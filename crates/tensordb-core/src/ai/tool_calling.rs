//! Hermes-style tool calling for NL-to-SQL with iterative schema discovery.
//!
//! Provides a multi-turn loop where the LLM can call `list_tables`,
//! `describe_table`, and `execute_sql` tools to discover schema and validate
//! SQL before returning a final answer.

use crate::error::{Result, TensorError};

use super::llm::{clean_sql_output, LlmEngine};

/// Maximum number of tool-calling rounds before giving up.
const MAX_TOOL_ROUNDS: usize = 3;

/// Maximum rows returned by the `execute_sql` tool (context budget).
const MAX_RESULT_ROWS: usize = 10;

// ── Tool definitions ────────────────────────────────────────────────────

/// Compact JSON tool definitions for the system prompt (~120 tokens).
fn build_tools_block() -> &'static str {
    r#"<tools>[
{"type":"function","function":{"name":"list_tables","description":"List all table names in the database","parameters":{"type":"object","properties":{}}}},
{"type":"function","function":{"name":"describe_table","description":"Get column names and types for a table","parameters":{"type":"object","properties":{"name":{"type":"string","description":"Table name"}},"required":["name"]}}},
{"type":"function","function":{"name":"execute_sql","description":"Execute a SQL query and return results (max 10 rows)","parameters":{"type":"object","properties":{"sql":{"type":"string","description":"SQL query"}},"required":["sql"]}}}
]</tools>"#
}

// ── ToolExecutor trait ──────────────────────────────────────────────────

/// Decouples AI tool calling from the Database implementation.
pub trait ToolExecutor {
    /// List all table names in the database.
    fn list_tables(&self) -> Result<Vec<String>>;

    /// Describe a table's columns: `(column_name, column_type)` pairs.
    fn describe_table(&self, name: &str) -> Result<Vec<(String, String)>>;

    /// Execute a SQL query and return a JSON string of the results.
    /// Errors are returned as JSON `{"error":"..."}`, not as `Err(...)`,
    /// so the model can self-correct.
    fn execute_sql_tool(&self, sql: &str) -> String;
}

// ── Tool call parsing ───────────────────────────────────────────────────

/// A parsed tool call from the model's output.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of the tool-calling orchestration loop.
#[derive(Debug)]
pub struct ToolCallingResult {
    /// The final SQL statement extracted from the model's output.
    pub sql: String,
    /// Number of tool-calling rounds used.
    pub rounds: usize,
    /// History of tool calls and their results.
    pub history: Vec<(ToolCall, String)>,
}

/// Parse a `<tool_call>...</tool_call>` block from the model's output.
///
/// Expects the format: `<tool_call>{"name":"...","arguments":{...}}</tool_call>`
pub fn parse_tool_call(text: &str) -> Option<ToolCall> {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";

    let start = text.find(start_tag)?;
    let json_start = start + start_tag.len();

    let json_end = text[json_start..].find(end_tag).map(|i| json_start + i)?;
    let json_str = text[json_start..json_end].trim();

    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;

    let name = parsed.get("name")?.as_str()?.to_string();
    let arguments = parsed
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    Some(ToolCall { name, arguments })
}

// ── Tool execution dispatch ─────────────────────────────────────────────

/// Execute a parsed tool call against a ToolExecutor and return a JSON result string.
fn execute_tool(call: &ToolCall, executor: &dyn ToolExecutor) -> String {
    match call.name.as_str() {
        "list_tables" => match executor.list_tables() {
            Ok(tables) => {
                let json = serde_json::json!({ "tables": tables });
                json.to_string()
            }
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        },
        "describe_table" => {
            let table_name = call
                .arguments
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            match executor.describe_table(table_name) {
                Ok(columns) => {
                    let cols: Vec<serde_json::Value> = columns
                        .into_iter()
                        .map(|(name, dtype)| serde_json::json!({ "column": name, "type": dtype }))
                        .collect();
                    serde_json::json!({ "columns": cols }).to_string()
                }
                Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
            }
        }
        "execute_sql" => {
            let sql = call
                .arguments
                .get("sql")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let result = executor.execute_sql_tool(sql);
            // Truncate to MAX_RESULT_ROWS if it looks like a JSON array
            truncate_result_rows(&result)
        }
        _ => serde_json::json!({ "error": format!("unknown tool: {}", call.name) }).to_string(),
    }
}

/// Truncate JSON array results to MAX_RESULT_ROWS.
fn truncate_result_rows(result: &str) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(result) {
        if let Some(rows) = val.get("rows").and_then(|r| r.as_array()) {
            if rows.len() > MAX_RESULT_ROWS {
                let truncated: Vec<&serde_json::Value> =
                    rows.iter().take(MAX_RESULT_ROWS).collect();
                return serde_json::json!({
                    "rows": truncated,
                    "truncated": true,
                    "total": rows.len()
                })
                .to_string();
            }
        }
    }
    result.to_string()
}

// ── Prompt building ─────────────────────────────────────────────────────

/// Build a multi-turn ChatML prompt with tool call history.
pub fn build_tool_prompt(
    system_prompt: &str,
    tools_block: &str,
    question: &str,
    schema_context: &str,
    history: &[(ToolCall, String)],
) -> String {
    let mut prompt = String::with_capacity(4096);

    // System message with tools block
    prompt.push_str("<|im_start|>system\n");
    prompt.push_str(system_prompt);
    prompt.push('\n');
    prompt.push_str(tools_block);
    prompt.push_str("<|im_end|>\n");

    // User question with schema context
    prompt.push_str("<|im_start|>user\n");
    if !schema_context.is_empty() {
        prompt.push_str(schema_context);
        prompt.push('\n');
    }
    prompt.push_str(question);
    prompt.push_str("<|im_end|>\n");

    // Replay tool call history
    for (call, result) in history {
        // Assistant's tool call
        prompt.push_str("<|im_start|>assistant\n");
        let call_json = serde_json::json!({
            "name": call.name,
            "arguments": call.arguments,
        });
        prompt.push_str(&format!("<tool_call>{}</tool_call>", call_json));
        prompt.push_str("<|im_end|>\n");

        // Tool result
        prompt.push_str("<|im_start|>user\n");
        prompt.push_str(&format!("<tool_result>{result}</tool_result>"));
        prompt.push_str("<|im_end|>\n");
    }

    // Open assistant turn for next generation
    prompt.push_str("<|im_start|>assistant\n");

    prompt
}

/// Build the system prompt for tool-calling mode.
///
/// Reuses the SQL dialect reference from `build_system_prompt()` but adjusts
/// the role description for tool calling.
fn build_tool_system_prompt() -> String {
    let mut p = String::with_capacity(2048);

    p.push_str(
        "You are TensorDB's SQL assistant with tool access. \
         Use the provided tools to discover the database schema before generating SQL. \
         When you have enough information, output exactly one SQL statement with no explanation.\n\n",
    );

    p.push_str("Workflow:\n");
    p.push_str("1. Call list_tables to see available tables\n");
    p.push_str("2. Call describe_table for relevant tables\n");
    p.push_str("3. Generate the SQL query using exact table and column names from the schema\n\n");

    p.push_str("Rules:\n");
    p.push_str("- Use ONLY tables and columns discovered via tools\n");
    p.push_str("- INSERT requires column names: INSERT INTO t (c1, c2) VALUES (v1, v2)\n");
    p.push_str("- Use single quotes for string literals\n");
    p.push_str("- For JOINs, use explicit column references: table.column\n");
    p.push_str("- When ready to answer, output the SQL directly (no tool call)\n\n");

    p.push_str("/no_think");

    p
}

// ── Orchestration loop ──────────────────────────────────────────────────

/// Run the tool-calling NL-to-SQL loop.
///
/// The model generates tokens; if it emits a `<tool_call>`, we parse and
/// execute the tool, append the result to the conversation history, rebuild
/// the prompt, and re-run. If the model outputs SQL (no tool call), we
/// extract it and return.
///
/// Max `MAX_TOOL_ROUNDS` rounds to prevent infinite loops.
pub fn nl_to_sql_with_tools(
    engine: &LlmEngine,
    question: &str,
    schema_context: &str,
    executor: &dyn ToolExecutor,
) -> Result<ToolCallingResult> {
    let system_prompt = build_tool_system_prompt();
    let tools_block = build_tools_block();

    let mut history: Vec<(ToolCall, String)> = Vec::new();
    let mut rounds = 0;

    loop {
        if rounds >= MAX_TOOL_ROUNDS {
            return Err(TensorError::LlmError(format!(
                "tool calling exceeded {MAX_TOOL_ROUNDS} rounds without producing SQL"
            )));
        }

        let prompt = build_tool_prompt(
            &system_prompt,
            tools_block,
            question,
            schema_context,
            &history,
        );

        let raw = engine.generate_for_tool_calling(&prompt, engine.max_tokens())?;
        rounds += 1;

        // Check if output contains a tool call
        if let Some(call) = parse_tool_call(&raw) {
            // Detect duplicate tool calls (prevents infinite loops)
            if history.iter().any(|(prev_call, _)| *prev_call == call) {
                // Model is repeating itself — force final generation
                return Err(TensorError::LlmError(
                    "model repeated a tool call; aborting".to_string(),
                ));
            }

            let result = execute_tool(&call, executor);
            history.push((call, result));
            continue;
        }

        // No tool call — treat as final SQL output
        let sql = clean_sql_output(&raw);
        if sql.is_empty() {
            return Err(TensorError::LlmError(
                "tool-calling loop produced empty SQL".to_string(),
            ));
        }

        return Ok(ToolCallingResult {
            sql,
            rounds,
            history,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tool_call_valid() {
        let text = r#"<tool_call>{"name":"list_tables","arguments":{}}</tool_call>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "list_tables");
        assert_eq!(call.arguments, serde_json::json!({}));
    }

    #[test]
    fn parse_tool_call_with_arguments() {
        let text =
            r#"<tool_call>{"name":"describe_table","arguments":{"name":"orders"}}</tool_call>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "describe_table");
        assert_eq!(call.arguments, serde_json::json!({"name": "orders"}));
    }

    #[test]
    fn parse_tool_call_with_whitespace() {
        let text = r#"<tool_call>
  {"name": "list_tables", "arguments": {}}
</tool_call>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "list_tables");
    }

    #[test]
    fn parse_tool_call_invalid_json() {
        let text = r#"<tool_call>not json</tool_call>"#;
        assert!(parse_tool_call(text).is_none());
    }

    #[test]
    fn parse_tool_call_missing_tags() {
        assert!(parse_tool_call("no tags here").is_none());
    }

    #[test]
    fn parse_tool_call_no_end_tag() {
        let text = r#"<tool_call>{"name":"list_tables","arguments":{}}"#;
        assert!(parse_tool_call(text).is_none());
    }

    #[test]
    fn parse_tool_call_missing_name() {
        let text = r#"<tool_call>{"arguments":{}}</tool_call>"#;
        assert!(parse_tool_call(text).is_none());
    }

    #[test]
    fn parse_tool_call_missing_arguments_defaults() {
        let text = r#"<tool_call>{"name":"list_tables"}</tool_call>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "list_tables");
        assert_eq!(call.arguments, serde_json::json!({}));
    }

    #[test]
    fn parse_tool_call_with_surrounding_text() {
        let text = r#"Let me check the tables. <tool_call>{"name":"list_tables","arguments":{}}</tool_call> Some trailing text."#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "list_tables");
    }

    #[test]
    fn build_tool_prompt_empty_history() {
        let prompt = build_tool_prompt(
            "system msg",
            "<tools>[]</tools>",
            "show me all users",
            "CREATE TABLE users (id INTEGER)",
            &[],
        );
        assert!(prompt.contains("system msg"));
        assert!(prompt.contains("<tools>[]</tools>"));
        assert!(prompt.contains("show me all users"));
        assert!(prompt.contains("CREATE TABLE users"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn build_tool_prompt_with_history() {
        let call = ToolCall {
            name: "list_tables".to_string(),
            arguments: serde_json::json!({}),
        };
        let result = r#"{"tables":["users","orders"]}"#.to_string();
        let history = vec![(call, result)];

        let prompt = build_tool_prompt("sys", "<tools>[]</tools>", "q", "", &history);
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("list_tables"));
        assert!(prompt.contains("<tool_result>"));
        assert!(prompt.contains("users"));
    }

    #[test]
    fn build_tool_prompt_no_schema_context() {
        let prompt = build_tool_prompt("sys", "<tools>[]</tools>", "question", "", &[]);
        // Should not have double newlines from empty schema
        assert!(!prompt.contains("\n\nquestion"));
    }

    #[test]
    fn truncate_result_rows_under_limit() {
        let input = r#"{"rows":[{"a":1},{"a":2}]}"#;
        assert_eq!(truncate_result_rows(input), input);
    }

    #[test]
    fn truncate_result_rows_over_limit() {
        let rows: Vec<serde_json::Value> = (0..20).map(|i| serde_json::json!({"id": i})).collect();
        let input = serde_json::json!({"rows": rows}).to_string();
        let output = truncate_result_rows(&input);
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed["rows"].as_array().unwrap().len(), MAX_RESULT_ROWS);
        assert_eq!(parsed["truncated"], true);
        assert_eq!(parsed["total"], 20);
    }

    #[test]
    fn truncate_result_rows_not_json() {
        let input = "not json";
        assert_eq!(truncate_result_rows(input), input);
    }

    #[test]
    fn tools_block_contains_all_tools() {
        let block = build_tools_block();
        assert!(block.contains("list_tables"));
        assert!(block.contains("describe_table"));
        assert!(block.contains("execute_sql"));
        assert!(block.starts_with("<tools>"));
        assert!(block.ends_with("</tools>"));
    }

    #[test]
    fn tool_system_prompt_has_required_sections() {
        let prompt = build_tool_system_prompt();
        assert!(prompt.contains("tool"));
        assert!(prompt.contains("list_tables"));
        assert!(prompt.contains("describe_table"));
        assert!(prompt.contains("/no_think"));
    }

    // ── Mock executor for orchestration tests ───────────────────────────

    struct MockExecutor {
        tables: Vec<String>,
        columns: std::collections::HashMap<String, Vec<(String, String)>>,
    }

    impl MockExecutor {
        fn new() -> Self {
            let mut columns = std::collections::HashMap::new();
            columns.insert(
                "users".to_string(),
                vec![
                    ("id".to_string(), "INTEGER".to_string()),
                    ("name".to_string(), "TEXT".to_string()),
                    ("balance".to_string(), "REAL".to_string()),
                ],
            );
            columns.insert(
                "orders".to_string(),
                vec![
                    ("id".to_string(), "INTEGER".to_string()),
                    ("user_id".to_string(), "INTEGER".to_string()),
                    ("total".to_string(), "REAL".to_string()),
                ],
            );
            Self {
                tables: vec!["users".to_string(), "orders".to_string()],
                columns,
            }
        }
    }

    impl ToolExecutor for MockExecutor {
        fn list_tables(&self) -> Result<Vec<String>> {
            Ok(self.tables.clone())
        }

        fn describe_table(&self, name: &str) -> Result<Vec<(String, String)>> {
            self.columns
                .get(name)
                .cloned()
                .ok_or_else(|| TensorError::LlmError(format!("table '{name}' not found")))
        }

        fn execute_sql_tool(&self, sql: &str) -> String {
            // Simple mock: return empty result
            let _ = sql;
            serde_json::json!({"rows": [], "message": "ok"}).to_string()
        }
    }

    #[test]
    fn execute_tool_list_tables() {
        let executor = MockExecutor::new();
        let call = ToolCall {
            name: "list_tables".to_string(),
            arguments: serde_json::json!({}),
        };
        let result = execute_tool(&call, &executor);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let tables = parsed["tables"].as_array().unwrap();
        assert_eq!(tables.len(), 2);
    }

    #[test]
    fn execute_tool_describe_table() {
        let executor = MockExecutor::new();
        let call = ToolCall {
            name: "describe_table".to_string(),
            arguments: serde_json::json!({"name": "users"}),
        };
        let result = execute_tool(&call, &executor);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        let columns = parsed["columns"].as_array().unwrap();
        assert_eq!(columns.len(), 3);
        assert_eq!(columns[0]["column"], "id");
    }

    #[test]
    fn execute_tool_describe_nonexistent() {
        let executor = MockExecutor::new();
        let call = ToolCall {
            name: "describe_table".to_string(),
            arguments: serde_json::json!({"name": "nonexistent"}),
        };
        let result = execute_tool(&call, &executor);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed["error"].as_str().is_some());
    }

    #[test]
    fn execute_tool_unknown() {
        let executor = MockExecutor::new();
        let call = ToolCall {
            name: "unknown_tool".to_string(),
            arguments: serde_json::json!({}),
        };
        let result = execute_tool(&call, &executor);
        assert!(result.contains("unknown tool"));
    }

    #[test]
    fn execute_tool_execute_sql() {
        let executor = MockExecutor::new();
        let call = ToolCall {
            name: "execute_sql".to_string(),
            arguments: serde_json::json!({"sql": "SELECT 1"}),
        };
        let result = execute_tool(&call, &executor);
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed["rows"].is_array());
    }
}
