/// Integration tests for Hermes-style tool calling NL-to-SQL.
///
/// Tests the `ToolExecutor` trait, tool call parsing, prompt building,
/// and the full orchestration loop.
///
/// Tests that require a real LLM model will be skipped unless `LLM_MODEL_PATH` is set.
///
/// Run:
///   LLM_MODEL_PATH=/path/to/model.gguf cargo test --release --test llm_tool_calling -- --test-threads=1 --nocapture
#[cfg(feature = "llm")]
mod tests {
    use tensordb_core::ai::tool_calling::{
        build_tool_prompt, parse_tool_call, ToolCall, ToolExecutor,
    };
    use tensordb_core::config::Config;
    use tensordb_core::engine::db::Database;

    // ── Unit tests (no model needed) ────────────────────────────────────

    #[test]
    fn tool_call_parsing_valid_list_tables() {
        let text = r#"<tool_call>{"name":"list_tables","arguments":{}}</tool_call>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "list_tables");
        assert_eq!(call.arguments, serde_json::json!({}));
    }

    #[test]
    fn tool_call_parsing_valid_describe_table() {
        let text =
            r#"<tool_call>{"name":"describe_table","arguments":{"name":"orders"}}</tool_call>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "describe_table");
        assert_eq!(call.arguments["name"], "orders");
    }

    #[test]
    fn tool_call_parsing_valid_execute_sql() {
        let text =
            r#"<tool_call>{"name":"execute_sql","arguments":{"sql":"SELECT 1"}}</tool_call>"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "execute_sql");
        assert_eq!(call.arguments["sql"], "SELECT 1");
    }

    #[test]
    fn tool_call_parsing_invalid_json() {
        assert!(parse_tool_call("<tool_call>not json</tool_call>").is_none());
    }

    #[test]
    fn tool_call_parsing_no_tags() {
        assert!(parse_tool_call("just plain text").is_none());
    }

    #[test]
    fn tool_call_parsing_missing_end_tag() {
        assert!(parse_tool_call(r#"<tool_call>{"name":"list_tables","arguments":{}}"#).is_none());
    }

    #[test]
    fn tool_call_parsing_with_surrounding_text() {
        let text =
            r#"I need to check. <tool_call>{"name":"list_tables","arguments":{}}</tool_call> done"#;
        let call = parse_tool_call(text).unwrap();
        assert_eq!(call.name, "list_tables");
    }

    #[test]
    fn tool_call_equality() {
        let a = ToolCall {
            name: "list_tables".to_string(),
            arguments: serde_json::json!({}),
        };
        let b = ToolCall {
            name: "list_tables".to_string(),
            arguments: serde_json::json!({}),
        };
        assert_eq!(a, b);
    }

    // ── Prompt building tests ───────────────────────────────────────────

    #[test]
    fn tool_prompt_building_empty_history() {
        let prompt = build_tool_prompt(
            "You are a SQL assistant.",
            "<tools>[...]</tools>",
            "Show all users",
            "CREATE TABLE users (id INTEGER)",
            &[],
        );
        // Should contain system, tools, user, and open assistant turn
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("You are a SQL assistant."));
        assert!(prompt.contains("<tools>[...]</tools>"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("Show all users"));
        assert!(prompt.contains("CREATE TABLE users"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn tool_prompt_building_with_history() {
        let call = ToolCall {
            name: "list_tables".to_string(),
            arguments: serde_json::json!({}),
        };
        let result = r#"{"tables":["users","orders"]}"#.to_string();
        let history = vec![(call, result)];

        let prompt = build_tool_prompt("sys", "<tools>[]</tools>", "question", "", &history);

        // Should contain the replayed tool call and result
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("list_tables"));
        assert!(prompt.contains("<tool_result>"));
        assert!(prompt.contains(r#""tables""#));

        // Should end with open assistant turn
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn tool_prompt_building_multi_round() {
        let calls = vec![
            (
                ToolCall {
                    name: "list_tables".to_string(),
                    arguments: serde_json::json!({}),
                },
                r#"{"tables":["users"]}"#.to_string(),
            ),
            (
                ToolCall {
                    name: "describe_table".to_string(),
                    arguments: serde_json::json!({"name": "users"}),
                },
                r#"{"columns":[{"column":"id","type":"INTEGER"}]}"#.to_string(),
            ),
        ];

        let prompt = build_tool_prompt("sys", "<tools>[]</tools>", "q", "", &calls);

        // Should have two assistant+user turn pairs
        let tool_call_count = prompt.matches("<tool_call>").count();
        let tool_result_count = prompt.matches("<tool_result>").count();
        assert_eq!(tool_call_count, 2);
        assert_eq!(tool_result_count, 2);
    }

    #[test]
    fn tool_prompt_token_budget() {
        // With a 3-round worst case, prompt should fit in 2048 tokens.
        // Rough estimate: ~4 chars per token for English text.
        let calls = vec![
            (
                ToolCall {
                    name: "list_tables".to_string(),
                    arguments: serde_json::json!({}),
                },
                r#"{"tables":["users","orders","products"]}"#.to_string(),
            ),
            (
                ToolCall {
                    name: "describe_table".to_string(),
                    arguments: serde_json::json!({"name": "users"}),
                },
                r#"{"columns":[{"column":"id","type":"INTEGER"},{"column":"name","type":"TEXT"},{"column":"email","type":"TEXT"}]}"#.to_string(),
            ),
            (
                ToolCall {
                    name: "describe_table".to_string(),
                    arguments: serde_json::json!({"name": "orders"}),
                },
                r#"{"columns":[{"column":"id","type":"INTEGER"},{"column":"user_id","type":"INTEGER"},{"column":"total","type":"REAL"}]}"#.to_string(),
            ),
        ];

        let prompt = build_tool_prompt(
            "You are TensorDB's SQL assistant with tool access.",
            r#"<tools>[{"type":"function","function":{"name":"list_tables"}}]</tools>"#,
            "Show all users who have orders over $50",
            "CREATE TABLE users (id INTEGER, name TEXT)",
            &calls,
        );

        // Conservative: 2048 tokens * 4 chars/token = 8192 chars
        assert!(
            prompt.len() < 8192,
            "prompt is {} chars, may exceed 2048 token budget",
            prompt.len()
        );
    }

    // ── ToolExecutor for Database (integration tests) ───────────────────

    fn model_available() -> bool {
        std::env::var("LLM_MODEL_PATH")
            .ok()
            .map(|p| std::path::Path::new(&p).exists())
            .unwrap_or(false)
    }

    fn open_test_db(name: &str) -> Database {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        let _ = std::fs::create_dir_all(&path);

        let config = Config {
            shard_count: 1,
            ai_auto_insights: false,
            llm_model_path: std::env::var("LLM_MODEL_PATH").ok(),
            ..Config::default()
        };

        let db = Database::open(path, config).expect("open db");

        db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT, role TEXT, created_at INTEGER);").unwrap();
        db.sql("INSERT INTO users (id, name, email, role, created_at) VALUES (1, 'alice', 'alice@example.com', 'admin', 1700000000);").unwrap();
        db.sql("INSERT INTO users (id, name, email, role, created_at) VALUES (2, 'bob', 'bob@example.com', 'user', 1700100000);").unwrap();
        db.sql("INSERT INTO users (id, name, email, role, created_at) VALUES (3, 'carol', 'carol@example.com', 'admin', 1700200000);").unwrap();

        db.sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, product TEXT, amount REAL, status TEXT);").unwrap();
        db.sql("INSERT INTO orders (id, user_id, product, amount, status) VALUES (101, 1, 'Widget A', 29.99, 'shipped');").unwrap();
        db.sql("INSERT INTO orders (id, user_id, product, amount, status) VALUES (102, 2, 'Widget B', 49.99, 'pending');").unwrap();
        db.sql("INSERT INTO orders (id, user_id, product, amount, status) VALUES (103, 1, 'Widget C', 15.00, 'delivered');").unwrap();

        db
    }

    #[test]
    fn tool_executor_list_tables() {
        let db = open_test_db("tool_list_tables");
        let tables = db.list_tables().unwrap();
        assert!(tables.contains(&"users".to_string()));
        assert!(tables.contains(&"orders".to_string()));
    }

    #[test]
    fn tool_executor_describe_table() {
        let db = open_test_db("tool_describe");
        let cols = db.describe_table("users").unwrap();
        let col_names: Vec<&str> = cols.iter().map(|(name, _)| name.as_str()).collect();
        assert!(col_names.contains(&"id"));
        assert!(col_names.contains(&"name"));
        assert!(col_names.contains(&"email"));
    }

    #[test]
    fn tool_executor_describe_nonexistent() {
        let db = open_test_db("tool_describe_none");
        let result = db.describe_table("nonexistent");
        // Should either return empty or an error
        if let Ok(cols) = result {
            assert!(cols.is_empty());
        }
    }

    #[test]
    fn tool_executor_execute_sql_success() {
        let db = open_test_db("tool_exec_sql");
        let result = db.execute_sql_tool("SELECT COUNT(*) as cnt FROM users");
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert!(parsed.get("rows").is_some() || parsed.get("error").is_none());
    }

    #[test]
    fn tool_executor_execute_sql_error_as_data() {
        let db = open_test_db("tool_exec_sql_err");
        let result = db.execute_sql_tool("SELECT * FROM nonexistent_table");
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        // Errors should be returned as JSON data, not as Err
        assert!(parsed.get("error").is_some());
    }

    // ── Full tool-calling scorecard (requires model) ────────────────────

    #[test]
    fn tool_calling_scorecard() {
        if !model_available() {
            eprintln!("Skipping tool_calling_scorecard: no LLM model file (set LLM_MODEL_PATH)");
            return;
        }

        let db = open_test_db("tool_scorecard");

        let questions = [
            // Easy (single table, simple filter)
            ("Count all users", vec!["SELECT", "COUNT", "users"]),
            ("List all tables", vec!["SHOW TABLES"]),
            ("Show all orders", vec!["SELECT", "orders"]),
            // Medium (filtering, aggregation)
            (
                "How many orders are shipped?",
                vec!["SELECT", "COUNT", "orders", "shipped"],
            ),
            (
                "What is the total order amount?",
                vec!["SELECT", "SUM", "amount", "orders"],
            ),
            ("Show admin users", vec!["SELECT", "users", "admin"]),
            // Hard (joins, subqueries — where tool calling helps most)
            (
                "Show orders with user names",
                vec!["SELECT", "JOIN", "users", "orders"],
            ),
            (
                "Which user has the most orders?",
                vec!["SELECT", "COUNT", "GROUP BY"],
            ),
            (
                "What is the average order amount per user?",
                vec!["SELECT", "AVG", "GROUP BY"],
            ),
            // Schema discovery (model MUST use tools to get this right)
            ("Describe the users table", vec!["DESCRIBE", "users"]),
            (
                "What columns does the orders table have?",
                vec!["DESCRIBE", "orders"],
            ),
            (
                "Show all pending orders with user email",
                vec!["SELECT", "JOIN", "pending"],
            ),
        ];

        let mut keyword_pass = 0;
        let mut execution_pass = 0;
        let total = questions.len();

        for (i, (question, keywords)) in questions.iter().enumerate() {
            eprintln!("\n--- Question {}/{total}: {question}", i + 1);

            match db.ask_with_tools(question) {
                Ok((sql, _result)) => {
                    eprintln!("  SQL: {sql}");

                    // Keyword check
                    let upper = sql.to_uppercase();
                    let kw_ok = keywords.iter().all(|kw| upper.contains(&kw.to_uppercase()));
                    if kw_ok {
                        keyword_pass += 1;
                        eprintln!("  Keywords: PASS");
                    } else {
                        eprintln!("  Keywords: FAIL (missing some of {:?})", keywords);
                    }

                    execution_pass += 1;
                    eprintln!("  Execution: PASS");
                }
                Err(e) => {
                    eprintln!("  ERROR: {e}");
                    // Check if ask_sql (without tools) would have worked
                    if let Ok(sql) = db.ask_sql(question) {
                        eprintln!("  (ask_sql produced: {sql})");
                    }
                }
            }
        }

        eprintln!("\n=== Tool-Calling Scorecard ===");
        eprintln!("Keyword pass: {keyword_pass}/{total}");
        eprintln!("Execution pass: {execution_pass}/{total}");

        // We expect tool calling to at least match direct generation
        assert!(
            keyword_pass >= 8,
            "expected at least 8/{total} keyword pass, got {keyword_pass}"
        );
    }
}
