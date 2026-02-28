use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use parking_lot::Mutex;

use crate::error::{Result, TensorError};

const DEFAULT_MAX_TOKENS: usize = 256;
const MODEL_FILENAME: &str = "Qwen3-0.6B-Q8_0.gguf";
const MODEL_URL: &str =
    "https://github.com/tensor-db/TensorDB/releases/download/v0.2.0-model/Qwen3-0.6B-Q8_0.gguf";

const SYSTEM_PROMPT: &str = "You are a SQL translator for TensorDB (a bitemporal database). \
TensorDB SQL supports: SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, \
SHOW TABLES, DESCRIBE <table>, time-travel (AS OF <timestamp>), \
aggregates (count, sum, avg, min, max), JOINs, CTEs, window functions. \
IMPORTANT: TensorDB does NOT have information_schema or pg_catalog. \
To list tables use SHOW TABLES. To describe a table use DESCRIBE <table>. \
Table names are plain identifiers — never use schema-qualified names like schema.table. \
Output ONLY a single SQL statement, nothing else — no explanation, no markdown. /no_think";

struct LoadedModel {
    backend: LlamaBackend,
    model: LlamaModel,
}

// Safety: LlamaBackend and LlamaModel are thread-safe (the C library uses internal locking).
// We wrap them in a Mutex<Option<>> anyway, so concurrent access is serialized.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

pub struct LlmEngine {
    inner: Mutex<Option<LoadedModel>>,
    model_path: PathBuf,
    loaded: AtomicBool,
    max_tokens: usize,
}

impl LlmEngine {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            inner: Mutex::new(None),
            model_path,
            loaded: AtomicBool::new(false),
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    fn ensure_loaded(&self) -> Result<()> {
        if self.loaded.load(Ordering::Acquire) {
            return Ok(());
        }

        let mut guard = self.inner.lock();
        // Double-check after acquiring lock
        if self.loaded.load(Ordering::Acquire) {
            return Ok(());
        }

        // Auto-download if model file doesn't exist
        if !self.model_path.exists() {
            Self::download_model(&self.model_path)?;
        }

        // Suppress verbose llama.cpp logs (layer info, graph reserves, etc.)
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(false),
        );

        let backend = LlamaBackend::init()
            .map_err(|e| TensorError::LlmError(format!("failed to init llama backend: {e}")))?;

        let model_params = LlamaModelParams::default();
        let model_params = std::pin::pin!(model_params);

        let model =
            LlamaModel::load_from_file(&backend, &self.model_path, &model_params).map_err(|e| {
                TensorError::LlmError(format!(
                    "failed to load model {}: {e}",
                    self.model_path.display()
                ))
            })?;

        *guard = Some(LoadedModel { backend, model });
        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    fn download_model(dest: &std::path::Path) -> Result<()> {
        eprintln!("Downloading TensorDB language model ({MODEL_FILENAME}, ~604 MB)...");
        eprintln!("  From: {MODEL_URL}");
        eprintln!("  To:   {}", dest.display());

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                TensorError::LlmError(format!(
                    "failed to create model directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        // Download to a temp file first, then rename (atomic-ish)
        let tmp_path = dest.with_extension("gguf.download");

        let resp = ureq::get(MODEL_URL)
            .call()
            .map_err(|e| TensorError::LlmError(format!("failed to download model: {e}")))?;

        let total: u64 = resp
            .header("content-length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        let mut reader = resp.into_reader();
        let mut file = std::fs::File::create(&tmp_path).map_err(|e| {
            TensorError::LlmError(format!("failed to create {}: {e}", tmp_path.display()))
        })?;

        let mut buf = vec![0u8; 1024 * 1024]; // 1 MB buffer
        let mut downloaded: u64 = 0;
        let mut last_pct: u64 = 0;

        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| TensorError::LlmError(format!("download read error: {e}")))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n])
                .map_err(|e| TensorError::LlmError(format!("write error: {e}")))?;
            downloaded += n as u64;

            if total > 0 {
                let pct = downloaded * 100 / total;
                if pct >= last_pct + 5 {
                    eprint!(
                        "\r  Progress: {pct}% ({} / {} MB)",
                        downloaded / (1024 * 1024),
                        total / (1024 * 1024),
                    );
                    last_pct = pct;
                }
            }
        }
        eprintln!("\r  Progress: 100% — download complete.       ");

        drop(file);
        std::fs::rename(&tmp_path, dest)
            .map_err(|e| TensorError::LlmError(format!("failed to finalize model file: {e}")))?;

        eprintln!("  Model ready.");
        Ok(())
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.ensure_loaded()?;

        let guard = self.inner.lock();
        let loaded = guard.as_ref().ok_or(TensorError::LlmNotAvailable)?;

        let ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));

        let mut ctx = loaded
            .model
            .new_context(&loaded.backend, ctx_params)
            .map_err(|e| TensorError::LlmError(format!("failed to create context: {e}")))?;

        // Tokenize the prompt
        let tokens = loaded
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| TensorError::LlmError(format!("tokenization failed: {e}")))?;

        if tokens.is_empty() {
            return Err(TensorError::LlmError(
                "prompt tokenized to empty sequence".to_string(),
            ));
        }

        // Feed prompt tokens into context
        let mut batch = LlamaBatch::new(512, 1);
        let last_index = (tokens.len() - 1) as i32;
        for (i, token) in (0i32..).zip(tokens.iter()) {
            batch
                .add(*token, i, &[0], i == last_index)
                .map_err(|e| TensorError::LlmError(format!("batch add failed: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| TensorError::LlmError(format!("initial decode failed: {e}")))?;

        // Generation loop
        let mut sampler =
            LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);

        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_cur = batch.n_tokens();
        let n_len = tokens.len() as i32 + max_tokens as i32;

        while n_cur <= n_len {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if loaded.model.is_eog_token(token) {
                break;
            }

            let piece = loaded
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| TensorError::LlmError(format!("token decode failed: {e}")))?;

            output.push_str(&piece);

            // Early stop: once we have a semicolon followed by a newline
            if output.contains(';') && piece.contains('\n') {
                break;
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| TensorError::LlmError(format!("batch add failed: {e}")))?;

            ctx.decode(&mut batch)
                .map_err(|e| TensorError::LlmError(format!("decode failed: {e}")))?;

            n_cur += 1;
        }

        Ok(output.trim().to_string())
    }

    pub fn nl_to_sql(&self, question: &str, schema_context: &str) -> Result<String> {
        // Build a ChatML-style prompt (Qwen3 instruct format).
        // The fine-tuned model handles TensorDB-specific SQL (SHOW TABLES, DESCRIBE,
        // temporal queries, etc.) without few-shot examples.
        let user_content = if schema_context.is_empty() {
            format!("Question: {question}")
        } else {
            format!("Schema:\n{schema_context}\n\nQuestion: {question}")
        };
        let prompt = format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
             <|im_start|>user\n{user_content}<|im_end|>\n\
             <|im_start|>assistant\n"
        );

        let raw = self.generate(&prompt, self.max_tokens)?;

        // Clean up: strip markdown fences if present, extract SQL
        let sql = clean_sql_output(&raw);

        if sql.is_empty() {
            return Err(TensorError::LlmError("LLM returned empty SQL".to_string()));
        }

        Ok(sql)
    }
}

/// SQL keywords that mark the start of a statement.
const SQL_KEYWORDS: &[&str] = &[
    "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "SHOW", "DESCRIBE",
    "EXPLAIN", "WITH", "ANALYZE", "COPY",
];

fn clean_sql_output(raw: &str) -> String {
    let mut s = raw.trim().to_string();

    // Strip ChatML end-of-turn token if present
    if let Some(pos) = s.find("<|im_end|>") {
        s = s[..pos].to_string();
    }

    // Handle Qwen3 thinking blocks: <think>...</think>
    // First try to get content AFTER </think>, but if the think block is unclosed
    // (model hit max tokens), fall back to extracting SQL from within it.
    let think_content = if let Some(start) = s.find("<think>") {
        if let Some(end) = s.find("</think>") {
            let inside = s[start + 7..end].to_string();
            s = format!("{}{}", &s[..start], &s[end + 8..]);
            Some(inside)
        } else {
            // Unclosed think tag — save the content for fallback extraction
            let inside = s[start + 7..].to_string();
            s = s[..start].to_string();
            Some(inside)
        }
    } else {
        None
    };

    let result = extract_sql_from_text(&s);

    // If we got nothing from the main text, try extracting SQL from inside the think block
    if result.is_empty() {
        if let Some(ref think_text) = think_content {
            return extract_sql_from_text(think_text);
        }
    }

    result
}

/// Extract a SQL statement from text, handling markdown fences, preamble, etc.
fn extract_sql_from_text(text: &str) -> String {
    let s = text.trim();

    // Strip markdown code fences
    let s = if s.starts_with("```sql") {
        s.strip_prefix("```sql").unwrap_or(s)
    } else if s.starts_with("```") {
        s.strip_prefix("```").unwrap_or(s)
    } else {
        s
    };
    let s = s.strip_suffix("```").unwrap_or(s).trim();

    // If the output has preamble text (e.g. "Answer: SELECT ..."), find the first SQL keyword
    let upper = s.to_uppercase();
    let sql_start = SQL_KEYWORDS.iter().filter_map(|kw| upper.find(kw)).min();

    let s = match sql_start {
        Some(pos) if pos > 0 => &s[pos..],
        _ => s,
    };

    // Take only the first SQL statement (up to and including the first semicolon)
    if let Some(pos) = s.find(';') {
        s[..pos].trim().to_string()
    } else {
        s.trim().to_string()
    }
}

impl std::fmt::Debug for LlmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmEngine")
            .field("model_path", &self.model_path)
            .field("loaded", &self.loaded.load(Ordering::Relaxed))
            .field("max_tokens", &self.max_tokens)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_sql_strips_fences() {
        assert_eq!(
            clean_sql_output("```sql\nSELECT * FROM users;\n```"),
            "SELECT * FROM users"
        );
    }

    #[test]
    fn clean_sql_takes_first_statement() {
        assert_eq!(clean_sql_output("SELECT 1; SELECT 2;"), "SELECT 1");
    }

    #[test]
    fn clean_sql_preserves_plain() {
        assert_eq!(clean_sql_output("SHOW TABLES;"), "SHOW TABLES");
    }

    #[test]
    fn clean_sql_handles_no_semicolon() {
        assert_eq!(
            clean_sql_output("SELECT count(*) FROM users"),
            "SELECT count(*) FROM users"
        );
    }

    #[test]
    fn clean_sql_strips_preamble_with_colon() {
        assert_eq!(
            clean_sql_output("Answer: SELECT count(*) FROM users;"),
            "SELECT count(*) FROM users"
        );
    }

    #[test]
    fn clean_sql_strips_multiline_preamble() {
        assert_eq!(
            clean_sql_output("Here is the SQL query:\n\nSELECT * FROM users WHERE balance > 500;"),
            "SELECT * FROM users WHERE balance > 500"
        );
    }

    #[test]
    fn clean_sql_strips_chatml_end_token() {
        assert_eq!(clean_sql_output("SELECT 1;<|im_end|>"), "SELECT 1");
    }

    #[test]
    fn clean_sql_strips_think_block() {
        assert_eq!(
            clean_sql_output(
                "<think>\nLet me think about this...\n</think>\nSELECT count(*) FROM users;"
            ),
            "SELECT count(*) FROM users"
        );
    }

    #[test]
    fn clean_sql_extracts_from_unclosed_think() {
        // When model hits max tokens inside a think block, extract SQL from within it
        assert_eq!(
            clean_sql_output(
                "<think>\nThe SQL should be SELECT COUNT(*) FROM users WHERE balance > 500;"
            ),
            "SELECT COUNT(*) FROM users WHERE balance > 500"
        );
    }

    #[test]
    fn clean_sql_handles_show_tables() {
        assert_eq!(
            clean_sql_output("The query is: SHOW TABLES;"),
            "SHOW TABLES"
        );
    }

    #[test]
    fn clean_sql_strips_trailing_commentary() {
        assert_eq!(
            clean_sql_output("SELECT * FROM users; -- this lists all users"),
            "SELECT * FROM users"
        );
    }
}
