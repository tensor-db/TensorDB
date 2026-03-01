//! NativeLlmEngine — pure-Rust inference for NL-to-SQL translation.
//!
//! Replaces the llama-cpp-2 dependency with a custom GGUF loader, BPE tokenizer,
//! Qwen2 transformer runtime, and constrained SQL grammar decoder.
//!
//! Key optimizations over the previous implementation:
//! - **Schema cache**: TTL-based caching of schema context avoids re-running
//!   SHOW TABLES + DESCRIBE on every `ask()` call.
//! - **KV cache prefix reuse**: The system prompt + schema prefix KV cache state
//!   is preserved across calls, avoiding redundant forward passes.
//! - **Constrained decoding**: SQL grammar decoder biases generation toward valid SQL.
//! - **Pure Rust**: No C++ dependencies (llama.cpp), simpler build, easier cross-compilation.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::error::{Result, TensorError};

use super::gguf::GgufFile;
use super::sampler::Sampler;
use super::schema_cache::SchemaCache;
use super::sql_grammar::SqlGrammarDecoder;
use super::tokenizer::BpeTokenizer;
use super::transformer::{KvCache, ModelConfig, TransformerModel};

const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_CONTEXT_SIZE: usize = 2048;
const DEFAULT_SCHEMA_CACHE_TTL_SECS: u64 = 60;
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
    model: TransformerModel,
    tokenizer: BpeTokenizer,
    grammar: SqlGrammarDecoder,
    config: ModelConfig,
}

pub struct LlmEngine {
    inner: Mutex<Option<LoadedModel>>,
    model_path: PathBuf,
    loaded: AtomicBool,
    max_tokens: usize,
    context_size: usize,
    schema_cache: SchemaCache,
    grammar_constrained: bool,
    /// Cached KV state for the system prompt prefix (reused across calls with same schema)
    prefix_kv_cache: Mutex<Option<PrefixKvState>>,
    kv_cache_prefix_enabled: bool,
}

/// Cached KV state for the system prompt + schema prefix.
struct PrefixKvState {
    kv_cache: KvCache,
    prefix_token_count: usize,
    schema_text_hash: u64,
}

impl LlmEngine {
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            inner: Mutex::new(None),
            model_path,
            loaded: AtomicBool::new(false),
            max_tokens: DEFAULT_MAX_TOKENS,
            context_size: DEFAULT_CONTEXT_SIZE,
            schema_cache: SchemaCache::new(DEFAULT_SCHEMA_CACHE_TTL_SECS),
            grammar_constrained: true,
            prefix_kv_cache: Mutex::new(None),
            kv_cache_prefix_enabled: true,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_context_size(mut self, context_size: usize) -> Self {
        self.context_size = context_size;
        self
    }

    pub fn with_schema_cache_ttl(self, ttl_secs: u64) -> Self {
        // Replace the schema cache with new TTL (must rebuild since SchemaCache isn't Clone)
        Self {
            schema_cache: SchemaCache::new(ttl_secs),
            ..self
        }
    }

    pub fn with_grammar_constrained(mut self, enabled: bool) -> Self {
        self.grammar_constrained = enabled;
        self
    }

    pub fn with_kv_cache_prefix(mut self, enabled: bool) -> Self {
        self.kv_cache_prefix_enabled = enabled;
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

        // Load GGUF file
        let gguf = GgufFile::open(&self.model_path)?;

        // Extract model config
        let config = ModelConfig::from_gguf(&gguf)?;

        // Load tokenizer
        let tokenizer = BpeTokenizer::from_gguf(&gguf)?;

        // Build grammar decoder
        let grammar = SqlGrammarDecoder::new(&tokenizer, self.grammar_constrained);

        // Load transformer model
        let model = TransformerModel::from_gguf(&gguf, &config)?;

        *guard = Some(LoadedModel {
            model,
            tokenizer,
            grammar,
            config,
        });
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

        let tokens = loaded.tokenizer.encode(prompt);
        if tokens.is_empty() {
            return Err(TensorError::LlmError(
                "prompt tokenized to empty sequence".to_string(),
            ));
        }

        let ctx_size = self.context_size;
        let mut kv_cache = KvCache::new(
            loaded.config.n_layers,
            loaded.config.n_kv_heads,
            loaded.config.head_dim,
            ctx_size,
        );

        // Prefill: process all prompt tokens
        let mut logits = loaded.model.forward_batch(&tokens, 0, &mut kv_cache);

        // Generation loop
        let mut sampler = Sampler::greedy();
        let mut output_tokens: Vec<u32> = Vec::new();
        let mut pos = tokens.len();

        for _ in 0..max_tokens {
            // Apply grammar constraints
            loaded.grammar.apply(&mut logits);

            let token = sampler.sample(&mut logits, &output_tokens);

            if loaded.tokenizer.is_eos(token) {
                break;
            }

            output_tokens.push(token);

            // Early stop: semicolon followed by newline
            let piece = loaded.tokenizer.decode(&[token]);
            if piece.contains('\n') {
                let output_so_far = loaded.tokenizer.decode(&output_tokens);
                if output_so_far.contains(';') {
                    break;
                }
            }

            if pos >= ctx_size - 1 {
                break; // Context window full
            }

            // Forward pass for the new token
            logits = loaded.model.forward(token, pos, &mut kv_cache);
            pos += 1;
        }

        let output = loaded.tokenizer.decode(&output_tokens);
        Ok(output.trim().to_string())
    }

    pub fn nl_to_sql(&self, question: &str, schema_context: &str) -> Result<String> {
        self.ensure_loaded()?;

        let guard = self.inner.lock();
        let loaded = guard.as_ref().ok_or(TensorError::LlmNotAvailable)?;

        // Build ChatML prompt
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

        let ctx_size = self.context_size;

        // Check for KV cache prefix reuse
        let schema_hash = simple_hash(schema_context.as_bytes());

        let (mut kv_cache, start_pos, logits) =
            if self.kv_cache_prefix_enabled && !schema_context.is_empty() {
                let prefix_guard = self.prefix_kv_cache.lock();
                if let Some(ref prefix_state) = *prefix_guard {
                    if prefix_state.schema_text_hash == schema_hash {
                        // Cache hit! Reuse the KV cache from the prefix
                        let mut kv = prefix_state.kv_cache.clone_state();
                        let start = prefix_state.prefix_token_count;

                        // Only tokenize and process the user question part
                        let suffix = format!(
                            "<|im_start|>user\n{user_content}<|im_end|>\n\
                             <|im_start|>assistant\n"
                        );
                        let suffix_tokens = loaded.tokenizer.encode(&suffix);

                        // Forward pass for suffix tokens only
                        let logits = loaded.model.forward_batch(&suffix_tokens, start, &mut kv);

                        drop(prefix_guard);
                        return self.generate_sql_from_logits(
                            loaded,
                            logits,
                            &mut kv,
                            start + suffix_tokens.len(),
                            ctx_size,
                        );
                    }
                }
                drop(prefix_guard);

                // Cache miss — process prefix first, cache it, then continue
                let full_tokens = loaded.tokenizer.encode(&prompt);
                let mut kv = KvCache::new(
                    loaded.config.n_layers,
                    loaded.config.n_kv_heads,
                    loaded.config.head_dim,
                    ctx_size,
                );

                // Compute prefix tokens (system prompt + schema)
                let prefix_prompt = format!(
                    "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n\
                     <|im_start|>user\nSchema:\n{schema_context}\n\n"
                );
                let prefix_tokens = loaded.tokenizer.encode(&prefix_prompt);
                let prefix_len = prefix_tokens.len();

                // Process prefix tokens first
                let _ = loaded.model.forward_batch(&prefix_tokens, 0, &mut kv);

                // Cache the prefix KV state for future reuse (clone before continuing)
                let kv_for_cache = kv.clone_state();
                let mut prefix_guard = self.prefix_kv_cache.lock();
                *prefix_guard = Some(PrefixKvState {
                    kv_cache: kv_for_cache,
                    prefix_token_count: prefix_len,
                    schema_text_hash: schema_hash,
                });
                drop(prefix_guard);

                // Continue processing the remaining suffix tokens
                let suffix_tokens = &full_tokens[prefix_len..];
                let logits = if suffix_tokens.is_empty() {
                    // Edge case: prompt is exactly the prefix
                    loaded.model.forward_batch(&prefix_tokens, 0, &mut kv)
                } else {
                    loaded
                        .model
                        .forward_batch(suffix_tokens, prefix_len, &mut kv)
                };

                (kv, full_tokens.len(), logits)
            } else {
                // No prefix caching — process full prompt
                let full_tokens = loaded.tokenizer.encode(&prompt);
                let mut kv = KvCache::new(
                    loaded.config.n_layers,
                    loaded.config.n_kv_heads,
                    loaded.config.head_dim,
                    ctx_size,
                );
                let logits = loaded.model.forward_batch(&full_tokens, 0, &mut kv);
                (kv, full_tokens.len(), logits)
            };

        let raw =
            self.generate_sql_from_logits(loaded, logits, &mut kv_cache, start_pos, ctx_size)?;

        let sql = clean_sql_output(&raw);

        if sql.is_empty() {
            return Err(TensorError::LlmError("LLM returned empty SQL".to_string()));
        }

        Ok(sql)
    }

    /// Generate SQL tokens from logits, using grammar constraints and sampling.
    fn generate_sql_from_logits(
        &self,
        loaded: &LoadedModel,
        mut logits: Vec<f32>,
        kv_cache: &mut KvCache,
        start_pos: usize,
        ctx_size: usize,
    ) -> Result<String> {
        let mut sampler = Sampler::greedy();
        let mut output_tokens: Vec<u32> = Vec::new();
        let mut pos = start_pos;

        for _ in 0..self.max_tokens {
            // Apply grammar constraints
            loaded.grammar.apply(&mut logits);

            let token = sampler.sample(&mut logits, &output_tokens);

            if loaded.tokenizer.is_eos(token) {
                break;
            }

            output_tokens.push(token);

            // Early stop: semicolon followed by newline
            let piece = loaded.tokenizer.decode(&[token]);
            if piece.contains('\n') {
                let output_so_far = loaded.tokenizer.decode(&output_tokens);
                if output_so_far.contains(';') {
                    break;
                }
            }

            if pos >= ctx_size - 1 {
                break;
            }

            logits = loaded.model.forward(token, pos, kv_cache);
            pos += 1;
        }

        let output = loaded.tokenizer.decode(&output_tokens);
        Ok(output.trim().to_string())
    }

    /// Invalidate the schema cache. Called after DDL statements.
    pub fn invalidate_schema_cache(&self) {
        self.schema_cache.invalidate();
        // Also invalidate prefix KV cache since schema changed
        let mut prefix_guard = self.prefix_kv_cache.lock();
        *prefix_guard = None;
    }

    /// Get a reference to the schema cache.
    pub fn schema_cache(&self) -> &SchemaCache {
        &self.schema_cache
    }
}

/// Simple non-cryptographic hash for schema text comparison.
fn simple_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    h
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
    let think_content = if let Some(start) = s.find("<think>") {
        if let Some(end) = s.find("</think>") {
            let inside = s[start + 7..end].to_string();
            s = format!("{}{}", &s[..start], &s[end + 8..]);
            Some(inside)
        } else {
            let inside = s[start + 7..].to_string();
            s = s[..start].to_string();
            Some(inside)
        }
    } else {
        None
    };

    let result = extract_sql_from_text(&s);

    if result.is_empty() {
        if let Some(ref think_text) = think_content {
            return extract_sql_from_text(think_text);
        }
    }

    result
}

fn extract_sql_from_text(text: &str) -> String {
    let s = text.trim();

    let s = if s.starts_with("```sql") {
        s.strip_prefix("```sql").unwrap_or(s)
    } else if s.starts_with("```") {
        s.strip_prefix("```").unwrap_or(s)
    } else {
        s
    };
    let s = s.strip_suffix("```").unwrap_or(s).trim();

    let upper = s.to_uppercase();
    let sql_start = SQL_KEYWORDS.iter().filter_map(|kw| upper.find(kw)).min();

    let s = match sql_start {
        Some(pos) if pos > 0 => &s[pos..],
        _ => s,
    };

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
            .field("context_size", &self.context_size)
            .field("grammar_constrained", &self.grammar_constrained)
            .field("kv_cache_prefix", &self.kv_cache_prefix_enabled)
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

    #[test]
    fn simple_hash_deterministic() {
        let h1 = simple_hash(b"hello world");
        let h2 = simple_hash(b"hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn simple_hash_differs() {
        let h1 = simple_hash(b"schema_v1");
        let h2 = simple_hash(b"schema_v2");
        assert_ne!(h1, h2);
    }
}
