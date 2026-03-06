use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crossbeam_channel::{bounded, unbounded, Sender};
use parking_lot::Mutex;

use crate::ai::{
    correlation_prefix_for_cluster, insight_prefix_for_source, insight_storage_key,
    parse_insight_id, AiCorrelationRef, AiInsight, AiRuntimeHandle, AiRuntimeStats,
};
use crate::config::Config;
use crate::engine::fast_write::{FastShardState, FastWritePath};
use crate::engine::shard::{
    ChangeEvent, ShardCommand, ShardOpenParams, ShardReadHandle, ShardRuntime, ShardStats,
    ShardStorageInfo, WriteBatchItem,
};
use crate::error::{Result, TensorError};
use crate::native_bridge::{build_hasher, Hasher};
use crate::sql::exec::{
    execute_sql, execute_sql_with_session, PreparedStatement, SqlResult, SqlSessionHandle,
};
use crate::storage::cache::{BlockCache, IndexCache};
use crate::storage::group_wal::{DurabilityThread, WalBatchQueue};
use crate::storage::manifest::Manifest;

#[derive(Debug, Clone)]
pub struct ExplainRow {
    pub shard_id: usize,
    pub bloom_hit: Option<bool>,
    pub sstable_block: Option<usize>,
    pub commit_ts_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixScanRow {
    pub user_key: Vec<u8>,
    pub doc: Vec<u8>,
    pub commit_ts: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DbStats {
    pub shard_count: usize,
    pub puts: u64,
    pub gets: u64,
    pub flushes: u64,
    pub compactions: u64,
    pub bloom_negatives: u64,
    pub mmap_block_reads: u64,
}

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub write_ops: usize,
    pub read_ops: usize,
    pub keyspace: usize,
    pub read_miss_ratio: f64,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            write_ops: 50_000,
            read_ops: 25_000,
            keyspace: 10_000,
            read_miss_ratio: 0.10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchReport {
    pub write_ops_per_sec: f64,
    pub fsync_every_n_records: usize,
    pub read_p50_us: u64,
    pub read_p95_us: u64,
    pub read_p99_us: u64,
    pub requested_read_miss_ratio: f64,
    pub observed_read_miss_ratio: f64,
    pub bloom_miss_rate: f64,
    pub mmap_reads: u64,
    pub hasher_impl: String,
}

#[derive(Debug, Clone)]
pub struct ActiveQuery {
    pub query: String,
    pub started_at_ms: u64,
    pub start_instant: Instant,
}

pub struct Database {
    root: PathBuf,
    config: Config,
    manifest: Arc<Mutex<Manifest>>,
    hasher: Arc<dyn Hasher + Send + Sync>,
    ai_runtime: Option<AiRuntimeHandle>,
    shard_senders: Vec<Sender<ShardCommand>>,
    shard_read_handles: Vec<ShardReadHandle>,
    shard_handles: Vec<JoinHandle<()>>,
    fast_write: Option<Arc<FastWritePath>>,
    durability_thread: Option<DurabilityThread>,
    /// Global epoch counter for EOAC (Epoch-Ordered Append-Only Concurrency).
    /// Monotonically increasing; advanced atomically on transaction commit.
    global_epoch: Arc<AtomicU64>,
    #[cfg(feature = "llm")]
    llm: Option<Arc<crate::ai::llm::LlmEngine>>,
    metrics: Arc<crate::util::metrics::MetricsRegistry>,
    startup_time: Instant,
    block_cache: Arc<BlockCache>,
    active_queries: Arc<Mutex<HashMap<u64, ActiveQuery>>>,
    next_query_id: Arc<AtomicU64>,
    audit_log: Arc<crate::auth::audit::AuditLog>,
    key_manager: Arc<crate::storage::key_manager::KeyManager>,
    compaction_window: Arc<parking_lot::RwLock<Option<(u8, u8)>>>,
}

impl Database {
    pub fn open(path: impl AsRef<Path>, config: Config) -> Result<Self> {
        config.validate()?;
        let root = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;

        let manifest = Manifest::load_or_create(&root, config.shard_count)?;
        if manifest.state.shards.len() != config.shard_count {
            return Err(TensorError::ManifestFormat(format!(
                "manifest shard count {} != config shard count {}",
                manifest.state.shards.len(),
                config.shard_count
            )));
        }

        let manifest = Arc::new(Mutex::new(manifest));
        let hasher = build_hasher();

        let block_cache = Arc::new(BlockCache::new(config.block_cache_bytes));
        let index_cache = Arc::new(IndexCache::new(config.index_cache_entries));

        let mut shard_senders = Vec::with_capacity(config.shard_count);
        let mut shard_read_handles = Vec::with_capacity(config.shard_count);
        let mut shard_handles = Vec::with_capacity(config.shard_count);
        let mut shard_runtimes_for_fast: Vec<(
            std::sync::Arc<crate::engine::shard::ShardShared>,
            PathBuf,
        )> = Vec::new();

        for shard_id in 0..config.shard_count {
            let (tx, rx) = unbounded();
            let shard_state = {
                let guard = manifest.lock();
                guard.state.shards[shard_id].clone()
            };

            let params = ShardOpenParams {
                shard_id,
                root: root.clone(),
                config: config.clone(),
                hasher: hasher.clone(),
                manifest: manifest.clone(),
                shard_state,
                block_cache: block_cache.clone(),
                index_cache: index_cache.clone(),
            };

            let (runtime, read_handle) = ShardRuntime::open(params)?;

            // Capture shared state and WAL path before moving runtime into thread
            if config.fast_write_enabled {
                shard_runtimes_for_fast.push((runtime.shared(), runtime.wal_path()));
            }

            let handle = thread::Builder::new()
                .name(format!("tensordb-shard-{shard_id}"))
                .spawn(move || runtime.run(rx))?;

            shard_senders.push(tx);
            shard_read_handles.push(read_handle);
            shard_handles.push(handle);
        }

        let ai_runtime = if config.ai_auto_insights {
            // Eagerly set has_subscribers so the fast write path falls back
            // to the channel path before the shard actors process Subscribe.
            for (shared, _) in &shard_runtimes_for_fast {
                shared
                    .has_subscribers
                    .store(true, std::sync::atomic::Ordering::Release);
            }

            let events = {
                let (tx, rx) = unbounded();
                for shard_tx in &shard_senders {
                    let _ = shard_tx.send(ShardCommand::Subscribe {
                        prefix: Vec::new(),
                        sender: tx.clone(),
                    });
                }
                rx
            };
            Some(AiRuntimeHandle::spawn(
                events,
                shard_senders.clone(),
                hasher.clone(),
                config.clone(),
            )?)
        } else {
            None
        };

        // Initialize fast write path
        let (fast_write, durability_thread) = if config.fast_write_enabled {
            let wal_queue = Arc::new(WalBatchQueue::new(config.shard_count));
            let wal_paths: Vec<PathBuf> = shard_runtimes_for_fast
                .iter()
                .map(|(_, p)| p.clone())
                .collect();

            let shard_states: Vec<FastShardState> = shard_runtimes_for_fast
                .into_iter()
                .zip(shard_senders.iter())
                .zip(std::iter::repeat(config.clone()))
                .map(|(((shared, _wal_path), sender), cfg)| FastShardState {
                    shared,
                    shard_sender: sender.clone(),
                    config: cfg,
                })
                .collect();

            let fast = Arc::new(FastWritePath::new(
                shard_states,
                wal_queue.clone(),
                hasher.clone(),
                config.fast_write_wal_batch_interval_us,
            ));

            let durability = DurabilityThread::spawn(
                wal_queue,
                wal_paths,
                config.fast_write_wal_batch_interval_us,
            );

            (Some(fast), Some(durability))
        } else {
            (None, None)
        };

        #[cfg(feature = "llm")]
        let llm = {
            const MODEL_FILENAME: &str = "Qwen3-0.6B-Q8_0.gguf";

            let model_path = if let Some(ref p) = config.llm_model_path {
                PathBuf::from(p)
            } else {
                // Search existing locations first; fall back to ~/.tensordb/models/
                // (the engine will auto-download there if needed)
                let candidates = [
                    root.join(".local/models").join(MODEL_FILENAME),
                    dirs::home_dir()
                        .unwrap_or_default()
                        .join(".tensordb/models")
                        .join(MODEL_FILENAME),
                    dirs::data_dir()
                        .unwrap_or_default()
                        .join("tensordb/models")
                        .join(MODEL_FILENAME),
                ];
                candidates
                    .iter()
                    .find(|p| p.exists())
                    .cloned()
                    .unwrap_or_else(|| {
                        // Default download location
                        dirs::home_dir()
                            .unwrap_or_else(|| PathBuf::from("."))
                            .join(".tensordb/models")
                            .join(MODEL_FILENAME)
                    })
            };

            let engine = crate::ai::llm::LlmEngine::new(model_path)
                .with_max_tokens(config.llm_max_tokens)
                .with_context_size(config.llm_context_size)
                .with_schema_cache_ttl(config.llm_schema_cache_ttl_secs)
                .with_grammar_constrained(config.llm_grammar_constrained);
            Some(Arc::new(engine))
        };

        // Initialize global epoch — start at 1 (0 reserved for non-transactional writes)
        let global_epoch = Arc::new(AtomicU64::new(1));

        let metrics = Arc::new(crate::util::metrics::MetricsRegistry::new(
            config.slow_query_threshold_us,
        ));

        let key_manager = Arc::new({
            let km = crate::storage::key_manager::KeyManager::new();
            if let Some(ref passphrase) = config.encryption_passphrase {
                let key = crate::storage::encryption::EncryptionKey::from_passphrase(passphrase);
                km.rotate_key(key);
            }
            km
        });

        Ok(Self {
            root,
            config,
            manifest,
            hasher,
            ai_runtime,
            shard_senders,
            shard_read_handles,
            shard_handles,
            fast_write,
            durability_thread,
            global_epoch,
            #[cfg(feature = "llm")]
            llm,
            metrics,
            startup_time: Instant::now(),
            block_cache,
            active_queries: Arc::new(Mutex::new(HashMap::new())),
            next_query_id: Arc::new(AtomicU64::new(1)),
            audit_log: Arc::new(crate::auth::audit::AuditLog::new()),
            key_manager,
            compaction_window: Arc::new(parking_lot::RwLock::new(None)),
        })
    }

    pub fn put(
        &self,
        user_key: &[u8],
        doc: Vec<u8>,
        valid_from: u64,
        valid_to: u64,
        schema_version: Option<u64>,
    ) -> Result<u64> {
        // Fast path: direct write, no channel
        if let Some(ref fast) = self.fast_write {
            if let Some(result) =
                fast.try_fast_put(user_key, &doc, valid_from, valid_to, schema_version)
            {
                return result;
            }
        }
        // Slow path: existing channel dispatch (unchanged, for backpressure fallback)
        let shard_id = self.shard_for(user_key);
        let (tx, rx) = bounded(1);
        self.shard_senders[shard_id]
            .send(ShardCommand::Put {
                user_key: user_key.to_vec(),
                doc,
                valid_from,
                valid_to,
                schema_version,
                resp: tx,
            })
            .map_err(|_| TensorError::ChannelClosed)?;
        rx.recv().map_err(|_| TensorError::ChannelClosed)?
    }

    /// Direct read — bypasses the shard actor channel entirely.
    pub fn get(
        &self,
        user_key: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
    ) -> Result<Option<Vec<u8>>> {
        let shard_id = self.shard_for(user_key);
        let result = self.shard_read_handles[shard_id].get(user_key, as_of, valid_at)?;
        Ok(result.value)
    }

    pub fn explain_get(
        &self,
        user_key: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
    ) -> Result<ExplainRow> {
        let shard_id = self.shard_for(user_key);
        let result = self.shard_read_handles[shard_id].get(user_key, as_of, valid_at)?;
        Ok(ExplainRow {
            shard_id,
            bloom_hit: result.bloom_hit,
            sstable_block: result.sstable_block,
            commit_ts_used: result.commit_ts_used,
        })
    }

    /// Direct prefix scan — bypasses the shard actor channel entirely.
    pub fn scan_prefix(
        &self,
        user_key_prefix: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<PrefixScanRow>> {
        if matches!(limit, Some(0)) {
            return Ok(Vec::new());
        }

        let mut merged = Vec::new();
        for read_handle in &self.shard_read_handles {
            let rows = read_handle.scan_prefix(user_key_prefix, as_of, valid_at, limit)?;
            merged.extend(rows.into_iter().map(|row| PrefixScanRow {
                user_key: row.user_key,
                doc: row.doc,
                commit_ts: row.commit_ts,
            }));
        }

        merged.sort_by(|a, b| a.user_key.cmp(&b.user_key));
        if let Some(limit) = limit {
            merged.truncate(limit);
        }
        Ok(merged)
    }

    pub fn write_batch(&self, entries: Vec<WriteBatchItem>) -> Result<Vec<u64>> {
        if entries.is_empty() {
            return Ok(Vec::new());
        }

        // Group entries by shard
        let mut by_shard: Vec<Vec<(usize, WriteBatchItem)>> =
            (0..self.config.shard_count).map(|_| Vec::new()).collect();
        for (orig_idx, entry) in entries.into_iter().enumerate() {
            let shard_id = self.shard_for(&entry.user_key);
            by_shard[shard_id].push((orig_idx, entry));
        }

        // Send batch commands to each shard
        let mut receivers = Vec::new();
        for (shard_id, shard_entries) in by_shard.into_iter().enumerate() {
            if shard_entries.is_empty() {
                continue;
            }
            let orig_indices: Vec<usize> = shard_entries.iter().map(|(i, _)| *i).collect();
            let items: Vec<WriteBatchItem> =
                shard_entries.into_iter().map(|(_, item)| item).collect();
            let (tx, rx) = bounded(1);
            self.shard_senders[shard_id]
                .send(ShardCommand::WriteBatch {
                    entries: items,
                    resp: tx,
                })
                .map_err(|_| TensorError::ChannelClosed)?;
            receivers.push((orig_indices, rx));
        }

        // Collect results and reassemble in original order
        let max_idx = receivers
            .iter()
            .flat_map(|(idxs, _)| idxs.iter())
            .copied()
            .max()
            .unwrap_or(0);
        let mut result = vec![0u64; max_idx + 1];

        for (orig_indices, rx) in receivers {
            let timestamps = rx.recv().map_err(|_| TensorError::ChannelClosed)??;
            for (i, ts) in orig_indices.into_iter().zip(timestamps) {
                result[i] = ts;
            }
        }

        Ok(result)
    }

    /// Write a batch of entries atomically with a single commit timestamp.
    /// All entries share the same epoch, making the batch appear as a single commit.
    pub fn write_batch_atomic(&self, entries: Vec<WriteBatchItem>) -> Result<u64> {
        if entries.is_empty() {
            return Ok(0);
        }

        // Allocate a single epoch for the entire batch
        let commit_ts = self
            .global_epoch
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;

        // Group entries by shard
        let mut by_shard: Vec<Vec<WriteBatchItem>> =
            (0..self.config.shard_count).map(|_| Vec::new()).collect();
        for entry in entries {
            let shard_id = self.shard_for(&entry.user_key);
            by_shard[shard_id].push(entry);
        }

        // Send batch commands to each shard
        let mut receivers = Vec::new();
        for (shard_id, shard_entries) in by_shard.into_iter().enumerate() {
            if shard_entries.is_empty() {
                continue;
            }
            let (tx, rx) = bounded(1);
            self.shard_senders[shard_id]
                .send(ShardCommand::WriteBatch {
                    entries: shard_entries,
                    resp: tx,
                })
                .map_err(|_| TensorError::ChannelClosed)?;
            receivers.push(rx);
        }

        // Wait for all shards to complete
        for rx in receivers {
            rx.recv().map_err(|_| TensorError::ChannelClosed)??;
        }

        Ok(commit_ts)
    }

    /// Subscribe to change events for keys matching a prefix.
    /// Returns a receiver that will receive ChangeEvent for each write.
    pub fn subscribe(&self, prefix: &[u8]) -> crossbeam_channel::Receiver<ChangeEvent> {
        // Eagerly mark all shards as having subscribers so the fast write path
        // falls back to the channel path (which emits change events).
        for rh in &self.shard_read_handles {
            rh.set_has_subscribers(true);
        }

        let (tx, rx) = unbounded();
        // Fan out to all shards
        for shard_tx in &self.shard_senders {
            let _ = shard_tx.send(ShardCommand::Subscribe {
                prefix: prefix.to_vec(),
                sender: tx.clone(),
            });
        }
        rx
    }

    /// Get the number of shards in this database.
    pub fn shard_count(&self) -> usize {
        self.config.shard_count
    }

    pub fn sql(&self, query: &str) -> Result<SqlResult> {
        // Register active query
        let query_id = self.next_query_id.fetch_add(1, Ordering::Relaxed);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.active_queries.lock().insert(
            query_id,
            ActiveQuery {
                query: if query.len() > 500 {
                    format!("{}...", &query[..500])
                } else {
                    query.to_string()
                },
                started_at_ms: now_ms,
                start_instant: Instant::now(),
            },
        );

        let start = Instant::now();
        let result = execute_sql(self, query);

        // Deregister active query
        self.active_queries.lock().remove(&query_id);

        let result = result?;
        let elapsed_us = start.elapsed().as_micros() as u64;

        // Record metrics
        let rows_returned = match &result {
            SqlResult::Rows(r) => Some(r.len() as u64),
            SqlResult::Affected { rows, .. } => Some(*rows),
            _ => None,
        };
        self.metrics
            .slow_query_log()
            .record(query, elapsed_us, rows_returned);
        self.metrics
            .counter("queries_total")
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .histogram("query_latency_us")
            .record(elapsed_us);

        // Invalidate LLM schema cache after DDL statements
        #[cfg(feature = "llm")]
        {
            let upper = query.trim().to_uppercase();
            if upper.starts_with("CREATE ")
                || upper.starts_with("DROP ")
                || upper.starts_with("ALTER ")
            {
                if let Some(ref llm) = self.llm {
                    llm.invalidate_schema_cache();
                }
            }
        }

        Ok(result)
    }

    /// Execute SQL using a persistent session handle, allowing transactions to
    /// span multiple calls. Used by the pgwire server.
    pub fn sql_session(&self, handle: &mut SqlSessionHandle, query: &str) -> Result<SqlResult> {
        let query_id = self.next_query_id.fetch_add(1, Ordering::Relaxed);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.active_queries.lock().insert(
            query_id,
            ActiveQuery {
                query: if query.len() > 500 {
                    format!("{}...", &query[..500])
                } else {
                    query.to_string()
                },
                started_at_ms: now_ms,
                start_instant: Instant::now(),
            },
        );

        let start = Instant::now();
        let result = execute_sql_with_session(self, handle, query);

        self.active_queries.lock().remove(&query_id);

        let result = result?;
        let elapsed_us = start.elapsed().as_micros() as u64;

        self.metrics
            .slow_query_log()
            .record(query, elapsed_us, None);
        self.metrics
            .counter("queries_total")
            .fetch_add(1, Ordering::Relaxed);
        self.metrics
            .histogram("query_latency_us")
            .record(elapsed_us);

        Ok(result)
    }

    /// Parse and plan a SQL statement once, returning a prepared statement that
    /// can be executed repeatedly without re-parsing.
    pub fn prepare(&self, sql: &str) -> Result<PreparedStatement> {
        PreparedStatement::new(sql)
    }

    /// Translate a natural language question to SQL using the embedded LLM,
    /// then execute the generated SQL. Returns `(generated_sql, result)`.
    #[cfg(feature = "llm")]
    pub fn ask(&self, question: &str) -> Result<(String, SqlResult)> {
        let llm = self.llm.as_ref().ok_or(TensorError::LlmNotAvailable)?;

        let (schema, _) = llm
            .schema_cache()
            .get_or_compute(|| self.gather_schema_context_inner(), |_| Vec::new())?;
        let sql = llm.nl_to_sql(question, &schema)?;
        match self.sql(&sql) {
            Ok(result) => Ok((sql, result)),
            Err(first_err) => {
                // Retry once with error feedback
                let retry_context = format!(
                    "{schema}\n\n\
                     Previous attempt generated: {sql}\n\
                     But it failed with: {first_err}\n\
                     Please generate a correct SQL statement."
                );
                let sql2 = llm.nl_to_sql(question, &retry_context)?;
                let result = self.sql(&sql2).map_err(|e| {
                    TensorError::LlmError(format!(
                        "generated SQL failed to execute: {e}\n  Generated SQL: {sql2}"
                    ))
                })?;
                Ok((sql2, result))
            }
        }
    }

    /// Translate a natural language question to SQL using the embedded LLM.
    /// Returns only the generated SQL without executing it.
    #[cfg(feature = "llm")]
    pub fn ask_sql(&self, question: &str) -> Result<String> {
        let llm = self.llm.as_ref().ok_or(TensorError::LlmNotAvailable)?;

        let (schema, _) = llm
            .schema_cache()
            .get_or_compute(|| self.gather_schema_context_inner(), |_| Vec::new())?;
        llm.nl_to_sql(question, &schema)
    }

    /// Build a schema context string for the LLM prompt by listing all tables
    /// and their column definitions. This is the inner implementation used by
    /// the schema cache.
    #[cfg(feature = "llm")]
    fn gather_schema_context_inner(&self) -> Result<String> {
        use crate::sql::exec::SqlResult;

        let mut ctx = String::new();

        // Get table list — use execute_sql directly to avoid DDL cache invalidation
        let tables_result = execute_sql(self, "SHOW TABLES")?;
        let table_names: Vec<String> = match tables_result {
            SqlResult::Rows(rows) => rows
                .into_iter()
                .filter_map(|row| {
                    serde_json::from_slice::<serde_json::Value>(&row)
                        .ok()
                        .and_then(|v| v.get("table").and_then(|t| t.as_str()).map(String::from))
                        .or_else(|| String::from_utf8(row).ok())
                })
                .collect(),
            _ => Vec::new(),
        };

        for table in &table_names {
            let describe_sql = format!("DESCRIBE {table}");
            if let Ok(desc) = execute_sql(self, &describe_sql) {
                // CREATE TABLE format — matches what the model saw in training.
                let mut cols = Vec::new();
                if let SqlResult::Rows(rows) = desc {
                    for row in rows {
                        if let Ok(text) = String::from_utf8(row) {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                                let name = v.get("column").and_then(|c| c.as_str()).unwrap_or("?");
                                let dtype =
                                    v.get("type").and_then(|t| t.as_str()).unwrap_or("TEXT");
                                cols.push(format!("{name} {dtype}"));
                            } else {
                                let trimmed = text.trim();
                                if !trimmed.is_empty() {
                                    cols.push(trimmed.to_string());
                                }
                            }
                        }
                    }
                }
                ctx.push_str(&format!("CREATE TABLE {table} ({});\n", cols.join(", ")));
            }
        }

        if ctx.is_empty() {
            ctx.push_str("(no tables exist yet)\n");
        }

        Ok(ctx)
    }

    /// Translate a natural language question to SQL using the embedded LLM
    /// with Hermes-style tool calling for iterative schema discovery.
    ///
    /// The model can call `list_tables`, `describe_table`, and `execute_sql`
    /// tools to discover the schema and validate SQL before returning a final answer.
    /// Returns `(generated_sql, execution_result)`.
    #[cfg(feature = "llm")]
    pub fn ask_with_tools(&self, question: &str) -> Result<(String, SqlResult)> {
        let llm = self.llm.as_ref().ok_or(TensorError::LlmNotAvailable)?;

        let (schema, _) = llm
            .schema_cache()
            .get_or_compute(|| self.gather_schema_context_inner(), |_| Vec::new())?;

        let result = crate::ai::tool_calling::nl_to_sql_with_tools(llm, question, &schema, self)?;

        match self.sql(&result.sql) {
            Ok(exec_result) => Ok((result.sql, exec_result)),
            Err(e) => Err(TensorError::LlmError(format!(
                "tool-calling generated SQL failed to execute: {e}\n  Generated SQL: {}",
                result.sql
            ))),
        }
    }

    pub fn ai_stats(&self) -> AiRuntimeStats {
        self.ai_runtime
            .as_ref()
            .map(|r| r.stats())
            .unwrap_or_default()
    }

    pub fn ai_insights_for_key(
        &self,
        user_key: &[u8],
        limit: Option<usize>,
    ) -> Result<Vec<AiInsight>> {
        let prefix = insight_prefix_for_source(user_key);
        let rows = self.scan_prefix(&prefix, None, None, limit)?;
        let mut out = Vec::new();
        for row in rows {
            if let Ok(insight) = serde_json::from_slice::<AiInsight>(&row.doc) {
                out.push(insight);
            }
        }
        out.sort_by(|a, b| b.source_commit_ts.cmp(&a.source_commit_ts));
        Ok(out)
    }

    pub fn ai_insight_by_id(&self, insight_id: &str) -> Result<Option<AiInsight>> {
        let Some((source_key, commit_ts)) = parse_insight_id(insight_id) else {
            return Ok(None);
        };
        let storage_key = insight_storage_key(&source_key, commit_ts);
        let row = self.get(&storage_key, None, None)?;
        let Some(row) = row else {
            return Ok(None);
        };
        let insight = serde_json::from_slice::<AiInsight>(&row).ok();
        Ok(insight)
    }

    pub fn ai_correlation_for_cluster(
        &self,
        cluster_id: &str,
        limit: Option<usize>,
    ) -> Result<Vec<AiCorrelationRef>> {
        let prefix = correlation_prefix_for_cluster(cluster_id);
        let rows = self.scan_prefix(&prefix, None, None, limit)?;
        let mut out = Vec::new();
        for row in rows {
            if let Ok(entry) = serde_json::from_slice::<AiCorrelationRef>(&row.doc) {
                out.push(entry);
            }
        }
        out.sort_by(|a, b| b.source_commit_ts.cmp(&a.source_commit_ts));
        Ok(out)
    }

    pub fn ai_correlation_for_key(
        &self,
        user_key: &[u8],
        limit: Option<usize>,
    ) -> Result<Vec<AiCorrelationRef>> {
        let latest = self.ai_insights_for_key(user_key, Some(1))?;
        let Some(insight) = latest.first() else {
            return Ok(Vec::new());
        };
        self.ai_correlation_for_cluster(&insight.cluster_id, limit)
    }

    /// Combines writer stats (from shard channels) with reader stats (from atomics).
    pub fn stats(&self) -> Result<DbStats> {
        let mut stats = DbStats {
            shard_count: self.config.shard_count,
            ..DbStats::default()
        };

        // Writer stats via channel
        for tx in &self.shard_senders {
            let (stx, srx) = bounded(1);
            tx.send(ShardCommand::Stats { resp: stx })
                .map_err(|_| TensorError::ChannelClosed)?;
            let shard_stats: ShardStats = srx.recv().map_err(|_| TensorError::ChannelClosed)?;
            stats.puts += shard_stats.puts;
            stats.gets += shard_stats.gets;
            stats.flushes += shard_stats.flushes;
            stats.compactions += shard_stats.compactions;
            stats.bloom_negatives += shard_stats.bloom_negatives;
            stats.mmap_block_reads += shard_stats.mmap_block_reads;
        }

        // Reader stats from atomics (direct reads bypass the channel)
        for rh in &self.shard_read_handles {
            let reader = rh.reader_stats();
            stats.gets += reader.gets;
            stats.bloom_negatives += reader.bloom_negatives;
            stats.mmap_block_reads += reader.mmap_block_reads;
        }

        Ok(stats)
    }

    pub fn bench(&self, opts: BenchOptions) -> Result<BenchReport> {
        let read_miss_ratio = opts.read_miss_ratio.clamp(0.0, 1.0);
        let start = Instant::now();
        for i in 0..opts.write_ops {
            let key = format!("bench/{:08}", i % opts.keyspace);
            let value = format!("{{\"n\":{i}}}").into_bytes();
            let _ = self.put(key.as_bytes(), value, 0, u64::MAX, Some(1))?;
        }
        let write_elapsed = start.elapsed().as_secs_f64().max(0.000_001);
        let write_ops_per_sec = opts.write_ops as f64 / write_elapsed;

        let mut samples = Vec::with_capacity(opts.read_ops);
        let stats_before = self.stats()?;
        let mut read_misses = 0usize;

        for _ in 0..opts.read_ops {
            let miss = fastrand::f64() < read_miss_ratio;
            let n = fastrand::usize(..opts.keyspace);
            let key = if miss {
                read_misses += 1;
                format!("bench-miss/{:08}", n + opts.keyspace)
            } else {
                format!("bench/{:08}", n)
            };
            let t0 = Instant::now();
            let _ = self.get(key.as_bytes(), None, None)?;
            let dt = t0.elapsed();
            samples.push(dt.as_micros() as u64);
        }

        samples.sort_unstable();
        let p50 = percentile(&samples, 0.50);
        let p95 = percentile(&samples, 0.95);
        let p99 = percentile(&samples, 0.99);

        let stats_after = self.stats()?;
        let gets_delta = (stats_after.gets.saturating_sub(stats_before.gets)).max(1);
        let bloom_delta = stats_after
            .bloom_negatives
            .saturating_sub(stats_before.bloom_negatives);
        let mmap_delta = stats_after
            .mmap_block_reads
            .saturating_sub(stats_before.mmap_block_reads);

        Ok(BenchReport {
            write_ops_per_sec,
            fsync_every_n_records: self.config.wal_fsync_every_n_records,
            read_p50_us: p50,
            read_p95_us: p95,
            read_p99_us: p99,
            requested_read_miss_ratio: read_miss_ratio,
            observed_read_miss_ratio: read_misses as f64 / opts.read_ops.max(1) as f64,
            bloom_miss_rate: bloom_delta as f64 / gets_delta as f64,
            mmap_reads: mmap_delta,
            hasher_impl: self.hasher.name().to_string(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.manifest.lock().path().to_path_buf()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn metrics(&self) -> &Arc<crate::util::metrics::MetricsRegistry> {
        &self.metrics
    }

    pub fn uptime_ms(&self) -> u64 {
        self.startup_time.elapsed().as_millis() as u64
    }

    pub fn block_cache(&self) -> &Arc<BlockCache> {
        &self.block_cache
    }

    pub fn active_queries_snapshot(&self) -> Vec<(u64, ActiveQuery)> {
        self.active_queries
            .lock()
            .iter()
            .map(|(&id, q)| (id, q.clone()))
            .collect()
    }

    pub fn storage_info(&self) -> Vec<(usize, ShardStorageInfo)> {
        self.shard_read_handles
            .iter()
            .enumerate()
            .map(|(i, rh)| (i, rh.storage_info()))
            .collect()
    }

    pub fn wal_sizes(&self) -> Vec<(usize, u64)> {
        (0..self.config.shard_count)
            .map(|shard_id| {
                let wal_path = self.root.join(format!("shard-{shard_id}")).join("wal.log");
                let size = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
                (shard_id, size)
            })
            .collect()
    }

    pub fn audit_log(&self) -> &Arc<crate::auth::audit::AuditLog> {
        &self.audit_log
    }

    pub fn key_manager(&self) -> &Arc<crate::storage::key_manager::KeyManager> {
        &self.key_manager
    }

    pub fn compaction_window(&self) -> Option<(u8, u8)> {
        *self.compaction_window.read()
    }

    pub fn set_compaction_window(&self, start_hour: u8, end_hour: u8) {
        *self.compaction_window.write() = Some((start_hour, end_hour));
    }

    pub fn clear_compaction_window(&self) {
        *self.compaction_window.write() = None;
    }

    /// Request compaction on a specific shard.
    pub fn request_compaction(&self, shard_id: usize) -> Result<()> {
        if shard_id >= self.shard_senders.len() {
            return Err(crate::error::sql_exec_err(format!(
                "shard {shard_id} does not exist"
            )));
        }
        let (tx, rx) = crossbeam_channel::bounded(1);
        let _ = self.shard_senders[shard_id].send(ShardCommand::ForceCompaction { resp: tx });
        rx.recv().map_err(|_| TensorError::ChannelClosed)?
    }

    /// Request compaction on all shards.
    pub fn request_compaction_all(&self) -> Result<()> {
        for shard_id in 0..self.shard_senders.len() {
            self.request_compaction(shard_id)?;
        }
        Ok(())
    }

    /// Ensure all pending fast-path WAL records have been flushed to disk.
    /// Call this when you need durability guarantees after fast-path writes.
    pub fn sync(&self) {
        if let Some(ref dt) = self.durability_thread {
            dt.sync();
        }
    }

    /// Returns the current global epoch.
    pub fn current_epoch(&self) -> u64 {
        self.global_epoch.load(Ordering::Acquire)
    }

    /// Atomically advances the global epoch and returns the new epoch value.
    /// Called by transaction COMMIT to assign a commit epoch.
    /// Also bumps all shard commit counters to at least the new epoch so that
    /// PITR queries using `AS OF EPOCH` can correctly filter cross-shard data:
    /// all writes after this epoch will have commit_ts > epoch.
    pub fn advance_epoch(&self) -> u64 {
        let new_epoch = self.global_epoch.fetch_add(1, Ordering::SeqCst) + 1;
        // Ensure all shard commit counters are at least new_epoch.
        // This guarantees that writes AFTER this epoch will have commit_ts > new_epoch,
        // so querying AS OF EPOCH new_epoch will correctly exclude future writes.
        for rh in &self.shard_read_handles {
            rh.bump_commit_counter(new_epoch);
        }
        new_epoch
    }

    pub(crate) fn shard_for(&self, key: &[u8]) -> usize {
        (self.hasher.hash64(key) as usize) % self.config.shard_count
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Disable fast write path first so no new writes go through it
        if let Some(ref fast) = self.fast_write {
            fast.disable();
        }
        // Flush pending WAL records
        if let Some(ref mut dt) = self.durability_thread {
            dt.shutdown();
        }
        if let Some(ai) = &mut self.ai_runtime {
            ai.shutdown();
        }
        for tx in &self.shard_senders {
            let _ = tx.send(ShardCommand::Shutdown);
        }
        while let Some(handle) = self.shard_handles.pop() {
            let _ = handle.join();
        }
    }
}

fn percentile(samples: &[u64], p: f64) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let idx = ((samples.len() - 1) as f64 * p).round() as usize;
    samples[idx.min(samples.len() - 1)]
}

// ── ToolExecutor implementation for Database ────────────────────────────

#[cfg(feature = "llm")]
impl crate::ai::tool_calling::ToolExecutor for Database {
    fn list_tables(&self) -> Result<Vec<String>> {
        let result = execute_sql(self, "SHOW TABLES")?;
        match result {
            SqlResult::Rows(rows) => Ok(rows
                .into_iter()
                .filter_map(|row| {
                    serde_json::from_slice::<serde_json::Value>(&row)
                        .ok()
                        .and_then(|v| v.get("table").and_then(|t| t.as_str()).map(String::from))
                        .or_else(|| String::from_utf8(row).ok())
                })
                .collect()),
            _ => Ok(Vec::new()),
        }
    }

    fn describe_table(&self, name: &str) -> Result<Vec<(String, String)>> {
        let describe_sql = format!("DESCRIBE {name}");
        let result = execute_sql(self, &describe_sql)?;
        match result {
            SqlResult::Rows(rows) => {
                let mut columns = Vec::new();
                for row in rows {
                    if let Ok(text) = String::from_utf8(row) {
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                            let col_name = v
                                .get("column")
                                .and_then(|c| c.as_str())
                                .unwrap_or("?")
                                .to_string();
                            let dtype = v
                                .get("type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("TEXT")
                                .to_string();
                            columns.push((col_name, dtype));
                        }
                    }
                }
                Ok(columns)
            }
            _ => Ok(Vec::new()),
        }
    }

    fn execute_sql_tool(&self, sql: &str) -> String {
        match execute_sql(self, sql) {
            Ok(SqlResult::Rows(rows)) => {
                let json_rows: Vec<serde_json::Value> = rows
                    .into_iter()
                    .filter_map(|row| {
                        String::from_utf8(row)
                            .ok()
                            .and_then(|s| serde_json::from_str(&s).ok())
                    })
                    .collect();
                serde_json::json!({ "rows": json_rows }).to_string()
            }
            Ok(SqlResult::Affected { rows, message, .. }) => {
                serde_json::json!({ "affected_rows": rows, "message": message }).to_string()
            }
            Ok(SqlResult::Explain(plan)) => serde_json::json!({ "plan": plan }).to_string(),
            Err(e) => serde_json::json!({ "error": e.to_string() }).to_string(),
        }
    }
}
