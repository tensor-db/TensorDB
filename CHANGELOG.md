# Changelog

All notable changes to TensorDB are documented in this file.

## [0.30] — Advanced Vector Search, Horizontal Scaling, Ecosystem

### Added

#### Vector Search
- **`VECTOR(n)` column type** — First-class vector storage with dimension validation
- **`CREATE VECTOR INDEX ... USING HNSW`** — Configurable HNSW indexes with `m`, `ef_construction`, and `metric` parameters
- **`CREATE VECTOR INDEX ... USING IVF_PQ`** — IVF-PQ indexes with `nlist`, `nprobe`, `pq_m`, and `pq_bits` parameters
- **`<->` distance operator** — k-NN search via `ORDER BY embedding <-> '[0.1, ...]' LIMIT k`
- **`vector_search()` table function** — Direct k-NN search returning pk, distance, score, and rank
- **`HYBRID_SCORE()` function** — Weighted combination of vector similarity and BM25 text relevance
- **Temporal vector queries** — `FOR SYSTEM_TIME AS OF` applied to vector search
- **Vector scalar functions** — `VECTOR_DISTANCE`, `COSINE_SIMILARITY`, `VECTOR_NORM`, `VECTOR_DIMS`
- **Vector quantization** — FP16 (via `half` crate) and INT8 scalar quantization, PQ codebook with k-means
- **IVF index** — K-means centroids, cell assignment, nprobe multi-cell search
- **Vector persistence** — LSM-backed storage under `__vec/` key prefix with automatic index maintenance on INSERT/UPDATE/DELETE
- **`DROP VECTOR INDEX`** — Clean removal of vector indexes and associated data
- 18 new integration tests in `tests/vector_search_sql.rs`

#### Horizontal Scaling (`tensordb-distributed` crate)
- **Shard routing** — Consistent hash ring with virtual nodes for key-to-shard mapping
- **Distributed database** — `DistributedDatabase` wrapper routing operations to local or remote shards
- **2PC distributed transactions** — `TxnCoordinator` and `TxnParticipant` with prepare/commit/abort protocol
- **Durable transaction log** — Append-only JSON WAL (`TxnLog`) for crash recovery of in-doubt transactions
- **Online shard rebalancing** — `ShardMigrator` state machine (Planning → Freezing → SnapshotStreaming → CatchingUp → Cutover → Complete)
- **Migration throttling** — Token-bucket `MigrationThrottle` rate limiter for background migration traffic
- **Gossip discovery** — Seed-based node discovery with membership merge protocol
- **Phi-accrual failure detector** — `HealthChecker` with adaptive heartbeat-based failure detection
- **Cluster node management** — `ClusterNode` with peer tracking, shard ownership, role management
- gRPC service definitions in `proto/tensor_cluster.proto` (ready for tonic integration)
- 25 unit tests across all distributed modules

#### Ecosystem
- **Dockerfile** — Multi-stage build producing minimal tensordb-server image (no LLM)
- **docker-compose.yml** — Ready-to-use setup with volume mount, healthcheck, port mapping
- **GitHub Actions: publish-crates.yml** — Automated crates.io publishing on release
- **GitHub Actions: publish-docker.yml** — Automated Docker image publishing to ghcr.io on release
- **FastAPI example app** — `examples/fastapi_app/app.py` with CRUD endpoints and vector search
- **Express.js example app** — `examples/express_app/index.js` with CRUD endpoints

### Changed
- `VectorError(String)` variant added to `TensorError` enum
- Vector config defaults added: `vector_hnsw_m`, `vector_hnsw_ef_construction`, `vector_ivf_threshold`, `vector_default_encoding`
- `SqlType::Vector { dims }` variant added to SQL type system
- `ColumnVector::Vector` variant added to vectorized batch engine
- `PlanNode::VectorSearch` variant added to query planner

## [0.29] — EOAC Architecture: Transactions, PITR, Incremental Backup

### Added
- **Epoch-Ordered Append-Only Concurrency (EOAC)** — Global `AtomicU64` epoch counter unifying transactions, MVCC, recovery, and time travel
- `BEGIN`/`COMMIT`/`ROLLBACK`/`SAVEPOINT` transactions with epoch-numbered commits
- `SELECT ... AS OF EPOCH <n>` for cross-shard point-in-time recovery
- `BACKUP TO '<path>' SINCE EPOCH <n>` for incremental backup (delta export since a given epoch)
- `BACKUP TO '<path>'` / `RESTORE FROM '<path>'` for full backup and restore
- `TXN_COMMIT` markers storing `max_commit_ts` for epoch→snapshot resolution
- Cross-shard epoch synchronization via `bump_commit_counter()` — ensures PITR correctness across shards
- `Database::advance_epoch()` API for manual epoch advancement
- `Database::current_epoch()` API for querying the current epoch
- **Encryption at rest** — AES-256-GCM block-level encryption for SSTable data and WAL frames (`--features encryption`)
- `EncryptionKey` with passphrase derivation (SHA-256) and key file support (32-byte raw or 64-char hex)
- WAL epoch tracking: `replay_until_epoch()`, `replay_epoch_range()`, `max_epoch()` methods
- 23 new integration tests in `tests/epoch_core.rs` covering transactions, PITR, and incremental backup

### Changed
- `advance_epoch()` now bumps all shard commit counters to the epoch value for cross-shard consistency
- Transaction `COMMIT` stores `max_commit_ts` (maximum across all writes) instead of `last_commit_ts`
- Query planner: `#[allow(clippy::too_many_arguments)]` on multi-parameter join functions

## [0.2.0] — Embedded LLM: Natural Language → SQL

### Added
- Embedded Qwen3-0.6B model for natural language to SQL translation via pure-Rust native inference engine
- `ASK '<question>'` SQL statement — translates NL to SQL and executes in one step
- `Database::ask()` and `Database::ask_sql()` Rust API methods
- `PyDatabase.ask()` Python binding returning `{"sql": ..., "result": ...}`
- `# <question>` prefix in both Rust and Python CLI shells (shows generated SQL, asks to confirm)
- `LlmEngine` with lazy model loading, greedy sampling, and output cleaning
- `llm` feature flag (default on) — pure Rust, no C++ dependencies
- `llm_model_path` and `llm_max_tokens` config fields
- `LlmNotAvailable` and `LlmError` error variants
- Schema-aware prompting: gathers table schemas via SHOW TABLES + DESCRIBE before translation
- **GGUF v3 loader** (`src/ai/gguf.rs`) — mmap-backed zero-copy tensor access, dequantization for Q8_0, Q4_0, F16, F32
- **BPE tokenizer** (`src/ai/tokenizer.rs`) — vocabulary and merge rules loaded from GGUF metadata, ChatML special tokens
- **Transformer runtime** (`src/ai/transformer.rs`) — Qwen2 architecture with RMSNorm, GQA attention, RoPE, SwiGLU FFN, KV cache
- **Token sampler** (`src/ai/sampler.rs`) — greedy (argmax), top-p nucleus sampling, temperature scaling, repetition penalty
- **SQL grammar decoder** (`src/ai/sql_grammar.rs`) — soft-constraint token filtering that biases generation toward valid SQL
- **Schema cache** (`src/ai/schema_cache.rs`) — TTL-based schema context caching with DDL invalidation
- **KV cache prefix reuse** — schema prompt KV state cached and reused across calls for ~3-15x speedup on repeated queries
- `llm_context_size`, `llm_schema_cache_ttl_secs`, `llm_grammar_constrained`, `llm_kv_cache_prefix` config fields

### Changed
- All crate versions bumped to 0.2.0
- Replaced `llama-cpp-2` C++ FFI dependency with pure-Rust inference engine — no C++ toolchain required
- Removed `encoding_rs` dependency (using std)
- DDL statements (CREATE/DROP/ALTER TABLE) now automatically invalidate the LLM schema cache

## [0.28] — Fast Write Engine

### Added
- Lock-free `FastWritePath` bypassing crossbeam channel (~1.9 µs writes, 20x faster than SQLite)
- Group-commit WAL via `DurabilityThread` — one fdatasync per batch cycle across all shards
- `fast_write_enabled` config (default: true) and `fast_write_wal_batch_interval_us` (default: 1000)
- Automatic fallback to channel path on memtable backpressure or subscriber activity
- `Database::sync()` for explicit durability flush
- Channel-path benchmark (`tensordb_point_write_channel`) for comparison

### Changed
- `ShardShared` fields now `pub(crate)` for direct access by fast write path
- `commit_counter` uses `AtomicU64` with `fetch_add` for lock-free timestamp assignment
- `has_subscribers` AtomicBool for fast-path/channel-path routing
- Shard actor syncs `local_commit_counter` from atomic before every write

## [0.27] — Replication Foundations

### Added
- Raft consensus: `RaftNode` with leader election, vote request/response, log replication
- Cluster membership: `NodeRegistry` with heartbeat, status tracking, shard assignments
- WAL shipping: `WalShipper` and `WalReceiver` for primary-to-standby replication
- Read replica routing with staleness tolerance
- Consistent hash ring for distributed shard routing
- Scatter-gather executor with merge strategies (concatenate, sort-merge, aggregate)
- Failover readiness scoring for standbys

## [0.26] — Schema Evolution

### Added
- `MigrationManager` with register, apply, rollback, apply_all
- `SchemaRegistry` with per-table column versioning and schema diff
- Migration tracking under `__schema/migration/` prefix

## [0.25] — Monitoring & Diagnostics

### Added
- `MetricsRegistry` with counters, gauges, and HDR histograms
- `SlowQueryLog` ring buffer with configurable threshold
- `TimerGuard` RAII wrapper for automatic latency recording
- JSON-serializable `MetricsSnapshot`

## [0.24] — Connection Pooling

### Added
- `ConnectionPool` with configurable max connections, min idle, idle timeout
- Pool warmup (pre-create min_idle connections)
- RAII `PooledConnection` guard with auto-release on drop
- Pool statistics: active/idle/total, acquired/released/timeout counts
- Idle eviction and LIFO connection reuse

## [0.23] — Authentication & Authorization

### Added
- `UserManager`: create, authenticate, change password, disable, grant/revoke roles
- `RoleManager`: built-in admin/reader/writer roles, custom roles
- Table-level permissions with `Privilege` enum (Select, Insert, Update, Delete, Create, Drop, Alter, Admin)
- `RbacChecker::check_privilege()` with direct + role-based resolution
- `SessionStore` with token-based TTL sessions

## [0.22] — Event Sourcing

### Added
- `create_event_store()`, `append_event()`, `get_events()`
- Aggregate projections via `get_aggregate_state()` with event replay
- Snapshot support via `save_snapshot()` for faster aggregate loading
- Idempotency keys under `__es/idem/` prefix
- `find_aggregates_by_event_type()` for cross-aggregate queries

## [0.21] — Change Data Capture v2

### Added
- `DurableCursor` with at-least-once delivery and position persistence
- `ConsumerGroupManager` with round-robin shard assignment and generation tracking
- Exactly-once ACK semantics

## [0.20] — Columnar Storage (Partial)

### Added
- Zone maps: per-block min/max, null count, HyperLogLog distinct count
- Dictionary encoding for low-cardinality string columns
- `APPROX_COUNT_DISTINCT` via HyperLogLog (256 registers, mergeable)

## [0.19] — Vectorized Execution

### Added
- `RecordBatch` columnar representation with typed `ColumnVector` (Int64, Float64, Boolean, Utf8)
- Vectorized operators: filter, project, sort, limit, hash aggregate, hash join
- Selection vector boolean combinators (and, or, not)
- Default batch size: 1024 rows

## [0.18] — Data Interchange

### Added
- Parquet read/write via `COPY TO/FROM` (behind `--features parquet`)
- `read_parquet()`, `read_csv()`, `read_json()` table functions
- CSV with RFC 4180 parsing, NDJSON streaming import/export
- Arrow in-memory columnar format for query execution

## [0.17] — SQL Completeness

### Added
- `CASE WHEN ... THEN ... ELSE ... END` expressions
- `CAST(expr AS type)` with type coercion
- `UNION`, `UNION ALL`, `INTERSECT`, `EXCEPT` set operations
- `INSERT ... RETURNING`
- `CREATE TABLE ... AS SELECT`
- 17 string functions, 13 math functions, 6 date/time functions
- `NULLIF`, `GREATEST`, `LEAST`, `IF`/`IIF`
- `STDDEV_POP`, `STDDEV_SAMP`, `VAR_POP`, `VAR_SAMP`, `STRING_AGG`
- `LIKE` / `ILIKE` pattern matching

## [0.15] — PostgreSQL Wire Protocol

### Added
- `tensordb-server` crate with pgwire v3 protocol implementation
- Simple query mode and extended query mode (Parse/Bind/Describe/Execute/Sync)
- Type OID mapping (BOOL, INT4, INT8, FLOAT8, TEXT, JSON, JSONB, TIMESTAMP, BYTEA)
- Password authentication

## [0.14] — Time-Series SQL

### Added
- `CREATE TIMESERIES TABLE ... WITH (bucket_size = '...')`
- `TIME_BUCKET()` and `TIME_BUCKET_GAPFILL()`
- `LOCF` (last observation carried forward) and `INTERPOLATE` (linear)
- `DELTA()` and `RATE()` functions
- `first()`/`last()` time-weighted aggregates

## [0.13] — Full-Text Search SQL

### Added
- `CREATE FULLTEXT INDEX ... ON table (columns)`
- `MATCH(column, 'query')` with BM25 relevance ranking
- `HIGHLIGHT(column, 'query')` with match markers
- Multi-column FTS with per-column boosting
- Automatic posting list maintenance on INSERT/UPDATE/DELETE

## [0.11] — Temporal SQL:2011

### Added
- `FOR SYSTEM_TIME AS OF`, `FROM...TO`, `BETWEEN...AND`, `ALL`
- `FOR APPLICATION_TIME AS OF`, `FROM...TO`, `BETWEEN...AND`

## [0.10] — Query Engine Performance

### Added
- Cost-based query planner with `PlanNode` tree and cost estimation
- `EXPLAIN ANALYZE` with execution_time_us, rows_returned, read/write ops, plan cost
- Prepared statements with `$1, $2, ...` parameter binding
- `ANALYZE table` for per-column statistics (distinct, null, min/max, top-N histograms)

## [0.9] — Storage Performance

### Added
- LZ4 block compression (SSTable V2 format)
- Adaptive compression (per-block codec selection)
- Batched WAL fsync with group commit window
- Bloom filter false positive rate tracking

## [0.6–0.7] — Storage & SQL Correctness

### Fixed
- Manifest persistence with per-level SSTable metadata
- Block and index cache wired into SSTable reader
- Immutable memtable queue (freeze → queue → background flush)
- Binary search probe for internal keys
- Numeric ORDER BY (typed comparison instead of lexicographic)
- Window function evaluation order (computed before LIMIT)
- Transaction-local reads (pending writes visible in SELECT)

### Added
- Typed delete tombstones
- WAL rotation (file-per-generation)

## [0.5] — Ecosystem

### Added
- Full-text search facet (inverted index, tokenizer, stemmer)
- Time-series facet (bucketed storage, downsampling, range queries)
- Streaming change feeds with prefix-filtered subscriptions
- io_uring async I/O for WAL and SSTable reads (`--features io-uring`)
- Comparative benchmark harness (TensorDB vs SQLite)

## [0.4] — SQL Surface & Developer Experience

### Added
- Subqueries and CTEs (`WITH ... AS`)
- Window functions: ROW_NUMBER, RANK, DENSE_RANK, LEAD, LAG
- Typed column schema with DDL enforcement
- Columnar row encoding with null bitmaps
- `COPY` for bulk import/export
- Python bindings (PyO3)
- Node.js bindings (napi-rs)

## [0.3] — Storage & Performance

### Added
- Multi-level compaction with size-budgeted leveling (L0 → L6)
- Block and index caching with configurable memory budgets (LRU)
- Prefix compression and restart points in SSTable blocks
- Write-batch API for atomic multi-key bulk ingest
- SIMD-accelerated bloom probes and checksums (`--features simd`)

## [0.2] — Query Engine

### Added
- Expression AST with full precedence parsing
- `WHERE` clauses with comparison operators and field access
- `UPDATE` and `DELETE` with temporal-aware semantics
- `JOIN` (inner, left, right, cross) with arbitrary ON clauses
- Aggregates: `SUM`, `AVG`, `MIN`, `MAX`
- `GROUP BY` on arbitrary expressions
- `HAVING` clause

## [0.1.0] — 2026-02-25

### Added
- Append-only fact ledger with WAL and CRC-framed records
- MVCC snapshot reads (`AS OF <commit_ts>`)
- Bitemporal filtering (`VALID AT <valid_ts>`)
- Sharded single-writer execution model
- LSM-style SSTables with bloom filters, block index, and mmap reads
- SQL subset: CREATE TABLE, INSERT, SELECT, DROP, SHOW TABLES, DESCRIBE, EXPLAIN
- Interactive CLI shell with TAB completion, persistent history, output modes
- Optional C++ acceleration (`--features native`)
- Benchmark harness with configurable workload matrix
