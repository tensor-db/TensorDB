# Changelog

All notable changes to TensorDB are documented in this file.

## [0.2.0] — Embedded LLM: Natural Language → SQL

### Added
- Embedded Qwen3-0.6B model for natural language to SQL translation via llama.cpp
- `ASK '<question>'` SQL statement — translates NL to SQL and executes in one step
- `Database::ask()` and `Database::ask_sql()` Rust API methods
- `PyDatabase.ask()` Python binding returning `{"sql": ..., "result": ...}`
- `# <question>` prefix in both Rust and Python CLI shells (shows generated SQL, asks to confirm)
- `LlmEngine` with lazy model loading, greedy sampling, and output cleaning
- `llm` feature flag (default on) backed by `llama-cpp-2` crate
- `llm_model_path` and `llm_max_tokens` config fields
- `LlmNotAvailable` and `LlmError` error variants
- Schema-aware prompting: gathers table schemas via SHOW TABLES + DESCRIBE before translation

### Changed
- All crate versions bumped to 0.2.0

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
