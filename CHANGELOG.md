# Changelog

All notable changes to TensorDB are documented in this file.

## [0.46] — Category Differentiation

### Added
- **Learned cost model** (`sql/learned_cost.rs`) — Online linear regression trained from observed query execution, per-plan-type weights via SGD with EMA decay
- **Anomaly detection** (`ai/anomaly.rs`) — Per-table write rate statistics using exponential moving average, flags events exceeding configurable sigma threshold
- **Graph queries** (`facet/graph.rs`) — Edge tables with adjacency/reverse adjacency lists, BFS/DFS traversal, shortest path queries, neighbor lookups

## [0.42–0.45] — Distributed & Cloud

### Added
- **Raft consensus** (`consensus/raft.rs`) — Leader election, vote request/response, log replication with AppendEntries, commit index tracking
- **Object store** (`storage/object_store.rs`) — `ObjectStore` trait with `LocalObjectStore` (filesystem) and `MemoryObjectStore` (testing) implementations
- **WAL replication** (`replication/wal_shipper.rs`) — Background WAL frame distribution to followers with per-follower offset tracking and backpressure
- **C FFI** (`ffi.rs`) — `tensordb_open()`, `tensordb_sql()`, `tensordb_close()`, `tensordb_version()` extern "C" functions for mobile and edge deployment

## [0.39–0.41] — Enterprise Security

### Added
- **TLS/mTLS** — `rustls`-based TLS termination for pgwire server, mutual TLS via `--tls-ca-cert`, `--ssl-mode=require` enforcement
- **Encryption key rotation** (`storage/key_manager.rs`) — `KeyManager` with versioned `BTreeMap<u32, EncryptionKey>`, `ROTATE ENCRYPTION KEY` SQL command, column-specific key derivation via HKDF
- **Audit log tamper detection** — SHA-256 hash chaining on audit events, `VERIFY AUDIT LOG` SQL command returning verified count and broken chain position

## [0.36–0.38] — Performance & Scalability

### Added
- **Query parallelism** (`sql/parallel.rs`) — Parallel prefix scan across shard read handles via rayon
- **Batch write optimization** — `ShardCommand::WriteBatchAtomic` with single epoch increment, memtable `insert_batch()` taking write lock once
- **External merge sort** (`sql/external_sort.rs`) — Spill-to-disk for large ORDER BY via sorted SSTable runs and K-way merge
- **Expression compilation** (`sql/compiled_expr.rs`) — Pattern-matching common predicates to closures (equality, range, IN-list via HashSet)
- **Zstd compression** — `CompressionCodec::Zstd` variant, per-level compression policies (LZ4 for L0-L2, Zstd for L3+), `compression-zstd` feature flag

## [0.34–0.35] — Advanced SQL

### Added
- **Recursive CTEs** — `WITH RECURSIVE` with anchor + recursive member, iterative execution until fixpoint
- **Foreign key constraints** — `REFERENCES` in column constraints, `FOREIGN KEY` table constraints with ON DELETE CASCADE/RESTRICT/SET NULL
- **Materialized views** — `CREATE MATERIALIZED VIEW`, `REFRESH MATERIALIZED VIEW`, `DROP MATERIALIZED VIEW`
- **Generated columns** — `GENERATED ALWAYS AS (expr)` with automatic value computation on INSERT/UPDATE
- **Triggers** — `CREATE TRIGGER` with BEFORE/AFTER timing and INSERT/UPDATE/DELETE events
- **User-defined functions** — `CREATE FUNCTION` with SQL body, `DROP FUNCTION`
- **Native date/time types** — `TIMESTAMP`, `DATE`, `INTERVAL` SQL types with temporal arithmetic (`NOW()`, `DATE_TRUNC()`, `DATE_PART()`, `DATE_ADD()`, `DATE_SUB()`)
- **JSON/JSONB operations** — `->` (JSON value), `->>` (text extraction), `@>` (containment), `JSON_EXTRACT()`, `JSON_TYPE()`, `JSON_ARRAY_LENGTH()`

## [0.33] — SQL Completeness

### Added
- **Multi-value INSERT** — `INSERT INTO t VALUES (1,'a'), (2,'b'), (3,'c')` with row count validation
- **Subqueries in WHERE** — `IN (SELECT ...)`, `EXISTS (SELECT ...)`, scalar subqueries via `materialize_subqueries()`
- **OFFSET clause** — `LIMIT n OFFSET m` for pagination, propagated through all SELECT paths
- **IF EXISTS / IF NOT EXISTS** — `CREATE TABLE IF NOT EXISTS`, `DROP TABLE IF EXISTS`, `CREATE INDEX IF NOT EXISTS`, `DROP INDEX IF EXISTS`
- **SELECT without FROM** — `SELECT 1`, `SELECT CURRENT_TIMESTAMP`, expressions evaluated against dummy context
- **FULL OUTER JOIN** — `JoinType::FullOuter` combining left outer + right outer with unmatched row tracking
- **Upsert** — `INSERT ... ON CONFLICT (cols) DO UPDATE SET ...` and `DO NOTHING`
- **RETURNING on UPDATE/DELETE** — `UpdateReturning` and `DeleteReturning` statement variants
- **Persistent transaction sessions** — `SqlSessionHandle` for cross-call BEGIN/COMMIT/ROLLBACK, pgwire per-connection session state

## [0.32] — Error Quality, Security & Audit, Operational Maturity

### Added

#### Phase 2: Error Quality & Developer Experience
- **Structured error codes** — `ErrorCode` enum with stable numeric codes (`T1001` syntax, `T2001` table not found, `T2002` column not found, `T3001` not-null violation, `T4003` query timeout, `T6001` permission denied) and `SqlError` struct with code, message, suggestion, and position fields
- **"Did you mean?" suggestions** — Levenshtein-based fuzzy matching for misspelled table names via `suggest_closest()` with configurable max edit distance; suggestions embedded in `SqlError.suggestion` field
- **`SUGGEST INDEX FOR '<query>'`** — Analyzes a query's WHERE, JOIN, and ORDER BY columns against existing indexes and recommends `CREATE INDEX` statements for unindexed columns
- **`SET` statement** — Session-level variables: `SET STRICT_MODE = ON|OFF`, `SET QUERY_TIMEOUT = <ms>`, `SET QUERY_MAX_MEMORY = <bytes>`, `SET COMPACTION_WINDOW = 'HH:MM-HH:MM'`
- **`VERIFY BACKUP '<path>'`** — Validates backup integrity (metadata, file enumeration, SSTable format, WAL CRC) without restoring; returns status (VALID/ISSUES_FOUND), file count, total bytes, and issue list
- **`VACUUM [table]`** — Scans for tombstones (empty-value records), triggers compaction to physically remove them, reports tombstone count
- `levenshtein()` and `suggest_closest()` public utility functions in `error.rs`
- `sql_parse_err()` and `sql_exec_err()` migration helpers for ergonomic error construction

#### Phase 3: Security & Audit
- **Audit log** (`auth/audit.rs`) — Append-only audit log under `__audit_log/` prefix tracking DDL changes (table/index/view create/drop), auth events, policy changes, and GDPR erasures; queryable via `SHOW AUDIT LOG [LIMIT n]`
- **Row-level security** (`auth/rls.rs`) — `CREATE POLICY <name> ON <table> FOR <op> [TO <role>, ...] USING (<expr>)` and `DROP POLICY <name> ON <table>` for per-row access control with `PolicyOperation` (Select/Insert/Update/Delete/All) and role-based filtering
- **GDPR erasure** — `FORGET KEY '<key>' FROM <table>` tombstones all temporal versions of a record and triggers compaction; records erasure in the audit log

#### Phase 4: Operational Maturity
- **Per-query resource limits** — `SET QUERY_TIMEOUT = <ms>` and `SET QUERY_MAX_MEMORY = <bytes>` with `SqlError::query_timeout()` and `SqlError::memory_limit()` error constructors (codes `T4003`/`T4004`)
- **Online DDL** — `ALTER TABLE <t> DROP COLUMN <c>` (metadata-only, excludes column from projections) and `ALTER TABLE <t> RENAME COLUMN <old> TO <new>` (updates metadata with `column_aliases` map for backward-compatible reads)
- **Plan stability** — `CREATE PLAN GUIDE '<name>' FOR '<sql>' USING '<hints>'`, `DROP PLAN GUIDE '<name>'`, `SHOW PLAN GUIDES` with storage under `__meta/plan_guide/` prefix
- **Backup dry-run** — `RESTORE DATABASE FROM '<path>' DRY_RUN` validates a backup without writing data, returning status, file count, total bytes, and issues
- **Compaction scheduling** — `SET COMPACTION_WINDOW = 'HH:MM-HH:MM'` to restrict compaction to off-peak hours; `compaction_window_start_hour`/`compaction_window_end_hour` config fields
- **WAL management** — `SHOW WAL STATUS` returns per-shard WAL file size, shard ID, and last flush info; `wal_archive_enabled`, `wal_archive_dir`, `wal_retention_count`, `wal_max_bytes` config fields

#### New SQL Commands
- `SET <variable> = <value>` — session-level configuration
- `SUGGEST INDEX FOR '<query>'` — index recommendation
- `VERIFY BACKUP '<path>'` — backup validation
- `VACUUM [table]` — tombstone cleanup
- `FORGET KEY '<key>' FROM <table>` — GDPR erasure
- `CREATE POLICY ... ON ... FOR ... USING (...)` — row-level security
- `DROP POLICY <name> ON <table>` — remove RLS policy
- `CREATE PLAN GUIDE ... FOR ... USING ...` — plan stability
- `DROP PLAN GUIDE '<name>'` — remove plan guide
- `SHOW WAL STATUS` — WAL monitoring
- `SHOW AUDIT LOG [LIMIT n]` — audit trail
- `SHOW PLAN GUIDES` — list plan guides
- `ALTER TABLE <t> DROP COLUMN <c>` — online DDL
- `ALTER TABLE <t> RENAME COLUMN <old> TO <new>` — online DDL
- `RESTORE ... DRY_RUN` — backup dry-run

#### New Modules
- `crates/tensordb-core/src/auth/audit.rs` — AuditLog, AuditEvent, AuditEventKind
- `crates/tensordb-core/src/auth/rls.rs` — PolicyManager, RowPolicy, PolicyOperation
- `crates/tensordb-core/src/sql/plan_guide.rs` — PlanGuideManager, PlanGuide

#### New Test Suites (15 files, 47 tests)
- `tests/error_codes.rs` — structured error code verification
- `tests/suggestions.rs` — Levenshtein fuzzy matching
- `tests/strict_mode.rs` — strict mode and SET statement
- `tests/vacuum.rs` — VACUUM tombstone cleanup
- `tests/verify_backup.rs` — backup verification
- `tests/suggest_index.rs` — index recommendation
- `tests/audit_log.rs` — audit log recording and querying
- `tests/row_level_security.rs` — RLS policy CRUD
- `tests/gdpr_erasure.rs` — FORGET KEY erasure
- `tests/resource_limits.rs` — query timeout and memory limits
- `tests/online_ddl.rs` — DROP/RENAME COLUMN
- `tests/plan_stability.rs` — plan guide CRUD
- `tests/backup_dry_run.rs` — RESTORE DRY_RUN
- `tests/compaction_scheduling.rs` — compaction window
- `tests/wal_management.rs` — SHOW WAL STATUS

### Changed
- `TensorError::SqlParse(String)` → `TensorError::SqlParse(SqlError)` with structured error codes
- `TensorError::SqlExec(String)` → `TensorError::SqlExec(SqlError)` with structured error codes
- 240+ error construction sites migrated to use `sql_parse_err()`/`sql_exec_err()` helpers
- `Database` struct extended with `audit_log` and `compaction_window` fields
- `SqlSession` extended with `session_user`, `session_roles`, `strict_mode`, `query_timeout_ms`, `query_max_memory_bytes`, `query_start`, `memory_used`
- `ShardCommand` extended with `ForceCompaction` variant for VACUUM
- `TableSchemaMetadata` extended with `column_aliases` field for RENAME COLUMN
- `Statement::Restore` extended with `dry_run: bool` field
- `Config` extended with `strict_mode`, `compaction_window_start_hour`, `compaction_window_end_hour`, `wal_archive_enabled`, `wal_archive_dir`, `wal_retention_count`, `wal_max_bytes`
- `parse_show()` extended with WAL STATUS, AUDIT LOG, PLAN GUIDES
- `parse_create()` extended with POLICY, PLAN GUIDE
- `parse_drop()` extended with POLICY, PLAN GUIDE
- `parse_alter()` extended with DROP COLUMN, RENAME COLUMN

## [0.31] — Observability & Diagnostics

### Added
- **`SHOW STATS`** — SQL command exposing uptime, shard count, puts/gets/flushes/compactions, cache hit rate, query latency histogram (p50/p99/avg), and all MetricsRegistry counters/gauges/histograms
- **`SHOW SLOW QUERIES`** — SQL-accessible slow query log with query text, duration_us, timestamp, and rows returned (configurable threshold via `slow_query_threshold_us`)
- **`SHOW ACTIVE QUERIES`** — List currently running queries with query_id, elapsed_ms, and started_at_ms (includes itself, like PostgreSQL's `pg_stat_activity`)
- **`SHOW STORAGE`** — Per-shard storage breakdown: memtable bytes, immutable memtable count/bytes, SSTable bytes, WAL bytes, L0 files, level sizes, block cache stats, plus a TOTAL summary row
- **`SHOW COMPACTION STATUS`** — Per-shard L0 file count, total SSTable files, level sizes, needs_compaction flag, plus global flush/compaction counts
- **Block cache hit/miss tracking** — `BlockCache` now tracks hits and misses with `AtomicU64` counters; `hit_count()`, `miss_count()`, `hit_rate()` accessors
- **Health HTTP endpoint** — `/health` JSON endpoint on pgwire port+1 returning uptime, shard count, puts/gets, cache hit rate, SSTable/memtable bytes, and readiness status (uses hyper 1.x)
- **Active query tracking** — `Database::sql()` registers/deregisters queries with timing, enabling `SHOW ACTIVE QUERIES`
- **Query metrics** — Every SQL query records execution time in `MetricsRegistry` histogram and slow query log
- `slow_query_threshold_us` config field (default: 10,000 µs = 10ms)
- `ShardStorageInfo` struct and `ShardReadHandle::storage_info()` for per-shard storage introspection
- `Database::metrics()`, `uptime_ms()`, `block_cache()`, `active_queries_snapshot()`, `storage_info()`, `wal_sizes()` accessors
- 14 new integration tests in `tests/observability.rs` covering all 5 SHOW commands, parser variants, cache hit rate, and query latency histogram

### Changed
- `Database` struct now holds `MetricsRegistry`, `startup_time`, `block_cache` reference, `active_queries` map, and `next_query_id` counter
- `tensordb-server` Cargo.toml now depends on `hyper`, `hyper-util`, and `http-body-util` for the health endpoint

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
- **Dockerfile** — Multi-stage build producing minimal tensordb-server image
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
