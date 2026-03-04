# TensorDB Test Plan

## Goals

- Prove correctness under normal and failure conditions.
- Validate temporal semantics (`AS OF`, `VALID AT`, SQL:2011 clauses) at scale.
- Quantify latency and throughput characteristics under realistic workloads.
- Demonstrate operational reliability under long-running mixed workloads.
- Establish reproducible baseline comparisons against SQLite, sled, and redb.

## Current Test Coverage

**270+ integration tests** across **45 test suites** in `tests/`:

| Suite | Coverage |
|-------|----------|
| `ai_core.rs` | AI runtime: insight synthesis, risk scoring, access stats, isolation from user change feeds |
| `ai_sql_functions.rs` | SQL-level AI functions: `EXPLAIN AI`, AI dashboard queries |
| `auth_rbac.rs` | Authentication, role-based access control, permissions, sessions |
| `bitemporal.rs` | Bitemporal queries: `AS OF`, `VALID AT`, combined filters |
| `change_feeds.rs` | Change feed subscriptions, prefix filtering, durable cursors, consumer groups |
| `compaction_history.rs` | Multi-level compaction preserving temporal history |
| `data_interchange.rs` | CSV/JSON/NDJSON import and export via `COPY TO/FROM` |
| `event_sourcing.rs` | Event stores, aggregate projections, snapshots, idempotency |
| `fts_sql.rs` | Full-text search: `CREATE FULLTEXT INDEX`, `MATCH()`, `HIGHLIGHT()`, BM25 |
| `multilevel_compaction.rs` | L0→L1→L2+ compaction correctness |
| `parquet_interchange.rs` | Parquet read/write, `read_parquet()` table function |
| `prefix_scan.rs` | Prefix scan correctness across shards |
| `query_engine_perf.rs` | Cost-based planner, `EXPLAIN ANALYZE`, table statistics |
| `readme_examples.rs` | Validates all SQL examples from README |
| `reopen_cycles.rs` | Database close/reopen with state preservation |
| `schema_evolution.rs` | Migration manager, schema versioning, schema diff |
| `sql_completeness.rs` | SQL surface coverage: CASE, CAST, UNION, set operations |
| `sql_correctness.rs` | SQL semantic correctness: NULL handling, ORDER BY, transactions |
| `sql_expressions.rs` | Expression evaluation: arithmetic, comparisons, functions |
| `sql_facet.rs` | Relational facet: typed tables, DDL, DML, views, indexes |
| `sstable_roundtrip.rs` | SSTable build/read/verify with LZ4 compression |
| `temporal_sql.rs` | SQL:2011 temporal clauses: SYSTEM_TIME, APPLICATION_TIME |
| `temporal_stress.rs` | High-cardinality version chains under flush/compaction |
| `time_travel.rs` | Time-travel reads across flush and compaction boundaries |
| `timeseries_sql.rs` | Time-series: TIME_BUCKET, gap filling, LOCF, INTERPOLATE, DELTA, RATE |
| `typed_schema.rs` | Typed column tables, columnar encoding, schema enforcement |
| `wal_faults.rs` | WAL fault injection: CRC mismatch, torn tail, recovery |
| `wal_recovery.rs` | WAL replay and state recovery after crash |
| `window_functions.rs` | Window functions: ROW_NUMBER, RANK, DENSE_RANK, LEAD, LAG |
| `write_batch.rs` | Atomic multi-key write batches |
| `error_codes.rs` | Structured error codes (T1xxx–T6xxx), code categories, display format |
| `suggestions.rs` | Levenshtein distance, fuzzy matching, "Did you mean?" suggestions |
| `strict_mode.rs` | SET statement, strict mode, query timeout, query max memory |
| `vacuum.rs` | VACUUM tombstone cleanup, VACUUM ALL |
| `verify_backup.rs` | VERIFY BACKUP validation, nonexistent path handling |
| `suggest_index.rs` | SUGGEST INDEX FOR query analysis, indexed vs unindexed columns |
| `audit_log.rs` | Audit log recording, SHOW AUDIT LOG, DDL event tracking |
| `row_level_security.rs` | CREATE/DROP POLICY, duplicate detection, role-based policies |
| `gdpr_erasure.rs` | FORGET KEY erasure, audit log integration |
| `resource_limits.rs` | SET QUERY_TIMEOUT, SET QUERY_MAX_MEMORY, timeout reset |
| `online_ddl.rs` | ALTER TABLE DROP/RENAME COLUMN, PK protection |
| `plan_stability.rs` | CREATE/DROP PLAN GUIDE, SHOW PLAN GUIDES, duplicate detection |
| `backup_dry_run.rs` | RESTORE DRY_RUN validation, invalid path handling |
| `compaction_scheduling.rs` | SET COMPACTION_WINDOW, format validation |
| `wal_management.rs` | SHOW WAL STATUS, per-shard WAL info |

## Acceptance Gates

A build is acceptable when all of the following pass:

1. `cargo fmt --all --check` — no formatting issues
2. `cargo clippy --workspace --all-targets -- -D warnings` — no lint warnings
3. `cargo test --workspace --all-targets` — all 270+ tests pass
4. `cargo test --workspace --all-targets --features native` — passes with C++ acceleration
5. `./scripts/ai_overhead_gate.sh` — AI overhead within regression thresholds
6. WAL fault-injection tests pass (CRC mismatch + torn tail recovery)
7. Temporal stress tests pass with forced flush/compaction/reopen cycles
8. README example tests pass (`tests/readme_examples.rs`)

## Test Layers

### A) Unit and Component Correctness

- Varint encode/decode roundtrip
- Internal key encode/decode consistency
- WAL framing and replay invariants
- Bloom filter encode/decode and `may_contain` behavior
- SSTable build/read structural validation (V1 and V2 formats)
- LZ4 block compression roundtrip
- Columnar row encoding/decoding with null bitmaps

### B) Fault Injection and Recovery

- WAL torn-tail replay: stops deterministically at last valid record
- WAL CRC mismatch: stops replay at last valid frame
- Reopen path: restores visible state from manifest + WAL replay
- Repeated reopen cycles: preserves deterministic results
- Manifest atomic replacement: no partial state visible after crash

### C) Temporal Semantics

- Multi-version `AS OF` read correctness
- `VALID AT` interval filtering at boundaries (`valid_from <= V < valid_to`)
- Combined `AS OF` and `VALID AT` behavior
- SQL:2011 `FOR SYSTEM_TIME` (AS OF, FROM...TO, BETWEEN...AND, ALL)
- SQL:2011 `FOR APPLICATION_TIME` (AS OF, FROM...TO, BETWEEN...AND)
- High-cardinality version chains under forced flush/compaction
- Compaction-preserved history across reopen cycles

### D) SQL Engine

- **DDL:** CREATE/DROP TABLE, VIEW, INDEX, FULLTEXT INDEX, TIMESERIES TABLE, ALTER TABLE ADD/DROP/RENAME COLUMN
- **DML:** INSERT, INSERT...RETURNING, UPDATE, DELETE, VACUUM, FORGET KEY
- **Queries:** SELECT with WHERE, ORDER BY, LIMIT, JOIN (inner/left/right/cross), GROUP BY, HAVING, subqueries, CTEs, UNION/INTERSECT/EXCEPT, window functions
- **Expressions:** Arithmetic, comparisons, CASE, CAST, LIKE/ILIKE, IS NULL, BETWEEN, IN
- **Functions:** All 50+ scalar, aggregate, window, time-series, and FTS functions
- **Temporal:** AS OF, VALID AT, all SQL:2011 temporal clause variants
- **Transactions:** BEGIN/COMMIT/ROLLBACK, transaction-local reads, rollback correctness
- **Data interchange:** COPY TO/FROM CSV/JSON/Parquet, table functions (read_csv, read_json, read_parquet)
- **Introspection:** SHOW TABLES, DESCRIBE, EXPLAIN, EXPLAIN ANALYZE, EXPLAIN AI, ANALYZE, SHOW WAL STATUS, SHOW AUDIT LOG, SHOW PLAN GUIDES
- **Administration:** SET STRICT_MODE/QUERY_TIMEOUT/QUERY_MAX_MEMORY/COMPACTION_WINDOW, SUGGEST INDEX, VERIFY BACKUP, VACUUM, RESTORE DRY_RUN
- **Prepared statements:** $1/$2 parameter binding, execution with parameters

### E) Specialized Engines

- **Full-text search:** Index creation, posting list maintenance, MATCH() with BM25, HIGHLIGHT(), multi-column
- **Time-series:** Bucketed storage, TIME_BUCKET, gap filling, LOCF, INTERPOLATE, DELTA, RATE
- **Vector search:** HNSW insert/search/delete, distance metrics (cosine, euclidean, dot product), exact vs approximate
- **Event sourcing:** Event stores, append with sequence numbering, aggregate projection, snapshots, idempotency
- **Schema evolution:** Migration register/apply/rollback, schema versioning, schema diff

### F) AI Runtime

- Auto-insight generation is asynchronous and in-process
- Insight writes are immutable internal facts under `__ai/insight/...`
- User change-feed subscribers never receive AI internal writes
- Inline risk scoring produces valid scores on writes
- AI advisor recommendations are generated
- `EXPLAIN AI` returns insights for a key
- AI overhead stays within regression gate thresholds

### G) Authentication & Security

- User creation, authentication, password change, disable
- Role creation, permission grants, role-based access
- Table-level privilege checking
- Session creation with TTL, token-based access, revocation
- Audit log: DDL events recorded, SHOW AUDIT LOG returns events
- Row-level security: CREATE/DROP POLICY, role-based policy matching
- GDPR erasure: FORGET KEY tombstones all versions, audit log integration
- Structured error codes: T1xxx syntax, T2xxx schema, T3xxx constraint, T4xxx execution, T6xxx auth

### H) Change Data Capture

- Prefix-filtered subscriptions receive matching writes
- Durable cursors persist position and resume after restart
- Consumer groups with shard assignment and rebalancing
- AI writes filtered from user-facing feeds

### I) Native Integration

- Rust/native deterministic output equivalence for fixture inputs
- Native call path instrumentation confirms invocation
- Feature-gating guarantees pure-Rust fallback works without C++

## Performance Validation Matrix

Sweep across key parameters:

| Parameter | Values |
|-----------|--------|
| `shard_count` | 1, 2, 4, 8 |
| `memtable_max_bytes` | 64 KiB, 1 MiB, 4 MiB |
| `sstable_block_bytes` | 4 KiB, 16 KiB, 64 KiB |
| `bloom_bits_per_key` | 8, 10, 14 |
| `fast_write_enabled` | true, false |
| `fast_write_wal_batch_interval_us` | 100, 1000, 5000 |
| `ai_auto_insights` | false, true |

Collect per run:
- Write throughput (ops/s) and latency distribution
- Read p50/p95/p99 latency
- Bloom false-positive rate
- mmap block read count
- Block cache hit ratio
- Compaction I/O amplification

## Baseline Comparisons

The benchmark suites compare TensorDB against:

- **SQLite** — Point read, point write, prefix scan, mixed workload
- **sled** — Point read, point write, batch write, prefix scan, mixed workload
- **redb** — Point read, point write, batch write, prefix scan, mixed workload

Comparison methodology:
- Criterion 0.5 with statistical analysis
- Warm-up phase before measurement
- Median of multiple iterations
- Same workload parameters across all engines

## Overnight Reliability Campaign

Use `scripts/overnight_iterate.sh` with:
- `ROUNDS >= 12`
- Mixed write/read workloads
- Periodic reopen/checkpoint behavior
- AI features enabled for subset of rounds

Required outcomes:
- No crashes or panics
- No invariant violations
- No data loss across reopen cycles
- Stable p99 without unbounded degradation
- AI overhead within gate thresholds

## Regression Policy

Any PR touching storage, WAL, write path, temporal semantics, SQL engine, or AI runtime must include:
- Updated or new tests covering the change
- Benchmark delta evidence (before/after)
- Note on whether temporal semantics were impacted
- `cargo fmt` and `cargo clippy` clean
