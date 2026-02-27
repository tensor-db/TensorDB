<p align="center">
  <img src="assets/spectradb-logo.svg" alt="SpectraDB" width="560">
</p>

<p align="center">
  <strong>An AI-native, bitemporal ledger database with MVCC, full SQL, vector search, and sub-microsecond performance.</strong>
</p>

<p align="center">
  <a href="https://github.com/spectra-db/SpectraDB/actions"><img src="https://github.com/spectra-db/SpectraDB/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-stable-orange.svg" alt="Rust"></a>
</p>

---

SpectraDB is an embedded database written in Rust that treats every write as an immutable fact. It separates **system time** (when data was recorded) from **business-valid time** (when data was true), giving you built-in time travel and auditability with zero application-level bookkeeping.

**65,000+ lines of Rust** | **224+ integration tests** | **30 test suites** | **8 workspace crates**

## Performance

| Operation | SpectraDB | SQLite | sled | redb |
|-----------|-----------|--------|------|------|
| Point Read | **276 ns** | 1,080 ns | 244 ns | 573 ns |
| Point Write | **1.9 µs** | 38.6 µs | — | — |
| Prefix Scan (1k keys) | **native** | — | — | — |
| Mixed 80r/20w | **native** | — | — | — |

- **4x faster reads** than SQLite
- **20x faster writes** than SQLite via lock-free fast write path
- **276 ns** point reads via direct shard bypass (no channel round-trip)
- **1.9 µs** point writes via `FastWritePath` with group-commit WAL

Benchmarks use Criterion 0.5. Run them yourself:

```bash
cargo bench --bench comparative    # SpectraDB vs SQLite
cargo bench --bench multi_engine   # SpectraDB vs SQLite vs sled vs redb
cargo bench --bench basic          # Microbenchmarks
```

## Key Features

### Core Database
- **Immutable Fact Ledger** — Append-only WAL with CRC-framed records. Data is never overwritten.
- **MVCC Snapshot Reads** — Query any past state with `AS OF <commit_ts>`.
- **Bitemporal Filtering** — SQL:2011 `SYSTEM_TIME` and `APPLICATION_TIME` temporal clauses.
- **LSM Storage Engine** — Memtable → SSTable (L0–L6) with bloom filters, prefix compression, mmap reads, LZ4 block compression.
- **Block & Index Caching** — LRU caches with configurable memory budgets.
- **Write Batch API** — Atomic multi-key writes with a single WAL frame.

### SQL Engine
- **Full SQL** — DDL, DML, SELECT, JOINs (inner/left/right/cross), GROUP BY, HAVING, CTEs, subqueries, UNION/INTERSECT/EXCEPT, window functions, CASE, CAST, LIKE/ILIKE, transactions.
- **50+ built-in functions** — String, numeric, date/time, aggregate, window, conditional, type conversion.
- **Cost-based query planner** — `PlanNode` tree with cost estimation, `EXPLAIN` and `EXPLAIN ANALYZE`.
- **Prepared statements** — Parse once, execute many with `$1, $2, ...` parameter binding.
- **Temporal SQL** — 7 SQL:2011 temporal clause variants for both system time and application time.
- **Vectorized execution** — Columnar `RecordBatch` engine with vectorized filter, project, aggregate, join, and sort.

### Specialized Engines
- **Full-Text Search** — `CREATE FULLTEXT INDEX`, `MATCH()`, `HIGHLIGHT()`, BM25 ranking, multi-column with per-column boosting.
- **Time-Series** — `CREATE TIMESERIES TABLE`, `TIME_BUCKET()`, gap filling (`LOCF`, `INTERPOLATE`), `DELTA()`, `RATE()`.
- **Vector Search** — HNSW approximate nearest neighbor, cosine/Euclidean/dot-product distance, `VectorIndex` API.
- **Event Sourcing** — Aggregate projections, snapshot support, idempotency keys, cross-aggregate event queries.
- **Schema Evolution** — Migration manager with versioned SQL migrations, schema diff, rollback support.

### Data Platform
- **Change Data Capture** — Prefix-filtered subscriptions, durable cursors, consumer groups with rebalancing.
- **Data Interchange** — `COPY TO/FROM` CSV, JSON, Parquet. Table functions: `read_csv()`, `read_json()`, `read_parquet()`.
- **PostgreSQL Wire Protocol** — `spectradb-server` crate accepts Postgres client connections via pgwire.
- **Authentication & RBAC** — User management, role-based access control, table-level permissions, session management.
- **Connection Pooling** — Configurable pool with warmup, idle eviction, and RAII connection guards.

### AI Runtime
- **Background Insight Synthesis** — In-process AI pipeline consuming change feeds.
- **Inline Risk Scoring** — Per-write risk assessment without external model servers.
- **AI Advisors** — Compaction scheduling, cache tuning, query optimization recommendations.
- **ML Pipeline** — Feature store, model registry, point-in-time joins, inference metrics.
- **`EXPLAIN AI`** — SQL command for AI insights, provenance, and risk scores per key.

### Language Bindings & Integrations
- **Rust** — Native embedded library (`spectradb-core`).
- **Python** — PyO3 bindings (`spectradb-python`) — `open()`, `put()`, `get()`, `sql()`.
- **Node.js** — napi-rs bindings (`spectradb-node`) — `open()`, `put()`, `get()`, `sql()`.
- **Interactive CLI** — TAB completion, persistent history, table/line/JSON output modes.
- **Optional C++ Acceleration** — `--features native` via `cxx` for Hasher, Compressor, BloomProbe.
- **Optional io_uring** — `--features io-uring` for Linux async I/O.
- **Optional SIMD** — `--features simd` for hardware-accelerated bloom probes and checksums.

## Who Is This For?

<table>
<tr>
<td width="50%" valign="top">

### Audit & Compliance Systems
Regulations demand provable data history. SpectraDB's append-only ledger with bitemporal queries lets you reconstruct the exact state of any record at any point in time — no application-level versioning required.

</td>
<td width="50%" valign="top">

### Event-Sourced Applications
Use SpectraDB as the append-only event store behind CQRS or event-driven architectures. Write batches provide atomic multi-event commits. Change feeds push events to downstream consumers in real time.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Temporal Data Management
Model entities with business-valid time ranges — contracts, policies, price schedules — and query them with full SQL. Ask "what was true on date X?" without building custom versioning logic.

</td>
<td width="50%" valign="top">

### Embedded & Local-First Apps
Ship a full temporal database as a Rust library, Python package, or Node.js module — no server process, no network, no Docker. Ideal for IoT gateways, on-device analytics, and desktop apps.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Real-Time Monitoring & IoT
The time-series engine handles high-frequency metric ingestion with bucketed storage and downsampling. Combine with change feeds to trigger alerts as data arrives.

</td>
<td width="50%" valign="top">

### AI & Search Applications
Vector search with HNSW indexing, full-text search with BM25 ranking, and an in-process AI runtime for insight synthesis — all in a single embedded engine.

</td>
</tr>
</table>

## Quickstart

```bash
# Run examples
cargo run --example quickstart     # Core features: SQL, time-travel, prepared statements
cargo run --example bitemporal     # Bitemporal ledger: AS OF + VALID AT queries
cargo run --example ai_native      # AI runtime: insights, risk scoring, query planning

# Build and launch interactive shell
cargo build -p spectradb-cli
cargo run -p spectradb-cli -- --path ./mydb

# Launch with AI auto-insights enabled
cargo run -p spectradb-cli -- --path ./mydb --ai-auto-insights

# Start PostgreSQL wire protocol server
cargo run -p spectradb-server -- --path ./mydb --port 5432
```

```sql
-- Create a typed table
CREATE TABLE accounts (id INTEGER PRIMARY KEY, name TEXT NOT NULL, balance REAL);

INSERT INTO accounts (id, name, balance) VALUES (1, 'alice', 1000.0);
INSERT INTO accounts (id, name, balance) VALUES (2, 'bob', 500.0);

-- JSON document table
CREATE TABLE events (pk TEXT PRIMARY KEY);
INSERT INTO events (pk, doc) VALUES ('evt-1', '{"type":"signup","user":"alice"}');
INSERT INTO events (pk, doc) VALUES ('evt-2', '{"type":"purchase","user":"bob","amount":49.99}');

-- Standard SQL
SELECT * FROM accounts WHERE balance > 500 ORDER BY name;
SELECT count(*), sum(balance) FROM accounts;
SELECT pk, doc FROM events ORDER BY pk LIMIT 10;

-- Joins
SELECT a.name, e.doc FROM accounts a
JOIN events e ON a.name = e.doc->>'user'
ORDER BY a.name;

-- Window functions
SELECT name, balance,
  ROW_NUMBER() OVER (ORDER BY balance DESC) AS rank,
  LAG(balance) OVER (ORDER BY balance) AS prev_balance
FROM accounts;

-- CTEs
WITH high_value AS (
  SELECT * FROM accounts WHERE balance > 500
)
SELECT * FROM high_value;

-- Time travel: read state as of commit 1
SELECT doc FROM events WHERE pk='evt-1' AS OF 1;

-- Bitemporal: SQL:2011 temporal queries
SELECT * FROM events FOR SYSTEM_TIME AS OF 1;
SELECT * FROM events FOR SYSTEM_TIME FROM 1 TO 10;
SELECT * FROM events FOR APPLICATION_TIME AS OF 1000;

-- Full-text search
CREATE FULLTEXT INDEX idx_events ON events (doc);
SELECT pk, doc FROM events WHERE MATCH(doc, 'signup');
SELECT HIGHLIGHT(doc, 'signup') FROM events WHERE MATCH(doc, 'signup');

-- Time-series
CREATE TIMESERIES TABLE metrics (ts TIMESTAMP, value REAL) WITH (bucket_size = '1h');
SELECT TIME_BUCKET('1h', ts), AVG(value) FROM metrics GROUP BY 1;

-- Prepared statements
-- $1, $2 are bound at execution time
SELECT * FROM accounts WHERE balance > $1;

-- EXPLAIN / EXPLAIN ANALYZE
EXPLAIN SELECT * FROM accounts WHERE id = 1;
EXPLAIN ANALYZE SELECT * FROM accounts ORDER BY balance DESC LIMIT 5;

-- Data interchange
COPY accounts TO '/tmp/accounts.csv' FORMAT CSV;
COPY accounts FROM '/tmp/accounts.csv' FORMAT CSV;
SELECT * FROM read_parquet('data.parquet');

-- Transactions
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- Schema management
ALTER TABLE accounts ADD COLUMN email TEXT;
SHOW TABLES;
DESCRIBE accounts;
ANALYZE accounts;
```

## SQL Function Reference

<details>
<summary><strong>String Functions (17)</strong></summary>

`UPPER`, `LOWER`, `LENGTH`, `SUBSTR`/`SUBSTRING`, `TRIM`, `LTRIM`, `RTRIM`, `REPLACE`, `CONCAT`, `CONCAT_WS`, `LEFT`, `RIGHT`, `LPAD`, `RPAD`, `REVERSE`, `SPLIT_PART`, `REPEAT`, `POSITION`/`STRPOS`, `INITCAP`
</details>

<details>
<summary><strong>Numeric Functions (13)</strong></summary>

`ABS`, `ROUND`, `CEIL`/`CEILING`, `FLOOR`, `MOD`, `POWER`/`POW`, `SQRT`, `LOG`/`LOG10`, `LN`, `EXP`, `SIGN`, `RANDOM`, `PI`
</details>

<details>
<summary><strong>Date/Time Functions (6)</strong></summary>

`NOW`/`CURRENT_TIMESTAMP`, `EPOCH`, `EXTRACT`/`DATE_PART`, `DATE_TRUNC`, `TO_CHAR`
</details>

<details>
<summary><strong>Aggregate Functions (11)</strong></summary>

`COUNT(*)`/`COUNT(col)`/`COUNT(DISTINCT col)`, `SUM`, `AVG`, `MIN`, `MAX`, `STRING_AGG`/`GROUP_CONCAT`, `STDDEV_POP`, `STDDEV_SAMP`, `VAR_POP`, `VAR_SAMP`
</details>

<details>
<summary><strong>Window Functions (5)</strong></summary>

`ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `LEAD()`, `LAG()`
</details>

<details>
<summary><strong>Time-Series Functions (6)</strong></summary>

`TIME_BUCKET`, `TIME_BUCKET_GAPFILL`, `LOCF`, `INTERPOLATE`, `DELTA`, `RATE`
</details>

<details>
<summary><strong>Full-Text Search Functions (2)</strong></summary>

`MATCH(column, query)`, `HIGHLIGHT(column, query)`
</details>

<details>
<summary><strong>Conditional & Utility (6)</strong></summary>

`COALESCE`, `NULLIF`, `GREATEST`, `LEAST`, `IF`/`IIF`, `TYPEOF`, `CAST`
</details>

## Architecture

SpectraDB is organized around three core principles: **immutable truth** (the append-only ledger), **temporal indexing** (bitemporal metadata on every fact), and **faceted queries** (pluggable query planes over the same data).

```mermaid
graph TB
    subgraph Client Layer
        CLI[Interactive Shell]
        API[Rust API<br/><code>db.sql&#40;...&#41;</code>]
        PY[Python Bindings<br/>PyO3]
        NODE[Node.js Bindings<br/>napi-rs]
        PG[pgwire Server<br/>PostgreSQL Protocol]
    end

    subgraph Query Engine
        Parser[SQL Parser<br/>50+ functions · CTEs · windows]
        Planner[Cost-Based Planner<br/>PlanNode tree · EXPLAIN ANALYZE]
        Executor[Query Executor<br/>scans · joins · aggregates · windows]
        VecEngine[Vectorized Engine<br/>columnar batches · RecordBatch]
    end

    subgraph Facet Layer
        RF[Relational Facet<br/>typed tables · views · indexes]
        FTS[Full-Text Search<br/>inverted index · BM25 · stemmer]
        TS[Time-Series<br/>time_bucket · gap fill · rate]
        VS[Vector Search<br/>HNSW · cosine · euclidean]
        ES[Event Sourcing<br/>aggregates · snapshots]
    end

    subgraph Shard Engine
        direction LR
        FW[Fast Write Path<br/>lock-free · 1.9µs]
        S0[Shard 0]
        S1[Shard 1]
        SN[Shard N]
    end

    CF[Change Feeds<br/>durable cursors · consumer groups]

    subgraph AI Runtime
        INS[Insight Synthesis<br/>background batching]
        RISK[Risk Scoring<br/>inline assessment]
        ADV[Advisors<br/>compaction · cache · query]
        ML[ML Pipeline<br/>feature store · model registry]
    end

    subgraph Storage Engine
        WAL[Write-Ahead Log<br/>CRC-framed · group commit · fdatasync]
        MT[Memtable<br/>sorted in-memory map]
        BC[Block & Index Cache<br/>LRU · configurable budgets]
        SST[SSTables<br/>bloom · LZ4 · mmap · zone maps]
        C[Multi-Level Compaction<br/>L0 → L1 → ... → L6]
    end

    MF[Manifest<br/>per-level file tracking]

    CLI --> Parser
    API --> Parser
    PY --> API
    NODE --> API
    PG --> Parser
    Parser --> Planner
    Planner --> Executor
    Executor --> VecEngine
    Executor --> RF
    RF --> FW
    FTS --> S0
    TS --> S0
    VS --> S0
    ES --> S0
    FW --> S0
    FW --> S1
    FW --> SN
    S0 --> CF
    S1 --> CF
    SN --> CF
    S0 --> INS
    INS --> RISK
    INS --> ADV
    INS --> ML
    S0 --> WAL
    S1 --> WAL
    SN --> WAL
    WAL --> MT
    MT -->|flush| SST
    SST --> BC
    SST --> C
    S0 --> MF
    S1 --> MF
    SN --> MF
```

### Write Path

1. **Route** — Key is hashed to a shard (`hash(key) % shard_count`).
2. **Fast Path** — If `fast_write_enabled`, the lock-free `FastWritePath` writes directly to the shard's memtable via atomic operations (~1.9 µs). Falls back to channel path when memtable is full or subscribers are active.
3. **WAL** — Group-commit `DurabilityThread` batches WAL records across shards, one `fdatasync` per flush cycle.
4. **Notify** — Matching change feed subscribers receive the event (when active).
5. **Buffer** — Entry is inserted into the in-memory memtable.
6. **Flush** — When memtable exceeds `memtable_max_bytes`, it is frozen and written as an LZ4-compressed SSTable.
7. **Compact** — Multi-level compaction promotes SSTables through L0 → L1 → ... → L6 with size-budgeted thresholds. All temporal versions are preserved.

### Read Path

1. **Direct Bypass** — `ShardReadHandle` reads directly from shared state — no channel round-trip (276 ns).
2. **Cache Check** — LRU block and index caches serve hot data without disk I/O.
3. **Bloom Check** — If the bloom filter says the key is absent, skip the SSTable.
4. **Memtable Scan** — Check the active and immutable memtables for the latest version.
5. **Level Lookup** — L0: search all files newest-first. L1+: binary search for the single overlapping file per level.
6. **Temporal Filter** — Apply `AS OF` (system time) and `VALID AT` (business time) predicates.
7. **Merge** — Return the most recent version satisfying all filters.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Append-only writes | Immutability simplifies recovery, enables time travel, eliminates in-place update corruption |
| Lock-free fast write path | Bypasses crossbeam channel for 20x improvement over channel-based writes |
| Single writer per shard | Avoids fine-grained locking while allowing parallel writes across shards |
| Group-commit WAL | One fdatasync per batch interval across all shards reduces I/O overhead |
| Bitemporal timestamps | Separates "when recorded" from "when true" — required for audit and compliance |
| Multi-level compaction | Size-budgeted leveling reduces read amplification while preserving all temporal versions |
| Direct shard reads | ShardReadHandle bypasses the actor channel entirely for sub-microsecond reads |
| Dual schema modes | JSON documents for flexibility; typed columns for structure and performance |

## Configuration

SpectraDB is configured through the `Config` struct. All parameters have sensible defaults.

<details>
<summary><strong>All 22 Configuration Parameters</strong></summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shard_count` | `usize` | `4` | Number of write shards |
| `wal_fsync_every_n_records` | `usize` | `128` | WAL fsync frequency |
| `memtable_max_bytes` | `usize` | `4 MB` | Max memtable size before flush |
| `sstable_block_bytes` | `usize` | `16 KB` | SSTable block size |
| `sstable_max_file_bytes` | `u64` | `64 MB` | Max SSTable file size |
| `bloom_bits_per_key` | `usize` | `10` | Bloom filter bits per key |
| `block_cache_bytes` | `usize` | `32 MB` | Block cache memory budget |
| `index_cache_entries` | `usize` | `1024` | Index cache entry count |
| `compaction_l0_threshold` | `usize` | `8` | L0 SSTable count before compaction |
| `compaction_l1_target_bytes` | `u64` | `10 MB` | L1 target size |
| `compaction_size_ratio` | `u64` | `10` | Level size ratio multiplier |
| `compaction_max_levels` | `usize` | `7` | Maximum compaction levels (L0–L6) |
| `fast_write_enabled` | `bool` | `true` | Enable lock-free fast write path |
| `fast_write_wal_batch_interval_us` | `u64` | `1000` | WAL group commit batch interval (µs) |
| `ai_auto_insights` | `bool` | `false` | Enable background AI insight synthesis |
| `ai_batch_window_ms` | `u64` | `20` | AI batch accumulation window |
| `ai_batch_max_events` | `usize` | `16` | Max events per AI batch |
| `ai_inline_risk_assessment` | `bool` | `false` | Inline risk score on writes |
| `ai_annotate_reads` | `bool` | `false` | Annotate reads with AI metadata |
| `ai_compaction_advisor` | `bool` | `false` | AI-driven compaction scheduling |
| `ai_cache_advisor` | `bool` | `false` | AI-driven cache admission/eviction |
| `ai_access_stats_size` | `usize` | `1024` | Hot-key tracker ring buffer size |

</details>

## Completed Releases

<details>
<summary><strong>v0.1 — Initial Release</strong></summary>

Core database: append-only ledger, WAL, memtable, SSTable, bloom filters, MVCC reads, bitemporal filtering, basic SQL (CREATE TABLE, INSERT, SELECT, AS OF, VALID AT), interactive CLI shell.
</details>

<details>
<summary><strong>v0.2 — Query Engine</strong></summary>

Expression AST with full precedence parsing, `WHERE` clauses, `UPDATE`/`DELETE`, `JOIN` (inner/left/right/cross), aggregates (`SUM`, `AVG`, `MIN`, `MAX`), `GROUP BY`, `HAVING`.
</details>

<details>
<summary><strong>v0.3 — Storage & Performance</strong></summary>

Multi-level compaction (L0–L7), block and index caching (LRU), prefix compression, write-batch API, SIMD-accelerated bloom probes (`--features simd`).
</details>

<details>
<summary><strong>v0.4 — SQL Surface & Developer Experience</strong></summary>

Subqueries, CTEs, window functions (ROW_NUMBER, RANK, DENSE_RANK, LEAD, LAG), typed column schema, columnar row encoding, index-backed execution, `COPY`, Python bindings, Node.js bindings.
</details>

<details>
<summary><strong>v0.5 — Ecosystem</strong></summary>

Full-text search facet, time-series facet, streaming change feeds, io_uring async I/O (`--features io-uring`), comparative benchmark harness.
</details>

<details>
<summary><strong>v0.6–v0.7 — Storage & SQL Correctness</strong></summary>

Manifest persistence with level metadata, block/index cache wired into SSTable reader, immutable memtable queue, binary search fix, typed delete tombstones, WAL rotation, numeric ORDER BY fix, transaction-local reads, window function evaluation order.
</details>

<details>
<summary><strong>v0.9 — Storage Performance</strong></summary>

LZ4 block compression (SSTable V2), adaptive compression, LRU eviction for block cache, predicate pushdown, parallel memtable flush, batched WAL fsync, bloom FP tracking.
</details>

<details>
<summary><strong>v0.10 — Query Engine Performance</strong></summary>

Cost-based query planner (`PlanNode` tree), `EXPLAIN ANALYZE` with execution metrics, prepared statements with `$N` parameters, table statistics via `ANALYZE`, per-column stats with histograms.
</details>

<details>
<summary><strong>v0.11 — Temporal SQL:2011</strong></summary>

`FOR SYSTEM_TIME AS OF`, `FROM...TO`, `BETWEEN...AND`, `ALL`. `FOR APPLICATION_TIME AS OF`, `FROM...TO`, `BETWEEN...AND`. Full SQL:2011 bitemporal query compliance.
</details>

<details>
<summary><strong>v0.13 — Full-Text Search SQL</strong></summary>

`CREATE FULLTEXT INDEX`, `MATCH()` with BM25 ranking, multi-column FTS with per-column boosting, `HIGHLIGHT()` with match markers, automatic posting list maintenance on INSERT/UPDATE/DELETE.
</details>

<details>
<summary><strong>v0.14 — Time-Series SQL</strong></summary>

`CREATE TIMESERIES TABLE`, `TIME_BUCKET()`, `TIME_BUCKET_GAPFILL()`, `LOCF`, `INTERPOLATE`, `DELTA`, `RATE`, `first()`/`last()` time-weighted aggregates.
</details>

<details>
<summary><strong>v0.15 — PostgreSQL Wire Protocol</strong></summary>

`spectradb-server` crate with pgwire v3 protocol, simple and extended query modes, type OID mapping, password authentication. Connect with any PostgreSQL client.
</details>

<details>
<summary><strong>v0.17 — SQL Completeness</strong></summary>

`CASE WHEN`, `CAST`, `UNION`/`INTERSECT`/`EXCEPT`, `INSERT...RETURNING`, `CREATE TABLE AS SELECT`, 17 string functions, 13 math functions, 6 date/time functions, `NULLIF`/`GREATEST`/`LEAST`/`IF`, `STDDEV`/`VARIANCE` aggregates, `LIKE`/`ILIKE`.
</details>

<details>
<summary><strong>v0.18 — Data Interchange</strong></summary>

Parquet read/write via `COPY TO/FROM`, CSV with RFC 4180, NDJSON streaming, `read_parquet()`/`read_csv()`/`read_json()` table functions, Arrow in-memory columnar format.
</details>

<details>
<summary><strong>v0.19 — Vectorized Execution</strong></summary>

Columnar `RecordBatch` representation, vectorized filter/project/aggregate/join/sort operators, selection vector operations, null-aware ordering.
</details>

<details>
<summary><strong>v0.20 — Columnar Storage (Partial)</strong></summary>

Zone maps (per-block min/max + HLL distinct count), dictionary encoding for low-cardinality columns, `APPROX_COUNT_DISTINCT` via HyperLogLog.
</details>

<details>
<summary><strong>v0.21 — Change Data Capture v2</strong></summary>

Durable cursors (`DurableCursor`) with at-least-once delivery, consumer groups (`ConsumerGroupManager`) with round-robin shard assignment and rebalancing, exactly-once ACK semantics.
</details>

<details>
<summary><strong>v0.22 — Event Sourcing</strong></summary>

`create_event_store()`, `append_event()`, `get_events()`, aggregate projections with `get_aggregate_state()`, snapshot support, idempotency keys, `find_aggregates_by_event_type()`.
</details>

<details>
<summary><strong>v0.23 — Authentication & Authorization</strong></summary>

User management (create, authenticate, disable, change password), role-based access control with built-in admin/reader/writer roles, table-level permissions, privilege checking, session management with token-based TTL.
</details>

<details>
<summary><strong>v0.24 — Connection Pooling</strong></summary>

Configurable connection pool with warmup, LIFO reuse, idle eviction, RAII connection guards, pool statistics.
</details>

<details>
<summary><strong>v0.25 — Monitoring & Diagnostics</strong></summary>

Metrics registry (counters, gauges, HDR histograms), slow query log, timer guard, JSON-serializable metrics snapshots.
</details>

<details>
<summary><strong>v0.26 — Schema Evolution</strong></summary>

Migration manager (register, apply, rollback, apply_all), schema version registry with per-table column history, schema diff.
</details>

<details>
<summary><strong>v0.27 — Replication Foundations</strong></summary>

Raft consensus (`RaftNode` with leader election, vote, log replication), cluster membership (`NodeRegistry`), shard assignment with configurable replication factor.
</details>

<details>
<summary><strong>v0.28 — Fast Write Engine</strong></summary>

Lock-free `FastWritePath` bypassing crossbeam channel (1.9 µs writes, 20x faster than SQLite). Group-commit WAL via `DurabilityThread` with one fdatasync per batch cycle. Automatic fallback to channel path on backpressure or subscriber activity.
</details>

## Roadmap

> **Strategy**: Fix foundations → Make it fast → Own the niche (bitemporal + AI + embedded) → Speak Postgres → Then expand.

### Upcoming

- **AI Runtime v2** — Pluggable model backends (ONNX, HTTP), anomaly detection, pattern learning, cross-shard correlation
- **Advanced Vector Search** — `VECTOR(n)` column type, IVF-PQ for billion-scale, hybrid search (vector + BM25 + temporal)
- **ML Pipelines** — In-database feature store, model registry, training data export, inference UDFs
- **Encryption & Compliance** — AES-256-GCM encryption at rest, key rotation, column-level encryption, audit log, GDPR erasure
- **Cloud-Native** — S3 storage backend, Helm chart, Kubernetes operator, compute-storage separation
- **Horizontal Scaling** — Distributed shard routing, online rebalancing, distributed transactions
- **v1.0 Stable Release** — Stable on-disk format, Jepsen testing, TPC-H/YCSB benchmarks, crates.io/PyPI/npm publishing

See the full roadmap with per-version feature tracking in the [design.md](design.md).

## Project Structure

```
spectradb/
├── crates/
│   ├── spectradb-core/          # Database engine (main crate, ~15k lines)
│   │   └── src/
│   │       ├── ai/              # AI runtime, inference, ML pipeline, advisors
│   │       ├── engine/          # Database, shard, fast write path, change feeds
│   │       ├── storage/         # SSTable, WAL, compaction, levels, cache, columnar, group WAL
│   │       ├── sql/             # Parser, executor, evaluator, planner, vectorized engine
│   │       ├── facet/           # Relational, FTS, time-series, vector search, event sourcing, schema evolution
│   │       ├── cluster/         # Raft consensus, replication, scaling, membership
│   │       ├── io/              # io_uring async I/O (optional)
│   │       ├── ledger/          # Key encoding with bitemporal metadata
│   │       └── util/            # Bloom filters, varint, checksums
│   ├── spectradb-cli/           # Interactive shell and CLI commands
│   ├── spectradb-server/        # PostgreSQL wire protocol server (pgwire)
│   ├── spectradb-native/        # Optional C++ acceleration (cxx)
│   ├── spectradb-python/        # Python bindings (PyO3 / maturin)
│   └── spectradb-node/          # Node.js bindings (napi-rs)
├── tests/                       # 224+ integration tests across 30 suites
├── benches/                     # Criterion benchmarks (basic, comparative, multi-engine)
├── examples/                    # quickstart.rs, bitemporal.rs, ai_native.rs
├── docs/                        # Interactive documentation site (Starlight/Astro)
├── scripts/                     # Benchmark matrix, AI overhead gate, overnight burn-in
└── .github/workflows/ci.yml     # CI: fmt, clippy, test, AI overhead gate
```

## Documentation

**[Interactive Documentation Site](docs/)** — 58 pages with live SQL playground, animated architecture diagrams, performance comparisons, and interactive configuration explorer.

```bash
cd docs && npm install && npm run dev
# Opens at http://localhost:4321
```

| Document | Description |
|----------|-------------|
| [docs/](docs/) | Interactive documentation site (Starlight/Astro) |
| [design.md](design.md) | Internal architecture, data model, storage format |
| [perf.md](perf.md) | Tuning knobs, benchmark methodology, optimization notes |
| [TEST_PLAN.md](TEST_PLAN.md) | Correctness, recovery, temporal, and soak test strategy |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and contribution guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Building

```bash
# Pure Rust (default)
cargo build
cargo test --workspace --all-targets

# With C++ acceleration
cargo test --workspace --all-targets --features native

# With SIMD-accelerated bloom probes and checksums
cargo test --features simd

# With io_uring async I/O (Linux only)
cargo test --features io-uring

# With Parquet support (Apache Arrow)
cargo test --features parquet

# Lint and format (CI enforces these)
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

# Run benchmarks
cargo bench --bench comparative
cargo bench --bench multi_engine
cargo bench --bench basic

# AI overhead regression gate
./scripts/ai_overhead_gate.sh

# Build Python bindings
cd crates/spectradb-python && maturin develop

# Build Node.js bindings
cd crates/spectradb-node && npm run build

# Build documentation site
cd docs && npm install && npm run build
```

## CI Pipeline

Two jobs run on every push and PR to `main`:

1. **test-rust** — `cargo fmt --check` → `cargo clippy -D warnings` → `cargo test --workspace` → AI overhead gate script
2. **test-native** — C++ toolchain → `cargo clippy --features native` → `cargo test --features native`

## Contributing

We welcome contributions. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

SpectraDB is licensed under the [MIT License](LICENSE).
