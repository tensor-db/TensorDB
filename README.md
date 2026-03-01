<p align="center">
  <img src="assets/tensordb-logo.svg" alt="TensorDB" width="560">
</p>

<p align="center">
  <strong>An AI-native, bitemporal ledger database with MVCC, full SQL, real transactions, PITR, vector search, and sub-microsecond performance.</strong>
</p>

<p align="center">
  <a href="https://github.com/tensor-db/tensorDB/actions"><img src="https://github.com/tensor-db/tensorDB/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-PolyForm--Noncommercial-blue.svg" alt="License: PolyForm Noncommercial"></a>
  <a href="https://pypi.org/project/tensordb/"><img src="https://img.shields.io/pypi/v/tensordb.svg" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/tensordb"><img src="https://img.shields.io/npm/v/tensordb.svg" alt="npm"></a>
  <a href="https://crates.io/crates/tensordb"><img src="https://img.shields.io/crates/v/tensordb.svg" alt="crates.io"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-stable-orange.svg" alt="Rust"></a>
</p>

---

TensorDB is an embedded database that treats every write as an immutable fact. It separates **system time** (when data was recorded) from **business-valid time** (when data was true), giving you built-in time travel, auditability, and point-in-time recovery with zero application-level bookkeeping. Written in Rust, it ships as a library for Rust, Python, and Node.js — no server process required.

## Performance

Full 4-way Criterion benchmark (optimized release build):

| Operation | TensorDB | SQLite | sled | redb |
|-----------|----------|--------|------|------|
| Point Read | **273 ns** | 1,145 ns | 278 ns | 570 ns |
| Point Write | **3.6 µs** | 41.9 µs | 4.3 µs | 1,392 µs |
| Batch Write (100) | 1,063 µs | 344 µs | 588 µs | 4,904 µs |
| Prefix Scan (1k) | 249 µs | 137 µs | 165 µs | 63 µs |
| Mixed 80r/20w | 34 µs | 9.5 µs | 1.1 µs | 277 µs |
| SQL SELECT (100 rows) | **56 µs** | — | — | — |

| Throughput (reads/sec) | 1k keys | 10k keys | 50k keys |
|------------------------|---------|----------|----------|
| TensorDB | **3.8M** | **3.5M** | **1.9M** |
| sled | 4.6M | 3.5M | 2.8M |

- **4.2x faster reads** than SQLite, on par with sled
- **11.6x faster writes** than SQLite via lock-free fast write path
- **273 ns** point reads via direct shard bypass (no channel round-trip)
- **3.6 µs** point writes via `FastWritePath` with group-commit WAL
- **3.8M reads/sec** sustained throughput at 1k dataset size

Benchmarks use Criterion 0.5. Run them yourself:

```bash
cargo bench --bench comparative    # TensorDB vs SQLite
cargo bench --bench multi_engine   # TensorDB vs SQLite vs sled vs redb
cargo bench --bench basic          # Microbenchmarks
```

## Install

```bash
# Python
pip install tensordb

# Node.js
npm install tensordb

# Rust
cargo add tensordb

# Interactive CLI
cargo install tensordb-cli
cargo run -p tensordb-cli -- --path ./mydb

# PostgreSQL wire protocol server
cargo run -p tensordb-server -- --data-dir ./mydb --port 5433

# Docker
docker compose up -d
psql -h localhost -p 5433
```

## Quickstart

```sql
-- Create a typed table
CREATE TABLE accounts (id INTEGER PRIMARY KEY, name TEXT NOT NULL, balance REAL);
INSERT INTO accounts (id, name, balance) VALUES (1, 'alice', 1000.0), (2, 'bob', 500.0);

-- Standard SQL: joins, window functions, CTEs
SELECT a.name, e.doc FROM accounts a
JOIN events e ON a.name = e.doc->>'user';

SELECT name, balance, ROW_NUMBER() OVER (ORDER BY balance DESC) AS rank FROM accounts;

WITH high_value AS (SELECT * FROM accounts WHERE balance > 500)
SELECT * FROM high_value;

-- Time travel: read state as of a specific commit or epoch
SELECT * FROM accounts AS OF 1;
SELECT * FROM accounts AS OF EPOCH 5;

-- Bitemporal: SQL:2011 temporal queries
SELECT * FROM accounts FOR SYSTEM_TIME AS OF 1;
SELECT * FROM accounts FOR APPLICATION_TIME AS OF 1000;

-- Transactions with savepoints
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
SAVEPOINT sp1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
ROLLBACK TO sp1;
COMMIT;

-- Full-text search
CREATE FULLTEXT INDEX idx_docs ON events (doc);
SELECT pk, HIGHLIGHT(doc, 'signup') FROM events WHERE MATCH(doc, 'signup');

-- Vector search: VECTOR(n) columns, HNSW/IVF-PQ indexes, k-NN via <-> operator
CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, embedding VECTOR(384));
CREATE VECTOR INDEX idx ON docs (embedding) USING HNSW WITH (m = 32, ef_construction = 200, metric = 'cosine');
INSERT INTO docs (id, title, embedding) VALUES (1, 'intro', '[0.1, 0.2, 0.3, ...]');
SELECT id, title, embedding <-> '[0.1, 0.2, ...]' AS distance FROM docs ORDER BY distance LIMIT 10;

-- Hybrid search: combine vector similarity with BM25 text relevance
SELECT id, HYBRID_SCORE(embedding <-> '[0.1, ...]', MATCH(body, 'quantum'), 0.7, 0.3) AS score
FROM docs WHERE MATCH(body, 'quantum') ORDER BY score DESC LIMIT 10;

-- Vector search table function
SELECT * FROM vector_search('docs', 'embedding', '[0.1, 0.2, ...]', 10);

-- Time-series
CREATE TIMESERIES TABLE metrics (ts TIMESTAMP, value REAL) WITH (bucket_size = '1h');
SELECT TIME_BUCKET('1h', ts), AVG(value) FROM metrics GROUP BY 1;

-- Data interchange
COPY accounts TO '/tmp/accounts.csv' FORMAT CSV;
SELECT * FROM read_parquet('data.parquet');

-- Incremental backup
BACKUP TO '/tmp/backup.json' SINCE EPOCH 3;
```

```bash
# Run built-in examples
cargo run --example quickstart     # Core features: SQL, time-travel, prepared statements
cargo run --example bitemporal     # Bitemporal ledger: AS OF + VALID AT queries
cargo run --example ai_native      # AI runtime: insights, risk scoring, query planning
```

## Key Features

### Core Database
- **Immutable Fact Ledger** — Append-only WAL with CRC-framed records. Data is never overwritten.
- **EOAC Transactions** — Epoch-Ordered Append-Only Concurrency with global epoch counter, `BEGIN`/`COMMIT`/`ROLLBACK`/`SAVEPOINT`.
- **MVCC Snapshot Reads** — Query any past state with `AS OF <commit_ts>` or `AS OF EPOCH <n>`.
- **Point-in-Time Recovery** — `SELECT ... AS OF EPOCH <n>` for cross-shard consistent snapshots.
- **Incremental Backup** — `BACKUP TO '<path>' SINCE EPOCH <n>` for delta exports.
- **Bitemporal Filtering** — SQL:2011 `SYSTEM_TIME` and `APPLICATION_TIME` temporal clauses.
- **LSM Storage Engine** — Memtable → SSTable (L0–L6) with bloom filters, prefix compression, mmap reads, LZ4 block compression.
- **Block & Index Caching** — LRU caches with configurable memory budgets.
- **Write Batch API** — Atomic multi-key writes with a single WAL frame.
- **Encryption at Rest** — AES-256-GCM block-level encryption (`--features encryption`).

### SQL Engine
- **Full SQL** — DDL, DML, SELECT, JOINs (inner/left/right/cross), GROUP BY, HAVING, CTEs, subqueries, UNION/INTERSECT/EXCEPT, window functions, CASE, CAST, LIKE/ILIKE, transactions.
- **60+ built-in functions** — String, numeric, date/time, aggregate, window, conditional, type conversion, vector search.
- **Cost-based query planner** — `PlanNode` tree with cost estimation, `EXPLAIN` and `EXPLAIN ANALYZE`.
- **Prepared statements** — Parse once, execute many with `$1, $2, ...` parameter binding.
- **Temporal SQL** — 7 SQL:2011 temporal clause variants for both system time and application time.
- **Vectorized execution** — Columnar `RecordBatch` engine with vectorized filter, project, aggregate, join, and sort.

### Specialized Engines
- **Full-Text Search** — `CREATE FULLTEXT INDEX`, `MATCH()`, `HIGHLIGHT()`, BM25 ranking, multi-column with per-column boosting.
- **Time-Series** — `CREATE TIMESERIES TABLE`, `TIME_BUCKET()`, gap filling (`LOCF`, `INTERPOLATE`), `DELTA()`, `RATE()`.
- **Vector Search** — `VECTOR(n)` column type, HNSW and IVF-PQ indexes, `<->` distance operator, `vector_search()` table function, hybrid search (vector + BM25 via `HYBRID_SCORE`), temporal vector queries, cosine/Euclidean/dot-product distance, FP16/INT8 quantization.
- **Event Sourcing** — Aggregate projections, snapshot support, idempotency keys, cross-aggregate event queries.
- **Schema Evolution** — Migration manager with versioned SQL migrations, schema diff, rollback support.

### Data Platform
- **Change Data Capture** — Prefix-filtered subscriptions, durable cursors, consumer groups with rebalancing.
- **Data Interchange** — `COPY TO/FROM` CSV, JSON, Parquet. Table functions: `read_csv()`, `read_json()`, `read_parquet()`.
- **PostgreSQL Wire Protocol** — `tensordb-server` crate accepts Postgres client connections via pgwire.
- **Authentication & RBAC** — User management, role-based access control, table-level permissions, session management.
- **Connection Pooling** — Configurable pool with warmup, idle eviction, and RAII connection guards.

### AI Runtime
- **Background Insight Synthesis** — In-process AI pipeline consuming change feeds.
- **Inline Risk Scoring** — Per-write risk assessment without external model servers.
- **AI Advisors** — Compaction scheduling, cache tuning, query optimization recommendations.
- **ML Pipeline** — Feature store, model registry, point-in-time joins, inference metrics.
- **`EXPLAIN AI`** — SQL command for AI insights, provenance, and risk scores per key.

### Language Bindings & Integrations
- **Rust** — Native embedded library (`tensordb-core`).
- **Python** — PyO3 bindings (`tensordb-python`) — `open()`, `put()`, `get()`, `sql()`.
- **Node.js** — napi-rs bindings (`tensordb-node`) — `open()`, `put()`, `get()`, `sql()`.
- **Interactive CLI** — TAB completion, persistent history, table/line/JSON output modes.
- **Optional C++ Acceleration** — `--features native` via `cxx` for Hasher, Compressor, BloomProbe.
- **Optional io_uring** — `--features io-uring` for Linux async I/O.
- **Optional SIMD** — `--features simd` for hardware-accelerated bloom probes and checksums.

## Use Cases

<table>
<tr>
<td width="50%" valign="top">

### Embedded Application Database
Drop-in embedded database for any app that needs real SQL — no server process, no Docker, no network. Use it from Rust, Python, or Node.js. Like SQLite, but with 4x faster reads and built-in version history.

</td>
<td width="50%" valign="top">

### Apps That Need an Undo Button
Every write is preserved. Roll back to any previous state with a single query. Build version history, audit trails, or time-travel debugging into your app without extra bookkeeping.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### AI-Powered Applications
Store vectors alongside your regular data. Run semantic search, full-text search, and SQL queries in one database. No need to sync between a vector store, a search engine, and a relational DB.

</td>
<td width="50%" valign="top">

### High-Throughput Ingestion
Sub-microsecond writes handle sensors, logs, metrics, and event streams at scale. The time-series engine adds bucketed aggregation, gap filling, and rate calculations out of the box.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Local-First & Edge Computing
Ship a full-featured database as a library — no infrastructure to manage. Works on desktops, IoT gateways, edge nodes, and anywhere you need data processing without a network round-trip.

</td>
<td width="50%" valign="top">

### Financial & Regulated Systems
Immutable append-only storage with bitemporal queries satisfies audit and compliance requirements. Reconstruct the exact state of any record at any point in time — system time and business time tracked separately.

</td>
</tr>
</table>

## Architecture

TensorDB is organized around four core principles: **immutable truth** (the append-only ledger), **epoch ordering** (global epoch counter unifying transactions, MVCC, and recovery), **temporal indexing** (bitemporal metadata on every fact), and **faceted queries** (pluggable query planes over the same data).

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
        Parser[SQL Parser<br/>60+ functions · CTEs · windows]
        Planner[Cost-Based Planner<br/>PlanNode tree · EXPLAIN ANALYZE]
        Executor[Query Executor<br/>scans · joins · aggregates · windows]
        VecEngine[Vectorized Engine<br/>columnar batches · RecordBatch]
    end

    subgraph Facet Layer
        RF[Relational Facet<br/>typed tables · views · indexes]
        FTS[Full-Text Search<br/>inverted index · BM25 · stemmer]
        TS[Time-Series<br/>time_bucket · gap fill · rate]
        VS[Vector Search<br/>HNSW · IVF-PQ · hybrid · temporal]
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
| Epoch-ordered concurrency | Global epoch counter unifies transactions, PITR, and incremental backup under one mechanism |
| Cross-shard epoch sync | advance_epoch() bumps all shard commit counters for consistent cross-shard point-in-time snapshots |

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
<summary><strong>Date/Time Functions (5)</strong></summary>

`NOW`/`CURRENT_TIMESTAMP`, `EPOCH`, `EXTRACT`/`DATE_PART`, `DATE_TRUNC`, `TO_CHAR`
</details>

<details>
<summary><strong>Aggregate Functions (10)</strong></summary>

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
<summary><strong>Vector Search Functions (5)</strong></summary>

`VECTOR_DISTANCE(v1, v2, metric)`, `COSINE_SIMILARITY(v1, v2)`, `VECTOR_NORM(v)`, `VECTOR_DIMS(v)`, `HYBRID_SCORE(vector_dist, bm25_score, vector_weight, text_weight)`
</details>

<details>
<summary><strong>Conditional & Utility (7)</strong></summary>

`COALESCE`, `NULLIF`, `GREATEST`, `LEAST`, `IF`/`IIF`, `TYPEOF`, `CAST`
</details>

## Configuration

TensorDB is configured through the `Config` struct. All parameters have sensible defaults.

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

## Project Structure

```
tensordb/
├── crates/
│   ├── tensordb-core/           # Database engine (main crate, ~31k lines)
│   │   └── src/
│   │       ├── ai/              # AI runtime, inference, ML pipeline, advisors
│   │       ├── engine/          # Database, shard, fast write path, change feeds
│   │       ├── storage/         # SSTable, WAL, compaction, levels, cache, columnar, group WAL
│   │       ├── sql/             # Parser, executor, evaluator, planner, vectorized engine
│   │       ├── facet/           # Relational, FTS, time-series, vector search, event sourcing, schema evolution
│   │       ├── cluster/         # Raft consensus, replication, scaling, membership
│   │       ├── auth/            # Authentication, RBAC, session management
│   │       ├── cdc/             # Change data capture, durable cursors, consumer groups
│   │       ├── io/              # io_uring async I/O (optional)
│   │       ├── ledger/          # Key encoding with bitemporal metadata
│   │       └── util/            # Varint encoding, metrics, time utilities
│   ├── tensordb-cli/            # Interactive shell and CLI commands
│   ├── tensordb-server/         # PostgreSQL wire protocol server (pgwire)
│   ├── tensordb-native/         # Optional C++ acceleration (cxx)
│   ├── tensordb-distributed/     # Horizontal scaling: routing, 2PC, rebalancing
│   ├── tensordb-python/         # Python bindings (PyO3 / maturin)
│   └── tensordb-node/           # Node.js bindings (napi-rs)
├── tests/                       # 740+ tests across 35 suites
├── benches/                     # Criterion benchmarks (basic, comparative, multi-engine)
├── examples/                    # quickstart.rs, bitemporal.rs, ai_native.rs, fastapi, express
├── docs/                        # Interactive documentation site (Starlight/Astro)
├── scripts/                     # Benchmark matrix, AI overhead gate, overnight burn-in
├── Dockerfile                   # Multi-stage Docker image for tensordb-server
├── docker-compose.yml           # Docker Compose example with volume and healthcheck
└── .github/workflows/           # CI, crates.io publish, Docker image publish
```

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
cd crates/tensordb-python && maturin develop

# Build Node.js bindings
cd crates/tensordb-node && npm run build

# Build documentation site
cd docs && npm install && npm run build
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

## CI Pipeline

**On every push and PR to `main`:**

1. **test-rust** — `cargo fmt --check` → `cargo clippy -D warnings` → `cargo test --workspace` → AI overhead gate script
2. **test-native** — C++ toolchain → `cargo clippy --features native` → `cargo test --features native`

**On release:**

3. **publish-crates** — Publish `tensordb-core` and `tensordb` to crates.io
4. **publish-docker** — Build and push multi-arch Docker image to `ghcr.io`

## Roadmap

> **Strategy**: Fix foundations → Make it fast → Own the niche (bitemporal + AI + embedded) → Speak Postgres → Delight users → Then scale.

### Done

- **v0.1–v0.10** — Core engine, SQL, storage, query planner, prepared statements
- **v0.11–v0.18** — Temporal SQL:2011, FTS, time-series, pgwire, data interchange, vectorized execution
- **v0.19–v0.26** — Columnar storage, CDC, event sourcing, auth/RBAC, connection pooling, monitoring, schema evolution
- **v0.27–v0.28** — Replication foundations, fast write engine
- **v0.29** — EOAC transactions, PITR, incremental backup, encryption at rest
- **v0.2.0** — Embedded LLM (Qwen3 0.6B via llama-cpp-2)
- **v0.30** — Advanced vector search (VECTOR(n), HNSW/IVF-PQ, hybrid search, temporal vectors), horizontal scaling (tensordb-distributed crate), ecosystem (Docker, CI publish workflows, example apps)

### Phase 1: Observability & Diagnostics

Surface the internals so users can answer "what is my database doing?" without guessing.

- **`SHOW STATS`** — SQL command exposing MetricsRegistry: cache hit rates, compaction stats, WAL size, bloom filter false positive rate, shard load distribution
- **`SHOW SLOW QUERIES`** — SQL-accessible slow query log with query text, duration, rows scanned, plan used
- **`SHOW ACTIVE QUERIES`** — List currently running queries with elapsed time and resource usage
- **`SHOW STORAGE`** — Per-table/index/WAL disk space breakdown: data size, index size, tombstone count, compaction amplification ratio
- **`SHOW COMPACTION STATUS`** — L0 file count, pending compactions, last compaction time, bytes written per level
- **Live query profiling** — Per-query latency histogram, rows scanned vs returned ratio, cache miss tracking
- **Health endpoint** — `/health` JSON endpoint for pgwire server with liveness, readiness, and storage metrics

### Phase 2: Error Quality & Developer Experience

Make errors helpful instead of cryptic. Reduce friction for new users.

- **Structured error codes** — Stable numeric codes (`T1001 SYNTAX_ERROR`, `T2001 TABLE_NOT_FOUND`, etc.) for programmatic handling, mapped to categories (syntax, schema, constraint, IO, auth)
- **"Did you mean?" suggestions** — Levenshtein-based fuzzy matching for misspelled table names, column names, SQL keywords, and function names
- **`SUGGEST INDEX FOR <query>`** — Analyze a query and recommend optimal indexes based on WHERE clauses, JOIN predicates, and ORDER BY columns
- **Progress indicators** — Long-running operations (`COPY`, `BACKUP`, `RESTORE`, compaction, bulk INSERT) report rows processed, bytes written, ETA
- **Strict mode** — `SET STRICT_MODE = ON` to fail on silent type coercion, truncation, and implicit NULL insertion instead of silently proceeding
- **`VERIFY BACKUP <path>`** — Validate backup file integrity (checksums, key counts, epoch consistency) without restoring
- **`VACUUM`** — Reclaim space from tombstones and old temporal versions, report bytes freed and compaction stats
- **CLI autocomplete** — Context-aware TAB completion for table names, column names, SQL keywords, and function names in the interactive shell

### Phase 3: Security & Audit

Enterprise-grade security beyond basic RBAC.

- **Audit log** — Append-only log of all DDL changes, auth events (login, failed attempts, permission changes), and data access patterns, queryable via `SELECT * FROM __audit_log`
- **Row-level security** — `CREATE POLICY` for per-row access control based on session user, role, or arbitrary predicates
- **Key rotation** — Rotate encryption keys without downtime: re-encrypt WAL and SSTables in background, track key versions per file
- **Column-level encryption** — `CREATE TABLE ... (ssn TEXT ENCRYPTED)` for encrypting sensitive columns at rest with per-column keys
- **GDPR erasure** — `FORGET KEY <key>` to cryptographically erase all temporal versions of a record while preserving ledger structure

### Phase 4: Operational Maturity

Production-hardening for teams running TensorDB in real workloads.

- **Per-query resource limits** — `SET QUERY_TIMEOUT = 5000` and `SET QUERY_MAX_MEMORY = '256MB'` to prevent runaway queries
- **Online DDL** — `ALTER TABLE ADD COLUMN`, `DROP COLUMN`, `RENAME COLUMN` without table locks or downtime
- **Plan stability** — `CREATE PLAN GUIDE` to pin query plans and prevent regressions from stale statistics
- **Backup dry-run** — `RESTORE FROM <path> DRY_RUN` to validate a restore without writing data, showing what would change
- **Compaction scheduling** — User-configurable compaction windows (`SET COMPACTION_WINDOW = '02:00-06:00'`) to avoid peak-hour I/O
- **WAL size management** — Configurable WAL retention, automatic WAL archival, and `SHOW WAL STATUS` for monitoring

### Phase 5: AI Runtime v2

Next-generation AI capabilities tightly integrated with the query engine.

- **Pluggable model backends** — ONNX Runtime, HTTP model servers, and local llama.cpp behind a unified `ModelBackend` trait
- **Anomaly detection** — Automatic detection of unusual write patterns, schema drift, and query performance regressions
- **Pattern learning** — Learn access patterns over time to auto-tune cache policies, prefetch strategies, and compaction schedules
- **Inference UDFs** — `SELECT predict(model_name, features...) FROM data` for in-database model inference
- **Cross-shard correlation** — AI runtime correlates change events across shards for holistic insight synthesis

### Phase 6: Cloud-Native & Scale

From embedded to distributed.

- **S3 storage backend** — Tiered storage with hot data on local SSD, cold data on S3-compatible object storage
- **Compute-storage separation** — Stateless query nodes reading from shared storage
- **gRPC transport** — Wire up `tensordb-distributed` with tonic/prost for actual multi-node deployment
- **Helm chart & Kubernetes operator** — Deploy TensorDB clusters on Kubernetes with auto-scaling and self-healing
- **ML Pipelines** — In-database feature store with point-in-time joins, model registry, training data export

### v1.0 Stable Release

- **Stable on-disk format** — Backward-compatible SSTable and WAL format with versioned headers
- **Jepsen testing** — Formal verification of transaction isolation and crash recovery guarantees
- **TPC-H / YCSB benchmarks** — Industry-standard benchmark results for credibility
- **Semantic versioning** — No breaking API or format changes within major versions
- **Migration tooling** — `tensordb-migrate` for upgrading on-disk format between major versions

See the full changelog in [CHANGELOG.md](CHANGELOG.md) and architecture details in [design.md](design.md).

## Contributing

We welcome contributions. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

TensorDB is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may use it freely for personal, educational, research, and non-commercial purposes. Commercial use requires a paid license — contact walebadr@users.noreply.github.com.
