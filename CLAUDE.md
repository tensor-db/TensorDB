# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build (pure Rust)
cargo build

# Run all tests (~800 tests across 51 suites)
cargo test --workspace --all-targets

# Run a single test suite
cargo test --test sql_correctness
cargo test --test sql_correctness

# Run a single test by name
cargo test --test sql_correctness -- test_name

# Test with optional feature flags
cargo test --workspace --all-targets --features native   # C++ acceleration
cargo test --features simd                                # SIMD bloom filters
cargo test --features io-uring                            # Linux async I/O
cargo test --features parquet                             # Apache Parquet support

# Lint and format (CI enforces both)
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings

# Benchmarks
cargo bench --bench comparative     # TensorDB vs SQLite
cargo bench --bench multi_engine    # 4-way comparison
cargo bench --bench basic           # Microbenchmarks

# Run CLI
cargo run -p tensordb-cli -- --path ./mydb

# Run PostgreSQL wire protocol server
cargo run -p tensordb-server -- --data-dir ./mydb --port 5433

# Python bindings (requires maturin)
cd crates/tensordb-python && maturin develop

# Node.js bindings (requires napi-rs)
cd crates/tensordb-node && npm install && npm run build

# Documentation site
cd docs && npm install && npm run dev   # http://localhost:4321
```

## Workspace Structure

8-crate Rust workspace. Default members: root, core, cli, server. Python and Node crates require their respective build tools. Distributed crate is feature-gated.

| Crate | Purpose |
|-------|---------|
| `tensordb` (root) | Re-exports `tensordb-core` |
| `tensordb-core` | Database engine: storage, SQL, facets (~31k LoC) |
| `tensordb-cli` | Interactive shell (rustyline, clap) |
| `tensordb-server` | PostgreSQL wire protocol server (tokio, pgwire v3) |
| `tensordb-native` | Optional C++ acceleration via cxx (behind `--features native`) |
| `tensordb-distributed` | Horizontal scaling: shard routing, 2PC transactions, rebalancing |
| `tensordb-python` | Python bindings (PyO3/maturin) |
| `tensordb-node` | Node.js bindings (napi-rs) |

## Architecture

### Data Model

Every record is an **immutable fact** with: `user_key`, `commit_ts` (system time), `valid_from`/`valid_to` (business time), and `payload`. Facts are never overwritten — updates create new facts with higher `commit_ts`, deletes create tombstones.

**Internal key encoding:** `user_key || 0x00 || commit_ts(8B big-endian) || kind(1B)` — enables prefix scans for all versions of a key and chronological ordering.

### Engine Layers (top to bottom)

1. **Client Layer** — Rust API, CLI, Python/PyO3, Node.js/napi-rs, pgwire server
2. **SQL Engine** (`crates/tensordb-core/src/sql/`) — Hand-written recursive descent parser, cost-based planner with `PlanNode` variants (PointLookup, PrefixScan, IndexScan, FullScan, HashJoin, etc.), executor, and vectorized batch engine (1024-row batches with typed column vectors)
3. **Facet Layer** (`crates/tensordb-core/src/facet/`) — Specialized engines: relational (typed tables/views/indexes), full-text search (BM25), time-series (bucketing/gap-fill), vector search (HNSW/IVF-PQ with persistence, hybrid search, temporal vectors, quantization), event sourcing
4. **Shard Engine** (`crates/tensordb-core/src/engine/`) — Database owns N shards (default 4). Fast write path (~1.9us lock-free) bypasses channels; direct reads (~276ns) bypass shard actors via `ShardReadHandle`
5. **Storage Engine** (`crates/tensordb-core/src/storage/`) — LSM-tree: WAL (CRC-framed) -> Memtable (BTreeMap) -> SSTables (LZ4 compressed, bloom filters) -> L0-L6 compaction

### Key Subsystems

- **EOAC Transactions** — `global_epoch: Arc<AtomicU64>` incremented per commit. Incomplete transactions (missing TXN_COMMIT marker) are rolled back on recovery.
- **MVCC** — Reads filter by `commit_ts` for snapshot isolation. Temporal queries: `AS OF` (system time), `VALID AT` (business time).
- **Anomaly Detection** (`crates/tensordb-core/src/ai/`) — Write rate anomaly detection, learned cost model.
- **Observability** — `MetricsRegistry` (counters, gauges, histograms, slow query log) wired into `Database::sql()`. 8 SQL diagnostic commands: `SHOW STATS`, `SHOW SLOW QUERIES`, `SHOW ACTIVE QUERIES`, `SHOW STORAGE`, `SHOW COMPACTION STATUS`, `SHOW WAL STATUS`, `SHOW AUDIT LOG`, `SHOW PLAN GUIDES`. Block cache hit/miss tracking. Health HTTP endpoint on pgwire port+1.
- **Structured Errors** — `ErrorCode` enum (T1xxx syntax, T2xxx schema, T3xxx constraint, T4xxx execution, T6xxx auth) with `SqlError` struct carrying code, message, suggestion, position. Levenshtein fuzzy matching for "Did you mean?" suggestions.
- **CDC** (`crates/tensordb-core/src/cdc/`) — Durable cursors with at-least-once delivery, consumer groups with rebalancing.
- **Auth/RBAC** (`crates/tensordb-core/src/auth/`) — Users, roles (admin/reader/writer), sessions with TTL, table-level privileges. Audit log (`audit.rs`), row-level security policies (`rls.rs`), GDPR erasure (`FORGET KEY`).
- **Plan Stability** (`crates/tensordb-core/src/sql/plan_guide.rs`) — `PlanGuideManager` for pinning query plans via `CREATE PLAN GUIDE`.

### Internal Key Prefixes

- `__meta/table/{name}` — Table schema metadata
- `__meta/view/{name}` — View definitions
- `__meta/index/{table}/{name}` — Index metadata
- `__meta/ts_table/{name}` — Time-series metadata
- `__meta/fts_index/{table}/{name}` — Full-text search index metadata
- `__meta/vector_index/{table}/{column}` — Vector index metadata (HNSW/IVF-PQ params)
- `__vec/{table}/{column}/{pk}` — Vector data (raw f32 bytes)
- `__vec_idx/{table}/{column}/...` — Vector index structures (HNSW graph, IVF centroids, PQ codebook)
- `__meta/policy/{table}/{name}` — Row-level security policies
- `__meta/plan_guide/{name}` — Plan guide definitions
- `__audit_log/{timestamp}/{seq}` — Audit log events

### Concurrency Model

- **Shard actor threads** (1 per shard) handle writes via crossbeam channels
- **ShardReadHandle** with `parking_lot::RwLock` enables lock-free concurrent reads
- **FastWritePath** uses atomic CAS for lock-free writes (falls back to channel if CDC subscribers exist)
- **DurabilityThread** batches WAL fsyncs across shards (default 1000us interval)

### Feature Flags

Optional feature flags: `native` (C++ FFI), `simd`, `io-uring`, `parquet` (Apache Arrow), `encryption` (AES-256-GCM).

## Code Conventions

- **Pure-Rust first.** The pure-Rust path must always be fully functional. Native optimizations stay behind `--features native`.
- **Preserve temporal invariants.** Bitemporal + MVCC semantics are the core guarantee.
- Use `thiserror` for error types (`TensorError` enum in `error.rs`, `Result<T>` alias).
- Use `Arc<T>` for shared ownership, `parking_lot::RwLock` for read-heavy shared state, `crossbeam_channel` for inter-thread communication.
- Keep modules focused — one concern per file.
- Don't commit generated artifacts: `target/`, `data/`, `bench_runs/`, `overnight_runs/`.

## CI Gates

All must pass: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --all-targets`, `cargo test --workspace --all-targets --features native`.
