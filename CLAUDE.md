# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Build (pure Rust, default features include llm)
cargo build

# Run all tests (~698 tests across 33 suites)
cargo test --workspace --all-targets

# Run a single test suite
cargo test --test ai_core
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

# AI overhead regression gate (CI runs this)
./scripts/ai_overhead_gate.sh

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

7-crate Rust workspace. Default members: root, core, cli, server. Python and Node crates require their respective build tools.

| Crate | Purpose |
|-------|---------|
| `tensordb` (root) | Re-exports `tensordb-core` |
| `tensordb-core` | Database engine: storage, SQL, AI runtime, facets (~31k LoC) |
| `tensordb-cli` | Interactive shell (rustyline, clap) |
| `tensordb-server` | PostgreSQL wire protocol server (tokio, pgwire v3) |
| `tensordb-native` | Optional C++ acceleration via cxx (behind `--features native`) |
| `tensordb-python` | Python bindings (PyO3/maturin) |
| `tensordb-node` | Node.js bindings (napi-rs) |

## Architecture

### Data Model

Every record is an **immutable fact** with: `user_key`, `commit_ts` (system time), `valid_from`/`valid_to` (business time), and `payload`. Facts are never overwritten — updates create new facts with higher `commit_ts`, deletes create tombstones.

**Internal key encoding:** `user_key || 0x00 || commit_ts(8B big-endian) || kind(1B)` — enables prefix scans for all versions of a key and chronological ordering.

### Engine Layers (top to bottom)

1. **Client Layer** — Rust API, CLI, Python/PyO3, Node.js/napi-rs, pgwire server
2. **SQL Engine** (`crates/tensordb-core/src/sql/`) — Hand-written recursive descent parser, cost-based planner with `PlanNode` variants (PointLookup, PrefixScan, IndexScan, FullScan, HashJoin, etc.), executor, and vectorized batch engine (1024-row batches with typed column vectors)
3. **Facet Layer** (`crates/tensordb-core/src/facet/`) — Specialized engines: relational (typed tables/views/indexes), full-text search (BM25), time-series (bucketing/gap-fill), vector search (HNSW), event sourcing
4. **Shard Engine** (`crates/tensordb-core/src/engine/`) — Database owns N shards (default 4). Fast write path (~1.9us lock-free) bypasses channels; direct reads (~276ns) bypass shard actors via `ShardReadHandle`
5. **Storage Engine** (`crates/tensordb-core/src/storage/`) — LSM-tree: WAL (CRC-framed) -> Memtable (BTreeMap) -> SSTables (LZ4 compressed, bloom filters) -> L0-L6 compaction

### Key Subsystems

- **EOAC Transactions** — `global_epoch: Arc<AtomicU64>` incremented per commit. Incomplete transactions (missing TXN_COMMIT marker) are rolled back on recovery.
- **MVCC** — Reads filter by `commit_ts` for snapshot isolation. Temporal queries: `AS OF` (system time), `VALID AT` (business time).
- **AI Runtime** (`crates/tensordb-core/src/ai/`) — Separate thread receiving `ChangeEvent`s, batching (20ms window, max 16 events), synthesizing insights stored under `__ai/` prefix. Includes embedded LLM (Qwen3 0.6B via llama-cpp-2, feature-gated).
- **CDC** (`crates/tensordb-core/src/cdc/`) — Durable cursors with at-least-once delivery, consumer groups with rebalancing.
- **Auth/RBAC** (`crates/tensordb-core/src/auth/`) — Users, roles (admin/reader/writer), sessions with TTL, table-level privileges.

### Internal Key Prefixes

- `__meta/table/{name}` — Table schema metadata
- `__meta/view/{name}` — View definitions
- `__meta/index/{table}/{name}` — Index metadata
- `__meta/ts_table/{name}` — Time-series metadata
- `__meta/fts_index/{table}/{name}` — Full-text search index metadata
- `__ai/insight/{hex_key}/{commit_ts}` — AI-generated insights
- `__ai/correlation/{cluster_id}/{commit_ts}/{hex_key}` — AI correlation refs

### Concurrency Model

- **Shard actor threads** (1 per shard) handle writes via crossbeam channels
- **ShardReadHandle** with `parking_lot::RwLock` enables lock-free concurrent reads
- **FastWritePath** uses atomic CAS for lock-free writes (falls back to channel if CDC subscribers exist)
- **DurabilityThread** batches WAL fsyncs across shards (default 1000us interval)
- **AI runtime thread** processes change events asynchronously

### Feature Flags

`default = ["llm"]`. Optional: `native` (C++ FFI), `simd`, `io-uring`, `parquet` (Apache Arrow), `encryption` (AES-256-GCM).

## Code Conventions

- **Pure-Rust first.** The pure-Rust path must always be fully functional. Native optimizations stay behind `--features native`.
- **Preserve temporal invariants.** Bitemporal + MVCC semantics are the core guarantee.
- Use `thiserror` for error types (`TensorError` enum in `error.rs`, `Result<T>` alias).
- Use `Arc<T>` for shared ownership, `parking_lot::RwLock` for read-heavy shared state, `crossbeam_channel` for inter-thread communication.
- Keep modules focused — one concern per file.
- Don't commit generated artifacts: `target/`, `data/`, `bench_runs/`, `overnight_runs/`.

## CI Gates

All must pass: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace --all-targets`, `cargo test --workspace --all-targets --features native`, `./scripts/ai_overhead_gate.sh`.
