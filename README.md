<p align="center">
  <img src="assets/spectradb-logo.svg" alt="SpectraDB" width="560">
</p>

<p align="center">
  <strong>An append-only, bitemporal ledger database with MVCC snapshot reads and a SQL query interface.</strong>
</p>

<p align="center">
  <a href="https://github.com/spectra-db/SpectraDB/actions"><img src="https://github.com/spectra-db/SpectraDB/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-stable-orange.svg" alt="Rust"></a>
</p>

---

SpectraDB is a single-node embedded database that treats every write as an immutable fact. It separates the **system timeline** (when data was recorded) from the **business-valid timeline** (when data was true), giving you built-in time travel and auditability with zero application-level bookkeeping.

## Key Features

- **Immutable Fact Ledger** — Write-ahead log with CRC-framed records. Data is never overwritten; updates create new versions.
- **MVCC Snapshot Reads** — Query any past state with `AS OF <commit_ts>`.
- **Bitemporal Filtering** — Separate system and valid-time dimensions with `VALID AT <valid_ts>`.
- **LSM Storage Engine** — Memtable, SSTable with bloom filters, block index, and mmap reads. Background compaction keeps read amplification low.
- **SQL Interface** — `CREATE TABLE`, `INSERT`, `SELECT`, `JOIN`, `GROUP BY`, views, indexes, transactions, `EXPLAIN`, and more.
- **Interactive Shell** — TAB completion, persistent history, table/line/JSON output modes.
- **Optional Native Acceleration** — C++ kernels behind `--features native` via `cxx`, with pure Rust as the default.

## Quickstart

```bash
# Build
cargo build -p spectradb-cli

# Launch interactive shell
cargo run -p spectradb-cli -- --path ./mydb
```

```sql
CREATE TABLE events (pk TEXT PRIMARY KEY);

INSERT INTO events (pk, doc) VALUES ('evt-1', '{"type":"signup","user":"alice"}');
INSERT INTO events (pk, doc) VALUES ('evt-2', '{"type":"purchase","user":"bob"}');

-- Query latest state
SELECT pk, doc FROM events ORDER BY pk LIMIT 10;

-- Time travel: read state as of commit 1
SELECT doc FROM events WHERE pk='evt-1' AS OF 1;

-- Bitemporal: what was valid at a specific point
SELECT doc FROM events VALID AT 1000;

-- Analytics
SELECT count(*) FROM events;
SELECT pk, count(*) FROM events GROUP BY pk;

-- Transactions
BEGIN;
INSERT INTO events (pk, doc) VALUES ('evt-3', '{"type":"refund","user":"bob"}');
COMMIT;
```

## Architecture

```
┌─────────────────────────────────────────────┐
│                  SQL Parser                  │
│            (CREATE, INSERT, SELECT, ...)     │
├─────────────────────────────────────────────┤
│               Query Executor                 │
│       (scans, joins, grouping, explain)      │
├─────────────────────────────────────────────┤
│            Relational Facet                   │
│     (table/view/index metadata, schema)      │
├──────────┬──────────┬──────────┬────────────┤
│  Shard 0 │  Shard 1 │  Shard 2 │  Shard N   │
│  (WAL +  │  (WAL +  │  (WAL +  │  (WAL +    │
│ memtable │ memtable │ memtable │ memtable   │
│ + SSTs)  │ + SSTs)  │ + SSTs)  │ + SSTs)    │
├──────────┴──────────┴──────────┴────────────┤
│              Storage Engine                   │
│  WAL ─► Memtable ─► SSTable (L0) ─► Compact  │
│  Bloom filters · Block index · mmap reads    │
└─────────────────────────────────────────────┘
```

Each shard maintains its own WAL, memtable, and SSTable set with a single-writer execution model. Keys are hash-routed to shards for write concurrency without locking.

## Performance

SpectraDB ships with a built-in benchmark harness:

```bash
cargo run -p spectradb-cli -- --path /tmp/bench bench \
  --write-ops 100000 --read-ops 50000 --keyspace 20000 --read-miss-ratio 0.20
```

Sample numbers (single machine, sanity run):

| Metric | Value |
|--------|-------|
| Write throughput | ~4,500 ops/s |
| Read p50 latency | ~530 µs |
| Read p95 latency | ~890 µs |
| Read p99 latency | ~1,030 µs |

Tuning knobs: `--wal-fsync-every-n-records`, `--memtable-max-bytes`, `--sstable-block-bytes`, `--bloom-bits-per-key`, `--shard-count`. See [perf.md](perf.md) for details.

## Project Structure

```
spectradb/
├── crates/
│   ├── spectradb-core/    # Storage engine, SQL parser/executor, facets
│   ├── spectradb-cli/     # Interactive shell and CLI commands
│   └── spectradb-native/  # Optional C++ acceleration (cxx)
├── tests/                 # Integration tests
├── benches/               # Criterion benchmarks
├── scripts/               # Benchmark matrix, overnight burn-in
├── design.md              # Architecture deep dive
├── perf.md                # Performance notes and tuning guide
└── TEST_PLAN.md           # Validation strategy
```

## Documentation

| Document | Description |
|----------|-------------|
| [design.md](design.md) | Internal architecture, data model, storage format |
| [perf.md](perf.md) | Tuning knobs, benchmark methodology, optimization roadmap |
| [TEST_PLAN.md](TEST_PLAN.md) | Correctness, recovery, temporal, and soak test strategy |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup and contribution guidelines |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Building

```bash
# Pure Rust (default)
cargo test

# With C++ acceleration
cargo test --features native

# Run benchmarks
cargo bench
```

## Contributing

We welcome contributions. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

SpectraDB is licensed under the [MIT License](LICENSE).
