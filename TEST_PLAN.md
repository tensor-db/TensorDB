# SpectraDB Value-Proof Test Plan

This plan is designed to prove SpectraDB’s value beyond basic correctness.

## 1) Goals

- Prove correctness under normal and failure conditions.
- Quantify latency/throughput characteristics under realistic workloads.
- Validate temporal semantics (`AS OF`, `VALID AT`) at scale.
- Demonstrate operational reliability under long-running mixed workloads.
- Establish reproducible baseline comparisons against common local DB options.

## 2) Acceptance Gates

A build is considered MVP-acceptable only when all of the following pass:

1. `cargo test` passes on pure Rust mode.
2. `cargo test --features native` passes on native-enabled machine.
3. WAL fault-injection tests pass (CRC mismatch + torn tail behavior).
4. Temporal stress test passes with forced flush/compaction/reopen.
5. SQL facet integration tests pass for currently implemented statements and execution modes:
   - `CREATE TABLE`, `CREATE VIEW`, `CREATE INDEX`, `ALTER TABLE ... ADD COLUMN`
   - `INSERT`, point `SELECT`, temporal filters (`AS OF`, `VALID AT`), `EXPLAIN SELECT`
   - multi-statement batches and in-call transactions (`BEGIN`, `COMMIT`, `ROLLBACK`)
6. Overnight burn-in run completes with zero invariant failures.

## 3) Test Layers

### A) Unit and Component Correctness

- Varint roundtrip correctness.
- Internal key encode/decode consistency.
- WAL framing and replay invariants.
- Bloom filter encode/decode and may-contain behavior.
- SSTable build/open/read structural validation.

### B) Fault Injection and Recovery

- WAL torn-tail replay should stop deterministically at last valid record.
- WAL CRC mismatch should hard-fail replay.
- Reopen path should restore visible state from manifest + WAL.
- Repeated reopen cycles should preserve deterministic results.

### C) Temporal Semantics

- Multi-version `AS OF` read correctness.
- `VALID AT` interval filtering correctness at boundaries.
- Combined `AS OF` and `VALID AT` behavior.
- High-cardinality version chains under forced flush/compaction.
- Compaction-preserved history checks across reopen cycles.

### D) SQL Facet

- Table lifecycle and row access semantics.
- Error behavior for missing tables/invalid statements.
- `EXPLAIN` observability fields present and stable.
- Batch parser behavior:
  - `stmt;` accepted
  - `stmt1; stmt2;` sequential execution
  - semicolons inside quoted JSON payloads are preserved.
- Transactional batch behavior:
  - `BEGIN; ...; COMMIT;` persists staged writes.
  - `BEGIN; ...; ROLLBACK;` discards staged writes.
  - nested `BEGIN` rejected.
  - trailing open transaction rejected (`BEGIN;` without close).
- Metadata-only features:
  - `CREATE VIEW` definition metadata persisted and selectable via source indirection.
  - `CREATE INDEX` metadata persisted across reopen; no index-backed acceleration assertions.
  - `ALTER TABLE ... ADD COLUMN` schema metadata revision persists; no row rewrite/backfill assertions.
- Introspection and lifecycle:
  - `SHOW TABLES`, `DESCRIBE <table>`.
  - `DROP TABLE` / `DROP VIEW` / `DROP INDEX` semantics and durability across reopen.
- Scan and relational shape:
  - `SELECT doc FROM <table>`
  - `SELECT pk, doc FROM <table>`
  - `SELECT COUNT(*) FROM <table>`
  - `SELECT pk, doc FROM <left> JOIN <right> ON <left>.pk=<right>.pk`
  - `SELECT pk, count(*) FROM ... GROUP BY pk`
  - `ORDER BY` + `LIMIT` determinism.

Next syntax milestone tests (not yet implemented):
- richer joins / grouping / windows / subqueries.
- update/delete temporal semantics.

### E) Native Integration

- Rust/native deterministic output equivalence for fixture inputs.
- Native call path instrumentation confirms invocation.
- Feature-gating guarantees pure-Rust fallback works without C++.

## 4) Performance Validation Matrix

Run matrix across key knobs:

- `shard_count`: 1, 2, 4, 8
- `memtable_max_bytes`: 64 KiB, 1 MiB, 4 MiB
- `sstable_block_bytes`: 4 KiB, 16 KiB, 64 KiB
- `bloom_bits_per_key`: 8, 10, 14
- `wal_fsync_every_n_records`: 1, 16, 128

Collect per run:

- write ops/s
- read p50/p95/p99
- requested vs observed read miss ratio
- bloom miss rate
- mmap block reads
- hasher path (`rust-fnv64` or `native-demo64`)

## 5) Baseline Comparisons

When tooling is available, compare SpectraDB point-read and insert workloads against:

- SQLite (`sqlite3` CLI)
- (optional future) RocksDB harness

Comparisons should report:

- workload definition
- machine metadata
- warm-up policy
- median of N runs
- variability (min/max)

## 6) Overnight Reliability Campaign

Use `scripts/overnight_iterate.sh` with:

- `ROUNDS >= 12`
- mixed write/read workloads
- periodic reopen/checkpoint behavior

Required outcomes:

- no crashes
- no invariant violations
- no data loss across reopen
- stable p99 without unbounded degradation

## 7) Regression Policy

Any PR touching storage, WAL, temporal semantics, or native bridge must include:

- affected test updates,
- benchmark delta evidence (before/after),
- explicit note whether temporal semantics were impacted.

## 8) Next Additions for SOTA Push

- Property-based temporal testing (`proptest`) for random histories.
- Deterministic crash-point injection across WAL/manifest write boundaries.
- End-to-end comparative harness with RocksDB and LMDB adapters.
- CPU profile + alloc profile capture in CI for hotpath tracking.
- SQL workload tiers aligned to DuckDB docs categories (statement/query/aggregate/introspection), with explicit unsupported-category skips until implemented.
