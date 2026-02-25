# SpectraDB Performance Notes

## 1) Tuning Knobs

- `wal_fsync_every_n_records`
  - Lower = stronger durability cadence, lower throughput.
  - Higher = better throughput, larger acknowledged window between fsyncs.

- `memtable_max_bytes`
  - Larger = fewer flushes and better write amortization.
  - Smaller = lower memory footprint, more flush churn.

- `sstable_block_bytes`
  - Larger = fewer index entries, better sequential scan locality.
  - Smaller = less over-read per point lookup.

- `bloom_bits_per_key`
  - Higher = lower false-positive rate, more memory/disk for bloom block.
  - Typical range: 8-14 bits/key.

- `shard_count`
  - More shards = higher write concurrency potential.
  - Too many shards can amplify background work and metadata overhead.

## 2) Expected MVP Profile

Writes:
- sequential WAL appends dominate steady-state ingest,
- memtable flush creates immutable SSTables,
- compaction cost rises when L0 count grows.

Reads:
- point gets are bloom-filtered first,
- mmap SSTable scans avoid explicit read syscalls for hot pages,
- latency tail depends on compaction pressure and cache warmth.
- current SQL executor supports point-key reads, prefix scans, pk-equality hash-join skeleton, and basic grouping skeleton (`pk,count(*) GROUP BY pk`), so scan/join cost now contributes materially to SQL latency.

Temporal queries:
- `AS OF` and `VALID AT` add version filtering cost,
- bounded key cardinality and bloom-index pruning keep this manageable.

DuckDB-category parity note:
- Current SQL perf numbers cover point lookup plus early scan/aggregate/order subsets.
- Missing rich relational operators (general joins/grouping/windows) still create a large gap vs DuckDB query workloads.

## 3) Benchmark Surface

CLI benchmark (`spectradb-cli bench`) reports:
- write throughput (`ops/s`)
- read p50/p95/p99 latency
- requested and observed read miss ratio
- bloom miss rate
- fsync cadence
- active hasher path (rust/native)
- mmap block read count

### Sample run (single machine, quick sanity only)

Command:

```bash
cargo run -q -p spectradb-cli -- --path /tmp/sdb_bench_rust --memtable-max-bytes 65536 bench --write-ops 20000 --read-ops 10000 --keyspace 5000 --read-miss-ratio 0.20
cargo run -q -p spectradb-cli --features native -- --path /tmp/sdb_bench_native --memtable-max-bytes 65536 bench --write-ops 20000 --read-ops 10000 --keyspace 5000 --read-miss-ratio 0.20
```

Observed output snapshot:
- Rust hasher: `write_ops_per_sec=4593.57`, `read_p50_us=538`, `read_p95_us=887`, `read_p99_us=1032`, `mmap_reads=40000`, `hasher=rust-fnv64-mix`
- Native hasher: `write_ops_per_sec=3996.17`, `read_p50_us=469`, `read_p95_us=853`, `read_p99_us=987`, `mmap_reads=40000`, `hasher=native-demo64`

Interpretation:
- Native path is correctly invoked (`hasher=native-demo64`).
- This demo hasher is a correctness/integration target, not yet a throughput winner.
- Next native iterations should focus on kernels with larger per-call compute payload (SIMD bloom probes, checksum/compression).

## 4) Performance Validation Plan

Short-loop checks:
- ingest throughput sweep by fsync cadence.
- read latency sweep by block size and bloom bits/key.
- native vs rust hasher delta.

Nightly/soak checks:
- mixed workload burn-in with periodic reopen/recovery.
- stable p99 regression budgets.
- disk amplification and compaction debt tracking.

## 5) Prioritized Optimization Roadmap (Efficiency, Speed, Portability)

1. Query-path efficiency:
- optimize current scan operators and preserve point-read fast path while adding predictable bounded-memory costs.
- next: grouping and joins with spill-aware execution.

2. Read amplification and cache locality:
- add restart points + prefix compression in SSTable blocks.
- introduce block/index caches with explicit hit-rate and memory budgets.

3. Compaction scalability:
- evolve from simple L0 flow to multi-level, overlap-aware compaction.
- reduce write amplification and compaction debt under sustained ingest.

4. Portability-preserving acceleration:
- keep pure-Rust path as default reference behavior.
- add optional native/SIMD kernels where per-call compute is high enough to beat FFI overhead.

5. I/O path portability:
- retain mmap + sync write baseline across platforms.
- optionally add platform-specific async write paths behind feature flags without changing correctness semantics.

6. Reproducible comparative harness:
- expand benchmark matrix to include both current point-lookups and future scan/aggregate/operator categories for apples-to-apples DuckDB/SQLite comparisons.
