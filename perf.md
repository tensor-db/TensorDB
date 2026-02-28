# TensorDB Performance Guide

## Benchmark Results

TensorDB ships with three Criterion benchmark suites. Results from a single-machine run:

| Operation | TensorDB | SQLite | sled | redb |
|-----------|-----------|--------|------|------|
| Point Read | **276 ns** | 1,080 ns | 244 ns | 573 ns |
| Point Write (fast path) | **1.9 µs** | 38.6 µs | — | — |
| Point Write (channel path) | ~161 µs | 38.6 µs | — | — |
| Batch Write (100 keys) | native | — | — | — |
| Prefix Scan (1k keys) | native | — | — | — |
| Mixed 80r/20w | native | — | — | — |

**Key takeaways:**
- Point reads are **4x faster** than SQLite thanks to direct shard bypass (no channel round-trip).
- Point writes are **20x faster** than SQLite via the lock-free fast write path with group-commit WAL.
- The channel-based write path (~161 µs) is the fallback for backpressure, subscriber activity, or when `fast_write_enabled=false`.

### Running Benchmarks

```bash
# TensorDB vs SQLite (head-to-head)
cargo bench --bench comparative

# TensorDB vs SQLite vs sled vs redb (four-way)
cargo bench --bench multi_engine

# TensorDB microbenchmarks
cargo bench --bench basic

# CLI-integrated benchmark with configurable workload
cargo run -p tensordb-cli -- --path /tmp/bench bench \
  --write-ops 100000 --read-ops 50000 --keyspace 20000 --read-miss-ratio 0.20
```

### AI Overhead Gate

CI runs an overhead regression check on every push:

```bash
./scripts/ai_overhead_gate.sh
```

Thresholds: max 90% write throughput drop, max 150% increase in p95/p99 latency when AI features are enabled vs disabled.

---

## Why Reads Are Fast (276 ns)

The read path bypasses the shard actor channel entirely. `ShardReadHandle` holds an `Arc<ShardShared>` and reads directly from shared memory:

1. Block cache check (LRU, configurable size)
2. Bloom filter probe (skip SSTable if negative)
3. Active memtable scan (read lock, microseconds)
4. Immutable memtables scan
5. SSTable level scan (L0: all files newest-first, L1+: binary search)
6. Temporal filter application

No channel send/receive, no thread context switch, no shard actor wake-up. Read locks are held only during the scan itself.

## Why Writes Are Fast (1.9 µs)

The fast write path (`FastWritePath`) eliminates the three most expensive operations in the traditional channel-based path:

| Cost | Channel Path | Fast Path |
|------|-------------|-----------|
| Channel send/receive | ~50 µs | Eliminated |
| Per-write WAL fsync | ~100 µs | Amortized via group commit |
| Shard actor wake-up | ~10 µs | Eliminated |

**How it works:**
1. Atomic `fetch_add(1)` on the shard's commit counter → `commit_ts`
2. Encode internal key and fact value
3. Write-lock memtable → insert (microseconds)
4. Enqueue pre-encoded WAL frame to `WalBatchQueue`
5. Return `commit_ts` to caller immediately

The `DurabilityThread` runs in the background, batching WAL frames across all shards into a single `fdatasync` call per interval. At 100K writes/sec with a 1ms batch interval, one fsync amortizes across ~100 writes.

**Fallback to channel path when:**
- Memtable is full (needs flush — shard actor handles compaction)
- Change-feed subscribers are active (shard actor emits events)
- `fast_write_enabled` is set to false

---

## Tuning Guide

### Storage Parameters

| Parameter | Default | Trade-off |
|-----------|---------|-----------|
| `memtable_max_bytes` | 4 MB | Larger = fewer flushes, better write amortization, more memory. Smaller = lower memory, more flush churn. |
| `sstable_block_bytes` | 16 KB | Larger = better sequential scan locality, fewer index entries. Smaller = less over-read per point lookup. |
| `sstable_max_file_bytes` | 64 MB | Max SSTable file size before splitting during compaction. |
| `bloom_bits_per_key` | 10 | Higher = lower false-positive rate (fewer unnecessary disk reads), more memory/disk. Typical range: 8–14. |
| `block_cache_bytes` | 32 MB | Larger = more hot data served from memory, fewer disk reads. Size based on your working set. |
| `index_cache_entries` | 1024 | Number of SSTable index blocks cached. Increase if you have many SSTables. |

### Write Path Parameters

| Parameter | Default | Trade-off |
|-----------|---------|-----------|
| `shard_count` | 4 | More shards = higher write concurrency. Too many = more background work and metadata overhead. Match to CPU core count. |
| `fast_write_enabled` | true | Enables lock-free fast write path (~1.9 µs). Disable for strict per-write durability (falls back to channel + immediate fsync). |
| `fast_write_wal_batch_interval_us` | 1000 | WAL group commit interval in microseconds. Lower = faster durability, more fsyncs. Higher = better throughput, larger acknowledged window. |
| `wal_fsync_every_n_records` | 128 | Channel-path fsync cadence. Lower = stronger durability, lower throughput. Higher = better throughput, larger window between fsyncs. |

### Compaction Parameters

| Parameter | Default | Trade-off |
|-----------|---------|-----------|
| `compaction_l0_threshold` | 8 | Number of L0 SSTables before triggering compaction. Lower = less read amplification, more compaction work. |
| `compaction_l1_target_bytes` | 10 MB | L1 size target. Larger = fewer L0→L1 compactions, larger files. |
| `compaction_size_ratio` | 10 | Each level is this many times larger than the previous. Standard LSM ratio. |
| `compaction_max_levels` | 7 | Maximum levels (L0 through L6). More levels = better space amplification for large datasets. |

### AI Parameters

| Parameter | Default | Trade-off |
|-----------|---------|-----------|
| `ai_auto_insights` | false | Enable background insight synthesis from change feeds. Adds background CPU/IO work. |
| `ai_batch_window_ms` | 20 | Lower = fresher insights, more processing overhead. Higher = better batching efficiency, staler insights. |
| `ai_batch_max_events` | 16 | Lower = smaller batch size, less burst cost. Higher = better amortization of synthesis. |
| `ai_inline_risk_assessment` | false | Enables synchronous risk scoring on every write. Adds latency to the write path. |
| `ai_annotate_reads` | false | Annotate read results with AI metadata. Adds overhead to reads. |
| `ai_compaction_advisor` | false | AI-driven compaction scheduling recommendations. |
| `ai_cache_advisor` | false | AI-driven cache size tuning based on access patterns. |
| `ai_access_stats_size` | 1024 | Ring buffer size for hot-key tracking. Larger = more accurate hot-key detection, more memory. |

---

## Performance Characteristics

### Write Profile

- **Steady state:** Fast write path dominates. Lock-free atomic increment + memtable insert + async WAL enqueue.
- **Memtable flush:** When memtable exceeds `memtable_max_bytes`, it's frozen and flushed to an L0 SSTable. During flush, the fast path falls back to the channel path briefly.
- **Compaction pressure:** When L0 files accumulate beyond `compaction_l0_threshold`, compaction merges them into L1. Write latency can spike during heavy compaction.
- **Group commit:** The `DurabilityThread` batches WAL writes. Durability is guaranteed within one batch interval, not per-write.

### Read Profile

- **Cache-warm reads:** Block cache hit → ~100 ns.
- **Bloom-filtered misses:** Bloom probe returns negative → skip SSTable entirely. Cost: ~50 ns per SSTable.
- **Memtable reads:** Active memtable scan with read lock → ~200–400 ns.
- **Cold SSTable reads:** mmap page fault + block decompression (LZ4) + scan → 1–10 µs depending on block size and data locality.
- **Temporal filtering:** `AS OF` and `VALID AT` predicates add version-filtering cost proportional to the number of versions per key.

### SQL Query Profile

- **Point lookups via SQL:** Parser + planner + executor overhead adds ~10–50 µs on top of raw key lookup.
- **Full table scans:** Proportional to row count. Vectorized execution engine processes rows in batches of 1024 for analytical queries.
- **Joins:** Hash joins for equi-joins (build hash table on smaller side, probe with larger). Nested loop joins for non-equi conditions.
- **Aggregates:** Vectorized hash aggregate for GROUP BY. Window functions computed after grouping, before LIMIT.

---

## Profiling

### Built-in Diagnostics

```sql
-- Query execution plan with cost estimates
EXPLAIN SELECT * FROM orders WHERE customer_id = 42;

-- Plan with actual execution metrics
EXPLAIN ANALYZE SELECT * FROM orders ORDER BY created_at DESC LIMIT 10;
-- Shows: execution_time_us, rows_returned, read_ops, write_ops, bloom_negatives, plan_cost

-- Table statistics (used by cost-based planner)
ANALYZE orders;

-- AI insights for a key
EXPLAIN AI 'orders/ord-123';
```

### CLI Benchmark Output

The CLI `bench` command reports:
- Write throughput (ops/s)
- Read p50/p95/p99 latency (µs)
- Bloom miss rate
- mmap block read count
- Hasher path (rust/native)
- AI runtime counters (when `--ai-auto-insights` is enabled)

### External Profiling

```bash
# CPU profiling with perf
perf record cargo bench --bench comparative
perf report

# Flamegraph
cargo install flamegraph
cargo flamegraph --bench comparative

# Memory profiling with heaptrack
heaptrack cargo run -p tensordb-cli -- --path /tmp/bench bench --write-ops 100000
```

---

## Optimization Roadmap

**Near-term:**
- Index scan execution for `WHERE pk = ?` and `WHERE indexed_col = ?`
- Parallel shard execution for scans and aggregates
- Expression compilation for hot WHERE predicates
- Predicate pushdown to SSTable block level

**Medium-term:**
- Pipeline execution for vectorized operators (fuse without materialization)
- Morsel-driven parallelism for table scans
- External merge sort for large ORDER BY
- Adaptive execution switching between vectorized (analytics) and row-based (OLTP)

**Long-term:**
- Columnar SSTable format for analytics workloads
- SIMD string operations (LIKE, SUBSTR, UPPER/LOWER)
- Late materialization (keep column references until final projection)
- Compression policies per compaction level (LZ4 for L0–L2, Zstd for L3+)
