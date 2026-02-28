# Epoch-Ordered Append-Only Concurrency (EOAC) Design

**Date:** 2026-02-28
**Status:** Approved
**Scope:** Close all 5 critical gaps vs Oracle-class databases

## Core Insight

TensorDB is append-only with bitemporal support. A single **global epoch counter** (`AtomicU64`) unifies transactions, MVCC, index acceleration, concurrency, and point-in-time recovery into one zero-overhead mechanism. No undo logs, no version chains, no vacuum.

---

## 1. Global Epoch Infrastructure

### Data Structures

```rust
// In Database
pub struct Database {
    global_epoch: Arc<AtomicU64>,  // monotonically increasing
    // ... existing fields
}

// Every record gains an epoch
pub struct InternalKey {
    user_key: Vec<u8>,
    commit_ts: u64,      // existing per-shard timestamp
    commit_epoch: u64,   // NEW: global epoch at commit time
}
```

### Epoch Advancement

- `global_epoch` starts at 1 on database open
- Advanced atomically at transaction commit (`fetch_add(1, SeqCst)`)
- Non-transactional writes (single PUT) use `load(Acquire)` for current epoch
- Epoch is encoded into WAL records and memtable entries

### Visibility Rule

A record is visible to reader at `read_epoch` if:
```
record.commit_epoch <= read_epoch
```

For non-transactional readers (no explicit BEGIN), `read_epoch = global_epoch.load(Acquire)`.

---

## 2. Zero-Cost Intent Transactions

### Transaction Lifecycle

```
BEGIN     → read_epoch = global_epoch.load(Acquire)
            intent_buffer = Vec::new()

WRITE     → intent_buffer.push((key, value, shard_id))
            // No locks, no WAL, no memtable access. ~50ns.

READ      → check intent_buffer (reverse scan for latest)
            then query storage with epoch filter: commit_epoch <= read_epoch

COMMIT    → commit_epoch = global_epoch.fetch_add(1, SeqCst)
            for each shard touched:
              write_lock(shard.memtable)
              bulk_insert(intents, commit_epoch)
              write WAL records with txn_id + commit_epoch
              write TXN_COMMIT marker
              unlock
            // ~1.5µs per write in the intent buffer

ROLLBACK  → drop(intent_buffer)
            // Literally free. Zero I/O.
```

### Cross-Shard Atomicity

All shards in a transaction share the same `commit_epoch`. Readers at epoch E either see ALL writes from a transaction with `commit_epoch <= E` or NONE.

### Crash Safety

- WAL records carry `txn_id` and `commit_epoch`
- `TXN_COMMIT(txn_id, commit_epoch, shard_count)` marker written to each shard WAL
- On crash recovery: scan WAL for TXN_COMMIT markers. Records with matching txn_id are committed; others are skipped.
- No undo log needed — uncommitted data never reached durable storage outside WAL, and WAL replay skips it.

### Nested Transactions / Savepoints

- `SAVEPOINT name` → record current intent_buffer length as savepoint
- `ROLLBACK TO name` → truncate intent_buffer to savepoint length
- `RELEASE name` → remove savepoint marker (intents stay in buffer)

### Wire Format

```
WAL Record (extended):
  [magic: 4B][length: 4B][crc32: 4B]
  [epoch: 8B][txn_id: 8B][payload...]

TXN_COMMIT Record:
  [magic: 4B = "TXNC"][length: 4B][crc32: 4B]
  [epoch: 8B][txn_id: 8B][shard_bitmap: 8B][intent_count: 4B]
```

---

## 3. Adaptive Pipeline Query Engine

### N-Way JOIN Support

**Parser Change:** Allow chained JOINs:
```sql
SELECT * FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id WHERE a.status = 'active'
```

**Join Graph Construction:**
```rust
struct JoinGraph {
    tables: Vec<TableRef>,         // [a, b, c]
    edges: Vec<JoinEdge>,          // [(a,b,a.id=b.a_id), (b,c,b.id=c.b_id)]
    predicates: Vec<Predicate>,    // [a.status = 'active']
}
```

**Join Reordering (Greedy Heuristic):**
1. Start with the smallest table (by estimated rows from ANALYZE)
2. Greedily add the table with the cheapest join to the current result set
3. Cost = algorithm_cost(join_type, left_rows, right_rows, has_index)

**Algorithm Selection Per Edge:**
| Condition | Algorithm | Cost |
|---|---|---|
| Equi-join + inner table indexed | Index Nested Loop | O(n * log m) |
| Equi-join + no index | Hash Join | O(n + m) |
| Non-equi / theta join | Nested Loop | O(n * m) |
| Cross join | Cartesian | O(n * m) |

### Predicate Pushdown

Before execution, push WHERE predicates to the earliest possible scan:
```
Original:  Scan(a) → Join(b) → Filter(a.status = 'active')
Optimized: Scan(a, WHERE status='active') → Join(b)
```

Rules:
- Single-table predicates push to that table's scan
- Join predicates stay at the join operator
- Post-join predicates (referencing both tables) cannot be pushed

### Pipeline Execution (Volcano-style with Fused Operators)

```rust
trait PipelineOp {
    fn next(&mut self) -> Option<Row>;
}

// Fused scan: combines scan + filter + project + limit
struct FusedScan {
    scanner: Box<dyn Iterator<Item = Row>>,
    predicate: Option<CompiledPredicate>,
    projection: Vec<usize>,  // column indices
    limit: Option<usize>,
    emitted: usize,
}

// Hash join operator
struct HashJoinOp {
    build_side: HashMap<Vec<u8>, Vec<Row>>,
    probe_side: Box<dyn PipelineOp>,
    join_key_left: usize,
    join_key_right: usize,
}
```

### Table Statistics (ANALYZE Integration)

`ANALYZE` already exists but results aren't fed to planner. Wire it:
```rust
struct TableStats {
    row_count: u64,
    column_stats: HashMap<String, ColumnStats>,
}

struct ColumnStats {
    distinct_count: u64,
    null_count: u64,
    min_value: Option<SqlValue>,
    max_value: Option<SqlValue>,
    avg_size_bytes: u64,
}
```

Store in `__meta/stats/<table>` as JSON. Planner reads at plan time.

---

## 4. Index-Accelerated Execution

### Predicate-to-Index Matching

At plan time, the planner examines WHERE predicates and available indexes:

```rust
fn match_index(predicate: &Expr, indexes: &[IndexMeta]) -> Option<IndexScanPlan> {
    match predicate {
        // col = literal → exact index lookup
        Expr::BinOp(Eq, Expr::Column(col), Expr::Literal(val)) => {
            find_index_for_column(col, indexes)
                .map(|idx| IndexScanPlan::Exact(idx, val))
        }
        // col > literal → prefix range scan
        Expr::BinOp(Gt, Expr::Column(col), Expr::Literal(val)) => {
            find_index_for_column(col, indexes)
                .map(|idx| IndexScanPlan::Range(idx, val, None))
        }
        // col IN (v1, v2, ...) → multi-point lookup
        Expr::In(Expr::Column(col), values) => {
            find_index_for_column(col, indexes)
                .map(|idx| IndexScanPlan::MultiPoint(idx, values))
        }
        // AND → try to use composite index
        Expr::BinOp(And, left, right) => {
            try_composite_index_match(left, right, indexes)
        }
        _ => None,
    }
}
```

### Index Scan Execution

```rust
fn execute_index_scan(
    db: &Database,
    table: &str,
    index: &IndexMeta,
    scan_type: IndexScanPlan,
    epoch: u64,
) -> Vec<Row> {
    let pk_list = match scan_type {
        IndexScanPlan::Exact(idx, val) => {
            // Prefix scan: __idx/<table>/<index>/<encoded_val>/
            let prefix = format!("__idx/{}/{}/{}/", table, idx.name, encode_value(&val));
            db.scan_prefix(prefix.as_bytes())
                .filter(|r| r.commit_epoch <= epoch && r.doc != INDEX_TOMBSTONE)
                .map(|r| extract_pk_from_index_key(&r.user_key))
                .collect()
        }
        IndexScanPlan::Range(idx, start, end) => {
            // Range scan on encoded values
            // ... prefix scan with bounds
        }
        IndexScanPlan::MultiPoint(idx, values) => {
            // Multiple exact lookups, deduplicate PKs
            values.iter()
                .flat_map(|v| execute_index_scan(db, table, &idx, Exact(idx, v), epoch))
                .collect()
        }
    };

    // Fetch actual rows by PK (point lookups — very fast)
    pk_list.iter()
        .filter_map(|pk| db.get(row_key(table, pk)))
        .collect()
}
```

### Index + Bloom Filter Fusion

For predicates on unindexed columns, use existing temporal bloom filters as pre-filter:
1. Check bloom filter — if negative, skip entire SST block
2. If positive, scan block and evaluate predicate

For indexed columns, skip bloom filter entirely (index is more precise).

### Index Nested Loop Join

When the inner table has an index on the join column:
```
for each row in outer_table:
    key = row[join_column]
    inner_rows = index_lookup(inner_table, index, key)
    for each inner_row in inner_rows:
        emit(merge(row, inner_row))
```

Cost: O(n * log m) where n = outer rows, m = inner rows (index lookup is O(log m)).

---

## 5. Epoch-Based Recovery

### WAL Format (Extended)

```
Standard Record:
  [magic: 4B "SWAL"][total_len: 4B][crc32: 4B]
  [epoch: 8B][txn_id: 8B][user_key_len: 4B][user_key][commit_ts: 8B][doc]

Batch Record:
  [magic: 4B "SWBC"][total_len: 4B][crc32: 4B]
  [epoch: 8B][txn_id: 8B][count: 4B]
  [entry_len: 4B][user_key_len: 4B][user_key][commit_ts: 8B][doc] * count

Commit Marker:
  [magic: 4B "TXNC"][total_len: 4B][crc32: 4B]
  [epoch: 8B][txn_id: 8B][shard_bitmap: 8B][intent_count: 4B]

Checkpoint Marker:
  [magic: 4B "CKPT"][total_len: 4B][crc32: 4B]
  [epoch: 8B][manifest_snapshot: variable]
```

### Crash Recovery Algorithm

```
1. For each shard WAL:
   a. Scan all records, collect txn_ids and their records
   b. Build committed_set: txn_ids that have a TXN_COMMIT marker
   c. For non-transactional records (txn_id = 0): always apply
   d. For transactional records: apply only if txn_id in committed_set
   e. Insert applied records into fresh memtable

2. Update global_epoch to max(epoch seen in any WAL) + 1
3. Resume normal operation
```

### Point-in-Time Recovery (PITR)

```sql
-- Restore database to epoch 12345
RESTORE DATABASE TO EPOCH 12345;

-- Or by timestamp (maps to nearest epoch via MANIFEST)
RESTORE DATABASE TO TIMESTAMP '2026-02-28T10:30:00Z';
```

**Algorithm:**
1. Start from most recent checkpoint before target epoch
2. Replay WAL from checkpoint epoch to target epoch
3. Skip records with `epoch > target_epoch`
4. Skip uncommitted transactions (no TXN_COMMIT marker with epoch <= target)

### Incremental Backup

```sql
BACKUP DATABASE TO '/path' INCREMENTAL SINCE EPOCH 10000;
```

**Algorithm:**
1. Record current epoch as `backup_epoch`
2. Identify SSTs created since last backup epoch (from MANIFEST)
3. Copy only those SSTs + WAL segments since last backup
4. Write `backup_metadata.json` with `{from_epoch, to_epoch, sst_list}`

### Epoch Checkpoints

Periodically (configurable):
1. Flush all memtables
2. Write CKPT marker to all shard WALs
3. Record checkpoint epoch in MANIFEST
4. Truncate WAL before checkpoint (safe — all data in SSTs)

---

## 6. Concurrency Model

### Epoch-Based Lock-Free Readers

```rust
impl Database {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            read_epoch: self.global_epoch.load(Acquire),
        }
    }
}

impl Snapshot {
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        // Read from memtable + SSTs, filter by commit_epoch <= read_epoch
    }

    pub fn scan_prefix(&self, prefix: &[u8]) -> Vec<Record> {
        // Same scan but filtered by epoch
    }
}
```

- Readers: pin read_epoch at query start. See consistent snapshot. No locks.
- Writers: use existing fast write path + intent buffer.
- Cross-shard: shared global_epoch ensures all shards see same snapshot boundary.

### Garbage Collection

Records where `commit_epoch < (min_active_read_epoch - retention_window)` are eligible for compaction. During compaction, only the latest version (by commit_epoch) of each key is kept (unless bitemporal retention requires older versions).

---

## Implementation Order

| Phase | Component | Estimated Complexity |
|---|---|---|
| A | Global epoch infrastructure + epoch in WAL | Medium |
| B | Intent-based transactions (BEGIN/COMMIT/ROLLBACK) | Medium |
| C | Index-accelerated scans (predicate matching) | Medium |
| D | N-way JOIN parsing + join graph | Medium |
| E | Join reordering + algorithm selection | Medium |
| F | Predicate pushdown | Low |
| G | Crash recovery with epoch filtering | Medium |
| H | PITR + incremental backup | Medium |
| I | Savepoints | Low |
| J | Table statistics wired to planner | Low |

Phases A-C are foundational. D-F transform the query engine. G-H complete recovery. I-J are polish.

---

## What This Beats

| Feature | SQLite | PostgreSQL | Oracle | TensorDB EOAC |
|---|---|---|---|---|
| Transaction begin | Write lock | Snapshot alloc + txn_id | Undo segment alloc | AtomicU64 load (~5ns) |
| Rollback cost | Undo journal replay | Mark aborted + cleanup | Undo log replay | Drop Vec (~0ns) |
| Read isolation | Shared lock or WAL snapshot | MVCC version chain walk | Undo segment read | Epoch compare (~1ns) |
| Cross-shard atomicity | N/A (single file) | 2PC + prepared txns | Distributed lock mgr | Single epoch counter |
| PITR | WAL replay from backup | Continuous archiving | RMAN + archived redo | Epoch-filtered WAL replay |
| Index selection | Automatic | Cost-based optimizer | Cost-based + hints | Predicate-to-index matching |
