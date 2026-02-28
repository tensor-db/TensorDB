# TensorDB — Architecture & Design

## What TensorDB Does

TensorDB is an embedded database where **every change is preserved and queryable**. When you write data, the old version isn't overwritten — it becomes part of the history. You can ask "what did this record look like last Tuesday?" or "what was the account balance we believed was correct on March 1st?" and get exact answers using standard SQL.

This matters because most databases only keep the latest state. If you need audit trails, compliance records, time-travel debugging, or undo capabilities, you typically build versioning logic in your application. TensorDB handles this at the storage level, so your application code stays simple.

**The core idea:** every write is an immutable fact with two timestamps — *when it was recorded* (system time) and *when it was true* (business time). This separation is called bitemporal modeling, and it lets you answer two distinct questions:

- **"What did we know at time T?"** — System time query (`AS OF`)
- **"What was true at time T?"** — Business time query (`VALID AT`)

TensorDB wraps this in a full SQL engine with joins, aggregates, window functions, full-text search, vector search, time-series analytics, and an embedded AI runtime — all in a single Rust library you can embed in any application.

---

## Data Model

Every record stored in TensorDB is a **fact** with five components:

| Field | Type | Purpose |
|-------|------|---------|
| `user_key` | bytes | The logical identifier (e.g., `"accounts/alice"`) |
| `commit_ts` | u64 | System timestamp — when this fact was recorded (monotonic per shard) |
| `valid_from` | u64 | Business timestamp — when this fact became true |
| `valid_to` | u64 | Business timestamp — when this fact stopped being true |
| `payload` | bytes | The actual data (JSON document or typed columnar encoding) |

Facts are never modified or deleted at the storage level. An "update" creates a new fact with a higher `commit_ts`. A "delete" creates a tombstone fact. The complete history is always available.

### Internal Key Encoding

Facts are stored with a composite key that enables efficient temporal lookups:

```
internal_key = user_key || 0x00 || commit_ts (big-endian u64) || kind (u8)
```

The `0x00` separator allows prefix scans to find all versions of a key. Big-endian encoding of `commit_ts` ensures newer versions sort after older ones within the same prefix. The `kind` byte distinguishes record types (currently `0x00` = PUT).

### Temporal Semantics

**MVCC (system time):** Each shard maintains a monotonic commit counter. Every write gets the next counter value as its `commit_ts`. Reading `AS OF T` returns the newest version where `commit_ts <= T`.

**Bitemporal (business time):** The valid interval is half-open: `[valid_from, valid_to)`. Reading `VALID AT V` returns versions where `valid_from <= V < valid_to`.

**Combined:** Both filters compose. The final result is the version with the highest `commit_ts` that passes both the system-time and business-time predicates.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                             │
│  Rust API │ Interactive CLI │ Python (PyO3) │ Node.js (napi-rs) │
│                    │ pgwire Server (PostgreSQL Protocol)         │
└─────────────┬───────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────┐
│                       SQL Engine                                │
│  Parser ──► Cost-Based Planner ──► Executor ──► Vectorized Eng  │
│  50+ functions │ CTEs │ Joins │ Windows │ Temporal Clauses       │
└─────────────┬───────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────┐
│                      Facet Layer                                │
│  Relational │ Full-Text Search │ Time-Series │ Vector │ Events  │
└─────────────┬───────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────┐
│                     Shard Engine                                │
│  Fast Write Path (lock-free, ~1.9µs)  ──► Shard 0..N           │
│  Channel Path (fallback, ~161µs)      ──► Direct Read (276ns)  │
│                                                                 │
│  Change Feeds │ Durable Cursors │ Consumer Groups               │
└─────────────┬───────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────┐
│                    Storage Engine                               │
│  Group-Commit WAL ──► Memtable ──► SSTable (LZ4, bloom, mmap)  │
│  L0 ──► L1 ──► ... ──► L6 (multi-level compaction)             │
│  Block Cache (LRU) │ Index Cache │ Zone Maps │ Fractal Index    │
└─────────────────────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────┐
│                     AI Runtime                                  │
│  Insight Synthesis │ Risk Scoring │ Advisors │ ML Pipeline      │
│  Inference Engine │ RAG Pipeline │ Feature Store                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Write Path

Writes take two paths depending on conditions. The fast path handles the common case; the channel path handles edge cases.

### Fast Write Path (~1.9 µs per write)

This is the default. It bypasses the shard actor's crossbeam channel entirely, writing directly to shared memory structures using atomic operations.

```
Client
  │
  ▼
Database::put(key, doc, valid_from, valid_to)
  │
  ├── hash(key) % shard_count ──► select shard
  │
  ▼
FastWritePath::try_fast_put()
  │
  ├── 1. Check: fast_write enabled?              (AtomicBool, Relaxed)
  ├── 2. Check: any change-feed subscribers?      (AtomicBool, Relaxed)
  ├── 3. Check: memtable below size threshold?    (read lock, check approx_bytes)
  │       └── If any check fails ──► fall back to channel path
  │
  ├── 4. commit_ts = commit_counter.fetch_add(1)  (AtomicU64, AcqRel)
  ├── 5. Encode internal_key and fact_value
  ├── 6. Write-lock memtable ──► insert entry
  └── 7. Enqueue WAL frame to WalBatchQueue
```

**Why it's fast:** No channel send/receive, no shard actor wake-up, no per-write fsync. The commit counter is a single atomic increment. The memtable write lock is held only for the insert (microseconds). WAL durability is deferred to the group-commit thread.

**Fallback conditions:**
- Memtable is full (needs flush — shard actor handles this)
- Change-feed subscribers are active (shard actor emits events)
- `fast_write_enabled` is false in config

### Channel Write Path (~161 µs per write)

The fallback path sends a `ShardCommand::Put` message through a crossbeam channel to the shard actor thread. The actor performs WAL append, memtable insert, change-feed notification, and responds through a bounded(1) reply channel.

This path is used when the fast path can't be taken, and it's the path that existed before the fast write optimization.

### Group-Commit WAL

Both paths feed into the same durability mechanism. Instead of one `fdatasync` per write, a dedicated `DurabilityThread` batches WAL records:

```
FastWritePath ──► WalBatchQueue (per-shard Mutex<Vec<frame>>)
                       │
                       ▼
              DurabilityThread (background)
                       │
                  ┌────┴────┐
                  │  Wait   │  condvar OR batch_interval_us timeout
                  └────┬────┘
                       │
            For each shard with pending frames:
              1. Drain queue
              2. Concatenate all frames
              3. write_all() to shard WAL file
              4. fdatasync()
```

**Configuration:** `fast_write_wal_batch_interval_us` (default: 1000 µs). At 100K writes/sec, one 100 µs fsync amortizes to ~1 µs per write.

**Shutdown sequence:** Set shutdown flag → signal condvar → join thread → final flush of remaining records.

---

## Read Path

Reads bypass the shard actor channel entirely. The `ShardReadHandle` accesses shared state directly through `Arc<ShardShared>`, achieving 276 ns point reads.

```
Client
  │
  ▼
Database::get(key, as_of, valid_at)
  │
  ├── hash(key) % shard_count ──► select ShardReadHandle
  │
  ▼
ShardReadHandle::get_visible(key, as_of, valid_at)
  │
  ├── 1. Block cache lookup
  ├── 2. Bloom filter probe ──► if negative, skip SSTable
  ├── 3. Active memtable scan (read lock)
  ├── 4. Immutable memtables scan (read lock)
  ├── 5. SSTable level scan:
  │       L0: search all files, newest first
  │       L1+: binary search for overlapping file per level
  ├── 6. Temporal filter: commit_ts <= as_of AND valid_from <= valid_at < valid_to
  └── 7. Return newest matching version
```

**Why it's fast:** No channel round-trip, no thread context switch. Read locks on the memtable are held only during the scan. SSTable reads use memory-mapped I/O (`memmap2`), so the OS page cache serves hot data.

---

## Storage Engine

### WAL (Write-Ahead Log)

Every write is durably logged before it's acknowledged. The WAL uses a simple frame format:

```
┌──────────┬──────────┬──────────┬───────────────┐
│ MAGIC    │ Length   │ CRC32    │ Payload       │
│ (4 LE)   │ (4 LE)  │ (4 LE)  │ (variable)    │
└──────────┴──────────┴──────────┴───────────────┘

Payload = encode_bytes(internal_key)
        + encode_bytes(fact_value)
        + flags (1 byte)
        + [optional: schema_version varint]
```

**Recovery:** On startup, the WAL is replayed from the beginning. CRC mismatches or torn tails cause a deterministic stop at the last valid frame — no data corruption, just a clean truncation point.

### Memtable

An ordered map (`BTreeMap`) keyed by internal key. Serves as the write-optimized staging area before data reaches SSTables. When `memtable_max_bytes` is exceeded, the active memtable is frozen, pushed to the immutable queue, and a new empty memtable takes its place.

### SSTable Format (V2)

SSTables are the on-disk format for persisted data. V2 adds LZ4 block compression:

```
┌─────────────────────────────────────────────────┐
│ Header: MAGIC(4) + VERSION(4) + block_size(4)   │  12 bytes
├─────────────────────────────────────────────────┤
│ Data Blocks (repeated):                         │
│   on_disk_len(4 LE) + LZ4-compressed block      │
│   Each entry: varint(klen) + varint(vlen)       │
│                + key bytes + value bytes         │
├─────────────────────────────────────────────────┤
│ Index Block:                                    │
│   [last_key + offset(8 LE) + len(4 LE)] × N    │
├─────────────────────────────────────────────────┤
│ Bloom Filter Block:                             │
│   Serialized bloom filter bitset                │
├─────────────────────────────────────────────────┤
│ Footer: index_offset(8) + bloom_offset(8)       │
│         + FOOTER_MAGIC(4)                        │  20 bytes
└─────────────────────────────────────────────────┘
```

**Lookup path:**
1. Bloom filter probe — if negative, the key definitely isn't in this SSTable
2. Binary search the index block for the candidate data block
3. Decompress the block (`lz4_flex::decompress_size_prepended`)
4. Scan entries, apply temporal visibility filters
5. Return the matching value (if any)

**Constants:** `SST_MAGIC = 0x53535442` ("SSTB"), `SST_VERSION_V2 = 2`, `FOOTER_MAGIC = 0x53534654` ("SSFT")

**V1 compatibility:** The reader detects the version from the header and handles uncompressed V1 blocks transparently.

### Multi-Level Compaction

SSTables are organized into levels (L0 through L6). When L0 accumulates too many files (`compaction_l0_threshold`, default 8), compaction merges them into L1. Each subsequent level is `compaction_size_ratio` times larger (default 10x).

```
L0:  [SST] [SST] [SST] [SST]  ──► compact when count >= threshold
L1:  [     SSTable (10 MB)    ]  ──► compact when size exceeds target
L2:  [          SSTable (100 MB)          ]
L3:  [                SSTable (1 GB)                  ]
...
L6:  [                          SSTable                            ]
```

**Key invariant:** All temporal versions are preserved through compaction. Compaction deduplicates by keeping the latest version per logical key at the same commit timestamp, but it never drops older temporal versions. The complete history survives.

**Compaction algorithm:**
1. Collect all entries from source SSTable readers
2. Sort by internal key (deterministic ordering via commit_ts)
3. Deduplicate — keep latest per logical key
4. Build new SSTable with `build_sstable()`
5. Atomically replace manifest

### Block & Index Cache

Two LRU caches reduce disk I/O for hot data:

- **Block cache** (`block_cache_bytes`, default 32 MB) — Caches decompressed SSTable data blocks. Keyed by (file_id, block_offset).
- **Index cache** (`index_cache_entries`, default 1024) — Caches SSTable index blocks. Avoids re-reading the index on repeated access to the same file.

### Zone Maps

Per-block metadata that enables block skipping during scans:

- Column min/max values per block
- Null count per column
- HyperLogLog sketch for approximate distinct count (256 registers, mergeable across blocks)

### Fractal Index

An adaptive secondary index that switches strategies based on access patterns:

- **HashOnly** — When >85% of accesses are point lookups
- **SortedOnly** — When >85% of accesses are range scans
- **Both** — When access patterns are mixed (default for <10 observations)

The `IndexRouter` tracks per-prefix access statistics and recommends strategies. On a point lookup that hits the sorted index, the entry is auto-promoted to the hash index.

### Manifest

The manifest tracks which SSTable files belong to which levels, plus the next file ID for each shard. It uses JSON format for readability and deterministic replay.

**Atomic update protocol:**
1. Write manifest to a temp file
2. `fsync` the temp file
3. Rename temp → `MANIFEST`
4. `fsync` the parent directory

This ensures the manifest is never in a partially-written state, even after a crash.

---

## SQL Engine

### Parser

The parser (`sql/parser.rs`, ~2,700 lines) converts SQL text into an AST. It handles:

**DDL:** `CREATE TABLE` (typed and JSON), `CREATE TABLE AS SELECT`, `CREATE VIEW`, `CREATE INDEX`, `CREATE FULLTEXT INDEX`, `CREATE TIMESERIES TABLE`, `ALTER TABLE ADD COLUMN`, `DROP TABLE/VIEW/INDEX`

**DML:** `INSERT` (with optional `RETURNING`), `UPDATE ... SET ... WHERE`, `DELETE ... WHERE`

**Queries:** `SELECT` with `FROM`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT`, subqueries, CTEs (`WITH ... AS`), `UNION`/`INTERSECT`/`EXCEPT`, all join types

**Temporal clauses (SQL:2011):**
- `FOR SYSTEM_TIME AS OF <ts>`
- `FOR SYSTEM_TIME FROM <t1> TO <t2>`
- `FOR SYSTEM_TIME BETWEEN <t1> AND <t2>`
- `FOR SYSTEM_TIME ALL`
- `FOR APPLICATION_TIME AS OF <ts>`
- `FOR APPLICATION_TIME FROM <t1> TO <t2>`
- `FOR APPLICATION_TIME BETWEEN <t1> AND <t2>`

**Table functions:** `read_csv('path')`, `read_json('path')`, `read_parquet('path')`

**Data transfer:** `COPY table TO/FROM 'path' [FORMAT CSV|JSON|PARQUET]`

**Transactions:** `BEGIN`, `COMMIT`, `ROLLBACK`

**Introspection:** `SHOW TABLES`, `DESCRIBE table`, `EXPLAIN`, `EXPLAIN ANALYZE`, `EXPLAIN AI`, `ANALYZE table`

**Prepared statements:** `$1`, `$2`, ... positional parameter placeholders

### Cost-Based Planner

The planner (`sql/planner.rs`) builds a `PlanNode` tree and estimates execution cost:

```
PlanNode variants:
  PointLookup  { table, key }              cost: 1.0
  PrefixScan   { table, prefix }           cost: varies
  FullScan     { table, estimated_rows }   cost: rows × 1.0
  Filter       { input, predicate }        cost: input × selectivity
  HashJoin     { left, right, keys }       cost: build(1.5/row) + probe(1.0/row)
  NestedLoopJoin { left, right, cond }     cost: left × right × 0.5
  Aggregate    { input, group_by, aggs }   cost: input + output
  Sort         { input, order_by }         cost: input + n × log(n) × 2.0
  Limit        { input, limit, offset }    cost: input (capped)
  Project      { input, columns }          cost: input
```

**Selectivity estimation:**
- Primary key equality: `1 / row_count`
- General equality: 0.1
- Range predicates (`<`, `>`, `BETWEEN`): 0.33
- `LIKE`: 0.25
- `AND`: product of child selectivities
- `OR`: `1 - (1 - s1)(1 - s2)`

**Table statistics** (collected via `ANALYZE table`): Per-column `ColumnStats` with distinct count, null count, min/max values, and top-N frequency histograms.

**Join selection:** If an equi-join is detected (equality condition on a single column), the planner chooses `HashJoin`. Otherwise, `NestedLoopJoin`.

### Executor

The executor (`sql/exec.rs`, ~5,000 lines) evaluates the parsed AST against the database. It handles all statement types, dispatches to the appropriate facet for specialized operations (FTS, time-series, etc.), and returns results as `SqlResult` (rows, affected count, or explain output).

### Vectorized Execution Engine

For analytical queries, the vectorized engine (`sql/vectorized.rs`, ~1,100 lines) processes data in columnar batches instead of row-at-a-time:

**Column types:** `Int64`, `Float64`, `Boolean`, `Utf8` — each with a parallel null bitmap.

**Batch:** `RecordBatch { schema, columns: Vec<ColumnVector>, num_rows }` (default batch size: 1024 rows)

**Operators:**
- `vectorized_filter` — Apply boolean selection vector, gather matching rows
- `vectorized_project` — Select column subsets by index
- `vectorized_sort` — Argsort with null-aware ordering
- `vectorized_limit` — Slice with offset
- `vectorized_hash_aggregate` — Group by key columns, compute SUM/COUNT/AVG/MIN/MAX
- `vectorized_hash_join` — Build hash table from right side, probe from left

**Boolean combinators:** `selection_and`, `selection_or`, `selection_not` for composing filter masks.

### Expression Evaluator

The evaluator (`sql/eval.rs`, ~1,600 lines) handles 50+ scalar functions:

| Category | Functions |
|----------|-----------|
| String | `UPPER`, `LOWER`, `LENGTH`, `SUBSTR`, `TRIM`, `LTRIM`, `RTRIM`, `REPLACE`, `CONCAT`, `CONCAT_WS`, `LEFT`, `RIGHT`, `LPAD`, `RPAD`, `REVERSE`, `SPLIT_PART`, `REPEAT`, `POSITION`, `INITCAP` |
| Numeric | `ABS`, `ROUND`, `CEIL`, `FLOOR`, `MOD`, `POWER`, `SQRT`, `LOG`, `LN`, `EXP`, `SIGN`, `RANDOM`, `PI` |
| Date/Time | `NOW`, `EPOCH`, `EXTRACT`, `DATE_TRUNC`, `TO_CHAR` |
| Conditional | `COALESCE`, `NULLIF`, `GREATEST`, `LEAST`, `IF`/`IIF`, `TYPEOF` |
| Aggregate | `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, `STRING_AGG`, `STDDEV_POP`, `STDDEV_SAMP`, `VAR_POP`, `VAR_SAMP` |
| Window | `ROW_NUMBER`, `RANK`, `DENSE_RANK`, `LEAD`, `LAG` |
| Time-Series | `TIME_BUCKET`, `TIME_BUCKET_GAPFILL`, `LOCF`, `INTERPOLATE`, `DELTA`, `RATE` |
| Full-Text | `MATCH`, `HIGHLIGHT` |
| Type | `CAST` (INTEGER, REAL, TEXT, BOOLEAN) |

---

## Facets

Facets are specialized query engines built over the same underlying fact ledger. Each facet maintains its own indexes and provides domain-specific operations, but all data ultimately comes from the same append-only storage.

### Relational Facet

Standard SQL tables with typed columns. Metadata stored under `__meta/table/<name>`, rows under `table/<name>/<pk>`.

**Schema modes:**
- **Typed columns** — `CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, balance REAL)` with columnar row encoding (null bitmaps, typed values)
- **JSON documents** — `CREATE TABLE t (pk TEXT PRIMARY KEY)` with JSON payloads accessible via `doc.field` or `doc->>'field'`

**Columnar encoding format:**
```
null_bitmap: ceil(n_cols / 8) bytes — bit i=1 means NULL
Per non-null column:
  Integer → 8 bytes LE i64
  Real    → 8 bytes LE f64
  Text    → varint(len) + UTF-8 bytes
  Boolean → 1 byte
  Blob    → varint(len) + raw bytes
```

### Full-Text Search Facet

Inverted index with BM25 relevance ranking. Supports multi-column indexes with per-column boosting.

```sql
CREATE FULLTEXT INDEX idx_docs ON documents (title, body);
SELECT * FROM documents WHERE MATCH(body, 'temporal database');
SELECT HIGHLIGHT(body, 'temporal') FROM documents WHERE MATCH(body, 'temporal');
```

**Internals:** Tokenizer → stemmer → posting list. `MATCH()` returns a BM25 relevance score. `HIGHLIGHT()` wraps matched terms in `<<match>>` markers. Posting lists are automatically maintained on INSERT/UPDATE/DELETE.

### Time-Series Facet

Bucketed storage optimized for high-frequency metric ingestion.

```sql
CREATE TIMESERIES TABLE metrics (ts TIMESTAMP, value REAL) WITH (bucket_size = '1h');

SELECT TIME_BUCKET('1h', ts) AS hour, AVG(value)
FROM metrics
GROUP BY 1
ORDER BY 1;

-- Gap filling with last observation carried forward
SELECT TIME_BUCKET_GAPFILL('1h', ts) AS hour, LOCF(AVG(value))
FROM metrics
GROUP BY 1;

-- Rate of change
SELECT ts, RATE(value) FROM metrics;
```

**Functions:** `TIME_BUCKET`, `TIME_BUCKET_GAPFILL`, `LOCF` (last observation carried forward), `INTERPOLATE` (linear interpolation), `DELTA` (difference from previous), `RATE` (per-second rate of change).

### Vector Search Facet

HNSW (Hierarchical Navigable Small World) approximate nearest neighbor search with multiple distance metrics.

**`VectorIndex` API:**
- `insert(id, vector)` — Add a vector
- `search(query_vector, k, ef)` — Find k nearest neighbors
- `delete(id)` — Remove a vector
- `stats()` — Index statistics

**Distance metrics:** Cosine, Euclidean (L2), Dot Product

**HNSW parameters:**
- `max_connections` (M): 16 — max neighbors per node per layer
- `ef_construction`: 200 — beam width during index build
- Level assignment: `-ln(rand()) × 1/ln(M)` (exponential distribution)

**Search strategy:** Uses ef-beam search. Automatically switches to exact brute-force when the index is small (`count < 1000`) or when k is large relative to the dataset (`k/count > 0.1`).

**`VectorStore`** manages multiple named indexes. `VectorSqlBridge` provides SQL-level functions: `vector_distance`, `cosine_similarity`, `vector_norm`, `vector_dims`.

### Event Sourcing Facet

First-class support for event sourcing patterns with aggregate projections and snapshot optimization.

**Key scheme:**
```
__es/meta/<store>                            store metadata
__es/data/<store>/<aggregate_id>/<seq:020>   events (zero-padded sequence)
__es/idem/<store>/<idempotency_key>          deduplication markers
__es/snap/<store>/<aggregate_id>/latest      latest snapshot
```

**Operations:**
- `create_event_store(name)` — Create a named event store
- `append_event(store, aggregate_id, event_type, payload)` — Append with automatic sequence numbering
- `get_events(store, aggregate_id)` — Retrieve event stream
- `get_aggregate_state(store, aggregate_id)` — Replay events (from snapshot if available) into aggregate state
- `save_snapshot(store, aggregate_id, state)` — Persist snapshot for faster future replays
- `find_aggregates_by_event_type(store, event_type)` — Cross-aggregate event query

**Idempotency:** Pass an `idempotency_key` with `append_event`. If a matching key exists in `__es/idem/`, the duplicate is rejected.

### Schema Evolution Facet

Versioned schema migrations with rollback support.

- `MigrationManager` — Register, apply, rollback, and list migrations. Stored under `__schema/migration/` with version, SQL, timestamp, and checksum.
- `SchemaRegistry` — Per-table column definitions with version history. `diff()` computes added/removed/modified columns between versions.

---

## Change Data Capture

TensorDB supports real-time streaming of changes to downstream consumers.

### Basic Subscriptions

```rust
let rx = db.subscribe(b"orders/");
// rx receives ChangeEvent for every write with key prefix "orders/"
```

Prefix-filtered, crossbeam-channel-based. Multiple subscribers can watch different prefixes.

### Durable Cursors

For at-least-once delivery that survives restarts:

- `DurableCursor` persists position under `__cdc/cursor/<consumer_id>`
- `poll()` returns the next batch of events
- `ack()` atomically updates the cursor position

### Consumer Groups

Kafka-style partition assignment for parallel consumption:

- `ConsumerGroupManager` handles round-robin shard assignment
- Rebalancing on consumer join/leave with generation tracking
- Each consumer in a group processes a disjoint subset of shards

**Isolation:** AI internal writes (under `__ai/` prefix) are automatically filtered from user-facing change feeds.

---

## AI Runtime

The AI runtime is embedded directly in the database engine. No external model servers, no network calls, no separate processes. It runs in-process alongside the storage engine.

### Insight Synthesis

When `ai_auto_insights` is enabled, a background thread consumes shard change-feed events, micro-batches them (configurable window and max events), and synthesizes operational insights. Insights are stored as immutable facts under `__ai/insight/...` and are queryable via `EXPLAIN AI '<key>'`.

### Inline Risk Scoring

When `ai_inline_risk_assessment` is enabled, every write gets a risk score computed synchronously before acknowledgment. High-risk writes are flagged for review.

### AI Advisors

Three advisors observe database behavior and make recommendations:

- **Compaction advisor** — Observes read patterns and recommends compaction priorities for key ranges with high read amplification
- **Cache advisor** — Uses Count-Min Sketch for hot-key frequency estimation, recommends cache size adjustments based on hit ratio and working set size
- **Query advisor** — Analyzes query patterns and suggests index creation or query rewrites

### Inference Engine

The inference engine (`ai/inference.rs`, ~1,300 lines) provides:

- **Scoring functions** — Configurable linear models with activations (Sigmoid, ReLU, Softmax). Weight matrices stored as row-major `Vec<f32>`.
- **RAG pipeline** — Document chunking at sentence boundaries, TF-IDF keyword scoring, hybrid retrieval combining vector similarity + BM25, configurable fusion weights.

**Softmax** uses numerically stable computation: `exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))`.

### ML Pipeline

The ML pipeline (`ai/ml_pipeline.rs`, ~960 lines) provides:

- **Feature store** — Create named feature sets, ingest per-entity features, point-in-time joins (filters by `updated_at <= as_of`).
- **Model registry** — Versioned model lifecycle: Registered → Validated → Deployed → Deprecated → Archived. Tracks inference count and average latency.
- **Pipeline advisor** — Monitors accuracy (retrain threshold: 0.7), feature staleness (refresh threshold: 0.5), and overall health score (0.4 × freshness + 0.6 × accuracy).

---

## Cluster Architecture

### Raft Consensus

Full Raft implementation for leader election and log replication:

- **Election:** Timeout 150–300 ms (node-seeded for stagger). Candidate increments term, votes for self, requests votes from peers. Majority wins.
- **Log replication:** Leader sends `AppendEntries` to followers with log consistency checks and conflict resolution. Commit index advances when a majority of nodes have replicated an entry.
- **Safety:** Votes granted only if candidate's log is at least as up-to-date. Leader appends Noop on election to establish commit point.

**`RaftNode` fields:** node_id, state (Follower/Candidate/Leader), current_term, voted_for, log entries, commit_index, last_applied, per-peer next_index/match_index.

### Replication

WAL-based replication from primary to standbys:

- **`WalShipper`** — Segments outbound WAL records, tracks per-standby acknowledgment state (lag in sequences and seconds, consecutive failures, health).
- **`WalReceiver`** — Inbound queue on standbys, replays WAL segments in sequence order.
- **Modes:** Async (drop oldest on overflow for best-effort), Sync (block until standby ACKs for strong durability).

**Failover readiness scoring:**
```
base = 1.0
if lag_seqs > 100: -0.3;  if lag_seqs > 10: -0.1
if lag_seconds > 30: -0.4; if lag_seconds > 5: -0.2
if consecutive_failures > 3: -0.3
score = max(0.0, base + penalties)
```

### Read Replica Routing

`ReadReplicaRouter` filters replicas by health and staleness tolerance, then round-robins reads among eligible replicas. Falls back to primary if no replica meets the staleness requirement.

### Horizontal Scaling

- **Consistent hash ring** — FNV-1a hash with virtual nodes (`"<node_id>#vn<i>"`). `get_node(key)` walks the ring clockwise.
- **Rebalance planning** — `compute_rebalance()` compares current placement against ring-ideal placement, emits transfer plans with estimated byte counts.
- **Scatter-gather** — `ScatterGatherExecutor` fans queries to target nodes, merges results with configurable strategies: Concatenate, SortMerge, or Aggregate (SUM/COUNT).

**Imbalance metric:** `(max_shards - min_shards) / avg_shards`. Default rebalance threshold: 0.25.

### Cluster Membership

Nodes register under `__cluster/node/<node_id>` with role, status, heartbeat, and shard assignments.

**Roles:** Primary, Standby, ReadReplica, Arbiter

**Status lifecycle:** Starting → Active → (Draining → Removed | Down)

**Shard assignment:** Round-robin distribution with configurable replication factor: each shard is assigned to `min(replication_factor, active_nodes)` nodes.

---

## Authentication & Authorization

### User Management

`UserManager` handles user lifecycle: create, authenticate (argon2/bcrypt password hashing), change password, disable, grant/revoke roles and permissions.

### Role-Based Access Control

`RoleManager` maintains roles with built-in defaults (admin, reader, writer). Each role contains a set of `Permission` entries.

**Privilege types:** Select, Insert, Update, Delete, Create, Drop, Alter, Admin — each optionally scoped to a specific table.

**Checking:** `RbacChecker::check_privilege()` resolves permissions from both direct user grants and inherited role grants.

### Sessions

`SessionStore` provides token-based session management with configurable TTL expiry and per-user revocation.

---

## PostgreSQL Wire Protocol

The `tensordb-server` crate implements the PostgreSQL v3 wire protocol, allowing any Postgres-compatible client (psql, JDBC, Python's psycopg2, etc.) to connect.

**Supported messages:**
- Startup with parameter negotiation
- Simple query mode (Q message)
- Extended query mode (Parse/Bind/Describe/Execute/Sync)
- Password authentication
- SSL rejection (TLS not yet supported)

**Type OID mapping:**

| TensorDB Type | PostgreSQL OID | Postgres Name |
|----------------|----------------|---------------|
| Boolean | 16 | BOOL |
| Integer | 23 / 20 | INT4 / INT8 |
| Real | 701 | FLOAT8 |
| Text | 25 | TEXT |
| JSON | 114 / 3802 | JSON / JSONB |
| Timestamp | 1114 | TIMESTAMP |
| Blob | 17 | BYTEA |

**Architecture:** Tokio async TCP listener. Each connection spawns an independent task sharing an `Arc<Database>`. Default port: 5433.

---

## Connection Pooling

`ConnectionPool` manages a pool of database connections for the wire protocol server:

- **Configuration:** `max_connections` (100), `min_idle` (5), `idle_timeout` (5 min), `acquire_timeout` (30s)
- **Acquire:** Pop from idle deque → check expiry → wrap in RAII `PooledConnection` guard. If no idle slot and below max, create new. Otherwise, timeout.
- **Release:** RAII guard returns the connection on drop (LIFO reuse for cache locality)
- **Warmup:** Pre-creates `min_idle` connections at startup

---

## Optional Native Acceleration

Feature flag: `--features native`

The pure-Rust path is always fully functional. The native path provides C++ implementations of performance-critical operations through the `cxx` bridge:

**Trait boundary in `native_bridge.rs`:**
- `Hasher` — Key hashing for shard routing
- `Compressor` — Block compression for SSTables
- `BloomProbe` — Bloom filter membership testing

The native implementations include call-counter instrumentation and deterministic equality tests against the Rust reference implementations.

---

## Data Interchange

### Import/Export

```sql
COPY accounts TO '/tmp/accounts.csv' FORMAT CSV;
COPY accounts FROM '/tmp/data.json' FORMAT JSON;
COPY accounts TO '/tmp/out.parquet' FORMAT PARQUET;
```

### Table Functions

```sql
SELECT * FROM read_csv('data.csv');          -- auto type detection
SELECT * FROM read_json('data.json');        -- JSON array or NDJSON
SELECT * FROM read_parquet('data.parquet');   -- behind --features parquet
```

Parquet support uses Apache Arrow 54 and the parquet 54 crate. CSV parsing follows RFC 4180.

---

## Storage Key Prefixes

TensorDB uses key prefixes to organize internal and user data:

| Prefix | Purpose |
|--------|---------|
| `table/<name>/<pk>` | SQL table row data |
| `__meta/table/<name>` | Table metadata (schema, columns) |
| `__meta/view/<name>` | View definitions |
| `__meta/index/<table>/<name>` | Index metadata |
| `__ai/insight/...` | AI-synthesized insights |
| `__cdc/cursor/<id>` | Durable cursor positions |
| `__es/data/<store>/<agg>/<seq>` | Event sourcing events |
| `__es/snap/<store>/<agg>/latest` | Event sourcing snapshots |
| `__es/idem/<store>/<key>` | Idempotency deduplication |
| `__schema/migration/...` | Schema migration records |
| `__cluster/node/<id>` | Cluster membership |

---

## Invariants

These properties hold at all times and are the foundation of TensorDB's correctness guarantees:

1. **WAL-before-ACK** — No write is acknowledged until its WAL frame is enqueued for durability (fast path) or appended to disk (channel path).
2. **Monotonic timestamps** — Shard commit timestamps are strictly increasing per shard.
3. **Temporal preservation** — All temporal versions survive compaction. History is never dropped.
4. **SSTable immutability** — Once published, an SSTable is never modified. Compaction creates new files and atomically updates the manifest.
5. **Manifest atomicity** — The manifest is replaced via write-temp/fsync/rename/fsync-dir. No partial state is visible.
6. **Deterministic recovery** — Manifest + WAL replay reproduces the exact visible state.
7. **AI isolation** — Internal AI facts (`__ai/...`) never appear in user-facing change feeds.
8. **Read consistency** — `ShardReadHandle` reads are consistent within a single `get_visible()` call (snapshot of memtable + levels at read time).
