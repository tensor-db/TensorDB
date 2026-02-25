# SpectraDB Design

## 1) Purpose and Model

SpectraDB separates immutable truth from query projections:
- Core truth: append-only bitemporal fact ledger.
- Facets: query planes built over the same truth.

MVP uses one facet (`RelationalFacet`) while preserving a layout that supports future search/vector/graph/time-series facets.

## 2) Data Model

Each fact version has:
- `user_key`
- `commit_ts` (system/transaction timeline)
- `valid_from`, `valid_to` (business-valid timeline)
- payload (`doc` bytes)

Internal key format:

```text
internal_key = user_key || 0x00 || commit_ts(be_u64) || kind(u8)
```

MVP `kind`:
- `0`: PUT

## 3) Semantics

### MVCC
- Each shard owns a monotonic `u64` commit counter.
- `PUT` assigns commit timestamp in shard writer loop.
- Read `AS OF T` sees newest committed version where `commit_ts <= T`.

### Bitemporal
- Valid interval is half-open: `[valid_from, valid_to)`.
- `VALID AT V` returns versions satisfying:

```text
valid_from <= V < valid_to
```

### Combined
- `AS OF` filters by commit timeline.
- `VALID AT` filters by domain-valid timeline.
- Final record is max visible `commit_ts` after both filters.

## 4) Storage Pipeline

```text
WAL append -> memtable -> SSTable flush (L0) -> compaction (simple MVP)
```

### WAL format

Frame:

```text
MAGIC(u32) | len(u32) | crc32(u32) | payload[len]
```

Payload (`FactWrite`):
- internal_key length varint + bytes
- fact length varint + bytes
- metadata presence flags + optional fields

Recovery behavior:
- CRC mismatch: hard error
- torn tail: deterministic stop at last valid frame

### Memtable
- Ordered map keyed by internal key.
- Write-optimized in-memory staging.
- Flushed when `memtable_max_bytes` threshold is exceeded.

### SSTable format

```text
Header:
  MAGIC(u32), VERSION(u32), block_size(u32)

Data blocks:
  [block_len u32][entries...]
  entry = [klen varint][vlen varint][key][value]

Index block (at end):
  count(u32), repeated(last_key, offset, len)

Bloom block (at end):
  bloom metadata + bitset bytes

Footer:
  index_offset(u64), bloom_offset(u64), footer_magic(u32)
```

Lookup path:
1. Bloom negative => fast not found
2. Binary-search index for candidate block start
3. Scan block entries (linear in MVP)

### Manifest
- Tracks per-shard WAL/SSTables and next file id.
- JSON metadata for MVP readability and deterministic replay.
- Atomic update pattern:
  - write temp
  - fsync temp
  - rename to MANIFEST
  - fsync parent dir

## 5) Shard Execution Model

Shard id:

```text
shard = hash(user_key) % shard_count
```

Each shard has a dedicated actor loop with:
- WAL handle
- active memtable
- immutable memtables (flush queue)
- L0 SSTable list
- local commit counter

Why this model:
- single writer per shard avoids fine-grained lock contention,
- sequential WAL I/O per shard,
- straightforward correctness boundaries.

## 6) SQL / Relational Facet

### 6.1 Implemented SQL Surface

- `CREATE TABLE t (pk TEXT PRIMARY KEY);`
- `INSERT INTO t (pk, doc) VALUES ('k', '{...json...}');`
- `SELECT doc FROM t_or_view [WHERE pk='k'] [AS OF <commit_ts>] [VALID AT <valid_ts>] [ORDER BY pk ASC|DESC] [LIMIT N];`
- `SELECT pk, doc FROM t_or_view ...`
- `SELECT count(*) FROM t_or_view ...`
- `SELECT pk, doc FROM left_t JOIN right_t ON left_t.pk=right_t.pk ...` (hash-join skeleton)
- `SELECT pk, count(*) FROM ... GROUP BY pk ...` (grouping skeleton)
- `EXPLAIN SELECT ...` (point-read explain line)
- `BEGIN`, `COMMIT`, `ROLLBACK`
- multi-statement batches (`stmt1; stmt2; ...`) with semicolon-aware splitting
- `CREATE VIEW v AS SELECT ...` (view metadata + select target indirection)
- `CREATE INDEX idx ON t (pk)` (metadata-only)
- `ALTER TABLE t ADD COLUMN c TEXT` (metadata-only)
- `SHOW TABLES`
- `DESCRIBE <table>`
- `DROP TABLE <table>`, `DROP VIEW <view>`, `DROP INDEX <idx> ON <table>`

### 6.2 Execution Semantics and Limits

- SQL call model:
  - `db.sql(...)` executes one batch string and returns only the final statement result.
  - transaction state is local to that single call.
- Transaction behavior:
  - writes are staged after `BEGIN` and only persisted on `COMMIT`.
  - `ROLLBACK` discards staged writes.
  - an open transaction at end-of-call returns an error (no cross-call session transaction).
- Query shape constraints:
  - supported projections: `doc`, `pk, doc`, `count(*)`, `pk, count(*)`.
  - supports full table/view scans and key-filtered reads with temporal predicates.
  - supports `ORDER BY pk` and `LIMIT`.
  - supports pk-equality join skeleton (`JOIN ... ON <left>.pk=<right>.pk`) via bounded hash join.
  - supports grouping skeleton (`SELECT pk, count(*) ... GROUP BY pk`).
  - no window functions, subqueries, CTEs, or general-purpose join/group expression support yet.
- Metadata-only boundaries:
  - `CREATE INDEX` persists definition metadata only; lookup planner/executor does not use it.
  - `ALTER TABLE ... ADD COLUMN` updates schema metadata only; row payload and query operators remain `(pk, doc)`-centric.
  - `CREATE VIEW` requires `SELECT doc ... WHERE pk='...'`; read path resolves to source table + temporal defaults, not full SQL rewrite/materialization.
  - `DROP` statements write append-only tombstones (`DROP TABLE` also tombstones live row keys to prevent stale-row resurfacing on recreate).
  - `EXPLAIN` currently supports point-select shape only (`SELECT doc ... WHERE pk='...'`).

### 6.3 Storage Mapping

- table metadata key: `__meta/table/<table>`
- view metadata key: `__meta/view/<view>`
- index metadata key: `__meta/index/<table>/<index>`
- row key: `table/<table>/<pk>`
- row payload: JSON bytes

### 6.4 DuckDB Category Gap Map (Current)

- Statements:
  - partial parity (`CREATE TABLE/VIEW/INDEX`, `ALTER TABLE ADD COLUMN`, `INSERT`, transactions, `EXPLAIN`, `DROP`)
  - missing `UPDATE`, `DELETE`, broader DDL/DML
- Query Syntax:
  - partial parity via point + scan `SELECT`, `ORDER BY pk`, `LIMIT`, pk-equality join skeleton, basic group-by skeleton
  - missing rich joins/window/subquery surface
- Aggregates / Functions:
  - partial (`count(*)` and `pk,count(*) GROUP BY pk` implemented; broad function/aggregate catalog missing)
- Data Types:
  - constrained/fixed row contract, no broad SQL type system
- Meta Queries:
  - partial (`SHOW TABLES`, `DESCRIBE` implemented)
- Indexes:
  - metadata persisted, execution support missing
- Views:
  - metadata-backed indirection only, no advanced view capabilities

### 6.5 Near-Term SQL Syntax Targets

- richer joins over row streams (non-pk predicates, multiple joins, and planner selection)
- richer grouping (`GROUP BY`) and additional aggregates (`min/max/sum/avg`)
- `UPDATE` / `DELETE` with temporal-aware semantics
- richer catalog and typed schema enforcement

## 7) Native Integration (Optional)

Feature flag: `native`

Design:
- core remains pure Rust and fully functional,
- `spectradb-native` crate hosts C++ code and `cxx` bridge,
- trait boundary in `native_bridge.rs`:
  - `Hasher`
  - `Compressor`
  - `BloomProbe`

Current demo module:
- native hasher with call-counter instrumentation,
- deterministic equality tests vs Rust reference implementation.

## 8) Invariants Checklist

- No ACK before WAL append.
- Shard commit timestamps are monotonic.
- Visible read result is highest commit meeting temporal predicates.
- SSTables are immutable once published.
- Manifest state is atomically replaced.
- Recovery from manifest + WAL replay reproduces visible state.

## 9) Facet Subscription Direction (Post-MVP)

Future facets should consume fact stream from ledger append points:
- append event contains key, commit_ts, valid interval, payload,
- facet-specific indexes can be updated asynchronously,
- core ledger remains source of truth for reconciliation and rebuilds.

MVP facet directly queries core storage, but interface boundaries already support decoupled facet pipelines.
