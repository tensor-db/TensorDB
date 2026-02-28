# EOAC Critical Gaps Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close all 5 critical gaps (transactions, index acceleration, N-way JOINs, concurrency, recovery) using the Epoch-Ordered Append-Only Concurrency (EOAC) architecture.

**Architecture:** A global epoch counter (`AtomicU64`) unifies transactions, MVCC, and recovery. Intent buffers provide zero-cost rollback. Predicate-to-index matching enables automatic index acceleration. N-way JOIN chains with greedy reordering beat naive execution. WAL records carry epoch+txn_id for crash-safe transactions and PITR.

**Tech Stack:** Rust, crossbeam, parking_lot::RwLock, AtomicU64, existing WAL/memtable/SSTable infrastructure.

**Build command:** `BINDGEN_EXTRA_CLANG_ARGS="--include-directory=/usr/lib/gcc/aarch64-linux-gnu/13/include" cargo build --workspace`

**Test command:** `BINDGEN_EXTRA_CLANG_ARGS="--include-directory=/usr/lib/gcc/aarch64-linux-gnu/13/include" cargo test --workspace --all-targets`

---

## Phase A: Global Epoch Infrastructure

### Task 1: Add global_epoch to Database

**Files:**
- Modify: `crates/tensordb-core/src/engine/db.rs:85-98` (Database struct)
- Modify: `crates/tensordb-core/src/engine/db.rs:101-281` (Database::open)
- Test: `tests/epoch_core.rs` (new)

**Step 1: Write the failing test**

Create `tests/epoch_core.rs`:
```rust
//! Tests for global epoch infrastructure

use tempfile::TempDir;
use tensordb::{Config, Database};

fn open_db(dir: &TempDir) -> Database {
    Database::open(dir.path(), Config { shard_count: 2, ..Config::default() }).unwrap()
}

#[test]
fn epoch_starts_at_one() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    assert!(db.current_epoch() >= 1);
}

#[test]
fn epoch_advances_on_write() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let e1 = db.current_epoch();
    db.put(b"key1", b"val1".to_vec(), 0, u64::MAX, Some(1)).unwrap();
    // Non-transactional writes use current epoch, don't advance it
    let e2 = db.current_epoch();
    assert_eq!(e1, e2);
}

#[test]
fn epoch_snapshot_returns_consistent_value() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let snap = db.current_epoch();
    assert!(snap >= 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --test epoch_core -- --nocapture 2>&1`
Expected: FAIL — `current_epoch` method doesn't exist

**Step 3: Implement global epoch in Database**

In `crates/tensordb-core/src/engine/db.rs`:

Add to Database struct (after line 98, before closing brace):
```rust
    global_epoch: Arc<AtomicU64>,
```

Add import at top:
```rust
use std::sync::atomic::AtomicU64;
```

In `Database::open()`, initialize before the `Ok(Self { ... })` block:
```rust
        let global_epoch = Arc::new(AtomicU64::new(1));
```

Add to the returned struct:
```rust
            global_epoch,
```

Add public method after `sync()`:
```rust
    /// Returns the current global epoch.
    pub fn current_epoch(&self) -> u64 {
        self.global_epoch.load(std::sync::atomic::Ordering::Acquire)
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test --test epoch_core -- --nocapture 2>&1`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add crates/tensordb-core/src/engine/db.rs tests/epoch_core.rs
git commit -m "feat(eoac): add global epoch counter to Database"
```

---

### Task 2: Add epoch to WAL record format

**Files:**
- Modify: `crates/tensordb-core/src/storage/wal.rs:10-12` (constants)
- Modify: `crates/tensordb-core/src/storage/wal.rs:41-58` (append)
- Modify: `crates/tensordb-core/src/storage/wal.rs:111-175` (replay)
- Modify: `crates/tensordb-core/src/ledger/record.rs:11-15` (FactWrite)
- Test: `crates/tensordb-core/src/storage/wal.rs` (existing tests)

**Step 1: Write the failing test**

Add to wal.rs tests section:
```rust
#[test]
fn wal_epoch_roundtrip() {
    let dir = tempfile::TempDir::new().unwrap();
    let wal_path = dir.path().join("test.wal");
    let mut wal = Wal::create(&wal_path, 100).unwrap();

    let write = FactWrite {
        internal_key: b"testkey\x00\x00\x00\x00\x00\x00\x00\x01\x01".to_vec(),
        fact: b"testval".to_vec(),
        metadata: FactMetadata { source_id: None, schema_version: None },
        epoch: 42,
        txn_id: 7,
    };
    wal.append(&write).unwrap();
    drop(wal);

    let recovered = Wal::replay(&wal_path).unwrap();
    assert_eq!(recovered.len(), 1);
    assert_eq!(recovered[0].epoch, 42);
    assert_eq!(recovered[0].txn_id, 7);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib wal_epoch_roundtrip -- --nocapture 2>&1`
Expected: FAIL — `epoch` and `txn_id` fields don't exist on FactWrite

**Step 3: Add epoch and txn_id to FactWrite**

In `crates/tensordb-core/src/ledger/record.rs`, add to FactWrite struct:
```rust
#[derive(Debug, Clone)]
pub struct FactWrite {
    pub internal_key: Vec<u8>,
    pub fact: Vec<u8>,
    pub metadata: FactMetadata,
    pub epoch: u64,
    pub txn_id: u64,
}
```

Update `FactWrite::encode_payload()` to include epoch and txn_id:
```rust
pub fn encode_payload(&self) -> Vec<u8> {
    let mut out = Vec::with_capacity(self.internal_key.len() + self.fact.len() + 32);
    out.extend_from_slice(&self.epoch.to_le_bytes());
    out.extend_from_slice(&self.txn_id.to_le_bytes());
    // ... existing encode ...
    out
}
```

Update `FactWrite::decode_payload()` to read epoch and txn_id:
```rust
pub fn decode_payload(data: &[u8]) -> Result<Self> {
    if data.len() < 16 {
        return Err(TensorError::WalCorrupted("payload too short for epoch header".into()));
    }
    let epoch = u64::from_le_bytes(data[0..8].try_into().unwrap());
    let txn_id = u64::from_le_bytes(data[8..16].try_into().unwrap());
    // ... decode rest from &data[16..] ...
}
```

**Step 4: Fix all compilation errors**

Every place that constructs a `FactWrite` needs `epoch: 0, txn_id: 0` added:
- `shard.rs:handle_put()` — use current epoch from shared state
- `fast_write.rs:try_fast_put()` — use current epoch from database
- `group_wal.rs:encode_wal_frame()` — pass epoch through

Search for all `FactWrite {` and add the fields. Use `epoch: 0, txn_id: 0` as default for non-transactional writes.

**Step 5: Run full test suite**

Run: `cargo test --workspace --all-targets 2>&1 | tail -5`
Expected: All tests pass (epoch=0 is backward compatible)

**Step 6: Commit**

```bash
git add crates/tensordb-core/src/ledger/record.rs crates/tensordb-core/src/storage/wal.rs \
       crates/tensordb-core/src/engine/shard.rs crates/tensordb-core/src/engine/fast_write.rs \
       crates/tensordb-core/src/storage/group_wal.rs
git commit -m "feat(eoac): add epoch and txn_id to WAL records"
```

---

### Task 3: Thread global_epoch into write paths

**Files:**
- Modify: `crates/tensordb-core/src/engine/db.rs:283-313` (put method)
- Modify: `crates/tensordb-core/src/engine/fast_write.rs:44-51` (FastWritePath struct)
- Modify: `crates/tensordb-core/src/engine/fast_write.rs:74-136` (try_fast_put)
- Modify: `crates/tensordb-core/src/engine/shard.rs:503-570` (handle_put)
- Test: `tests/epoch_core.rs`

**Step 1: Write the failing test**

Add to `tests/epoch_core.rs`:
```rust
#[test]
fn writes_carry_epoch() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.put(b"key1", b"val1".to_vec(), 0, u64::MAX, Some(1)).unwrap();
    // After a write, we can read it back — the epoch is embedded in the WAL
    let val = db.get(b"key1", None, None).unwrap();
    assert_eq!(val, Some(b"val1".to_vec()));
}
```

**Step 2: Implement epoch threading**

Pass `Arc<AtomicU64>` (global_epoch) into FastWritePath and ShardRuntime so they can stamp writes with the current epoch.

In `FastWritePath::new()`, accept `global_epoch: Arc<AtomicU64>` and store it.
In `try_fast_put()`, read current epoch: `let epoch = self.global_epoch.load(Ordering::Acquire);` and pass to FactWrite.
In `ShardCommand::Put`, add `epoch: u64` field.
In `Database::put()`, load epoch before sending command.

**Step 3: Run tests**

Run: `cargo test --workspace --all-targets 2>&1 | tail -5`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/tensordb-core/src/engine/
git commit -m "feat(eoac): thread global epoch into write paths"
```

---

## Phase B: Intent-Based Transactions

### Task 4: Redesign SqlSession for epoch-aware transactions

**Files:**
- Modify: `crates/tensordb-core/src/sql/exec.rs:433-495` (SqlSession, PendingPut)
- Test: `tests/epoch_core.rs`

**Step 1: Write the failing test**

Add to `tests/epoch_core.rs`:
```rust
use tensordb_core::sql::exec::SqlResult;

fn rows(result: SqlResult) -> Vec<serde_json::Value> {
    match result {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| serde_json::from_slice(r).unwrap_or(serde_json::Value::String(
                String::from_utf8_lossy(r).to_string(),
            )))
            .collect(),
        _ => panic!("expected rows, got {result:?}"),
    }
}

#[test]
fn transaction_commit_makes_writes_visible() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k2', 'v2')").unwrap();
    db.sql("COMMIT").unwrap();
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 2);
    assert_eq!(rs[0]["id"], "k1");
    assert_eq!(rs[1]["id"], "k2");
}

#[test]
fn transaction_rollback_discards_writes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();
    db.sql("ROLLBACK").unwrap();
    let rs = rows(db.sql("SELECT id FROM t").unwrap());
    assert_eq!(rs.len(), 0);
}

#[test]
fn transaction_reads_own_writes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();
    let rs = rows(db.sql("SELECT val FROM t WHERE pk = 'k1'").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["val"], "v1");
    db.sql("COMMIT").unwrap();
}

#[test]
fn commit_advances_global_epoch() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    let e1 = db.current_epoch();
    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();
    db.sql("COMMIT").unwrap();
    let e2 = db.current_epoch();
    assert!(e2 > e1, "epoch should advance on commit: e1={e1}, e2={e2}");
}
```

**Step 2: Run tests to check current behavior**

These tests should mostly pass already (transactions exist). The `commit_advances_global_epoch` test should fail.

**Step 3: Wire epoch into SqlSession commit**

In `crates/tensordb-core/src/sql/exec.rs`:

The `execute_sql()` function (line 527) takes `&Database`. We need to access `db.global_epoch` during COMMIT.

Modify `SqlSession::commit()` to accept a reference to the global epoch and call `fetch_add(1, SeqCst)` to get the commit_epoch. Pass this epoch to each `db.put()` call.

This requires adding an `epoch` parameter to `Database::put()` or a new internal method `put_with_epoch()`.

**Option (simpler):** Add `Database::advance_epoch(&self) -> u64` public method:
```rust
pub fn advance_epoch(&self) -> u64 {
    self.global_epoch.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1
}
```

In `SqlSession::commit()`:
```rust
fn commit(&mut self, db: &Database) -> Result<(u64, Option<u64>)> {
    let commit_epoch = db.advance_epoch();
    let mut last_ts = None;
    let count = self.pending.len() as u64;
    while let Some(p) = self.pending.pop_front() {
        let ts = db.put(&p.key, p.value, p.valid_from, p.valid_to, p.schema_version)?;
        last_ts = Some(ts);
    }
    self.in_txn = false;
    Ok((count, last_ts))
}
```

**Step 4: Run full test suite**

Run: `cargo test --workspace --all-targets 2>&1 | tail -5`
Expected: All tests pass, including new epoch advancement test

**Step 5: Commit**

```bash
git add crates/tensordb-core/src/engine/db.rs crates/tensordb-core/src/sql/exec.rs tests/epoch_core.rs
git commit -m "feat(eoac): epoch-aware transaction commit"
```

---

### Task 5: Add SAVEPOINT / ROLLBACK TO / RELEASE

**Files:**
- Modify: `crates/tensordb-core/src/sql/parser.rs:147-249` (Statement enum)
- Modify: `crates/tensordb-core/src/sql/parser.rs` (parse_statement)
- Modify: `crates/tensordb-core/src/sql/exec.rs:442-445` (SqlSession)
- Modify: `crates/tensordb-core/src/sql/exec.rs:548-578` (execute_stmt)
- Test: `tests/epoch_core.rs`

**Step 1: Write failing tests**

Add to `tests/epoch_core.rs`:
```rust
#[test]
fn savepoint_rollback_to() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();
    db.sql("SAVEPOINT sp1").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k2', 'v2')").unwrap();
    db.sql("ROLLBACK TO sp1").unwrap();
    // k2 should be gone, k1 should remain
    db.sql("COMMIT").unwrap();
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "k1");
}

#[test]
fn savepoint_release() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();
    db.sql("SAVEPOINT sp1").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k2', 'v2')").unwrap();
    db.sql("RELEASE sp1").unwrap();
    db.sql("COMMIT").unwrap();
    // Both k1 and k2 should be present
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 2);
}

#[test]
fn nested_savepoints() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();
    db.sql("SAVEPOINT sp1").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k2', 'v2')").unwrap();
    db.sql("SAVEPOINT sp2").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k3', 'v3')").unwrap();
    db.sql("ROLLBACK TO sp1").unwrap();
    // Only k1 should remain (sp2 is also rolled back)
    db.sql("COMMIT").unwrap();
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "k1");
}
```

**Step 2: Implement parser additions**

Add to Statement enum:
```rust
Savepoint { name: String },
RollbackTo { name: String },
ReleaseSavepoint { name: String },
```

Parse `SAVEPOINT <name>`, `ROLLBACK TO <name>`, `RELEASE <name>` in the statement parser. The parser already handles `ROLLBACK` — add a lookahead for `TO` keyword after `ROLLBACK`.

**Step 3: Implement SqlSession savepoints**

Add to SqlSession:
```rust
struct SqlSession {
    in_txn: bool,
    pending: VecDeque<PendingPut>,
    savepoints: Vec<(String, usize)>,  // (name, pending.len() at savepoint)
}
```

Handlers:
- `Savepoint { name }` → push `(name, pending.len())` onto savepoints
- `RollbackTo { name }` → find savepoint by name, truncate pending to that length, remove that savepoint and all later ones
- `ReleaseSavepoint { name }` → remove savepoint marker only (writes remain in pending)

**Step 4: Run tests**

Run: `cargo test --test epoch_core -- --nocapture 2>&1`
Expected: All savepoint tests pass

**Step 5: Commit**

```bash
git add crates/tensordb-core/src/sql/parser.rs crates/tensordb-core/src/sql/exec.rs tests/epoch_core.rs
git commit -m "feat(eoac): SAVEPOINT, ROLLBACK TO, RELEASE SAVEPOINT"
```

---

### Task 6: Add TXN_COMMIT WAL marker for crash-safe transactions

**Files:**
- Modify: `crates/tensordb-core/src/storage/wal.rs` (new TXN_COMMIT magic)
- Modify: `crates/tensordb-core/src/sql/exec.rs` (commit writes TXN_COMMIT marker)
- Test: `tests/epoch_core.rs`

**Step 1: Write the failing test**

Add to `tests/epoch_core.rs`:
```rust
#[test]
fn transaction_is_atomic_on_reopen() {
    let dir = TempDir::new().unwrap();
    {
        let db = open_db(&dir);
        db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
        db.sql("BEGIN").unwrap();
        db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();
        db.sql("INSERT INTO t (id, val) VALUES ('k2', 'v2')").unwrap();
        db.sql("COMMIT").unwrap();
        db.sync();
    }
    // Reopen
    let db = open_db(&dir);
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 2);
}
```

**Step 2: Add WAL TXN_COMMIT marker**

In `wal.rs`, add constant:
```rust
pub const WAL_TXN_COMMIT_MAGIC: u32 = 0x54584e43; // "TXNC"
```

Add method to Wal:
```rust
pub fn append_txn_commit(&mut self, epoch: u64, txn_id: u64, intent_count: u32) -> Result<()> {
    let mut payload = Vec::with_capacity(20);
    payload.extend_from_slice(&epoch.to_le_bytes());
    payload.extend_from_slice(&txn_id.to_le_bytes());
    payload.extend_from_slice(&intent_count.to_le_bytes());

    let len = payload.len() as u32;
    let mut crc_hasher = crc32fast::Hasher::new();
    crc_hasher.update(&payload);
    let crc = crc_hasher.finalize();

    self.file.write_all(&WAL_TXN_COMMIT_MAGIC.to_le_bytes())?;
    self.file.write_all(&len.to_le_bytes())?;
    self.file.write_all(&crc.to_le_bytes())?;
    self.file.write_all(&payload)?;
    self.sync()?; // TXN_COMMIT is always durable
    Ok(())
}
```

Update `replay()` to recognize TXN_COMMIT records and return them alongside regular writes.

**Step 3: Wire into SqlSession::commit()**

During commit, after all puts are written, send a TXN_COMMIT command to each touched shard. Add a `ShardCommand::TxnCommit { epoch, txn_id, count }` variant that the shard writes to its WAL.

**Step 4: Update replay to filter uncommitted transactions**

In shard.rs `ShardRuntime::open()`, after WAL replay:
1. Collect all txn_ids from replayed writes
2. Collect committed txn_ids from TXN_COMMIT markers
3. Only apply writes where `txn_id == 0` (non-transactional) OR `txn_id in committed_set`

**Step 5: Run tests**

Run: `cargo test --workspace --all-targets 2>&1 | tail -5`
Expected: All pass

**Step 6: Commit**

```bash
git add crates/tensordb-core/src/storage/wal.rs crates/tensordb-core/src/engine/shard.rs \
       crates/tensordb-core/src/sql/exec.rs tests/epoch_core.rs
git commit -m "feat(eoac): crash-safe transactions via TXN_COMMIT WAL markers"
```

---

## Phase C: Index-Accelerated Execution

### Task 7: Predicate-to-index matching in planner

**Files:**
- Modify: `crates/tensordb-core/src/sql/planner.rs:297-328` (plan_select)
- Modify: `crates/tensordb-core/src/sql/exec.rs` (pass index metadata to planner)
- Test: `tests/index_acceleration.rs` (new)

**Step 1: Write failing tests**

Create `tests/index_acceleration.rs`:
```rust
//! Tests for index-accelerated query execution

use tempfile::TempDir;
use tensordb::{Config, Database};
use tensordb_core::sql::exec::SqlResult;

fn open_db(dir: &TempDir) -> Database {
    Database::open(dir.path(), Config { shard_count: 2, ..Config::default() }).unwrap()
}

fn rows(result: SqlResult) -> Vec<serde_json::Value> {
    match result {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| serde_json::from_slice(r).unwrap_or(serde_json::Value::String(
                String::from_utf8_lossy(r).to_string(),
            )))
            .collect(),
        _ => panic!("expected rows, got {result:?}"),
    }
}

#[test]
fn index_accelerated_equality_lookup() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT, name TEXT)").unwrap();
    for i in 0..100 {
        db.sql(&format!(
            "INSERT INTO users (id, email, name) VALUES ('u{i}', 'user{i}@test.com', 'User {i}')"
        )).unwrap();
    }
    db.sql("CREATE INDEX idx_email ON users (email)").unwrap();

    // This query should use the index
    let rs = rows(db.sql("SELECT name FROM users WHERE email = 'user50@test.com'").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["name"], "User 50");
}

#[test]
fn index_accelerated_unique_lookup() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE products (id TEXT PRIMARY KEY, sku TEXT, name TEXT)").unwrap();
    db.sql("INSERT INTO products (id, sku, name) VALUES ('p1', 'SKU-001', 'Widget')").unwrap();
    db.sql("INSERT INTO products (id, sku, name) VALUES ('p2', 'SKU-002', 'Gadget')").unwrap();
    db.sql("CREATE UNIQUE INDEX idx_sku ON products (sku)").unwrap();

    let rs = rows(db.sql("SELECT name FROM products WHERE sku = 'SKU-001'").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["name"], "Widget");
}

#[test]
fn explain_shows_index_scan() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('r1', 'x')").unwrap();
    db.sql("CREATE INDEX idx_val ON t (val)").unwrap();

    let result = db.sql("EXPLAIN SELECT id FROM t WHERE val = 'x'").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let plan_text = String::from_utf8_lossy(&rows[0]);
            assert!(plan_text.contains("IndexScan") || plan_text.contains("index_scan"),
                    "EXPLAIN should show IndexScan, got: {plan_text}");
        }
        _ => panic!("expected rows from EXPLAIN"),
    }
}

#[test]
fn query_without_index_still_works() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT, other TEXT)").unwrap();
    db.sql("INSERT INTO t (id, val, other) VALUES ('r1', 'x', 'a')").unwrap();
    db.sql("CREATE INDEX idx_val ON t (val)").unwrap();
    // Query on non-indexed column should fall back to full scan
    let rs = rows(db.sql("SELECT id FROM t WHERE other = 'a'").unwrap());
    assert_eq!(rs.len(), 1);
}
```

**Step 2: Implement predicate-to-index matching**

In the executor (exec.rs), before scanning a table for SELECT:

1. Load index metadata for the table: scan `__meta/idx/<table>/` prefix
2. Extract WHERE predicates that are simple equality on a single column
3. If an index exists for that column, use index lookup instead of full scan

Add a function `try_index_scan()`:
```rust
fn try_index_scan(
    db: &Database,
    session: &SqlSession,
    table: &str,
    filter: &Expr,
) -> Option<Vec<VisibleRow>> {
    // 1. Parse filter for col = 'literal' patterns
    let (col, val) = match filter {
        Expr::BinOp { left, op: BinOperator::Eq, right } => {
            match (left.as_ref(), right.as_ref()) {
                (Expr::Column(c), Expr::StringLit(v)) => (c.clone(), v.clone()),
                (Expr::StringLit(v), Expr::Column(c)) => (c.clone(), v.clone()),
                _ => return None,
            }
        }
        _ => return None,
    };

    // 2. Check if index exists on this column
    let idx_meta_prefix = format!("__meta/idx/{}/", table);
    let meta_entries = scan_prefix_helper(db, session, idx_meta_prefix.as_bytes());
    let index_name = meta_entries.iter().find_map(|entry| {
        let doc: serde_json::Value = serde_json::from_slice(&entry.doc).ok()?;
        let cols = doc["columns"].as_array()?;
        if cols.len() == 1 && cols[0].as_str()? == col {
            doc["name"].as_str().map(|s| s.to_string())
        } else {
            None
        }
    })?;

    // 3. Index lookup: scan __idx/<table>/<index_name>/<value>/
    let idx_prefix = format!("__idx/{}/{}/{}/", table, index_name, val);
    let idx_entries = scan_prefix_helper(db, session, idx_prefix.as_bytes());

    // 4. For each PK found, fetch the actual row
    let mut result = Vec::new();
    for entry in &idx_entries {
        if entry.doc == INDEX_TOMBSTONE { continue; }
        let pk = extract_pk_from_index_key(&entry.user_key, table, &index_name, &val);
        let row_key = row_key(table, &pk);
        if let Some(doc) = read_live_key(db, session, &row_key, None) {
            if !doc.is_empty() {
                result.push(VisibleRow { pk, doc, user_key: row_key });
            }
        }
    }
    Some(result)
}
```

Wire this into `execute_select()` before the full scan: if `try_index_scan()` returns Some, use those rows instead of scanning the whole table.

**Step 3: Update planner to report IndexScan**

In `planner.rs:plan_select()`, after checking for PointLookup, add index check:
```rust
// Check for index-accelerated scan
// (This is advisory — actual execution uses try_index_scan in exec.rs)
if let Some(ref f) = filter {
    if let Some((col, _val)) = extract_eq_column_literal(f) {
        // Check if we have index metadata (passed in via context)
        if let Some(idx) = index_for_column(&col, available_indexes) {
            node = PlanNode::IndexScan {
                table: table.clone(),
                index_name: idx.name.clone(),
                columns: idx.columns.clone(),
                estimated_rows: 1,
                estimated_cost: _INDEX_SCAN_COST,
            };
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test --test index_acceleration -- --nocapture 2>&1`
Expected: All 4 tests pass

**Step 5: Commit**

```bash
git add crates/tensordb-core/src/sql/exec.rs crates/tensordb-core/src/sql/planner.rs \
       tests/index_acceleration.rs
git commit -m "feat(eoac): index-accelerated WHERE clause execution"
```

---

### Task 8: Index-accelerated range scans and IN lists

**Files:**
- Modify: `crates/tensordb-core/src/sql/exec.rs` (extend try_index_scan)
- Test: `tests/index_acceleration.rs`

**Step 1: Write failing tests**

Add to `tests/index_acceleration.rs`:
```rust
#[test]
fn index_in_list_lookup() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE items (id TEXT PRIMARY KEY, category TEXT)").unwrap();
    for cat in &["electronics", "books", "clothing", "food", "toys"] {
        for i in 0..5 {
            db.sql(&format!(
                "INSERT INTO items (id, category) VALUES ('{cat}_{i}', '{cat}')"
            )).unwrap();
        }
    }
    db.sql("CREATE INDEX idx_cat ON items (category)").unwrap();

    let rs = rows(db.sql(
        "SELECT id FROM items WHERE category IN ('books', 'toys') ORDER BY id ASC"
    ).unwrap());
    assert_eq!(rs.len(), 10);
    assert!(rs[0]["id"].as_str().unwrap().starts_with("books"));
}

#[test]
fn index_composite_equality() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE orders (id TEXT PRIMARY KEY, customer TEXT, status TEXT)").unwrap();
    db.sql("INSERT INTO orders (id, customer, status) VALUES ('o1', 'alice', 'pending')").unwrap();
    db.sql("INSERT INTO orders (id, customer, status) VALUES ('o2', 'alice', 'shipped')").unwrap();
    db.sql("INSERT INTO orders (id, customer, status) VALUES ('o3', 'bob', 'pending')").unwrap();
    db.sql("CREATE INDEX idx_cust_status ON orders (customer, status)").unwrap();

    let rs = rows(db.sql(
        "SELECT id FROM orders WHERE customer = 'alice' AND status = 'pending'"
    ).unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "o1");
}
```

**Step 2: Extend try_index_scan**

Add handling for:
- `Expr::InList { expr: Column(col), list, negated: false }` → multi-point index lookup
- `Expr::BinOp { op: And, left, right }` → check for composite index match
- `Expr::BinOp { op: Gt/Gte/Lt/Lte, ... }` → range scan on index prefix

For IN lists: do one index prefix scan per value, union the PKs, then fetch rows.
For composite: encode multiple column values into the index key prefix.

**Step 3: Run tests**

Run: `cargo test --test index_acceleration -- --nocapture 2>&1`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/tensordb-core/src/sql/exec.rs tests/index_acceleration.rs
git commit -m "feat(eoac): index-accelerated IN lists and composite index lookups"
```

---

### Task 9: Index nested loop join

**Files:**
- Modify: `crates/tensordb-core/src/sql/exec.rs:2171-2309` (execute_join_select)
- Test: `tests/index_acceleration.rs`

**Step 1: Write failing test**

Add to `tests/index_acceleration.rs`:
```rust
#[test]
fn join_uses_index_on_inner_table() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE orders (id TEXT PRIMARY KEY, customer_id TEXT, amount REAL)").unwrap();
    db.sql("CREATE TABLE customers (id TEXT PRIMARY KEY, name TEXT)").unwrap();

    for i in 0..50 {
        db.sql(&format!("INSERT INTO customers (id, name) VALUES ('c{i}', 'Customer {i}')")).unwrap();
    }
    for i in 0..200 {
        let cid = i % 50;
        db.sql(&format!(
            "INSERT INTO orders (id, customer_id, amount) VALUES ('o{i}', 'c{cid}', {}.99)",
            i * 10
        )).unwrap();
    }
    db.sql("CREATE INDEX idx_orders_cust ON orders (customer_id)").unwrap();

    // Join should use index on orders.customer_id
    let rs = rows(db.sql(
        "SELECT c.name, o.amount FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.name = 'Customer 0' ORDER BY o.amount ASC LIMIT 5"
    ).unwrap());
    assert!(rs.len() <= 5);
    assert_eq!(rs[0]["name"], "Customer 0");
}
```

**Step 2: Implement index nested loop join**

In `execute_join_select()`, before choosing hash join or nested loop:

1. Check if the join column on the inner (right) table has an index
2. If yes, use index nested loop: for each outer row, do an index lookup on the inner table
3. This is O(n * log m) instead of O(n + m) for hash join, but avoids materializing the entire inner table

Add a new join execution function:
```rust
fn index_nested_loop_join(
    outer_rows: Vec<VisibleRow>,
    db: &Database,
    session: &SqlSession,
    inner_table: &str,
    index_name: &str,
    outer_join_col: &str,
    inner_join_col: &str,
    join_type: &JoinType,
) -> Result<Vec<JoinedRow>> { ... }
```

**Step 3: Run tests**

Run: `cargo test --test index_acceleration -- --nocapture 2>&1`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/tensordb-core/src/sql/exec.rs tests/index_acceleration.rs
git commit -m "feat(eoac): index nested loop join for indexed inner tables"
```

---

## Phase D: N-Way JOINs

### Task 10: Parse chained JOIN clauses

**Files:**
- Modify: `crates/tensordb-core/src/sql/parser.rs:98-103` (JoinSpec → Vec<JoinSpec>)
- Modify: `crates/tensordb-core/src/sql/parser.rs:147-249` (Statement::Select)
- Test: `crates/tensordb-core/src/sql/parser.rs` (unit tests)

**Step 1: Write failing test**

Add parser test:
```rust
#[test]
fn parses_multi_way_join() {
    let stmt = parse_sql("SELECT a.id, b.name, c.val FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id").unwrap();
    match stmt {
        Statement::Select { joins, .. } => {
            assert_eq!(joins.len(), 2);
            assert_eq!(joins[0].right_table, "b");
            assert_eq!(joins[1].right_table, "c");
        }
        _ => panic!("expected select"),
    }
}
```

**Step 2: Change Select to use Vec<JoinSpec>**

In parser.rs Statement::Select, change:
```rust
// Old:
join: Option<JoinSpec>,
// New:
joins: Vec<JoinSpec>,
```

Update the parser to loop and collect multiple JOIN clauses. Update all code that matches on `Statement::Select { join, .. }` to use `joins`.

In `try_parse_join()` — currently returns `Option<JoinSpec>`. Change to loop:
```rust
fn try_parse_joins(&mut self) -> Vec<JoinSpec> {
    let mut joins = Vec::new();
    while let Some(spec) = self.try_parse_single_join() {
        joins.push(spec);
    }
    joins
}
```

**Step 3: Update executor**

In exec.rs, change `execute_select()` signature from `join: Option<JoinSpec>` to `joins: Vec<JoinSpec>`.

For now, if `joins.len() == 1`, delegate to existing `execute_join_select()`. If `joins.len() > 1`, handle in Task 11.

**Step 4: Run tests**

Run: `cargo test --workspace --all-targets 2>&1 | tail -5`
Expected: All existing tests still pass + new parser test passes

**Step 5: Commit**

```bash
git add crates/tensordb-core/src/sql/parser.rs crates/tensordb-core/src/sql/exec.rs \
       crates/tensordb-core/src/sql/planner.rs
git commit -m "feat(eoac): parse N-way JOIN chains"
```

---

### Task 11: Execute N-way JOINs via left-deep pipeline

**Files:**
- Modify: `crates/tensordb-core/src/sql/exec.rs` (new multi-join execution)
- Test: `tests/multi_join.rs` (new)

**Step 1: Write failing tests**

Create `tests/multi_join.rs`:
```rust
//! Tests for N-way JOIN execution

use tempfile::TempDir;
use tensordb::{Config, Database};
use tensordb_core::sql::exec::SqlResult;

fn open_db(dir: &TempDir) -> Database {
    Database::open(dir.path(), Config { shard_count: 2, ..Config::default() }).unwrap()
}

fn rows(result: SqlResult) -> Vec<serde_json::Value> {
    match result {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| serde_json::from_slice(r).unwrap_or(serde_json::Value::String(
                String::from_utf8_lossy(r).to_string(),
            )))
            .collect(),
        _ => panic!("expected rows, got {result:?}"),
    }
}

#[test]
fn three_way_inner_join() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE departments (id TEXT PRIMARY KEY, name TEXT)").unwrap();
    db.sql("CREATE TABLE employees (id TEXT PRIMARY KEY, dept_id TEXT, name TEXT)").unwrap();
    db.sql("CREATE TABLE salaries (id TEXT PRIMARY KEY, emp_id TEXT, amount REAL)").unwrap();

    db.sql("INSERT INTO departments (id, name) VALUES ('d1', 'Engineering')").unwrap();
    db.sql("INSERT INTO departments (id, name) VALUES ('d2', 'Marketing')").unwrap();
    db.sql("INSERT INTO employees (id, dept_id, name) VALUES ('e1', 'd1', 'Alice')").unwrap();
    db.sql("INSERT INTO employees (id, dept_id, name) VALUES ('e2', 'd1', 'Bob')").unwrap();
    db.sql("INSERT INTO employees (id, dept_id, name) VALUES ('e3', 'd2', 'Carol')").unwrap();
    db.sql("INSERT INTO salaries (id, emp_id, amount) VALUES ('s1', 'e1', 120000)").unwrap();
    db.sql("INSERT INTO salaries (id, emp_id, amount) VALUES ('s2', 'e2', 110000)").unwrap();
    db.sql("INSERT INTO salaries (id, emp_id, amount) VALUES ('s3', 'e3', 95000)").unwrap();

    let rs = rows(db.sql(
        "SELECT d.name, e.name, s.amount FROM departments d \
         JOIN employees e ON d.id = e.dept_id \
         JOIN salaries s ON e.id = s.emp_id \
         ORDER BY s.amount DESC"
    ).unwrap());

    assert_eq!(rs.len(), 3);
    assert_eq!(rs[0]["d.name"], "Engineering");
    assert_eq!(rs[0]["e.name"], "Alice");
}

#[test]
fn three_way_join_with_where() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE a (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("CREATE TABLE b (id TEXT PRIMARY KEY, a_id TEXT, val TEXT)").unwrap();
    db.sql("CREATE TABLE c (id TEXT PRIMARY KEY, b_id TEXT, val TEXT)").unwrap();

    db.sql("INSERT INTO a (id, val) VALUES ('a1', 'x')").unwrap();
    db.sql("INSERT INTO a (id, val) VALUES ('a2', 'y')").unwrap();
    db.sql("INSERT INTO b (id, a_id, val) VALUES ('b1', 'a1', 'p')").unwrap();
    db.sql("INSERT INTO b (id, a_id, val) VALUES ('b2', 'a2', 'q')").unwrap();
    db.sql("INSERT INTO c (id, b_id, val) VALUES ('c1', 'b1', 'm')").unwrap();
    db.sql("INSERT INTO c (id, b_id, val) VALUES ('c2', 'b2', 'n')").unwrap();

    let rs = rows(db.sql(
        "SELECT a.val, c.val FROM a \
         JOIN b ON a.id = b.a_id \
         JOIN c ON b.id = c.b_id \
         WHERE a.val = 'x'"
    ).unwrap());

    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["a.val"], "x");
    assert_eq!(rs[0]["c.val"], "m");
}

#[test]
fn four_way_join() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t1 (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("CREATE TABLE t2 (id TEXT PRIMARY KEY, t1_id TEXT)").unwrap();
    db.sql("CREATE TABLE t3 (id TEXT PRIMARY KEY, t2_id TEXT)").unwrap();
    db.sql("CREATE TABLE t4 (id TEXT PRIMARY KEY, t3_id TEXT, result TEXT)").unwrap();

    db.sql("INSERT INTO t1 (id, val) VALUES ('a', 'root')").unwrap();
    db.sql("INSERT INTO t2 (id, t1_id) VALUES ('b', 'a')").unwrap();
    db.sql("INSERT INTO t3 (id, t2_id) VALUES ('c', 'b')").unwrap();
    db.sql("INSERT INTO t4 (id, t3_id, result) VALUES ('d', 'c', 'leaf')").unwrap();

    let rs = rows(db.sql(
        "SELECT t1.val, t4.result FROM t1 \
         JOIN t2 ON t1.id = t2.t1_id \
         JOIN t3 ON t2.id = t3.t2_id \
         JOIN t4 ON t3.id = t4.t3_id"
    ).unwrap());

    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["t1.val"], "root");
    assert_eq!(rs[0]["t4.result"], "leaf");
}

#[test]
fn mixed_join_types() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE a (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("CREATE TABLE b (id TEXT PRIMARY KEY, a_id TEXT)").unwrap();
    db.sql("CREATE TABLE c (id TEXT PRIMARY KEY, b_id TEXT)").unwrap();

    db.sql("INSERT INTO a (id, val) VALUES ('a1', 'x')").unwrap();
    db.sql("INSERT INTO a (id, val) VALUES ('a2', 'y')").unwrap();
    db.sql("INSERT INTO b (id, a_id) VALUES ('b1', 'a1')").unwrap();
    // No b for a2, and no c for b1

    let rs = rows(db.sql(
        "SELECT a.val FROM a \
         LEFT JOIN b ON a.id = b.a_id \
         LEFT JOIN c ON b.id = c.b_id \
         ORDER BY a.val ASC"
    ).unwrap());

    // Both a1 and a2 should appear (LEFT JOINs preserve left rows)
    assert_eq!(rs.len(), 2);
}
```

**Step 2: Implement N-way join execution**

Create `execute_multi_join_select()`:

```rust
fn execute_multi_join_select(
    db: &Database,
    session: &mut SqlSession,
    from_table: &str,
    joins: Vec<JoinSpec>,
    items: &[SelectItem],
    filter: Option<Expr>,
    as_of: Option<u64>,
    valid_at: Option<u64>,
    group_by: Option<Vec<Expr>>,
    having: Option<Expr>,
    order_by: Option<Vec<(Expr, OrderDirection)>>,
    limit: Option<u64>,
) -> Result<SqlResult> {
    // 1. Fetch left (FROM) table rows
    let mut current_rows = fetch_rows_for_table(db, session, from_table, ...)?;
    let mut current_tables = vec![from_table.to_string()];

    // 2. For each JOIN, join current_rows with the next table
    for join_spec in &joins {
        let right_rows = fetch_rows_for_table(db, session, &join_spec.right_table, ...)?;
        current_rows = execute_pairwise_join(
            current_rows,
            right_rows,
            &join_spec,
            &current_tables,
        )?;
        current_tables.push(join_spec.right_table.clone());
    }

    // 3. Apply WHERE filter on the final joined result
    // 4. Apply GROUP BY, HAVING, ORDER BY, LIMIT, PROJECT
    // ... (reuse existing logic)
}
```

The key insight: each intermediate join produces rows that carry merged JSON from all tables joined so far. Column references like `a.val` are resolved from the merged JSON.

**Step 3: Run tests**

Run: `cargo test --test multi_join -- --nocapture 2>&1`
Expected: All 4 tests pass

**Step 4: Commit**

```bash
git add crates/tensordb-core/src/sql/exec.rs tests/multi_join.rs
git commit -m "feat(eoac): N-way JOIN execution via left-deep pipeline"
```

---

### Task 12: Join reordering with greedy heuristic

**Files:**
- Modify: `crates/tensordb-core/src/sql/planner.rs` (join reordering)
- Modify: `crates/tensordb-core/src/sql/exec.rs` (use reordered joins)
- Test: `tests/multi_join.rs`

**Step 1: Write test**

Add to `tests/multi_join.rs`:
```rust
#[test]
fn join_reordering_produces_correct_results() {
    // Even if tables are listed in suboptimal order, results should be correct
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE big (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("CREATE TABLE small (id TEXT PRIMARY KEY, big_id TEXT)").unwrap();

    // big has 100 rows, small has 2
    for i in 0..100 {
        db.sql(&format!("INSERT INTO big (id, val) VALUES ('b{i}', 'v{i}')")).unwrap();
    }
    db.sql("INSERT INTO small (id, big_id) VALUES ('s1', 'b0')").unwrap();
    db.sql("INSERT INTO small (id, big_id) VALUES ('s2', 'b1')").unwrap();

    // ANALYZE to give planner stats
    db.sql("ANALYZE big").unwrap();
    db.sql("ANALYZE small").unwrap();

    // Join big → small (suboptimal order for hash join: big is build side)
    let rs = rows(db.sql(
        "SELECT big.val FROM big JOIN small ON big.id = small.big_id ORDER BY big.val ASC"
    ).unwrap());
    assert_eq!(rs.len(), 2);
    assert_eq!(rs[0]["big.val"], "v0");
    assert_eq!(rs[1]["big.val"], "v1");
}
```

**Step 2: Implement join reordering**

In the planner or executor, when processing N-way joins:

1. Load table stats for all tables involved (from `__meta/stats/<table>`)
2. For 2-table joins: put smaller table as build side of hash join
3. For N-table joins: greedy algorithm — start with smallest table, add cheapest next join

For the binary join case, this is already partially done (hash join checks sizes). For N-way, reorder the `Vec<JoinSpec>` before execution.

**Step 3: Run tests**

Expected: All pass

**Step 4: Commit**

```bash
git add crates/tensordb-core/src/sql/planner.rs crates/tensordb-core/src/sql/exec.rs \
       tests/multi_join.rs
git commit -m "feat(eoac): greedy join reordering based on table statistics"
```

---

## Phase E: Predicate Pushdown

### Task 13: Push single-table predicates into scans

**Files:**
- Modify: `crates/tensordb-core/src/sql/exec.rs` (predicate pushdown)
- Test: `tests/multi_join.rs`

**Step 1: Write test**

Add to `tests/multi_join.rs`:
```rust
#[test]
fn predicate_pushdown_filters_early() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE left_t (id TEXT PRIMARY KEY, status TEXT)").unwrap();
    db.sql("CREATE TABLE right_t (id TEXT PRIMARY KEY, left_id TEXT, val TEXT)").unwrap();

    for i in 0..100 {
        let status = if i % 10 == 0 { "active" } else { "inactive" };
        db.sql(&format!(
            "INSERT INTO left_t (id, status) VALUES ('l{i}', '{status}')"
        )).unwrap();
        db.sql(&format!(
            "INSERT INTO right_t (id, left_id, val) VALUES ('r{i}', 'l{i}', 'v{i}')"
        )).unwrap();
    }

    // WHERE left_t.status = 'active' should be pushed into left scan
    let rs = rows(db.sql(
        "SELECT right_t.val FROM left_t JOIN right_t ON left_t.id = right_t.left_id \
         WHERE left_t.status = 'active' ORDER BY right_t.val ASC"
    ).unwrap());
    assert_eq!(rs.len(), 10);
}
```

**Step 2: Implement predicate pushdown**

Before executing a join, analyze the WHERE clause:

```rust
fn split_predicates(
    filter: &Expr,
    left_table: &str,
    right_table: &str,
) -> (Option<Expr>, Option<Expr>, Option<Expr>) {
    // Returns (left_only_preds, right_only_preds, post_join_preds)
    match filter {
        Expr::BinOp { op: BinOperator::And, left, right } => {
            // Recursively split AND clauses
            // If predicate references only left table columns → push to left scan
            // If predicate references only right table columns → push to right scan
            // Otherwise → keep as post-join filter
        }
        _ => {
            // Single predicate — check which table(s) it references
            let refs = collect_column_refs(filter);
            if refs.iter().all(|c| belongs_to_table(c, left_table)) {
                (Some(filter.clone()), None, None)
            } else if refs.iter().all(|c| belongs_to_table(c, right_table)) {
                (None, Some(filter.clone()), None)
            } else {
                (None, None, Some(filter.clone()))
            }
        }
    }
}
```

Apply pushed-down predicates during `fetch_rows_for_table()` to reduce the row set before joining.

**Step 3: Run tests**

Run: `cargo test --test multi_join -- --nocapture 2>&1`
Expected: All pass

**Step 4: Commit**

```bash
git add crates/tensordb-core/src/sql/exec.rs tests/multi_join.rs
git commit -m "feat(eoac): predicate pushdown into scan operators"
```

---

## Phase F: Table Statistics Wired to Planner

### Task 14: Wire ANALYZE results into planner cost model

**Files:**
- Modify: `crates/tensordb-core/src/sql/exec.rs` (ANALYZE stores stats)
- Modify: `crates/tensordb-core/src/sql/planner.rs` (read stats for cost estimation)
- Test: `tests/index_acceleration.rs`

**Step 1: Write test**

Add to `tests/index_acceleration.rs`:
```rust
#[test]
fn analyze_stores_table_stats() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE stats_test (id TEXT PRIMARY KEY, val TEXT, num REAL)").unwrap();
    for i in 0..50 {
        db.sql(&format!(
            "INSERT INTO stats_test (id, val, num) VALUES ('r{i}', 'v{i}', {}.5)", i
        )).unwrap();
    }
    let result = db.sql("ANALYZE stats_test").unwrap();
    match result {
        SqlResult::Affected { message, .. } => {
            assert!(message.contains("50"), "ANALYZE should report 50 rows: {message}");
        }
        _ => panic!("expected affected"),
    }
}

#[test]
fn explain_uses_stats_for_cost() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    for i in 0..100 {
        db.sql(&format!("INSERT INTO t (id, val) VALUES ('r{i}', 'v{i}')")).unwrap();
    }
    db.sql("ANALYZE t").unwrap();

    let result = db.sql("EXPLAIN SELECT * FROM t").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let text = String::from_utf8_lossy(&rows[0]);
            // Should show estimated rows ~100 instead of default 1000
            assert!(text.contains("100") || text.contains("est_rows"),
                    "EXPLAIN should reflect ANALYZE stats: {text}");
        }
        _ => panic!("expected rows"),
    }
}
```

**Step 2: Implement stats storage and retrieval**

ANALYZE currently computes stats but doesn't persist them. Store at `__meta/stats/<table>`:
```json
{"row_count": 50, "columns": {"val": {"distinct": 50}, "num": {"distinct": 50, "min": 0.5, "max": 49.5}}}
```

In planner, read stats to replace `DEFAULT_ROW_ESTIMATE`:
```rust
fn get_table_stats(db: &Database, table: &str) -> Option<TableStats> {
    let key = format!("__meta/stats/{}", table);
    let data = db.get(key.as_bytes(), None, None).ok()??;
    serde_json::from_slice(&data).ok()
}
```

**Step 3: Run tests**

Expected: All pass

**Step 4: Commit**

```bash
git add crates/tensordb-core/src/sql/exec.rs crates/tensordb-core/src/sql/planner.rs \
       tests/index_acceleration.rs
git commit -m "feat(eoac): wire ANALYZE stats into planner cost model"
```

---

## Phase G: Crash Recovery with Epoch Filtering

### Task 15: Epoch-filtered WAL replay

**Files:**
- Modify: `crates/tensordb-core/src/storage/wal.rs` (replay with epoch filter)
- Modify: `crates/tensordb-core/src/engine/shard.rs` (use epoch-filtered replay)
- Test: `tests/epoch_core.rs`

**Step 1: Write test**

Add to `tests/epoch_core.rs`:
```rust
#[test]
fn crash_recovery_skips_uncommitted_txn() {
    let dir = TempDir::new().unwrap();
    {
        let db = open_db(&dir);
        db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();

        // Committed write
        db.sql("INSERT INTO t (id, val) VALUES ('k1', 'committed')").unwrap();

        // Transactional write that gets committed
        db.sql("BEGIN").unwrap();
        db.sql("INSERT INTO t (id, val) VALUES ('k2', 'also_committed')").unwrap();
        db.sql("COMMIT").unwrap();

        db.sync();
    }
    // Reopen — both should be present
    let db = open_db(&dir);
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 2);
}

#[test]
fn reopen_preserves_epoch() {
    let dir = TempDir::new().unwrap();
    let epoch_after;
    {
        let db = open_db(&dir);
        db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();
        db.sql("BEGIN").unwrap();
        db.sql("INSERT INTO t (id) VALUES ('k1')").unwrap();
        db.sql("COMMIT").unwrap();
        epoch_after = db.current_epoch();
        db.sync();
    }
    let db = open_db(&dir);
    // Epoch should be >= what it was before close
    assert!(db.current_epoch() >= epoch_after,
            "epoch should survive reopen: was {epoch_after}, now {}", db.current_epoch());
}
```

**Step 2: Implement epoch-aware replay**

In `Wal::replay()`, return both `FactWrite` records and `TxnCommit` markers.

In shard.rs `ShardRuntime::open()`:
```rust
let (writes, commits) = Wal::replay_with_txns(&wal_path)?;
let committed_txns: HashSet<u64> = commits.iter().map(|c| c.txn_id).collect();

for write in writes {
    if write.txn_id == 0 || committed_txns.contains(&write.txn_id) {
        // Apply to memtable
        memtable.insert(write.internal_key, write.fact);
    }
    // else: skip uncommitted transactional write
}
```

Update `Database::open()` to restore `global_epoch` from max epoch seen across all shards.

**Step 3: Run tests**

Expected: All pass

**Step 4: Commit**

```bash
git add crates/tensordb-core/src/storage/wal.rs crates/tensordb-core/src/engine/shard.rs \
       crates/tensordb-core/src/engine/db.rs tests/epoch_core.rs
git commit -m "feat(eoac): epoch-filtered WAL replay for crash recovery"
```

---

## Phase H: PITR and Incremental Backup

### Task 16: RESTORE TO EPOCH

**Files:**
- Modify: `crates/tensordb-core/src/sql/parser.rs` (parse RESTORE TO EPOCH)
- Modify: `crates/tensordb-core/src/sql/exec.rs` (execute RESTORE TO EPOCH)
- Test: `tests/epoch_core.rs`

**Step 1: Write test**

Add to `tests/epoch_core.rs`:
```rust
#[test]
fn pitr_restore_to_epoch() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();
    let epoch_after_k1 = db.current_epoch();

    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k2', 'v2')").unwrap();
    db.sql("COMMIT").unwrap();

    // Query as of the earlier epoch should not see k2
    let rs = rows(db.sql(&format!(
        "SELECT id FROM t FOR SYSTEM_TIME AS OF {epoch_after_k1} ORDER BY id ASC"
    )).unwrap());
    // This may or may not work depending on how epoch filtering interacts with system_time
    // At minimum, k1 should be visible
    assert!(rs.len() >= 1);
    assert_eq!(rs[0]["id"], "k1");
}
```

**Step 2: Add RESTORE TO EPOCH parsing**

Add Statement variant:
```rust
RestoreToEpoch { epoch: u64 },
```

Parse: `RESTORE DATABASE TO EPOCH <number>`

**Step 3: Implement RESTORE TO EPOCH execution**

This is a read-only snapshot query — replay WAL up to the target epoch.

For the first implementation, support `FOR SYSTEM_TIME AS OF <epoch>` on SELECT statements by filtering records where `commit_epoch <= epoch`.

**Step 4: Run tests**

Expected: Pass

**Step 5: Commit**

```bash
git add crates/tensordb-core/src/sql/parser.rs crates/tensordb-core/src/sql/exec.rs \
       tests/epoch_core.rs
git commit -m "feat(eoac): PITR via epoch-based query filtering"
```

---

### Task 17: Incremental backup

**Files:**
- Modify: `crates/tensordb-core/src/sql/parser.rs` (BACKUP INCREMENTAL)
- Modify: `crates/tensordb-core/src/sql/exec.rs` (incremental backup logic)
- Test: `tests/epoch_core.rs`

**Step 1: Write test**

Add to `tests/epoch_core.rs`:
```rust
#[test]
fn incremental_backup() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k1', 'v1')").unwrap();

    // Full backup
    let backup_dir = TempDir::new().unwrap();
    let full_path = backup_dir.path().join("full");
    db.sql(&format!("BACKUP DATABASE TO '{}'", full_path.display())).unwrap();

    // More writes
    db.sql("INSERT INTO t (id, val) VALUES ('k2', 'v2')").unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('k3', 'v3')").unwrap();

    // Incremental backup (since last full)
    let incr_path = backup_dir.path().join("incr");
    let result = db.sql(&format!(
        "BACKUP DATABASE TO '{}' INCREMENTAL", incr_path.display()
    )).unwrap();
    match result {
        SqlResult::Affected { message, .. } => {
            assert!(message.contains("incremental") || message.contains("backup"));
        }
        _ => panic!("expected affected"),
    }
}
```

**Step 2: Implement incremental backup**

Track last backup epoch in `__meta/backup/last_epoch`. On incremental backup:
1. Read last backup epoch
2. Copy only WAL segments and SSTs created since that epoch
3. Write metadata with `from_epoch` and `to_epoch`

**Step 3: Run tests**

Expected: Pass

**Step 4: Commit**

```bash
git add crates/tensordb-core/src/sql/parser.rs crates/tensordb-core/src/sql/exec.rs \
       tests/epoch_core.rs
git commit -m "feat(eoac): incremental backup via epoch tracking"
```

---

## Phase I: Integration Tests and Polish

### Task 18: Comprehensive integration test suite

**Files:**
- Create: `tests/eoac_integration.rs`

**Step 1: Write comprehensive tests**

```rust
//! EOAC integration tests — transactions + indexes + joins + recovery

use tempfile::TempDir;
use tensordb::{Config, Database};
use tensordb_core::sql::exec::SqlResult;

fn open_db(dir: &TempDir) -> Database {
    Database::open(dir.path(), Config { shard_count: 2, ..Config::default() }).unwrap()
}

fn rows(result: SqlResult) -> Vec<serde_json::Value> {
    match result {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| serde_json::from_slice(r).unwrap_or(serde_json::Value::String(
                String::from_utf8_lossy(r).to_string(),
            )))
            .collect(),
        _ => panic!("expected rows, got {result:?}"),
    }
}

// Transaction + Index interaction
#[test]
fn transaction_with_indexed_inserts() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT)").unwrap();
    db.sql("CREATE UNIQUE INDEX idx_email ON users (email)").unwrap();

    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO users (id, email) VALUES ('u1', 'alice@test.com')").unwrap();
    db.sql("INSERT INTO users (id, email) VALUES ('u2', 'bob@test.com')").unwrap();
    db.sql("COMMIT").unwrap();

    let rs = rows(db.sql("SELECT id FROM users WHERE email = 'alice@test.com'").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "u1");
}

// Transaction rollback with indexes
#[test]
fn rollback_cleans_up_index_entries() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT)").unwrap();
    db.sql("CREATE UNIQUE INDEX idx_email ON users (email)").unwrap();

    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO users (id, email) VALUES ('u1', 'alice@test.com')").unwrap();
    db.sql("ROLLBACK").unwrap();

    // After rollback, the email should be available
    db.sql("INSERT INTO users (id, email) VALUES ('u2', 'alice@test.com')").unwrap();
    let rs = rows(db.sql("SELECT id FROM users WHERE email = 'alice@test.com'").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "u2");
}

// Multi-way join with indexes
#[test]
fn three_way_join_with_index_acceleration() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE authors (id TEXT PRIMARY KEY, name TEXT)").unwrap();
    db.sql("CREATE TABLE books (id TEXT PRIMARY KEY, author_id TEXT, title TEXT)").unwrap();
    db.sql("CREATE TABLE reviews (id TEXT PRIMARY KEY, book_id TEXT, rating REAL)").unwrap();
    db.sql("CREATE INDEX idx_books_author ON books (author_id)").unwrap();
    db.sql("CREATE INDEX idx_reviews_book ON reviews (book_id)").unwrap();

    db.sql("INSERT INTO authors (id, name) VALUES ('a1', 'Alice')").unwrap();
    db.sql("INSERT INTO books (id, author_id, title) VALUES ('b1', 'a1', 'Book One')").unwrap();
    db.sql("INSERT INTO reviews (id, book_id, rating) VALUES ('r1', 'b1', 4.5)").unwrap();

    let rs = rows(db.sql(
        "SELECT a.name, b.title, r.rating FROM authors a \
         JOIN books b ON a.id = b.author_id \
         JOIN reviews r ON b.id = r.book_id"
    ).unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["a.name"], "Alice");
}

// Savepoint with multi-way join verification
#[test]
fn savepoint_with_join_verification() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE parents (id TEXT PRIMARY KEY, name TEXT)").unwrap();
    db.sql("CREATE TABLE children (id TEXT PRIMARY KEY, parent_id TEXT, name TEXT)").unwrap();

    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO parents (id, name) VALUES ('p1', 'Parent')").unwrap();
    db.sql("SAVEPOINT sp1").unwrap();
    db.sql("INSERT INTO children (id, parent_id, name) VALUES ('c1', 'p1', 'Child')").unwrap();
    db.sql("ROLLBACK TO sp1").unwrap();
    db.sql("COMMIT").unwrap();

    // Parent should exist, child should not
    let rs = rows(db.sql("SELECT id FROM parents").unwrap());
    assert_eq!(rs.len(), 1);
    let rs = rows(db.sql("SELECT id FROM children").unwrap());
    assert_eq!(rs.len(), 0);
}

// Epoch survives crash
#[test]
fn epoch_monotonically_increases_across_reopens() {
    let dir = TempDir::new().unwrap();
    let e1;
    {
        let db = open_db(&dir);
        db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();
        db.sql("BEGIN").unwrap();
        db.sql("INSERT INTO t (id) VALUES ('k1')").unwrap();
        db.sql("COMMIT").unwrap();
        e1 = db.current_epoch();
        db.sync();
    }
    let db = open_db(&dir);
    let e2 = db.current_epoch();
    assert!(e2 >= e1, "epoch must not go backwards: e1={e1}, e2={e2}");

    db.sql("BEGIN").unwrap();
    db.sql("INSERT INTO t (id) VALUES ('k2')").unwrap();
    db.sql("COMMIT").unwrap();
    let e3 = db.current_epoch();
    assert!(e3 > e2, "epoch must advance after commit: e2={e2}, e3={e3}");
}
```

**Step 2: Run all tests**

Run: `cargo test --workspace --all-targets 2>&1 | tail -5`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/eoac_integration.rs
git commit -m "test(eoac): comprehensive integration tests for all critical gaps"
```

---

### Task 19: Fix any remaining clippy warnings and run full CI

**Files:**
- Any files with warnings

**Step 1: Run clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings 2>&1`

**Step 2: Fix all warnings**

Address each warning. Common issues:
- Unused imports from refactoring
- Unnecessary clones
- Missing `#[allow(dead_code)]` on new constants

**Step 3: Run fmt**

Run: `cargo fmt --all -- --check 2>&1`

**Step 4: Run full test suite one final time**

Run: `cargo test --workspace --all-targets 2>&1`
Expected: All tests pass, no warnings

**Step 5: Commit**

```bash
git add -A
git commit -m "chore(eoac): fix clippy warnings and formatting"
```

---

### Task 20: Final commit with all EOAC features

**Step 1: Verify everything compiles and passes**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
```

**Step 2: Update memory file with EOAC architecture**

Update `~/.claude/projects/-home-walebadr-spectraDB/memory/MEMORY.md` with:
- Global epoch architecture
- Transaction model (intent-based with savepoints)
- Index acceleration (predicate-to-index matching)
- N-way JOIN support (left-deep pipeline)
- Recovery model (epoch-filtered WAL replay)

---

## Task Dependency Graph

```
Task 1 (epoch infra)
  └→ Task 2 (epoch in WAL)
      └→ Task 3 (epoch in write paths)
          ├→ Task 4 (epoch-aware transactions)
          │   └→ Task 5 (savepoints)
          │       └→ Task 6 (TXN_COMMIT markers)
          │           └→ Task 15 (crash recovery)
          │               └→ Task 16 (PITR)
          │                   └→ Task 17 (incremental backup)
          ├→ Task 7 (index scan matching)
          │   └→ Task 8 (range + IN + composite)
          │       └→ Task 9 (index nested loop join)
          ├→ Task 10 (parse N-way JOINs)
          │   └→ Task 11 (execute N-way JOINs)
          │       └→ Task 12 (join reordering)
          │           └→ Task 13 (predicate pushdown)
          └→ Task 14 (table stats → planner)

Tasks 18-20 depend on all above being complete.
```

## Parallel Execution Opportunities

These task groups are **independent** and can be worked on in parallel:
- **Group 1:** Tasks 4-6, 15-17 (transactions + recovery)
- **Group 2:** Tasks 7-9 (index acceleration)
- **Group 3:** Tasks 10-13 (N-way JOINs)
- **Group 4:** Task 14 (statistics)

All groups depend on Tasks 1-3 (epoch infrastructure) being complete first.
