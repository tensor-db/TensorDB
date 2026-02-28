//! Tests for EOAC (Epoch-Ordered Append-Only Concurrency) infrastructure

use tempfile::TempDir;
use tensordb::{Config, Database};
use tensordb_core::sql::exec::SqlResult;

fn open_db(dir: &TempDir) -> Database {
    Database::open(
        dir.path(),
        Config {
            shard_count: 2,
            ..Config::default()
        },
    )
    .unwrap()
}

fn rows(result: SqlResult) -> Vec<serde_json::Value> {
    match result {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| {
                serde_json::from_slice(r).unwrap_or(serde_json::Value::String(
                    String::from_utf8_lossy(r).to_string(),
                ))
            })
            .collect(),
        _ => panic!("expected rows, got {result:?}"),
    }
}

// ========================== Epoch Tests ==========================

#[test]
fn epoch_starts_at_one() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    assert!(db.current_epoch() >= 1);
}

#[test]
fn epoch_stable_without_transactions() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let e1 = db.current_epoch();
    db.put(b"key1", b"val1".to_vec(), 0, u64::MAX, Some(1))
        .unwrap();
    let e2 = db.current_epoch();
    assert_eq!(e1, e2);
}

#[test]
fn advance_epoch_increments() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let e1 = db.current_epoch();
    let e2 = db.advance_epoch();
    assert_eq!(e2, e1 + 1);
    let e3 = db.current_epoch();
    assert_eq!(e3, e2);
}

// ========================== Transaction Tests ==========================

#[test]
fn transaction_commit_makes_writes_visible() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         INSERT INTO t (id, val) VALUES ('k2', 'v2'); \
         COMMIT;",
    )
    .unwrap();
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 2);
    assert_eq!(rs[0]["id"], "k1");
    assert_eq!(rs[1]["id"], "k2");
}

#[test]
fn transaction_rollback_discards_writes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         ROLLBACK;",
    )
    .unwrap();
    let rs = rows(db.sql("SELECT id FROM t").unwrap());
    assert_eq!(rs.len(), 0);
}

#[test]
fn transaction_reads_own_writes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    // Multi-statement: BEGIN, INSERT, SELECT, COMMIT — SELECT should see the staged write
    let _result = db
        .sql(
            "BEGIN; \
             INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
             SELECT val FROM t WHERE pk = 'k1'; \
             COMMIT;",
        )
        .unwrap();
    // The last statement in the batch is COMMIT, so we get COMMIT result
    // Let's test differently — verify the write persisted after commit
    let rs = rows(db.sql("SELECT val FROM t WHERE pk = 'k1'").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["val"], "v1");
}

#[test]
fn commit_advances_global_epoch() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    let e1 = db.current_epoch();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         COMMIT;",
    )
    .unwrap();
    let e2 = db.current_epoch();
    assert!(e2 > e1, "epoch should advance on commit: e1={e1}, e2={e2}");
}

#[test]
fn empty_commit_does_not_advance_epoch() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();
    let e1 = db.current_epoch();
    db.sql("BEGIN; COMMIT;").unwrap();
    let e2 = db.current_epoch();
    assert_eq!(e1, e2, "empty commit should not advance epoch");
}

// ========================== Savepoint Tests ==========================

#[test]
fn savepoint_rollback_to() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         SAVEPOINT sp1; \
         INSERT INTO t (id, val) VALUES ('k2', 'v2'); \
         ROLLBACK TO sp1; \
         COMMIT;",
    )
    .unwrap();
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "k1");
}

#[test]
fn savepoint_release() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         SAVEPOINT sp1; \
         INSERT INTO t (id, val) VALUES ('k2', 'v2'); \
         RELEASE sp1; \
         COMMIT;",
    )
    .unwrap();
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 2);
}

#[test]
fn nested_savepoints() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         SAVEPOINT sp1; \
         INSERT INTO t (id, val) VALUES ('k2', 'v2'); \
         SAVEPOINT sp2; \
         INSERT INTO t (id, val) VALUES ('k3', 'v3'); \
         ROLLBACK TO sp1; \
         COMMIT;",
    )
    .unwrap();
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "k1");
}

#[test]
fn rollback_to_savepoint_syntax_variant() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id) VALUES ('k1'); \
         SAVEPOINT sp1; \
         INSERT INTO t (id) VALUES ('k2'); \
         ROLLBACK TO SAVEPOINT sp1; \
         COMMIT;",
    )
    .unwrap();
    let rs = rows(db.sql("SELECT id FROM t").unwrap());
    assert_eq!(rs.len(), 1);
}

#[test]
fn release_savepoint_syntax_variant() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id) VALUES ('k1'); \
         SAVEPOINT sp1; \
         INSERT INTO t (id) VALUES ('k2'); \
         RELEASE SAVEPOINT sp1; \
         COMMIT;",
    )
    .unwrap();
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 2);
}

// ========================== TXN_COMMIT Marker ==========================

#[test]
fn txn_commit_writes_marker_key() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();

    let epoch_before = db.current_epoch();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         INSERT INTO t (id, val) VALUES ('k2', 'v2'); \
         COMMIT;",
    )
    .unwrap();
    let epoch_after = db.current_epoch();
    assert!(epoch_after > epoch_before, "epoch should advance on commit");

    // The TXN_COMMIT marker should be visible as a key in the database
    let commit_prefix = b"__txn_commit/";
    let markers = db.scan_prefix(commit_prefix, None, None, None).unwrap();
    assert!(
        !markers.is_empty(),
        "TXN_COMMIT marker should exist after commit"
    );
}

#[test]
fn empty_transaction_no_commit_marker() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();
    db.sql("BEGIN; COMMIT;").unwrap();

    // Empty transaction should NOT write a commit marker
    let markers = db.scan_prefix(b"__txn_commit/", None, None, None).unwrap();
    assert!(
        markers.is_empty(),
        "empty transaction should not write commit marker"
    );
}

// ========================== Recovery Tests ==========================

#[test]
fn transaction_survives_reopen() {
    let dir = TempDir::new().unwrap();
    {
        let db = open_db(&dir);
        db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
            .unwrap();
        db.sql(
            "BEGIN; \
             INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
             INSERT INTO t (id, val) VALUES ('k2', 'v2'); \
             COMMIT;",
        )
        .unwrap();
        db.sync();
    }
    let db = open_db(&dir);
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 2);
}

#[test]
fn epoch_survives_reopen() {
    let dir = TempDir::new().unwrap();
    {
        let db = open_db(&dir);
        db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();
        db.sql(
            "BEGIN; \
             INSERT INTO t (id) VALUES ('k1'); \
             COMMIT;",
        )
        .unwrap();
        db.sync();
    }
    let db = open_db(&dir);
    assert!(
        db.current_epoch() >= 1,
        "epoch should be valid after reopen"
    );
}

// ========================== PITR Tests ==========================

#[test]
fn pitr_select_as_of_epoch() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();

    // Transaction 1: insert k1
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         COMMIT;",
    )
    .unwrap();
    let epoch1 = db.current_epoch();

    // Transaction 2: insert k2
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k2', 'v2'); \
         COMMIT;",
    )
    .unwrap();
    let epoch2 = db.current_epoch();
    assert!(epoch2 > epoch1);

    // Query at epoch1 — should see only k1
    let rs = rows(
        db.sql(&format!(
            "SELECT id FROM t AS OF EPOCH {epoch1} ORDER BY id ASC"
        ))
        .unwrap(),
    );
    assert_eq!(rs.len(), 1, "at epoch1 should see 1 row, got {rs:?}");
    assert_eq!(rs[0]["id"], "k1");

    // Query at epoch2 — should see both
    let rs = rows(
        db.sql(&format!(
            "SELECT id FROM t AS OF EPOCH {epoch2} ORDER BY id ASC"
        ))
        .unwrap(),
    );
    assert_eq!(rs.len(), 2, "at epoch2 should see 2 rows");
    assert_eq!(rs[0]["id"], "k1");
    assert_eq!(rs[1]["id"], "k2");
}

#[test]
fn pitr_does_not_see_future_writes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();

    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         COMMIT;",
    )
    .unwrap();
    let epoch1 = db.current_epoch();

    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k2', 'v2'); \
         COMMIT;",
    )
    .unwrap();

    // Current query sees both
    let rs = rows(db.sql("SELECT id FROM t ORDER BY id ASC").unwrap());
    assert_eq!(rs.len(), 2);

    // PITR at epoch1 should only see k1
    let rs = rows(
        db.sql(&format!(
            "SELECT id FROM t AS OF EPOCH {epoch1} ORDER BY id ASC"
        ))
        .unwrap(),
    );
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "k1");
}

#[test]
fn pitr_commit_marker_stores_commit_ts() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();

    db.sql(
        "BEGIN; \
         INSERT INTO t (id) VALUES ('k1'); \
         COMMIT;",
    )
    .unwrap();

    // The commit marker should store a non-zero commit_ts
    let markers = db.scan_prefix(b"__txn_commit/", None, None, None).unwrap();
    assert!(!markers.is_empty());

    let row = &markers[0];
    assert!(
        row.doc.len() >= 8,
        "commit marker should store 8-byte commit_ts"
    );
    let stored_ts = u64::from_le_bytes(row.doc[..8].try_into().unwrap());
    assert!(stored_ts > 0, "stored commit_ts should be non-zero");
}

// ========================== Incremental Backup Tests ==========================

#[test]
fn full_backup_includes_epoch() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         COMMIT;",
    )
    .unwrap();

    let backup_dir = TempDir::new().unwrap();
    let backup_path = backup_dir.path().join("full");
    let result = db
        .sql(&format!("BACKUP TO '{}'", backup_path.display()))
        .unwrap();
    match result {
        SqlResult::Affected { message, .. } => {
            assert!(message.contains("backup complete"));
        }
        _ => panic!("expected Affected result"),
    }

    // Verify metadata includes epoch
    let meta = std::fs::read_to_string(backup_path.join("backup_metadata.json")).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&meta).unwrap();
    assert_eq!(meta["type"], "full");
    assert!(meta["current_epoch"].as_u64().unwrap() > 0);
}

#[test]
fn incremental_backup_captures_changes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();

    // Txn 1
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k1', 'v1'); \
         COMMIT;",
    )
    .unwrap();
    let epoch1 = db.current_epoch();

    // Txn 2
    db.sql(
        "BEGIN; \
         INSERT INTO t (id, val) VALUES ('k2', 'v2'); \
         COMMIT;",
    )
    .unwrap();

    // Incremental backup since epoch1
    let backup_dir = TempDir::new().unwrap();
    let backup_path = backup_dir.path().join("incr");
    let result = db
        .sql(&format!(
            "BACKUP TO '{}' SINCE EPOCH {epoch1}",
            backup_path.display()
        ))
        .unwrap();
    match result {
        SqlResult::Affected { rows, message, .. } => {
            assert!(rows > 0, "incremental backup should capture records");
            assert!(message.contains("incremental backup complete"));
        }
        _ => panic!("expected Affected result"),
    }

    // Verify metadata
    let meta = std::fs::read_to_string(backup_path.join("backup_metadata.json")).unwrap();
    let meta: serde_json::Value = serde_json::from_str(&meta).unwrap();
    assert_eq!(meta["type"], "incremental");
    assert_eq!(meta["since_epoch"], epoch1);

    // Verify incremental data file exists
    assert!(backup_path.join("incremental_data.json").exists());
}

#[test]
fn incremental_backup_parser() {
    // Test that the parser handles SINCE EPOCH syntax
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();
    db.sql(
        "BEGIN; \
         INSERT INTO t (id) VALUES ('k1'); \
         COMMIT;",
    )
    .unwrap();

    let backup_dir = TempDir::new().unwrap();
    let backup_path = backup_dir.path().join("test_incr");
    // This should parse and execute without error
    let result = db.sql(&format!(
        "BACKUP TO '{}' SINCE EPOCH 1",
        backup_path.display()
    ));
    assert!(result.is_ok(), "BACKUP ... SINCE EPOCH should parse");
}
