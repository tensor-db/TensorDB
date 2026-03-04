// Phase 4: WAL management tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_show_wal_status() {
    let (db, _dir) = setup();
    // Insert some data to create WAL entries
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("INSERT INTO t (id, name) VALUES (1, 'hello')")
        .unwrap();

    let result = db.sql("SHOW WAL STATUS").unwrap();
    if let SqlResult::Rows(rows) = result {
        assert!(!rows.is_empty(), "should have per-shard WAL info");
        let json: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
        assert!(json.get("shard").is_some());
        assert!(json.get("wal_bytes").is_some());
    } else {
        panic!("expected Rows");
    }
}

#[test]
fn test_wal_status_has_all_shards() {
    let (db, _dir) = setup();
    let result = db.sql("SHOW WAL STATUS").unwrap();
    if let SqlResult::Rows(rows) = result {
        // Default shard count is 4
        assert_eq!(rows.len(), 4, "should have 4 shards");
    }
}
