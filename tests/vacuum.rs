// Phase 2: VACUUM tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_vacuum_reports_tombstone_count() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("INSERT INTO items (id, name) VALUES (1, 'a')")
        .unwrap();
    db.sql("INSERT INTO items (id, name) VALUES (2, 'b')")
        .unwrap();
    db.sql("DELETE FROM items WHERE id = 1").unwrap();

    let result = db.sql("VACUUM items").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("vacuum complete"), "got: {message}");
    } else {
        panic!("expected Affected result");
    }
}

#[test]
fn test_vacuum_all_tables() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("INSERT INTO t1 (id, name) VALUES (1, 'a')").unwrap();
    db.sql("DELETE FROM t1 WHERE id = 1").unwrap();

    let result = db.sql("VACUUM").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("vacuum complete"));
    }
}
