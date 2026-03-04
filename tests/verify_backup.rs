// Phase 2: VERIFY BACKUP tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_backup_then_verify() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("INSERT INTO t (id, name) VALUES (1, 'hello')")
        .unwrap();

    let backup_dir = tempfile::tempdir().unwrap();
    let backup_path = backup_dir.path().to_string_lossy().to_string();
    db.sql(&format!("BACKUP DATABASE TO '{backup_path}'"))
        .unwrap();

    let result = db.sql(&format!("VERIFY BACKUP '{backup_path}'")).unwrap();
    if let SqlResult::Rows(rows) = result {
        assert!(!rows.is_empty());
        let json: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
        assert_eq!(json["status"], "VALID");
    } else {
        panic!("expected Rows result");
    }
}

#[test]
fn test_verify_nonexistent_backup() {
    let (db, _dir) = setup();
    let result = db.sql("VERIFY BACKUP '/nonexistent/path'").unwrap();
    if let SqlResult::Rows(rows) = result {
        let json: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
        assert_eq!(json["status"], "ISSUES_FOUND");
    }
}
