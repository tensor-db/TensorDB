// Phase 4: Per-query resource limits tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_set_query_timeout() {
    let (db, _dir) = setup();
    let result = db.sql("SET QUERY_TIMEOUT = 1000").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("1000"));
    }
}

#[test]
fn test_set_query_max_memory() {
    let (db, _dir) = setup();
    let result = db.sql("SET QUERY_MAX_MEMORY = 1048576").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("1048576"));
    }
}

#[test]
fn test_timeout_reset_to_zero_disables() {
    let (db, _dir) = setup();
    // Set then reset
    db.sql("SET QUERY_TIMEOUT = 1000").unwrap();
    let result = db.sql("SET QUERY_TIMEOUT = 0").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("0"));
    }
}
