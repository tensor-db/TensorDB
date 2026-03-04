// Phase 2: Strict mode tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_set_strict_mode_on_off() {
    let (db, _dir) = setup();
    let result = db.sql("SET STRICT_MODE = ON").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("strict_mode = true"));
    }
    let result = db.sql("SET STRICT_MODE = OFF").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("strict_mode = false"));
    }
}

#[test]
fn test_set_query_timeout() {
    let (db, _dir) = setup();
    let result = db.sql("SET QUERY_TIMEOUT = 5000").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("5000"));
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
fn test_set_unknown_variable_errors() {
    let (db, _dir) = setup();
    let err = db.sql("SET UNKNOWN_VAR = 42").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("unknown session variable"));
}
