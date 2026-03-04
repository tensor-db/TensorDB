// Phase 2: SUGGEST INDEX tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_suggest_index_for_unindexed_where() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        .unwrap();

    let result = db
        .sql("SUGGEST INDEX FOR 'SELECT * FROM users WHERE age > 30'")
        .unwrap();
    if let SqlResult::Rows(rows) = result {
        assert!(!rows.is_empty(), "should suggest an index on age");
        let json: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
        assert!(json["suggested_index"].as_str().unwrap().contains("age"));
        assert!(json["reason"].as_str().unwrap().contains("WHERE"));
    } else {
        panic!("expected Rows result");
    }
}

#[test]
fn test_suggest_index_no_suggestion_for_indexed_column() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("CREATE INDEX idx_users_name ON users (name)")
        .unwrap();

    let result = db
        .sql("SUGGEST INDEX FOR 'SELECT * FROM users WHERE name = 1'")
        .unwrap();
    if let SqlResult::Rows(rows) = result {
        // name is already indexed, no suggestion
        assert!(
            rows.is_empty(),
            "should not suggest index for already-indexed column"
        );
    }
}
