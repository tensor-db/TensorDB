// Phase 4: Plan guide stability tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_create_plan_guide() {
    let (db, _dir) = setup();
    let result = db.sql("CREATE PLAN GUIDE 'fast_users' FOR 'SELECT * FROM users WHERE id = 1' USING 'USE_INDEX(idx_users_id)'").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("created plan guide"));
    }
}

#[test]
fn test_show_plan_guides() {
    let (db, _dir) = setup();
    db.sql("CREATE PLAN GUIDE 'guide1' FOR 'SELECT * FROM t' USING 'FORCE_SCAN'")
        .unwrap();
    db.sql(
        "CREATE PLAN GUIDE 'guide2' FOR 'SELECT * FROM t WHERE id = 1' USING 'USE_INDEX(idx_t_id)'",
    )
    .unwrap();

    let result = db.sql("SHOW PLAN GUIDES").unwrap();
    if let SqlResult::Rows(rows) = result {
        assert_eq!(rows.len(), 2);
        let json: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
        assert!(json.get("name").is_some());
        assert!(json.get("sql_pattern").is_some());
        assert!(json.get("hints").is_some());
    }
}

#[test]
fn test_drop_plan_guide() {
    let (db, _dir) = setup();
    db.sql("CREATE PLAN GUIDE 'test_guide' FOR 'SELECT 1' USING 'FORCE_SCAN'")
        .unwrap();

    let result = db.sql("DROP PLAN GUIDE 'test_guide'").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("dropped plan guide"));
    }

    // Should be gone
    let result = db.sql("SHOW PLAN GUIDES").unwrap();
    if let SqlResult::Rows(rows) = result {
        assert!(rows.is_empty(), "guide should be gone");
    }
}

#[test]
fn test_duplicate_plan_guide_errors() {
    let (db, _dir) = setup();
    db.sql("CREATE PLAN GUIDE 'dup' FOR 'SELECT 1' USING 'FORCE_SCAN'")
        .unwrap();

    let err = db
        .sql("CREATE PLAN GUIDE 'dup' FOR 'SELECT 1' USING 'FORCE_SCAN'")
        .unwrap_err();
    assert!(format!("{err}").contains("already exists"));
}
