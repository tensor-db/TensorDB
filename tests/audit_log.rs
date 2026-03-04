// Phase 3: Audit log tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_create_table_logged_to_audit() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();

    let result = db.sql("SHOW AUDIT LOG 10").unwrap();
    if let SqlResult::Rows(rows) = result {
        assert!(!rows.is_empty(), "audit log should have entries");
        let json: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
        let details = json["details"].to_string();
        assert!(
            details.contains("TableCreated") || details.contains("users"),
            "should log table creation: {details}"
        );
    } else {
        panic!("expected Rows");
    }
}

#[test]
fn test_drop_table_logged_to_audit() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE temp (id INTEGER PRIMARY KEY)")
        .unwrap();
    db.sql("DROP TABLE temp").unwrap();

    let result = db.sql("SHOW AUDIT LOG 10").unwrap();
    if let SqlResult::Rows(rows) = result {
        // Should have both create and drop events
        assert!(rows.len() >= 2, "audit log should have at least 2 entries");
    }
}

#[test]
fn test_show_audit_log_returns_rows() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE t1 (id INTEGER PRIMARY KEY)").unwrap();
    db.sql("CREATE TABLE t2 (id INTEGER PRIMARY KEY)").unwrap();

    let result = db.sql("SHOW AUDIT LOG").unwrap();
    if let SqlResult::Rows(rows) = result {
        assert!(rows.len() >= 2);
        // Each row should be valid JSON with expected fields
        for row in &rows {
            let json: serde_json::Value = serde_json::from_slice(row).unwrap();
            assert!(json.get("timestamp").is_some());
            assert!(json.get("event_type").is_some());
            assert!(json.get("session_user").is_some());
        }
    }
}
