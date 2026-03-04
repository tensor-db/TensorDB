// Phase 3: GDPR erasure tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_forget_key_erases_data() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES (2, 'Bob')")
        .unwrap();

    let result = db.sql("FORGET KEY '1' FROM users").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(
            message.contains("erased"),
            "should report erasure: {message}"
        );
    }
}

#[test]
fn test_forget_key_audit_logged() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")
        .unwrap();
    db.sql("FORGET KEY '1' FROM users").unwrap();

    let result = db.sql("SHOW AUDIT LOG 10").unwrap();
    if let SqlResult::Rows(rows) = result {
        let has_erasure = rows.iter().any(|r| {
            let json: serde_json::Value = serde_json::from_slice(r).unwrap_or_default();
            json["details"].to_string().contains("GdprErasure")
        });
        assert!(has_erasure, "audit log should contain GdprErasure event");
    }
}
