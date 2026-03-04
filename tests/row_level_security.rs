// Phase 3: Row-level security tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_create_and_drop_policy() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, owner TEXT)")
        .unwrap();

    let result = db
        .sql("CREATE POLICY user_isolation ON users FOR SELECT USING (owner = 'alice')")
        .unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("created policy"));
    }

    let result = db.sql("DROP POLICY user_isolation ON users").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("dropped policy"));
    }
}

#[test]
fn test_create_policy_with_roles() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
        .unwrap();

    let result = db
        .sql("CREATE POLICY admin_only ON docs FOR ALL TO admin, superadmin USING (id > 0)")
        .unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("created policy admin_only"));
    }
}

#[test]
fn test_duplicate_policy_errors() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY)").unwrap();
    db.sql("CREATE POLICY p1 ON t FOR SELECT USING (id > 0)")
        .unwrap();

    let err = db
        .sql("CREATE POLICY p1 ON t FOR SELECT USING (id > 0)")
        .unwrap_err();
    assert!(format!("{err}").contains("already exists"));
}

#[test]
fn test_drop_nonexistent_policy_errors() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY)").unwrap();

    let err = db.sql("DROP POLICY nonexistent ON t").unwrap_err();
    assert!(format!("{err}").contains("does not exist"));
}
