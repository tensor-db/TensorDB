use tempfile::tempdir;
use tensordb::config::Config;
use tensordb::sql::exec::SqlResult;
use tensordb::Database;

fn setup_db() -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 1,
        ..Config::default()
    };
    let db = Database::open(dir.path(), cfg).unwrap();
    (dir, db)
}

fn sql(db: &Database, query: &str) -> String {
    match db.sql(query).unwrap() {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| String::from_utf8_lossy(r).to_string())
            .collect::<Vec<_>>()
            .join("\n"),
        SqlResult::Affected { message, .. } => message,
        SqlResult::Explain(e) => e,
    }
}

fn sql_rows(db: &Database, query: &str) -> Vec<String> {
    match db.sql(query).unwrap() {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| String::from_utf8_lossy(r).to_string())
            .collect(),
        _ => vec![],
    }
}

fn setup_users(db: &Database) {
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age) VALUES (3, 'Charlie', 35)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age) VALUES (4, 'Diana', 28)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age) VALUES (5, 'Eve', 32)")
        .unwrap();
}

// --- Step 1.1: OFFSET ---

#[test]
fn test_offset() {
    let (_dir, db) = setup_db();
    setup_users(&db);
    let rows = sql_rows(&db, "SELECT * FROM users ORDER BY id LIMIT 3 OFFSET 2");
    assert_eq!(rows.len(), 3, "OFFSET 2 LIMIT 3 should return 3 rows");
}

#[test]
fn test_offset_exceeds_rows() {
    let (_dir, db) = setup_db();
    setup_users(&db);
    let rows = sql_rows(&db, "SELECT * FROM users ORDER BY id OFFSET 100");
    assert_eq!(rows.len(), 0, "OFFSET beyond row count should return empty");
}

// --- Step 1.2: IF EXISTS / IF NOT EXISTS ---

#[test]
fn test_create_if_not_exists() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t1 (id INTEGER PRIMARY KEY)").unwrap();
    let result = sql(
        &db,
        "CREATE TABLE IF NOT EXISTS t1 (id INTEGER PRIMARY KEY)",
    );
    assert!(result.contains("IF NOT EXISTS"));
}

#[test]
fn test_drop_if_exists() {
    let (_dir, db) = setup_db();
    let result = sql(&db, "DROP TABLE IF EXISTS nonexistent");
    assert!(result.contains("IF EXISTS"));
}

#[test]
fn test_create_index_if_not_exists() {
    let (_dir, db) = setup_db();
    setup_users(&db);
    db.sql("CREATE INDEX idx_name ON users (name)").unwrap();
    let result = sql(&db, "CREATE INDEX IF NOT EXISTS idx_name ON users (name)");
    assert!(result.contains("IF NOT EXISTS"));
}

#[test]
fn test_drop_index_if_exists() {
    let (_dir, db) = setup_db();
    setup_users(&db);
    let result = sql(&db, "DROP INDEX IF EXISTS nonexistent ON users");
    assert!(result.contains("IF EXISTS"));
}

// --- Step 1.3: SELECT without FROM ---

#[test]
fn test_select_without_from() {
    let (_dir, db) = setup_db();
    let rows = sql_rows(&db, "SELECT 1 + 1 AS result");
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("2"));
}

#[test]
fn test_select_string_without_from() {
    let (_dir, db) = setup_db();
    let rows = sql_rows(&db, "SELECT 'hello' AS greeting");
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("hello"));
}

// --- Step 1.4: Multi-value INSERT ---

#[test]
fn test_multi_row_insert() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    let result = sql(
        &db,
        "INSERT INTO items (id, name) VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    );
    assert!(result.contains("3"));
    let rows = sql_rows(&db, "SELECT * FROM items ORDER BY id");
    assert_eq!(rows.len(), 3);
}

// --- Step 1.5: FULL OUTER JOIN ---

#[test]
fn test_full_outer_join() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE left_t (id INTEGER PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql("CREATE TABLE right_t (id INTEGER PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql("INSERT INTO left_t (id, val) VALUES (1, 'a')")
        .unwrap();
    db.sql("INSERT INTO left_t (id, val) VALUES (2, 'b')")
        .unwrap();
    db.sql("INSERT INTO right_t (id, val) VALUES (2, 'x')")
        .unwrap();
    db.sql("INSERT INTO right_t (id, val) VALUES (3, 'y')")
        .unwrap();

    let rows = sql_rows(
        &db,
        "SELECT * FROM left_t FULL OUTER JOIN right_t ON left_t.id = right_t.id",
    );
    // FULL OUTER JOIN produces rows for matched + unmatched on both sides
    assert!(
        rows.len() >= 3,
        "FULL OUTER JOIN should produce at least 3 rows"
    );
}

// --- Step 1.6: RETURNING on UPDATE/DELETE ---

#[test]
fn test_update_returning() {
    let (_dir, db) = setup_db();
    setup_users(&db);
    let rows = sql_rows(
        &db,
        "UPDATE users SET age = 31 WHERE name = 'Alice' RETURNING *",
    );
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("31"), "Updated age should be 31");
}

#[test]
fn test_delete_returning() {
    let (_dir, db) = setup_db();
    setup_users(&db);
    let rows = sql_rows(
        &db,
        "DELETE FROM users WHERE name = 'Bob' RETURNING name, age",
    );
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("Bob"));
}

// --- Step 1.7: Subqueries in WHERE ---

#[test]
fn test_in_subquery() {
    let (_dir, db) = setup_db();
    setup_users(&db);
    db.sql("CREATE TABLE vip (id INTEGER PRIMARY KEY)").unwrap();
    db.sql("INSERT INTO vip (id) VALUES (1)").unwrap();
    db.sql("INSERT INTO vip (id) VALUES (3)").unwrap();

    let rows = sql_rows(&db, "SELECT * FROM users WHERE id IN (SELECT id FROM vip)");
    assert_eq!(rows.len(), 2, "IN subquery should match 2 VIP users");
}

#[test]
fn test_exists_subquery() {
    let (_dir, db) = setup_db();
    setup_users(&db);
    db.sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER)")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id) VALUES (100, 1)")
        .unwrap();

    let rows = sql_rows(
        &db,
        "SELECT * FROM users WHERE EXISTS (SELECT id FROM orders WHERE user_id = 1)",
    );
    // EXISTS is non-correlated, returns true → all users returned
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_scalar_subquery() {
    let (_dir, db) = setup_db();
    setup_users(&db);
    let rows = sql_rows(&db, "SELECT * FROM users WHERE age > (SELECT 29 AS val)");
    // Users with age > 29: Alice(30), Charlie(35), Eve(32) = 3
    assert_eq!(rows.len(), 3, "Scalar subquery should filter to 3 rows");
}

// --- Step 1.8: Upsert ---

#[test]
fn test_upsert_do_update() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE kv (k TEXT PRIMARY KEY, v INTEGER)")
        .unwrap();
    db.sql("INSERT INTO kv (k, v) VALUES ('key1', 10)").unwrap();

    sql(
        &db,
        "INSERT INTO kv (k, v) VALUES ('key1', 20) ON CONFLICT (k) DO UPDATE SET v = 20",
    );
    let rows = sql_rows(&db, "SELECT * FROM kv WHERE k = 'key1'");
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("20"), "Upsert should update value to 20");
}

#[test]
fn test_upsert_do_nothing() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE kv (k TEXT PRIMARY KEY, v INTEGER)")
        .unwrap();
    db.sql("INSERT INTO kv (k, v) VALUES ('key1', 10)").unwrap();

    sql(
        &db,
        "INSERT INTO kv (k, v) VALUES ('key1', 99) ON CONFLICT (k) DO NOTHING",
    );
    let rows = sql_rows(&db, "SELECT * FROM kv WHERE k = 'key1'");
    assert_eq!(rows.len(), 1);
    assert!(
        rows[0].contains("10"),
        "DO NOTHING should keep original value"
    );
}

// --- Step 1.9: Persistent Sessions ---

#[test]
fn test_persistent_session_begin_commit() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE sess_t (id INTEGER PRIMARY KEY, val TEXT)")
        .unwrap();

    let mut handle = tensordb::SqlSessionHandle::new();

    db.sql_session(&mut handle, "BEGIN").unwrap();
    assert!(handle.in_transaction());

    db.sql_session(
        &mut handle,
        "INSERT INTO sess_t (id, val) VALUES (1, 'hello')",
    )
    .unwrap();

    db.sql_session(&mut handle, "COMMIT").unwrap();
    assert!(!handle.in_transaction());

    let rows = sql_rows(&db, "SELECT * FROM sess_t");
    assert_eq!(rows.len(), 1);
}

#[test]
fn test_persistent_session_rollback() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE sess_t2 (id INTEGER PRIMARY KEY, val TEXT)")
        .unwrap();

    let mut handle = tensordb::SqlSessionHandle::new();

    db.sql_session(&mut handle, "BEGIN").unwrap();
    db.sql_session(
        &mut handle,
        "INSERT INTO sess_t2 (id, val) VALUES (1, 'hello')",
    )
    .unwrap();
    db.sql_session(&mut handle, "ROLLBACK").unwrap();
    assert!(!handle.in_transaction());

    let rows = sql_rows(&db, "SELECT * FROM sess_t2");
    assert_eq!(rows.len(), 0);
}
