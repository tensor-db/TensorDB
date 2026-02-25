use spectradb::{Config, Database};
use tempfile::TempDir;

fn open_db(dir: &TempDir) -> Database {
    Database::open(
        dir.path(),
        Config {
            shard_count: 2,
            ..Config::default()
        },
    )
    .unwrap()
}

#[test]
fn typed_create_table_and_insert() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT, age INTEGER)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age) VALUES ('u1', 'Alice', 30)")
        .unwrap();
    let result = db.sql("SELECT doc FROM users WHERE pk = 'u1'").unwrap();
    match result {
        spectradb_core::sql::exec::SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let doc: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
            assert_eq!(doc["name"], "Alice");
            // Age is stored as JSON number - may be integer or float
            assert_eq!(doc["age"].as_f64().unwrap() as i64, 30);
        }
        _ => panic!("expected rows"),
    }
}

#[test]
fn typed_insert_select_columns() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT NOT NULL, price REAL)")
        .unwrap();
    db.sql("INSERT INTO products (id, name, price) VALUES ('p1', 'Widget', 9.99)")
        .unwrap();
    db.sql("INSERT INTO products (id, name, price) VALUES ('p2', 'Gadget', 24.50)")
        .unwrap();

    let result = db.sql("SELECT name, price FROM products ORDER BY name ASC").unwrap();
    match result {
        spectradb_core::sql::exec::SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
            let r0: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
            assert_eq!(r0["name"], "Gadget");
            assert_eq!(r0["price"], 24.5);
            let r1: serde_json::Value = serde_json::from_slice(&rows[1]).unwrap();
            assert_eq!(r1["name"], "Widget");
        }
        _ => panic!("expected rows"),
    }
}

#[test]
fn typed_update_and_delete() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE items (id TEXT PRIMARY KEY, qty INTEGER)")
        .unwrap();
    db.sql("INSERT INTO items (id, qty) VALUES ('a', 10)")
        .unwrap();
    db.sql("INSERT INTO items (id, qty) VALUES ('b', 20)")
        .unwrap();

    // Update
    db.sql("UPDATE items SET doc = '{\"id\":\"a\",\"qty\":15}' WHERE pk = 'a'")
        .unwrap();

    let result = db.sql("SELECT doc FROM items WHERE pk = 'a'").unwrap();
    match result {
        spectradb_core::sql::exec::SqlResult::Rows(rows) => {
            let doc: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
            assert_eq!(doc["qty"], 15);
        }
        _ => panic!("expected rows"),
    }

    // Delete
    db.sql("DELETE FROM items WHERE pk = 'b'").unwrap();
    let result = db.sql("SELECT count(*) FROM items").unwrap();
    match result {
        spectradb_core::sql::exec::SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(std::str::from_utf8(&rows[0]).unwrap(), "1");
        }
        _ => panic!("expected rows"),
    }
}
