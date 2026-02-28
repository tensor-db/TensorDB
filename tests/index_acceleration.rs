//! Tests for index-accelerated query scans (equality, composite, IN-list).

use tempfile::TempDir;
use tensordb::{Config, Database};
use tensordb_core::sql::exec::SqlResult;

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

fn rows(result: SqlResult) -> Vec<serde_json::Value> {
    match result {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| {
                serde_json::from_slice(r).unwrap_or(serde_json::Value::String(
                    String::from_utf8_lossy(r).to_string(),
                ))
            })
            .collect(),
        _ => panic!("expected rows, got {result:?}"),
    }
}

fn setup_users(db: &Database) {
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT, city TEXT, age INTEGER)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, city, age) VALUES ('u1', 'Alice', 'NYC', 30)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, city, age) VALUES ('u2', 'Bob', 'SF', 25)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, city, age) VALUES ('u3', 'Carol', 'NYC', 35)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, city, age) VALUES ('u4', 'Dave', 'LA', 28)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, city, age) VALUES ('u5', 'Eve', 'SF', 22)")
        .unwrap();
}

// ===================== Single-column index equality scan =====================

#[test]
fn index_eq_scan_returns_matching_rows() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_city ON users (city)").unwrap();

    let rs = rows(
        db.sql("SELECT id, name FROM users WHERE city = 'NYC'")
            .unwrap(),
    );
    assert_eq!(rs.len(), 2);
    let ids: Vec<&str> = rs.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"u1"));
    assert!(ids.contains(&"u3"));
}

#[test]
fn index_eq_scan_no_match() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_city ON users (city)").unwrap();

    let rs = rows(
        db.sql("SELECT id FROM users WHERE city = 'Chicago'")
            .unwrap(),
    );
    assert_eq!(rs.len(), 0);
}

#[test]
fn index_eq_scan_single_result() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_city ON users (city)").unwrap();

    let rs = rows(db.sql("SELECT id FROM users WHERE city = 'LA'").unwrap());
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "u4");
}

// ===================== Composite index scan =====================

#[test]
fn composite_index_scan_two_columns() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_city_age ON users (city, age)")
        .unwrap();

    let rs = rows(
        db.sql("SELECT id FROM users WHERE city = 'NYC' AND age = 30")
            .unwrap(),
    );
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "u1");
}

#[test]
fn composite_index_scan_no_match() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_city_age ON users (city, age)")
        .unwrap();

    let rs = rows(
        db.sql("SELECT id FROM users WHERE city = 'NYC' AND age = 99")
            .unwrap(),
    );
    assert_eq!(rs.len(), 0);
}

// ===================== Composite index with string columns =====================

#[test]
fn composite_index_scan_string_columns() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, a TEXT, b TEXT)")
        .unwrap();
    db.sql("INSERT INTO t (id, a, b) VALUES ('r1', 'x', 'y')")
        .unwrap();
    db.sql("INSERT INTO t (id, a, b) VALUES ('r2', 'x', 'z')")
        .unwrap();
    db.sql("INSERT INTO t (id, a, b) VALUES ('r3', 'w', 'y')")
        .unwrap();
    db.sql("CREATE INDEX idx_ab ON t (a, b)").unwrap();

    let rs = rows(
        db.sql("SELECT id FROM t WHERE a = 'x' AND b = 'y'")
            .unwrap(),
    );
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "r1");
}

// ===================== IN-list index scan =====================

#[test]
fn index_in_scan_multiple_values() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_city ON users (city)").unwrap();

    let rs = rows(
        db.sql("SELECT id FROM users WHERE city IN ('NYC', 'LA')")
            .unwrap(),
    );
    assert_eq!(rs.len(), 3);
    let ids: Vec<&str> = rs.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"u1"));
    assert!(ids.contains(&"u3"));
    assert!(ids.contains(&"u4"));
}

#[test]
fn index_in_scan_single_value() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_city ON users (city)").unwrap();

    let rs = rows(db.sql("SELECT id FROM users WHERE city IN ('SF')").unwrap());
    assert_eq!(rs.len(), 2);
    let ids: Vec<&str> = rs.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert!(ids.contains(&"u2"));
    assert!(ids.contains(&"u5"));
}

// ===================== Without index — falls back to full scan =====================

#[test]
fn query_without_index_still_works() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    // No index created — should still return correct results via full scan

    let rs = rows(db.sql("SELECT id FROM users WHERE city = 'NYC'").unwrap());
    assert_eq!(rs.len(), 2);
}

// ===================== Index with additional filter predicates =====================

#[test]
fn index_scan_with_additional_filter() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_city ON users (city)").unwrap();

    // Index on city, but also filter by age — index narrows to NYC rows, then age filter applied
    let rs = rows(
        db.sql("SELECT id FROM users WHERE city = 'NYC' AND age = 35")
            .unwrap(),
    );
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["id"], "u3");
}

// ===================== Range scan with index =====================

#[test]
fn index_range_scan_gt() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_age ON users (age)").unwrap();

    let rs = rows(db.sql("SELECT id FROM users WHERE age > 28").unwrap());
    let ids: Vec<&str> = rs.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert_eq!(rs.len(), 2); // Alice(30), Carol(35)
    assert!(ids.contains(&"u1"));
    assert!(ids.contains(&"u3"));
}

#[test]
fn index_range_scan_gte() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_age ON users (age)").unwrap();

    let rs = rows(db.sql("SELECT id FROM users WHERE age >= 28").unwrap());
    assert_eq!(rs.len(), 3); // Dave(28), Alice(30), Carol(35)
}

#[test]
fn index_range_scan_lt() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_age ON users (age)").unwrap();

    let rs = rows(db.sql("SELECT id FROM users WHERE age < 28").unwrap());
    let ids: Vec<&str> = rs.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert_eq!(rs.len(), 2); // Bob(25), Eve(22)
    assert!(ids.contains(&"u2"));
    assert!(ids.contains(&"u5"));
}

#[test]
fn index_range_scan_lte() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_age ON users (age)").unwrap();

    let rs = rows(db.sql("SELECT id FROM users WHERE age <= 25").unwrap());
    assert_eq!(rs.len(), 2); // Bob(25), Eve(22)
}

// ===================== Index nested loop join =====================

#[test]
fn index_nested_loop_join_inner() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE orders (id TEXT PRIMARY KEY, user_id TEXT, amount REAL)")
        .unwrap();
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("CREATE INDEX idx_user_id ON orders (user_id)")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES ('u1', 'Alice')")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES ('u2', 'Bob')")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES ('o1', 'u1', 100)")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES ('o2', 'u1', 200)")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES ('o3', 'u2', 50)")
        .unwrap();

    let rs = rows(
        db.sql("SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.user_id ORDER BY orders.amount ASC")
            .unwrap(),
    );
    assert_eq!(rs.len(), 3);
}

#[test]
fn index_nested_loop_left_join() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("CREATE TABLE orders (id TEXT PRIMARY KEY, user_id TEXT, amount REAL)")
        .unwrap();
    db.sql("CREATE INDEX idx_user_id ON orders (user_id)")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES ('u1', 'Alice')")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES ('u2', 'Bob')")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES ('u3', 'Carol')")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES ('o1', 'u1', 100)")
        .unwrap();

    let rs = rows(
        db.sql("SELECT users.name FROM users LEFT JOIN orders ON users.id = orders.user_id")
            .unwrap(),
    );
    // Alice matched, Bob unmatched, Carol unmatched = 3 rows
    assert_eq!(rs.len(), 3);
}

// ===================== N-way JOIN =====================

#[test]
fn three_way_inner_join() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("CREATE TABLE orders (id TEXT PRIMARY KEY, user_id TEXT, product_id TEXT)")
        .unwrap();
    db.sql("CREATE TABLE products (id TEXT PRIMARY KEY, title TEXT)")
        .unwrap();

    db.sql("INSERT INTO users (id, name) VALUES ('u1', 'Alice')")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES ('u2', 'Bob')")
        .unwrap();
    db.sql("INSERT INTO products (id, title) VALUES ('p1', 'Widget')")
        .unwrap();
    db.sql("INSERT INTO products (id, title) VALUES ('p2', 'Gadget')")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, product_id) VALUES ('o1', 'u1', 'p1')")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, product_id) VALUES ('o2', 'u1', 'p2')")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, product_id) VALUES ('o3', 'u2', 'p1')")
        .unwrap();

    // Create indexes for INL join
    db.sql("CREATE INDEX idx_order_user ON orders (user_id)")
        .unwrap();
    db.sql("CREATE INDEX idx_order_product ON orders (product_id)")
        .unwrap();

    // 3-way join: users → orders → products
    let rs = rows(
        db.sql("SELECT users.name, products.title FROM users JOIN orders ON users.id = orders.user_id JOIN products ON orders.product_id = products.id")
            .unwrap(),
    );
    assert_eq!(rs.len(), 3);
}

#[test]
fn three_way_join_with_filter() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE a (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql("CREATE TABLE b (id TEXT PRIMARY KEY, a_id TEXT, val TEXT)")
        .unwrap();
    db.sql("CREATE TABLE c (id TEXT PRIMARY KEY, b_id TEXT, val TEXT)")
        .unwrap();

    db.sql("INSERT INTO a (id, val) VALUES ('a1', 'x')")
        .unwrap();
    db.sql("INSERT INTO b (id, a_id, val) VALUES ('b1', 'a1', 'y')")
        .unwrap();
    db.sql("INSERT INTO b (id, a_id, val) VALUES ('b2', 'a1', 'z')")
        .unwrap();
    db.sql("INSERT INTO c (id, b_id, val) VALUES ('c1', 'b1', 'w')")
        .unwrap();
    db.sql("INSERT INTO c (id, b_id, val) VALUES ('c2', 'b2', 'w')")
        .unwrap();

    let rs = rows(
        db.sql("SELECT count(*) FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id")
            .unwrap(),
    );
    assert_eq!(rs.len(), 1);
    // Should be "2" — two paths: a1→b1→c1, a1→b2→c2
    let count: String = String::from_utf8(
        match db
            .sql("SELECT count(*) FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id")
            .unwrap()
        {
            tensordb_core::sql::exec::SqlResult::Rows(r) => r[0].clone(),
            _ => panic!("expected rows"),
        },
    )
    .unwrap();
    assert_eq!(count, "2");
}

// ===================== Index survives insert after creation =====================

#[test]
fn index_reflects_new_inserts() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE INDEX idx_city ON users (city)").unwrap();

    // Insert a new row after index creation
    db.sql("INSERT INTO users (id, name, city, age) VALUES ('u6', 'Frank', 'NYC', 40)")
        .unwrap();

    let rs = rows(db.sql("SELECT id FROM users WHERE city = 'NYC'").unwrap());
    assert_eq!(rs.len(), 3);
}

// ===================== Join reordering: prefers indexed joins =====================

#[test]
fn join_reordering_prefers_indexed_table() {
    // Setup: 3 tables, but only products has an index on the join column.
    // The optimizer should reorder to execute the indexed join first.
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);

    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("CREATE TABLE orders (id TEXT PRIMARY KEY, user_id TEXT, product_id TEXT)")
        .unwrap();
    db.sql("CREATE TABLE products (id TEXT PRIMARY KEY, title TEXT, order_id TEXT)")
        .unwrap();

    // Only index on products.order_id
    db.sql("CREATE INDEX idx_prod_order ON products (order_id)")
        .unwrap();

    db.sql("INSERT INTO users (id, name) VALUES ('u1', 'Alice')")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, product_id) VALUES ('o1', 'u1', 'p1')")
        .unwrap();
    db.sql("INSERT INTO products (id, title, order_id) VALUES ('p1', 'Widget', 'o1')")
        .unwrap();

    // Query: users → orders → products (orders has no index, products does)
    // The reorderer should prefer to join products (indexed) before orders (no index)
    // but in this case both joins reference different columns so order may not change.
    // The key test is that the result is still correct regardless of reordering.
    let rs = rows(
        db.sql("SELECT users.name, products.title FROM users JOIN orders ON users.id = orders.user_id JOIN products ON orders.id = products.order_id")
            .unwrap(),
    );
    assert_eq!(rs.len(), 1);
    assert_eq!(rs[0]["users.name"], "Alice");
    assert_eq!(rs[0]["products.title"], "Widget");
}

// ===================== Predicate pushdown =====================

#[test]
fn predicate_pushdown_filters_before_join() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_users(&db);
    db.sql("CREATE TABLE orders (id TEXT PRIMARY KEY, user_id TEXT, amount REAL)")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES ('o1', 'u1', 100)")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES ('o2', 'u3', 200)")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES ('o3', 'u2', 50)")
        .unwrap();
    db.sql("CREATE INDEX idx_user_id ON orders (user_id)")
        .unwrap();

    // Predicate `users.city = 'NYC'` should be pushed down to filter users
    // before the join, so only Alice(u1) and Carol(u3) are joined with orders.
    let rs = rows(
        db.sql("SELECT users.name, orders.amount FROM users JOIN orders ON users.id = orders.user_id WHERE users.city = 'NYC'")
            .unwrap(),
    );
    assert_eq!(rs.len(), 2);
    let names: Vec<&str> = rs
        .iter()
        .map(|r| r["users.name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"Alice"));
    assert!(names.contains(&"Carol"));
}
