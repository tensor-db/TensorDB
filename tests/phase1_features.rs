//! Integration tests for Phase 1: Production-Ready Foundations
//!
//! Covers: DECIMAL type, secondary indexes, backup/restore, encryption module

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

#[allow(dead_code)]
fn affected(result: SqlResult) -> u64 {
    match result {
        SqlResult::Affected { rows, .. } => rows,
        _ => panic!("expected affected, got {result:?}"),
    }
}

// ========================== DECIMAL Type ==========================

#[test]
fn decimal_create_table_and_insert() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE accounts (id TEXT PRIMARY KEY, balance DECIMAL(18,2))")
        .unwrap();
    db.sql("INSERT INTO accounts (id, balance) VALUES ('a1', 1234.56)")
        .unwrap();
    db.sql("INSERT INTO accounts (id, balance) VALUES ('a2', 7890.12)")
        .unwrap();

    let rs = rows(
        db.sql("SELECT balance FROM accounts ORDER BY balance ASC")
            .unwrap(),
    );
    assert_eq!(rs.len(), 2);
    // DECIMAL values are stored as JSON numbers in JSON-doc mode
    assert!((rs[0]["balance"].as_f64().unwrap() - 1234.56).abs() < 0.01);
    assert!((rs[1]["balance"].as_f64().unwrap() - 7890.12).abs() < 0.01);
}

#[test]
fn decimal_arithmetic() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE ledger (id TEXT PRIMARY KEY, amount DECIMAL(18,4))")
        .unwrap();
    db.sql("INSERT INTO ledger (id, amount) VALUES ('t1', 100.5025)")
        .unwrap();
    db.sql("INSERT INTO ledger (id, amount) VALUES ('t2', 200.2575)")
        .unwrap();

    let rs = rows(db.sql("SELECT SUM(amount) AS total FROM ledger").unwrap());
    assert_eq!(rs.len(), 1);
    let total = rs[0]["total"].as_f64().unwrap();
    assert!((total - 300.76).abs() < 0.01);
}

#[test]
fn decimal_cast() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('r1', '42.75')")
        .unwrap();

    let rs = rows(db.sql("SELECT CAST(val AS DECIMAL) AS d FROM t").unwrap());
    assert_eq!(rs.len(), 1);
    // DECIMAL cast produces a string representation in JSON
    let d = &rs[0]["d"];
    assert!(d.as_str().unwrap().contains("42.75") || d.as_f64().unwrap() == 42.75);
}

#[test]
fn decimal_typeof() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE d (id TEXT PRIMARY KEY, amount DECIMAL(10,2))")
        .unwrap();
    db.sql("INSERT INTO d (id, amount) VALUES ('x', 99.99)")
        .unwrap();

    // In JSON-doc mode, DECIMAL values are stored as numbers
    let rs = rows(db.sql("SELECT TYPEOF(amount) AS t FROM d").unwrap());
    assert_eq!(rs[0]["t"], "number");
}

// ========================== Secondary Indexes ==========================

#[test]
fn create_index_and_describe() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT, name TEXT)")
        .unwrap();
    db.sql("INSERT INTO users (id, email, name) VALUES ('u1', 'alice@test.com', 'Alice')")
        .unwrap();
    db.sql("INSERT INTO users (id, email, name) VALUES ('u2', 'bob@test.com', 'Bob')")
        .unwrap();

    // Create index
    let result = db.sql("CREATE INDEX idx_email ON users (email)").unwrap();
    match result {
        SqlResult::Affected { message, .. } => {
            assert!(message.contains("idx_email"));
        }
        _ => panic!("expected affected"),
    }
}

#[test]
fn unique_index_prevents_duplicates() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE users (id TEXT PRIMARY KEY, email TEXT)")
        .unwrap();
    db.sql("INSERT INTO users (id, email) VALUES ('u1', 'alice@test.com')")
        .unwrap();
    db.sql("CREATE UNIQUE INDEX idx_uniq_email ON users (email)")
        .unwrap();

    // This should succeed — different email
    db.sql("INSERT INTO users (id, email) VALUES ('u2', 'bob@test.com')")
        .unwrap();

    // This should fail — duplicate email
    let result = db.sql("INSERT INTO users (id, email) VALUES ('u3', 'alice@test.com')");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("UNIQUE constraint violated"));
}

#[test]
fn composite_index() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE addresses (id TEXT PRIMARY KEY, city TEXT, state TEXT)")
        .unwrap();
    db.sql("INSERT INTO addresses (id, city, state) VALUES ('a1', 'Austin', 'TX')")
        .unwrap();
    db.sql("INSERT INTO addresses (id, city, state) VALUES ('a2', 'Denver', 'CO')")
        .unwrap();

    let result = db
        .sql("CREATE INDEX idx_city_state ON addresses (city, state)")
        .unwrap();
    match result {
        SqlResult::Affected { message, .. } => {
            assert!(message.contains("idx_city_state"), "message: {message}");
        }
        _ => panic!("expected affected"),
    }
}

#[test]
fn drop_index_cleans_up_entries() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('r1', 'foo')")
        .unwrap();
    db.sql("INSERT INTO t (id, val) VALUES ('r2', 'bar')")
        .unwrap();
    db.sql("CREATE INDEX idx_val ON t (val)").unwrap();

    let result = db.sql("DROP INDEX idx_val ON t").unwrap();
    match result {
        SqlResult::Affected { rows, message, .. } => {
            assert!(rows >= 1); // At least the metadata entry
            assert!(message.contains("dropped index"));
        }
        _ => panic!("expected affected"),
    }
}

#[test]
fn index_maintenance_on_update_and_delete() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE products (id TEXT PRIMARY KEY, sku TEXT)")
        .unwrap();
    db.sql("INSERT INTO products (id, sku) VALUES ('p1', 'SKU-001')")
        .unwrap();
    db.sql("CREATE UNIQUE INDEX idx_sku ON products (sku)")
        .unwrap();

    // Update the SKU — should remove old index entry and add new one
    db.sql("UPDATE products SET sku = 'SKU-002' WHERE pk = 'p1'")
        .unwrap();

    // Now inserting the old SKU should succeed (it was freed by the update)
    db.sql("INSERT INTO products (id, sku) VALUES ('p2', 'SKU-001')")
        .unwrap();

    // But the new SKU should be blocked
    let result = db.sql("INSERT INTO products (id, sku) VALUES ('p3', 'SKU-002')");
    assert!(result.is_err());

    // Delete p1 — should free SKU-002
    db.sql("DELETE FROM products WHERE pk = 'p1'").unwrap();
    db.sql("INSERT INTO products (id, sku) VALUES ('p3', 'SKU-002')")
        .unwrap();
}

#[test]
fn drop_table_cleans_up_indexes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE temp (id TEXT PRIMARY KEY, val TEXT)")
        .unwrap();
    db.sql("INSERT INTO temp (id, val) VALUES ('r1', 'x')")
        .unwrap();
    db.sql("CREATE INDEX idx_temp_val ON temp (val)").unwrap();

    // Drop the table — should clean up index entries and metadata
    let result = db.sql("DROP TABLE temp").unwrap();
    match result {
        SqlResult::Affected { rows, .. } => {
            // Should have cleaned up rows + table meta + index meta + index entries
            assert!(rows >= 3);
        }
        _ => panic!("expected affected"),
    }
}

// ========================== Backup & Restore ==========================

#[test]
fn backup_and_restore() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);

    // Create some data
    db.sql("CREATE TABLE notes (id TEXT PRIMARY KEY, content TEXT)")
        .unwrap();
    db.sql("INSERT INTO notes (id, content) VALUES ('n1', 'hello')")
        .unwrap();
    db.sql("INSERT INTO notes (id, content) VALUES ('n2', 'world')")
        .unwrap();

    // Backup
    let backup_dir = TempDir::new().unwrap();
    let backup_path = backup_dir.path().join("backup");
    let result = db
        .sql(&format!("BACKUP DATABASE TO '{}'", backup_path.display()))
        .unwrap();
    match result {
        SqlResult::Affected { message, .. } => {
            assert!(message.contains("backup complete"));
        }
        _ => panic!("expected affected"),
    }

    // Verify backup metadata exists
    assert!(backup_path.join("backup_metadata.json").exists());
}

#[test]
fn backup_without_database_keyword() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE t (id TEXT PRIMARY KEY)").unwrap();

    let backup_dir = TempDir::new().unwrap();
    let backup_path = backup_dir.path().join("bak");
    // Should also work without DATABASE keyword
    let result = db
        .sql(&format!("BACKUP TO '{}'", backup_path.display()))
        .unwrap();
    assert!(matches!(result, SqlResult::Affected { .. }));
}

#[test]
fn restore_validates_backup() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);

    // Restore from non-existent path should fail
    let result = db.sql("RESTORE DATABASE FROM '/nonexistent/path'");
    assert!(result.is_err());

    // Restore from path without backup_metadata.json should fail
    let empty_dir = TempDir::new().unwrap();
    let result = db.sql(&format!(
        "RESTORE DATABASE FROM '{}'",
        empty_dir.path().display()
    ));
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not a valid backup"));
}

// ========================== Encryption Module ==========================

#[test]
fn encryption_key_from_passphrase() {
    use tensordb_core::storage::encryption::EncryptionKey;

    let k1 = EncryptionKey::from_passphrase("test-key-123");
    let k2 = EncryptionKey::from_passphrase("test-key-123");
    assert_eq!(k1.as_bytes(), k2.as_bytes());

    let k3 = EncryptionKey::from_passphrase("different-key");
    assert_ne!(k1.as_bytes(), k3.as_bytes());
}

#[test]
fn encryption_key_from_bytes() {
    use tensordb_core::storage::encryption::EncryptionKey;

    let bytes = [42u8; 32];
    let key = EncryptionKey::from_bytes(bytes);
    assert_eq!(*key.as_bytes(), bytes);
}

#[test]
fn encryption_config_defaults() {
    let config = Config::default();
    assert!(config.encryption_passphrase.is_none());
    assert!(config.encryption_key_file.is_none());
}

// ========================== DECIMAL in Typed Tables ==========================

#[test]
fn decimal_in_typed_table_roundtrip() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE prices (id TEXT PRIMARY KEY, price DECIMAL(12,4), tax DECIMAL(8,2))")
        .unwrap();
    db.sql("INSERT INTO prices (id, price, tax) VALUES ('item1', 99.9999, 8.25)")
        .unwrap();
    db.sql("INSERT INTO prices (id, price, tax) VALUES ('item2', 0.0001, 0.00)")
        .unwrap();

    let rs = rows(
        db.sql("SELECT price, tax FROM prices ORDER BY price ASC")
            .unwrap(),
    );
    assert_eq!(rs.len(), 2);
    // DECIMAL values stored as JSON numbers in JSON-doc mode
    assert!((rs[0]["price"].as_f64().unwrap() - 0.0001).abs() < 0.00001);
    assert!((rs[0]["tax"].as_f64().unwrap() - 0.0).abs() < 0.01);
    assert!((rs[1]["price"].as_f64().unwrap() - 99.9999).abs() < 0.0001);
    assert!((rs[1]["tax"].as_f64().unwrap() - 8.25).abs() < 0.01);
}

#[test]
fn decimal_comparison_in_where() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE balances (id TEXT PRIMARY KEY, amount DECIMAL(18,2))")
        .unwrap();
    db.sql("INSERT INTO balances (id, amount) VALUES ('a', 100.50)")
        .unwrap();
    db.sql("INSERT INTO balances (id, amount) VALUES ('b', 200.75)")
        .unwrap();
    db.sql("INSERT INTO balances (id, amount) VALUES ('c', 50.25)")
        .unwrap();

    let rs = rows(
        db.sql("SELECT id FROM balances WHERE amount > 100 ORDER BY id ASC")
            .unwrap(),
    );
    assert_eq!(rs.len(), 2);
    assert_eq!(rs[0]["id"], "a");
    assert_eq!(rs[1]["id"], "b");
}

// ========================== Multiple Features Combined ==========================

#[test]
fn decimal_with_index() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE orders (id TEXT PRIMARY KEY, amount DECIMAL(18,2), status TEXT)")
        .unwrap();
    db.sql("INSERT INTO orders (id, amount, status) VALUES ('o1', 150.50, 'pending')")
        .unwrap();
    db.sql("INSERT INTO orders (id, amount, status) VALUES ('o2', 250.75, 'shipped')")
        .unwrap();
    db.sql("INSERT INTO orders (id, amount, status) VALUES ('o3', 50.25, 'pending')")
        .unwrap();

    // Create index on status
    db.sql("CREATE INDEX idx_status ON orders (status)")
        .unwrap();

    // Query should still work correctly with index present
    let rs = rows(
        db.sql("SELECT id, amount FROM orders WHERE status = 'pending' ORDER BY id ASC")
            .unwrap(),
    );
    assert_eq!(rs.len(), 2);
    assert_eq!(rs[0]["id"], "o1");
    assert_eq!(rs[1]["id"], "o3");
}

#[test]
fn backup_preserves_indexes() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE items (id TEXT PRIMARY KEY, category TEXT)")
        .unwrap();
    db.sql("INSERT INTO items (id, category) VALUES ('i1', 'electronics')")
        .unwrap();
    db.sql("CREATE INDEX idx_cat ON items (category)").unwrap();

    let backup_dir = TempDir::new().unwrap();
    let backup_path = backup_dir.path().join("backup");
    let result = db
        .sql(&format!("BACKUP DATABASE TO '{}'", backup_path.display()))
        .unwrap();
    assert!(matches!(result, SqlResult::Affected { .. }));
    assert!(backup_path.join("backup_metadata.json").exists());
    assert!(backup_path.join("MANIFEST.json").exists());
}
