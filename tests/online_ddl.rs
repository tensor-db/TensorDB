// Phase 4: Online DDL tests (DROP COLUMN, RENAME COLUMN)
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_drop_column() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
        .unwrap();

    let result = db.sql("ALTER TABLE users DROP COLUMN age").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("dropped column age"));
    }

    // Verify schema no longer has 'age'
    let desc = db.sql("DESCRIBE users").unwrap();
    if let SqlResult::Rows(rows) = desc {
        for row in &rows {
            let json: serde_json::Value = serde_json::from_slice(row).unwrap();
            assert_ne!(json["column"], "age", "age column should be gone");
        }
    }
}

#[test]
fn test_rename_column() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)")
        .unwrap();
    db.sql("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 9.99)")
        .unwrap();

    let result = db
        .sql("ALTER TABLE products RENAME COLUMN name TO product_name")
        .unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("renamed column"));
    }

    // Verify schema has new column name
    let desc = db.sql("DESCRIBE products").unwrap();
    if let SqlResult::Rows(rows) = desc {
        let col_names: Vec<String> = rows
            .iter()
            .map(|r| {
                let json: serde_json::Value = serde_json::from_slice(r).unwrap();
                json["column"].as_str().unwrap().to_string()
            })
            .collect();
        assert!(
            col_names.contains(&"product_name".to_string()),
            "should have product_name: {col_names:?}"
        );
        assert!(
            !col_names.contains(&"name".to_string()),
            "should not have old name"
        );
    }
}

#[test]
fn test_cannot_drop_pk_column() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();

    let err = db.sql("ALTER TABLE t DROP COLUMN id").unwrap_err();
    assert!(
        format!("{err}").contains("primary key"),
        "Should prevent dropping PK: {err}"
    );
}

#[test]
fn test_drop_nonexistent_column_errors() {
    let (db, _dir) = setup();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();

    let err = db.sql("ALTER TABLE t DROP COLUMN nonexistent").unwrap_err();
    assert!(
        format!("{err}").contains("does not exist"),
        "Should error: {err}"
    );
}
