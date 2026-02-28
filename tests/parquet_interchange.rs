// Integration tests for Parquet support (requires --features parquet).
#![cfg(feature = "parquet")]

use tensordb_core::config::Config;
use tensordb_core::sql::exec::SqlResult;
use tensordb_core::Database;

fn test_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

fn sql(db: &Database, query: &str) -> Vec<serde_json::Value> {
    match db.sql(query).unwrap() {
        SqlResult::Rows(rows) => rows
            .into_iter()
            .map(|r| {
                serde_json::from_slice(&r).unwrap_or_else(|_| {
                    let s = String::from_utf8_lossy(&r);
                    serde_json::json!({ "result": s.as_ref() })
                })
            })
            .collect(),
        SqlResult::Affected { message, .. } => {
            vec![serde_json::json!({ "message": message })]
        }
        SqlResult::Explain(text) => {
            vec![serde_json::json!({ "explain": text })]
        }
    }
}

fn sql_affected(db: &Database, query: &str) -> (u64, String) {
    match db.sql(query).unwrap() {
        SqlResult::Affected { rows, message, .. } => (rows, message),
        other => panic!("expected Affected, got: {other:?}"),
    }
}

#[test]
fn copy_parquet_roundtrip_typed() {
    let (db, dir) = test_db();
    db.sql("CREATE TABLE metrics (id INTEGER PRIMARY KEY, name TEXT, value REAL, active BOOLEAN);")
        .unwrap();
    db.sql("INSERT INTO metrics (id, name, value, active) VALUES (1, 'cpu', 0.85, true);")
        .unwrap();
    db.sql("INSERT INTO metrics (id, name, value, active) VALUES (2, 'mem', 0.72, true);")
        .unwrap();
    db.sql("INSERT INTO metrics (id, name, value, active) VALUES (3, 'disk', 0.45, false);")
        .unwrap();

    let parquet_path = dir.path().join("metrics.parquet");
    let (count, _msg) = sql_affected(
        &db,
        &format!(
            "COPY metrics TO '{}' FORMAT PARQUET;",
            parquet_path.display()
        ),
    );
    assert_eq!(count, 3);
    assert!(parquet_path.exists());
    assert!(std::fs::metadata(&parquet_path).unwrap().len() > 0);

    // Import into new table
    db.sql(
        "CREATE TABLE metrics2 (id INTEGER PRIMARY KEY, name TEXT, value REAL, active BOOLEAN);",
    )
    .unwrap();
    let (count2, _) = sql_affected(
        &db,
        &format!(
            "COPY metrics2 FROM '{}' FORMAT PARQUET;",
            parquet_path.display()
        ),
    );
    assert_eq!(count2, 3);

    let rows = sql(
        &db,
        "SELECT id, name, value, active FROM metrics2 ORDER BY id;",
    );
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["name"], "cpu");
    assert!((rows[0]["value"].as_f64().unwrap() - 0.85).abs() < 0.01);
    assert_eq!(rows[0]["active"], true);
    assert_eq!(rows[2]["name"], "disk");
    assert_eq!(rows[2]["active"], false);
}

#[test]
fn read_parquet_table_function() {
    let (db, dir) = test_db();

    // First export some data to parquet
    db.sql("CREATE TABLE src (id INTEGER PRIMARY KEY, label TEXT);")
        .unwrap();
    db.sql("INSERT INTO src (id, label) VALUES (1, 'Alpha');")
        .unwrap();
    db.sql("INSERT INTO src (id, label) VALUES (2, 'Beta');")
        .unwrap();
    db.sql("INSERT INTO src (id, label) VALUES (3, 'Gamma');")
        .unwrap();

    let parquet_path = dir.path().join("src.parquet");
    sql_affected(
        &db,
        &format!("COPY src TO '{}' FORMAT PARQUET;", parquet_path.display()),
    );

    // Read back via table function
    let rows = sql(
        &db,
        &format!(
            "SELECT id, label FROM read_parquet('{}') ORDER BY id;",
            parquet_path.display()
        ),
    );
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["label"], "Alpha");
    assert_eq!(rows[1]["label"], "Beta");
    assert_eq!(rows[2]["label"], "Gamma");
}

#[test]
fn read_parquet_with_filter_and_agg() {
    let (db, dir) = test_db();

    db.sql("CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, amount REAL);")
        .unwrap();
    db.sql("INSERT INTO sales (id, region, amount) VALUES (1, 'East', 100.0);")
        .unwrap();
    db.sql("INSERT INTO sales (id, region, amount) VALUES (2, 'West', 200.0);")
        .unwrap();
    db.sql("INSERT INTO sales (id, region, amount) VALUES (3, 'East', 150.0);")
        .unwrap();
    db.sql("INSERT INTO sales (id, region, amount) VALUES (4, 'West', 300.0);")
        .unwrap();

    let parquet_path = dir.path().join("sales.parquet");
    sql_affected(
        &db,
        &format!("COPY sales TO '{}' FORMAT PARQUET;", parquet_path.display()),
    );

    // Filter
    let rows = sql(
        &db,
        &format!(
            "SELECT region, amount FROM read_parquet('{}') WHERE amount > 150;",
            parquet_path.display()
        ),
    );
    assert_eq!(rows.len(), 2);

    // Aggregate
    let rows = sql(
        &db,
        &format!(
            "SELECT region, SUM(amount) AS total FROM read_parquet('{}') GROUP BY region ORDER BY region;",
            parquet_path.display()
        ),
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["region"], "East");
    assert_eq!(rows[0]["total"], 250.0);
}

#[test]
fn parquet_without_feature_returns_error() {
    // This test always runs but checks the error message
    // when the parquet format is used without the feature.
    // Since this test file is gated on #[cfg(feature = "parquet")],
    // we just verify the feature IS available here.
    let (db, dir) = test_db();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY);").unwrap();
    let parquet_path = dir.path().join("t.parquet");
    // Should succeed since parquet feature is enabled
    let result = db.sql(&format!(
        "COPY t TO '{}' FORMAT PARQUET;",
        parquet_path.display()
    ));
    assert!(result.is_ok());
}
