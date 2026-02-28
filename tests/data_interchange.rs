/// Integration tests for v0.18 Data Interchange features.
/// Tests: COPY TO/FROM CSV/JSON/NDJSON (typed tables), read_csv/read_json/read_ndjson table functions.
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

// ---------- COPY TO CSV (typed table) ----------

#[test]
fn copy_to_csv_typed_table() {
    let (db, dir) = test_db();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30);")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25);")
        .unwrap();

    let csv_path = dir.path().join("users.csv");
    let (count, _msg) = sql_affected(
        &db,
        &format!("COPY users TO '{}' FORMAT CSV;", csv_path.display()),
    );
    assert_eq!(count, 2);

    let content = std::fs::read_to_string(&csv_path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert!(lines[0].contains("id"));
    assert!(lines[0].contains("name"));
    assert!(lines[0].contains("age"));
    assert!(lines.len() >= 3); // header + 2 data rows
}

// ---------- COPY TO/FROM CSV round-trip ----------

#[test]
fn copy_csv_roundtrip_typed() {
    let (db, dir) = test_db();
    db.sql("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);")
        .unwrap();
    db.sql("INSERT INTO products (id, name, price) VALUES (1, 'Widget', 9.99);")
        .unwrap();
    db.sql("INSERT INTO products (id, name, price) VALUES (2, 'Gadget', 19.50);")
        .unwrap();

    let csv_path = dir.path().join("products.csv");
    sql_affected(
        &db,
        &format!("COPY products TO '{}' FORMAT CSV;", csv_path.display()),
    );

    // Create a new table and import
    db.sql("CREATE TABLE products2 (id INTEGER PRIMARY KEY, name TEXT, price REAL);")
        .unwrap();
    let (count, _) = sql_affected(
        &db,
        &format!("COPY products2 FROM '{}' FORMAT CSV;", csv_path.display()),
    );
    assert_eq!(count, 2);

    let rows = sql(&db, "SELECT id, name, price FROM products2 ORDER BY id;");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["name"], "Widget");
    assert!((rows[0]["price"].as_f64().unwrap() - 9.99).abs() < 0.01);
    assert_eq!(rows[1]["name"], "Gadget");
}

// ---------- COPY TO/FROM JSON ----------

#[test]
fn copy_json_roundtrip_typed() {
    let (db, dir) = test_db();
    db.sql("CREATE TABLE items (id INTEGER PRIMARY KEY, label TEXT);")
        .unwrap();
    db.sql("INSERT INTO items (id, label) VALUES (1, 'Alpha');")
        .unwrap();
    db.sql("INSERT INTO items (id, label) VALUES (2, 'Beta');")
        .unwrap();

    let json_path = dir.path().join("items.json");
    sql_affected(
        &db,
        &format!("COPY items TO '{}' FORMAT JSON;", json_path.display()),
    );

    // Verify JSON is valid array
    let content = std::fs::read_to_string(&json_path).unwrap();
    let arr: Vec<serde_json::Value> = serde_json::from_str(&content).unwrap();
    assert_eq!(arr.len(), 2);

    // Import into new table
    db.sql("CREATE TABLE items2 (id INTEGER PRIMARY KEY, label TEXT);")
        .unwrap();
    let (count, _) = sql_affected(
        &db,
        &format!("COPY items2 FROM '{}' FORMAT JSON;", json_path.display()),
    );
    assert_eq!(count, 2);

    let rows = sql(&db, "SELECT id, label FROM items2 ORDER BY id;");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["label"], "Alpha");
    assert_eq!(rows[1]["label"], "Beta");
}

// ---------- COPY TO/FROM NDJSON ----------

#[test]
fn copy_ndjson_roundtrip_typed() {
    let (db, dir) = test_db();
    db.sql("CREATE TABLE logs (id INTEGER PRIMARY KEY, level TEXT, msg TEXT);")
        .unwrap();
    db.sql("INSERT INTO logs (id, level, msg) VALUES (1, 'INFO', 'started');")
        .unwrap();
    db.sql("INSERT INTO logs (id, level, msg) VALUES (2, 'ERROR', 'failed');")
        .unwrap();

    let ndjson_path = dir.path().join("logs.ndjson");
    sql_affected(
        &db,
        &format!("COPY logs TO '{}' FORMAT NDJSON;", ndjson_path.display()),
    );

    // Verify NDJSON
    let content = std::fs::read_to_string(&ndjson_path).unwrap();
    let data_lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(data_lines.len(), 2);
    for line in &data_lines {
        let _: serde_json::Value = serde_json::from_str(line).unwrap();
    }

    // Import
    db.sql("CREATE TABLE logs2 (id INTEGER PRIMARY KEY, level TEXT, msg TEXT);")
        .unwrap();
    let (count, _) = sql_affected(
        &db,
        &format!("COPY logs2 FROM '{}' FORMAT NDJSON;", ndjson_path.display()),
    );
    assert_eq!(count, 2);

    let rows = sql(&db, "SELECT id, level, msg FROM logs2 ORDER BY id;");
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["level"], "INFO");
    assert_eq!(rows[1]["msg"], "failed");
}

// ---------- CSV with quoted fields ----------

#[test]
fn copy_csv_with_commas_in_values() {
    let (db, dir) = test_db();
    db.sql("CREATE TABLE notes (id INTEGER PRIMARY KEY, text TEXT);")
        .unwrap();
    db.sql("INSERT INTO notes (id, text) VALUES (1, 'hello, world');")
        .unwrap();
    db.sql("INSERT INTO notes (id, text) VALUES (2, 'no comma');")
        .unwrap();

    let csv_path = dir.path().join("notes.csv");
    sql_affected(
        &db,
        &format!("COPY notes TO '{}' FORMAT CSV;", csv_path.display()),
    );

    // The "hello, world" value should be properly quoted
    let content = std::fs::read_to_string(&csv_path).unwrap();
    assert!(content.contains("\"hello, world\""));

    // Round-trip
    db.sql("CREATE TABLE notes2 (id INTEGER PRIMARY KEY, text TEXT);")
        .unwrap();
    sql_affected(
        &db,
        &format!("COPY notes2 FROM '{}' FORMAT CSV;", csv_path.display()),
    );

    let rows = sql(&db, "SELECT id, text FROM notes2 ORDER BY id;");
    assert_eq!(rows[0]["text"], "hello, world");
}

// ---------- Table functions: read_csv ----------

#[test]
fn read_csv_table_function() {
    let (db, dir) = test_db();
    let csv_path = dir.path().join("data.csv");
    std::fs::write(
        &csv_path,
        "name,age,active\nAlice,30,true\nBob,25,false\nCharlie,35,true\n",
    )
    .unwrap();

    let rows = sql(
        &db,
        &format!(
            "SELECT name, age, active FROM read_csv('{}');",
            csv_path.display()
        ),
    );
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["name"], "Alice");
    assert_eq!(rows[0]["age"].as_f64().unwrap() as i64, 30);
    assert_eq!(rows[0]["active"], true);
    assert_eq!(rows[1]["name"], "Bob");
    assert_eq!(rows[1]["age"].as_f64().unwrap() as i64, 25);
    assert_eq!(rows[1]["active"], false);
}

// ---------- Table functions: read_json ----------

#[test]
fn read_json_table_function() {
    let (db, dir) = test_db();
    let json_path = dir.path().join("data.json");
    std::fs::write(
        &json_path,
        r#"[{"city":"NYC","pop":8000000},{"city":"LA","pop":4000000}]"#,
    )
    .unwrap();

    let rows = sql(
        &db,
        &format!(
            "SELECT city, pop FROM read_json('{}');",
            json_path.display()
        ),
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["city"], "NYC");
    assert_eq!(rows[0]["pop"].as_f64().unwrap() as i64, 8000000);
    assert_eq!(rows[1]["city"], "LA");
}

// ---------- Table functions: read_ndjson ----------

#[test]
fn read_ndjson_table_function() {
    let (db, dir) = test_db();
    let ndjson_path = dir.path().join("data.ndjson");
    std::fs::write(
        &ndjson_path,
        "{\"x\":1,\"y\":\"a\"}\n{\"x\":2,\"y\":\"b\"}\n{\"x\":3,\"y\":\"c\"}\n",
    )
    .unwrap();

    let rows = sql(
        &db,
        &format!("SELECT x, y FROM read_ndjson('{}');", ndjson_path.display()),
    );
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["x"].as_f64().unwrap() as i64, 1);
    assert_eq!(rows[0]["y"], "a");
    assert_eq!(rows[2]["x"].as_f64().unwrap() as i64, 3);
    assert_eq!(rows[2]["y"], "c");
}

// ---------- read_csv with WHERE filter ----------

#[test]
fn read_csv_with_where() {
    let (db, dir) = test_db();
    let csv_path = dir.path().join("scores.csv");
    std::fs::write(
        &csv_path,
        "student,score\nAlice,85\nBob,72\nCharlie,93\nDiana,68\n",
    )
    .unwrap();

    let rows = sql(
        &db,
        &format!(
            "SELECT student, score FROM read_csv('{}') WHERE score > 80;",
            csv_path.display()
        ),
    );
    assert_eq!(rows.len(), 2);
    let names: Vec<&str> = rows
        .iter()
        .map(|r| r["student"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"Alice"));
    assert!(names.contains(&"Charlie"));
}

// ---------- read_csv with ORDER BY ----------

#[test]
fn read_csv_with_order_by() {
    let (db, dir) = test_db();
    let csv_path = dir.path().join("nums.csv");
    std::fs::write(&csv_path, "val\n3\n1\n4\n1\n5\n").unwrap();

    let rows = sql(
        &db,
        &format!(
            "SELECT val FROM read_csv('{}') ORDER BY val ASC;",
            csv_path.display()
        ),
    );
    let vals: Vec<i64> = rows
        .iter()
        .map(|r| r["val"].as_f64().unwrap() as i64)
        .collect();
    assert_eq!(vals, vec![1, 1, 3, 4, 5]);
}

// ---------- read_csv with aggregation ----------

#[test]
fn read_csv_with_aggregation() {
    let (db, dir) = test_db();
    let csv_path = dir.path().join("sales.csv");
    std::fs::write(
        &csv_path,
        "region,amount\nEast,100\nWest,200\nEast,150\nWest,300\n",
    )
    .unwrap();

    let rows = sql(
        &db,
        &format!(
            "SELECT region, SUM(amount) AS total FROM read_csv('{}') GROUP BY region ORDER BY region;",
            csv_path.display()
        ),
    );
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["region"], "East");
    assert_eq!(rows[0]["total"], 250.0);
    assert_eq!(rows[1]["region"], "West");
    assert_eq!(rows[1]["total"], 500.0);
}
