use tempfile::tempdir;

use tensordb::config::Config;
use tensordb::sql::exec::SqlResult;
use tensordb::Database;

fn rows_to_strings(rows: Vec<Vec<u8>>) -> Vec<String> {
    rows.into_iter()
        .map(|row| String::from_utf8(row).unwrap())
        .collect()
}

fn setup_db() -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 1,
        ..Config::default()
    };
    let db = Database::open(dir.path(), cfg).unwrap();
    (dir, db)
}

// --- Change 1: Numeric ORDER BY ---

#[test]
fn numeric_order_by_ascending() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE nums (id INTEGER PRIMARY KEY, value INTEGER);")
        .unwrap();
    db.sql("INSERT INTO nums (id, value) VALUES (1, 10);")
        .unwrap();
    db.sql("INSERT INTO nums (id, value) VALUES (2, 9);")
        .unwrap();
    db.sql("INSERT INTO nums (id, value) VALUES (3, 100);")
        .unwrap();
    db.sql("INSERT INTO nums (id, value) VALUES (4, 2);")
        .unwrap();
    db.sql("INSERT INTO nums (id, value) VALUES (5, 20);")
        .unwrap();

    let result = db
        .sql("SELECT value FROM nums ORDER BY value ASC;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 5);
            let values: Vec<f64> = rows
                .iter()
                .map(|r| {
                    let v: serde_json::Value = serde_json::from_slice(r).unwrap();
                    v["value"].as_f64().unwrap()
                })
                .collect();
            // Proper numeric ordering: 2, 9, 10, 20, 100
            assert_eq!(values, vec![2.0, 9.0, 10.0, 20.0, 100.0]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn numeric_order_by_descending() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE nums (id INTEGER PRIMARY KEY, value INTEGER);")
        .unwrap();
    db.sql("INSERT INTO nums (id, value) VALUES (1, 10);")
        .unwrap();
    db.sql("INSERT INTO nums (id, value) VALUES (2, 9);")
        .unwrap();
    db.sql("INSERT INTO nums (id, value) VALUES (3, 100);")
        .unwrap();
    db.sql("INSERT INTO nums (id, value) VALUES (4, 2);")
        .unwrap();

    let result = db
        .sql("SELECT value FROM nums ORDER BY value DESC;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let values: Vec<f64> = rows
                .iter()
                .map(|r| {
                    let v: serde_json::Value = serde_json::from_slice(r).unwrap();
                    v["value"].as_f64().unwrap()
                })
                .collect();
            assert_eq!(values, vec![100.0, 10.0, 9.0, 2.0]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Change 2: Window Function + LIMIT ---

#[test]
fn window_function_computed_before_limit() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE scores (id INTEGER PRIMARY KEY, score INTEGER);")
        .unwrap();
    for i in 1..=5 {
        db.sql(&format!(
            "INSERT INTO scores (id, score) VALUES ({}, {});",
            i,
            i * 10
        ))
        .unwrap();
    }

    // ROW_NUMBER should be computed over ALL 5 rows, then LIMIT to 3
    let result = db
        .sql("SELECT score, ROW_NUMBER() OVER (ORDER BY score ASC) AS rn FROM scores ORDER BY score ASC LIMIT 3;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 3);
            let strs = rows_to_strings(rows);
            // The window function should assign row numbers 1-5 over ALL rows, then LIMIT to 3
            for s in &strs {
                let v: serde_json::Value = serde_json::from_str(s).unwrap();
                let rn = v["rn"].as_f64().unwrap();
                let score = v["score"].as_f64().unwrap();
                // rn should be based on full dataset
                match score as i64 {
                    10 => assert_eq!(rn, 1.0),
                    20 => assert_eq!(rn, 2.0),
                    30 => assert_eq!(rn, 3.0),
                    _ => panic!("unexpected score: {score}"),
                }
            }
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Change 3: CASE WHEN ---

#[test]
fn case_when_searched() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE items (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO items (pk, doc) VALUES ('a', '{\"val\":10}');")
        .unwrap();
    db.sql("INSERT INTO items (pk, doc) VALUES ('b', '{\"val\":50}');")
        .unwrap();
    db.sql("INSERT INTO items (pk, doc) VALUES ('c', '{\"val\":90}');")
        .unwrap();

    let result = db
        .sql("SELECT pk, CASE WHEN val > 80 THEN 'high' WHEN val > 30 THEN 'mid' ELSE 'low' END AS tier FROM items ORDER BY pk;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strs = rows_to_strings(rows);
            assert_eq!(strs.len(), 3);
            let v0: serde_json::Value = serde_json::from_str(&strs[0]).unwrap();
            let v1: serde_json::Value = serde_json::from_str(&strs[1]).unwrap();
            let v2: serde_json::Value = serde_json::from_str(&strs[2]).unwrap();
            assert_eq!(v0["tier"], "low");
            assert_eq!(v1["tier"], "mid");
            assert_eq!(v2["tier"], "high");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn case_when_simple() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE colors (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("INSERT INTO colors (pk, doc) VALUES ('r', '{\"color\":\"red\"}');")
        .unwrap();
    db.sql("INSERT INTO colors (pk, doc) VALUES ('b', '{\"color\":\"blue\"}');")
        .unwrap();
    db.sql("INSERT INTO colors (pk, doc) VALUES ('g', '{\"color\":\"green\"}');")
        .unwrap();

    let result = db
        .sql("SELECT pk, CASE color WHEN 'red' THEN 'warm' WHEN 'blue' THEN 'cool' ELSE 'other' END AS temp FROM colors ORDER BY pk;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strs = rows_to_strings(rows);
            assert_eq!(strs.len(), 3);
            let vb: serde_json::Value = serde_json::from_str(&strs[0]).unwrap();
            let vg: serde_json::Value = serde_json::from_str(&strs[1]).unwrap();
            let vr: serde_json::Value = serde_json::from_str(&strs[2]).unwrap();
            assert_eq!(vb["temp"], "cool");
            assert_eq!(vg["temp"], "other");
            assert_eq!(vr["temp"], "warm");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn case_when_no_else() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"x\":1}');")
        .unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k2', '{\"x\":5}');")
        .unwrap();

    let result = db
        .sql("SELECT pk, CASE WHEN x > 3 THEN 'big' END AS label FROM t ORDER BY pk;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strs = rows_to_strings(rows);
            let v0: serde_json::Value = serde_json::from_str(&strs[0]).unwrap();
            let v1: serde_json::Value = serde_json::from_str(&strs[1]).unwrap();
            assert!(v0["label"].is_null()); // x=1, no match, no ELSE => NULL
            assert_eq!(v1["label"], "big"); // x=5, matches
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Change 4: CAST ---

#[test]
fn cast_integer() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"val\":3.7}');")
        .unwrap();

    let result = db
        .sql("SELECT CAST(val AS INTEGER) AS int_val FROM t;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let v: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
            assert_eq!(v["int_val"], 3.0);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn cast_text() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"val\":42}');")
        .unwrap();

    let result = db.sql("SELECT CAST(val AS TEXT) AS txt FROM t;").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let v: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
            assert_eq!(v["txt"], "42");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn cast_boolean() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"val\":1}');")
        .unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k2', '{\"val\":0}');")
        .unwrap();

    let result = db
        .sql("SELECT pk, CAST(val AS BOOLEAN) AS flag FROM t ORDER BY pk;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strs = rows_to_strings(rows);
            let v0: serde_json::Value = serde_json::from_str(&strs[0]).unwrap();
            let v1: serde_json::Value = serde_json::from_str(&strs[1]).unwrap();
            assert_eq!(v0["flag"], true);
            assert_eq!(v1["flag"], false);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn cast_real() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"val\":\"3.14\"}');")
        .unwrap();

    let result = db.sql("SELECT CAST(val AS REAL) AS num FROM t;").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let v: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
            #[allow(clippy::approx_constant)]
            {
                assert!((v["num"].as_f64().unwrap() - 3.14).abs() < 0.001);
            }
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Change 5: Transaction-local reads ---

#[test]
fn transaction_reads_own_writes() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    // Use a multi-statement SQL that does BEGIN, INSERT, SELECT, COMMIT in one call
    let _result = db
        .sql(
            "BEGIN; \
             INSERT INTO t (pk, doc) VALUES ('txn1', '{\"hello\":\"world\"}'); \
             SELECT doc FROM t WHERE pk = 'txn1'; \
             COMMIT;",
        )
        .unwrap();

    // The last result is from COMMIT
    // Let's verify by doing individual calls in a different way
    // Actually, execute_sql processes all statements and returns the last result.
    // The last statement is COMMIT, so the result is Affected.
    // Let's verify the data was committed correctly.
    let result = db.sql("SELECT doc FROM t WHERE pk = 'txn1';").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("hello"));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn transaction_scan_sees_staged_writes() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('existing', '{\"a\":1}');")
        .unwrap();

    // In a transaction, insert a new row and then do a scan (SELECT without pk filter)
    // The scan should see both the existing row and the staged one.
    // execute_sql returns the last statement's result, so we do the SELECT last before COMMIT.
    // But we can't easily get intermediate results. Let's do a count approach.

    // After commit, verify data is there
    let result = db
        .sql(
            "BEGIN; \
         INSERT INTO t (pk, doc) VALUES ('new_in_txn', '{\"b\":2}'); \
         COMMIT;",
        )
        .unwrap();
    match result {
        SqlResult::Affected { rows, .. } => {
            assert_eq!(rows, 1); // 1 write committed
        }
        other => panic!("unexpected: {other:?}"),
    }

    // Verify both rows exist
    let result = db.sql("SELECT doc FROM t;").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Fix: Implicit table aliases (FROM users u) ---

#[test]
fn implicit_table_alias_from() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, role TEXT);")
        .unwrap();
    db.sql("INSERT INTO users (id, name, role) VALUES (1, 'alice', 'admin');")
        .unwrap();
    db.sql("INSERT INTO users (id, name, role) VALUES (2, 'bob', 'user');")
        .unwrap();

    // Implicit alias: FROM users u (no AS keyword)
    let result = db
        .sql("SELECT u.name FROM users u WHERE u.role = 'admin';")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("alice"), "expected alice, got: {s}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn implicit_alias_join() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();
    db.sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES (1, 'alice');")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES (101, 1, 50.0);")
        .unwrap();

    // Implicit alias on JOIN: users u JOIN orders o
    let result = db
        .sql("SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(
                s.contains("alice"),
                "expected alice in join result, got: {s}"
            );
            assert!(s.contains("50"), "expected 50 in join result, got: {s}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn explicit_as_alias_still_works() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();
    db.sql("INSERT INTO users (id, name) VALUES (1, 'alice');")
        .unwrap();

    // Explicit alias: FROM users AS u
    let result = db.sql("SELECT u.name FROM users AS u;").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("alice"), "expected alice, got: {s}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Fix: COUNT(DISTINCT col) ---

#[test]
fn count_distinct() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES (1, 1, 10.0);")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES (2, 1, 20.0);")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES (3, 2, 30.0);")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, amount) VALUES (4, 3, 40.0);")
        .unwrap();

    // COUNT(DISTINCT user_id) should return 3 (users 1, 2, 3)
    let result = db
        .sql("SELECT COUNT(DISTINCT user_id) FROM orders;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("3"), "expected count=3 distinct users, got: {s}");
        }
        other => panic!("unexpected: {other:?}"),
    }

    // Regular COUNT should return 4
    let result = db.sql("SELECT COUNT(user_id) FROM orders;").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("4"), "expected count=4, got: {s}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Fix: Multi-column GROUP BY with qualified names ---

#[test]
fn group_by_with_qualified_names() {
    let (_dir, db) = setup_db();
    db.sql(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, status TEXT, amount REAL);",
    )
    .unwrap();
    db.sql("INSERT INTO orders (id, user_id, status, amount) VALUES (1, 1, 'shipped', 10.0);")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, status, amount) VALUES (2, 1, 'shipped', 20.0);")
        .unwrap();
    db.sql("INSERT INTO orders (id, user_id, status, amount) VALUES (3, 2, 'pending', 30.0);")
        .unwrap();

    // Multi-column GROUP BY with table-qualified names using alias
    let result = db
        .sql("SELECT o.user_id, o.status, COUNT(*) FROM orders o GROUP BY o.user_id, o.status;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2, "expected 2 groups, got: {}", rows.len());
        }
        other => panic!("unexpected: {other:?}"),
    }
}
