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

fn setup_test_table(db: &Database) {
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"name\":\"Alice\",\"age\":30,\"balance\":100.5}');").unwrap();
    db.sql(
        "INSERT INTO t (pk, doc) VALUES ('k2', '{\"name\":\"Bob\",\"age\":25,\"balance\":200.0}');",
    )
    .unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k3', '{\"name\":\"Charlie\",\"age\":35,\"balance\":50.0}');").unwrap();
}

// --- WHERE clause tests ---

#[test]
fn where_doc_field_comparison() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db.sql("SELECT doc FROM t WHERE doc.age > 28;").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
            let strs = rows_to_strings(rows);
            assert!(strs.iter().any(|s| s.contains("Alice")));
            assert!(strs.iter().any(|s| s.contains("Charlie")));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn where_and_or_combined() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db
        .sql("SELECT doc FROM t WHERE doc.age > 28 AND doc.balance > 60;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("Alice"));
        }
        other => panic!("unexpected: {other:?}"),
    }

    let result = db
        .sql("SELECT doc FROM t WHERE doc.age < 26 OR doc.age > 34;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
            let strs = rows_to_strings(rows);
            assert!(strs.iter().any(|s| s.contains("Bob")));
            assert!(strs.iter().any(|s| s.contains("Charlie")));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn where_like_pattern() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db
        .sql("SELECT doc FROM t WHERE doc.name LIKE 'Al%';")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("Alice"));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn where_between() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db
        .sql("SELECT doc FROM t WHERE doc.age BETWEEN 26 AND 34;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("Alice"));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn where_in_list() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db
        .sql("SELECT doc FROM t WHERE pk IN ('k1', 'k3');")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
            let strs = rows_to_strings(rows);
            assert!(strs.iter().any(|s| s.contains("Alice")));
            assert!(strs.iter().any(|s| s.contains("Charlie")));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn where_not_expression() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db.sql("SELECT doc FROM t WHERE NOT doc.age > 30;").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
            let strs = rows_to_strings(rows);
            assert!(strs.iter().any(|s| s.contains("Alice")));
            assert!(strs.iter().any(|s| s.contains("Bob")));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- UPDATE tests ---

#[test]
fn update_with_pk_filter() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db
        .sql("UPDATE t SET doc = '{\"name\":\"Alice\",\"age\":31,\"balance\":100.5}' WHERE pk = 'k1';")
        .unwrap();
    match result {
        SqlResult::Affected { rows, .. } => assert_eq!(rows, 1),
        other => panic!("unexpected: {other:?}"),
    }

    let check = db.sql("SELECT doc FROM t WHERE pk = 'k1';").unwrap();
    match check {
        SqlResult::Rows(rows) => {
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("31"));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn update_with_doc_field_filter() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db
        .sql("UPDATE t SET doc = '{\"name\":\"Updated\",\"age\":99,\"balance\":0}' WHERE doc.balance < 100;")
        .unwrap();
    match result {
        SqlResult::Affected { rows, .. } => assert_eq!(rows, 1), // Only Charlie with balance 50
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn update_all_rows() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db.sql("UPDATE t SET doc = '{\"reset\":true}';").unwrap();
    match result {
        SqlResult::Affected { rows, .. } => assert_eq!(rows, 3),
        other => panic!("unexpected: {other:?}"),
    }
}

// --- DELETE tests ---

#[test]
fn delete_with_pk_filter() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db.sql("DELETE FROM t WHERE pk = 'k1';").unwrap();
    match result {
        SqlResult::Affected { rows, .. } => assert_eq!(rows, 1),
        other => panic!("unexpected: {other:?}"),
    }

    let check = db.sql("SELECT count(*) FROM t;").unwrap();
    match check {
        SqlResult::Rows(rows) => {
            assert_eq!(rows_to_strings(rows), vec!["2"]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn delete_with_doc_field_filter() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db.sql("DELETE FROM t WHERE doc.balance < 100;").unwrap();
    match result {
        SqlResult::Affected { rows, .. } => assert_eq!(rows, 1), // Only Charlie
        other => panic!("unexpected: {other:?}"),
    }

    let check = db.sql("SELECT count(*) FROM t;").unwrap();
    match check {
        SqlResult::Rows(rows) => {
            assert_eq!(rows_to_strings(rows), vec!["2"]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn delete_all_rows() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db.sql("DELETE FROM t;").unwrap();
    match result {
        SqlResult::Affected { rows, .. } => assert_eq!(rows, 3),
        other => panic!("unexpected: {other:?}"),
    }

    let check = db.sql("SELECT count(*) FROM t;").unwrap();
    match check {
        SqlResult::Rows(rows) => {
            assert_eq!(rows_to_strings(rows), vec!["0"]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- JOIN tests ---

#[allow(dead_code)]
fn setup_join_tables(db: &Database) {
    db.sql("CREATE TABLE orders (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("CREATE TABLE customers (pk TEXT PRIMARY KEY);")
        .unwrap();

    db.sql("INSERT INTO orders (pk, doc) VALUES ('o1', '{\"customer\":\"c1\",\"amount\":100}');")
        .unwrap();
    db.sql("INSERT INTO orders (pk, doc) VALUES ('o2', '{\"customer\":\"c2\",\"amount\":200}');")
        .unwrap();
    db.sql("INSERT INTO orders (pk, doc) VALUES ('o3', '{\"customer\":\"c1\",\"amount\":50}');")
        .unwrap();

    db.sql("INSERT INTO customers (pk, doc) VALUES ('c1', '{\"name\":\"Alice\"}');")
        .unwrap();
    db.sql("INSERT INTO customers (pk, doc) VALUES ('c2', '{\"name\":\"Bob\"}');")
        .unwrap();
    db.sql("INSERT INTO customers (pk, doc) VALUES ('c3', '{\"name\":\"Charlie\"}');")
        .unwrap();
}

#[test]
fn left_join_includes_unmatched() {
    let (_dir, db) = setup_db();

    db.sql("CREATE TABLE a (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("CREATE TABLE b (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO a (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap();
    db.sql("INSERT INTO a (pk, doc) VALUES ('k2', '{\"v\":2}');")
        .unwrap();
    db.sql("INSERT INTO b (pk, doc) VALUES ('k2', '{\"v\":22}');")
        .unwrap();

    let result = db
        .sql("SELECT pk, doc FROM a LEFT JOIN b ON a.pk = b.pk ORDER BY pk ASC;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
            let s1 = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s1.contains("\"pk\":\"k1\""));
            assert!(s1.contains("null")); // right side is null for k1
            let s2 = String::from_utf8(rows[1].clone()).unwrap();
            assert!(s2.contains("\"pk\":\"k2\""));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn right_join_includes_unmatched() {
    let (_dir, db) = setup_db();

    db.sql("CREATE TABLE a (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("CREATE TABLE b (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO a (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap();
    db.sql("INSERT INTO b (pk, doc) VALUES ('k1', '{\"v\":11}');")
        .unwrap();
    db.sql("INSERT INTO b (pk, doc) VALUES ('k2', '{\"v\":22}');")
        .unwrap();

    let result = db
        .sql("SELECT pk, doc FROM a RIGHT JOIN b ON a.pk = b.pk ORDER BY pk ASC;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn cross_join() {
    let (_dir, db) = setup_db();

    db.sql("CREATE TABLE a (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("CREATE TABLE b (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO a (pk, doc) VALUES ('a1', '{\"v\":1}');")
        .unwrap();
    db.sql("INSERT INTO a (pk, doc) VALUES ('a2', '{\"v\":2}');")
        .unwrap();
    db.sql("INSERT INTO b (pk, doc) VALUES ('b1', '{\"v\":10}');")
        .unwrap();
    db.sql("INSERT INTO b (pk, doc) VALUES ('b2', '{\"v\":20}');")
        .unwrap();

    let result = db.sql("SELECT count(*) FROM a CROSS JOIN b;").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows_to_strings(rows), vec!["4"]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Aggregate tests ---

#[test]
fn aggregate_sum_avg_min_max() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db
        .sql(
            "SELECT sum(doc.balance), avg(doc.balance), min(doc.balance), max(doc.balance) FROM t;",
        )
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("350.5")); // sum = 100.5 + 200.0 + 50.0
            assert!(s.contains("50")); // min
            assert!(s.contains("200")); // max
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn group_by_doc_field() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE sales (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s1', '{\"region\":\"east\",\"amount\":100}');")
        .unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s2', '{\"region\":\"west\",\"amount\":200}');")
        .unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s3', '{\"region\":\"east\",\"amount\":150}');")
        .unwrap();

    let result = db
        .sql("SELECT doc.region, sum(doc.amount) FROM sales GROUP BY doc.region;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
            let strs = rows_to_strings(rows);
            // Should have east=250 and west=200
            let has_east = strs.iter().any(|s| s.contains("east") && s.contains("250"));
            let has_west = strs.iter().any(|s| s.contains("west") && s.contains("200"));
            assert!(has_east, "expected east sum=250 in {:?}", strs);
            assert!(has_west, "expected west sum=200 in {:?}", strs);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn having_filters_groups() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE sales (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s1', '{\"region\":\"east\",\"amount\":100}');")
        .unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s2', '{\"region\":\"west\",\"amount\":200}');")
        .unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s3', '{\"region\":\"east\",\"amount\":150}');")
        .unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s4', '{\"region\":\"east\",\"amount\":50}');")
        .unwrap();

    let result = db
        .sql("SELECT doc.region, count(*) FROM sales GROUP BY doc.region HAVING count(*) > 1;")
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("east"));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- UPDATE/DELETE in transactions ---

#[test]
fn update_delete_in_transaction() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    let result = db.sql("BEGIN; UPDATE t SET doc = '{\"name\":\"Updated\",\"age\":0,\"balance\":0}' WHERE pk = 'k1'; DELETE FROM t WHERE pk = 'k3'; COMMIT;").unwrap();
    match result {
        SqlResult::Affected { rows, .. } => assert!(rows > 0),
        other => panic!("unexpected: {other:?}"),
    }

    let check = db.sql("SELECT count(*) FROM t;").unwrap();
    match check {
        SqlResult::Rows(rows) => {
            assert_eq!(rows_to_strings(rows), vec!["2"]);
        }
        other => panic!("unexpected: {other:?}"),
    }

    let k1 = db.sql("SELECT doc FROM t WHERE pk = 'k1';").unwrap();
    match k1 {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let s = String::from_utf8(rows[0].clone()).unwrap();
            assert!(s.contains("Updated"));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Temporal filters with new features ---

#[test]
fn temporal_where_with_expressions() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    let c1 = match db
        .sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap()
    {
        SqlResult::Affected {
            commit_ts: Some(ts),
            ..
        } => ts,
        other => panic!("unexpected: {other:?}"),
    };

    let _c2 = match db
        .sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"v\":2}');")
        .unwrap()
    {
        SqlResult::Affected {
            commit_ts: Some(ts),
            ..
        } => ts,
        other => panic!("unexpected: {other:?}"),
    };

    // Use AS OF to read old version with expression-based WHERE
    let result = db
        .sql(&format!("SELECT doc FROM t WHERE pk = 'k1' AS OF {c1};"))
        .unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0], br#"{"v":1}"#.to_vec());
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// --- Comparison operators ---

#[test]
fn comparison_operators_all() {
    let (_dir, db) = setup_db();
    setup_test_table(&db);

    // != (not equal)
    let r = db
        .sql("SELECT count(*) FROM t WHERE doc.age != 30;")
        .unwrap();
    match r {
        SqlResult::Rows(rows) => assert_eq!(rows_to_strings(rows), vec!["2"]),
        other => panic!("unexpected: {other:?}"),
    }

    // <= (less than or equal)
    let r = db
        .sql("SELECT count(*) FROM t WHERE doc.age <= 30;")
        .unwrap();
    match r {
        SqlResult::Rows(rows) => assert_eq!(rows_to_strings(rows), vec!["2"]),
        other => panic!("unexpected: {other:?}"),
    }

    // >= (greater than or equal)
    let r = db
        .sql("SELECT count(*) FROM t WHERE doc.age >= 30;")
        .unwrap();
    match r {
        SqlResult::Rows(rows) => assert_eq!(rows_to_strings(rows), vec!["2"]),
        other => panic!("unexpected: {other:?}"),
    }
}
