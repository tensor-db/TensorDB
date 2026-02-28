use tempfile::tempdir;

use tensordb::config::Config;
use tensordb::sql::exec::{SqlResult, TableStats};
use tensordb::{Database, PreparedStatement};

fn setup_db() -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 1,
        ..Config::default()
    };
    let db = Database::open(dir.path(), cfg).unwrap();
    (dir, db)
}

fn rows_to_strings(rows: Vec<Vec<u8>>) -> Vec<String> {
    rows.into_iter()
        .map(|row| String::from_utf8(row).unwrap())
        .collect()
}

// ---- ANALYZE tests ----

#[test]
fn analyze_collects_table_stats() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE stats_t (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();
    db.sql("INSERT INTO stats_t (id, name) VALUES (1, 'Alice');")
        .unwrap();
    db.sql("INSERT INTO stats_t (id, name) VALUES (2, 'Bob');")
        .unwrap();
    db.sql("INSERT INTO stats_t (id, name) VALUES (3, 'Carol');")
        .unwrap();

    let result = db.sql("ANALYZE stats_t;").unwrap();
    match result {
        SqlResult::Affected { rows, message, .. } => {
            assert_eq!(rows, 3);
            assert!(message.contains("ANALYZE stats_t"));
            assert!(message.contains("3 rows"));
        }
        other => panic!("unexpected ANALYZE result: {other:?}"),
    }

    // Verify stats are persisted under __meta/stats/stats_t
    let raw = db
        .get(b"__meta/stats/stats_t", None, None)
        .unwrap()
        .expect("stats should be stored");
    let stats: TableStats = serde_json::from_slice(&raw).unwrap();
    assert_eq!(stats.row_count, 3);
    assert!(stats.approx_byte_size > 0);
    assert!(stats.last_updated_ms > 0);
}

#[test]
fn analyze_nonexistent_table_errors() {
    let (_dir, db) = setup_db();
    let err = db.sql("ANALYZE no_such_table;").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("does not exist"), "got: {msg}");
}

#[test]
fn analyze_updates_stats_after_more_inserts() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE grow_t (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("INSERT INTO grow_t (pk, doc) VALUES ('a', '{}');")
        .unwrap();

    let result = db.sql("ANALYZE grow_t;").unwrap();
    match &result {
        SqlResult::Affected { rows, .. } => assert_eq!(*rows, 1),
        other => panic!("unexpected: {other:?}"),
    }

    // Insert more
    db.sql("INSERT INTO grow_t (pk, doc) VALUES ('b', '{}');")
        .unwrap();
    db.sql("INSERT INTO grow_t (pk, doc) VALUES ('c', '{}');")
        .unwrap();

    let result = db.sql("ANALYZE grow_t;").unwrap();
    match &result {
        SqlResult::Affected { rows, .. } => assert_eq!(*rows, 3),
        other => panic!("unexpected: {other:?}"),
    }

    let raw = db
        .get(b"__meta/stats/grow_t", None, None)
        .unwrap()
        .expect("stats should be stored");
    let stats: TableStats = serde_json::from_slice(&raw).unwrap();
    assert_eq!(stats.row_count, 3);
}

// ---- EXPLAIN ANALYZE tests ----

#[test]
fn explain_analyze_select() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE ea_t (id INTEGER PRIMARY KEY, val TEXT);")
        .unwrap();
    db.sql("INSERT INTO ea_t (id, val) VALUES (1, 'x');")
        .unwrap();
    db.sql("INSERT INTO ea_t (id, val) VALUES (2, 'y');")
        .unwrap();
    db.sql("INSERT INTO ea_t (id, val) VALUES (3, 'z');")
        .unwrap();

    let result = db.sql("EXPLAIN ANALYZE SELECT val FROM ea_t;").unwrap();
    match result {
        SqlResult::Explain(text) => {
            assert!(
                text.contains("execution_time_us="),
                "should report execution time, got: {text}"
            );
            assert!(
                text.contains("rows_returned=3"),
                "should report 3 rows returned, got: {text}"
            );
        }
        other => panic!("unexpected EXPLAIN ANALYZE result: {other:?}"),
    }
}

#[test]
fn explain_analyze_insert() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE ea_ins (pk TEXT PRIMARY KEY);")
        .unwrap();

    let result = db
        .sql("EXPLAIN ANALYZE INSERT INTO ea_ins (pk, doc) VALUES ('k', '{}');")
        .unwrap();
    match result {
        SqlResult::Explain(text) => {
            assert!(
                text.contains("execution_time_us="),
                "should report execution time, got: {text}"
            );
            assert!(
                text.contains("rows_returned=1"),
                "should report 1 affected row, got: {text}"
            );
        }
        other => panic!("unexpected EXPLAIN ANALYZE result: {other:?}"),
    }
}

#[test]
fn explain_analyze_with_filter() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE ea_filt (id INTEGER PRIMARY KEY, val INTEGER);")
        .unwrap();
    for i in 1..=10 {
        db.sql(&format!(
            "INSERT INTO ea_filt (id, val) VALUES ({i}, {});",
            i * 10
        ))
        .unwrap();
    }

    let result = db
        .sql("EXPLAIN ANALYZE SELECT val FROM ea_filt WHERE val > 50;")
        .unwrap();
    match result {
        SqlResult::Explain(text) => {
            assert!(text.contains("execution_time_us="), "got: {text}");
            // val > 50 means val in {60, 70, 80, 90, 100} = 5 rows
            assert!(text.contains("rows_returned=5"), "got: {text}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// ---- PreparedStatement tests ----

#[test]
fn prepared_statement_basic() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE prep_t (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();
    db.sql("INSERT INTO prep_t (id, name) VALUES (1, 'Alice');")
        .unwrap();
    db.sql("INSERT INTO prep_t (id, name) VALUES (2, 'Bob');")
        .unwrap();

    let prepared = db
        .prepare("SELECT name FROM prep_t ORDER BY id ASC;")
        .unwrap();

    // Execute the same prepared statement multiple times
    for _ in 0..3 {
        let result = prepared.execute(&db).unwrap();
        match result {
            SqlResult::Rows(rows) => {
                let strs = rows_to_strings(rows);
                assert_eq!(strs, vec![r#"{"name":"Alice"}"#, r#"{"name":"Bob"}"#]);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}

#[test]
fn prepared_statement_sees_new_data() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE prep_live (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("INSERT INTO prep_live (pk, doc) VALUES ('a', '{\"v\":1}');")
        .unwrap();

    let prepared = db
        .prepare("SELECT doc FROM prep_live ORDER BY pk;")
        .unwrap();

    let result = prepared.execute(&db).unwrap();
    match &result {
        SqlResult::Rows(rows) => assert_eq!(rows.len(), 1),
        other => panic!("unexpected: {other:?}"),
    }

    // Insert more data after prepare
    db.sql("INSERT INTO prep_live (pk, doc) VALUES ('b', '{\"v\":2}');")
        .unwrap();

    // The prepared statement should see the new data
    let result = prepared.execute(&db).unwrap();
    match result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
            let strs = rows_to_strings(rows);
            assert_eq!(strs, vec![r#"{"v":1}"#, r#"{"v":2}"#]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn prepared_statement_parse_error() {
    let (_dir, db) = setup_db();
    let err = db.prepare("GIBBERISH STATEMENT;").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("sql parse error"), "got: {msg}");
}

#[test]
fn prepared_statement_insert() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE prep_ins (pk TEXT PRIMARY KEY);")
        .unwrap();

    let prepared = db
        .prepare("INSERT INTO prep_ins (pk, doc) VALUES ('x', '{\"n\":1}');")
        .unwrap();

    let result = prepared.execute(&db).unwrap();
    match result {
        SqlResult::Affected { rows, .. } => assert_eq!(rows, 1),
        other => panic!("unexpected: {other:?}"),
    }

    // Verify data was actually inserted
    let read_result = db.sql("SELECT doc FROM prep_ins WHERE pk='x';").unwrap();
    match read_result {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0], br#"{"n":1}"#.to_vec());
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// ---- Cost-based EXPLAIN tests ----

#[test]
fn explain_shows_plan_tree() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE plan_t (id INTEGER PRIMARY KEY, val TEXT);")
        .unwrap();
    db.sql("INSERT INTO plan_t (id, val) VALUES (1, 'a');")
        .unwrap();

    let result = db.sql("EXPLAIN SELECT val FROM plan_t;").unwrap();
    match result {
        SqlResult::Explain(text) => {
            assert!(text.contains("Query Plan:"), "got: {text}");
            assert!(text.contains("FullScan"), "got: {text}");
            assert!(text.contains("Estimated cost:"), "got: {text}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn explain_point_lookup_shows_plan_and_storage() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE plan_pk (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("INSERT INTO plan_pk (pk, doc) VALUES ('x', '{\"v\":1}');")
        .unwrap();

    let result = db
        .sql("EXPLAIN SELECT doc FROM plan_pk WHERE pk='x';")
        .unwrap();
    match result {
        SqlResult::Explain(text) => {
            assert!(text.contains("PointLookup"), "got: {text}");
            assert!(
                text.contains("Storage:"),
                "plan should include storage info, got: {text}"
            );
            assert!(text.contains("shard_id="), "got: {text}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn explain_with_order_by_shows_sort() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE sort_t (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();

    let result = db
        .sql("EXPLAIN SELECT name FROM sort_t ORDER BY name ASC;")
        .unwrap();
    match result {
        SqlResult::Explain(text) => {
            assert!(text.contains("Sort"), "got: {text}");
            assert!(text.contains("FullScan"), "got: {text}");
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// ---- Enhanced EXPLAIN ANALYZE tests ----

#[test]
fn explain_analyze_shows_operations() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE ea_ops (id INTEGER PRIMARY KEY, val TEXT);")
        .unwrap();
    for i in 1..=5 {
        db.sql(&format!(
            "INSERT INTO ea_ops (id, val) VALUES ({i}, 'v{i}');"
        ))
        .unwrap();
    }

    let result = db.sql("EXPLAIN ANALYZE SELECT val FROM ea_ops;").unwrap();
    match result {
        SqlResult::Explain(text) => {
            assert!(text.contains("execution_time_us="), "got: {text}");
            assert!(text.contains("rows_returned=5"), "got: {text}");
            assert!(
                text.contains("operations:"),
                "should show operation counts, got: {text}"
            );
            assert!(
                text.contains("plan_cost="),
                "should show plan cost, got: {text}"
            );
        }
        other => panic!("unexpected: {other:?}"),
    }
}

// ---- Enhanced TableStats with column-level stats ----

#[test]
fn analyze_typed_table_produces_column_stats() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE col_t (id INTEGER PRIMARY KEY, name TEXT, score REAL);")
        .unwrap();
    db.sql("INSERT INTO col_t (id, name, score) VALUES (1, 'Alice', 95.5);")
        .unwrap();
    db.sql("INSERT INTO col_t (id, name, score) VALUES (2, 'Bob', 87.0);")
        .unwrap();
    db.sql("INSERT INTO col_t (id, name, score) VALUES (3, 'Alice', 92.0);")
        .unwrap();

    db.sql("ANALYZE col_t;").unwrap();

    let raw = db
        .get(b"__meta/stats/col_t", None, None)
        .unwrap()
        .expect("stats should be stored");
    let stats: TableStats = serde_json::from_slice(&raw).unwrap();

    assert_eq!(stats.row_count, 3);
    assert!(stats.avg_row_bytes > 0);
    assert_eq!(
        stats.columns.len(),
        3,
        "should have stats for id, name, score"
    );

    // Check 'name' column stats
    let name_stats = stats.columns.iter().find(|c| c.name == "name").unwrap();
    assert_eq!(name_stats.distinct_count, 2, "Alice appears twice");
    assert_eq!(name_stats.null_count, 0);
    assert!(name_stats.top_values.len() >= 2);
    // Alice should be the most frequent
    assert_eq!(name_stats.top_values[0].0, "Alice");
    assert_eq!(name_stats.top_values[0].1, 2);
}

// ---- Prepared Statement parameter binding tests ----

#[test]
fn prepared_statement_with_params() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE param_t (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();
    db.sql("INSERT INTO param_t (id, name) VALUES (1, 'Alice');")
        .unwrap();
    db.sql("INSERT INTO param_t (id, name) VALUES (2, 'Bob');")
        .unwrap();
    db.sql("INSERT INTO param_t (id, name) VALUES (3, 'Carol');")
        .unwrap();

    // Use $1 as a parameter placeholder (parsed as column name, substituted at execute time)
    let prepared = PreparedStatement::new("SELECT name FROM param_t WHERE name = $1;").unwrap();
    assert_eq!(prepared.param_count(), 1);

    let result = prepared.execute_with_params(&db, &["Alice"]).unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strs = rows_to_strings(rows);
            assert_eq!(strs, vec![r#"{"name":"Alice"}"#]);
        }
        other => panic!("unexpected: {other:?}"),
    }

    // Same prepared statement, different param
    let result = prepared.execute_with_params(&db, &["Bob"]).unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strs = rows_to_strings(rows);
            assert_eq!(strs, vec![r#"{"name":"Bob"}"#]);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn prepared_statement_param_count_mismatch() {
    let prepared = PreparedStatement::new("SELECT * FROM t WHERE a = $1 AND b = $2;").unwrap();
    assert_eq!(prepared.param_count(), 2);

    let (_dir, db) = setup_db();
    let err = prepared
        .execute_with_params(&db, &["only_one"])
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("expected 2 parameters"), "got: {msg}");
}
