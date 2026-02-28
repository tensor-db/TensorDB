//! Integration tests for SQL:2011 Temporal Query Clauses
//!
//! Tests FOR SYSTEM_TIME and FOR APPLICATION_TIME parsing and execution.

use tempfile::tempdir;

use tensordb::config::Config;
use tensordb::sql::exec::SqlResult;
use tensordb::Database;

fn setup_db() -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let db = Database::open(
        dir.path(),
        Config {
            shard_count: 1,
            ..Config::default()
        },
    )
    .unwrap();
    (dir, db)
}

fn row_strings(result: &SqlResult) -> Vec<String> {
    match result {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| String::from_utf8_lossy(r).to_string())
            .collect(),
        _ => panic!("expected Rows, got {result:?}"),
    }
}

// ---------- Parsing tests ----------

#[test]
fn parse_for_system_time_as_of() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap();

    // FOR SYSTEM_TIME AS OF should parse and execute
    let result = db
        .sql("SELECT doc FROM t FOR SYSTEM_TIME AS OF 999999;")
        .unwrap();
    // With a huge system time, we should see data written before that time
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
}

#[test]
fn parse_for_system_time_from_to() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    let ct = match db
        .sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap()
    {
        SqlResult::Affected { commit_ts, .. } => commit_ts.unwrap(),
        _ => panic!("expected Affected"),
    };

    // FROM ct TO ct+100 — start of range = ct, so data at ct is visible
    let result = db
        .sql(&format!(
            "SELECT doc FROM t FOR SYSTEM_TIME FROM {ct} TO {};",
            ct + 100
        ))
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
}

#[test]
fn parse_for_system_time_between_and() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    let ct = match db
        .sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap()
    {
        SqlResult::Affected { commit_ts, .. } => commit_ts.unwrap(),
        _ => panic!("expected Affected"),
    };

    // BETWEEN ct AND ct+100 — start of range = ct
    let result = db
        .sql(&format!(
            "SELECT doc FROM t FOR SYSTEM_TIME BETWEEN {ct} AND {};",
            ct + 100
        ))
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
}

#[test]
fn parse_for_system_time_all() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap();

    // SYSTEM_TIME ALL means no system time filter — returns latest
    let result = db.sql("SELECT doc FROM t FOR SYSTEM_TIME ALL;").unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
}

#[test]
fn parse_for_application_time_as_of() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    // Insert with business time validity via the low-level API,
    // then query via SQL with APPLICATION_TIME
    let key = b"table/t/k1";
    db.put(
        key,
        br#"{"v":1}"#.to_vec(),
        100, // valid_from
        200, // valid_to
        None,
    )
    .unwrap();

    // Query at application time 150 (within validity)
    let result = db
        .sql("SELECT doc FROM t FOR APPLICATION_TIME AS OF 150;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("\"v\":1"));
}

#[test]
fn parse_for_application_time_from_to() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap();

    // APPLICATION_TIME FROM .. TO should parse correctly
    let result = db
        .sql("SELECT doc FROM t FOR APPLICATION_TIME FROM 0 TO 999999;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
}

#[test]
fn parse_for_application_time_between_and() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap();

    let result = db
        .sql("SELECT doc FROM t FOR APPLICATION_TIME BETWEEN 0 AND 999999;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
}

// ---------- Semantic tests ----------

#[test]
fn system_time_as_of_point_in_time() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE events (pk TEXT PRIMARY KEY);")
        .unwrap();

    let ct1 = match db
        .sql("INSERT INTO events (pk, doc) VALUES ('e1', '{\"v\":1}');")
        .unwrap()
    {
        SqlResult::Affected { commit_ts, .. } => commit_ts.unwrap(),
        _ => panic!("expected Affected"),
    };

    let _ct2 = match db
        .sql("INSERT INTO events (pk, doc) VALUES ('e2', '{\"v\":2}');")
        .unwrap()
    {
        SqlResult::Affected { commit_ts, .. } => commit_ts.unwrap(),
        _ => panic!("expected Affected"),
    };

    // At system time ct1, only e1 should be visible
    let result = db
        .sql(&format!(
            "SELECT doc FROM events FOR SYSTEM_TIME AS OF {ct1};"
        ))
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("\"v\":1"));
}

#[test]
fn combined_system_and_application_time() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE ledger (pk TEXT PRIMARY KEY);")
        .unwrap();

    // Insert with specific business time validity
    let key = b"table/ledger/txn1";
    let ct1 = db
        .put(key, br#"{"amount":100}"#.to_vec(), 1000, 2000, None)
        .unwrap();

    // Both temporal clauses: system time + application time
    let result = db
        .sql(&format!(
            "SELECT doc FROM ledger FOR SYSTEM_TIME AS OF {ct1} FOR APPLICATION_TIME AS OF 1500;"
        ))
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("100"));
}

#[test]
fn temporal_clause_with_where() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE accounts (id INTEGER PRIMARY KEY, name TEXT, balance REAL);")
        .unwrap();
    db.sql("INSERT INTO accounts (id, name, balance) VALUES (1, 'alice', 1000.0);")
        .unwrap();
    db.sql("INSERT INTO accounts (id, name, balance) VALUES (2, 'bob', 500.0);")
        .unwrap();

    // Temporal clause combined with WHERE filter
    let result = db
        .sql("SELECT name FROM accounts WHERE balance > 600 FOR SYSTEM_TIME ALL;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("alice"));
}

#[test]
fn temporal_parse_error_invalid_keyword() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    // Invalid temporal keyword after FOR
    let result = db.sql("SELECT doc FROM t FOR INVALID_TIME AS OF 100;");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("SYSTEM_TIME") || err.contains("APPLICATION_TIME"));
}

#[test]
fn temporal_parse_error_missing_value() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    // SYSTEM_TIME without AS OF/FROM/BETWEEN/ALL
    let result = db.sql("SELECT doc FROM t FOR SYSTEM_TIME;");
    assert!(result.is_err());
}

#[test]
fn temporal_with_order_by() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('a', '{\"n\":1}');")
        .unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('b', '{\"n\":2}');")
        .unwrap();

    let result = db
        .sql("SELECT doc FROM t FOR SYSTEM_TIME ALL ORDER BY pk DESC;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 2);
    // DESC order: b first, then a
    assert!(rows[0].contains("\"n\":2"));
    assert!(rows[1].contains("\"n\":1"));
}

#[test]
fn temporal_with_limit() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('a', '{\"n\":1}');")
        .unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('b', '{\"n\":2}');")
        .unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('c', '{\"n\":3}');")
        .unwrap();

    let result = db
        .sql("SELECT doc FROM t FOR SYSTEM_TIME ALL ORDER BY pk ASC LIMIT 2;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 2);
}

#[test]
fn explain_with_temporal_clause() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    let result = db
        .sql("EXPLAIN SELECT doc FROM t FOR SYSTEM_TIME AS OF 100;")
        .unwrap();
    if let SqlResult::Explain(text) = result {
        assert!(text.contains("Plan") || text.contains("Scan"));
    } else {
        panic!("expected Explain result");
    }
}
