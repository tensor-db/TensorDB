use tempfile::tempdir;

use tensordb::config::Config;
use tensordb::sql::exec::SqlResult;
use tensordb::Database;

fn rows_to_strings(rows: Vec<Vec<u8>>) -> Vec<String> {
    rows.into_iter()
        .map(|row| String::from_utf8(row).unwrap())
        .collect()
}

#[test]
fn sql_create_insert_select_as_of_explain() {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 1,
        ..Config::default()
    };
    let db = Database::open(dir.path(), cfg).unwrap();

    let create = db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    match create {
        SqlResult::Affected { rows, .. } => assert_eq!(rows, 1),
        _ => panic!("unexpected result for create"),
    }

    let c1 = match db
        .sql("INSERT INTO t (pk, doc) VALUES ('k', '{\"v\":1}');")
        .unwrap()
    {
        SqlResult::Affected {
            commit_ts: Some(ts),
            ..
        } => ts,
        other => panic!("unexpected insert result: {other:?}"),
    };

    let c2 = match db
        .sql("INSERT INTO t (pk, doc) VALUES ('k', '{\"v\":2}');")
        .unwrap()
    {
        SqlResult::Affected {
            commit_ts: Some(ts),
            ..
        } => ts,
        other => panic!("unexpected insert result: {other:?}"),
    };

    assert!(c2 > c1);

    let latest = db.sql("SELECT doc FROM t WHERE pk='k';").unwrap();
    match latest {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0], br#"{"v":2}"#.to_vec());
        }
        _ => panic!("unexpected latest select result"),
    }

    let old = db
        .sql(&format!("SELECT doc FROM t WHERE pk='k' AS OF {c1};"))
        .unwrap();
    match old {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0], br#"{"v":1}"#.to_vec());
        }
        _ => panic!("unexpected as-of select result"),
    }

    let explain = db
        .sql(&format!(
            "EXPLAIN SELECT doc FROM t WHERE pk='k' AS OF {c2} VALID AT 0;"
        ))
        .unwrap();
    match explain {
        SqlResult::Explain(line) => {
            assert!(line.contains("shard_id="));
            assert!(line.contains("bloom_hit="));
            assert!(line.contains("sstable_block="));
            assert!(line.contains("commit_ts_used="));
        }
        _ => panic!("unexpected explain result"),
    }
}

#[test]
fn sql_insert_requires_existing_table() {
    let dir = tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();

    let err = db
        .sql("INSERT INTO missing (pk, doc) VALUES ('k', '{\"v\":1}');")
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("does not exist"));
}

#[test]
fn sql_select_scan_pk_doc_count_order_limit() {
    let dir = tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();

    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    let c1 = match db
        .sql("INSERT INTO t (pk, doc) VALUES ('k2', '{\"v\":2}');")
        .unwrap()
    {
        SqlResult::Affected {
            commit_ts: Some(ts),
            ..
        } => ts,
        other => panic!("unexpected insert result: {other:?}"),
    };
    let c2 = match db
        .sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"v\":1}');")
        .unwrap()
    {
        SqlResult::Affected {
            commit_ts: Some(ts),
            ..
        } => ts,
        other => panic!("unexpected insert result: {other:?}"),
    };
    assert_ne!(c2, 0);

    let scan = db.sql("SELECT doc FROM t;").unwrap();
    match scan {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
            assert!(rows.contains(&br#"{"v":1}"#.to_vec()));
            assert!(rows.contains(&br#"{"v":2}"#.to_vec()));
        }
        other => panic!("unexpected scan result: {other:?}"),
    }

    let ordered = db
        .sql("SELECT doc FROM t ORDER BY pk DESC LIMIT 1;")
        .unwrap();
    match ordered {
        SqlResult::Rows(rows) => {
            assert_eq!(rows, vec![br#"{"v":2}"#.to_vec()]);
        }
        other => panic!("unexpected ordered result: {other:?}"),
    }

    let pk_doc = db
        .sql("SELECT pk, doc FROM t WHERE pk='k1' ORDER BY pk ASC LIMIT 5;")
        .unwrap();
    match pk_doc {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let row = String::from_utf8(rows[0].clone()).unwrap();
            assert!(row.contains("\"pk\":\"k1\""));
            assert!(row.contains("\"doc\":{\"v\":1}"));
        }
        other => panic!("unexpected pk/doc result: {other:?}"),
    }

    let count = db.sql("SELECT count(*) FROM t;").unwrap();
    match count {
        SqlResult::Rows(rows) => assert_eq!(rows_to_strings(rows), vec!["2".to_string()]),
        other => panic!("unexpected count result: {other:?}"),
    }

    let old_count = db
        .sql(&format!(
            "SELECT count(*) FROM t WHERE pk='k2' AS OF {c1} VALID AT 0;"
        ))
        .unwrap();
    match old_count {
        SqlResult::Rows(rows) => assert_eq!(rows_to_strings(rows), vec!["1".to_string()]),
        other => panic!("unexpected as-of count result: {other:?}"),
    }
}

#[test]
fn sql_show_describe_and_drop() {
    let dir = tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();

    db.sql("CREATE TABLE users (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("CREATE TABLE orders (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("INSERT INTO orders (pk, doc) VALUES ('o1', '{\"v\":1}');")
        .unwrap();
    db.sql("CREATE VIEW v_user AS SELECT doc FROM users WHERE pk='u1';")
        .unwrap();
    db.sql("CREATE INDEX users_pk_idx ON users (pk);").unwrap();

    let tables = db.sql("SHOW TABLES;").unwrap();
    match tables {
        SqlResult::Rows(rows) => {
            let names = rows_to_strings(rows);
            assert_eq!(names, vec!["orders".to_string(), "users".to_string()]);
        }
        other => panic!("unexpected show tables result: {other:?}"),
    }

    let describe = db.sql("DESCRIBE users;").unwrap();
    match describe {
        SqlResult::Rows(rows) => {
            assert!(!rows.is_empty());
            let lines = rows_to_strings(rows);
            assert!(lines.iter().any(|line| line.contains("\"column\":\"pk\"")));
            assert!(lines.iter().any(|line| line.contains("\"column\":\"doc\"")));
        }
        other => panic!("unexpected describe result: {other:?}"),
    }

    db.sql("DROP INDEX users_pk_idx ON users;").unwrap();
    db.sql("CREATE INDEX users_pk_idx ON users (pk);").unwrap();

    db.sql("DROP VIEW v_user;").unwrap();
    let select_view_err = db.sql("SELECT doc FROM v_user;").unwrap_err();
    assert!(format!("{select_view_err}").contains("does not exist"));

    db.sql("DROP TABLE orders;").unwrap();
    let describe_err = db.sql("DESCRIBE orders;").unwrap_err();
    assert!(format!("{describe_err}").contains("does not exist"));

    db.sql("CREATE TABLE orders (pk TEXT PRIMARY KEY);")
        .unwrap();
    let recreated = db.sql("SELECT doc FROM orders WHERE pk='o1';").unwrap();
    match recreated {
        SqlResult::Rows(rows) => assert!(rows.is_empty()),
        other => panic!("unexpected recreate select result: {other:?}"),
    }
    let tables_after = db.sql("SHOW TABLES;").unwrap();
    match tables_after {
        SqlResult::Rows(rows) => {
            let names = rows_to_strings(rows);
            assert_eq!(names, vec!["orders".to_string(), "users".to_string()]);
        }
        other => panic!("unexpected show tables result: {other:?}"),
    }
}

#[test]
fn sql_join_and_group_by_skeleton() {
    let dir = tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();

    db.sql("CREATE TABLE left_t (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("CREATE TABLE right_t (pk TEXT PRIMARY KEY);")
        .unwrap();

    db.sql("INSERT INTO left_t (pk, doc) VALUES ('k1', '{\"l\":1}');")
        .unwrap();
    db.sql("INSERT INTO left_t (pk, doc) VALUES ('k2', '{\"l\":2}');")
        .unwrap();
    db.sql("INSERT INTO right_t (pk, doc) VALUES ('k2', '{\"r\":2}');")
        .unwrap();
    db.sql("INSERT INTO right_t (pk, doc) VALUES ('k3', '{\"r\":3}');")
        .unwrap();

    let join_rows = db
        .sql("SELECT pk, doc FROM left_t JOIN right_t ON left_t.pk=right_t.pk ORDER BY pk ASC;")
        .unwrap();
    match join_rows {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let row = String::from_utf8(rows[0].clone()).unwrap();
            assert!(row.contains("\"pk\":\"k2\""));
            assert!(row.contains("\"left_doc\":{\"l\":2}"));
            assert!(row.contains("\"right_doc\":{\"r\":2}"));
        }
        other => panic!("unexpected join result: {other:?}"),
    }

    let join_count = db
        .sql("SELECT count(*) FROM left_t JOIN right_t ON left_t.pk=right_t.pk;")
        .unwrap();
    match join_count {
        SqlResult::Rows(rows) => assert_eq!(rows_to_strings(rows), vec!["1".to_string()]),
        other => panic!("unexpected join count result: {other:?}"),
    }

    let grouped = db
        .sql("SELECT pk, count(*) FROM left_t JOIN right_t ON left_t.pk=right_t.pk GROUP BY pk;")
        .unwrap();
    match grouped {
        SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let row = String::from_utf8(rows[0].clone()).unwrap();
            assert!(row.contains("\"pk\":\"k2\""));
            assert!(row.contains("\"count\":1"));
        }
        other => panic!("unexpected grouped result: {other:?}"),
    }
}
