//! Integration tests for v0.14 Time-Series SQL Integration
//!
//! Tests CREATE TIMESERIES TABLE, time_bucket(), time_bucket_gapfill(),
//! FIRST(), LAST(), and time-series bucket storage.

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

fn parse_json(s: &str) -> serde_json::Value {
    serde_json::from_str(s).unwrap()
}

// ---------- CREATE TIMESERIES TABLE ----------

#[test]
fn create_timeseries_table_basic() {
    let (_dir, db) = setup_db();
    let result = db
        .sql("CREATE TIMESERIES TABLE metrics (id INTEGER PRIMARY KEY, ts INTEGER, value REAL) WITH (bucket_size = '1h');")
        .unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("created timeseries table metrics"));
        assert!(message.contains("bucket_size=1h"));
        assert!(message.contains("ts_column=ts"));
    } else {
        panic!("expected Affected result");
    }
}

#[test]
fn create_timeseries_table_default_bucket() {
    let (_dir, db) = setup_db();
    let result = db
        .sql(
            "CREATE TIMESERIES TABLE temps (id INTEGER PRIMARY KEY, ts INTEGER, temperature REAL);",
        )
        .unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("created timeseries table temps"));
        assert!(message.contains("bucket_size=1h")); // default
    } else {
        panic!("expected Affected result");
    }
}

#[test]
fn create_timeseries_table_custom_bucket() {
    let (_dir, db) = setup_db();
    let result = db
        .sql("CREATE TIMESERIES TABLE fast_metrics (id INTEGER PRIMARY KEY, ts INTEGER, cpu REAL) WITH (bucket_size = '5m');")
        .unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("bucket_size=5m"));
    } else {
        panic!("expected Affected result");
    }
}

#[test]
fn create_timeseries_table_duplicate_fails() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TIMESERIES TABLE metrics (id INTEGER PRIMARY KEY, ts INTEGER, value REAL);")
        .unwrap();
    let result =
        db.sql("CREATE TIMESERIES TABLE metrics (id INTEGER PRIMARY KEY, ts INTEGER, value REAL);");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}

// ---------- INSERT and time-series bucket storage ----------

#[test]
fn insert_into_timeseries_table() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TIMESERIES TABLE metrics (id INTEGER PRIMARY KEY, ts INTEGER, value REAL) WITH (bucket_size = '3600');")
        .unwrap();

    db.sql("INSERT INTO metrics (id, ts, value) VALUES (1, 1000, 42.5);")
        .unwrap();
    db.sql("INSERT INTO metrics (id, ts, value) VALUES (2, 2000, 55.0);")
        .unwrap();
    db.sql("INSERT INTO metrics (id, ts, value) VALUES (3, 5000, 100.0);")
        .unwrap();

    // Verify the rows are readable
    let result = db.sql("SELECT value FROM metrics;").unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 3);
}

// ---------- time_bucket() function ----------

#[test]
fn time_bucket_basic() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE events (id INTEGER PRIMARY KEY, ts INTEGER, count INTEGER);")
        .unwrap();
    db.sql("INSERT INTO events (id, ts, count) VALUES (1, 100, 5);")
        .unwrap();
    db.sql("INSERT INTO events (id, ts, count) VALUES (2, 200, 10);")
        .unwrap();
    db.sql("INSERT INTO events (id, ts, count) VALUES (3, 3700, 20);")
        .unwrap();
    db.sql("INSERT INTO events (id, ts, count) VALUES (4, 3800, 15);")
        .unwrap();

    // time_bucket('1h', ts) should group 100 and 200 into bucket 0,
    // and 3700 and 3800 into bucket 3600
    let result = db
        .sql("SELECT time_bucket('1h', ts) AS bucket, SUM(count) AS total FROM events GROUP BY time_bucket('1h', ts) ORDER BY time_bucket('1h', ts) ASC;")
        .unwrap();
    let rows = row_strings(&result);

    assert_eq!(rows.len(), 2);

    let r0 = parse_json(&rows[0]);
    assert_eq!(r0["bucket"], 0.0); // bucket at 0s
    assert_eq!(r0["total"], 15.0); // 5 + 10

    let r1 = parse_json(&rows[1]);
    assert_eq!(r1["bucket"], 3600.0); // bucket at 3600s
    assert_eq!(r1["total"], 35.0); // 20 + 15
}

#[test]
fn time_bucket_with_avg() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE readings (id INTEGER PRIMARY KEY, ts INTEGER, temp REAL);")
        .unwrap();
    db.sql("INSERT INTO readings (id, ts, temp) VALUES (1, 0, 20.0);")
        .unwrap();
    db.sql("INSERT INTO readings (id, ts, temp) VALUES (2, 30, 22.0);")
        .unwrap();
    db.sql("INSERT INTO readings (id, ts, temp) VALUES (3, 60, 24.0);")
        .unwrap();

    // 1-minute buckets
    let result = db
        .sql("SELECT time_bucket('1m', ts) AS bucket, AVG(temp) AS avg_temp FROM readings GROUP BY time_bucket('1m', ts) ORDER BY time_bucket('1m', ts) ASC;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 2);

    let r0 = parse_json(&rows[0]);
    assert_eq!(r0["bucket"], 0.0);
    assert_eq!(r0["avg_temp"], 21.0); // avg(20, 22) = 21

    let r1 = parse_json(&rows[1]);
    assert_eq!(r1["bucket"], 60.0);
    assert_eq!(r1["avg_temp"], 24.0);
}

// ---------- FIRST() and LAST() aggregates ----------

#[test]
fn first_last_aggregates() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE sensor (id INTEGER PRIMARY KEY, ts INTEGER, value REAL);")
        .unwrap();
    db.sql("INSERT INTO sensor (id, ts, value) VALUES (1, 100, 10.0);")
        .unwrap();
    db.sql("INSERT INTO sensor (id, ts, value) VALUES (2, 300, 30.0);")
        .unwrap();
    db.sql("INSERT INTO sensor (id, ts, value) VALUES (3, 200, 20.0);")
        .unwrap();

    let result = db
        .sql("SELECT FIRST(value, ts), LAST(value, ts) FROM sensor;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    let r = parse_json(&rows[0]);
    assert_eq!(r["first"], 10.0); // value at ts=100 (smallest)
    assert_eq!(r["last"], 30.0); // value at ts=300 (largest)
}

#[test]
fn first_last_with_group_by() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE sensor (id INTEGER PRIMARY KEY, ts INTEGER, value REAL, device TEXT);")
        .unwrap();
    db.sql("INSERT INTO sensor (id, ts, value, device) VALUES (1, 100, 10.0, 'A');")
        .unwrap();
    db.sql("INSERT INTO sensor (id, ts, value, device) VALUES (2, 200, 20.0, 'A');")
        .unwrap();
    db.sql("INSERT INTO sensor (id, ts, value, device) VALUES (3, 100, 100.0, 'B');")
        .unwrap();
    db.sql("INSERT INTO sensor (id, ts, value, device) VALUES (4, 200, 200.0, 'B');")
        .unwrap();

    let result = db
        .sql("SELECT device, FIRST(value, ts) AS first_val, LAST(value, ts) AS last_val FROM sensor GROUP BY device ORDER BY device ASC;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 2);

    let r0 = parse_json(&rows[0]);
    assert_eq!(r0["device"], "A");
    assert_eq!(r0["first_val"], 10.0);
    assert_eq!(r0["last_val"], 20.0);

    let r1 = parse_json(&rows[1]);
    assert_eq!(r1["device"], "B");
    assert_eq!(r1["first_val"], 100.0);
    assert_eq!(r1["last_val"], 200.0);
}

// ---------- time_bucket_gapfill ----------

#[test]
fn time_bucket_gapfill_fills_gaps() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE metrics (id INTEGER PRIMARY KEY, ts INTEGER, value REAL);")
        .unwrap();

    // Insert data into buckets 0 and 120 (60-second buckets), skipping bucket 60
    db.sql("INSERT INTO metrics (id, ts, value) VALUES (1, 10, 1.0);")
        .unwrap();
    db.sql("INSERT INTO metrics (id, ts, value) VALUES (2, 130, 3.0);")
        .unwrap();

    let result = db
        .sql("SELECT time_bucket_gapfill('1m', ts) AS bucket, AVG(value) AS avg_val FROM metrics GROUP BY time_bucket_gapfill('1m', ts) ORDER BY time_bucket_gapfill('1m', ts) ASC;")
        .unwrap();
    let rows = row_strings(&result);

    // Should have 3 buckets: 0, 60, 120
    assert_eq!(rows.len(), 3);

    let r0 = parse_json(&rows[0]);
    assert_eq!(r0["bucket"].as_f64().unwrap() as u64, 0);
    assert_eq!(r0["avg_val"], 1.0);

    let r1 = parse_json(&rows[1]);
    assert_eq!(r1["bucket"].as_f64().unwrap() as u64, 60);
    assert!(r1["avg_val"].is_null()); // Gap — no data

    let r2 = parse_json(&rows[2]);
    assert_eq!(r2["bucket"].as_f64().unwrap() as u64, 120);
    assert_eq!(r2["avg_val"], 3.0);
}

// ---------- parse_interval_seconds ----------

#[test]
fn parse_interval_seconds_variants() {
    use tensordb::sql::eval::parse_interval_seconds;

    assert_eq!(parse_interval_seconds("1s"), Some(1));
    assert_eq!(parse_interval_seconds("30s"), Some(30));
    assert_eq!(parse_interval_seconds("1m"), Some(60));
    assert_eq!(parse_interval_seconds("5m"), Some(300));
    assert_eq!(parse_interval_seconds("1h"), Some(3600));
    assert_eq!(parse_interval_seconds("6h"), Some(21600));
    assert_eq!(parse_interval_seconds("1d"), Some(86400));
    assert_eq!(parse_interval_seconds("7d"), Some(604800));
    assert_eq!(parse_interval_seconds("1w"), Some(604800));
    assert_eq!(parse_interval_seconds("3600"), Some(3600));
    assert_eq!(parse_interval_seconds("invalid"), None);
}

// ---------- DESCRIBE on timeseries table ----------

#[test]
fn describe_timeseries_table() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TIMESERIES TABLE metrics (id INTEGER PRIMARY KEY, ts INTEGER, value REAL) WITH (bucket_size = '1h');")
        .unwrap();

    let result = db.sql("DESCRIBE metrics;").unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 3);
    assert!(rows[0].contains("\"id\""));
    assert!(rows[1].contains("\"ts\""));
    assert!(rows[2].contains("\"value\""));
}

// ---------- SELECT from timeseries with WHERE ----------

#[test]
fn select_timeseries_with_where() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TIMESERIES TABLE metrics (id INTEGER PRIMARY KEY, ts INTEGER, value REAL) WITH (bucket_size = '1h');")
        .unwrap();
    db.sql("INSERT INTO metrics (id, ts, value) VALUES (1, 100, 10.0);")
        .unwrap();
    db.sql("INSERT INTO metrics (id, ts, value) VALUES (2, 200, 20.0);")
        .unwrap();
    db.sql("INSERT INTO metrics (id, ts, value) VALUES (3, 300, 30.0);")
        .unwrap();

    let result = db
        .sql("SELECT value FROM metrics WHERE value > 15;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 2);
}

// ---------- Multiple value columns ----------

#[test]
fn timeseries_multiple_value_columns() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TIMESERIES TABLE multi (id INTEGER PRIMARY KEY, ts INTEGER, cpu REAL, mem REAL) WITH (bucket_size = '1h');")
        .unwrap();

    let result = db.sql("DESCRIBE multi;").unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 4); // id, ts, cpu, mem

    db.sql("INSERT INTO multi (id, ts, cpu, mem) VALUES (1, 100, 75.5, 4096.0);")
        .unwrap();
    db.sql("INSERT INTO multi (id, ts, cpu, mem) VALUES (2, 200, 80.0, 3800.0);")
        .unwrap();

    let result = db.sql("SELECT AVG(cpu), AVG(mem) FROM multi;").unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    let r = parse_json(&rows[0]);
    assert_eq!(r["avg_cpu"], 77.75);
    assert_eq!(r["avg_mem"], 3948.0);
}

// ---------- Error cases ----------

#[test]
fn create_timeseries_no_pk_fails() {
    let (_dir, db) = setup_db();
    let result = db.sql("CREATE TIMESERIES TABLE bad (ts INTEGER, value REAL);");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("PRIMARY KEY"));
}

#[test]
fn invalid_bucket_size_fails() {
    let (_dir, db) = setup_db();
    let result = db.sql(
        "CREATE TIMESERIES TABLE bad (id INTEGER PRIMARY KEY, ts INTEGER) WITH (bucket_size = 'xyz');",
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("invalid bucket_size"));
}
