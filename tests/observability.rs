use tempfile::tempdir;

use tensordb::config::Config;
use tensordb::sql::exec::SqlResult;
use tensordb::sql::parser::{parse_sql, Statement};
use tensordb::Database;

fn setup_db() -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 2,
        slow_query_threshold_us: 0, // log all queries as "slow" for testing
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

// --- Parser tests ---

#[test]
fn parse_show_stats() {
    let stmt = parse_sql("SHOW STATS").unwrap();
    assert!(matches!(stmt, Statement::ShowStats));
}

#[test]
fn parse_show_slow_queries() {
    let stmt = parse_sql("SHOW SLOW QUERIES").unwrap();
    assert!(matches!(stmt, Statement::ShowSlowQueries));
}

#[test]
fn parse_show_active_queries() {
    let stmt = parse_sql("SHOW ACTIVE QUERIES").unwrap();
    assert!(matches!(stmt, Statement::ShowActiveQueries));
}

#[test]
fn parse_show_storage() {
    let stmt = parse_sql("SHOW STORAGE").unwrap();
    assert!(matches!(stmt, Statement::ShowStorage));
}

#[test]
fn parse_show_compaction_status() {
    let stmt = parse_sql("SHOW COMPACTION STATUS").unwrap();
    assert!(matches!(stmt, Statement::ShowCompactionStatus));
}

#[test]
fn parse_show_tables_still_works() {
    let stmt = parse_sql("SHOW TABLES").unwrap();
    assert!(matches!(stmt, Statement::ShowTables));
}

// --- Executor tests ---

#[test]
fn show_stats_returns_rows() {
    let (_dir, db) = setup_db();
    let result = db.sql("SHOW STATS").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strings = rows_to_strings(rows);
            // Should have basic metrics
            assert!(strings.iter().any(|s| s.contains("uptime_ms")));
            assert!(strings.iter().any(|s| s.contains("shard_count")));
            assert!(strings.iter().any(|s| s.contains("total_puts")));
            assert!(strings.iter().any(|s| s.contains("total_gets")));
            assert!(strings.iter().any(|s| s.contains("cache_hit_rate")));
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn show_stats_after_operations() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();
    db.sql("INSERT INTO t1 (id, name) VALUES (1, 'alice');")
        .unwrap();
    db.sql("SELECT * FROM t1;").unwrap();

    let result = db.sql("SHOW STATS").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strings = rows_to_strings(rows);
            // queries_total should reflect the queries we've run
            let queries_row = strings
                .iter()
                .find(|s| s.contains("\"queries_total\""))
                .expect("should have queries_total metric");
            // We ran CREATE, INSERT, SELECT before SHOW STATS
            let v: serde_json::Value = serde_json::from_str(queries_row).unwrap();
            let val: u64 = v["value"].as_str().unwrap().parse().unwrap();
            assert!(val >= 3, "expected at least 3 queries, got {val}");
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn show_slow_queries_captures_queries() {
    let (_dir, db) = setup_db();
    // threshold is 0, so all queries logged
    db.sql("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();
    db.sql("INSERT INTO t1 (id, name) VALUES (1, 'test');")
        .unwrap();

    let result = db.sql("SHOW SLOW QUERIES").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strings = rows_to_strings(rows);
            // Should have logged the CREATE and INSERT
            assert!(
                strings.len() >= 2,
                "expected at least 2 slow queries, got {}",
                strings.len()
            );
            // Entries should have query and duration_us fields
            for s in &strings {
                let v: serde_json::Value = serde_json::from_str(s).unwrap();
                assert!(v.get("query").is_some());
                assert!(v.get("duration_us").is_some());
                assert!(v.get("timestamp_ms").is_some());
            }
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn show_active_queries_includes_self() {
    let (_dir, db) = setup_db();
    let result = db.sql("SHOW ACTIVE QUERIES").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            // The SHOW ACTIVE QUERIES should show itself as running
            // (it registered itself before executing)
            // Note: by the time we reach the executor, the query is registered
            // but by the time results return, it's deregistered.
            // The executor snapshot happens while the query is still active,
            // so it should see at least the SHOW ACTIVE QUERIES itself.
            // Actually, the active query tracking wraps execute_sql, so
            // inside execute_sql the query IS registered.
            let strings = rows_to_strings(rows);
            assert!(
                !strings.is_empty(),
                "expected at least 1 active query (self)"
            );
            let has_self = strings.iter().any(|s| s.contains("SHOW ACTIVE QUERIES"));
            assert!(has_self, "should see SHOW ACTIVE QUERIES itself");
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn show_storage_per_shard() {
    let (_dir, db) = setup_db();
    // Insert some data to have non-zero memtable bytes
    db.sql("CREATE TABLE t1 (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();
    for i in 0..10 {
        db.sql(&format!(
            "INSERT INTO t1 (id, name) VALUES ({i}, 'row_{i}');"
        ))
        .unwrap();
    }

    let result = db.sql("SHOW STORAGE").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strings = rows_to_strings(rows);
            // Should have per-shard rows + TOTAL summary
            // 2 shards + 1 total = 3 rows
            assert!(
                strings.len() >= 3,
                "expected at least 3 rows (2 shards + total), got {}",
                strings.len()
            );

            // Check shard rows have expected fields
            let shard_row: serde_json::Value = serde_json::from_str(&strings[0]).unwrap();
            assert!(shard_row.get("memtable_bytes").is_some());
            assert!(shard_row.get("sstable_bytes").is_some());
            assert!(shard_row.get("wal_bytes").is_some());
            assert!(shard_row.get("l0_files").is_some());
            assert!(shard_row.get("level_sizes").is_some());

            // Last row should be TOTAL
            let total_row: serde_json::Value =
                serde_json::from_str(strings.last().unwrap()).unwrap();
            assert_eq!(total_row["shard"], "TOTAL");
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn show_compaction_status() {
    let (_dir, db) = setup_db();
    let result = db.sql("SHOW COMPACTION STATUS").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strings = rows_to_strings(rows);
            // 2 shards + GLOBAL summary = 3 rows
            assert!(
                strings.len() >= 3,
                "expected at least 3 rows, got {}",
                strings.len()
            );

            // Check shard rows
            let shard_row: serde_json::Value = serde_json::from_str(&strings[0]).unwrap();
            assert!(shard_row.get("l0_file_count").is_some());
            assert!(shard_row.get("total_sstable_files").is_some());
            assert!(shard_row.get("level_sizes").is_some());
            assert!(shard_row.get("needs_compaction").is_some());

            // Global summary
            let global_row: serde_json::Value =
                serde_json::from_str(strings.last().unwrap()).unwrap();
            assert_eq!(global_row["shard"], "GLOBAL");
            assert!(global_row.get("total_flushes").is_some());
            assert!(global_row.get("total_compactions").is_some());
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn cache_hit_rate_in_stats() {
    let (_dir, db) = setup_db();
    // Read same key twice to generate cache hit
    db.put(b"key1", b"val1".to_vec(), 0, u64::MAX, None)
        .unwrap();
    let _ = db.get(b"key1", None, None).unwrap();
    let _ = db.get(b"key1", None, None).unwrap();

    let result = db.sql("SHOW STATS").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strings = rows_to_strings(rows);
            let hit_rate_row = strings
                .iter()
                .find(|s| s.contains("\"cache_hit_rate\""))
                .expect("should have cache_hit_rate");
            let v: serde_json::Value = serde_json::from_str(hit_rate_row).unwrap();
            let rate: f64 = v["value"].as_str().unwrap().parse().unwrap();
            // With small test data, rate might be 0 (all in memtable), which is fine
            assert!((0.0..=1.0).contains(&rate), "hit rate should be [0, 1]");
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}

#[test]
fn show_stats_query_latency_histogram() {
    let (_dir, db) = setup_db();
    // Run some queries to populate histogram
    db.sql("CREATE TABLE t1 (id INTEGER PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t1 (id) VALUES (1);").unwrap();
    db.sql("SELECT * FROM t1;").unwrap();

    let result = db.sql("SHOW STATS").unwrap();
    match result {
        SqlResult::Rows(rows) => {
            let strings = rows_to_strings(rows);
            assert!(strings.iter().any(|s| s.contains("query_latency_us_count")));
            assert!(strings
                .iter()
                .any(|s| s.contains("query_latency_us_avg_us")));
            assert!(strings
                .iter()
                .any(|s| s.contains("query_latency_us_p50_us")));
            assert!(strings
                .iter()
                .any(|s| s.contains("query_latency_us_p99_us")));
        }
        other => panic!("expected Rows, got {:?}", other),
    }
}
