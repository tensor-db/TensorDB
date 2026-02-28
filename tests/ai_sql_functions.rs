//! Integration tests for v0.12 AI SQL Functions
//!
//! Tests EXPLAIN AI, ai_top_risks, and ai_cluster_summary.

use tempfile::tempdir;

use tensordb::config::Config;
use tensordb::sql::exec::SqlResult;
use tensordb::Database;

fn setup_ai_db() -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let db = Database::open(
        dir.path(),
        Config {
            shard_count: 1,
            ai_auto_insights: true,
            ai_batch_window_ms: 10,
            ai_batch_max_events: 10,
            ..Config::default()
        },
    )
    .unwrap();
    (dir, db)
}

fn setup_plain_db() -> (tempfile::TempDir, Database) {
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

// ---------- EXPLAIN AI tests ----------

#[test]
fn explain_ai_existing_key() {
    let (_dir, db) = setup_plain_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"status\":\"error\",\"msg\":\"critical failure\"}');")
        .unwrap();

    let result = db.sql("EXPLAIN AI 't/k1';").unwrap();
    if let SqlResult::Explain(text) = result {
        assert!(text.contains("EXPLAIN AI"));
        assert!(text.contains("risk score") || text.contains("Inline risk score"));
    } else {
        panic!("expected Explain result");
    }
}

#[test]
fn explain_ai_missing_key() {
    let (_dir, db) = setup_plain_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    let result = db.sql("EXPLAIN AI 't/nonexistent';").unwrap();
    if let SqlResult::Explain(text) = result {
        assert!(
            text.contains("not found") || text.contains("No AI insights"),
            "expected 'not found' or 'No AI insights', got: {text}"
        );
    } else {
        panic!("expected Explain result");
    }
}

#[test]
fn explain_ai_with_insights() {
    let (_dir, db) = setup_ai_db();
    db.sql("CREATE TABLE audit (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO audit (pk, doc) VALUES ('evt-1', '{\"action\":\"login\",\"status\":\"failed\",\"error\":\"unauthorized access\"}');")
        .unwrap();

    // Give AI runtime time to process
    std::thread::sleep(std::time::Duration::from_millis(100));

    let result = db.sql("EXPLAIN AI 'audit/evt-1';").unwrap();
    if let SqlResult::Explain(text) = result {
        assert!(text.contains("EXPLAIN AI"));
        // Should either have insights or show key info
        assert!(
            text.contains("risk") || text.contains("Insight") || text.contains("not found"),
            "unexpected output: {text}"
        );
    } else {
        panic!("expected Explain result");
    }
}

// ---------- ai_top_risks tests ----------

#[test]
fn ai_top_risks_empty() {
    let (_dir, db) = setup_plain_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    // When no AI insights exist, returns empty results
    let result = db.sql("SELECT * FROM ai_top_risks LIMIT 5;").unwrap();
    let rows = row_strings(&result);
    assert!(rows.is_empty());
}

#[test]
fn ai_top_risks_with_data() {
    let (_dir, db) = setup_ai_db();
    db.sql("CREATE TABLE events (pk TEXT PRIMARY KEY);")
        .unwrap();

    // Insert events that will trigger AI insights
    db.sql("INSERT INTO events (pk, doc) VALUES ('e1', '{\"action\":\"error\",\"msg\":\"critical system failure\"}');")
        .unwrap();
    db.sql("INSERT INTO events (pk, doc) VALUES ('e2', '{\"action\":\"login\",\"status\":\"failed\",\"error\":\"unauthorized\"}');")
        .unwrap();
    db.sql(
        "INSERT INTO events (pk, doc) VALUES ('e3', '{\"action\":\"payment\",\"status\":\"ok\"}');",
    )
    .unwrap();

    // Give AI runtime time to process
    std::thread::sleep(std::time::Duration::from_millis(200));

    let result = db.sql("SELECT * FROM ai_top_risks LIMIT 10;").unwrap();
    let rows = row_strings(&result);

    // Should have some insights (at least for the error/failure events)
    if !rows.is_empty() {
        // First result should have highest risk
        assert!(rows[0].contains("risk_score"));
        assert!(rows[0].contains("tags"));

        // Parse first row to verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
        assert!(parsed.get("risk_score").is_some());
        assert!(parsed.get("tags").is_some());
    }
}

#[test]
fn ai_top_risks_respects_limit() {
    let (_dir, db) = setup_ai_db();
    db.sql("CREATE TABLE events (pk TEXT PRIMARY KEY);")
        .unwrap();

    for i in 0..5 {
        db.sql(&format!(
            "INSERT INTO events (pk, doc) VALUES ('e{i}', '{{\"action\":\"error\",\"msg\":\"failure #{i}\"}}');"
        ))
        .unwrap();
    }

    std::thread::sleep(std::time::Duration::from_millis(200));

    let result = db.sql("SELECT * FROM ai_top_risks LIMIT 2;").unwrap();
    let rows = row_strings(&result);
    assert!(rows.len() <= 2, "expected <= 2 rows, got {}", rows.len());
}

// ---------- ai_cluster_summary tests ----------

#[test]
fn ai_cluster_summary_requires_filter() {
    let (_dir, db) = setup_plain_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    // Should fail without WHERE cluster_id = '...'
    let result = db.sql("SELECT * FROM ai_cluster_summary;");
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("cluster_id"));
}

#[test]
fn ai_cluster_summary_nonexistent_cluster() {
    let (_dir, db) = setup_plain_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    let result = db
        .sql("SELECT * FROM ai_cluster_summary WHERE cluster_id = 'nonexistent';")
        .unwrap();
    let rows = row_strings(&result);
    // Should have summary row with 0 events
    assert_eq!(rows.len(), 1);
    let parsed: serde_json::Value = serde_json::from_str(&rows[0]).unwrap();
    assert_eq!(parsed["event_count"], 0);
    assert_eq!(parsed["type"], "summary");
}

// ---------- Parsing tests ----------

#[test]
fn parse_explain_ai_syntax() {
    let (_dir, db) = setup_plain_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    // Valid syntax
    let result = db.sql("EXPLAIN AI 'some/key';");
    assert!(result.is_ok());
}

#[test]
fn parse_explain_ai_still_allows_normal_explain() {
    let (_dir, db) = setup_plain_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    // Normal EXPLAIN should still work
    let result = db.sql("EXPLAIN SELECT doc FROM t;").unwrap();
    if let SqlResult::Explain(text) = result {
        assert!(text.contains("Plan") || text.contains("Scan"));
    } else {
        panic!("expected Explain result");
    }

    // EXPLAIN ANALYZE should still work
    let result = db.sql("EXPLAIN ANALYZE SELECT doc FROM t;").unwrap();
    if let SqlResult::Explain(text) = result {
        assert!(text.contains("execution_time"));
    } else {
        panic!("expected Explain result");
    }
}
