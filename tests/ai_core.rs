use std::time::Duration;

use tempfile::TempDir;
use tensordb::{Config, Database};

fn open_ai_db(dir: &TempDir) -> Database {
    let cfg = Config {
        shard_count: 1,
        ai_auto_insights: true,
        ..Config::default()
    };
    Database::open(dir.path(), cfg).unwrap()
}

#[test]
fn ai_runtime_generates_and_persists_insights() {
    let dir = TempDir::new().unwrap();
    let db = open_ai_db(&dir);

    db.put(
        b"txn/1001",
        b"{\"event\":\"refund requested\",\"note\":\"chargeback dispute observed\"}".to_vec(),
        0,
        u64::MAX,
        Some(1),
    )
    .unwrap();

    let mut insights = Vec::new();
    for _ in 0..60 {
        insights = db.ai_insights_for_key(b"txn/1001", Some(5)).unwrap();
        if !insights.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    assert!(
        !insights.is_empty(),
        "expected ai insight to be generated for write"
    );
    assert_eq!(insights[0].source_key_hex, "74786e2f31303031");
    assert!(
        insights[0].tags.iter().any(|t| t == "payments-risk"),
        "expected payments-risk tag, got {:?}",
        insights[0].tags
    );
    assert!(
        insights[0].model_id.contains("core-ai"),
        "unexpected model id {}",
        insights[0].model_id
    );
    assert!(
        !insights[0].insight_id.is_empty(),
        "insight id should be populated"
    );
    assert!(
        !insights[0].cluster_id.is_empty(),
        "cluster id should be populated"
    );
    assert!(
        !insights[0].provenance.hint_basis.is_empty(),
        "provenance hint basis should be populated"
    );

    let fetched = db
        .ai_insight_by_id(&insights[0].insight_id)
        .unwrap()
        .expect("insight should be retrievable by id");
    assert_eq!(fetched.source_commit_ts, insights[0].source_commit_ts);

    let correlated = db.ai_correlation_for_key(b"txn/1001", Some(10)).unwrap();
    assert!(!correlated.is_empty(), "expected correlation rows");
    assert_eq!(correlated[0].insight_id, insights[0].insight_id);

    let stats = db.ai_stats();
    assert!(stats.enabled);
    assert!(stats.events_received >= 1);
    assert!(stats.insights_written >= 1);
}

#[test]
fn ai_internal_writes_do_not_leak_into_change_feed() {
    let dir = TempDir::new().unwrap();
    let db = open_ai_db(&dir);
    let rx = db.subscribe(b"");

    db.put(
        b"feed/live",
        b"{\"message\":\"timeout while processing payment\"}".to_vec(),
        0,
        u64::MAX,
        Some(1),
    )
    .unwrap();

    let mut events = Vec::new();
    while let Ok(evt) = rx.recv_timeout(Duration::from_millis(100)) {
        events.push(evt);
    }

    assert_eq!(events.len(), 1, "internal ai writes should be hidden");
    assert_eq!(events[0].user_key, b"feed/live");
}

#[test]
fn ai_core_runtime_generates_insights_without_external_process() {
    let dir = TempDir::new().unwrap();
    let cfg = Config {
        shard_count: 1,
        ai_auto_insights: true,
        ..Config::default()
    };
    let db = Database::open(dir.path(), cfg).unwrap();

    db.put(
        b"incident/native",
        b"{\"msg\":\"timeout in checkout\"}".to_vec(),
        0,
        u64::MAX,
        Some(1),
    )
    .unwrap();

    let mut insights = Vec::new();
    for _ in 0..60 {
        insights = db.ai_insights_for_key(b"incident/native", Some(5)).unwrap();
        if !insights.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    assert!(!insights.is_empty(), "expected native in-core insight");
    assert!(
        insights[0].model_id.contains("core-ai"),
        "expected native model id, got {}",
        insights[0].model_id
    );
    assert!(
        !insights[0].cluster_id.is_empty(),
        "expected native insight to include cluster id"
    );

    let stats = db.ai_stats();
    assert!(stats.enabled);
}
