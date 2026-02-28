// Integration tests for v0.22 Event Sourcing
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::facet::event_sourcing::*;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_event_store_lifecycle() {
    let (db, _dir) = setup();

    // Create store
    create_event_store(&db, "orders", Some(100)).unwrap();

    // Verify it exists
    let meta = get_event_store(&db, "orders").unwrap().unwrap();
    assert_eq!(meta.name, "orders");
    assert_eq!(meta.snapshot_interval, Some(100));

    // List stores
    let stores = list_event_stores(&db).unwrap();
    assert!(!stores.is_empty());
}

#[test]
fn test_event_append_and_replay() {
    let (db, _dir) = setup();
    create_event_store(&db, "accounts", None).unwrap();

    // Append events for an account
    let s1 = append_event(
        &db,
        "accounts",
        "acc-001",
        "AccountOpened",
        serde_json::json!({"name": "Alice", "balance": 0}),
        None,
    )
    .unwrap();
    assert_eq!(s1, 1);

    let s2 = append_event(
        &db,
        "accounts",
        "acc-001",
        "Deposited",
        serde_json::json!({"balance": 500, "amount": 500}),
        None,
    )
    .unwrap();
    assert_eq!(s2, 2);

    let s3 = append_event(
        &db,
        "accounts",
        "acc-001",
        "Withdrawn",
        serde_json::json!({"balance": 300, "amount": 200}),
        None,
    )
    .unwrap();
    assert_eq!(s3, 3);

    // Get all events
    let events = get_events(&db, "accounts", "acc-001", None).unwrap();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].event_type, "AccountOpened");
    assert_eq!(events[1].event_type, "Deposited");
    assert_eq!(events[2].event_type, "Withdrawn");

    // Get events from seq 2 onwards
    let later = get_events(&db, "accounts", "acc-001", Some(2)).unwrap();
    assert_eq!(later.len(), 2);

    // Get aggregate state
    let state = get_aggregate_state(&db, "accounts", "acc-001").unwrap();
    assert_eq!(state["balance"], 300);
    assert_eq!(state["name"], "Alice");
    assert_eq!(state["__sequence_num"], 3);
}

#[test]
fn test_idempotency_prevents_duplicates() {
    let (db, _dir) = setup();
    create_event_store(&db, "payments", None).unwrap();

    append_event(
        &db,
        "payments",
        "pay-1",
        "Charged",
        serde_json::json!({"amount": 50}),
        Some("idem-key-1"),
    )
    .unwrap();

    // Same idempotency key should fail
    let result = append_event(
        &db,
        "payments",
        "pay-1",
        "Charged",
        serde_json::json!({"amount": 50}),
        Some("idem-key-1"),
    );
    assert!(result.is_err());

    // Different key should succeed
    append_event(
        &db,
        "payments",
        "pay-1",
        "Refunded",
        serde_json::json!({"amount": -50}),
        Some("idem-key-2"),
    )
    .unwrap();
}

#[test]
fn test_snapshot_optimized_replay() {
    let (db, _dir) = setup();
    create_event_store(&db, "counters", None).unwrap();

    // Create 10 events
    for i in 1..=10 {
        append_event(
            &db,
            "counters",
            "cnt-1",
            "Increment",
            serde_json::json!({"value": i}),
            None,
        )
        .unwrap();
    }

    // Save snapshot at seq 7
    save_snapshot(
        &db,
        "counters",
        "cnt-1",
        7,
        serde_json::json!({"value": 7, "snapshot": true}),
    )
    .unwrap();

    // State should be built from snapshot + events 8-10
    let state = get_aggregate_state(&db, "counters", "cnt-1").unwrap();
    assert_eq!(state["value"], 10); // Latest event value
    assert_eq!(state["snapshot"], true); // From snapshot
    assert_eq!(state["__sequence_num"], 10);
}

#[test]
fn test_find_aggregates_by_event_type() {
    let (db, _dir) = setup();
    create_event_store(&db, "orders", None).unwrap();

    append_event(
        &db,
        "orders",
        "order-A",
        "Created",
        serde_json::json!({}),
        None,
    )
    .unwrap();
    append_event(
        &db,
        "orders",
        "order-B",
        "Created",
        serde_json::json!({}),
        None,
    )
    .unwrap();
    append_event(
        &db,
        "orders",
        "order-A",
        "Shipped",
        serde_json::json!({}),
        None,
    )
    .unwrap();

    let created = find_aggregates_by_event_type(&db, "orders", "Created").unwrap();
    assert_eq!(created.len(), 2);

    let shipped = find_aggregates_by_event_type(&db, "orders", "Shipped").unwrap();
    assert_eq!(shipped.len(), 1);
    assert_eq!(shipped[0], "order-A");
}
