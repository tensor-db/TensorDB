use std::collections::HashMap;

use crate::engine::db::Database;
use crate::error::{Result, TensorError};

/// Key prefix for event store metadata.
const EVENT_STORE_META_PREFIX: &str = "__es/meta/";
/// Key prefix for event data: __es/data/<store_name>/<aggregate_id>/<sequence_num>
const EVENT_STORE_DATA_PREFIX: &str = "__es/data/";
/// Key prefix for idempotency tracking.
const IDEMPOTENCY_PREFIX: &str = "__es/idem/";
/// Key prefix for aggregate snapshots.
const SNAPSHOT_PREFIX: &str = "__es/snap/";

/// Event store metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EventStoreMetadata {
    pub name: String,
    pub event_types: Vec<String>,
    pub created_at: u64,
    pub snapshot_interval: Option<u64>, // Snapshot every N events
}

/// A stored event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StoredEvent {
    pub aggregate_id: String,
    pub sequence_num: u64,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub timestamp: u64,
    pub idempotency_key: Option<String>,
}

/// Aggregate snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AggregateSnapshot {
    pub aggregate_id: String,
    pub sequence_num: u64,
    pub state: serde_json::Value,
    pub timestamp: u64,
}

/// Create an event store.
pub fn create_event_store(db: &Database, name: &str, snapshot_interval: Option<u64>) -> Result<()> {
    let key = format!("{}{}", EVENT_STORE_META_PREFIX, name);
    let meta = EventStoreMetadata {
        name: name.to_string(),
        event_types: Vec::new(),
        created_at: current_timestamp_ms(),
        snapshot_interval,
    };
    let value = serde_json::to_vec(&meta)
        .map_err(|e| TensorError::SqlExec(format!("failed to serialize event store: {e}")))?;
    db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
    Ok(())
}

/// Get event store metadata.
pub fn get_event_store(db: &Database, name: &str) -> Result<Option<EventStoreMetadata>> {
    let key = format!("{}{}", EVENT_STORE_META_PREFIX, name);
    match db.get(key.as_bytes(), None, None)? {
        Some(bytes) => {
            let meta: EventStoreMetadata = serde_json::from_slice(&bytes)
                .map_err(|e| TensorError::SqlExec(format!("failed to parse event store: {e}")))?;
            Ok(Some(meta))
        }
        None => Ok(None),
    }
}

/// List all event stores.
pub fn list_event_stores(db: &Database) -> Result<Vec<EventStoreMetadata>> {
    let rows = db.scan_prefix(EVENT_STORE_META_PREFIX.as_bytes(), None, None, None)?;
    let mut stores = Vec::new();
    for row in rows {
        if let Ok(meta) = serde_json::from_slice::<EventStoreMetadata>(&row.doc) {
            stores.push(meta);
        }
    }
    Ok(stores)
}

/// Append an event to an event store.
/// Returns the sequence number assigned to the event.
pub fn append_event(
    db: &Database,
    store_name: &str,
    aggregate_id: &str,
    event_type: &str,
    payload: serde_json::Value,
    idempotency_key: Option<&str>,
) -> Result<u64> {
    // Check idempotency key
    if let Some(idem_key) = idempotency_key {
        let idem_storage_key = format!("{}{}/{}", IDEMPOTENCY_PREFIX, store_name, idem_key);
        if let Some(_existing) = db.get(idem_storage_key.as_bytes(), None, None)? {
            return Err(TensorError::SqlExec(format!(
                "duplicate idempotency key: {idem_key}"
            )));
        }
    }

    // Get next sequence number for this aggregate
    let seq_num = next_sequence_number(db, store_name, aggregate_id)?;

    // Store the event
    let event = StoredEvent {
        aggregate_id: aggregate_id.to_string(),
        sequence_num: seq_num,
        event_type: event_type.to_string(),
        payload,
        timestamp: current_timestamp_ms(),
        idempotency_key: idempotency_key.map(|s| s.to_string()),
    };

    let event_key = format!(
        "{}{}/{}/{:020}",
        EVENT_STORE_DATA_PREFIX, store_name, aggregate_id, seq_num
    );
    let event_value = serde_json::to_vec(&event)
        .map_err(|e| TensorError::SqlExec(format!("failed to serialize event: {e}")))?;
    db.put(event_key.as_bytes(), event_value, 0, u64::MAX, None)?;

    // Store idempotency marker
    if let Some(idem_key) = idempotency_key {
        let idem_storage_key = format!("{}{}/{}", IDEMPOTENCY_PREFIX, store_name, idem_key);
        let idem_val = serde_json::json!({"seq": seq_num, "aggregate_id": aggregate_id});
        db.put(
            idem_storage_key.as_bytes(),
            serde_json::to_vec(&idem_val).unwrap_or_default(),
            0,
            u64::MAX,
            None,
        )?;
    }

    Ok(seq_num)
}

/// Get all events for an aggregate, optionally from a given sequence number.
pub fn get_events(
    db: &Database,
    store_name: &str,
    aggregate_id: &str,
    from_seq: Option<u64>,
) -> Result<Vec<StoredEvent>> {
    let prefix = format!(
        "{}{}/{}/",
        EVENT_STORE_DATA_PREFIX, store_name, aggregate_id
    );
    let rows = db.scan_prefix(prefix.as_bytes(), None, None, None)?;

    let mut events = Vec::new();
    for row in rows {
        if let Ok(event) = serde_json::from_slice::<StoredEvent>(&row.doc) {
            if let Some(from) = from_seq {
                if event.sequence_num < from {
                    continue;
                }
            }
            events.push(event);
        }
    }

    events.sort_by_key(|e| e.sequence_num);
    Ok(events)
}

/// Get the current state of an aggregate by replaying all events.
/// Uses `fold_fn` to reduce events into a state.
pub fn get_aggregate_state(
    db: &Database,
    store_name: &str,
    aggregate_id: &str,
) -> Result<serde_json::Value> {
    // Check for snapshot first
    let snap_key = format!("{}{}/{}/latest", SNAPSHOT_PREFIX, store_name, aggregate_id);
    let (initial_state, from_seq) = match db.get(snap_key.as_bytes(), None, None)? {
        Some(bytes) => {
            if let Ok(snap) = serde_json::from_slice::<AggregateSnapshot>(&bytes) {
                (snap.state, Some(snap.sequence_num + 1))
            } else {
                (serde_json::Value::Object(serde_json::Map::new()), None)
            }
        }
        None => (serde_json::Value::Object(serde_json::Map::new()), None),
    };

    let events = get_events(db, store_name, aggregate_id, from_seq)?;

    // Merge events into state: each event's payload fields override state fields
    let mut state = initial_state;
    for event in &events {
        if let (Some(state_obj), Some(payload_obj)) =
            (state.as_object_mut(), event.payload.as_object())
        {
            for (k, v) in payload_obj {
                state_obj.insert(k.clone(), v.clone());
            }
            state_obj.insert(
                "__last_event_type".to_string(),
                serde_json::json!(event.event_type),
            );
            state_obj.insert(
                "__sequence_num".to_string(),
                serde_json::json!(event.sequence_num),
            );
            state_obj.insert(
                "__aggregate_id".to_string(),
                serde_json::json!(event.aggregate_id),
            );
        }
    }

    Ok(state)
}

/// Save a snapshot of aggregate state.
pub fn save_snapshot(
    db: &Database,
    store_name: &str,
    aggregate_id: &str,
    sequence_num: u64,
    state: serde_json::Value,
) -> Result<()> {
    let snap = AggregateSnapshot {
        aggregate_id: aggregate_id.to_string(),
        sequence_num,
        state,
        timestamp: current_timestamp_ms(),
    };
    let snap_key = format!("{}{}/{}/latest", SNAPSHOT_PREFIX, store_name, aggregate_id);
    let snap_value = serde_json::to_vec(&snap)
        .map_err(|e| TensorError::SqlExec(format!("failed to serialize snapshot: {e}")))?;
    db.put(snap_key.as_bytes(), snap_value, 0, u64::MAX, None)?;
    Ok(())
}

/// Get the next sequence number for an aggregate.
fn next_sequence_number(db: &Database, store_name: &str, aggregate_id: &str) -> Result<u64> {
    let prefix = format!(
        "{}{}/{}/",
        EVENT_STORE_DATA_PREFIX, store_name, aggregate_id
    );
    let rows = db.scan_prefix(prefix.as_bytes(), None, None, None)?;

    let mut max_seq = 0u64;
    for row in rows {
        if let Ok(event) = serde_json::from_slice::<StoredEvent>(&row.doc) {
            max_seq = max_seq.max(event.sequence_num);
        }
    }

    Ok(max_seq + 1)
}

/// Get aggregate IDs that have events matching a given event type.
pub fn find_aggregates_by_event_type(
    db: &Database,
    store_name: &str,
    event_type: &str,
) -> Result<Vec<String>> {
    let prefix = format!("{}{}/", EVENT_STORE_DATA_PREFIX, store_name);
    let rows = db.scan_prefix(prefix.as_bytes(), None, None, None)?;

    let mut aggregate_ids: HashMap<String, bool> = HashMap::new();
    for row in rows {
        if let Ok(event) = serde_json::from_slice::<StoredEvent>(&row.doc) {
            if event.event_type == event_type {
                aggregate_ids.insert(event.aggregate_id, true);
            }
        }
    }

    let mut ids: Vec<String> = aggregate_ids.into_keys().collect();
    ids.sort();
    Ok(ids)
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn setup() -> (Database, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), Config::default()).unwrap();
        (db, dir)
    }

    #[test]
    fn test_create_and_get_event_store() {
        let (db, _dir) = setup();
        create_event_store(&db, "orders", Some(100)).unwrap();

        let meta = get_event_store(&db, "orders").unwrap().unwrap();
        assert_eq!(meta.name, "orders");
        assert_eq!(meta.snapshot_interval, Some(100));
    }

    #[test]
    fn test_append_and_get_events() {
        let (db, _dir) = setup();
        create_event_store(&db, "orders", None).unwrap();

        let seq1 = append_event(
            &db,
            "orders",
            "order-123",
            "OrderCreated",
            serde_json::json!({"item": "Widget", "qty": 5}),
            None,
        )
        .unwrap();
        assert_eq!(seq1, 1);

        let seq2 = append_event(
            &db,
            "orders",
            "order-123",
            "OrderShipped",
            serde_json::json!({"tracking": "TRK-456"}),
            None,
        )
        .unwrap();
        assert_eq!(seq2, 2);

        let events = get_events(&db, "orders", "order-123", None).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "OrderCreated");
        assert_eq!(events[1].event_type, "OrderShipped");
    }

    #[test]
    fn test_idempotency_key_prevents_duplicate() {
        let (db, _dir) = setup();
        create_event_store(&db, "payments", None).unwrap();

        append_event(
            &db,
            "payments",
            "pay-1",
            "PaymentReceived",
            serde_json::json!({"amount": 100}),
            Some("idm-abc-123"),
        )
        .unwrap();

        // Duplicate should fail
        let result = append_event(
            &db,
            "payments",
            "pay-1",
            "PaymentReceived",
            serde_json::json!({"amount": 100}),
            Some("idm-abc-123"),
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("duplicate idempotency key"));
    }

    #[test]
    fn test_aggregate_state() {
        let (db, _dir) = setup();
        create_event_store(&db, "users", None).unwrap();

        append_event(
            &db,
            "users",
            "user-1",
            "UserCreated",
            serde_json::json!({"name": "Alice", "email": "alice@example.com"}),
            None,
        )
        .unwrap();

        append_event(
            &db,
            "users",
            "user-1",
            "EmailChanged",
            serde_json::json!({"email": "new_alice@example.com"}),
            None,
        )
        .unwrap();

        let state = get_aggregate_state(&db, "users", "user-1").unwrap();
        assert_eq!(state["name"], "Alice"); // From first event
        assert_eq!(state["email"], "new_alice@example.com"); // Updated by second event
        assert_eq!(state["__sequence_num"], 2);
    }

    #[test]
    fn test_snapshot_and_replay() {
        let (db, _dir) = setup();
        create_event_store(&db, "counters", None).unwrap();

        // Append 5 events
        for i in 1..=5 {
            append_event(
                &db,
                "counters",
                "cnt-1",
                "Incremented",
                serde_json::json!({"count": i}),
                None,
            )
            .unwrap();
        }

        // Save snapshot at seq 3
        save_snapshot(
            &db,
            "counters",
            "cnt-1",
            3,
            serde_json::json!({"count": 3, "snapshotted": true}),
        )
        .unwrap();

        // Get state — should load snapshot + replay events 4 and 5
        let state = get_aggregate_state(&db, "counters", "cnt-1").unwrap();
        assert_eq!(state["count"], 5); // Latest event overrides
        assert_eq!(state["snapshotted"], true); // From snapshot
    }

    #[test]
    fn test_find_aggregates_by_event_type() {
        let (db, _dir) = setup();
        create_event_store(&db, "orders", None).unwrap();

        append_event(
            &db,
            "orders",
            "order-1",
            "OrderCreated",
            serde_json::json!({}),
            None,
        )
        .unwrap();
        append_event(
            &db,
            "orders",
            "order-2",
            "OrderCreated",
            serde_json::json!({}),
            None,
        )
        .unwrap();
        append_event(
            &db,
            "orders",
            "order-1",
            "OrderShipped",
            serde_json::json!({}),
            None,
        )
        .unwrap();

        let created = find_aggregates_by_event_type(&db, "orders", "OrderCreated").unwrap();
        assert_eq!(created, vec!["order-1", "order-2"]);

        let shipped = find_aggregates_by_event_type(&db, "orders", "OrderShipped").unwrap();
        assert_eq!(shipped, vec!["order-1"]);
    }

    #[test]
    fn test_get_events_from_seq() {
        let (db, _dir) = setup();
        create_event_store(&db, "log", None).unwrap();

        for i in 1..=5 {
            append_event(
                &db,
                "log",
                "agg-1",
                "Event",
                serde_json::json!({"n": i}),
                None,
            )
            .unwrap();
        }

        // Get events from seq 3 onwards
        let events = get_events(&db, "log", "agg-1", Some(3)).unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].sequence_num, 3);
        assert_eq!(events[2].sequence_num, 5);
    }

    #[test]
    fn test_list_event_stores() {
        let (db, _dir) = setup();
        create_event_store(&db, "store_a", None).unwrap();
        create_event_store(&db, "store_b", Some(50)).unwrap();

        let stores = list_event_stores(&db).unwrap();
        assert!(stores.len() >= 2);
    }
}
