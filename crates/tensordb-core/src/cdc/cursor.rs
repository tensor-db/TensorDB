use std::collections::HashMap;

use crate::engine::db::Database;
use crate::engine::shard::ChangeEvent;
use crate::error::{Result, TensorError};

/// Key prefix for durable cursor storage.
const CURSOR_PREFIX: &str = "__cdc/cursor/";

/// A persistent cursor position for a change feed consumer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CursorPosition {
    /// Consumer ID.
    pub consumer_id: String,
    /// Last acknowledged commit timestamp per shard.
    pub shard_positions: HashMap<usize, u64>,
    /// Total events consumed so far.
    pub events_consumed: u64,
    /// Timestamp of last ACK.
    pub last_ack_ts: u64,
}

impl CursorPosition {
    pub fn new(consumer_id: &str) -> Self {
        CursorPosition {
            consumer_id: consumer_id.to_string(),
            shard_positions: HashMap::new(),
            events_consumed: 0,
            last_ack_ts: 0,
        }
    }

    fn storage_key(&self) -> String {
        format!("{}{}", CURSOR_PREFIX, self.consumer_id)
    }
}

/// A durable cursor that persists its position in TensorDB itself.
/// Allows consumers to resume from their last acknowledged position
/// after restart.
pub struct DurableCursor {
    position: CursorPosition,
    prefix_filter: Vec<u8>,
    receiver: Option<crossbeam_channel::Receiver<ChangeEvent>>,
    pending_events: Vec<ChangeEvent>,
}

impl DurableCursor {
    /// Create or resume a durable cursor.
    /// If a cursor with this consumer_id already exists, resume from its last position.
    pub fn open(db: &Database, consumer_id: &str, prefix: &[u8]) -> Result<Self> {
        let position = load_cursor_position(db, consumer_id)?
            .unwrap_or_else(|| CursorPosition::new(consumer_id));

        let receiver = Some(db.subscribe(prefix));

        Ok(DurableCursor {
            position,
            prefix_filter: prefix.to_vec(),
            receiver,
            pending_events: Vec::new(),
        })
    }

    /// Poll for new change events. Returns events since last ACK.
    /// Events are buffered until explicitly acknowledged via `ack()`.
    pub fn poll(&mut self, max_events: usize) -> Vec<ChangeEvent> {
        let rx = match &self.receiver {
            Some(rx) => rx,
            None => return Vec::new(),
        };

        let mut events = Vec::new();
        let deadline = std::time::Duration::from_millis(10);

        while events.len() < max_events {
            match rx.recv_timeout(deadline) {
                Ok(event) => {
                    // Filter events that are before our last acknowledged position
                    let shard_id = 0; // TODO: when multi-shard routing is available
                    let last_pos = self
                        .position
                        .shard_positions
                        .get(&shard_id)
                        .copied()
                        .unwrap_or(0);

                    if event.commit_ts > last_pos {
                        events.push(event);
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }

        self.pending_events.extend(events.clone());
        events
    }

    /// Acknowledge all pending events up to and including the given commit_ts.
    /// This persists the cursor position durably.
    pub fn ack(&mut self, db: &Database, up_to_commit_ts: u64) -> Result<u64> {
        let acked_count = self
            .pending_events
            .iter()
            .filter(|e| e.commit_ts <= up_to_commit_ts)
            .count();

        // Remove acknowledged events
        self.pending_events
            .retain(|e| e.commit_ts > up_to_commit_ts);

        // Update position
        self.position.shard_positions.insert(0, up_to_commit_ts);
        self.position.events_consumed += acked_count as u64;
        self.position.last_ack_ts = current_timestamp_ms();

        // Persist position
        save_cursor_position(db, &self.position)?;

        Ok(acked_count as u64)
    }

    /// Get the current cursor position.
    pub fn position(&self) -> &CursorPosition {
        &self.position
    }

    /// Get the prefix filter.
    pub fn prefix(&self) -> &[u8] {
        &self.prefix_filter
    }

    /// Get number of pending (unacknowledged) events.
    pub fn pending_count(&self) -> usize {
        self.pending_events.len()
    }
}

/// Load a cursor position from the database.
fn load_cursor_position(db: &Database, consumer_id: &str) -> Result<Option<CursorPosition>> {
    let key = format!("{}{}", CURSOR_PREFIX, consumer_id);
    match db.get(key.as_bytes(), None, None) {
        Ok(Some(bytes)) => {
            let pos: CursorPosition = serde_json::from_slice(&bytes)
                .map_err(|e| TensorError::SqlExec(format!("failed to parse cursor: {e}")))?;
            Ok(Some(pos))
        }
        Ok(None) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Save a cursor position to the database.
fn save_cursor_position(db: &Database, pos: &CursorPosition) -> Result<()> {
    let key = pos.storage_key();
    let value = serde_json::to_vec(pos)
        .map_err(|e| TensorError::SqlExec(format!("failed to serialize cursor: {e}")))?;
    db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
    Ok(())
}

/// List all durable cursors.
pub fn list_cursors(db: &Database) -> Result<Vec<CursorPosition>> {
    let prefix = CURSOR_PREFIX.as_bytes();
    let rows = db.scan_prefix(prefix, None, None, None)?;
    let mut cursors = Vec::new();
    for row in rows {
        if let Ok(pos) = serde_json::from_slice::<CursorPosition>(&row.doc) {
            cursors.push(pos);
        }
    }
    Ok(cursors)
}

/// Delete a durable cursor.
pub fn delete_cursor(db: &Database, consumer_id: &str) -> Result<()> {
    let key = format!("{}{}", CURSOR_PREFIX, consumer_id);
    db.put(
        key.as_bytes(),
        b"{\"deleted\":true}".to_vec(),
        0,
        u64::MAX,
        None,
    )?;
    Ok(())
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

    #[test]
    fn test_cursor_position_new() {
        let pos = CursorPosition::new("my_consumer");
        assert_eq!(pos.consumer_id, "my_consumer");
        assert!(pos.shard_positions.is_empty());
        assert_eq!(pos.events_consumed, 0);
    }

    #[test]
    fn test_cursor_position_storage_key() {
        let pos = CursorPosition::new("consumer_1");
        assert_eq!(pos.storage_key(), "__cdc/cursor/consumer_1");
    }

    #[test]
    fn test_cursor_roundtrip_serialization() {
        let mut pos = CursorPosition::new("test_consumer");
        pos.shard_positions.insert(0, 100);
        pos.shard_positions.insert(1, 200);
        pos.events_consumed = 42;

        let json = serde_json::to_vec(&pos).unwrap();
        let restored: CursorPosition = serde_json::from_slice(&json).unwrap();

        assert_eq!(restored.consumer_id, "test_consumer");
        assert_eq!(restored.shard_positions.get(&0), Some(&100));
        assert_eq!(restored.shard_positions.get(&1), Some(&200));
        assert_eq!(restored.events_consumed, 42);
    }

    #[test]
    fn test_durable_cursor_open_new() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), Config::default()).unwrap();
        let cursor = DurableCursor::open(&db, "new_consumer", b"table/orders/").unwrap();
        assert_eq!(cursor.position().consumer_id, "new_consumer");
        assert_eq!(cursor.position().events_consumed, 0);
        assert_eq!(cursor.prefix(), b"table/orders/");
    }

    #[test]
    fn test_durable_cursor_persist_and_resume() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), Config::default()).unwrap();

        // Create cursor and simulate advancing it
        let mut pos = CursorPosition::new("persistent_consumer");
        pos.shard_positions.insert(0, 500);
        pos.events_consumed = 10;
        save_cursor_position(&db, &pos).unwrap();

        // Reopen cursor — should resume from saved position
        let cursor = DurableCursor::open(&db, "persistent_consumer", b"").unwrap();
        assert_eq!(cursor.position().events_consumed, 10);
        assert_eq!(cursor.position().shard_positions.get(&0), Some(&500));
    }

    #[test]
    fn test_list_cursors() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), Config::default()).unwrap();

        save_cursor_position(&db, &CursorPosition::new("consumer_a")).unwrap();
        save_cursor_position(&db, &CursorPosition::new("consumer_b")).unwrap();

        let cursors = list_cursors(&db).unwrap();
        assert!(cursors.len() >= 2);
    }
}
