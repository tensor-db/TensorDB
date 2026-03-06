//! WAL shipper: tails local WAL and ships frames to follower nodes.
//!
//! The shipper runs as a background task, reading new WAL frames from the
//! leader and sending them to followers for replay. Each follower tracks
//! its own replication offset.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;

/// Tracks replication state for a single follower.
#[derive(Debug, Clone)]
pub struct FollowerState {
    pub node_id: String,
    pub address: String,
    /// Last WAL offset successfully replicated to this follower.
    pub last_replicated_offset: u64,
    /// Last time we received an ACK from this follower (unix ms).
    pub last_ack_ms: u64,
    /// Replication lag in bytes.
    pub lag_bytes: u64,
    /// Whether this follower is actively receiving.
    pub is_active: bool,
}

/// A WAL frame to be shipped to followers.
#[derive(Debug, Clone)]
pub struct WalFrame {
    pub offset: u64,
    pub shard_id: u32,
    pub data: Vec<u8>,
    pub timestamp_ms: u64,
}

/// WAL shipper that tails the local WAL and distributes frames to followers.
pub struct WalShipper {
    /// Map of follower node_id -> replication state.
    followers: RwLock<HashMap<String, FollowerState>>,
    /// Current WAL write offset on the leader.
    leader_offset: AtomicU64,
    /// Whether replication is running.
    running: AtomicBool,
    /// Pending frames buffer (frames not yet ACKed by all followers).
    pending_frames: RwLock<Vec<WalFrame>>,
    /// Maximum pending frames before backpressure.
    max_pending: usize,
}

impl WalShipper {
    pub fn new(max_pending: usize) -> Self {
        Self {
            followers: RwLock::new(HashMap::new()),
            leader_offset: AtomicU64::new(0),
            running: AtomicBool::new(false),
            pending_frames: RwLock::new(Vec::new()),
            max_pending,
        }
    }

    /// Register a follower for replication.
    pub fn add_follower(&self, node_id: String, address: String) {
        let state = FollowerState {
            node_id: node_id.clone(),
            address,
            last_replicated_offset: 0,
            last_ack_ms: 0,
            lag_bytes: 0,
            is_active: true,
        };
        self.followers.write().unwrap().insert(node_id, state);
    }

    /// Remove a follower from replication.
    pub fn remove_follower(&self, node_id: &str) -> bool {
        self.followers.write().unwrap().remove(node_id).is_some()
    }

    /// Enqueue a WAL frame for replication to all followers.
    /// Returns the number of pending frames.
    pub fn enqueue_frame(&self, frame: WalFrame) -> usize {
        self.leader_offset.store(frame.offset, Ordering::SeqCst);
        let mut pending = self.pending_frames.write().unwrap();
        pending.push(frame);

        // Trim frames that have been ACKed by all followers
        let min_offset = self.min_replicated_offset();
        pending.retain(|f| f.offset > min_offset);

        pending.len()
    }

    /// Get frames that need to be sent to a specific follower.
    pub fn frames_for_follower(&self, node_id: &str) -> Vec<WalFrame> {
        let followers = self.followers.read().unwrap();
        let follower_offset = followers
            .get(node_id)
            .map_or(0, |f| f.last_replicated_offset);
        let pending = self.pending_frames.read().unwrap();
        pending
            .iter()
            .filter(|f| f.offset > follower_offset)
            .cloned()
            .collect()
    }

    /// Record that a follower has acknowledged frames up to `offset`.
    pub fn ack_follower(&self, node_id: &str, offset: u64) {
        let mut followers = self.followers.write().unwrap();
        if let Some(state) = followers.get_mut(node_id) {
            state.last_replicated_offset = offset;
            state.last_ack_ms = current_timestamp_ms();
            let leader = self.leader_offset.load(Ordering::SeqCst);
            state.lag_bytes = leader.saturating_sub(offset);
        }
    }

    /// Get the minimum replicated offset across all active followers.
    pub fn min_replicated_offset(&self) -> u64 {
        let followers = self.followers.read().unwrap();
        followers
            .values()
            .filter(|f| f.is_active)
            .map(|f| f.last_replicated_offset)
            .min()
            .unwrap_or(0)
    }

    /// Get replication status for all followers.
    pub fn status(&self) -> Vec<FollowerState> {
        self.followers.read().unwrap().values().cloned().collect()
    }

    /// Check if replication has backpressure (too many unACKed frames).
    pub fn has_backpressure(&self) -> bool {
        self.pending_frames.read().unwrap().len() >= self.max_pending
    }

    /// Start the replication loop.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
    }

    /// Stop the replication loop.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if the shipper is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Current leader WAL offset.
    pub fn leader_offset(&self) -> u64 {
        self.leader_offset.load(Ordering::SeqCst)
    }

    /// Number of registered followers.
    pub fn follower_count(&self) -> usize {
        self.followers.read().unwrap().len()
    }
}

impl Default for WalShipper {
    fn default() -> Self {
        Self::new(10_000)
    }
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_remove_follower() {
        let shipper = WalShipper::new(100);
        shipper.add_follower("f1".to_string(), "127.0.0.1:9200".to_string());
        assert_eq!(shipper.follower_count(), 1);
        assert!(shipper.remove_follower("f1"));
        assert_eq!(shipper.follower_count(), 0);
    }

    #[test]
    fn enqueue_and_ack_frames() {
        let shipper = WalShipper::new(100);
        shipper.add_follower("f1".to_string(), "127.0.0.1:9200".to_string());

        let frame1 = WalFrame {
            offset: 1,
            shard_id: 0,
            data: vec![1, 2, 3],
            timestamp_ms: 1000,
        };
        let frame2 = WalFrame {
            offset: 2,
            shard_id: 0,
            data: vec![4, 5, 6],
            timestamp_ms: 1001,
        };
        shipper.enqueue_frame(frame1);
        shipper.enqueue_frame(frame2);

        let frames = shipper.frames_for_follower("f1");
        assert_eq!(frames.len(), 2);

        shipper.ack_follower("f1", 1);
        let frames = shipper.frames_for_follower("f1");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].offset, 2);
    }

    #[test]
    fn backpressure_detection() {
        let shipper = WalShipper::new(3);
        shipper.add_follower("f1".to_string(), "127.0.0.1:9200".to_string());

        for i in 1..=3 {
            shipper.enqueue_frame(WalFrame {
                offset: i,
                shard_id: 0,
                data: vec![],
                timestamp_ms: 1000 + i,
            });
        }
        assert!(shipper.has_backpressure());
    }

    #[test]
    fn replication_lag_tracking() {
        let shipper = WalShipper::new(100);
        shipper.add_follower("f1".to_string(), "127.0.0.1:9200".to_string());

        shipper.enqueue_frame(WalFrame {
            offset: 100,
            shard_id: 0,
            data: vec![],
            timestamp_ms: 1000,
        });

        shipper.ack_follower("f1", 50);
        let status = shipper.status();
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].lag_bytes, 50);
    }
}
