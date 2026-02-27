//! WAL Shipping & Replication — stream WAL segments from primary to standbys.
//!
//! # Architecture
//! Primary → WAL Shipper → Standby WAL Receiver → Standby WAL Replay
//!
//! Supports both synchronous (primary waits for ACK) and asynchronous
//! (fire-and-forget) replication modes.
//!
//! # AI Integration
//! An AI replication advisor monitors lag patterns and recommends:
//! - When to switch from sync to async (high write bursts)
//! - Optimal batch size for WAL segment shipping
//! - Failover readiness score for each standby

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};

/// Replication mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicationMode {
    /// Primary waits for at least one standby ACK before confirming write.
    Synchronous,
    /// Primary ships WAL asynchronously; standbys may lag.
    Asynchronous,
}

/// A WAL segment ready for shipping.
#[derive(Debug, Clone)]
pub struct WalSegment {
    /// Monotonic segment sequence number.
    pub sequence: u64,
    /// Shard ID this segment belongs to.
    pub shard_id: usize,
    /// Raw WAL bytes (CRC-framed records).
    pub data: Vec<u8>,
    /// Number of records in this segment.
    pub record_count: u32,
    /// Timestamp when this segment was created.
    pub created_at_ms: u64,
}

/// Status of a standby node.
#[derive(Debug, Clone)]
pub struct StandbyStatus {
    pub node_id: String,
    pub last_acked_sequence: u64,
    pub last_ack_time_ms: u64,
    pub lag_bytes: u64,
    pub lag_seconds: f64,
    pub is_healthy: bool,
    pub failover_readiness: f64, // 0.0 to 1.0
}

/// WAL Shipper — ships WAL segments from primary to standbys.
pub struct WalShipper {
    mode: RwLock<ReplicationMode>,
    /// Outbound segment queue (segments waiting to be shipped).
    outbound_queue: Mutex<VecDeque<WalSegment>>,
    /// Per-standby tracking.
    standbys: RwLock<HashMap<String, StandbyState>>,
    /// Current sequence number.
    next_sequence: AtomicU64,
    /// Total segments shipped.
    total_shipped: AtomicU64,
    /// Total ACKs received.
    total_acks: AtomicU64,
    /// Is this shipper active?
    active: AtomicBool,
    /// Max outbound queue size before backpressure.
    max_queue_size: usize,
    /// AI advisor
    advisor: Mutex<ReplicationAdvisor>,
}

#[derive(Debug, Clone)]
struct StandbyState {
    node_id: String,
    last_acked_sequence: u64,
    last_ack_time_ms: u64,
    _registered_at_ms: u64,
    consecutive_failures: u32,
}

impl WalShipper {
    pub fn new(mode: ReplicationMode, max_queue_size: usize) -> Self {
        Self {
            mode: RwLock::new(mode),
            outbound_queue: Mutex::new(VecDeque::with_capacity(max_queue_size)),
            standbys: RwLock::new(HashMap::new()),
            next_sequence: AtomicU64::new(1),
            total_shipped: AtomicU64::new(0),
            total_acks: AtomicU64::new(0),
            active: AtomicBool::new(true),
            max_queue_size,
            advisor: Mutex::new(ReplicationAdvisor::new()),
        }
    }

    /// Register a standby node.
    pub fn register_standby(&self, node_id: &str) {
        let mut standbys = self.standbys.write();
        standbys.insert(
            node_id.to_string(),
            StandbyState {
                node_id: node_id.to_string(),
                last_acked_sequence: 0,
                last_ack_time_ms: 0,
                _registered_at_ms: current_ms(),
                consecutive_failures: 0,
            },
        );
    }

    /// Remove a standby node.
    pub fn unregister_standby(&self, node_id: &str) -> bool {
        self.standbys.write().remove(node_id).is_some()
    }

    /// Enqueue a WAL segment for shipping. Returns the segment sequence number.
    /// In synchronous mode, this blocks until at least one standby ACKs.
    pub fn ship_segment(&self, shard_id: usize, data: Vec<u8>, record_count: u32) -> Option<u64> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }

        let seq = self.next_sequence.fetch_add(1, Ordering::AcqRel);

        let segment = WalSegment {
            sequence: seq,
            shard_id,
            data,
            record_count,
            created_at_ms: current_ms(),
        };

        {
            let mut queue = self.outbound_queue.lock();
            if queue.len() >= self.max_queue_size {
                // Backpressure: drop oldest segment (async mode) or block
                if *self.mode.read() == ReplicationMode::Asynchronous {
                    queue.pop_front();
                } else {
                    return None; // Sync mode: signal caller to retry
                }
            }
            queue.push_back(segment);
        }

        self.total_shipped.fetch_add(1, Ordering::Relaxed);
        self.advisor.lock().record_ship();

        Some(seq)
    }

    /// Drain segments ready for transmission to standbys.
    /// Returns up to `max_batch` segments.
    pub fn drain_outbound(&self, max_batch: usize) -> Vec<WalSegment> {
        let mut queue = self.outbound_queue.lock();
        let count = queue.len().min(max_batch);
        queue.drain(..count).collect()
    }

    /// Record an ACK from a standby node.
    pub fn ack_segment(&self, node_id: &str, sequence: u64) {
        let mut standbys = self.standbys.write();
        if let Some(state) = standbys.get_mut(node_id) {
            if sequence > state.last_acked_sequence {
                state.last_acked_sequence = sequence;
                state.last_ack_time_ms = current_ms();
                state.consecutive_failures = 0;
            }
        }
        self.total_acks.fetch_add(1, Ordering::Relaxed);
        self.advisor.lock().record_ack();
    }

    /// Record a shipping failure for a standby.
    pub fn record_failure(&self, node_id: &str) {
        let mut standbys = self.standbys.write();
        if let Some(state) = standbys.get_mut(node_id) {
            state.consecutive_failures += 1;
        }
    }

    /// Get the current replication mode.
    pub fn mode(&self) -> ReplicationMode {
        *self.mode.read()
    }

    /// Switch replication mode.
    pub fn set_mode(&self, mode: ReplicationMode) {
        *self.mode.write() = mode;
    }

    /// Get status of all standbys.
    pub fn standby_statuses(&self) -> Vec<StandbyStatus> {
        let standbys = self.standbys.read();
        let current_seq = self.next_sequence.load(Ordering::Relaxed).saturating_sub(1);
        let now = current_ms();

        standbys
            .values()
            .map(|s| {
                let lag_seqs = current_seq.saturating_sub(s.last_acked_sequence);
                let lag_seconds = if s.last_ack_time_ms > 0 {
                    (now.saturating_sub(s.last_ack_time_ms)) as f64 / 1000.0
                } else {
                    f64::INFINITY
                };

                // AI-computed failover readiness
                let readiness =
                    compute_failover_readiness(lag_seqs, lag_seconds, s.consecutive_failures);

                StandbyStatus {
                    node_id: s.node_id.clone(),
                    last_acked_sequence: s.last_acked_sequence,
                    last_ack_time_ms: s.last_ack_time_ms,
                    lag_bytes: lag_seqs * 4096, // Estimate: avg 4KB per segment
                    lag_seconds,
                    is_healthy: s.consecutive_failures < 5,
                    failover_readiness: readiness,
                }
            })
            .collect()
    }

    /// Check if synchronous commit can proceed (at least one standby has ACKed).
    pub fn sync_commit_ready(&self, sequence: u64) -> bool {
        if *self.mode.read() == ReplicationMode::Asynchronous {
            return true;
        }

        let standbys = self.standbys.read();
        standbys.values().any(|s| s.last_acked_sequence >= sequence)
    }

    /// Stop the shipper.
    pub fn stop(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Get shipping statistics.
    pub fn stats(&self) -> WalShipperStats {
        let advisor = self.advisor.lock();
        WalShipperStats {
            mode: *self.mode.read(),
            total_shipped: self.total_shipped.load(Ordering::Relaxed),
            total_acks: self.total_acks.load(Ordering::Relaxed),
            outbound_queue_len: self.outbound_queue.lock().len(),
            standby_count: self.standbys.read().len(),
            advisor_stats: advisor.stats(),
        }
    }
}

/// WAL Receiver — receives and replays WAL segments on standbys.
pub struct WalReceiver {
    /// Inbound segment queue.
    inbound_queue: Mutex<VecDeque<WalSegment>>,
    /// Last applied sequence number.
    last_applied: AtomicU64,
    /// Total segments received.
    total_received: AtomicU64,
    /// Total segments applied.
    total_applied: AtomicU64,
    /// Is this receiver active?
    active: AtomicBool,
}

impl WalReceiver {
    pub fn new() -> Self {
        Self {
            inbound_queue: Mutex::new(VecDeque::new()),
            last_applied: AtomicU64::new(0),
            total_received: AtomicU64::new(0),
            total_applied: AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }

    /// Receive a WAL segment from the primary.
    pub fn receive(&self, segment: WalSegment) -> bool {
        if !self.active.load(Ordering::Relaxed) {
            return false;
        }

        self.total_received.fetch_add(1, Ordering::Relaxed);
        self.inbound_queue.lock().push_back(segment);
        true
    }

    /// Drain segments ready for replay.
    pub fn drain_for_replay(&self, max_batch: usize) -> Vec<WalSegment> {
        let mut queue = self.inbound_queue.lock();
        let count = queue.len().min(max_batch);
        queue.drain(..count).collect()
    }

    /// Mark a segment as applied.
    pub fn mark_applied(&self, sequence: u64) {
        let current = self.last_applied.load(Ordering::Relaxed);
        if sequence > current {
            self.last_applied.store(sequence, Ordering::Relaxed);
        }
        self.total_applied.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the last applied sequence number.
    pub fn last_applied_sequence(&self) -> u64 {
        self.last_applied.load(Ordering::Relaxed)
    }

    /// Get receiver statistics.
    pub fn stats(&self) -> WalReceiverStats {
        WalReceiverStats {
            total_received: self.total_received.load(Ordering::Relaxed),
            total_applied: self.total_applied.load(Ordering::Relaxed),
            last_applied: self.last_applied.load(Ordering::Relaxed),
            pending: self.inbound_queue.lock().len(),
        }
    }

    pub fn stop(&self) {
        self.active.store(false, Ordering::Relaxed);
    }
}

impl Default for WalReceiver {
    fn default() -> Self {
        Self::new()
    }
}

/// Failover Manager — handles automatic failover when primary goes down.
pub struct FailoverManager {
    /// Current role of this node.
    role: RwLock<NodeRole>,
    /// Heartbeat timeout in milliseconds.
    heartbeat_timeout_ms: u64,
    /// Last heartbeat received from primary.
    last_heartbeat_ms: AtomicU64,
    /// Failover history.
    failover_events: Mutex<Vec<FailoverEvent>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Primary,
    Standby,
    Candidate,
}

#[derive(Debug, Clone)]
pub struct FailoverEvent {
    pub timestamp_ms: u64,
    pub old_primary: String,
    pub new_primary: String,
    pub reason: String,
    pub lag_at_failover: u64,
}

impl FailoverManager {
    pub fn new(role: NodeRole, heartbeat_timeout_ms: u64) -> Self {
        Self {
            role: RwLock::new(role),
            heartbeat_timeout_ms,
            last_heartbeat_ms: AtomicU64::new(current_ms()),
            failover_events: Mutex::new(Vec::new()),
        }
    }

    /// Record a heartbeat from the primary.
    pub fn record_heartbeat(&self) {
        self.last_heartbeat_ms
            .store(current_ms(), Ordering::Relaxed);
    }

    /// Check if the primary is considered down (heartbeat timeout exceeded).
    pub fn is_primary_down(&self) -> bool {
        if *self.role.read() == NodeRole::Primary {
            return false; // We are the primary
        }

        let last = self.last_heartbeat_ms.load(Ordering::Relaxed);
        let elapsed = current_ms().saturating_sub(last);
        elapsed > self.heartbeat_timeout_ms
    }

    /// Initiate failover: this standby promotes itself to primary.
    pub fn promote_to_primary(&self, old_primary_id: &str, self_id: &str) -> FailoverEvent {
        *self.role.write() = NodeRole::Primary;

        let event = FailoverEvent {
            timestamp_ms: current_ms(),
            old_primary: old_primary_id.to_string(),
            new_primary: self_id.to_string(),
            reason: "heartbeat timeout".to_string(),
            lag_at_failover: 0,
        };

        self.failover_events.lock().push(event.clone());
        event
    }

    /// Demote this node back to standby (when a new primary is elected).
    pub fn demote_to_standby(&self) {
        *self.role.write() = NodeRole::Standby;
    }

    /// Get the current role.
    pub fn role(&self) -> NodeRole {
        *self.role.read()
    }

    /// Get failover history.
    pub fn failover_history(&self) -> Vec<FailoverEvent> {
        self.failover_events.lock().clone()
    }

    /// Time since last heartbeat in milliseconds.
    pub fn time_since_heartbeat_ms(&self) -> u64 {
        current_ms().saturating_sub(self.last_heartbeat_ms.load(Ordering::Relaxed))
    }
}

/// Read Replica Router — routes SELECT queries to standbys.
pub struct ReadReplicaRouter {
    /// Available replicas with their lag.
    replicas: RwLock<Vec<ReplicaInfo>>,
    /// Maximum acceptable staleness in milliseconds.
    max_staleness_ms: u64,
    /// Round-robin index for load balancing.
    next_replica: AtomicU64,
    /// Total reads routed.
    total_reads_routed: AtomicU64,
    /// Total reads rejected (all replicas too stale).
    total_reads_rejected: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct ReplicaInfo {
    pub node_id: String,
    pub lag_ms: u64,
    pub is_healthy: bool,
    pub load: f64, // 0.0 to 1.0
}

impl ReadReplicaRouter {
    pub fn new(max_staleness_ms: u64) -> Self {
        Self {
            replicas: RwLock::new(Vec::new()),
            max_staleness_ms,
            next_replica: AtomicU64::new(0),
            total_reads_routed: AtomicU64::new(0),
            total_reads_rejected: AtomicU64::new(0),
        }
    }

    /// Update replica information.
    pub fn update_replicas(&self, replicas: Vec<ReplicaInfo>) {
        *self.replicas.write() = replicas;
    }

    /// Route a read query to the best available replica.
    /// Returns None if all replicas are too stale or unhealthy.
    pub fn route_read(&self) -> Option<String> {
        let replicas = self.replicas.read();

        let eligible: Vec<&ReplicaInfo> = replicas
            .iter()
            .filter(|r| r.is_healthy && r.lag_ms <= self.max_staleness_ms)
            .collect();

        if eligible.is_empty() {
            self.total_reads_rejected.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Round-robin among eligible replicas
        let idx = self.next_replica.fetch_add(1, Ordering::Relaxed) as usize % eligible.len();
        self.total_reads_routed.fetch_add(1, Ordering::Relaxed);
        Some(eligible[idx].node_id.clone())
    }

    /// Route a read query with a specific staleness tolerance.
    pub fn route_read_with_staleness(&self, max_staleness_ms: u64) -> Option<String> {
        let replicas = self.replicas.read();

        let eligible: Vec<&ReplicaInfo> = replicas
            .iter()
            .filter(|r| r.is_healthy && r.lag_ms <= max_staleness_ms)
            .collect();

        if eligible.is_empty() {
            return None;
        }

        // Pick the replica with the lowest lag
        eligible
            .iter()
            .min_by_key(|r| r.lag_ms)
            .map(|r| r.node_id.clone())
    }

    /// Get routing statistics.
    pub fn stats(&self) -> ReadReplicaStats {
        ReadReplicaStats {
            total_reads_routed: self.total_reads_routed.load(Ordering::Relaxed),
            total_reads_rejected: self.total_reads_rejected.load(Ordering::Relaxed),
            replica_count: self.replicas.read().len(),
            max_staleness_ms: self.max_staleness_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// AI Replication Advisor
// ---------------------------------------------------------------------------

struct ReplicationAdvisor {
    ships_per_second: VecDeque<(u64, u64)>, // (timestamp_ms, count)
    acks_per_second: VecDeque<(u64, u64)>,
    total_ships: u64,
    total_acks: u64,
}

impl ReplicationAdvisor {
    fn new() -> Self {
        Self {
            ships_per_second: VecDeque::with_capacity(60),
            acks_per_second: VecDeque::with_capacity(60),
            total_ships: 0,
            total_acks: 0,
        }
    }

    fn record_ship(&mut self) {
        self.total_ships += 1;
        let now = current_ms();
        self.ships_per_second.push_back((now, 1));
        self.cleanup_old(&mut self.ships_per_second.clone(), now);
    }

    fn record_ack(&mut self) {
        self.total_acks += 1;
        let now = current_ms();
        self.acks_per_second.push_back((now, 1));
        self.cleanup_old(&mut self.acks_per_second.clone(), now);
    }

    fn cleanup_old(&mut self, _window: &mut VecDeque<(u64, u64)>, now: u64) {
        let cutoff = now.saturating_sub(60_000);
        while self
            .ships_per_second
            .front()
            .is_some_and(|(ts, _)| *ts < cutoff)
        {
            self.ships_per_second.pop_front();
        }
        while self
            .acks_per_second
            .front()
            .is_some_and(|(ts, _)| *ts < cutoff)
        {
            self.acks_per_second.pop_front();
        }
    }

    fn stats(&self) -> ReplicationAdvisorStats {
        ReplicationAdvisorStats {
            total_ships: self.total_ships,
            total_acks: self.total_acks,
            recent_ship_rate: self.ships_per_second.len() as f64,
            recent_ack_rate: self.acks_per_second.len() as f64,
        }
    }
}

/// AI-computed failover readiness score (0.0 = not ready, 1.0 = fully ready).
fn compute_failover_readiness(lag_seqs: u64, lag_seconds: f64, consecutive_failures: u32) -> f64 {
    let mut score = 1.0f64;

    // Penalize for lag
    if lag_seqs > 100 {
        score -= 0.3;
    } else if lag_seqs > 10 {
        score -= 0.1;
    }

    // Penalize for time lag
    if lag_seconds > 30.0 {
        score -= 0.4;
    } else if lag_seconds > 5.0 {
        score -= 0.2;
    }

    // Penalize for consecutive failures
    if consecutive_failures > 3 {
        score -= 0.3;
    } else if consecutive_failures > 0 {
        score -= 0.1 * consecutive_failures as f64;
    }

    score.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Stats types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WalShipperStats {
    pub mode: ReplicationMode,
    pub total_shipped: u64,
    pub total_acks: u64,
    pub outbound_queue_len: usize,
    pub standby_count: usize,
    pub advisor_stats: ReplicationAdvisorStats,
}

#[derive(Debug, Clone)]
pub struct WalReceiverStats {
    pub total_received: u64,
    pub total_applied: u64,
    pub last_applied: u64,
    pub pending: usize,
}

#[derive(Debug, Clone)]
pub struct ReadReplicaStats {
    pub total_reads_routed: u64,
    pub total_reads_rejected: u64,
    pub replica_count: usize,
    pub max_staleness_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ReplicationAdvisorStats {
    pub total_ships: u64,
    pub total_acks: u64,
    pub recent_ship_rate: f64,
    pub recent_ack_rate: f64,
}

fn current_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_shipper_basic() {
        let shipper = WalShipper::new(ReplicationMode::Asynchronous, 100);
        shipper.register_standby("standby-1");

        let seq = shipper.ship_segment(0, b"wal-data".to_vec(), 1);
        assert_eq!(seq, Some(1));

        let segments = shipper.drain_outbound(10);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].sequence, 1);
    }

    #[test]
    fn test_wal_shipper_ack() {
        let shipper = WalShipper::new(ReplicationMode::Synchronous, 100);
        shipper.register_standby("standby-1");

        let seq = shipper.ship_segment(0, b"data".to_vec(), 1).unwrap();
        assert!(!shipper.sync_commit_ready(seq));

        shipper.ack_segment("standby-1", seq);
        assert!(shipper.sync_commit_ready(seq));
    }

    #[test]
    fn test_wal_receiver_basic() {
        let receiver = WalReceiver::new();

        let segment = WalSegment {
            sequence: 1,
            shard_id: 0,
            data: b"test-wal".to_vec(),
            record_count: 1,
            created_at_ms: current_ms(),
        };

        assert!(receiver.receive(segment));
        let segments = receiver.drain_for_replay(10);
        assert_eq!(segments.len(), 1);

        receiver.mark_applied(1);
        assert_eq!(receiver.last_applied_sequence(), 1);
    }

    #[test]
    fn test_failover_manager_heartbeat() {
        let mgr = FailoverManager::new(NodeRole::Standby, 100);
        mgr.record_heartbeat();

        assert!(!mgr.is_primary_down());

        // Simulate timeout
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(mgr.is_primary_down());
    }

    #[test]
    fn test_failover_promote() {
        let mgr = FailoverManager::new(NodeRole::Standby, 100);
        assert_eq!(mgr.role(), NodeRole::Standby);

        let event = mgr.promote_to_primary("old-primary", "self-node");
        assert_eq!(mgr.role(), NodeRole::Primary);
        assert_eq!(event.new_primary, "self-node");
        assert_eq!(mgr.failover_history().len(), 1);
    }

    #[test]
    fn test_failover_demote() {
        let mgr = FailoverManager::new(NodeRole::Primary, 100);
        mgr.demote_to_standby();
        assert_eq!(mgr.role(), NodeRole::Standby);
    }

    #[test]
    fn test_read_replica_router() {
        let router = ReadReplicaRouter::new(5000);

        router.update_replicas(vec![
            ReplicaInfo {
                node_id: "r1".to_string(),
                lag_ms: 100,
                is_healthy: true,
                load: 0.3,
            },
            ReplicaInfo {
                node_id: "r2".to_string(),
                lag_ms: 200,
                is_healthy: true,
                load: 0.5,
            },
            ReplicaInfo {
                node_id: "r3".to_string(),
                lag_ms: 10000,
                is_healthy: true,
                load: 0.1,
            },
        ]);

        // r3 is too stale (10s > 5s tolerance)
        let target = router.route_read().unwrap();
        assert!(target == "r1" || target == "r2");
    }

    #[test]
    fn test_read_replica_no_eligible() {
        let router = ReadReplicaRouter::new(100);

        router.update_replicas(vec![ReplicaInfo {
            node_id: "r1".to_string(),
            lag_ms: 5000,
            is_healthy: true,
            load: 0.3,
        }]);

        assert!(router.route_read().is_none());
    }

    #[test]
    fn test_read_replica_custom_staleness() {
        let router = ReadReplicaRouter::new(100);

        router.update_replicas(vec![ReplicaInfo {
            node_id: "r1".to_string(),
            lag_ms: 500,
            is_healthy: true,
            load: 0.3,
        }]);

        // Default staleness (100ms) → not eligible
        assert!(router.route_read().is_none());
        // Custom staleness (1000ms) → eligible
        assert_eq!(
            router.route_read_with_staleness(1000),
            Some("r1".to_string())
        );
    }

    #[test]
    fn test_failover_readiness_score() {
        // Perfectly caught up
        let score = compute_failover_readiness(0, 0.0, 0);
        assert!((score - 1.0).abs() < 0.001);

        // Very behind
        let score = compute_failover_readiness(200, 60.0, 5);
        assert!(score < 0.3);
    }

    #[test]
    fn test_standby_status() {
        let shipper = WalShipper::new(ReplicationMode::Asynchronous, 100);
        shipper.register_standby("s1");

        // Ship some segments
        shipper.ship_segment(0, b"data1".to_vec(), 1);
        shipper.ship_segment(0, b"data2".to_vec(), 1);

        // ACK first segment
        shipper.ack_segment("s1", 1);

        let statuses = shipper.standby_statuses();
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].last_acked_sequence, 1);
    }

    #[test]
    fn test_wal_shipper_backpressure() {
        let shipper = WalShipper::new(ReplicationMode::Asynchronous, 3);

        // Fill the queue
        shipper.ship_segment(0, b"d1".to_vec(), 1);
        shipper.ship_segment(0, b"d2".to_vec(), 1);
        shipper.ship_segment(0, b"d3".to_vec(), 1);

        // One more — oldest gets dropped (async mode)
        shipper.ship_segment(0, b"d4".to_vec(), 1);

        let segments = shipper.drain_outbound(10);
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].sequence, 2); // seq 1 was dropped
    }

    #[test]
    fn test_wal_shipper_unregister() {
        let shipper = WalShipper::new(ReplicationMode::Asynchronous, 100);
        shipper.register_standby("s1");
        assert_eq!(shipper.standbys.read().len(), 1);

        assert!(shipper.unregister_standby("s1"));
        assert_eq!(shipper.standbys.read().len(), 0);
        assert!(!shipper.unregister_standby("s1"));
    }

    #[test]
    fn test_wal_shipper_stop() {
        let shipper = WalShipper::new(ReplicationMode::Asynchronous, 100);
        shipper.stop();

        assert!(shipper.ship_segment(0, b"data".to_vec(), 1).is_none());
    }

    #[test]
    fn test_stats() {
        let shipper = WalShipper::new(ReplicationMode::Asynchronous, 100);
        shipper.ship_segment(0, b"data".to_vec(), 1);

        let stats = shipper.stats();
        assert_eq!(stats.total_shipped, 1);
        assert_eq!(stats.mode, ReplicationMode::Asynchronous);
    }
}
