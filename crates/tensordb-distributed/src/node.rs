//! ClusterNode — the top-level coordinator for a distributed TensorDB node.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::config::DistributedConfig;
use crate::error::Result;

/// State of a peer node in the cluster.
#[derive(Debug, Clone)]
pub struct PeerState {
    pub node_id: String,
    pub address: String,
    pub shard_ids: Vec<u32>,
    pub last_heartbeat_ms: u64,
    pub is_healthy: bool,
}

/// Role of this node in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Leader,
    Follower,
    Candidate,
}

/// The top-level coordinator for a distributed TensorDB node.
pub struct ClusterNode {
    pub config: DistributedConfig,
    pub role: RwLock<NodeRole>,
    pub peers: RwLock<HashMap<String, PeerState>>,
    pub local_shards: RwLock<Vec<u32>>,
    pub current_epoch: std::sync::atomic::AtomicU64,
}

impl ClusterNode {
    /// Create a new cluster node with the given config.
    pub fn new(config: DistributedConfig) -> Arc<Self> {
        let shards: Vec<u32> = (0..config.initial_shard_count).collect();
        Arc::new(Self {
            config,
            role: RwLock::new(NodeRole::Follower),
            peers: RwLock::new(HashMap::new()),
            local_shards: RwLock::new(shards),
            current_epoch: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Register a peer node.
    pub fn add_peer(&self, peer: PeerState) {
        self.peers.write().insert(peer.node_id.clone(), peer);
    }

    /// Remove a peer by node ID.
    pub fn remove_peer(&self, node_id: &str) -> bool {
        self.peers.write().remove(node_id).is_some()
    }

    /// Check if a shard is locally owned.
    pub fn owns_shard(&self, shard_id: u32) -> bool {
        self.local_shards.read().contains(&shard_id)
    }

    /// Get the node ID that owns a given shard.
    pub fn shard_owner(&self, shard_id: u32) -> Option<String> {
        if self.owns_shard(shard_id) {
            return Some(self.config.node_id.clone());
        }
        let peers = self.peers.read();
        for peer in peers.values() {
            if peer.shard_ids.contains(&shard_id) {
                return Some(peer.node_id.clone());
            }
        }
        None
    }

    /// Get peer address by node ID.
    pub fn peer_address(&self, node_id: &str) -> Option<String> {
        self.peers.read().get(node_id).map(|p| p.address.clone())
    }

    /// Get cluster status summary.
    pub fn cluster_status(&self) -> ClusterStatus {
        let peers = self.peers.read();
        ClusterStatus {
            node_id: self.config.node_id.clone(),
            role: *self.role.read(),
            epoch: self
                .current_epoch
                .load(std::sync::atomic::Ordering::Relaxed),
            local_shard_count: self.local_shards.read().len(),
            peer_count: peers.len(),
            healthy_peers: peers.values().filter(|p| p.is_healthy).count(),
        }
    }

    /// Update heartbeat timestamp for a peer.
    pub fn record_heartbeat(&self, node_id: &str, epoch: u64) -> Result<()> {
        let mut peers = self.peers.write();
        if let Some(peer) = peers.get_mut(node_id) {
            peer.last_heartbeat_ms = current_time_ms();
            peer.is_healthy = true;
            // Update epoch if peer has a newer one
            let current = self
                .current_epoch
                .load(std::sync::atomic::Ordering::Relaxed);
            if epoch > current {
                self.current_epoch
                    .store(epoch, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(())
    }
}

/// Summary of cluster state.
#[derive(Debug, Clone)]
pub struct ClusterStatus {
    pub node_id: String,
    pub role: NodeRole,
    pub epoch: u64,
    pub local_shard_count: usize,
    pub peer_count: usize,
    pub healthy_peers: usize,
}

fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_node_creation() {
        let cfg = DistributedConfig {
            initial_shard_count: 4,
            ..DistributedConfig::default()
        };
        let node = ClusterNode::new(cfg);
        assert_eq!(node.local_shards.read().len(), 4);
        assert!(node.owns_shard(0));
        assert!(node.owns_shard(3));
        assert!(!node.owns_shard(4));
    }

    #[test]
    fn test_peer_management() {
        let node = ClusterNode::new(DistributedConfig::default());
        node.add_peer(PeerState {
            node_id: "peer-1".to_string(),
            address: "127.0.0.1:9101".to_string(),
            shard_ids: vec![4, 5],
            last_heartbeat_ms: 0,
            is_healthy: true,
        });

        assert_eq!(node.shard_owner(4), Some("peer-1".to_string()));
        assert_eq!(
            node.peer_address("peer-1"),
            Some("127.0.0.1:9101".to_string())
        );
        assert!(node.remove_peer("peer-1"));
        assert!(!node.remove_peer("peer-1"));
    }

    #[test]
    fn test_cluster_status() {
        let node = ClusterNode::new(DistributedConfig::default());
        let status = node.cluster_status();
        assert_eq!(status.local_shard_count, 4);
        assert_eq!(status.peer_count, 0);
    }
}
