//! Distributed configuration for TensorDB cluster nodes.

use serde::{Deserialize, Serialize};

/// Configuration for a TensorDB cluster node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Unique identifier for this node.
    pub node_id: String,
    /// gRPC listen address (e.g., "0.0.0.0:9100").
    pub listen_addr: String,
    /// Advertised address for other nodes to connect to.
    pub advertise_addr: String,
    /// Seed nodes for initial cluster discovery.
    pub seed_nodes: Vec<String>,
    /// Number of shards this node owns initially.
    pub initial_shard_count: u32,
    /// Maximum concurrent gRPC connections per peer.
    pub max_connections_per_peer: usize,
    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval_ms: u64,
    /// Heartbeat timeout for failure detection (ms).
    pub heartbeat_timeout_ms: u64,
    /// 2PC prepare timeout (ms).
    pub txn_prepare_timeout_ms: u64,
    /// Migration rate limit (bytes/sec).
    pub migration_rate_limit: u64,
    /// Rebalance threshold — trigger rebalance if shard count imbalance exceeds this.
    pub rebalance_threshold: f64,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            node_id: format!("node-{}", fastrand::u64(..)),
            listen_addr: "0.0.0.0:9100".to_string(),
            advertise_addr: "127.0.0.1:9100".to_string(),
            seed_nodes: Vec::new(),
            initial_shard_count: 4,
            max_connections_per_peer: 4,
            heartbeat_interval_ms: 1000,
            heartbeat_timeout_ms: 5000,
            txn_prepare_timeout_ms: 30_000,
            migration_rate_limit: 50 * 1024 * 1024, // 50 MB/s
            rebalance_threshold: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = DistributedConfig::default();
        assert_eq!(cfg.listen_addr, "0.0.0.0:9100");
        assert_eq!(cfg.initial_shard_count, 4);
        assert_eq!(cfg.heartbeat_interval_ms, 1000);
    }

    #[test]
    fn test_config_serialization() {
        let cfg = DistributedConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: DistributedConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.listen_addr, cfg.listen_addr);
    }
}
