use crate::engine::db::Database;
use crate::error::{Result, TensorError};

/// Key prefix for node registry.
const NODE_PREFIX: &str = "__cluster/node/";
/// Key for cluster configuration.
const CLUSTER_CONFIG_KEY: &str = "__cluster/config";

/// Role of a node in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NodeRole {
    Leader,
    Follower,
    Candidate,
    Observer,
}

/// Status of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NodeStatus {
    Active,
    Joining,
    Leaving,
    Down,
}

/// Information about a cluster node.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub address: String,
    pub port: u16,
    pub role: NodeRole,
    pub status: NodeStatus,
    pub shard_assignments: Vec<usize>,
    pub joined_at: u64,
    pub last_heartbeat: u64,
    pub raft_term: u64,
}

/// Cluster configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterConfig {
    pub cluster_id: String,
    pub replication_factor: usize,
    pub min_nodes: usize,
    pub heartbeat_interval_ms: u64,
    pub election_timeout_ms: u64,
    pub node_count: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        ClusterConfig {
            cluster_id: "tensordb-cluster".to_string(),
            replication_factor: 3,
            min_nodes: 1,
            heartbeat_interval_ms: 100,
            election_timeout_ms: 300,
            node_count: 0,
        }
    }
}

/// Manages cluster node membership.
pub struct NodeRegistry;

impl NodeRegistry {
    /// Register a new node in the cluster.
    pub fn register_node(db: &Database, node: &NodeInfo) -> Result<()> {
        let key = format!("{}{}", NODE_PREFIX, node.node_id);
        if let Some(bytes) = db.get(key.as_bytes(), None, None)? {
            if let Ok(existing) = serde_json::from_slice::<NodeInfo>(&bytes) {
                if existing.status != NodeStatus::Down {
                    return Err(TensorError::SqlExec(format!(
                        "node already registered: {}",
                        node.node_id
                    )));
                }
            }
        }
        let value = serde_json::to_vec(node)
            .map_err(|e| TensorError::SqlExec(format!("failed to serialize node: {e}")))?;
        db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        Ok(())
    }

    /// Update a node's information.
    pub fn update_node(db: &Database, node: &NodeInfo) -> Result<()> {
        let key = format!("{}{}", NODE_PREFIX, node.node_id);
        let value = serde_json::to_vec(node)
            .map_err(|e| TensorError::SqlExec(format!("failed to serialize node: {e}")))?;
        db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        Ok(())
    }

    /// Get a node by ID.
    pub fn get_node(db: &Database, node_id: &str) -> Result<Option<NodeInfo>> {
        let key = format!("{}{}", NODE_PREFIX, node_id);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let node: NodeInfo = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse node: {e}")))?;
                Ok(Some(node))
            }
            None => Ok(None),
        }
    }

    /// List all nodes.
    pub fn list_nodes(db: &Database) -> Result<Vec<NodeInfo>> {
        let rows = db.scan_prefix(NODE_PREFIX.as_bytes(), None, None, None)?;
        let mut nodes = Vec::new();
        for row in rows {
            if let Ok(node) = serde_json::from_slice::<NodeInfo>(&row.doc) {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }

    /// List active nodes only.
    pub fn active_nodes(db: &Database) -> Result<Vec<NodeInfo>> {
        let nodes = Self::list_nodes(db)?;
        Ok(nodes
            .into_iter()
            .filter(|n| n.status == NodeStatus::Active)
            .collect())
    }

    /// Mark a node as down.
    pub fn mark_down(db: &Database, node_id: &str) -> Result<()> {
        if let Some(mut node) = Self::get_node(db, node_id)? {
            node.status = NodeStatus::Down;
            Self::update_node(db, &node)?;
        }
        Ok(())
    }

    /// Mark a node as leaving.
    pub fn mark_leaving(db: &Database, node_id: &str) -> Result<()> {
        if let Some(mut node) = Self::get_node(db, node_id)? {
            node.status = NodeStatus::Leaving;
            Self::update_node(db, &node)?;
        }
        Ok(())
    }

    /// Update a node's heartbeat timestamp.
    pub fn heartbeat(db: &Database, node_id: &str) -> Result<()> {
        if let Some(mut node) = Self::get_node(db, node_id)? {
            node.last_heartbeat = current_timestamp_ms();
            Self::update_node(db, &node)?;
        }
        Ok(())
    }

    /// Find the current leader.
    pub fn find_leader(db: &Database) -> Result<Option<NodeInfo>> {
        let nodes = Self::list_nodes(db)?;
        Ok(nodes.into_iter().find(|n| n.role == NodeRole::Leader))
    }

    /// Get or create cluster configuration.
    pub fn get_cluster_config(db: &Database) -> Result<ClusterConfig> {
        match db.get(CLUSTER_CONFIG_KEY.as_bytes(), None, None)? {
            Some(bytes) => {
                let config: ClusterConfig = serde_json::from_slice(&bytes).map_err(|e| {
                    TensorError::SqlExec(format!("failed to parse cluster config: {e}"))
                })?;
                Ok(config)
            }
            None => Ok(ClusterConfig::default()),
        }
    }

    /// Save cluster configuration.
    pub fn save_cluster_config(db: &Database, config: &ClusterConfig) -> Result<()> {
        let value = serde_json::to_vec(config).map_err(|e| {
            TensorError::SqlExec(format!("failed to serialize cluster config: {e}"))
        })?;
        db.put(CLUSTER_CONFIG_KEY.as_bytes(), value, 0, u64::MAX, None)?;
        Ok(())
    }

    /// Compute shard assignments for the cluster using consistent hashing.
    pub fn compute_shard_assignments(
        nodes: &[NodeInfo],
        total_shards: usize,
        replication_factor: usize,
    ) -> std::collections::HashMap<String, Vec<usize>> {
        let mut assignments: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();

        let active: Vec<&NodeInfo> = nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Active)
            .collect();

        if active.is_empty() {
            return assignments;
        }

        let rf = replication_factor.min(active.len());

        for shard_id in 0..total_shards {
            for replica in 0..rf {
                let node_idx = (shard_id + replica) % active.len();
                assignments
                    .entry(active[node_idx].node_id.clone())
                    .or_default()
                    .push(shard_id);
            }
        }

        assignments
    }
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

    fn make_node(id: &str, role: NodeRole, status: NodeStatus) -> NodeInfo {
        NodeInfo {
            node_id: id.to_string(),
            address: "127.0.0.1".to_string(),
            port: 5433,
            role,
            status,
            shard_assignments: vec![],
            joined_at: current_timestamp_ms(),
            last_heartbeat: current_timestamp_ms(),
            raft_term: 0,
        }
    }

    #[test]
    fn test_register_and_get_node() {
        let (db, _dir) = setup();
        let node = make_node("node1", NodeRole::Follower, NodeStatus::Active);
        NodeRegistry::register_node(&db, &node).unwrap();

        let retrieved = NodeRegistry::get_node(&db, "node1").unwrap().unwrap();
        assert_eq!(retrieved.node_id, "node1");
        assert_eq!(retrieved.role, NodeRole::Follower);
        assert_eq!(retrieved.status, NodeStatus::Active);
    }

    #[test]
    fn test_duplicate_node_rejected() {
        let (db, _dir) = setup();
        let node = make_node("node1", NodeRole::Follower, NodeStatus::Active);
        NodeRegistry::register_node(&db, &node).unwrap();
        assert!(NodeRegistry::register_node(&db, &node).is_err());
    }

    #[test]
    fn test_rejoin_after_down() {
        let (db, _dir) = setup();
        let node = make_node("node1", NodeRole::Follower, NodeStatus::Active);
        NodeRegistry::register_node(&db, &node).unwrap();
        NodeRegistry::mark_down(&db, "node1").unwrap();

        // Should allow re-register after marked down
        let rejoined = make_node("node1", NodeRole::Follower, NodeStatus::Active);
        NodeRegistry::register_node(&db, &rejoined).unwrap();
    }

    #[test]
    fn test_list_nodes() {
        let (db, _dir) = setup();
        NodeRegistry::register_node(&db, &make_node("n1", NodeRole::Leader, NodeStatus::Active))
            .unwrap();
        NodeRegistry::register_node(
            &db,
            &make_node("n2", NodeRole::Follower, NodeStatus::Active),
        )
        .unwrap();
        NodeRegistry::register_node(&db, &make_node("n3", NodeRole::Follower, NodeStatus::Down))
            .unwrap();

        let all = NodeRegistry::list_nodes(&db).unwrap();
        assert_eq!(all.len(), 3);

        let active = NodeRegistry::active_nodes(&db).unwrap();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_find_leader() {
        let (db, _dir) = setup();
        NodeRegistry::register_node(
            &db,
            &make_node("n1", NodeRole::Follower, NodeStatus::Active),
        )
        .unwrap();
        NodeRegistry::register_node(&db, &make_node("n2", NodeRole::Leader, NodeStatus::Active))
            .unwrap();

        let leader = NodeRegistry::find_leader(&db).unwrap().unwrap();
        assert_eq!(leader.node_id, "n2");
    }

    #[test]
    fn test_mark_leaving() {
        let (db, _dir) = setup();
        NodeRegistry::register_node(
            &db,
            &make_node("n1", NodeRole::Follower, NodeStatus::Active),
        )
        .unwrap();
        NodeRegistry::mark_leaving(&db, "n1").unwrap();

        let node = NodeRegistry::get_node(&db, "n1").unwrap().unwrap();
        assert_eq!(node.status, NodeStatus::Leaving);
    }

    #[test]
    fn test_shard_assignment_single_node() {
        let nodes = vec![make_node("n1", NodeRole::Leader, NodeStatus::Active)];
        let assignments = NodeRegistry::compute_shard_assignments(&nodes, 4, 3);
        assert_eq!(assignments.get("n1").unwrap().len(), 4);
    }

    #[test]
    fn test_shard_assignment_multi_node() {
        let nodes = vec![
            make_node("n1", NodeRole::Leader, NodeStatus::Active),
            make_node("n2", NodeRole::Follower, NodeStatus::Active),
            make_node("n3", NodeRole::Follower, NodeStatus::Active),
        ];
        let assignments = NodeRegistry::compute_shard_assignments(&nodes, 4, 2);

        // Each shard replicated to 2 nodes → 8 total assignments
        let total: usize = assignments.values().map(|v| v.len()).sum();
        assert_eq!(total, 8);
    }

    #[test]
    fn test_cluster_config() {
        let (db, _dir) = setup();
        let config = ClusterConfig {
            cluster_id: "test-cluster".to_string(),
            replication_factor: 2,
            min_nodes: 3,
            heartbeat_interval_ms: 50,
            election_timeout_ms: 200,
            node_count: 3,
        };
        NodeRegistry::save_cluster_config(&db, &config).unwrap();

        let loaded = NodeRegistry::get_cluster_config(&db).unwrap();
        assert_eq!(loaded.cluster_id, "test-cluster");
        assert_eq!(loaded.replication_factor, 2);
    }

    #[test]
    fn test_heartbeat_update() {
        let (db, _dir) = setup();
        let node = make_node("n1", NodeRole::Follower, NodeStatus::Active);
        NodeRegistry::register_node(&db, &node).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(10));
        NodeRegistry::heartbeat(&db, "n1").unwrap();

        let updated = NodeRegistry::get_node(&db, "n1").unwrap().unwrap();
        assert!(updated.last_heartbeat > node.last_heartbeat);
    }
}
