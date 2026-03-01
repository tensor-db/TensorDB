//! Seed-based node discovery for cluster bootstrapping.

use std::collections::HashSet;

/// Gossip-based discovery: contacts seed nodes to learn about the cluster.
pub struct GossipDiscovery {
    seed_nodes: Vec<String>,
    known_nodes: parking_lot::RwLock<HashSet<String>>,
}

impl GossipDiscovery {
    /// Create a new gossip discovery with the given seed nodes.
    pub fn new(seed_nodes: Vec<String>) -> Self {
        let known: HashSet<String> = seed_nodes.iter().cloned().collect();
        Self {
            seed_nodes,
            known_nodes: parking_lot::RwLock::new(known),
        }
    }

    /// Add a newly discovered node.
    pub fn add_node(&self, address: String) {
        self.known_nodes.write().insert(address);
    }

    /// Remove a node (e.g., after it leaves).
    pub fn remove_node(&self, address: &str) {
        self.known_nodes.write().remove(address);
    }

    /// Get all known nodes.
    pub fn known_nodes(&self) -> Vec<String> {
        self.known_nodes.read().iter().cloned().collect()
    }

    /// Get seed nodes.
    pub fn seed_nodes(&self) -> &[String] {
        &self.seed_nodes
    }

    /// Merge a list of nodes from a peer's gossip response.
    pub fn merge(&self, nodes: &[String]) -> usize {
        let mut known = self.known_nodes.write();
        let before = known.len();
        for node in nodes {
            known.insert(node.clone());
        }
        known.len() - before
    }

    /// Check if we know about a specific node.
    pub fn knows(&self, address: &str) -> bool {
        self.known_nodes.read().contains(address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discovery_basic() {
        let discovery = GossipDiscovery::new(vec!["seed-1:9100".to_string()]);
        assert!(discovery.knows("seed-1:9100"));
        assert!(!discovery.knows("unknown:9100"));
    }

    #[test]
    fn test_merge_nodes() {
        let discovery = GossipDiscovery::new(vec!["seed-1:9100".to_string()]);
        let added = discovery.merge(&["node-2:9100".to_string(), "node-3:9100".to_string()]);
        assert_eq!(added, 2);
        assert_eq!(discovery.known_nodes().len(), 3);
    }

    #[test]
    fn test_remove_node() {
        let discovery = GossipDiscovery::new(vec!["seed-1:9100".to_string()]);
        discovery.add_node("node-2:9100".to_string());
        assert_eq!(discovery.known_nodes().len(), 2);
        discovery.remove_node("node-2:9100");
        assert_eq!(discovery.known_nodes().len(), 1);
    }
}
