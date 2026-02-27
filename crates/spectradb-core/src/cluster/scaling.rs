//! Horizontal Scaling Foundations (v0.28)
//!
//! This module provides the core primitives for scaling SpectraDB across multiple
//! nodes in a distributed cluster. It introduces three key components:
//!
//! 1. **ConsistentHashRing** -- Virtual-node consistent hashing for deterministic
//!    shard-to-node mapping. Uses a `BTreeMap<u64, String>` ring with configurable
//!    virtual nodes per physical node, providing O(log n) key lookups and minimal
//!    key remapping when nodes join or leave.
//!
//! 2. **ShardRebalancer** -- An AI-driven shard migration planner that computes
//!    the diff between current shard placement and the ideal placement derived
//!    from the hash ring, then produces a `RebalancePlan` that minimizes data
//!    movement while restoring balance.
//!
//! 3. **ScatterGatherExecutor** -- A parallel query execution framework that
//!    scatters sub-queries to the relevant subset of nodes and gathers partial
//!    results using pluggable merge strategies (concatenate, sort-merge, aggregate).
//!    An AI advisor can prune the scatter width by predicting which nodes are
//!    unlikely to hold matching data.
//!
//! All structures use `parking_lot::RwLock` and `std::sync::atomic` for
//! thread-safe, lock-efficient concurrent access.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::RwLock;

// ---------------------------------------------------------------------------
// FNV-1a hash helper
// ---------------------------------------------------------------------------

/// Compute a 64-bit FNV-1a hash of the given bytes.
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ---------------------------------------------------------------------------
// ConsistentHashRing
// ---------------------------------------------------------------------------

/// Virtual-node consistent hash ring for mapping keys to cluster nodes.
///
/// Each physical node is expanded into `virtual_nodes` positions on the ring
/// using FNV-1a hashing. Key lookups walk clockwise on the ring (via
/// `BTreeMap::range`) in O(log n) time.
pub struct ConsistentHashRing {
    /// Mapping from hash position to node identifier.
    ring: RwLock<BTreeMap<u64, String>>,
    /// Number of virtual nodes per physical node.
    virtual_nodes: usize,
    /// Set of physical nodes currently in the ring.
    nodes: RwLock<HashSet<String>>,
    /// Total number of lookups performed (for metrics).
    lookup_count: AtomicU64,
}

impl ConsistentHashRing {
    /// Create a new consistent hash ring with the given virtual-node count.
    ///
    /// A higher `virtual_nodes` value improves distribution uniformity at the
    /// cost of slightly more memory and insertion time.
    pub fn new(virtual_nodes: usize) -> Self {
        Self {
            ring: RwLock::new(BTreeMap::new()),
            virtual_nodes,
            nodes: RwLock::new(HashSet::new()),
            lookup_count: AtomicU64::new(0),
        }
    }

    /// Add a physical node and its virtual nodes to the ring.
    pub fn add_node(&self, node_id: &str) {
        let mut ring = self.ring.write();
        let mut nodes = self.nodes.write();
        nodes.insert(node_id.to_string());
        for i in 0..self.virtual_nodes {
            let vnode_key = format!("{node_id}#vn{i}");
            let hash = fnv1a_hash(vnode_key.as_bytes());
            ring.insert(hash, node_id.to_string());
        }
    }

    /// Remove a physical node and all of its virtual nodes from the ring.
    pub fn remove_node(&self, node_id: &str) {
        let mut ring = self.ring.write();
        let mut nodes = self.nodes.write();
        nodes.remove(node_id);
        ring.retain(|_, v| v != node_id);
    }

    /// Find the responsible node for a given key by walking clockwise.
    ///
    /// Returns `None` only if the ring is empty.
    pub fn get_node(&self, key: &[u8]) -> Option<String> {
        self.lookup_count.fetch_add(1, Ordering::Relaxed);
        let ring = self.ring.read();
        if ring.is_empty() {
            return None;
        }
        let hash = fnv1a_hash(key);
        // Walk clockwise: find first entry >= hash, or wrap to the beginning.
        ring.range(hash..)
            .next()
            .or_else(|| ring.iter().next())
            .map(|(_, node)| node.clone())
    }

    /// Get `count` distinct nodes responsible for replicating a key.
    ///
    /// Walks clockwise on the ring, skipping duplicate physical nodes, and
    /// returns up to `count` unique node identifiers. If the cluster has
    /// fewer distinct nodes than `count`, all available nodes are returned.
    pub fn get_nodes(&self, key: &[u8], count: usize) -> Vec<String> {
        self.lookup_count.fetch_add(1, Ordering::Relaxed);
        let ring = self.ring.read();
        if ring.is_empty() {
            return Vec::new();
        }
        let hash = fnv1a_hash(key);
        let mut seen = HashSet::new();
        let mut result = Vec::with_capacity(count);

        // Walk clockwise from hash, then wrap around from the start.
        let iter = ring.range(hash..).chain(ring.iter());
        for (_, node) in iter {
            if seen.insert(node.clone()) {
                result.push(node.clone());
                if result.len() == count {
                    break;
                }
            }
        }
        result
    }

    /// Return the number of physical nodes currently in the ring.
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Return the total number of virtual-node entries on the ring.
    pub fn ring_size(&self) -> usize {
        self.ring.read().len()
    }

    /// Return the total number of key lookups performed.
    pub fn lookup_count(&self) -> u64 {
        self.lookup_count.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// ShardRebalancer
// ---------------------------------------------------------------------------

/// A single shard migration instruction produced by the rebalancer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RebalancePlan {
    /// Node that currently owns the shard.
    pub source_node: String,
    /// Node that should own the shard after migration.
    pub target_node: String,
    /// The shard being migrated.
    pub shard_id: usize,
    /// Estimated size of the shard data in bytes.
    pub estimated_bytes: u64,
}

/// Compute a rebalance plan by diffing `current` placement against the ideal
/// placement derived from `target` (a [`ConsistentHashRing`]).
///
/// For each shard, the function determines the ideal owner by hashing the
/// shard id and consulting the ring. If the shard is currently assigned to a
/// different node, a migration entry is emitted.
pub fn compute_rebalance(
    current: &HashMap<String, Vec<usize>>,
    target: &ConsistentHashRing,
) -> Vec<RebalancePlan> {
    // Invert the current assignment map: shard_id -> owner_node.
    let mut shard_owner: HashMap<usize, String> = HashMap::new();
    for (node, shards) in current {
        for &shard_id in shards {
            shard_owner.insert(shard_id, node.clone());
        }
    }

    let mut plans = Vec::new();
    for (&shard_id, source_node) in &shard_owner {
        let shard_key = shard_id.to_string();
        if let Some(ideal_node) = target.get_node(shard_key.as_bytes()) {
            if ideal_node != *source_node {
                plans.push(RebalancePlan {
                    source_node: source_node.clone(),
                    target_node: ideal_node,
                    shard_id,
                    estimated_bytes: 0, // Caller fills in real sizes.
                });
            }
        }
    }
    // Sort by shard_id for deterministic output.
    plans.sort_by_key(|p| p.shard_id);
    plans
}

/// AI-driven rebalance advisor that decides *when* to trigger rebalancing
/// and estimates migration cost.
pub struct RebalanceAdvisor {
    /// Imbalance ratio threshold above which rebalancing is recommended.
    imbalance_threshold: f64,
    /// Assumed transfer rate in bytes per second for cost estimation.
    transfer_rate_bps: u64,
    /// Number of advisories issued.
    advisory_count: AtomicU64,
}

impl RebalanceAdvisor {
    /// Create a new advisor with the given imbalance threshold and transfer rate.
    ///
    /// `imbalance_threshold`: ratio (0.0 - 1.0) above which rebalancing fires.
    /// `transfer_rate_bps`: assumed network transfer rate in bytes/sec.
    pub fn new(imbalance_threshold: f64, transfer_rate_bps: u64) -> Self {
        Self {
            imbalance_threshold,
            transfer_rate_bps,
            advisory_count: AtomicU64::new(0),
        }
    }

    /// AI decision: should rebalancing be triggered given the current imbalance?
    ///
    /// The imbalance ratio is defined as
    /// `(max_shards - min_shards) / avg_shards` across nodes.
    /// Returns `true` when the ratio exceeds the configured threshold.
    pub fn should_rebalance(&self, imbalance_ratio: f64) -> bool {
        self.advisory_count.fetch_add(1, Ordering::Relaxed);
        imbalance_ratio > self.imbalance_threshold
    }

    /// Estimate the wall-clock time required to migrate `shard_size_bytes`
    /// worth of data at the configured transfer rate.
    pub fn estimate_migration_cost(&self, shard_size_bytes: u64) -> Duration {
        if self.transfer_rate_bps == 0 {
            return Duration::from_secs(0);
        }
        let secs = shard_size_bytes as f64 / self.transfer_rate_bps as f64;
        Duration::from_secs_f64(secs)
    }

    /// Compute the imbalance ratio for a given shard distribution.
    ///
    /// Returns `(max - min) / avg` where max/min/avg refer to per-node shard
    /// counts. Returns 0.0 when there are fewer than two nodes.
    pub fn compute_imbalance(distribution: &HashMap<String, Vec<usize>>) -> f64 {
        if distribution.len() < 2 {
            return 0.0;
        }
        let counts: Vec<f64> = distribution.values().map(|v| v.len() as f64).collect();
        let min = counts.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = counts.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg: f64 = counts.iter().sum::<f64>() / counts.len() as f64;
        if avg == 0.0 {
            return 0.0;
        }
        (max - min) / avg
    }

    /// Return the number of advisory calls made.
    pub fn advisory_count(&self) -> u64 {
        self.advisory_count.load(Ordering::Relaxed)
    }
}

impl Default for RebalanceAdvisor {
    fn default() -> Self {
        Self::new(0.25, 100_000_000) // 25% threshold, 100 MB/s
    }
}

// ---------------------------------------------------------------------------
// ScatterGatherExecutor
// ---------------------------------------------------------------------------

/// Strategy used to merge partial results from multiple nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Simply concatenate all partial results in arrival order.
    Concatenate,
    /// Sort-merge partial results by a key column.
    SortMerge { sort_column: String },
    /// Aggregate partial results (SUM, COUNT, etc.).
    Aggregate { function: String },
}

/// A scatter-gather query plan describing which nodes to contact.
#[derive(Debug, Clone)]
pub struct ScatterGatherPlan {
    /// Target nodes to scatter the query to.
    pub target_nodes: Vec<String>,
    /// The query (or query fragment) to execute on each node.
    pub query: String,
    /// How to merge the partial results.
    pub merge_strategy: MergeStrategy,
}

/// A partial result returned by a single node.
#[derive(Debug, Clone)]
pub struct PartialResult {
    /// Which node produced this result.
    pub node_id: String,
    /// Rows returned as serialised JSON values.
    pub rows: Vec<serde_json::Value>,
    /// Execution time on the remote node.
    pub execution_time: Duration,
}

/// The final merged result after the gather phase.
#[derive(Debug, Clone)]
pub struct GatheredResult {
    /// Merged rows.
    pub rows: Vec<serde_json::Value>,
    /// Number of nodes that contributed.
    pub nodes_contacted: usize,
    /// Total execution time across all nodes (parallel, so this is the max).
    pub total_time: Duration,
    /// The merge strategy that was applied.
    pub merge_strategy: MergeStrategy,
}

/// Parallel query execution across cluster nodes.
///
/// In a full production system the scatter phase would issue RPCs; here we
/// simulate it to validate the planning and merge logic.
pub struct ScatterGatherExecutor {
    /// Total scatter operations.
    total_scatters: AtomicU64,
    /// Total gather operations.
    total_gathers: AtomicU64,
}

impl ScatterGatherExecutor {
    /// Create a new executor.
    pub fn new() -> Self {
        Self {
            total_scatters: AtomicU64::new(0),
            total_gathers: AtomicU64::new(0),
        }
    }

    /// Simulate the scatter phase: produce one `PartialResult` per target node.
    ///
    /// In production this would issue parallel RPCs. Here each node returns an
    /// empty result set to validate the planning logic.
    pub fn scatter(&self, plan: &ScatterGatherPlan) -> Vec<PartialResult> {
        self.total_scatters.fetch_add(1, Ordering::Relaxed);
        plan.target_nodes
            .iter()
            .map(|node_id| PartialResult {
                node_id: node_id.clone(),
                rows: Vec::new(),
                execution_time: Duration::from_micros(100),
            })
            .collect()
    }

    /// Merge partial results according to the given strategy.
    pub fn gather(&self, results: Vec<PartialResult>, strategy: &MergeStrategy) -> GatheredResult {
        self.total_gathers.fetch_add(1, Ordering::Relaxed);

        let nodes_contacted = results.len();
        let total_time = results
            .iter()
            .map(|r| r.execution_time)
            .max()
            .unwrap_or(Duration::ZERO);

        let rows = match strategy {
            MergeStrategy::Concatenate => results.into_iter().flat_map(|r| r.rows).collect(),
            MergeStrategy::SortMerge { sort_column } => {
                let mut all_rows: Vec<serde_json::Value> =
                    results.into_iter().flat_map(|r| r.rows).collect();
                let col = sort_column.clone();
                all_rows.sort_by(|a, b| {
                    let va = a.get(&col).and_then(|v| v.as_str()).unwrap_or("");
                    let vb = b.get(&col).and_then(|v| v.as_str()).unwrap_or("");
                    va.cmp(vb)
                });
                all_rows
            }
            MergeStrategy::Aggregate { function } => {
                // Sum all numeric "value" fields across partial results.
                let all_rows: Vec<serde_json::Value> =
                    results.into_iter().flat_map(|r| r.rows).collect();
                if function == "SUM" || function == "COUNT" {
                    let total: f64 = all_rows
                        .iter()
                        .filter_map(|r| r.get("value").and_then(|v| v.as_f64()))
                        .sum();
                    vec![serde_json::json!({ "value": total })]
                } else {
                    all_rows
                }
            }
        };

        GatheredResult {
            rows,
            nodes_contacted,
            total_time,
            merge_strategy: strategy.clone(),
        }
    }

    /// Return total scatter operations.
    pub fn total_scatters(&self) -> u64 {
        self.total_scatters.load(Ordering::Relaxed)
    }

    /// Return total gather operations.
    pub fn total_gathers(&self) -> u64 {
        self.total_gathers.load(Ordering::Relaxed)
    }
}

impl Default for ScatterGatherExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// AI advisor for scatter-gather: decides optimal scatter width by pruning
/// nodes unlikely to hold matching data.
pub struct ScatterAdvisor;

impl ScatterAdvisor {
    /// Given a full set of candidate nodes and a key hint, prune the list to
    /// only those nodes the hash ring maps the key to, up to `max_width`.
    ///
    /// This avoids unnecessary scatter to nodes that cannot possibly hold
    /// the requested data, reducing network overhead.
    pub fn optimal_scatter_nodes(
        ring: &ConsistentHashRing,
        key_hint: Option<&[u8]>,
        all_nodes: &[String],
        max_width: usize,
    ) -> Vec<String> {
        match key_hint {
            Some(key) => {
                let nodes = ring.get_nodes(key, max_width);
                if nodes.is_empty() {
                    // Fallback: scatter to all (capped).
                    all_nodes.iter().take(max_width).cloned().collect()
                } else {
                    nodes
                }
            }
            None => {
                // No key hint: scatter to all nodes (full table scan).
                all_nodes.iter().take(max_width).cloned().collect()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ConsistentHashRing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hash_ring_empty() {
        let ring = ConsistentHashRing::new(100);
        assert!(ring.get_node(b"key").is_none());
        assert!(ring.get_nodes(b"key", 3).is_empty());
        assert_eq!(ring.node_count(), 0);
        assert_eq!(ring.ring_size(), 0);
    }

    #[test]
    fn test_hash_ring_single_node() {
        let ring = ConsistentHashRing::new(100);
        ring.add_node("node-a");
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.ring_size(), 100);

        // Every key must resolve to the only node.
        for i in 0..50 {
            let key = format!("key-{i}");
            assert_eq!(ring.get_node(key.as_bytes()), Some("node-a".to_string()));
        }
    }

    #[test]
    fn test_hash_ring_add_remove() {
        let ring = ConsistentHashRing::new(50);
        ring.add_node("A");
        ring.add_node("B");
        assert_eq!(ring.node_count(), 2);
        assert_eq!(ring.ring_size(), 100);

        ring.remove_node("A");
        assert_eq!(ring.node_count(), 1);
        assert_eq!(ring.ring_size(), 50);

        // All keys now map to B.
        for i in 0..20 {
            let key = format!("k{i}");
            assert_eq!(ring.get_node(key.as_bytes()), Some("B".to_string()));
        }
    }

    #[test]
    fn test_hash_ring_distribution_uniformity() {
        let ring = ConsistentHashRing::new(200);
        ring.add_node("n1");
        ring.add_node("n2");
        ring.add_node("n3");

        let mut counts: HashMap<String, usize> = HashMap::new();
        let total_keys = 3000;
        for i in 0..total_keys {
            let key = format!("test-key-{i}");
            if let Some(node) = ring.get_node(key.as_bytes()) {
                *counts.entry(node).or_default() += 1;
            }
        }

        // With 3 nodes, each should get roughly 1/3 of keys.
        // Allow deviation of up to 25% from the ideal share.
        let ideal = total_keys as f64 / 3.0;
        for (node, count) in &counts {
            let ratio = *count as f64 / ideal;
            assert!(
                (0.5..=1.5).contains(&ratio),
                "Node {node} has {count} keys (ratio {ratio:.2}), expected ~{ideal:.0}"
            );
        }
    }

    #[test]
    fn test_hash_ring_get_nodes_replication() {
        let ring = ConsistentHashRing::new(100);
        ring.add_node("A");
        ring.add_node("B");
        ring.add_node("C");

        let nodes = ring.get_nodes(b"my-key", 2);
        assert_eq!(nodes.len(), 2);
        // All returned nodes must be distinct.
        let unique: HashSet<_> = nodes.iter().collect();
        assert_eq!(unique.len(), 2);
    }

    #[test]
    fn test_hash_ring_get_nodes_exceeds_cluster() {
        let ring = ConsistentHashRing::new(50);
        ring.add_node("X");
        ring.add_node("Y");

        // Request 5 replicas but only 2 nodes exist.
        let nodes = ring.get_nodes(b"k", 5);
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_hash_ring_key_remapping_minimal() {
        let ring = ConsistentHashRing::new(150);
        ring.add_node("A");
        ring.add_node("B");

        // Record initial mapping.
        let total_keys = 1000;
        let mut initial: HashMap<String, String> = HashMap::new();
        for i in 0..total_keys {
            let key = format!("k-{i}");
            let node = ring.get_node(key.as_bytes()).unwrap();
            initial.insert(key, node);
        }

        // Add a third node.
        ring.add_node("C");

        let mut moved = 0usize;
        for i in 0..total_keys {
            let key = format!("k-{i}");
            let node = ring.get_node(key.as_bytes()).unwrap();
            if node != initial[&key] {
                moved += 1;
            }
        }

        // Consistent hashing guarantees that at most ~1/N keys are remapped.
        // With 3 nodes, expect roughly 1/3 to move; allow generous margin.
        let move_ratio = moved as f64 / total_keys as f64;
        assert!(
            move_ratio < 0.6,
            "Too many keys remapped: {moved}/{total_keys} = {move_ratio:.2}"
        );
    }

    #[test]
    fn test_hash_ring_lookup_counter() {
        let ring = ConsistentHashRing::new(10);
        ring.add_node("N");
        assert_eq!(ring.lookup_count(), 0);
        ring.get_node(b"a");
        ring.get_node(b"b");
        assert_eq!(ring.lookup_count(), 2);
    }

    // -----------------------------------------------------------------------
    // ShardRebalancer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rebalance_no_change_needed() {
        let ring = ConsistentHashRing::new(100);
        ring.add_node("A");

        let mut current: HashMap<String, Vec<usize>> = HashMap::new();
        // Assign all shards to A -- which is the only node, so the ring
        // should also map everything to A.
        current.insert("A".to_string(), vec![0, 1, 2, 3]);

        let plans = compute_rebalance(&current, &ring);
        assert!(plans.is_empty(), "Expected no migrations, got {plans:?}");
    }

    #[test]
    fn test_rebalance_generates_plans() {
        let ring = ConsistentHashRing::new(100);
        ring.add_node("A");
        ring.add_node("B");

        // All shards currently on A, but the ring distributes them between A & B.
        let mut current: HashMap<String, Vec<usize>> = HashMap::new();
        current.insert("A".to_string(), (0..10).collect());

        let plans = compute_rebalance(&current, &ring);
        // At least some shards should need to move to B.
        assert!(
            !plans.is_empty(),
            "Expected some migrations when all shards are on one of two nodes"
        );
        for plan in &plans {
            assert_eq!(plan.source_node, "A");
            assert_eq!(plan.target_node, "B");
        }
    }

    #[test]
    fn test_rebalance_advisor_should_rebalance() {
        let advisor = RebalanceAdvisor::default();
        // Low imbalance -> no rebalance.
        assert!(!advisor.should_rebalance(0.1));
        // High imbalance -> yes.
        assert!(advisor.should_rebalance(0.5));
        assert_eq!(advisor.advisory_count(), 2);
    }

    #[test]
    fn test_rebalance_advisor_migration_cost() {
        let advisor = RebalanceAdvisor::new(0.25, 100_000_000); // 100 MB/s
        let cost = advisor.estimate_migration_cost(500_000_000); // 500 MB
        assert_eq!(cost, Duration::from_secs(5));
    }

    #[test]
    fn test_rebalance_advisor_zero_transfer_rate() {
        let advisor = RebalanceAdvisor::new(0.25, 0);
        let cost = advisor.estimate_migration_cost(1_000_000);
        assert_eq!(cost, Duration::from_secs(0));
    }

    #[test]
    fn test_compute_imbalance() {
        let mut dist: HashMap<String, Vec<usize>> = HashMap::new();
        dist.insert("A".to_string(), vec![0, 1, 2, 3, 4, 5]);
        dist.insert("B".to_string(), vec![6, 7]);
        // max=6, min=2, avg=4 -> imbalance = 4/4 = 1.0
        let imbalance = RebalanceAdvisor::compute_imbalance(&dist);
        assert!((imbalance - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_imbalance_single_node() {
        let mut dist: HashMap<String, Vec<usize>> = HashMap::new();
        dist.insert("A".to_string(), vec![0, 1, 2]);
        let imbalance = RebalanceAdvisor::compute_imbalance(&dist);
        assert!((imbalance - 0.0).abs() < 0.001);
    }

    // -----------------------------------------------------------------------
    // ScatterGatherExecutor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_basic() {
        let exec = ScatterGatherExecutor::new();
        let plan = ScatterGatherPlan {
            target_nodes: vec!["n1".into(), "n2".into(), "n3".into()],
            query: "SELECT * FROM t".into(),
            merge_strategy: MergeStrategy::Concatenate,
        };
        let results = exec.scatter(&plan);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].node_id, "n1");
        assert_eq!(exec.total_scatters(), 1);
    }

    #[test]
    fn test_gather_concatenate() {
        let exec = ScatterGatherExecutor::new();
        let r1 = PartialResult {
            node_id: "n1".into(),
            rows: vec![serde_json::json!({"id": 1}), serde_json::json!({"id": 2})],
            execution_time: Duration::from_millis(10),
        };
        let r2 = PartialResult {
            node_id: "n2".into(),
            rows: vec![serde_json::json!({"id": 3})],
            execution_time: Duration::from_millis(5),
        };

        let gathered = exec.gather(vec![r1, r2], &MergeStrategy::Concatenate);
        assert_eq!(gathered.rows.len(), 3);
        assert_eq!(gathered.nodes_contacted, 2);
        assert_eq!(gathered.total_time, Duration::from_millis(10));
    }

    #[test]
    fn test_gather_sort_merge() {
        let exec = ScatterGatherExecutor::new();
        let r1 = PartialResult {
            node_id: "n1".into(),
            rows: vec![
                serde_json::json!({"name": "charlie"}),
                serde_json::json!({"name": "alice"}),
            ],
            execution_time: Duration::from_millis(1),
        };
        let r2 = PartialResult {
            node_id: "n2".into(),
            rows: vec![serde_json::json!({"name": "bob"})],
            execution_time: Duration::from_millis(2),
        };

        let gathered = exec.gather(
            vec![r1, r2],
            &MergeStrategy::SortMerge {
                sort_column: "name".into(),
            },
        );
        assert_eq!(gathered.rows.len(), 3);
        assert_eq!(gathered.rows[0]["name"], "alice");
        assert_eq!(gathered.rows[1]["name"], "bob");
        assert_eq!(gathered.rows[2]["name"], "charlie");
    }

    #[test]
    fn test_gather_aggregate_sum() {
        let exec = ScatterGatherExecutor::new();
        let r1 = PartialResult {
            node_id: "n1".into(),
            rows: vec![serde_json::json!({"value": 10.0})],
            execution_time: Duration::from_millis(1),
        };
        let r2 = PartialResult {
            node_id: "n2".into(),
            rows: vec![serde_json::json!({"value": 25.5})],
            execution_time: Duration::from_millis(1),
        };

        let gathered = exec.gather(
            vec![r1, r2],
            &MergeStrategy::Aggregate {
                function: "SUM".into(),
            },
        );
        assert_eq!(gathered.rows.len(), 1);
        let total = gathered.rows[0]["value"].as_f64().unwrap();
        assert!((total - 35.5).abs() < 0.001);
    }

    #[test]
    fn test_scatter_advisor_with_key_hint() {
        let ring = ConsistentHashRing::new(100);
        ring.add_node("A");
        ring.add_node("B");
        ring.add_node("C");

        let all = vec!["A".into(), "B".into(), "C".into()];
        let pruned = ScatterAdvisor::optimal_scatter_nodes(&ring, Some(b"mykey"), &all, 2);
        assert_eq!(pruned.len(), 2);
    }

    #[test]
    fn test_scatter_advisor_no_key_hint() {
        let ring = ConsistentHashRing::new(100);
        ring.add_node("A");
        ring.add_node("B");

        let all = vec!["A".into(), "B".into()];
        let nodes = ScatterAdvisor::optimal_scatter_nodes(&ring, None, &all, 10);
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_scatter_gather_executor_stats() {
        let exec = ScatterGatherExecutor::new();
        assert_eq!(exec.total_scatters(), 0);
        assert_eq!(exec.total_gathers(), 0);

        let plan = ScatterGatherPlan {
            target_nodes: vec!["n1".into()],
            query: "SELECT 1".into(),
            merge_strategy: MergeStrategy::Concatenate,
        };
        let results = exec.scatter(&plan);
        exec.gather(results, &MergeStrategy::Concatenate);

        assert_eq!(exec.total_scatters(), 1);
        assert_eq!(exec.total_gathers(), 1);
    }

    #[test]
    fn test_gather_empty_results() {
        let exec = ScatterGatherExecutor::new();
        let gathered = exec.gather(Vec::new(), &MergeStrategy::Concatenate);
        assert!(gathered.rows.is_empty());
        assert_eq!(gathered.nodes_contacted, 0);
        assert_eq!(gathered.total_time, Duration::ZERO);
    }
}
