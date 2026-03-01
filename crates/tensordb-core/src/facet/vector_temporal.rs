//! Temporal HNSW: versioned vector nodes with time-filtered search.
//! Enables `FOR SYSTEM_TIME AS OF` queries on vector indexes.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::facet::vector_search::DistanceMetric;

// ── Temporal HNSW Graph ─────────────────────────────────────────────────────

/// A versioned node in the temporal HNSW graph.
#[derive(Debug, Clone)]
pub struct TemporalHnswNode {
    pub pk: String,
    pub vector: Vec<f32>,
    /// Connections at each level.
    pub connections: Vec<Vec<usize>>,
    /// Epoch when this node was inserted (commit_ts).
    pub valid_from: u64,
    /// Epoch when this node was deleted (u64::MAX = still alive).
    pub valid_to: u64,
}

/// HNSW graph with temporal awareness.
#[derive(Debug)]
pub struct TemporalHnswGraph {
    pub nodes: Vec<TemporalHnswNode>,
    pub entry_point: Option<usize>,
    pub max_connections: usize,
    pub ef_construction: usize,
    pub max_level: usize,
    pub metric: DistanceMetric,
}

impl TemporalHnswGraph {
    pub fn new(max_connections: usize, ef_construction: usize, metric: DistanceMetric) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_connections,
            ef_construction,
            max_level: 0,
            metric,
        }
    }

    /// Insert a new vector with temporal bounds.
    pub fn insert(&mut self, pk: String, vector: Vec<f32>, valid_from: u64) -> usize {
        let level = self.random_level();
        let node_id = self.nodes.len();

        let node = TemporalHnswNode {
            pk,
            vector,
            connections: vec![Vec::new(); level + 1],
            valid_from,
            valid_to: u64::MAX,
        };
        self.nodes.push(node);

        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.max_level = level;
            return node_id;
        }

        let ep = self.entry_point.unwrap();

        // Search from top to insertion level
        let mut current = ep;
        for lev in (level + 1..=self.max_level).rev() {
            current = self.greedy_closest(current, &self.nodes[node_id].vector, lev, None);
        }

        // Insert at each level from level down to 0
        for lev in (0..=level.min(self.max_level)).rev() {
            let neighbors = self.search_level(
                &self.nodes[node_id].vector,
                current,
                self.ef_construction,
                lev,
                None,
            );

            let max_conn = if lev == 0 {
                self.max_connections * 2
            } else {
                self.max_connections
            };

            let selected: Vec<usize> = neighbors.iter().take(max_conn).map(|e| e.id).collect();

            // Connect node to selected neighbors
            if lev < self.nodes[node_id].connections.len() {
                self.nodes[node_id].connections[lev] = selected.clone();
            }

            // Connect neighbors back to node
            for &neighbor_id in &selected {
                if lev < self.nodes[neighbor_id].connections.len() {
                    self.nodes[neighbor_id].connections[lev].push(node_id);
                    // Prune if over capacity
                    if self.nodes[neighbor_id].connections[lev].len() > max_conn {
                        let query = self.nodes[neighbor_id].vector.clone();
                        let mut conn_dists: Vec<(usize, f32)> = self.nodes[neighbor_id].connections
                            [lev]
                            .iter()
                            .map(|&id| (id, self.metric.compute(&query, &self.nodes[id].vector)))
                            .collect();
                        conn_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        self.nodes[neighbor_id].connections[lev] = conn_dists
                            .iter()
                            .take(max_conn)
                            .map(|&(id, _)| id)
                            .collect();
                    }
                }
            }

            if !selected.is_empty() {
                current = selected[0];
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(node_id);
        }

        node_id
    }

    /// Mark a node as deleted at the given epoch.
    pub fn delete(&mut self, pk: &str, valid_to: u64) -> bool {
        for node in &mut self.nodes {
            if node.pk == pk && node.valid_to == u64::MAX {
                node.valid_to = valid_to;
                return true;
            }
        }
        false
    }

    /// Search for k nearest neighbors, optionally filtering by temporal point.
    /// If `as_of` is `Some(ts)`, only nodes where `valid_from <= ts < valid_to` are returned.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        as_of: Option<u64>,
    ) -> Vec<TemporalSearchResult> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        let ep = self.entry_point.unwrap();

        // Descend from top level to level 1
        let mut current = ep;
        for lev in (1..=self.max_level).rev() {
            current = self.greedy_closest(current, query, lev, as_of);
        }

        // Search at level 0 with ef candidates
        let candidates = self.search_level(query, current, ef.max(k), 0, as_of);

        candidates
            .into_iter()
            .take(k)
            .map(|e| TemporalSearchResult {
                pk: self.nodes[e.id].pk.clone(),
                distance: e.distance,
                valid_from: self.nodes[e.id].valid_from,
                valid_to: self.nodes[e.id].valid_to,
            })
            .collect()
    }

    /// Check if a node is visible at the given temporal point.
    fn is_visible(&self, node_id: usize, as_of: Option<u64>) -> bool {
        match as_of {
            None => self.nodes[node_id].valid_to == u64::MAX,
            Some(ts) => self.nodes[node_id].valid_from <= ts && ts < self.nodes[node_id].valid_to,
        }
    }

    fn greedy_closest(
        &self,
        start: usize,
        query: &[f32],
        level: usize,
        as_of: Option<u64>,
    ) -> usize {
        let mut current = start;
        let mut current_dist = self.metric.compute(query, &self.nodes[current].vector);

        loop {
            let mut changed = false;
            if level < self.nodes[current].connections.len() {
                for &neighbor in &self.nodes[current].connections[level] {
                    if !self.is_visible(neighbor, as_of) {
                        continue;
                    }
                    let dist = self.metric.compute(query, &self.nodes[neighbor].vector);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    fn search_level(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
        as_of: Option<u64>,
    ) -> Vec<ScoredNode> {
        let mut visited = vec![false; self.nodes.len()];
        let mut candidates: BinaryHeap<std::cmp::Reverse<ScoredNode>> = BinaryHeap::new();
        let mut results: BinaryHeap<ScoredNode> = BinaryHeap::new();

        let dist = self.metric.compute(query, &self.nodes[entry].vector);
        visited[entry] = true;

        candidates.push(std::cmp::Reverse(ScoredNode {
            id: entry,
            distance: dist,
        }));
        if self.is_visible(entry, as_of) {
            results.push(ScoredNode {
                id: entry,
                distance: dist,
            });
        }

        while let Some(std::cmp::Reverse(current)) = candidates.pop() {
            if let Some(worst) = results.peek() {
                if results.len() >= ef && current.distance > worst.distance {
                    break;
                }
            }

            if level < self.nodes[current.id].connections.len() {
                for &neighbor in &self.nodes[current.id].connections[level] {
                    if visited[neighbor] {
                        continue;
                    }
                    visited[neighbor] = true;

                    let ndist = self.metric.compute(query, &self.nodes[neighbor].vector);

                    let should_add =
                        results.len() < ef || results.peek().is_none_or(|w| ndist < w.distance);

                    if should_add {
                        candidates.push(std::cmp::Reverse(ScoredNode {
                            id: neighbor,
                            distance: ndist,
                        }));
                        if self.is_visible(neighbor, as_of) {
                            results.push(ScoredNode {
                                id: neighbor,
                                distance: ndist,
                            });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut sorted: Vec<ScoredNode> = results.into_iter().collect();
        sorted.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        sorted
    }

    fn random_level(&self) -> usize {
        let mut level = 0;
        let ml = 1.0 / (self.max_connections as f64).ln();
        while fastrand::f64() < (-fastrand::f64().ln() * ml).exp().recip() && level < 16 {
            level += 1;
        }
        level
    }
}

// ── Supporting types ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TemporalSearchResult {
    pub pk: String,
    pub distance: f32,
    pub valid_from: u64,
    pub valid_to: u64,
}

#[derive(Debug, Clone)]
struct ScoredNode {
    id: usize,
    distance: f32,
}

impl PartialEq for ScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for ScoredNode {}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_hnsw_basic() {
        let mut graph = TemporalHnswGraph::new(16, 200, DistanceMetric::Euclidean);

        // Insert vectors at different epochs
        graph.insert("pk1".to_string(), vec![1.0, 0.0, 0.0], 100);
        graph.insert("pk2".to_string(), vec![0.0, 1.0, 0.0], 200);
        graph.insert("pk3".to_string(), vec![0.0, 0.0, 1.0], 300);

        // Search without temporal filter (current state)
        let results = graph.search(&[1.0, 0.0, 0.0], 2, 50, None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].pk, "pk1");
    }

    #[test]
    fn test_temporal_hnsw_time_filter() {
        let mut graph = TemporalHnswGraph::new(16, 200, DistanceMetric::Euclidean);

        graph.insert("pk1".to_string(), vec![1.0, 0.0], 100);
        graph.insert("pk2".to_string(), vec![0.9, 0.1], 200);
        graph.insert("pk3".to_string(), vec![0.8, 0.2], 300);

        // Search as of epoch 150: only pk1 visible
        let results = graph.search(&[1.0, 0.0], 3, 50, Some(150));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk, "pk1");

        // Search as of epoch 250: pk1 and pk2 visible
        let results = graph.search(&[1.0, 0.0], 3, 50, Some(250));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_temporal_hnsw_delete() {
        let mut graph = TemporalHnswGraph::new(16, 200, DistanceMetric::Euclidean);

        graph.insert("pk1".to_string(), vec![1.0, 0.0], 100);
        graph.insert("pk2".to_string(), vec![0.0, 1.0], 100);

        // Delete pk1 at epoch 500
        assert!(graph.delete("pk1", 500));

        // Current state: only pk2
        let results = graph.search(&[1.0, 0.0], 2, 50, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].pk, "pk2");

        // As of epoch 200: both visible
        let results = graph.search(&[1.0, 0.0], 2, 50, Some(200));
        assert_eq!(results.len(), 2);
    }
}
