//! Graph query support: edge tables, BFS/DFS traversal, path queries.
//!
//! Implements basic graph operations on top of the relational storage layer.
//! Edges are stored in typed tables with `from_id` and `to_id` columns,
//! using secondary indexes for efficient traversal.

use std::collections::{HashMap, HashSet, VecDeque};

/// A graph edge.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from_id: String,
    pub to_id: String,
    pub edge_type: String,
    pub properties: HashMap<String, String>,
}

/// A node in a graph traversal result.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub depth: usize,
    pub path: Vec<String>,
}

/// Result of a graph traversal.
#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub nodes: Vec<GraphNode>,
    pub edges_traversed: usize,
}

/// Graph query engine operating over edge lists.
pub struct GraphEngine {
    /// Adjacency list: from_id -> [(to_id, edge_type, properties)]
    adjacency: HashMap<String, Vec<Edge>>,
    /// Reverse adjacency: to_id -> [(from_id, edge_type, properties)]
    reverse_adjacency: HashMap<String, Vec<Edge>>,
}

impl GraphEngine {
    /// Create an empty graph engine.
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
        }
    }

    /// Build a graph engine from a list of edges.
    pub fn from_edges(edges: Vec<Edge>) -> Self {
        let mut engine = Self::new();
        for edge in edges {
            engine.add_edge(edge);
        }
        engine
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: Edge) {
        self.reverse_adjacency
            .entry(edge.to_id.clone())
            .or_default()
            .push(edge.clone());
        self.adjacency
            .entry(edge.from_id.clone())
            .or_default()
            .push(edge);
    }

    /// Get outgoing edges from a node.
    pub fn outgoing(&self, node_id: &str) -> &[Edge] {
        self.adjacency.get(node_id).map_or(&[], |v| v.as_slice())
    }

    /// Get incoming edges to a node.
    pub fn incoming(&self, node_id: &str) -> &[Edge] {
        self.reverse_adjacency
            .get(node_id)
            .map_or(&[], |v| v.as_slice())
    }

    /// Breadth-first search from a starting node.
    /// Returns all reachable nodes up to `max_depth`.
    pub fn bfs(&self, start: &str, max_depth: usize) -> TraversalResult {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        let mut edges_traversed = 0;

        visited.insert(start.to_string());
        queue.push_back(GraphNode {
            id: start.to_string(),
            depth: 0,
            path: vec![start.to_string()],
        });

        while let Some(node) = queue.pop_front() {
            if node.depth >= max_depth {
                result.push(node);
                continue;
            }

            let neighbors = self.outgoing(&node.id);
            for edge in neighbors {
                edges_traversed += 1;
                if !visited.contains(&edge.to_id) {
                    visited.insert(edge.to_id.clone());
                    let mut path = node.path.clone();
                    path.push(edge.to_id.clone());
                    queue.push_back(GraphNode {
                        id: edge.to_id.clone(),
                        depth: node.depth + 1,
                        path,
                    });
                }
            }
            result.push(node);
        }

        TraversalResult {
            nodes: result,
            edges_traversed,
        }
    }

    /// Depth-first search from a starting node.
    pub fn dfs(&self, start: &str, max_depth: usize) -> TraversalResult {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut edges_traversed = 0;

        self.dfs_recursive(
            start,
            0,
            max_depth,
            &mut visited,
            &mut result,
            &mut edges_traversed,
            vec![start.to_string()],
        );

        TraversalResult {
            nodes: result,
            edges_traversed,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs_recursive(
        &self,
        node_id: &str,
        depth: usize,
        max_depth: usize,
        visited: &mut HashSet<String>,
        result: &mut Vec<GraphNode>,
        edges_traversed: &mut usize,
        path: Vec<String>,
    ) {
        visited.insert(node_id.to_string());
        result.push(GraphNode {
            id: node_id.to_string(),
            depth,
            path: path.clone(),
        });

        if depth >= max_depth {
            return;
        }

        for edge in self.outgoing(node_id) {
            *edges_traversed += 1;
            if !visited.contains(&edge.to_id) {
                let mut new_path = path.clone();
                new_path.push(edge.to_id.clone());
                self.dfs_recursive(
                    &edge.to_id,
                    depth + 1,
                    max_depth,
                    visited,
                    result,
                    edges_traversed,
                    new_path,
                );
            }
        }
    }

    /// Find shortest path between two nodes using BFS.
    /// Returns None if no path exists.
    pub fn shortest_path(&self, from: &str, to: &str, max_depth: usize) -> Option<Vec<String>> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        visited.insert(from.to_string());
        queue.push_back((from.to_string(), vec![from.to_string()]));

        while let Some((current, path)) = queue.pop_front() {
            if current == to {
                return Some(path);
            }
            if path.len() > max_depth {
                continue;
            }

            for edge in self.outgoing(&current) {
                if !visited.contains(&edge.to_id) {
                    visited.insert(edge.to_id.clone());
                    let mut new_path = path.clone();
                    new_path.push(edge.to_id.clone());
                    queue.push_back((edge.to_id.clone(), new_path));
                }
            }
        }

        None
    }

    /// Count nodes in the graph.
    pub fn node_count(&self) -> usize {
        let mut nodes = HashSet::new();
        for (from, edges) in &self.adjacency {
            nodes.insert(from.as_str());
            for edge in edges {
                nodes.insert(edge.to_id.as_str());
            }
        }
        nodes.len()
    }

    /// Count edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|v| v.len()).sum()
    }

    /// Get all neighbors of a node (both incoming and outgoing).
    pub fn neighbors(&self, node_id: &str) -> Vec<String> {
        let mut result = HashSet::new();
        for edge in self.outgoing(node_id) {
            result.insert(edge.to_id.clone());
        }
        for edge in self.incoming(node_id) {
            result.insert(edge.from_id.clone());
        }
        result.into_iter().collect()
    }
}

impl Default for GraphEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_edge(from: &str, to: &str) -> Edge {
        Edge {
            from_id: from.to_string(),
            to_id: to.to_string(),
            edge_type: "knows".to_string(),
            properties: HashMap::new(),
        }
    }

    fn sample_graph() -> GraphEngine {
        // A -> B -> D
        // A -> C -> D
        // D -> E
        GraphEngine::from_edges(vec![
            make_edge("A", "B"),
            make_edge("A", "C"),
            make_edge("B", "D"),
            make_edge("C", "D"),
            make_edge("D", "E"),
        ])
    }

    #[test]
    fn test_bfs() {
        let graph = sample_graph();
        let result = graph.bfs("A", 3);
        let ids: HashSet<_> = result.nodes.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains("A"));
        assert!(ids.contains("B"));
        assert!(ids.contains("C"));
        assert!(ids.contains("D"));
        assert!(ids.contains("E"));
    }

    #[test]
    fn test_bfs_depth_limit() {
        let graph = sample_graph();
        let result = graph.bfs("A", 1);
        let ids: HashSet<_> = result.nodes.iter().map(|n| n.id.as_str()).collect();
        assert!(ids.contains("A"));
        assert!(ids.contains("B"));
        assert!(ids.contains("C"));
        assert!(!ids.contains("E")); // Too deep
    }

    #[test]
    fn test_dfs() {
        let graph = sample_graph();
        let result = graph.dfs("A", 10);
        let ids: HashSet<_> = result.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_shortest_path() {
        let graph = sample_graph();

        let path = graph.shortest_path("A", "E", 10).unwrap();
        assert_eq!(path.first().unwrap(), "A");
        assert_eq!(path.last().unwrap(), "E");
        assert_eq!(path.len(), 4); // A -> B -> D -> E or A -> C -> D -> E
    }

    #[test]
    fn test_no_path() {
        let graph = sample_graph();
        assert!(graph.shortest_path("E", "A", 10).is_none());
    }

    #[test]
    fn test_node_and_edge_count() {
        let graph = sample_graph();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 5);
    }

    #[test]
    fn test_neighbors() {
        let graph = sample_graph();
        let neighbors = graph.neighbors("D");
        assert!(neighbors.contains(&"B".to_string()) || neighbors.contains(&"C".to_string()));
        assert!(neighbors.contains(&"E".to_string()));
    }

    #[test]
    fn test_incoming_outgoing() {
        let graph = sample_graph();
        assert_eq!(graph.outgoing("A").len(), 2);
        assert_eq!(graph.incoming("D").len(), 2);
        assert_eq!(graph.outgoing("E").len(), 0);
    }
}
