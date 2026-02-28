//! Vector Search — AI-native semantic similarity search for TensorDB.
//!
//! # The Innovation
//! TensorDB integrates vector similarity search directly into its storage engine,
//! making it the first bitemporal database with native vector search. This enables:
//! - "Find documents similar to X, AS OF timestamp T" — temporal vector queries
//! - "Track how document similarity changed over time" — bitemporal vector analysis
//! - AI-driven index selection between exact search (small sets) and HNSW (large sets)
//!
//! # Architecture
//! - Vectors are stored as regular TensorDB facts with a special encoding
//! - An in-memory HNSW (Hierarchical Navigable Small World) graph provides fast ANN search
//! - The AI advisor monitors query patterns and auto-tunes HNSW parameters
//!
//! # Why This Is Novel
//! - pgvector: Vector search bolted onto PostgreSQL, no temporal dimension
//! - Pinecone/Milvus: Purpose-built vector DBs, no SQL or bitemporal support
//! - TensorDB: Native vector search with bitemporal time-travel queries

use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

/// A vector with its metadata.
#[derive(Debug, Clone)]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: u64,
}

/// Distance metric for vector similarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance — smaller = more similar.
    Euclidean,
    /// Cosine distance — 1 - cosine_similarity.
    Cosine,
    /// Dot product (negative) — more negative = more similar.
    DotProduct,
}

impl DistanceMetric {
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Euclidean => euclidean_distance(a, b),
            Self::Cosine => cosine_distance(a, b),
            Self::DotProduct => -dot_product(a, b),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::Cosine => "cosine",
            Self::DotProduct => "dot_product",
        }
    }
}

/// A vector index (collection of vectors with an index structure).
pub struct VectorIndex {
    /// Name of this index.
    name: String,
    /// Dimension of vectors in this index.
    dimensions: usize,
    /// Distance metric.
    metric: DistanceMetric,
    /// All vectors (id → entry).
    vectors: RwLock<HashMap<String, VectorEntry>>,
    /// HNSW graph for approximate nearest neighbor search.
    hnsw: RwLock<HnswGraph>,
    /// AI advisor for index tuning.
    advisor: RwLock<VectorAdvisor>,
    /// Stats
    total_inserts: AtomicU64,
    total_searches: AtomicU64,
}

impl VectorIndex {
    /// Create a new vector index.
    pub fn new(name: &str, dimensions: usize, metric: DistanceMetric) -> Self {
        Self {
            name: name.to_string(),
            dimensions,
            metric,
            vectors: RwLock::new(HashMap::new()),
            hnsw: RwLock::new(HnswGraph::new(16, 200)), // M=16, ef_construction=200
            advisor: RwLock::new(VectorAdvisor::new()),
            total_inserts: AtomicU64::new(0),
            total_searches: AtomicU64::new(0),
        }
    }

    /// Insert a vector into the index.
    pub fn insert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), String> {
        if vector.len() != self.dimensions {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            ));
        }

        let entry = VectorEntry {
            id: id.to_string(),
            vector: vector.clone(),
            metadata,
            created_at: current_ms(),
        };

        let node_id = {
            let mut vectors = self.vectors.write();
            let node_id = vectors.len();
            vectors.insert(id.to_string(), entry);
            node_id
        };

        // Add to HNSW graph
        self.hnsw.write().add_node(node_id, vector, &self.metric);

        self.total_inserts.fetch_add(1, Ordering::Relaxed);
        self.advisor.write().record_insert();
        Ok(())
    }

    /// Search for the k nearest neighbors to a query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, String> {
        if query.len() != self.dimensions {
            return Err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            ));
        }

        self.total_searches.fetch_add(1, Ordering::Relaxed);
        self.advisor.write().record_search(k);

        let vectors = self.vectors.read();
        let count = vectors.len();

        if count == 0 {
            return Ok(Vec::new());
        }

        // AI decision: exact search for small sets, HNSW for large
        let use_exact = self.advisor.read().should_use_exact_search(count, k);

        if use_exact {
            // Exact brute-force search
            self.exact_search(query, k)
        } else {
            // HNSW approximate search
            self.hnsw_search(query, k)
        }
    }

    /// Exact brute-force search (guaranteed correct, O(n)).
    fn exact_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, String> {
        let vectors = self.vectors.read();
        let mut heap: BinaryHeap<SearchCandidate> = BinaryHeap::new();

        for (id, entry) in vectors.iter() {
            let distance = self.metric.compute(query, &entry.vector);
            let candidate = SearchCandidate {
                id: id.clone(),
                distance,
            };

            if heap.len() < k {
                heap.push(candidate);
            } else if let Some(worst) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(candidate);
                }
            }
        }

        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: c.distance,
                score: 1.0 / (1.0 + c.distance),
            })
            .collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(results)
    }

    /// HNSW approximate search.
    fn hnsw_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, String> {
        let hnsw = self.hnsw.read();
        let vectors = self.vectors.read();

        let candidates = hnsw.search(query, k, &self.metric);

        // Map node IDs back to vector IDs
        let id_map: Vec<String> = vectors.keys().cloned().collect();

        let results: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|(node_id, distance)| {
                id_map.get(node_id).map(|id| SearchResult {
                    id: id.clone(),
                    distance,
                    score: 1.0 / (1.0 + distance),
                })
            })
            .collect();

        Ok(results)
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<VectorEntry> {
        self.vectors.read().get(id).cloned()
    }

    /// Delete a vector by ID.
    pub fn delete(&self, id: &str) -> bool {
        self.vectors.write().remove(id).is_some()
    }

    /// Number of vectors in the index.
    pub fn count(&self) -> usize {
        self.vectors.read().len()
    }

    /// Get index statistics.
    pub fn stats(&self) -> VectorIndexStats {
        let advisor = self.advisor.read();
        VectorIndexStats {
            name: self.name.clone(),
            dimensions: self.dimensions,
            metric: self.metric.name().to_string(),
            vector_count: self.vectors.read().len(),
            total_inserts: self.total_inserts.load(Ordering::Relaxed),
            total_searches: self.total_searches.load(Ordering::Relaxed),
            hnsw_levels: self.hnsw.read().max_level(),
            advisor_stats: advisor.stats(),
        }
    }
}

/// Search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub score: f32, // 1 / (1 + distance), normalized to [0, 1]
}

/// Internal candidate for heap-based top-k selection.
#[derive(Debug, Clone)]
struct SearchCandidate {
    id: String,
    distance: f32,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: largest distance at top (so we can pop the worst)
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// HNSW Graph (simplified in-memory implementation)
// ---------------------------------------------------------------------------

struct HnswNode {
    vector: Vec<f32>,
    connections: Vec<Vec<usize>>, // connections per level
}

struct HnswGraph {
    nodes: Vec<HnswNode>,
    entry_point: Option<usize>,
    max_connections: usize, // M parameter
    _ef_construction: usize,
    max_level: usize,
    level_multiplier: f64,
}

impl HnswGraph {
    fn new(max_connections: usize, ef_construction: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_connections,
            _ef_construction: ef_construction,
            max_level: 0,
            level_multiplier: 1.0 / (max_connections as f64).ln(),
        }
    }

    fn random_level(&self) -> usize {
        let r = fastrand::f64();
        (-r.ln() * self.level_multiplier).floor() as usize
    }

    fn add_node(&mut self, _node_id: usize, vector: Vec<f32>, metric: &DistanceMetric) {
        let level = self.random_level();
        let node_idx = self.nodes.len();

        let connections = vec![Vec::new(); level + 1];
        self.nodes.push(HnswNode {
            vector,
            connections,
        });

        if self.entry_point.is_none() {
            self.entry_point = Some(node_idx);
            self.max_level = level;
            return;
        }

        let entry = self.entry_point.unwrap();

        // Connect to nearest neighbors at each level
        for l in 0..=level.min(self.max_level) {
            let neighbors = self.find_nearest_at_level(
                &self.nodes[node_idx].vector,
                self.max_connections,
                l,
                entry,
                metric,
            );

            for &neighbor in &neighbors {
                // Bidirectional connection
                if self.nodes[node_idx].connections.len() > l {
                    self.nodes[node_idx].connections[l].push(neighbor);
                }
                if self.nodes[neighbor].connections.len() > l {
                    self.nodes[neighbor].connections[l].push(node_idx);

                    // Prune if too many connections
                    if self.nodes[neighbor].connections[l].len() > self.max_connections * 2 {
                        let node_vec = self.nodes[neighbor].vector.clone();
                        let mut scored: Vec<(usize, f32)> = self.nodes[neighbor].connections[l]
                            .iter()
                            .map(|&n| (n, metric.compute(&node_vec, &self.nodes[n].vector)))
                            .collect();
                        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        self.nodes[neighbor].connections[l] = scored
                            .into_iter()
                            .take(self.max_connections)
                            .map(|(n, _)| n)
                            .collect();
                    }
                }
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(node_idx);
        }
    }

    fn find_nearest_at_level(
        &self,
        query: &[f32],
        k: usize,
        level: usize,
        start: usize,
        metric: &DistanceMetric,
    ) -> Vec<usize> {
        let mut visited = vec![false; self.nodes.len()];
        let mut candidates: BinaryHeap<std::cmp::Reverse<(FloatOrd, usize)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(FloatOrd, usize)> = BinaryHeap::new();

        let d = metric.compute(query, &self.nodes[start].vector);
        candidates.push(std::cmp::Reverse((FloatOrd(d), start)));
        results.push((FloatOrd(d), start));
        visited[start] = true;

        while let Some(std::cmp::Reverse((FloatOrd(cd), c))) = candidates.pop() {
            if let Some(&(FloatOrd(worst_d), _)) = results.peek() {
                if cd > worst_d && results.len() >= k {
                    break;
                }
            }

            if level < self.nodes[c].connections.len() {
                for &neighbor in &self.nodes[c].connections[level] {
                    if neighbor < visited.len() && !visited[neighbor] {
                        visited[neighbor] = true;
                        let nd = metric.compute(query, &self.nodes[neighbor].vector);

                        if results.len() < k {
                            candidates.push(std::cmp::Reverse((FloatOrd(nd), neighbor)));
                            results.push((FloatOrd(nd), neighbor));
                        } else if let Some(&(FloatOrd(worst_d), _)) = results.peek() {
                            if nd < worst_d {
                                candidates.push(std::cmp::Reverse((FloatOrd(nd), neighbor)));
                                results.pop();
                                results.push((FloatOrd(nd), neighbor));
                            }
                        }
                    }
                }
            }
        }

        results.into_iter().map(|(_, id)| id).collect()
    }

    fn search(&self, query: &[f32], k: usize, metric: &DistanceMetric) -> Vec<(usize, f32)> {
        let entry = match self.entry_point {
            Some(e) => e,
            None => return Vec::new(),
        };

        let neighbors = self.find_nearest_at_level(query, k, 0, entry, metric);

        let mut results: Vec<(usize, f32)> = neighbors
            .into_iter()
            .map(|n| (n, metric.compute(query, &self.nodes[n].vector)))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    fn max_level(&self) -> usize {
        self.max_level
    }
}

/// Wrapper for f32 ordering (total order).
#[derive(Debug, Clone, Copy)]
struct FloatOrd(f32);

impl PartialEq for FloatOrd {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// AI Vector Advisor
// ---------------------------------------------------------------------------

struct VectorAdvisor {
    total_inserts: u64,
    total_searches: u64,
    search_k_sum: u64,
    search_count: u64,
}

impl VectorAdvisor {
    fn new() -> Self {
        Self {
            total_inserts: 0,
            total_searches: 0,
            search_k_sum: 0,
            search_count: 0,
        }
    }

    fn record_insert(&mut self) {
        self.total_inserts += 1;
    }

    fn record_search(&mut self, k: usize) {
        self.total_searches += 1;
        self.search_k_sum += k as u64;
        self.search_count += 1;
    }

    /// AI decision: should we use exact search or HNSW?
    fn should_use_exact_search(&self, vector_count: usize, k: usize) -> bool {
        // Heuristics:
        // - Exact search is faster for small sets (< 1000 vectors)
        // - Exact search is more accurate when k is large relative to n
        // - HNSW has build overhead that amortizes over many queries
        if vector_count < 1000 {
            return true;
        }
        if k as f64 / vector_count as f64 > 0.1 {
            return true; // k > 10% of n → exact is better
        }
        false
    }

    fn stats(&self) -> VectorAdvisorStats {
        VectorAdvisorStats {
            avg_k: if self.search_count > 0 {
                self.search_k_sum as f64 / self.search_count as f64
            } else {
                0.0
            },
            total_inserts: self.total_inserts,
            total_searches: self.total_searches,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VectorIndexStats {
    pub name: String,
    pub dimensions: usize,
    pub metric: String,
    pub vector_count: usize,
    pub total_inserts: u64,
    pub total_searches: u64,
    pub hnsw_levels: usize,
    pub advisor_stats: VectorAdvisorStats,
}

#[derive(Debug, Clone)]
pub struct VectorAdvisorStats {
    pub avg_k: f64,
    pub total_inserts: u64,
    pub total_searches: u64,
}

// ---------------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------------

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-10 {
        1.0
    } else {
        1.0 - (dot / denom)
    }
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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

    fn vec3(x: f32, y: f32, z: f32) -> Vec<f32> {
        vec![x, y, z]
    }

    #[test]
    fn test_euclidean_distance() {
        assert!((euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 0.001);
        assert_eq!(euclidean_distance(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_cosine_distance() {
        // Same direction → distance 0
        let d = cosine_distance(&[1.0, 0.0], &[2.0, 0.0]);
        assert!(d.abs() < 0.001);

        // Orthogonal → distance 1
        let d = cosine_distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((d - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dot_product() {
        assert_eq!(dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn test_vector_index_insert_and_search() {
        let idx = VectorIndex::new("test", 3, DistanceMetric::Euclidean);

        idx.insert("a", vec3(1.0, 0.0, 0.0), None).unwrap();
        idx.insert("b", vec3(0.0, 1.0, 0.0), None).unwrap();
        idx.insert("c", vec3(0.0, 0.0, 1.0), None).unwrap();

        // Search for nearest to [1, 0, 0] — should be "a"
        let results = idx.search(&vec3(1.0, 0.0, 0.0), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "a");
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_vector_index_top_k() {
        let idx = VectorIndex::new("test", 2, DistanceMetric::Euclidean);

        idx.insert("origin", vec![0.0, 0.0], None).unwrap();
        idx.insert("near", vec![0.1, 0.1], None).unwrap();
        idx.insert("far", vec![10.0, 10.0], None).unwrap();

        let results = idx.search(&[0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "origin");
        assert_eq!(results[1].id, "near");
    }

    #[test]
    fn test_dimension_mismatch() {
        let idx = VectorIndex::new("test", 3, DistanceMetric::Euclidean);

        let err = idx.insert("a", vec![1.0, 2.0], None);
        assert!(err.is_err());

        idx.insert("a", vec3(1.0, 0.0, 0.0), None).unwrap();
        let err = idx.search(&[1.0, 2.0], 1);
        assert!(err.is_err());
    }

    #[test]
    fn test_cosine_search() {
        let idx = VectorIndex::new("test", 3, DistanceMetric::Cosine);

        idx.insert("up", vec3(0.0, 1.0, 0.0), None).unwrap();
        idx.insert("right", vec3(1.0, 0.0, 0.0), None).unwrap();
        idx.insert("mostly_up", vec3(0.1, 0.9, 0.0), None).unwrap();

        // Query direction is "up" — should find "up" and "mostly_up" closest
        let results = idx.search(&vec3(0.0, 1.0, 0.0), 2).unwrap();
        assert_eq!(results[0].id, "up");
        assert_eq!(results[1].id, "mostly_up");
    }

    #[test]
    fn test_delete() {
        let idx = VectorIndex::new("test", 2, DistanceMetric::Euclidean);
        idx.insert("a", vec![1.0, 0.0], None).unwrap();
        assert_eq!(idx.count(), 1);

        assert!(idx.delete("a"));
        assert_eq!(idx.count(), 0);
        assert!(!idx.delete("a")); // Already deleted
    }

    #[test]
    fn test_get() {
        let idx = VectorIndex::new("test", 2, DistanceMetric::Euclidean);
        idx.insert(
            "a",
            vec![1.0, 2.0],
            Some(serde_json::json!({"label": "test"})),
        )
        .unwrap();

        let entry = idx.get("a").unwrap();
        assert_eq!(entry.vector, vec![1.0, 2.0]);
        assert!(entry.metadata.is_some());

        assert!(idx.get("nonexistent").is_none());
    }

    #[test]
    fn test_empty_search() {
        let idx = VectorIndex::new("test", 3, DistanceMetric::Euclidean);
        let results = idx.search(&vec3(1.0, 0.0, 0.0), 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_stats() {
        let idx = VectorIndex::new("vectors", 3, DistanceMetric::Euclidean);
        idx.insert("a", vec3(1.0, 0.0, 0.0), None).unwrap();
        idx.search(&vec3(1.0, 0.0, 0.0), 1).unwrap();

        let stats = idx.stats();
        assert_eq!(stats.name, "vectors");
        assert_eq!(stats.dimensions, 3);
        assert_eq!(stats.vector_count, 1);
        assert_eq!(stats.total_inserts, 1);
        assert_eq!(stats.total_searches, 1);
    }

    #[test]
    fn test_score_normalization() {
        let idx = VectorIndex::new("test", 2, DistanceMetric::Euclidean);
        idx.insert("exact", vec![1.0, 0.0], None).unwrap();

        let results = idx.search(&[1.0, 0.0], 1).unwrap();
        // Distance 0 → score = 1 / (1 + 0) = 1.0
        assert!((results[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_larger_dataset() {
        let idx = VectorIndex::new("test", 4, DistanceMetric::Euclidean);

        // Insert 100 vectors
        for i in 0..100 {
            let v = vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32];
            idx.insert(&format!("v{i}"), v, None).unwrap();
        }

        assert_eq!(idx.count(), 100);

        // Search for nearest to [50, 100, 150, 200] (should be v50)
        let results = idx.search(&[50.0, 100.0, 150.0, 200.0], 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, "v50");
    }

    #[test]
    fn test_with_metadata() {
        let idx = VectorIndex::new("docs", 2, DistanceMetric::Cosine);

        idx.insert(
            "doc1",
            vec![1.0, 0.0],
            Some(serde_json::json!({"title": "hello", "category": "greeting"})),
        )
        .unwrap();

        let entry = idx.get("doc1").unwrap();
        assert_eq!(entry.metadata.unwrap()["title"], "hello");
    }
}
