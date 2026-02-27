//! Vector Search SQL Surface — SQL-accessible vector operations for SpectraDB.
//!
//! # The Innovation
//! This module bridges SpectraDB's native vector search engine with its SQL surface,
//! enabling vector similarity queries to be expressed as standard SQL statements.
//! Unlike pgvector (which bolts vector ops onto PostgreSQL) or standalone vector DBs
//! (Pinecone, Milvus), SpectraDB unifies vector search, bitemporal time-travel, and
//! SQL in a single engine. The result: queries like
//!
//! ```sql
//! SELECT * FROM vector_search('embeddings', '[1.0, 0.5, 0.3]', 10)
//!     AS OF 1000 VALID AT 2000;
//! ```
//!
//! combine semantic similarity with bitemporal filtering in one pass.
//!
//! # Components
//! - **`VectorStore`** — manages multiple named vector indexes (create, drop, insert,
//!   search, list, stats) behind a `RwLock<HashMap>` for concurrent access.
//! - **`VectorSqlBridge`** — bidirectional conversion between SQL string literals and
//!   vector values, plus distance-metric parsing.
//! - **Vector SQL functions** — `vector_distance`, `cosine_similarity`, `vector_norm`,
//!   `vector_dims` — evaluated inline in the SQL expression evaluator.
//! - **`VectorQueryAdvisor`** — AI-driven advisor that monitors search patterns and
//!   recommends index parameter tuning (ef_search, exact thresholds).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::facet::vector_search::{
    DistanceMetric, SearchResult, VectorEntry, VectorIndex, VectorIndexStats,
};

// ---------------------------------------------------------------------------
// VectorStore — multi-index manager
// ---------------------------------------------------------------------------

/// Manages multiple named vector indexes, providing a unified entry point
/// for SQL-driven vector operations.
pub struct VectorStore {
    indexes: RwLock<HashMap<String, VectorIndex>>,
    total_ops: AtomicU64,
}

impl VectorStore {
    /// Create a new, empty `VectorStore`.
    pub fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
            total_ops: AtomicU64::new(0),
        }
    }

    /// Create a new vector index with the given name, dimensionality, and metric.
    pub fn create_index(
        &self,
        name: &str,
        dimensions: usize,
        metric: DistanceMetric,
    ) -> Result<(), String> {
        let mut indexes = self.indexes.write();
        if indexes.contains_key(name) {
            return Err(format!("vector index '{name}' already exists"));
        }
        indexes.insert(name.to_string(), VectorIndex::new(name, dimensions, metric));
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Drop (remove) a vector index by name.
    pub fn drop_index(&self, name: &str) -> Result<(), String> {
        let mut indexes = self.indexes.write();
        if indexes.remove(name).is_none() {
            return Err(format!("vector index '{name}' does not exist"));
        }
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Insert a vector into the named index.
    pub fn insert(
        &self,
        index_name: &str,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) -> Result<(), String> {
        let indexes = self.indexes.read();
        let idx = indexes
            .get(index_name)
            .ok_or_else(|| format!("vector index '{index_name}' does not exist"))?;
        idx.insert(id, vector, metadata)?;
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Search the named index for the `k` nearest neighbors to `query_vector`.
    pub fn search(
        &self,
        index_name: &str,
        query_vector: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, String> {
        let indexes = self.indexes.read();
        let idx = indexes
            .get(index_name)
            .ok_or_else(|| format!("vector index '{index_name}' does not exist"))?;
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        idx.search(query_vector, k)
    }

    /// Retrieve a single vector entry by id from the named index.
    pub fn get(&self, index_name: &str, id: &str) -> Result<Option<VectorEntry>, String> {
        let indexes = self.indexes.read();
        let idx = indexes
            .get(index_name)
            .ok_or_else(|| format!("vector index '{index_name}' does not exist"))?;
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        Ok(idx.get(id))
    }

    /// List metadata for all indexes.
    pub fn list_indexes(&self) -> Vec<VectorIndexInfo> {
        let indexes = self.indexes.read();
        indexes
            .values()
            .map(|idx| {
                let stats = idx.stats();
                VectorIndexInfo {
                    name: stats.name,
                    dimensions: stats.dimensions,
                    metric: stats.metric,
                    vector_count: stats.vector_count,
                }
            })
            .collect()
    }

    /// Get detailed statistics for a specific index.
    pub fn stats(&self, index_name: &str) -> Option<VectorIndexStats> {
        let indexes = self.indexes.read();
        indexes.get(index_name).map(|idx| idx.stats())
    }

    /// Total operations performed across all indexes.
    pub fn total_ops(&self) -> u64 {
        self.total_ops.load(Ordering::Relaxed)
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary information about a vector index (returned by `list_indexes`).
#[derive(Debug, Clone)]
pub struct VectorIndexInfo {
    pub name: String,
    pub dimensions: usize,
    pub metric: String,
    pub vector_count: usize,
}

// ---------------------------------------------------------------------------
// VectorSqlBridge — SQL <-> vector conversions
// ---------------------------------------------------------------------------

/// Bidirectional conversion between SQL string representations and vector values.
pub struct VectorSqlBridge;

impl VectorSqlBridge {
    /// Parse a vector literal string like `"[1.0, 2.0, 3.0]"` into a `Vec<f32>`.
    ///
    /// Accepts both bracket and bare formats:
    /// - `[1.0, 2.0, 3.0]`
    /// - `1.0, 2.0, 3.0`
    pub fn parse_vector_literal(s: &str) -> Result<Vec<f32>, String> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err("empty vector literal".to_string());
        }

        // Strip optional brackets
        let inner = if trimmed.starts_with('[') && trimmed.ends_with(']') {
            &trimmed[1..trimmed.len() - 1]
        } else {
            trimmed
        };

        let inner = inner.trim();
        if inner.is_empty() {
            return Err("empty vector literal".to_string());
        }

        inner
            .split(',')
            .map(|part| {
                let t = part.trim();
                t.parse::<f32>()
                    .map_err(|e| format!("invalid vector element '{t}': {e}"))
            })
            .collect()
    }

    /// Format a vector as a bracket-delimited string: `[1.0, 2.0, 3.0]`.
    pub fn format_vector(v: &[f32]) -> String {
        let elems: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
        format!("[{}]", elems.join(", "))
    }

    /// Parse a distance metric name (case-insensitive) into a `DistanceMetric`.
    pub fn parse_distance_metric(s: &str) -> Result<DistanceMetric, String> {
        match s.trim().to_lowercase().as_str() {
            "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
            "cosine" => Ok(DistanceMetric::Cosine),
            "dot_product" | "dot" | "dotproduct" | "inner_product" => {
                Ok(DistanceMetric::DotProduct)
            }
            other => Err(format!(
                "unknown distance metric '{other}': expected euclidean, cosine, or dot_product"
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Vector SQL functions
// ---------------------------------------------------------------------------

/// Compute the distance between two vectors using the specified metric.
pub fn vector_distance(v1: &[f32], v2: &[f32], metric: &DistanceMetric) -> Result<f64, String> {
    if v1.len() != v2.len() {
        return Err(format!(
            "vector_distance: dimension mismatch ({} vs {})",
            v1.len(),
            v2.len()
        ));
    }
    Ok(metric.compute(v1, v2) as f64)
}

/// Compute cosine similarity between two vectors (1 - cosine distance).
/// Returns a value in [-1, 1] where 1 means identical direction.
pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> Result<f64, String> {
    if v1.len() != v2.len() {
        return Err(format!(
            "cosine_similarity: dimension mismatch ({} vs {})",
            v1.len(),
            v2.len()
        ));
    }
    let dist = DistanceMetric::Cosine.compute(v1, v2);
    Ok((1.0 - dist) as f64)
}

/// Compute the L2 (Euclidean) norm of a vector.
pub fn vector_norm(v: &[f32]) -> f64 {
    let sum_sq: f32 = v.iter().map(|x| x * x).sum();
    sum_sq.sqrt() as f64
}

/// Return the number of dimensions in a vector.
pub fn vector_dims(v: &[f32]) -> i64 {
    v.len() as i64
}

// ---------------------------------------------------------------------------
// VectorQueryAdvisor — AI-driven parameter tuning
// ---------------------------------------------------------------------------

/// AI advisor that monitors vector query patterns and recommends index
/// parameter adjustments for optimal recall-vs-latency tradeoffs.
///
/// The advisor tracks:
/// - Average `k` values in searches (large k benefits from higher ef_search)
/// - Query frequency (high throughput may warrant exact-threshold increases)
/// - Insert-to-search ratio (write-heavy vs read-heavy workloads)
///
/// Based on these observations it recommends:
/// - `ef_search` — HNSW search beam width (higher = better recall, slower)
/// - `exact_threshold` — vector count below which exact search is preferred
pub struct VectorQueryAdvisor {
    search_k_values: Vec<usize>,
    total_searches: u64,
    total_inserts: u64,
    last_recommendation_at: u64,
}

impl VectorQueryAdvisor {
    /// Create a new advisor with default state.
    pub fn new() -> Self {
        Self {
            search_k_values: Vec::new(),
            total_searches: 0,
            total_inserts: 0,
            last_recommendation_at: 0,
        }
    }

    /// Record a search operation with the requested `k`.
    pub fn record_search(&mut self, k: usize) {
        self.total_searches += 1;
        self.search_k_values.push(k);
        // Keep a sliding window of the last 1000 k-values
        if self.search_k_values.len() > 1000 {
            self.search_k_values.drain(0..500);
        }
    }

    /// Record an insert operation.
    pub fn record_insert(&mut self) {
        self.total_inserts += 1;
    }

    /// Average `k` value across recent searches.
    pub fn avg_k(&self) -> f64 {
        if self.search_k_values.is_empty() {
            return 10.0; // sensible default
        }
        let sum: usize = self.search_k_values.iter().sum();
        sum as f64 / self.search_k_values.len() as f64
    }

    /// Recommend an `ef_search` value based on observed query patterns.
    ///
    /// Higher average k -> higher ef_search for better recall at top-k.
    /// Minimum ef_search is 50; maximum is 500.
    pub fn recommend_ef_search(&mut self) -> usize {
        self.last_recommendation_at = self.total_searches;
        let avg = self.avg_k();
        // ef_search should be at least 2x the average k, clamped to [50, 500]
        let ef = (avg * 2.0).round() as usize;
        ef.clamp(50, 500)
    }

    /// Recommend an exact-search threshold based on workload balance.
    ///
    /// Read-heavy workloads benefit from a lower threshold (use HNSW sooner);
    /// write-heavy workloads benefit from a higher threshold (delay HNSW build cost).
    pub fn recommend_exact_threshold(&self) -> usize {
        if self.total_inserts == 0 && self.total_searches == 0 {
            return 1000; // default
        }
        let total = self.total_inserts + self.total_searches;
        let search_ratio = self.total_searches as f64 / total as f64;

        if search_ratio > 0.7 {
            // Read-heavy: lower threshold to engage HNSW sooner
            500
        } else if search_ratio < 0.3 {
            // Write-heavy: higher threshold to avoid HNSW rebuild overhead
            5000
        } else {
            1000 // balanced
        }
    }

    /// Summary statistics for the advisor.
    pub fn stats(&self) -> VectorQueryAdvisorStats {
        VectorQueryAdvisorStats {
            total_searches: self.total_searches,
            total_inserts: self.total_inserts,
            avg_k: self.avg_k(),
            recommended_ef_search: if self.total_searches > 0 {
                let avg = self.avg_k();
                (avg * 2.0).round() as usize
            } else {
                50
            },
            recommended_exact_threshold: self.recommend_exact_threshold(),
        }
    }
}

impl Default for VectorQueryAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics snapshot from the `VectorQueryAdvisor`.
#[derive(Debug, Clone)]
pub struct VectorQueryAdvisorStats {
    pub total_searches: u64,
    pub total_inserts: u64,
    pub avg_k: f64,
    pub recommended_ef_search: usize,
    pub recommended_exact_threshold: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- VectorStore CRUD tests --

    #[test]
    fn test_store_create_and_list_indexes() {
        let store = VectorStore::new();
        store
            .create_index("emb128", 128, DistanceMetric::Cosine)
            .unwrap();
        store
            .create_index("emb3", 3, DistanceMetric::Euclidean)
            .unwrap();

        let list = store.list_indexes();
        assert_eq!(list.len(), 2);

        let names: Vec<&str> = list.iter().map(|i| i.name.as_str()).collect();
        assert!(names.contains(&"emb128"));
        assert!(names.contains(&"emb3"));
    }

    #[test]
    fn test_store_create_duplicate_index() {
        let store = VectorStore::new();
        store
            .create_index("idx", 3, DistanceMetric::Euclidean)
            .unwrap();
        let err = store.create_index("idx", 3, DistanceMetric::Euclidean);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("already exists"));
    }

    #[test]
    fn test_store_drop_index() {
        let store = VectorStore::new();
        store
            .create_index("temp", 2, DistanceMetric::Cosine)
            .unwrap();
        assert_eq!(store.list_indexes().len(), 1);

        store.drop_index("temp").unwrap();
        assert_eq!(store.list_indexes().len(), 0);
    }

    #[test]
    fn test_store_drop_nonexistent() {
        let store = VectorStore::new();
        let err = store.drop_index("nope");
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("does not exist"));
    }

    #[test]
    fn test_store_insert_and_search() {
        let store = VectorStore::new();
        store
            .create_index("docs", 3, DistanceMetric::Euclidean)
            .unwrap();

        store
            .insert("docs", "v1", vec![1.0, 0.0, 0.0], None)
            .unwrap();
        store
            .insert("docs", "v2", vec![0.0, 1.0, 0.0], None)
            .unwrap();
        store
            .insert("docs", "v3", vec![0.0, 0.0, 1.0], None)
            .unwrap();

        let results = store.search("docs", &[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_store_get_entry() {
        let store = VectorStore::new();
        store
            .create_index("idx", 2, DistanceMetric::Euclidean)
            .unwrap();

        store
            .insert(
                "idx",
                "item1",
                vec![3.0, 4.0],
                Some(serde_json::json!({"label": "test"})),
            )
            .unwrap();

        let entry = store.get("idx", "item1").unwrap().unwrap();
        assert_eq!(entry.vector, vec![3.0, 4.0]);
        assert!(entry.metadata.is_some());

        let missing = store.get("idx", "no_such_item").unwrap();
        assert!(missing.is_none());
    }

    #[test]
    fn test_store_stats() {
        let store = VectorStore::new();
        store.create_index("s", 3, DistanceMetric::Cosine).unwrap();
        store.insert("s", "a", vec![1.0, 0.0, 0.0], None).unwrap();

        let stats = store.stats("s").unwrap();
        assert_eq!(stats.name, "s");
        assert_eq!(stats.dimensions, 3);
        assert_eq!(stats.vector_count, 1);

        assert!(store.stats("nonexistent").is_none());
    }

    #[test]
    fn test_store_insert_into_nonexistent_index() {
        let store = VectorStore::new();
        let err = store.insert("ghost", "a", vec![1.0], None);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("does not exist"));
    }

    // -- VectorSqlBridge tests --

    #[test]
    fn test_parse_vector_literal_brackets() {
        let v = VectorSqlBridge::parse_vector_literal("[1.0, 2.5, 3.0]").unwrap();
        assert_eq!(v, vec![1.0, 2.5, 3.0]);
    }

    #[test]
    fn test_parse_vector_literal_bare() {
        let v = VectorSqlBridge::parse_vector_literal("4.0, 5.0, 6.0").unwrap();
        assert_eq!(v, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_parse_vector_literal_whitespace() {
        let v = VectorSqlBridge::parse_vector_literal("  [ 1.0 , 2.0 , 3.0 ]  ").unwrap();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_vector_literal_empty() {
        assert!(VectorSqlBridge::parse_vector_literal("").is_err());
        assert!(VectorSqlBridge::parse_vector_literal("[]").is_err());
    }

    #[test]
    fn test_parse_vector_literal_invalid() {
        assert!(VectorSqlBridge::parse_vector_literal("[1.0, abc, 3.0]").is_err());
    }

    #[test]
    fn test_format_vector() {
        let s = VectorSqlBridge::format_vector(&[1.0, 2.5, 3.0]);
        assert_eq!(s, "[1, 2.5, 3]");
    }

    #[test]
    fn test_parse_distance_metric() {
        assert_eq!(
            VectorSqlBridge::parse_distance_metric("euclidean").unwrap(),
            DistanceMetric::Euclidean
        );
        assert_eq!(
            VectorSqlBridge::parse_distance_metric("L2").unwrap(),
            DistanceMetric::Euclidean
        );
        assert_eq!(
            VectorSqlBridge::parse_distance_metric("cosine").unwrap(),
            DistanceMetric::Cosine
        );
        assert_eq!(
            VectorSqlBridge::parse_distance_metric("dot_product").unwrap(),
            DistanceMetric::DotProduct
        );
        assert_eq!(
            VectorSqlBridge::parse_distance_metric("DOT").unwrap(),
            DistanceMetric::DotProduct
        );
        assert!(VectorSqlBridge::parse_distance_metric("hamming").is_err());
    }

    // -- Vector SQL function tests --

    #[test]
    fn test_vector_distance_euclidean() {
        let d = vector_distance(&[0.0, 0.0], &[3.0, 4.0], &DistanceMetric::Euclidean).unwrap();
        assert!((d - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_distance_dimension_mismatch() {
        let err = vector_distance(&[1.0, 2.0], &[1.0], &DistanceMetric::Euclidean);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_dimension_mismatch() {
        let err = cosine_similarity(&[1.0], &[1.0, 2.0]);
        assert!(err.is_err());
    }

    #[test]
    fn test_vector_norm() {
        let n = vector_norm(&[3.0, 4.0]);
        assert!((n - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_norm_zero() {
        let n = vector_norm(&[0.0, 0.0, 0.0]);
        assert!((n - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_vector_dims() {
        assert_eq!(vector_dims(&[1.0, 2.0, 3.0]), 3);
        assert_eq!(vector_dims(&[]), 0);
    }

    // -- VectorQueryAdvisor tests --

    #[test]
    fn test_advisor_default_avg_k() {
        let advisor = VectorQueryAdvisor::new();
        assert!((advisor.avg_k() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_advisor_tracks_k_values() {
        let mut advisor = VectorQueryAdvisor::new();
        advisor.record_search(5);
        advisor.record_search(15);
        // avg = (5 + 15) / 2 = 10
        assert!((advisor.avg_k() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_advisor_ef_search_recommendation() {
        let mut advisor = VectorQueryAdvisor::new();
        // Record k=100 searches
        for _ in 0..10 {
            advisor.record_search(100);
        }
        // avg_k = 100, recommend ef_search = 200 (2x avg), clamped to [50, 500]
        let ef = advisor.recommend_ef_search();
        assert_eq!(ef, 200);
    }

    #[test]
    fn test_advisor_ef_search_clamp_low() {
        let mut advisor = VectorQueryAdvisor::new();
        advisor.record_search(5);
        // avg_k = 5, 2x = 10, but minimum is 50
        let ef = advisor.recommend_ef_search();
        assert_eq!(ef, 50);
    }

    #[test]
    fn test_advisor_ef_search_clamp_high() {
        let mut advisor = VectorQueryAdvisor::new();
        for _ in 0..10 {
            advisor.record_search(500);
        }
        // avg_k = 500, 2x = 1000, clamped to 500
        let ef = advisor.recommend_ef_search();
        assert_eq!(ef, 500);
    }

    #[test]
    fn test_advisor_exact_threshold_read_heavy() {
        let mut advisor = VectorQueryAdvisor::new();
        // 80 searches, 20 inserts => search_ratio = 0.8 > 0.7 => threshold = 500
        for _ in 0..80 {
            advisor.record_search(10);
        }
        for _ in 0..20 {
            advisor.record_insert();
        }
        assert_eq!(advisor.recommend_exact_threshold(), 500);
    }

    #[test]
    fn test_advisor_exact_threshold_write_heavy() {
        let mut advisor = VectorQueryAdvisor::new();
        // 10 searches, 90 inserts => search_ratio = 0.1 < 0.3 => threshold = 5000
        for _ in 0..10 {
            advisor.record_search(10);
        }
        for _ in 0..90 {
            advisor.record_insert();
        }
        assert_eq!(advisor.recommend_exact_threshold(), 5000);
    }

    #[test]
    fn test_advisor_exact_threshold_balanced() {
        let mut advisor = VectorQueryAdvisor::new();
        for _ in 0..50 {
            advisor.record_search(10);
        }
        for _ in 0..50 {
            advisor.record_insert();
        }
        // search_ratio = 0.5, balanced => 1000
        assert_eq!(advisor.recommend_exact_threshold(), 1000);
    }

    #[test]
    fn test_advisor_stats() {
        let mut advisor = VectorQueryAdvisor::new();
        advisor.record_search(10);
        advisor.record_search(20);
        advisor.record_insert();

        let stats = advisor.stats();
        assert_eq!(stats.total_searches, 2);
        assert_eq!(stats.total_inserts, 1);
        assert!((stats.avg_k - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_advisor_sliding_window() {
        let mut advisor = VectorQueryAdvisor::new();
        // Fill beyond the 1000 window
        for i in 0..1200 {
            advisor.record_search(i);
        }
        // After draining, only recent values remain (window is trimmed)
        assert!(advisor.search_k_values.len() <= 1000);
        assert_eq!(advisor.total_searches, 1200);
    }

    // -- Multi-index management tests --

    #[test]
    fn test_store_multi_index_isolation() {
        let store = VectorStore::new();
        store
            .create_index("idx_a", 2, DistanceMetric::Euclidean)
            .unwrap();
        store
            .create_index("idx_b", 3, DistanceMetric::Cosine)
            .unwrap();

        store.insert("idx_a", "a1", vec![1.0, 0.0], None).unwrap();
        store
            .insert("idx_b", "b1", vec![1.0, 0.0, 0.0], None)
            .unwrap();

        // Each index only sees its own vectors
        let r_a = store.search("idx_a", &[1.0, 0.0], 10).unwrap();
        assert_eq!(r_a.len(), 1);
        assert_eq!(r_a[0].id, "a1");

        let r_b = store.search("idx_b", &[1.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(r_b.len(), 1);
        assert_eq!(r_b[0].id, "b1");
    }

    #[test]
    fn test_store_total_ops_tracking() {
        let store = VectorStore::new();
        assert_eq!(store.total_ops(), 0);

        store
            .create_index("x", 2, DistanceMetric::Euclidean)
            .unwrap();
        assert_eq!(store.total_ops(), 1);

        store.insert("x", "a", vec![1.0, 2.0], None).unwrap();
        assert_eq!(store.total_ops(), 2);

        store.search("x", &[1.0, 2.0], 1).unwrap();
        assert_eq!(store.total_ops(), 3);

        let _ = store.get("x", "a").unwrap();
        assert_eq!(store.total_ops(), 4);

        store.drop_index("x").unwrap();
        assert_eq!(store.total_ops(), 5);
    }

    // -- Edge case tests --

    #[test]
    fn test_store_dimension_mismatch_on_insert() {
        let store = VectorStore::new();
        store
            .create_index("strict", 3, DistanceMetric::Euclidean)
            .unwrap();

        let err = store.insert("strict", "bad", vec![1.0, 2.0], None);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("dimension mismatch"));
    }

    #[test]
    fn test_store_search_nonexistent_index() {
        let store = VectorStore::new();
        let err = store.search("ghost", &[1.0], 1);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("does not exist"));
    }

    #[test]
    fn test_parse_vector_single_element() {
        let v = VectorSqlBridge::parse_vector_literal("[42.0]").unwrap();
        assert_eq!(v, vec![42.0]);
    }

    #[test]
    fn test_format_and_reparse_roundtrip() {
        let original = vec![1.5, 2.25, 3.0, -4.5];
        let formatted = VectorSqlBridge::format_vector(&original);
        let reparsed = VectorSqlBridge::parse_vector_literal(&formatted).unwrap();
        assert_eq!(original.len(), reparsed.len());
        for (a, b) in original.iter().zip(reparsed.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }
}
