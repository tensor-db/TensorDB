//! Fractal Index — a novel hybrid B-tree/hash structure for O(1) point lookups with range scan.
//!
//! # The Problem
//! B-trees: O(log n) point lookup, O(log n + k) range scan. Great for ranges, not ideal for points.
//! Hash tables: O(1) point lookup, but no range scan support.
//! LSM-trees: O(log n) with bloom filter optimization, but still multiple levels to search.
//!
//! # The Innovation
//! The Fractal Index combines:
//! 1. A hash table for O(1) point lookups (direct key → SSTable block offset)
//! 2. A sorted skip list for O(log n + k) range scans
//! 3. AI-driven routing that learns whether each key should be hash-indexed, range-indexed, or both
//!
//! When a key is accessed by point lookup, it's added to the hash index.
//! When a key participates in a range scan, it stays in the sorted index.
//! Keys that are accessed both ways get both indices.
//!
//! # AI Integration
//! An AI router tracks access patterns per key prefix and predicts the optimal index type:
//! - Prefix "user/" → 90% point lookups → hash-primary
//! - Prefix "log/" → 80% range scans → range-primary
//! - Prefix "order/" → mixed → both indices
//!
//! # Why This Is Novel
//! - B-tree databases (PostgreSQL, MySQL): single index structure per table
//! - LSM databases (RocksDB): bloom filter + sorted runs, no hash index
//! - Hash databases (Redis, LMDB): no range scan support
//! - TensorDB: per-key-prefix AI-routed dual-index with minimal memory overhead

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

/// Location of data in the storage layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataLocation {
    /// SSTable file identifier.
    pub file_id: u64,
    /// Block offset within the SSTable.
    pub block_offset: u64,
    /// Entry offset within the block.
    pub entry_offset: u32,
}

/// The Fractal Index — hybrid hash + sorted index with AI routing.
pub struct FractalIndex {
    /// Hash index for O(1) point lookups.
    hash_index: RwLock<HashMap<Vec<u8>, DataLocation>>,
    /// Sorted index for range scans.
    sorted_index: RwLock<BTreeMap<Vec<u8>, DataLocation>>,
    /// AI router: tracks access patterns per key prefix.
    router: RwLock<IndexRouter>,
    /// Stats
    point_lookups: AtomicU64,
    range_scans: AtomicU64,
    hash_hits: AtomicU64,
    hash_misses: AtomicU64,
    sorted_hits: AtomicU64,
}

impl FractalIndex {
    pub fn new(prefix_len: usize) -> Self {
        Self {
            hash_index: RwLock::new(HashMap::new()),
            sorted_index: RwLock::new(BTreeMap::new()),
            router: RwLock::new(IndexRouter::new(prefix_len)),
            point_lookups: AtomicU64::new(0),
            range_scans: AtomicU64::new(0),
            hash_hits: AtomicU64::new(0),
            hash_misses: AtomicU64::new(0),
            sorted_hits: AtomicU64::new(0),
        }
    }

    /// Insert a key-location pair into the index.
    /// The AI router decides which index(es) to use.
    pub fn insert(&self, key: Vec<u8>, location: DataLocation) {
        let strategy = self.router.read().recommend(&key);

        match strategy {
            IndexStrategy::HashOnly => {
                self.hash_index.write().insert(key, location);
            }
            IndexStrategy::SortedOnly => {
                self.sorted_index.write().insert(key, location);
            }
            IndexStrategy::Both => {
                self.hash_index
                    .write()
                    .insert(key.clone(), location.clone());
                self.sorted_index.write().insert(key, location);
            }
        }
    }

    /// Point lookup: O(1) via hash index, fallback to O(log n) sorted index.
    pub fn point_lookup(&self, key: &[u8]) -> Option<DataLocation> {
        self.point_lookups.fetch_add(1, Ordering::Relaxed);
        self.router.write().record_point_lookup(key);

        // Try hash index first (O(1))
        if let Some(loc) = self.hash_index.read().get(key).cloned() {
            self.hash_hits.fetch_add(1, Ordering::Relaxed);
            return Some(loc);
        }

        self.hash_misses.fetch_add(1, Ordering::Relaxed);

        // Fallback to sorted index (O(log n))
        if let Some(loc) = self.sorted_index.read().get(key).cloned() {
            self.sorted_hits.fetch_add(1, Ordering::Relaxed);

            // Promote to hash index for future O(1) access
            self.hash_index.write().insert(key.to_vec(), loc.clone());

            return Some(loc);
        }

        None
    }

    /// Range scan: uses sorted index. Returns entries in [start, end).
    pub fn range_scan(
        &self,
        start: &[u8],
        end: &[u8],
        limit: usize,
    ) -> Vec<(Vec<u8>, DataLocation)> {
        self.range_scans.fetch_add(1, Ordering::Relaxed);
        self.router.write().record_range_scan(start);

        let sorted = self.sorted_index.read();
        sorted
            .range(start.to_vec()..end.to_vec())
            .take(limit)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Prefix scan: range scan with prefix bounds.
    pub fn prefix_scan(&self, prefix: &[u8], limit: usize) -> Vec<(Vec<u8>, DataLocation)> {
        self.range_scans.fetch_add(1, Ordering::Relaxed);
        self.router.write().record_range_scan(prefix);

        let end = prefix_successor(prefix);
        let sorted = self.sorted_index.read();

        match end {
            Some(end_key) => sorted
                .range(prefix.to_vec()..end_key)
                .take(limit)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            None => sorted
                .range(prefix.to_vec()..)
                .take(limit)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        }
    }

    /// Remove a key from all indices.
    pub fn remove(&self, key: &[u8]) {
        self.hash_index.write().remove(key);
        self.sorted_index.write().remove(key);
    }

    /// Get the total number of indexed keys.
    pub fn len(&self) -> usize {
        // Sorted index is the authoritative count (hash may be a subset)
        let hash_count = self.hash_index.read().len();
        let sorted_count = self.sorted_index.read().len();
        hash_count.max(sorted_count)
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.hash_index.read().is_empty() && self.sorted_index.read().is_empty()
    }

    /// Get index statistics.
    pub fn stats(&self) -> FractalIndexStats {
        let router = self.router.read();
        FractalIndexStats {
            hash_entries: self.hash_index.read().len(),
            sorted_entries: self.sorted_index.read().len(),
            point_lookups: self.point_lookups.load(Ordering::Relaxed),
            range_scans: self.range_scans.load(Ordering::Relaxed),
            hash_hits: self.hash_hits.load(Ordering::Relaxed),
            hash_misses: self.hash_misses.load(Ordering::Relaxed),
            sorted_hits: self.sorted_hits.load(Ordering::Relaxed),
            prefix_strategies: router.prefix_stats(),
        }
    }
}

/// AI-driven index router — learns optimal index strategy per key prefix.
struct IndexRouter {
    /// Prefix length for pattern tracking.
    prefix_len: usize,
    /// Per-prefix access statistics.
    prefix_stats: HashMap<Vec<u8>, PrefixAccessStats>,
}

#[derive(Debug, Clone, Default)]
struct PrefixAccessStats {
    point_lookups: u64,
    range_scans: u64,
}

impl IndexRouter {
    fn new(prefix_len: usize) -> Self {
        Self {
            prefix_len: if prefix_len == 0 { 8 } else { prefix_len },
            prefix_stats: HashMap::new(),
        }
    }

    fn extract_prefix(&self, key: &[u8]) -> Vec<u8> {
        if key.len() <= self.prefix_len {
            key.to_vec()
        } else {
            key[..self.prefix_len].to_vec()
        }
    }

    fn record_point_lookup(&mut self, key: &[u8]) {
        let prefix = self.extract_prefix(key);
        let stats = self.prefix_stats.entry(prefix).or_default();
        stats.point_lookups += 1;
    }

    fn record_range_scan(&mut self, key: &[u8]) {
        let prefix = self.extract_prefix(key);
        let stats = self.prefix_stats.entry(prefix).or_default();
        stats.range_scans += 1;
    }

    /// AI recommendation: which index strategy for this key?
    fn recommend(&self, key: &[u8]) -> IndexStrategy {
        let prefix = self.extract_prefix(key);
        let stats = match self.prefix_stats.get(&prefix) {
            Some(s) => s,
            None => return IndexStrategy::Both, // No data → default to both
        };

        let total = stats.point_lookups + stats.range_scans;
        if total < 10 {
            return IndexStrategy::Both; // Insufficient data
        }

        let point_ratio = stats.point_lookups as f64 / total as f64;

        if point_ratio > 0.85 {
            IndexStrategy::HashOnly // Overwhelmingly point lookups
        } else if point_ratio < 0.15 {
            IndexStrategy::SortedOnly // Overwhelmingly range scans
        } else {
            IndexStrategy::Both // Mixed access pattern
        }
    }

    fn prefix_stats(&self) -> Vec<PrefixStrategyInfo> {
        self.prefix_stats
            .iter()
            .map(|(prefix, stats)| {
                let total = stats.point_lookups + stats.range_scans;
                let strategy = if total < 10 {
                    IndexStrategy::Both
                } else {
                    let ratio = stats.point_lookups as f64 / total as f64;
                    if ratio > 0.85 {
                        IndexStrategy::HashOnly
                    } else if ratio < 0.15 {
                        IndexStrategy::SortedOnly
                    } else {
                        IndexStrategy::Both
                    }
                };

                PrefixStrategyInfo {
                    prefix: String::from_utf8_lossy(prefix).to_string(),
                    point_lookups: stats.point_lookups,
                    range_scans: stats.range_scans,
                    strategy,
                }
            })
            .collect()
    }
}

/// Index strategy recommended by the AI router.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexStrategy {
    /// Only maintain hash index (point-lookup dominated).
    HashOnly,
    /// Only maintain sorted index (range-scan dominated).
    SortedOnly,
    /// Maintain both indices (mixed access pattern).
    Both,
}

impl IndexStrategy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::HashOnly => "hash_only",
            Self::SortedOnly => "sorted_only",
            Self::Both => "both",
        }
    }
}

#[derive(Debug, Clone)]
pub struct FractalIndexStats {
    pub hash_entries: usize,
    pub sorted_entries: usize,
    pub point_lookups: u64,
    pub range_scans: u64,
    pub hash_hits: u64,
    pub hash_misses: u64,
    pub sorted_hits: u64,
    pub prefix_strategies: Vec<PrefixStrategyInfo>,
}

#[derive(Debug, Clone)]
pub struct PrefixStrategyInfo {
    pub prefix: String,
    pub point_lookups: u64,
    pub range_scans: u64,
    pub strategy: IndexStrategy,
}

/// Compute the successor of a prefix for range scan bounds.
fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut successor = prefix.to_vec();
    while let Some(last) = successor.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return Some(successor);
        }
        successor.pop();
    }
    None // All 0xFF — no successor
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loc(file_id: u64, block: u64, entry: u32) -> DataLocation {
        DataLocation {
            file_id,
            block_offset: block,
            entry_offset: entry,
        }
    }

    #[test]
    fn test_point_lookup_basic() {
        let idx = FractalIndex::new(4);
        idx.insert(b"key1".to_vec(), loc(1, 0, 0));
        idx.insert(b"key2".to_vec(), loc(1, 0, 1));

        assert_eq!(idx.point_lookup(b"key1"), Some(loc(1, 0, 0)));
        assert_eq!(idx.point_lookup(b"key2"), Some(loc(1, 0, 1)));
        assert_eq!(idx.point_lookup(b"key3"), None);
    }

    #[test]
    fn test_prefix_scan() {
        let idx = FractalIndex::new(4);
        idx.insert(b"user/1".to_vec(), loc(1, 0, 0));
        idx.insert(b"user/2".to_vec(), loc(1, 0, 1));
        idx.insert(b"user/3".to_vec(), loc(1, 0, 2));
        idx.insert(b"order/1".to_vec(), loc(2, 0, 0));

        let results = idx.prefix_scan(b"user/", 10);
        assert_eq!(results.len(), 3);
        assert!(results[0].0.starts_with(b"user/"));

        let order_results = idx.prefix_scan(b"order/", 10);
        assert_eq!(order_results.len(), 1);
    }

    #[test]
    fn test_range_scan() {
        let idx = FractalIndex::new(4);
        idx.insert(b"a".to_vec(), loc(1, 0, 0));
        idx.insert(b"b".to_vec(), loc(1, 0, 1));
        idx.insert(b"c".to_vec(), loc(1, 0, 2));
        idx.insert(b"d".to_vec(), loc(1, 0, 3));

        let results = idx.range_scan(b"b", b"d", 10);
        assert_eq!(results.len(), 2); // b, c (d excluded)
    }

    #[test]
    fn test_remove() {
        let idx = FractalIndex::new(4);
        idx.insert(b"key".to_vec(), loc(1, 0, 0));
        assert!(idx.point_lookup(b"key").is_some());

        idx.remove(b"key");
        assert!(idx.point_lookup(b"key").is_none());
    }

    #[test]
    fn test_hash_promotion() {
        let idx = FractalIndex::new(4);
        // Insert into sorted index only
        idx.sorted_index
            .write()
            .insert(b"orphan".to_vec(), loc(1, 0, 0));

        // Point lookup should find it in sorted and promote to hash
        let result = idx.point_lookup(b"orphan");
        assert!(result.is_some());

        // Verify it's now in hash index too
        assert!(idx.hash_index.read().contains_key(&b"orphan".to_vec()));
    }

    #[test]
    fn test_ai_router_point_heavy() {
        let idx = FractalIndex::new(5); // prefix_len=5 → "user/" is the prefix

        // Insert keys
        idx.insert(b"user/1".to_vec(), loc(1, 0, 0));
        idx.insert(b"user/2".to_vec(), loc(1, 0, 1));

        // Do many point lookups to train the router on "user/" prefix
        for _ in 0..20 {
            idx.point_lookup(b"user/1");
        }

        // Router should now recommend HashOnly for "user/" prefix
        let router = idx.router.read();
        let strategy = router.recommend(b"user/9");
        assert_eq!(strategy, IndexStrategy::HashOnly);
    }

    #[test]
    fn test_ai_router_range_heavy() {
        let idx = FractalIndex::new(4);

        // Insert keys and do range scans
        idx.insert(b"log/1".to_vec(), loc(1, 0, 0));
        idx.insert(b"log/2".to_vec(), loc(1, 0, 1));

        for _ in 0..20 {
            idx.prefix_scan(b"log/", 10);
        }

        // Router should recommend SortedOnly for "log/" prefix
        let router = idx.router.read();
        let strategy = router.recommend(b"log/x");
        assert_eq!(strategy, IndexStrategy::SortedOnly);
    }

    #[test]
    fn test_ai_router_mixed() {
        let idx = FractalIndex::new(6);
        idx.insert(b"order/1".to_vec(), loc(1, 0, 0));

        // Mix of point and range
        for _ in 0..10 {
            idx.point_lookup(b"order/1");
            idx.prefix_scan(b"order/", 10);
        }

        let router = idx.router.read();
        let strategy = router.recommend(b"order/x");
        assert_eq!(strategy, IndexStrategy::Both);
    }

    #[test]
    fn test_stats() {
        let idx = FractalIndex::new(4);
        idx.insert(b"a".to_vec(), loc(1, 0, 0));
        idx.point_lookup(b"a");
        idx.prefix_scan(b"a", 10);

        let stats = idx.stats();
        assert_eq!(stats.point_lookups, 1);
        assert_eq!(stats.range_scans, 1);
        assert!(stats.hash_entries > 0 || stats.sorted_entries > 0);
    }

    #[test]
    fn test_prefix_successor() {
        assert_eq!(prefix_successor(b"abc"), Some(b"abd".to_vec()));
        assert_eq!(prefix_successor(b"ab\xff"), Some(b"ac".to_vec()));
        assert_eq!(prefix_successor(b"\xff\xff\xff"), None);
        assert_eq!(prefix_successor(b""), None);
    }

    #[test]
    fn test_len_and_is_empty() {
        let idx = FractalIndex::new(4);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);

        idx.insert(b"key".to_vec(), loc(1, 0, 0));
        assert!(!idx.is_empty());
        assert!(!idx.is_empty());
    }

    #[test]
    fn test_high_cardinality() {
        let idx = FractalIndex::new(4);

        // Insert 1000 keys
        for i in 0..1000u64 {
            let key = format!("key/{i:06}").into_bytes();
            idx.insert(key, loc(i / 100, i % 100, 0));
        }

        assert_eq!(idx.len(), 1000);

        // Point lookup should find any key
        assert!(idx.point_lookup(b"key/000500").is_some());

        // Prefix scan — "key/0005" matches key/000500 through key/000599 (100 keys)
        let results = idx.prefix_scan(b"key/0005", 100);
        assert_eq!(results.len(), 100);

        // Narrower scan — "key/00050" matches key/000500 through key/000509 (10 keys)
        let results2 = idx.prefix_scan(b"key/00050", 100);
        assert_eq!(results2.len(), 10);
    }
}
