use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

/// Bounded, fixed-size map that tracks access counts and evicts cold entries.
/// Designed for approximate hot-key tracking with bounded memory.
pub struct LruCountMap {
    map: HashMap<Vec<u8>, u64>,
    max_entries: usize,
}

impl LruCountMap {
    pub fn new(max_entries: usize) -> Self {
        Self {
            map: HashMap::with_capacity(max_entries.min(256)),
            max_entries,
        }
    }

    pub fn increment(&mut self, key: &[u8]) {
        if let Some(count) = self.map.get_mut(key) {
            *count += 1;
            return;
        }
        // Evict coldest entry if at capacity
        if self.map.len() >= self.max_entries {
            if let Some(coldest) = self
                .map
                .iter()
                .min_by_key(|(_, &count)| count)
                .map(|(k, _)| k.clone())
            {
                self.map.remove(&coldest);
            }
        }
        self.map.insert(key.to_vec(), 1);
    }

    pub fn get_count(&self, key: &[u8]) -> u64 {
        self.map.get(key).copied().unwrap_or(0)
    }

    pub fn top_n(&self, n: usize) -> Vec<(Vec<u8>, u64)> {
        let mut entries: Vec<(Vec<u8>, u64)> =
            self.map.iter().map(|(k, &v)| (k.clone(), v)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.truncate(n);
        entries
    }
}

/// Per-shard access pattern statistics.
/// Atomic counters for the hot path; LruCountMap behind Mutex for detailed tracking.
pub struct AccessStats {
    pub hot_keys: Mutex<LruCountMap>,
    pub prefix_counts: Mutex<LruCountMap>,
    pub total_gets: AtomicU64,
    pub total_scans: AtomicU64,
    pub total_puts: AtomicU64,
}

impl AccessStats {
    pub fn new(capacity: usize) -> Self {
        Self {
            hot_keys: Mutex::new(LruCountMap::new(capacity)),
            prefix_counts: Mutex::new(LruCountMap::new(capacity)),
            total_gets: AtomicU64::new(0),
            total_scans: AtomicU64::new(0),
            total_puts: AtomicU64::new(0),
        }
    }

    /// Record a point read access. Lock-free atomic path; best-effort detailed tracking.
    pub fn record_get(&self, user_key: &[u8]) {
        self.total_gets.fetch_add(1, Ordering::Relaxed);
        if let Some(mut map) = self.hot_keys.try_lock() {
            map.increment(user_key);
        }
    }

    /// Record a prefix scan access.
    pub fn record_scan(&self, prefix: &[u8]) {
        self.total_scans.fetch_add(1, Ordering::Relaxed);
        if let Some(mut map) = self.prefix_counts.try_lock() {
            map.increment(prefix);
        }
    }

    /// Record a write.
    pub fn record_put(&self) {
        self.total_puts.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_count_map_basic() {
        let mut map = LruCountMap::new(3);
        map.increment(b"a");
        map.increment(b"a");
        map.increment(b"b");
        assert_eq!(map.get_count(b"a"), 2);
        assert_eq!(map.get_count(b"b"), 1);
        assert_eq!(map.get_count(b"c"), 0);
    }

    #[test]
    fn lru_count_map_evicts_coldest() {
        let mut map = LruCountMap::new(2);
        map.increment(b"hot");
        map.increment(b"hot");
        map.increment(b"warm");
        // At capacity (2). Insert "new" should evict "warm" (count=1) not "hot" (count=2)
        map.increment(b"new");
        assert_eq!(map.get_count(b"hot"), 2);
        assert_eq!(map.get_count(b"warm"), 0); // evicted
        assert_eq!(map.get_count(b"new"), 1);
    }

    #[test]
    fn lru_count_map_top_n() {
        let mut map = LruCountMap::new(10);
        for _ in 0..5 {
            map.increment(b"a");
        }
        for _ in 0..3 {
            map.increment(b"b");
        }
        for _ in 0..1 {
            map.increment(b"c");
        }
        let top = map.top_n(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, b"a".to_vec());
        assert_eq!(top[1].0, b"b".to_vec());
    }

    #[test]
    fn access_stats_record() {
        let stats = AccessStats::new(100);
        stats.record_get(b"key1");
        stats.record_get(b"key1");
        stats.record_scan(b"prefix/");
        stats.record_put();
        assert_eq!(stats.total_gets.load(Ordering::Relaxed), 2);
        assert_eq!(stats.total_scans.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_puts.load(Ordering::Relaxed), 1);
        assert_eq!(stats.hot_keys.lock().get_count(b"key1"), 2);
    }
}
