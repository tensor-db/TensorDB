use std::sync::Arc;

use crate::native_bridge::Hasher;

/// Count-Min Sketch for approximate frequency estimation.
/// Uses 4 hash rows x `width` columns. O(1) space per query, O(1) increment.
/// Periodically decays counters to forget stale frequency data.
pub struct CountMinSketch {
    rows: usize,
    width: usize,
    table: Vec<Vec<u32>>,
    hasher: Arc<dyn Hasher + Send + Sync>,
    total_insertions: u64,
    decay_threshold: u64,
}

impl CountMinSketch {
    pub fn new(width: usize, hasher: Arc<dyn Hasher + Send + Sync>) -> Self {
        let rows = 4;
        Self {
            rows,
            width,
            table: vec![vec![0u32; width]; rows],
            hasher,
            total_insertions: 0,
            decay_threshold: width as u64 * 10,
        }
    }

    /// Increment the frequency estimate for a key.
    pub fn increment(&mut self, key: &[u8]) {
        for row in 0..self.rows {
            let idx = self.hash_for_row(key, row);
            self.table[row][idx] = self.table[row][idx].saturating_add(1);
        }
        self.total_insertions += 1;
        if self.total_insertions >= self.decay_threshold {
            self.decay();
        }
    }

    /// Estimate the frequency of a key (minimum across all hash rows).
    pub fn estimate(&self, key: &[u8]) -> u32 {
        let mut min = u32::MAX;
        for row in 0..self.rows {
            let idx = self.hash_for_row(key, row);
            min = min.min(self.table[row][idx]);
        }
        min
    }

    fn hash_for_row(&self, key: &[u8], row: usize) -> usize {
        let mut data = key.to_vec();
        data.extend_from_slice(&(row as u64).to_le_bytes());
        (self.hasher.hash64(&data) as usize) % self.width
    }

    /// Halve all counters to decay old frequency data.
    fn decay(&mut self) {
        for row in &mut self.table {
            for cell in row.iter_mut() {
                *cell >>= 1;
            }
        }
        self.total_insertions = 0;
    }
}

/// AI cache advisor that tracks access frequency for cache eviction decisions.
pub struct CacheAdvisor {
    sketch: CountMinSketch,
}

impl CacheAdvisor {
    pub fn new(width: usize, hasher: Arc<dyn Hasher + Send + Sync>) -> Self {
        Self {
            sketch: CountMinSketch::new(width, hasher),
        }
    }

    /// Record an access to a cache key (block identified by path_hash + offset).
    pub fn record_access(&mut self, path_hash: u64, block_offset: u64) {
        let key = cache_key_bytes(path_hash, block_offset);
        self.sketch.increment(&key);
    }

    /// Estimate the frequency of a cache key.
    pub fn estimate_frequency(&self, path_hash: u64, block_offset: u64) -> u32 {
        let key = cache_key_bytes(path_hash, block_offset);
        self.sketch.estimate(&key)
    }

    /// Should we admit this new entry? Returns true if the new entry's estimated
    /// frequency exceeds the victim's frequency (TinyLFU-inspired admission).
    pub fn should_admit(
        &self,
        new_path_hash: u64,
        new_block_offset: u64,
        victim_path_hash: u64,
        victim_block_offset: u64,
    ) -> bool {
        let new_freq = self.estimate_frequency(new_path_hash, new_block_offset);
        let victim_freq = self.estimate_frequency(victim_path_hash, victim_block_offset);
        new_freq >= victim_freq
    }
}

fn cache_key_bytes(path_hash: u64, block_offset: u64) -> Vec<u8> {
    let mut key = Vec::with_capacity(16);
    key.extend_from_slice(&path_hash.to_le_bytes());
    key.extend_from_slice(&block_offset.to_le_bytes());
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_bridge::build_hasher;

    #[test]
    fn count_min_sketch_basic() {
        let hasher = build_hasher();
        let mut sketch = CountMinSketch::new(1024, hasher);
        for _ in 0..100 {
            sketch.increment(b"hot-key");
        }
        sketch.increment(b"cold-key");
        assert!(sketch.estimate(b"hot-key") >= 50); // May be approximate
        assert!(sketch.estimate(b"cold-key") >= 1);
        assert!(sketch.estimate(b"hot-key") > sketch.estimate(b"cold-key"));
    }

    #[test]
    fn count_min_sketch_decay() {
        let hasher = build_hasher();
        let mut sketch = CountMinSketch::new(64, hasher);
        // Insert enough to trigger decay (threshold = 64 * 10 = 640)
        for _ in 0..650 {
            sketch.increment(b"key");
        }
        // After decay, counters should be halved
        let est = sketch.estimate(b"key");
        assert!(est < 650);
    }

    #[test]
    fn cache_advisor_admission() {
        let hasher = build_hasher();
        let mut advisor = CacheAdvisor::new(1024, hasher);
        // Record many accesses for block (1, 0)
        for _ in 0..50 {
            advisor.record_access(1, 0);
        }
        // Record few accesses for block (2, 0)
        advisor.record_access(2, 0);

        // New entry (3, 0) never accessed — should not displace (1, 0)
        assert!(!advisor.should_admit(3, 0, 1, 0));
        // But should displace (2, 0) since both are ~equally cold
        // (actually (3,0) has freq 0, (2,0) has freq 1, so it won't displace)
        assert!(!advisor.should_admit(3, 0, 2, 0));
    }
}
