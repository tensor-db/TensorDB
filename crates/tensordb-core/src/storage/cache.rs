use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::native_bridge::Hasher;

#[derive(Clone, Hash, PartialEq, Eq)]
struct BlockCacheKey {
    path_hash: u64,
    block_offset: u64,
}

struct BlockCacheInner {
    map: HashMap<BlockCacheKey, Arc<Vec<u8>>>,
    order: VecDeque<BlockCacheKey>,
    current_bytes: usize,
    max_bytes: usize,
}

pub struct BlockCache {
    inner: Mutex<BlockCacheInner>,
}

impl BlockCache {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(BlockCacheInner {
                map: HashMap::new(),
                order: VecDeque::new(),
                current_bytes: 0,
                max_bytes,
            }),
        }
    }

    pub fn get(&self, path_hash: u64, block_offset: u64) -> Option<Arc<Vec<u8>>> {
        let mut inner = self.inner.lock();
        let key = BlockCacheKey {
            path_hash,
            block_offset,
        };
        if let Some(val) = inner.map.get(&key).cloned() {
            // Move to back for LRU
            if let Some(pos) = inner.order.iter().position(|k| k == &key) {
                inner.order.remove(pos);
                inner.order.push_back(key);
            }
            Some(val)
        } else {
            None
        }
    }

    pub fn insert(&self, path_hash: u64, block_offset: u64, data: Vec<u8>) {
        let mut inner = self.inner.lock();
        if inner.max_bytes == 0 {
            return;
        }

        let key = BlockCacheKey {
            path_hash,
            block_offset,
        };

        if inner.map.contains_key(&key) {
            return;
        }

        let data_len = data.len();

        // Evict until we have room
        while inner.current_bytes + data_len > inner.max_bytes {
            if let Some(evict_key) = inner.order.pop_front() {
                if let Some(evicted) = inner.map.remove(&evict_key) {
                    inner.current_bytes = inner.current_bytes.saturating_sub(evicted.len());
                }
            } else {
                break;
            }
        }

        inner.current_bytes += data_len;
        inner.order.push_back(key.clone());
        inner.map.insert(key, Arc::new(data));
    }

    pub fn size(&self) -> usize {
        self.inner.lock().current_bytes
    }

    pub fn entry_count(&self) -> usize {
        self.inner.lock().map.len()
    }
}

pub struct IndexCache {
    inner: Mutex<IndexCacheInner>,
}

struct IndexCacheInner {
    map: HashMap<u64, Arc<Vec<super::sstable::IndexEntry>>>,
    order: VecDeque<u64>,
    max_entries: usize,
}

impl IndexCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            inner: Mutex::new(IndexCacheInner {
                map: HashMap::new(),
                order: VecDeque::new(),
                max_entries,
            }),
        }
    }

    pub fn get(&self, path_hash: u64) -> Option<Arc<Vec<super::sstable::IndexEntry>>> {
        let mut inner = self.inner.lock();
        if let Some(val) = inner.map.get(&path_hash).cloned() {
            // Move to back for LRU
            if let Some(pos) = inner.order.iter().position(|k| *k == path_hash) {
                inner.order.remove(pos);
                inner.order.push_back(path_hash);
            }
            Some(val)
        } else {
            None
        }
    }

    pub fn insert(&self, path_hash: u64, index: Vec<super::sstable::IndexEntry>) {
        let mut inner = self.inner.lock();
        if inner.max_entries == 0 || inner.map.contains_key(&path_hash) {
            return;
        }
        while inner.map.len() >= inner.max_entries {
            if let Some(key) = inner.order.pop_front() {
                inner.map.remove(&key);
            } else {
                break;
            }
        }
        inner.order.push_back(path_hash);
        inner.map.insert(path_hash, Arc::new(index));
    }
}

// ---------------------------------------------------------------------------
// AiBlockCache — frequency-aware eviction (TinyLFU-inspired)
// ---------------------------------------------------------------------------

#[derive(Clone, Hash, PartialEq, Eq)]
struct AiBlockCacheKey {
    path_hash: u64,
    block_offset: u64,
}

struct AiCacheEntry {
    data: Arc<Vec<u8>>,
    access_count: u32,
}

/// Count-Min Sketch embedded in the cache for admission filtering.
struct CacheSketch {
    rows: usize,
    width: usize,
    table: Vec<Vec<u32>>,
    hasher: Arc<dyn Hasher + Send + Sync>,
    total: u64,
    decay_threshold: u64,
}

impl CacheSketch {
    fn new(width: usize, hasher: Arc<dyn Hasher + Send + Sync>) -> Self {
        let rows = 4;
        Self {
            rows,
            width,
            table: vec![vec![0u32; width]; rows],
            hasher,
            total: 0,
            decay_threshold: width as u64 * 10,
        }
    }

    fn increment(&mut self, key: &AiBlockCacheKey) {
        let key_bytes = ai_cache_key_bytes(key);
        for row in 0..self.rows {
            let idx = self.hash_row(&key_bytes, row);
            self.table[row][idx] = self.table[row][idx].saturating_add(1);
        }
        self.total += 1;
        if self.total >= self.decay_threshold {
            self.decay();
        }
    }

    fn estimate(&self, key: &AiBlockCacheKey) -> u32 {
        let key_bytes = ai_cache_key_bytes(key);
        let mut min = u32::MAX;
        for row in 0..self.rows {
            let idx = self.hash_row(&key_bytes, row);
            min = min.min(self.table[row][idx]);
        }
        min
    }

    fn hash_row(&self, key_bytes: &[u8], row: usize) -> usize {
        let mut data = key_bytes.to_vec();
        data.extend_from_slice(&(row as u64).to_le_bytes());
        (self.hasher.hash64(&data) as usize) % self.width
    }

    fn decay(&mut self) {
        for row in &mut self.table {
            for cell in row.iter_mut() {
                *cell >>= 1;
            }
        }
        self.total = 0;
    }
}

fn ai_cache_key_bytes(key: &AiBlockCacheKey) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&key.path_hash.to_le_bytes());
    bytes.extend_from_slice(&key.block_offset.to_le_bytes());
    bytes
}

struct AiBlockCacheInner {
    map: HashMap<AiBlockCacheKey, AiCacheEntry>,
    sketch: CacheSketch,
    current_bytes: usize,
    max_bytes: usize,
}

/// AI-driven block cache with frequency-aware eviction.
/// Uses a Count-Min Sketch for admission filtering (TinyLFU-inspired).
pub struct AiBlockCache {
    inner: Mutex<AiBlockCacheInner>,
}

impl AiBlockCache {
    pub fn new(max_bytes: usize, hasher: Arc<dyn Hasher + Send + Sync>) -> Self {
        Self {
            inner: Mutex::new(AiBlockCacheInner {
                map: HashMap::new(),
                sketch: CacheSketch::new(1024, hasher),
                current_bytes: 0,
                max_bytes,
            }),
        }
    }

    pub fn get(&self, path_hash: u64, block_offset: u64) -> Option<Arc<Vec<u8>>> {
        let mut inner = self.inner.lock();
        let key = AiBlockCacheKey {
            path_hash,
            block_offset,
        };
        // Increment sketch first (before mutable borrow of map)
        inner.sketch.increment(&key);
        if let Some(entry) = inner.map.get_mut(&key) {
            entry.access_count += 1;
            Some(entry.data.clone())
        } else {
            None
        }
    }

    pub fn insert(&self, path_hash: u64, block_offset: u64, data: Vec<u8>) {
        let mut inner = self.inner.lock();
        if inner.max_bytes == 0 {
            return;
        }

        let key = AiBlockCacheKey {
            path_hash,
            block_offset,
        };

        if inner.map.contains_key(&key) {
            return;
        }

        let data_len = data.len();
        let new_freq = inner.sketch.estimate(&key);

        // Evict entries with lowest frequency-to-age ratio
        while inner.current_bytes + data_len > inner.max_bytes {
            let victim = inner
                .map
                .iter()
                .min_by_key(|(_, entry)| entry.access_count)
                .map(|(k, _)| k.clone());
            if let Some(victim_key) = victim {
                let victim_freq = inner.sketch.estimate(&victim_key);
                // Only admit if new entry has higher estimated frequency than victim
                if new_freq < victim_freq && inner.current_bytes > 0 {
                    return; // Don't admit — victim is more valuable
                }
                if let Some(evicted) = inner.map.remove(&victim_key) {
                    inner.current_bytes = inner.current_bytes.saturating_sub(evicted.data.len());
                }
            } else {
                break;
            }
        }

        inner.current_bytes += data_len;
        inner.map.insert(
            key,
            AiCacheEntry {
                data: Arc::new(data),
                access_count: 1,
            },
        );
    }

    pub fn size(&self) -> usize {
        self.inner.lock().current_bytes
    }

    pub fn entry_count(&self) -> usize {
        self.inner.lock().map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_cache_insert_and_get() {
        let cache = BlockCache::new(1024);
        cache.insert(1, 0, vec![1, 2, 3]);
        let got = cache.get(1, 0).unwrap();
        assert_eq!(*got, vec![1, 2, 3]);
        assert!(cache.get(2, 0).is_none());
    }

    #[test]
    fn block_cache_eviction() {
        let cache = BlockCache::new(10);
        cache.insert(1, 0, vec![0; 6]);
        cache.insert(2, 0, vec![0; 6]);
        // First entry should have been evicted
        assert!(cache.get(1, 0).is_none());
        assert!(cache.get(2, 0).is_some());
    }

    #[test]
    fn block_cache_disabled_when_zero() {
        let cache = BlockCache::new(0);
        cache.insert(1, 0, vec![1, 2, 3]);
        assert!(cache.get(1, 0).is_none());
    }

    #[test]
    fn index_cache_basic() {
        let cache = IndexCache::new(2);
        let idx = vec![super::super::sstable::IndexEntry {
            last_key: vec![1],
            offset: 0,
            len: 10,
        }];
        cache.insert(1, idx.clone());
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
    }

    #[test]
    fn ai_block_cache_insert_and_get() {
        let hasher = crate::native_bridge::build_hasher();
        let cache = AiBlockCache::new(1024, hasher);
        cache.insert(1, 0, vec![1, 2, 3]);
        let got = cache.get(1, 0).unwrap();
        assert_eq!(*got, vec![1, 2, 3]);
        assert!(cache.get(2, 0).is_none());
    }

    #[test]
    fn ai_block_cache_eviction() {
        let hasher = crate::native_bridge::build_hasher();
        let cache = AiBlockCache::new(10, hasher);
        cache.insert(1, 0, vec![0; 6]);
        cache.insert(2, 0, vec![0; 6]);
        // Only one should fit
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn ai_block_cache_frequency_tracking() {
        let hasher = crate::native_bridge::build_hasher();
        let cache = AiBlockCache::new(100, hasher);
        cache.insert(1, 0, vec![1, 2, 3]);
        // Access multiple times to build frequency
        for _ in 0..5 {
            cache.get(1, 0);
        }
        assert!(cache.get(1, 0).is_some());
    }

    #[test]
    fn ai_block_cache_disabled_when_zero() {
        let hasher = crate::native_bridge::build_hasher();
        let cache = AiBlockCache::new(0, hasher);
        cache.insert(1, 0, vec![1, 2, 3]);
        assert!(cache.get(1, 0).is_none());
    }
}
