use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use parking_lot::Mutex;

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
        let inner = self.inner.lock();
        let key = BlockCacheKey {
            path_hash,
            block_offset,
        };
        inner.map.get(&key).cloned()
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
        self.inner.lock().map.get(&path_hash).cloned()
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
}
