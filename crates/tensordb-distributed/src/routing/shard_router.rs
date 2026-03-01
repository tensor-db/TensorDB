//! Shard routing via consistent hashing.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

/// Consistent hash ring for routing keys to shards/nodes.
#[derive(Debug, Clone)]
pub struct ShardRouter {
    ring: BTreeMap<u64, u32>,
    replicas: usize,
    shard_count: u32,
}

impl ShardRouter {
    /// Create a new shard router with the given number of shards and virtual nodes.
    pub fn new(shard_count: u32, replicas: usize) -> Self {
        let mut ring = BTreeMap::new();
        for shard_id in 0..shard_count {
            for replica in 0..replicas {
                let hash = hash_key(&format!("shard-{shard_id}-{replica}"));
                ring.insert(hash, shard_id);
            }
        }
        Self {
            ring,
            replicas,
            shard_count,
        }
    }

    /// Route a key to its owning shard ID.
    pub fn route(&self, key: &[u8]) -> u32 {
        if self.ring.is_empty() {
            return 0;
        }
        let hash = hash_bytes(key);
        // Find first ring position >= hash
        if let Some((&_pos, &shard)) = self.ring.range(hash..).next() {
            shard
        } else {
            // Wrap around to first entry
            *self.ring.values().next().unwrap()
        }
    }

    /// Add a new shard to the ring.
    pub fn add_shard(&mut self, shard_id: u32) {
        for replica in 0..self.replicas {
            let hash = hash_key(&format!("shard-{shard_id}-{replica}"));
            self.ring.insert(hash, shard_id);
        }
        self.shard_count += 1;
    }

    /// Remove a shard from the ring.
    pub fn remove_shard(&mut self, shard_id: u32) {
        self.ring.retain(|_, &mut s| s != shard_id);
        self.shard_count = self.shard_count.saturating_sub(1);
    }

    /// Number of shards in the ring.
    pub fn shard_count(&self) -> u32 {
        self.shard_count
    }
}

fn hash_key(key: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistent_routing() {
        let router = ShardRouter::new(4, 128);
        let shard1 = router.route(b"key-1");
        let shard2 = router.route(b"key-1");
        assert_eq!(shard1, shard2, "same key should always route to same shard");
    }

    #[test]
    fn test_distribution() {
        let router = ShardRouter::new(4, 128);
        let mut counts = [0u32; 4];
        for i in 0..1000 {
            let key = format!("key-{i}");
            let shard = router.route(key.as_bytes());
            counts[shard as usize] += 1;
        }
        // Each shard should get roughly 250 keys (±100 for randomness)
        for (i, &count) in counts.iter().enumerate() {
            assert!(
                count > 100 && count < 500,
                "shard {i} got {count} keys, expected ~250"
            );
        }
    }

    #[test]
    fn test_add_remove_shard() {
        let mut router = ShardRouter::new(4, 128);
        assert_eq!(router.shard_count(), 4);

        router.add_shard(4);
        assert_eq!(router.shard_count(), 5);

        router.remove_shard(4);
        assert_eq!(router.shard_count(), 4);
    }
}
