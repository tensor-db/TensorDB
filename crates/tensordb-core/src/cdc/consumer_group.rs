use std::collections::HashMap;

use crate::engine::db::Database;
use crate::error::{Result, TensorError};

/// Key prefix for consumer group metadata.
const GROUP_PREFIX: &str = "__cdc/group/";

/// A consumer group: multiple consumers sharing a change feed
/// with partition-based (shard-based) assignment.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConsumerGroup {
    /// Group ID.
    pub group_id: String,
    /// Members of this group.
    pub members: Vec<GroupMember>,
    /// Current partition assignments: member_id -> list of shard IDs.
    pub assignments: HashMap<String, Vec<usize>>,
    /// Generation ID (incremented on each rebalance).
    pub generation_id: u64,
    /// Prefix filter for the change feed.
    pub prefix: Vec<u8>,
    /// Total shard count.
    pub shard_count: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GroupMember {
    pub member_id: String,
    pub joined_at: u64,
}

/// Result of a partition assignment.
#[derive(Debug, Clone)]
pub struct PartitionAssignment {
    pub member_id: String,
    pub shard_ids: Vec<usize>,
    pub generation_id: u64,
}

impl ConsumerGroup {
    pub fn new(group_id: &str, prefix: &[u8], shard_count: usize) -> Self {
        ConsumerGroup {
            group_id: group_id.to_string(),
            members: Vec::new(),
            assignments: HashMap::new(),
            generation_id: 0,
            prefix: prefix.to_vec(),
            shard_count,
        }
    }

    fn storage_key(&self) -> String {
        format!("{}{}", GROUP_PREFIX, self.group_id)
    }

    /// Add a member to the group and trigger rebalance.
    pub fn join(&mut self, member_id: &str) -> PartitionAssignment {
        // Remove if already present
        self.members.retain(|m| m.member_id != member_id);

        self.members.push(GroupMember {
            member_id: member_id.to_string(),
            joined_at: current_timestamp_ms(),
        });

        self.rebalance();

        PartitionAssignment {
            member_id: member_id.to_string(),
            shard_ids: self.assignments.get(member_id).cloned().unwrap_or_default(),
            generation_id: self.generation_id,
        }
    }

    /// Remove a member from the group and trigger rebalance.
    pub fn leave(&mut self, member_id: &str) {
        self.members.retain(|m| m.member_id != member_id);
        self.assignments.remove(member_id);
        self.rebalance();
    }

    /// Rebalance: assign shards to members using round-robin.
    fn rebalance(&mut self) {
        self.generation_id += 1;
        self.assignments.clear();

        if self.members.is_empty() {
            return;
        }

        // Round-robin assignment of shards to members
        for shard_id in 0..self.shard_count {
            let member_idx = shard_id % self.members.len();
            let member_id = &self.members[member_idx].member_id;
            self.assignments
                .entry(member_id.clone())
                .or_default()
                .push(shard_id);
        }
    }

    /// Get the current assignment for a member.
    pub fn assignment_for(&self, member_id: &str) -> Option<PartitionAssignment> {
        self.assignments
            .get(member_id)
            .map(|shards| PartitionAssignment {
                member_id: member_id.to_string(),
                shard_ids: shards.clone(),
                generation_id: self.generation_id,
            })
    }
}

/// Manager for consumer groups, backed by TensorDB storage.
pub struct ConsumerGroupManager;

impl ConsumerGroupManager {
    /// Create or load a consumer group.
    pub fn get_or_create(db: &Database, group_id: &str, prefix: &[u8]) -> Result<ConsumerGroup> {
        let key = format!("{}{}", GROUP_PREFIX, group_id);
        match db.get(key.as_bytes(), None, None) {
            Ok(Some(bytes)) => {
                let group: ConsumerGroup = serde_json::from_slice(&bytes).map_err(|e| {
                    TensorError::SqlExec(format!("failed to parse consumer group: {e}"))
                })?;
                Ok(group)
            }
            Ok(None) => {
                let shard_count = db.shard_count();
                Ok(ConsumerGroup::new(group_id, prefix, shard_count))
            }
            Err(e) => Err(e),
        }
    }

    /// Persist a consumer group.
    pub fn save(db: &Database, group: &ConsumerGroup) -> Result<()> {
        let key = group.storage_key();
        let value = serde_json::to_vec(group)
            .map_err(|e| TensorError::SqlExec(format!("failed to serialize group: {e}")))?;
        db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        Ok(())
    }

    /// List all consumer groups.
    pub fn list(db: &Database) -> Result<Vec<ConsumerGroup>> {
        let prefix = GROUP_PREFIX.as_bytes();
        let rows = db.scan_prefix(prefix, None, None, None)?;
        let mut groups = Vec::new();
        for row in rows {
            if let Ok(group) = serde_json::from_slice::<ConsumerGroup>(&row.doc) {
                groups.push(group);
            }
        }
        Ok(groups)
    }

    /// Delete a consumer group.
    pub fn delete(db: &Database, group_id: &str) -> Result<()> {
        let key = format!("{}{}", GROUP_PREFIX, group_id);
        db.put(
            key.as_bytes(),
            b"{\"deleted\":true}".to_vec(),
            0,
            u64::MAX,
            None,
        )?;
        Ok(())
    }
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn test_consumer_group_creation() {
        let group = ConsumerGroup::new("my_group", b"table/orders/", 4);
        assert_eq!(group.group_id, "my_group");
        assert_eq!(group.shard_count, 4);
        assert!(group.members.is_empty());
    }

    #[test]
    fn test_single_member_gets_all_shards() {
        let mut group = ConsumerGroup::new("g1", b"", 4);
        let assignment = group.join("consumer_1");
        assert_eq!(assignment.shard_ids, vec![0, 1, 2, 3]);
        assert_eq!(assignment.generation_id, 1);
    }

    #[test]
    fn test_two_members_split_shards() {
        let mut group = ConsumerGroup::new("g1", b"", 4);
        group.join("consumer_1");
        let assignment2 = group.join("consumer_2");

        // After rebalance with 2 members and 4 shards:
        // consumer_1 gets shards 0, 2
        // consumer_2 gets shards 1, 3
        let a1 = group.assignment_for("consumer_1").unwrap();
        let a2 = group.assignment_for("consumer_2").unwrap();

        assert_eq!(a1.shard_ids.len(), 2);
        assert_eq!(a2.shard_ids.len(), 2);
        assert_eq!(a1.shard_ids, vec![0, 2]);
        assert_eq!(a2.shard_ids, vec![1, 3]);
        assert_eq!(assignment2.generation_id, 2);
    }

    #[test]
    fn test_member_leave_rebalances() {
        let mut group = ConsumerGroup::new("g1", b"", 4);
        group.join("c1");
        group.join("c2");
        assert_eq!(group.members.len(), 2);

        group.leave("c2");
        assert_eq!(group.members.len(), 1);

        let a = group.assignment_for("c1").unwrap();
        assert_eq!(a.shard_ids, vec![0, 1, 2, 3]); // c1 gets all shards
    }

    #[test]
    fn test_three_members_four_shards() {
        let mut group = ConsumerGroup::new("g1", b"", 4);
        group.join("a");
        group.join("b");
        group.join("c");

        // 4 shards / 3 members:
        // a: shard 0, 3
        // b: shard 1
        // c: shard 2
        let aa = group.assignment_for("a").unwrap();
        let ab = group.assignment_for("b").unwrap();
        let ac = group.assignment_for("c").unwrap();

        let total: usize = aa.shard_ids.len() + ab.shard_ids.len() + ac.shard_ids.len();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_group_serialization_roundtrip() {
        let mut group = ConsumerGroup::new("g1", b"prefix", 4);
        group.join("member_1");
        group.join("member_2");

        let json = serde_json::to_vec(&group).unwrap();
        let restored: ConsumerGroup = serde_json::from_slice(&json).unwrap();

        assert_eq!(restored.group_id, "g1");
        assert_eq!(restored.members.len(), 2);
        assert_eq!(restored.generation_id, group.generation_id);
    }

    #[test]
    fn test_group_persist_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), Config::default()).unwrap();

        let mut group = ConsumerGroup::new("test_group", b"", db.shard_count());
        group.join("consumer_a");
        ConsumerGroupManager::save(&db, &group).unwrap();

        let loaded = ConsumerGroupManager::get_or_create(&db, "test_group", b"").unwrap();
        assert_eq!(loaded.group_id, "test_group");
        assert_eq!(loaded.members.len(), 1);
    }
}
