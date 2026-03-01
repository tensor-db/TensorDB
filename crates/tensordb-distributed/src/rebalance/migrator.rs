//! ShardMigrator — handles online shard migration between nodes.
//!
//! Migration protocol:
//! 1. Source freezes shard (rejects writes, serves reads)
//! 2. Source streams snapshot to target via gRPC
//! 3. Source streams WAL tail until lag < threshold
//! 4. Brief freeze, final WAL entries, ownership switch
//! 5. Hash ring updated on all nodes

use crate::error::{DistributedError, Result};

/// State of a shard migration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationState {
    /// Planning the migration.
    Planning,
    /// Shard frozen on source, preparing snapshot.
    Freezing,
    /// Streaming snapshot data to target.
    SnapshotStreaming,
    /// Streaming WAL tail to catch up.
    CatchingUp,
    /// Final cutover: brief freeze + ownership switch.
    Cutover,
    /// Migration complete.
    Complete,
    /// Migration failed.
    Failed(String),
}

/// Tracks a single shard migration.
#[derive(Debug, Clone)]
pub struct Migration {
    pub shard_id: u32,
    pub source_node: String,
    pub target_node: String,
    pub state: MigrationState,
    pub bytes_transferred: u64,
    pub started_at_ms: u64,
}

/// Manages shard migrations.
pub struct ShardMigrator {
    active_migrations: parking_lot::RwLock<Vec<Migration>>,
}

impl ShardMigrator {
    /// Create a new migrator.
    pub fn new() -> Self {
        Self {
            active_migrations: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Plan a new migration.
    pub fn plan_migration(&self, shard_id: u32, source: String, target: String) -> Result<()> {
        let mut migrations = self.active_migrations.write();
        if migrations.iter().any(|m| m.shard_id == shard_id) {
            return Err(DistributedError::MigrationError(format!(
                "shard {shard_id} already has an active migration"
            )));
        }
        migrations.push(Migration {
            shard_id,
            source_node: source,
            target_node: target,
            state: MigrationState::Planning,
            bytes_transferred: 0,
            started_at_ms: current_time_ms(),
        });
        Ok(())
    }

    /// Advance migration to the next state.
    pub fn advance(&self, shard_id: u32) -> Result<MigrationState> {
        let mut migrations = self.active_migrations.write();
        let migration = migrations
            .iter_mut()
            .find(|m| m.shard_id == shard_id)
            .ok_or_else(|| {
                DistributedError::MigrationError(format!("no migration for shard {shard_id}"))
            })?;

        migration.state = match &migration.state {
            MigrationState::Planning => MigrationState::Freezing,
            MigrationState::Freezing => MigrationState::SnapshotStreaming,
            MigrationState::SnapshotStreaming => MigrationState::CatchingUp,
            MigrationState::CatchingUp => MigrationState::Cutover,
            MigrationState::Cutover => MigrationState::Complete,
            MigrationState::Complete => MigrationState::Complete,
            MigrationState::Failed(msg) => MigrationState::Failed(msg.clone()),
        };

        Ok(migration.state.clone())
    }

    /// Mark migration as failed.
    pub fn fail(&self, shard_id: u32, reason: String) -> Result<()> {
        let mut migrations = self.active_migrations.write();
        if let Some(migration) = migrations.iter_mut().find(|m| m.shard_id == shard_id) {
            migration.state = MigrationState::Failed(reason);
        }
        Ok(())
    }

    /// Remove completed migrations.
    pub fn cleanup_completed(&self) {
        let mut migrations = self.active_migrations.write();
        migrations.retain(|m| m.state != MigrationState::Complete);
    }

    /// Get active migration count.
    pub fn active_count(&self) -> usize {
        self.active_migrations.read().len()
    }
}

impl Default for ShardMigrator {
    fn default() -> Self {
        Self::new()
    }
}

fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_lifecycle() {
        let migrator = ShardMigrator::new();
        migrator
            .plan_migration(0, "node-a".into(), "node-b".into())
            .unwrap();
        assert_eq!(migrator.active_count(), 1);

        // Advance through all states
        assert_eq!(migrator.advance(0).unwrap(), MigrationState::Freezing);
        assert_eq!(
            migrator.advance(0).unwrap(),
            MigrationState::SnapshotStreaming
        );
        assert_eq!(migrator.advance(0).unwrap(), MigrationState::CatchingUp);
        assert_eq!(migrator.advance(0).unwrap(), MigrationState::Cutover);
        assert_eq!(migrator.advance(0).unwrap(), MigrationState::Complete);

        migrator.cleanup_completed();
        assert_eq!(migrator.active_count(), 0);
    }

    #[test]
    fn test_duplicate_migration() {
        let migrator = ShardMigrator::new();
        migrator.plan_migration(0, "a".into(), "b".into()).unwrap();
        assert!(migrator.plan_migration(0, "a".into(), "c".into()).is_err());
    }
}
