//! Distributed error types for TensorDB.

use std::fmt;

/// Errors that can occur in the distributed layer.
#[derive(Debug)]
pub enum DistributedError {
    /// Error from the core database.
    Core(tensordb_core::error::TensorError),
    /// Transport/network error.
    Transport(String),
    /// Shard routing error (key doesn't map to any owned shard).
    ShardNotOwned { shard_id: u32, key: String },
    /// Shard is frozen for migration, rejecting writes.
    ShardFrozen(u32),
    /// 2PC transaction error.
    TransactionError(String),
    /// Cluster membership error.
    ClusterError(String),
    /// Rebalance / migration error.
    MigrationError(String),
    /// Node not found.
    NodeNotFound(String),
    /// Timeout.
    Timeout(String),
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(e) => write!(f, "core: {e}"),
            Self::Transport(e) => write!(f, "transport: {e}"),
            Self::ShardNotOwned { shard_id, key } => {
                write!(f, "shard {shard_id} not owned (key: {key})")
            }
            Self::ShardFrozen(id) => write!(f, "shard {id} frozen for migration"),
            Self::TransactionError(e) => write!(f, "transaction: {e}"),
            Self::ClusterError(e) => write!(f, "cluster: {e}"),
            Self::MigrationError(e) => write!(f, "migration: {e}"),
            Self::NodeNotFound(id) => write!(f, "node not found: {id}"),
            Self::Timeout(e) => write!(f, "timeout: {e}"),
        }
    }
}

impl std::error::Error for DistributedError {}

impl From<tensordb_core::error::TensorError> for DistributedError {
    fn from(e: tensordb_core::error::TensorError) -> Self {
        Self::Core(e)
    }
}

/// Result alias for distributed operations.
pub type Result<T> = std::result::Result<T, DistributedError>;
