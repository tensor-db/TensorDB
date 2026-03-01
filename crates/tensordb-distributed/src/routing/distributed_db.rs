//! DistributedDatabase — wraps a local Database with distributed routing.

use std::sync::Arc;

use crate::error::{DistributedError, Result};
use crate::node::ClusterNode;
use crate::routing::shard_router::ShardRouter;

/// A distributed database that routes operations to the correct node.
pub struct DistributedDatabase {
    /// The local database instance.
    local_db: Arc<tensordb_core::engine::db::Database>,
    /// Cluster node coordinator.
    node: Arc<ClusterNode>,
    /// Shard router.
    router: ShardRouter,
}

impl DistributedDatabase {
    /// Create a new distributed database.
    pub fn new(
        local_db: Arc<tensordb_core::engine::db::Database>,
        node: Arc<ClusterNode>,
        shard_count: u32,
    ) -> Self {
        Self {
            local_db,
            node,
            router: ShardRouter::new(shard_count, 128),
        }
    }

    /// Route a key to its owning shard, then execute locally or return the target node.
    pub fn put(&self, key: &[u8], value: Vec<u8>) -> Result<PutResult> {
        let shard_id = self.router.route(key);

        if self.node.owns_shard(shard_id) {
            let commit_ts = self
                .local_db
                .put(key, value, 0, u64::MAX, Some(1))
                .map_err(DistributedError::Core)?;
            Ok(PutResult::Local { commit_ts })
        } else {
            let owner = self
                .node
                .shard_owner(shard_id)
                .ok_or(DistributedError::ShardNotOwned {
                    shard_id,
                    key: String::from_utf8_lossy(key).to_string(),
                })?;
            let addr = self
                .node
                .peer_address(&owner)
                .ok_or(DistributedError::NodeNotFound(owner.clone()))?;
            Ok(PutResult::Forward {
                node_id: owner,
                address: addr,
            })
        }
    }

    /// Get a value, routing to the correct shard.
    pub fn get(&self, key: &[u8]) -> Result<GetResult> {
        let shard_id = self.router.route(key);

        if self.node.owns_shard(shard_id) {
            let result = self
                .local_db
                .get(key, None, None)
                .map_err(DistributedError::Core)?;
            Ok(GetResult::Local { value: result })
        } else {
            let owner = self
                .node
                .shard_owner(shard_id)
                .ok_or(DistributedError::ShardNotOwned {
                    shard_id,
                    key: String::from_utf8_lossy(key).to_string(),
                })?;
            let addr = self
                .node
                .peer_address(&owner)
                .ok_or(DistributedError::NodeNotFound(owner.clone()))?;
            Ok(GetResult::Forward {
                node_id: owner,
                address: addr,
            })
        }
    }

    /// Execute SQL locally.
    pub fn sql(&self, query: &str) -> Result<Vec<Vec<u8>>> {
        let result = self.local_db.sql(query).map_err(DistributedError::Core)?;
        match result {
            tensordb_core::sql::exec::SqlResult::Rows(rows) => Ok(rows),
            tensordb_core::sql::exec::SqlResult::Affected { message, .. } => {
                Ok(vec![message.into_bytes()])
            }
            tensordb_core::sql::exec::SqlResult::Explain(plan) => Ok(vec![plan.into_bytes()]),
        }
    }

    /// Get the local database reference.
    pub fn local_db(&self) -> &tensordb_core::engine::db::Database {
        &self.local_db
    }

    /// Get the shard router.
    pub fn router(&self) -> &ShardRouter {
        &self.router
    }
}

/// Result of a put operation.
#[derive(Debug)]
pub enum PutResult {
    /// Written locally.
    Local { commit_ts: u64 },
    /// Needs to be forwarded to another node.
    Forward { node_id: String, address: String },
}

/// Result of a get operation.
#[derive(Debug)]
pub enum GetResult {
    /// Found locally.
    Local { value: Option<Vec<u8>> },
    /// Needs to be forwarded to another node.
    Forward { node_id: String, address: String },
}
