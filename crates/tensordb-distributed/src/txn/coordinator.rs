//! 2PC Transaction Coordinator.
//!
//! Coordinates distributed transactions across multiple nodes using two-phase commit.
//! Integrates with TensorDB's EOAC epoch system.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::error::{DistributedError, Result};

/// State of a distributed transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TxnState {
    Active,
    Preparing,
    Prepared,
    Committing,
    Committed,
    Aborting,
    Aborted,
}

/// A distributed transaction tracked by the coordinator.
#[derive(Debug, Clone)]
pub struct DistributedTxn {
    pub txn_id: String,
    pub state: TxnState,
    pub participants: Vec<String>,       // node IDs
    pub writes: Vec<(Vec<u8>, Vec<u8>)>, // (key, value) pairs
    pub epoch: u64,
    pub created_at_ms: u64,
}

/// 2PC coordinator that manages distributed transactions.
pub struct TxnCoordinator {
    active_txns: RwLock<HashMap<String, DistributedTxn>>,
    prepare_timeout_ms: u64,
}

impl TxnCoordinator {
    /// Create a new coordinator with the given prepare timeout.
    pub fn new(prepare_timeout_ms: u64) -> Arc<Self> {
        Arc::new(Self {
            active_txns: RwLock::new(HashMap::new()),
            prepare_timeout_ms,
        })
    }

    /// Begin a new distributed transaction.
    pub fn begin(&self, txn_id: String, participants: Vec<String>) -> Result<()> {
        let mut txns = self.active_txns.write();
        if txns.contains_key(&txn_id) {
            return Err(DistributedError::TransactionError(format!(
                "transaction {txn_id} already exists"
            )));
        }
        txns.insert(
            txn_id.clone(),
            DistributedTxn {
                txn_id,
                state: TxnState::Active,
                participants,
                writes: Vec::new(),
                epoch: 0,
                created_at_ms: current_time_ms(),
            },
        );
        Ok(())
    }

    /// Add a write to the transaction's buffer.
    pub fn buffer_write(&self, txn_id: &str, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let mut txns = self.active_txns.write();
        let txn = txns
            .get_mut(txn_id)
            .ok_or_else(|| DistributedError::TransactionError(format!("txn {txn_id} not found")))?;
        if txn.state != TxnState::Active {
            return Err(DistributedError::TransactionError(format!(
                "txn {txn_id} is in state {:?}, cannot buffer writes",
                txn.state
            )));
        }
        txn.writes.push((key, value));
        Ok(())
    }

    /// Transition to preparing state and return the transaction data.
    pub fn start_prepare(&self, txn_id: &str) -> Result<DistributedTxn> {
        let mut txns = self.active_txns.write();
        let txn = txns
            .get_mut(txn_id)
            .ok_or_else(|| DistributedError::TransactionError(format!("txn {txn_id} not found")))?;
        if txn.state != TxnState::Active {
            return Err(DistributedError::TransactionError(format!(
                "txn {txn_id} cannot prepare from state {:?}",
                txn.state
            )));
        }
        txn.state = TxnState::Preparing;
        Ok(txn.clone())
    }

    /// Mark transaction as prepared (all participants responded PREPARED).
    pub fn mark_prepared(&self, txn_id: &str, epoch: u64) -> Result<()> {
        let mut txns = self.active_txns.write();
        let txn = txns
            .get_mut(txn_id)
            .ok_or_else(|| DistributedError::TransactionError(format!("txn {txn_id} not found")))?;
        txn.state = TxnState::Prepared;
        txn.epoch = epoch;
        Ok(())
    }

    /// Mark transaction as committed.
    pub fn mark_committed(&self, txn_id: &str) -> Result<()> {
        let mut txns = self.active_txns.write();
        let txn = txns
            .get_mut(txn_id)
            .ok_or_else(|| DistributedError::TransactionError(format!("txn {txn_id} not found")))?;
        txn.state = TxnState::Committed;
        Ok(())
    }

    /// Mark transaction as aborted.
    pub fn mark_aborted(&self, txn_id: &str) -> Result<()> {
        let mut txns = self.active_txns.write();
        let txn = txns
            .get_mut(txn_id)
            .ok_or_else(|| DistributedError::TransactionError(format!("txn {txn_id} not found")))?;
        txn.state = TxnState::Aborted;
        Ok(())
    }

    /// Clean up completed transactions.
    pub fn cleanup_completed(&self) {
        let mut txns = self.active_txns.write();
        txns.retain(|_, txn| txn.state != TxnState::Committed && txn.state != TxnState::Aborted);
    }

    /// Get transactions that have timed out during prepare phase.
    pub fn find_timed_out(&self) -> Vec<String> {
        let now = current_time_ms();
        let txns = self.active_txns.read();
        txns.values()
            .filter(|txn| {
                txn.state == TxnState::Preparing
                    && now - txn.created_at_ms > self.prepare_timeout_ms
            })
            .map(|txn| txn.txn_id.clone())
            .collect()
    }

    /// Number of active transactions.
    pub fn active_count(&self) -> usize {
        self.active_txns.read().len()
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
    fn test_txn_lifecycle() {
        let coord = TxnCoordinator::new(30_000);
        coord
            .begin("txn-1".into(), vec!["node-a".into(), "node-b".into()])
            .unwrap();
        assert_eq!(coord.active_count(), 1);

        coord
            .buffer_write("txn-1", b"key".to_vec(), b"val".to_vec())
            .unwrap();

        let txn = coord.start_prepare("txn-1").unwrap();
        assert_eq!(txn.writes.len(), 1);

        coord.mark_prepared("txn-1", 100).unwrap();
        coord.mark_committed("txn-1").unwrap();

        coord.cleanup_completed();
        assert_eq!(coord.active_count(), 0);
    }

    #[test]
    fn test_duplicate_begin() {
        let coord = TxnCoordinator::new(30_000);
        coord.begin("txn-1".into(), vec![]).unwrap();
        assert!(coord.begin("txn-1".into(), vec![]).is_err());
    }

    #[test]
    fn test_abort() {
        let coord = TxnCoordinator::new(30_000);
        coord.begin("txn-1".into(), vec![]).unwrap();
        coord.mark_aborted("txn-1").unwrap();
        coord.cleanup_completed();
        assert_eq!(coord.active_count(), 0);
    }
}
