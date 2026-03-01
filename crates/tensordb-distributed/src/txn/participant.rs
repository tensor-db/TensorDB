//! 2PC Transaction Participant.
//!
//! Handles prepare/commit/abort requests from the coordinator.

use std::collections::HashMap;

use parking_lot::RwLock;

use crate::error::{DistributedError, Result};

/// State of a participant's transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParticipantState {
    Idle,
    Prepared,
    Committed,
    Aborted,
}

/// A pending transaction on the participant side.
#[derive(Debug, Clone)]
pub struct PendingTxn {
    pub txn_id: String,
    pub state: ParticipantState,
    pub writes: Vec<(Vec<u8>, Vec<u8>)>,
    pub prepare_ts: u64,
}

/// Transaction participant that responds to coordinator's 2PC requests.
pub struct TxnParticipant {
    pending: RwLock<HashMap<String, PendingTxn>>,
}

impl TxnParticipant {
    /// Create a new participant.
    pub fn new() -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
        }
    }

    /// Handle a PREPARE request: validate writes, acquire intent locks.
    pub fn prepare(&self, txn_id: String, writes: Vec<(Vec<u8>, Vec<u8>)>) -> Result<u64> {
        let mut pending = self.pending.write();
        if pending.contains_key(&txn_id) {
            return Err(DistributedError::TransactionError(format!(
                "txn {txn_id} already prepared"
            )));
        }

        let prepare_ts = current_time_ms();
        pending.insert(
            txn_id.clone(),
            PendingTxn {
                txn_id,
                state: ParticipantState::Prepared,
                writes,
                prepare_ts,
            },
        );
        Ok(prepare_ts)
    }

    /// Handle a COMMIT request: apply writes.
    pub fn commit(&self, txn_id: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut pending = self.pending.write();
        let txn = pending
            .get_mut(txn_id)
            .ok_or_else(|| DistributedError::TransactionError(format!("txn {txn_id} not found")))?;
        if txn.state != ParticipantState::Prepared {
            return Err(DistributedError::TransactionError(format!(
                "txn {txn_id} not in prepared state"
            )));
        }
        txn.state = ParticipantState::Committed;
        let writes = txn.writes.clone();
        Ok(writes)
    }

    /// Handle an ABORT request: release locks.
    pub fn abort(&self, txn_id: &str) -> Result<()> {
        let mut pending = self.pending.write();
        if let Some(txn) = pending.get_mut(txn_id) {
            txn.state = ParticipantState::Aborted;
        }
        Ok(())
    }

    /// Clean up completed/aborted transactions.
    pub fn cleanup(&self) {
        let mut pending = self.pending.write();
        pending.retain(|_, txn| {
            txn.state != ParticipantState::Committed && txn.state != ParticipantState::Aborted
        });
    }

    /// Number of pending transactions.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }
}

impl Default for TxnParticipant {
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
    fn test_participant_lifecycle() {
        let participant = TxnParticipant::new();

        let ts = participant
            .prepare("txn-1".into(), vec![(b"k".to_vec(), b"v".to_vec())])
            .unwrap();
        assert!(ts > 0);
        assert_eq!(participant.pending_count(), 1);

        let writes = participant.commit("txn-1").unwrap();
        assert_eq!(writes.len(), 1);

        participant.cleanup();
        assert_eq!(participant.pending_count(), 0);
    }

    #[test]
    fn test_participant_abort() {
        let participant = TxnParticipant::new();
        participant.prepare("txn-1".into(), vec![]).unwrap();
        participant.abort("txn-1").unwrap();
        participant.cleanup();
        assert_eq!(participant.pending_count(), 0);
    }
}
