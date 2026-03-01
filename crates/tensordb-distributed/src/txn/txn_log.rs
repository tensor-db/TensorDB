//! Durable transaction log for crash recovery of 2PC.
//!
//! Writes PREPARE/COMMIT/ABORT records to a WAL-like append-only log
//! so that in-doubt transactions can be resolved after a crash.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{DistributedError, Result};

/// A single entry in the transaction log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxnLogEntry {
    pub txn_id: String,
    pub action: TxnAction,
    pub epoch: u64,
    pub participants: Vec<String>,
    pub timestamp_ms: u64,
}

/// The action recorded in a log entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TxnAction {
    Prepare,
    Commit,
    Abort,
}

/// Append-only transaction log for durable 2PC state.
pub struct TxnLog {
    path: PathBuf,
    file: File,
}

impl TxnLog {
    /// Open or create a transaction log at the given path.
    pub fn open(path: PathBuf) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| DistributedError::TransactionError(format!("open txn log: {e}")))?;
        Ok(Self { path, file })
    }

    /// Append a log entry.
    pub fn append(&mut self, entry: &TxnLogEntry) -> Result<()> {
        let line = serde_json::to_string(entry)
            .map_err(|e| DistributedError::TransactionError(e.to_string()))?;
        writeln!(self.file, "{line}")
            .map_err(|e| DistributedError::TransactionError(format!("write txn log: {e}")))?;
        Ok(())
    }

    /// Read all entries from the log (for recovery).
    pub fn read_all(&self) -> Result<Vec<TxnLogEntry>> {
        let file = File::open(&self.path)
            .map_err(|e| DistributedError::TransactionError(format!("read txn log: {e}")))?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        for line in reader.lines() {
            let line =
                line.map_err(|e| DistributedError::TransactionError(format!("read line: {e}")))?;
            if line.trim().is_empty() {
                continue;
            }
            let entry: TxnLogEntry = serde_json::from_str(&line)
                .map_err(|e| DistributedError::TransactionError(format!("parse entry: {e}")))?;
            entries.push(entry);
        }
        Ok(entries)
    }

    /// Find transactions that were PREPARED but never COMMITTED or ABORTED.
    pub fn find_in_doubt(&self) -> Result<Vec<String>> {
        let entries = self.read_all()?;
        let mut state: std::collections::HashMap<String, TxnAction> =
            std::collections::HashMap::new();
        for entry in &entries {
            state.insert(entry.txn_id.clone(), entry.action.clone());
        }
        let in_doubt: Vec<String> = state
            .into_iter()
            .filter(|(_, action)| *action == TxnAction::Prepare)
            .map(|(txn_id, _)| txn_id)
            .collect();
        Ok(in_doubt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_txn_log_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("__dtxn.wal");

        let mut log = TxnLog::open(path).unwrap();
        log.append(&TxnLogEntry {
            txn_id: "txn-1".to_string(),
            action: TxnAction::Prepare,
            epoch: 100,
            participants: vec!["node-a".to_string()],
            timestamp_ms: 1000,
        })
        .unwrap();
        log.append(&TxnLogEntry {
            txn_id: "txn-1".to_string(),
            action: TxnAction::Commit,
            epoch: 100,
            participants: vec![],
            timestamp_ms: 1001,
        })
        .unwrap();

        let entries = log.read_all().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].action, TxnAction::Prepare);
        assert_eq!(entries[1].action, TxnAction::Commit);
    }

    #[test]
    fn test_in_doubt_detection() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("__dtxn.wal");

        let mut log = TxnLog::open(path).unwrap();
        log.append(&TxnLogEntry {
            txn_id: "txn-1".to_string(),
            action: TxnAction::Prepare,
            epoch: 100,
            participants: vec![],
            timestamp_ms: 1000,
        })
        .unwrap();
        // txn-1 is in-doubt (PREPARED but not COMMITTED)

        log.append(&TxnLogEntry {
            txn_id: "txn-2".to_string(),
            action: TxnAction::Prepare,
            epoch: 101,
            participants: vec![],
            timestamp_ms: 1001,
        })
        .unwrap();
        log.append(&TxnLogEntry {
            txn_id: "txn-2".to_string(),
            action: TxnAction::Commit,
            epoch: 101,
            participants: vec![],
            timestamp_ms: 1002,
        })
        .unwrap();

        let in_doubt = log.find_in_doubt().unwrap();
        assert_eq!(in_doubt, vec!["txn-1"]);
    }
}
