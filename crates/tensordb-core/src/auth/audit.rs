//! Append-only audit log stored under `__audit_log/` prefix.

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::engine::db::Database;
use crate::error::Result;

const AUDIT_PREFIX: &str = "__audit_log/";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventKind {
    // Auth
    Login {
        username: String,
        success: bool,
    },
    UserCreated {
        username: String,
    },
    UserDisabled {
        username: String,
    },
    PasswordChanged {
        username: String,
    },
    RoleGranted {
        username: String,
        role: String,
    },
    RoleRevoked {
        username: String,
        role: String,
    },
    // DDL
    TableCreated {
        table: String,
    },
    TableDropped {
        table: String,
    },
    TableAltered {
        table: String,
        change: String,
    },
    IndexCreated {
        table: String,
        index: String,
    },
    IndexDropped {
        table: String,
        index: String,
    },
    ViewCreated {
        view: String,
    },
    ViewDropped {
        view: String,
    },
    // Security
    PolicyCreated {
        table: String,
        policy: String,
    },
    PolicyDropped {
        table: String,
        policy: String,
    },
    GdprErasure {
        table: String,
        key: String,
    },
    PrivilegeDenied {
        username: String,
        privilege: String,
        table: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp_ms: u64,
    pub seq: u64,
    pub kind: AuditEventKind,
    pub session_user: String,
}

pub struct AuditLog {
    seq: AtomicU64,
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

impl AuditLog {
    pub fn new() -> Self {
        Self {
            seq: AtomicU64::new(0),
        }
    }

    pub fn log(&self, db: &Database, kind: AuditEventKind, user: &str) -> Result<()> {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let ts = current_timestamp_ms();
        let event = AuditEvent {
            timestamp_ms: ts,
            seq,
            kind,
            session_user: user.to_string(),
        };
        let key = format!("{AUDIT_PREFIX}{ts:016}/{seq:010}").into_bytes();
        let value = serde_json::to_vec(&event)?;
        db.put(&key, value, 0, u64::MAX, None)?;
        Ok(())
    }

    pub fn query_recent(&self, db: &Database, limit: usize) -> Result<Vec<AuditEvent>> {
        let prefix = AUDIT_PREFIX.as_bytes();
        let rows = db.scan_prefix(prefix, None, None, Some(limit * 2))?;
        let mut events: Vec<AuditEvent> = Vec::new();
        for row in rows {
            if row.doc.is_empty() {
                continue;
            }
            if let Ok(event) = serde_json::from_slice::<AuditEvent>(&row.doc) {
                events.push(event);
            }
        }
        // Sort by timestamp descending, take the most recent
        events.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));
        events.truncate(limit);
        Ok(events)
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}
