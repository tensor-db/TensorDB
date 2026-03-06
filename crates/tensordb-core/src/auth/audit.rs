//! Append-only audit log stored under `__audit_log/` prefix.
//! Hash-chained for tamper detection: each event includes the hash of the previous event.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

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
    /// Hash of the previous event in the chain (hex-encoded SHA-256).
    #[serde(default)]
    pub prev_hash: String,
    /// Hash of this event (hex-encoded SHA-256).
    #[serde(default)]
    pub event_hash: String,
}

pub struct AuditLog {
    seq: AtomicU64,
    /// Last hash in the chain for tamper detection.
    last_hash: Mutex<[u8; 32]>,
}

/// Verification result from `verify_audit_log`.
#[derive(Debug)]
pub struct AuditVerification {
    pub verified: usize,
    pub broken_at: Option<u64>,
    pub total: usize,
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
            last_hash: Mutex::new([0u8; 32]),
        }
    }

    pub fn log(&self, db: &Database, kind: AuditEventKind, user: &str) -> Result<()> {
        let seq = self.seq.fetch_add(1, Ordering::SeqCst);
        let ts = current_timestamp_ms();

        // Get the previous hash for chain linking
        let prev_hash = {
            let guard = self.last_hash.lock().unwrap();
            hex_encode(&*guard)
        };

        // Build event without event_hash first
        let mut event = AuditEvent {
            timestamp_ms: ts,
            seq,
            kind,
            session_user: user.to_string(),
            prev_hash,
            event_hash: String::new(),
        };

        // Compute event hash over all fields except event_hash itself
        let hash_input = format!(
            "{}:{}:{}:{}:{}",
            event.timestamp_ms,
            event.seq,
            serde_json::to_string(&event.kind).unwrap_or_default(),
            event.session_user,
            event.prev_hash,
        );
        let event_hash_bytes = sha256_simple(hash_input.as_bytes());
        event.event_hash = hex_encode(&event_hash_bytes);

        // Update the chain
        {
            let mut guard = self.last_hash.lock().unwrap();
            *guard = event_hash_bytes;
        }

        let key = format!("{AUDIT_PREFIX}{ts:016}/{seq:010}").into_bytes();
        let value = serde_json::to_vec(&event)?;
        db.put(&key, value, 0, u64::MAX, None)?;
        Ok(())
    }

    /// Verify the integrity of the audit log hash chain.
    pub fn verify(&self, db: &Database) -> Result<AuditVerification> {
        let prefix = AUDIT_PREFIX.as_bytes();
        let rows = db.scan_prefix(prefix, None, None, None)?;
        let mut events: Vec<AuditEvent> = Vec::new();
        for row in rows {
            if row.doc.is_empty() {
                continue;
            }
            if let Ok(event) = serde_json::from_slice::<AuditEvent>(&row.doc) {
                events.push(event);
            }
        }

        // Sort by (timestamp_ms, seq) ascending
        events.sort_by(|a, b| a.timestamp_ms.cmp(&b.timestamp_ms).then(a.seq.cmp(&b.seq)));

        let total = events.len();
        let mut prev_hash = [0u8; 32];
        let mut verified = 0;

        for event in &events {
            // Verify prev_hash matches expected
            if event.prev_hash != hex_encode(&prev_hash) {
                return Ok(AuditVerification {
                    verified,
                    broken_at: Some(event.seq),
                    total,
                });
            }

            // Verify event_hash
            let hash_input = format!(
                "{}:{}:{}:{}:{}",
                event.timestamp_ms,
                event.seq,
                serde_json::to_string(&event.kind).unwrap_or_default(),
                event.session_user,
                event.prev_hash,
            );
            let computed_hash = sha256_simple(hash_input.as_bytes());
            if event.event_hash != hex_encode(&computed_hash) {
                return Ok(AuditVerification {
                    verified,
                    broken_at: Some(event.seq),
                    total,
                });
            }

            prev_hash = computed_hash;
            verified += 1;
        }

        Ok(AuditVerification {
            verified,
            broken_at: None,
            total,
        })
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

/// Simple SHA-256 implementation (no external dependency).
/// Uses the standard NIST algorithm.
fn sha256_simple(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Pre-processing: add padding
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    // Process each 64-byte block
    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = [0u8; 32];
    for (i, val) in h.iter().enumerate() {
        result[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}
