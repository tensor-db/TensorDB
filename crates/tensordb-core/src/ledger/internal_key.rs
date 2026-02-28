use crate::error::{Result, TensorError};

pub const KIND_PUT: u8 = 0;
/// Marker record indicating a transaction boundary. Written to WAL after all
/// transaction writes to enable crash-safe recovery: incomplete transactions
/// (those without a trailing TXN_COMMIT) can be identified and rolled back.
pub const KIND_TXN_COMMIT: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedInternalKey {
    pub user_key: Vec<u8>,
    pub commit_ts: u64,
    pub kind: u8,
}

pub fn encode_internal_key(user_key: &[u8], commit_ts: u64, kind: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(user_key.len() + 10);
    out.extend_from_slice(user_key);
    out.push(0);
    out.extend_from_slice(&commit_ts.to_be_bytes());
    out.push(kind);
    out
}

pub fn decode_internal_key(key: &[u8]) -> Result<DecodedInternalKey> {
    if key.len() < 10 {
        return Err(TensorError::SstableFormat(
            "internal key too short".to_string(),
        ));
    }
    let split = key.len() - 10;
    if key[split] != 0 {
        return Err(TensorError::SstableFormat(
            "internal key suffix malformed".to_string(),
        ));
    }
    let mut ts_bytes = [0u8; 8];
    ts_bytes.copy_from_slice(&key[split + 1..split + 9]);
    Ok(DecodedInternalKey {
        user_key: key[..split].to_vec(),
        commit_ts: u64::from_be_bytes(ts_bytes),
        kind: key[split + 9],
    })
}

pub fn user_prefix_bounds(user_key: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut start = user_key.to_vec();
    start.push(0);
    let mut end = user_key.to_vec();
    end.push(1);
    (start, end)
}
