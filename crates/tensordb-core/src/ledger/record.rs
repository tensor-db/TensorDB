use crate::error::Result;
use crate::util::varint::{decode_bytes, decode_u64, encode_bytes, encode_u64};

#[derive(Debug, Clone, Default)]
pub struct FactMetadata {
    pub source_id: Option<u64>,
    pub schema_version: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct FactWrite {
    pub internal_key: Vec<u8>,
    pub fact: Vec<u8>,
    pub metadata: FactMetadata,
    /// Global epoch at write time (0 for legacy/non-transactional writes).
    pub epoch: u64,
    /// Transaction ID (0 for non-transactional writes).
    pub txn_id: u64,
}

impl FactWrite {
    pub fn encode_payload(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.internal_key.len() + self.fact.len() + 48);
        encode_bytes(&self.internal_key, &mut out);
        encode_bytes(&self.fact, &mut out);

        let mut flags = 0u8;
        if self.metadata.source_id.is_some() {
            flags |= 1;
        }
        if self.metadata.schema_version.is_some() {
            flags |= 2;
        }
        if self.epoch != 0 {
            flags |= 4;
        }
        if self.txn_id != 0 {
            flags |= 8;
        }
        out.push(flags);
        if let Some(v) = self.metadata.source_id {
            encode_u64(v, &mut out);
        }
        if let Some(v) = self.metadata.schema_version {
            encode_u64(v, &mut out);
        }
        if self.epoch != 0 {
            encode_u64(self.epoch, &mut out);
        }
        if self.txn_id != 0 {
            encode_u64(self.txn_id, &mut out);
        }
        out
    }

    pub fn decode_payload(payload: &[u8]) -> Result<Self> {
        let mut idx = 0usize;
        let internal_key = decode_bytes(payload, &mut idx)?;
        let fact = decode_bytes(payload, &mut idx)?;
        let flags = payload.get(idx).copied().unwrap_or(0);
        idx += usize::from(idx < payload.len());

        let source_id = if (flags & 1) != 0 {
            Some(decode_u64(payload, &mut idx)?)
        } else {
            None
        };
        let schema_version = if (flags & 2) != 0 {
            Some(decode_u64(payload, &mut idx)?)
        } else {
            None
        };
        let epoch = if (flags & 4) != 0 {
            decode_u64(payload, &mut idx)?
        } else {
            0
        };
        let txn_id = if (flags & 8) != 0 {
            decode_u64(payload, &mut idx)?
        } else {
            0
        };

        Ok(Self {
            internal_key,
            fact,
            metadata: FactMetadata {
                source_id,
                schema_version,
            },
            epoch,
            txn_id,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FactValue {
    pub doc: Vec<u8>,
    pub valid_from: u64,
    pub valid_to: u64,
}

impl FactValue {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.doc.len() + 24);
        out.extend_from_slice(&self.valid_from.to_be_bytes());
        out.extend_from_slice(&self.valid_to.to_be_bytes());
        encode_bytes(&self.doc, &mut out);
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            return Err(crate::error::TensorError::SstableFormat(
                "fact value too short".to_string(),
            ));
        }
        let mut idx = 0;
        let mut a = [0u8; 8];
        a.copy_from_slice(&bytes[idx..idx + 8]);
        idx += 8;
        let valid_from = u64::from_be_bytes(a);
        a.copy_from_slice(&bytes[idx..idx + 8]);
        idx += 8;
        let valid_to = u64::from_be_bytes(a);
        let doc = decode_bytes(bytes, &mut idx)?;
        Ok(Self {
            doc,
            valid_from,
            valid_to,
        })
    }
}
