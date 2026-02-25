use std::collections::BTreeMap;

use crate::error::Result;
use crate::ledger::internal_key::{decode_internal_key, user_prefix_bounds};
use crate::ledger::record::FactValue;

#[derive(Clone, Default)]
pub struct Memtable {
    map: BTreeMap<Vec<u8>, Vec<u8>>,
    approx_bytes: usize,
}

impl Memtable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, internal_key: Vec<u8>, value: Vec<u8>) {
        let new_size = internal_key.len() + value.len();
        if let Some(old_val) = self.map.get(&internal_key) {
            self.approx_bytes = self
                .approx_bytes
                .saturating_sub(internal_key.len() + old_val.len());
        }
        self.approx_bytes += new_size;
        self.map.insert(internal_key, value);
    }

    pub fn approx_bytes(&self) -> usize {
        self.approx_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn drain(self) -> BTreeMap<Vec<u8>, Vec<u8>> {
        self.map
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &Vec<u8>)> {
        self.map.iter()
    }

    pub fn visible_get(
        &self,
        user_key: &[u8],
        as_of: u64,
        valid_at: Option<u64>,
    ) -> Result<Option<(u64, Vec<u8>)>> {
        let (start, end) = user_prefix_bounds(user_key);
        for (k, v) in self.map.range(start..end).rev() {
            let decoded = decode_internal_key(k)?;
            if decoded.commit_ts > as_of {
                continue;
            }

            let fact = FactValue::decode(v)?;
            if let Some(valid_ts) = valid_at {
                if !(fact.valid_from <= valid_ts && valid_ts < fact.valid_to) {
                    continue;
                }
            }
            return Ok(Some((decoded.commit_ts, fact.doc)));
        }
        Ok(None)
    }
}
