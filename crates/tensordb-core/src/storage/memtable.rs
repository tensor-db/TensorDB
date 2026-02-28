use std::collections::BTreeMap;
use std::ops::Bound;

use crate::error::Result;
use crate::ledger::internal_key::{decode_internal_key, user_prefix_bounds};
use crate::ledger::record::FactValue;

/// Visible entry from a prefix scan: user_key → (commit_ts, doc).
pub type PrefixVisibleMap = BTreeMap<Vec<u8>, (u64, Vec<u8>)>;

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

    /// Prefix-bounded scan: returns visible entries for user keys starting with `prefix`.
    /// Uses BTreeMap::range() for O(k log n) instead of O(n) full iteration.
    pub fn scan_prefix_visible(
        &self,
        prefix: &[u8],
        as_of: u64,
        valid_at: Option<u64>,
    ) -> Result<PrefixVisibleMap> {
        let mut best: PrefixVisibleMap = BTreeMap::new();

        let start = Bound::Included(prefix.to_vec());
        let end = match prefix_successor(prefix) {
            Some(succ) => Bound::Excluded(succ),
            None => Bound::Unbounded,
        };

        for (internal_key, value) in self.map.range::<Vec<u8>, _>((start, end)) {
            let decoded = decode_internal_key(internal_key)?;
            if decoded.commit_ts > as_of {
                continue;
            }
            if !decoded.user_key.starts_with(prefix) {
                continue;
            }

            let fact = FactValue::decode(value)?;
            if let Some(valid_ts) = valid_at {
                if !(fact.valid_from <= valid_ts && valid_ts < fact.valid_to) {
                    continue;
                }
            }

            match best.get(decoded.user_key.as_slice()) {
                Some((best_ts, _)) if *best_ts >= decoded.commit_ts => {}
                _ => {
                    best.insert(decoded.user_key, (decoded.commit_ts, fact.doc));
                }
            }
        }

        Ok(best)
    }
}

/// Compute the successor of a byte prefix for range queries.
/// Increments the last non-0xFF byte; truncates trailing 0xFF bytes.
/// Returns None if the prefix is all 0xFF (meaning no upper bound).
pub fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut succ = prefix.to_vec();
    while let Some(last) = succ.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return Some(succ);
        }
        succ.pop();
    }
    None
}
