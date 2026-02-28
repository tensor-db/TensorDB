use std::path::Path;

use crate::error::Result;
use crate::native_bridge::Hasher;
use crate::storage::sstable::{build_sstable, SsTableReader};

pub fn compact_l0(
    readers: &[&SsTableReader],
    out_path: impl AsRef<Path>,
    block_size: usize,
    bloom_bits_per_key: usize,
    hasher: &dyn Hasher,
) -> Result<()> {
    let mut all = Vec::new();
    for r in readers {
        all.extend(r.iter_all_entries()?);
    }

    all.sort_by(|a, b| a.0.cmp(&b.0));
    all.dedup_by(|a, b| a.0 == b.0);
    build_sstable(out_path, &all, block_size, bloom_bits_per_key, hasher)?;
    Ok(())
}
