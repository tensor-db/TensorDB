use crate::error::{Result, SpectraError};

#[derive(Clone, Debug)]
pub struct Config {
    pub wal_fsync_every_n_records: usize,
    pub memtable_max_bytes: usize,
    pub sstable_block_bytes: usize,
    pub bloom_bits_per_key: usize,
    pub shard_count: usize,
    pub compaction_l0_threshold: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            wal_fsync_every_n_records: 128,
            memtable_max_bytes: 4 * 1024 * 1024,
            sstable_block_bytes: 16 * 1024,
            bloom_bits_per_key: 10,
            shard_count: 4,
            compaction_l0_threshold: 8,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.wal_fsync_every_n_records == 0 {
            return Err(SpectraError::Config(
                "wal_fsync_every_n_records must be > 0".to_string(),
            ));
        }
        if self.memtable_max_bytes < 1024 {
            return Err(SpectraError::Config(
                "memtable_max_bytes must be >= 1024".to_string(),
            ));
        }
        if self.sstable_block_bytes < 512 {
            return Err(SpectraError::Config(
                "sstable_block_bytes must be >= 512".to_string(),
            ));
        }
        if self.bloom_bits_per_key < 4 {
            return Err(SpectraError::Config(
                "bloom_bits_per_key must be >= 4".to_string(),
            ));
        }
        if self.shard_count == 0 {
            return Err(SpectraError::Config("shard_count must be > 0".to_string()));
        }
        if self.compaction_l0_threshold == 0 {
            return Err(SpectraError::Config(
                "compaction_l0_threshold must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}
