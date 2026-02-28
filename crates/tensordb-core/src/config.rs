use crate::error::{Result, TensorError};

#[derive(Clone, Debug)]
pub struct Config {
    pub wal_fsync_every_n_records: usize,
    pub memtable_max_bytes: usize,
    pub sstable_block_bytes: usize,
    pub bloom_bits_per_key: usize,
    pub shard_count: usize,
    pub ai_auto_insights: bool,
    pub ai_batch_window_ms: u64,
    pub ai_batch_max_events: usize,
    pub compaction_l0_threshold: usize,
    // Phase 2 additions
    pub compaction_l1_target_bytes: u64,
    pub compaction_size_ratio: u64,
    pub compaction_max_levels: usize,
    pub sstable_max_file_bytes: u64,
    pub block_cache_bytes: usize,
    pub index_cache_entries: usize,
    // AI-native core engine flags
    pub ai_inline_risk_assessment: bool,
    pub ai_annotate_reads: bool,
    pub ai_compaction_advisor: bool,
    pub ai_cache_advisor: bool,
    pub ai_access_stats_size: usize,
    // Fast write path
    pub fast_write_enabled: bool,
    pub fast_write_wal_batch_interval_us: u64,
    // LLM (natural language → SQL)
    /// Path to GGUF model file. If None, auto-discovers in `<db_root>/.local/models/`.
    pub llm_model_path: Option<String>,
    /// Max tokens for LLM generation (default: 256).
    pub llm_max_tokens: usize,
    // Encryption at rest
    /// Passphrase for AES-256-GCM encryption. If set, SSTable blocks and WAL frames
    /// are encrypted transparently. Requires the `encryption` feature flag.
    pub encryption_passphrase: Option<String>,
    /// Path to a key file (32 raw bytes or 64 hex chars). Alternative to passphrase.
    pub encryption_key_file: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            wal_fsync_every_n_records: 128,
            memtable_max_bytes: 4 * 1024 * 1024,
            sstable_block_bytes: 16 * 1024,
            bloom_bits_per_key: 10,
            shard_count: 4,
            ai_auto_insights: false,
            ai_batch_window_ms: 20,
            ai_batch_max_events: 16,
            compaction_l0_threshold: 8,
            compaction_l1_target_bytes: 10 * 1024 * 1024,
            compaction_size_ratio: 10,
            compaction_max_levels: 7,
            sstable_max_file_bytes: 64 * 1024 * 1024,
            block_cache_bytes: 32 * 1024 * 1024,
            index_cache_entries: 1024,
            ai_inline_risk_assessment: false,
            ai_annotate_reads: false,
            ai_compaction_advisor: false,
            ai_cache_advisor: false,
            ai_access_stats_size: 1024,
            fast_write_enabled: true,
            fast_write_wal_batch_interval_us: 1000,
            llm_model_path: None,
            llm_max_tokens: 256,
            encryption_passphrase: None,
            encryption_key_file: None,
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.wal_fsync_every_n_records == 0 {
            return Err(TensorError::Config(
                "wal_fsync_every_n_records must be > 0".to_string(),
            ));
        }
        if self.memtable_max_bytes < 1024 {
            return Err(TensorError::Config(
                "memtable_max_bytes must be >= 1024".to_string(),
            ));
        }
        if self.sstable_block_bytes < 512 {
            return Err(TensorError::Config(
                "sstable_block_bytes must be >= 512".to_string(),
            ));
        }
        if self.bloom_bits_per_key < 4 {
            return Err(TensorError::Config(
                "bloom_bits_per_key must be >= 4".to_string(),
            ));
        }
        if self.shard_count == 0 {
            return Err(TensorError::Config("shard_count must be > 0".to_string()));
        }
        if self.compaction_l0_threshold == 0 {
            return Err(TensorError::Config(
                "compaction_l0_threshold must be > 0".to_string(),
            ));
        }
        if self.compaction_max_levels < 2 {
            return Err(TensorError::Config(
                "compaction_max_levels must be >= 2".to_string(),
            ));
        }
        if self.ai_auto_insights && self.ai_batch_window_ms == 0 {
            return Err(TensorError::Config(
                "ai_batch_window_ms must be > 0".to_string(),
            ));
        }
        if self.ai_auto_insights && self.ai_batch_max_events == 0 {
            return Err(TensorError::Config(
                "ai_batch_max_events must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}
