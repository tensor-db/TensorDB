use std::fs;
use std::fs::File;
use std::path::PathBuf;

use crate::error::Result;
use crate::ledger::internal_key::decode_internal_key;
use crate::native_bridge::Hasher;
use crate::storage::sstable::{build_sstable, SsTableReader};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LevelFileInfo {
    pub file_name: String,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub file_size: u64,
}

pub struct LevelManager {
    shard_dir: PathBuf,
    levels: Vec<Vec<LevelFile>>,
    max_levels: usize,
    l1_target_bytes: u64,
    size_ratio: u64,
    sstable_max_bytes: u64,
}

struct LevelFile {
    info: LevelFileInfo,
    reader: SsTableReader,
}

impl LevelManager {
    pub fn new(
        shard_dir: PathBuf,
        max_levels: usize,
        l1_target_bytes: u64,
        size_ratio: u64,
        sstable_max_bytes: u64,
    ) -> Self {
        let mut levels = Vec::with_capacity(max_levels);
        for _ in 0..max_levels {
            levels.push(Vec::new());
        }
        Self {
            shard_dir,
            levels,
            max_levels,
            l1_target_bytes,
            size_ratio,
            sstable_max_bytes,
        }
    }

    pub fn from_sstables(
        shard_dir: PathBuf,
        readers: Vec<SsTableReader>,
        max_levels: usize,
        l1_target_bytes: u64,
        size_ratio: u64,
        sstable_max_bytes: u64,
    ) -> Self {
        let mut mgr = Self::new(
            shard_dir,
            max_levels,
            l1_target_bytes,
            size_ratio,
            sstable_max_bytes,
        );
        // All existing SSTables go to L0
        for reader in readers {
            let file_size = fs::metadata(&reader.path).map(|m| m.len()).unwrap_or(0);
            let (min_key, max_key) = extract_key_range(&reader);
            let file_name = reader
                .path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            let info = LevelFileInfo {
                file_name,
                min_key,
                max_key,
                file_size,
            };
            mgr.levels[0].push(LevelFile { info, reader });
        }
        mgr
    }

    pub fn l0_count(&self) -> usize {
        self.levels[0].len()
    }

    pub fn total_file_count(&self) -> usize {
        self.levels.iter().map(|l| l.len()).sum()
    }

    pub fn level_sizes(&self) -> Vec<u64> {
        self.levels
            .iter()
            .map(|level| level.iter().map(|f| f.info.file_size).sum())
            .collect()
    }

    pub fn all_readers(&self) -> Vec<&SsTableReader> {
        let mut out = Vec::new();
        for level in &self.levels {
            for f in level {
                out.push(&f.reader);
            }
        }
        out
    }

    pub fn l0_readers(&self) -> Vec<&SsTableReader> {
        self.levels[0].iter().map(|f| &f.reader).collect()
    }

    pub fn all_readers_owned(&self) -> Vec<&SsTableReader> {
        self.all_readers()
    }

    pub fn add_l0_file(&mut self, reader: SsTableReader) {
        let file_size = fs::metadata(&reader.path).map(|m| m.len()).unwrap_or(0);
        let (min_key, max_key) = extract_key_range(&reader);
        let file_name = reader
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        let info = LevelFileInfo {
            file_name,
            min_key,
            max_key,
            file_size,
        };
        self.levels[0].push(LevelFile { info, reader });
    }

    pub fn needs_compaction(&self, l0_threshold: usize) -> Option<CompactionTask> {
        // Check L0 → L1
        if self.levels[0].len() > l0_threshold {
            return Some(CompactionTask {
                source_level: 0,
                target_level: 1,
            });
        }

        // Check Ln → Ln+1 based on size budget
        for level in 1..self.max_levels.saturating_sub(1) {
            let level_size: u64 = self.levels[level].iter().map(|f| f.info.file_size).sum();
            let budget = self.level_budget(level);
            if level_size > budget {
                return Some(CompactionTask {
                    source_level: level,
                    target_level: level + 1,
                });
            }
        }

        None
    }

    fn level_budget(&self, level: usize) -> u64 {
        if level == 0 {
            return u64::MAX;
        }
        self.l1_target_bytes * self.size_ratio.pow((level - 1) as u32)
    }

    pub fn execute_compaction(
        &mut self,
        task: &CompactionTask,
        block_size: usize,
        bloom_bits_per_key: usize,
        hasher: &dyn Hasher,
        next_file_id: &mut u64,
    ) -> Result<CompactionResult> {
        let source_level = task.source_level;
        let target_level = task.target_level;

        // Collect all entries from source level
        let mut all_entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        for lf in &self.levels[source_level] {
            all_entries.extend(lf.reader.iter_all_entries()?);
        }

        // For L0→L1, also include overlapping target files
        if !self.levels[target_level].is_empty() {
            let source_min = all_entries
                .first()
                .map(|(k, _)| k.clone())
                .unwrap_or_default();
            let source_max = all_entries
                .last()
                .map(|(k, _)| k.clone())
                .unwrap_or_default();

            let mut target_idx_to_remove = Vec::new();
            for (idx, lf) in self.levels[target_level].iter().enumerate() {
                if overlaps(&lf.info.min_key, &lf.info.max_key, &source_min, &source_max) {
                    all_entries.extend(lf.reader.iter_all_entries()?);
                    target_idx_to_remove.push(idx);
                }
            }

            // Remove overlapping target files in reverse order
            for idx in target_idx_to_remove.into_iter().rev() {
                let removed = self.levels[target_level].remove(idx);
                let _ = fs::remove_file(&removed.reader.path);
            }
        }

        // Sort and deduplicate by keeping ALL temporal versions
        // (sort by key, entries with the same internal key are deduplicated)
        all_entries.sort_by(|a, b| a.0.cmp(&b.0));
        all_entries.dedup_by(|a, b| a.0 == b.0);

        // Write output SSTables, splitting at max file size
        let mut new_files = Vec::new();
        let mut chunk_start = 0;
        let mut chunk_bytes = 0usize;

        for i in 0..all_entries.len() {
            chunk_bytes += all_entries[i].0.len() + all_entries[i].1.len() + 16;
            if chunk_bytes >= self.sstable_max_bytes as usize || i == all_entries.len() - 1 {
                let chunk = &all_entries[chunk_start..=i];
                let file_name = format!("l{target_level}-{}.sst", *next_file_id);
                *next_file_id += 1;
                let sst_path = self.shard_dir.join(&file_name);
                build_sstable(
                    &sst_path,
                    chunk,
                    block_size,
                    bloom_bits_per_key,
                    hasher,
                )?;
                new_files.push(sst_path);
                chunk_start = i + 1;
                chunk_bytes = 0;
            }
        }

        // Fsync shard directory
        File::open(&self.shard_dir)?.sync_all()?;

        // Remove old source files
        let old_source_files: Vec<PathBuf> = self.levels[source_level]
            .iter()
            .map(|f| f.reader.path.clone())
            .collect();
        self.levels[source_level].clear();

        for path in &old_source_files {
            let _ = fs::remove_file(path);
        }

        // Open new readers and add to target level
        for path in &new_files {
            let reader = SsTableReader::open(path)?;
            let file_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let (min_key, max_key) = extract_key_range(&reader);
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();
            self.levels[target_level].push(LevelFile {
                info: LevelFileInfo {
                    file_name,
                    min_key,
                    max_key,
                    file_size,
                },
                reader,
            });
        }

        // Sort target level by min key for L1+
        if target_level >= 1 {
            self.levels[target_level].sort_by(|a, b| a.info.min_key.cmp(&b.info.min_key));
        }

        Ok(CompactionResult {
            files_removed: old_source_files.len(),
            files_created: new_files.len(),
        })
    }

    pub fn file_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for level in &self.levels {
            for f in level {
                names.push(f.info.file_name.clone());
            }
        }
        names
    }

    pub fn level_file_infos(&self) -> Vec<Vec<LevelFileInfo>> {
        self.levels
            .iter()
            .map(|level| level.iter().map(|f| f.info.clone()).collect())
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct CompactionTask {
    pub source_level: usize,
    pub target_level: usize,
}

#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub files_removed: usize,
    pub files_created: usize,
}

fn extract_key_range(reader: &SsTableReader) -> (Vec<u8>, Vec<u8>) {
    if reader.index.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // The index entries store internal keys as last_key of each block
    // We need the user key range
    let mut min_key = Vec::new();
    let mut max_key = Vec::new();

    if let Ok(entries) = reader.iter_all_entries() {
        if let Some((first, _)) = entries.first() {
            if let Ok(dk) = decode_internal_key(first) {
                min_key = dk.user_key;
            }
        }
        if let Some((last, _)) = entries.last() {
            if let Ok(dk) = decode_internal_key(last) {
                max_key = dk.user_key;
            }
        }
    }

    (min_key, max_key)
}

fn overlaps(a_min: &[u8], a_max: &[u8], b_min: &[u8], b_max: &[u8]) -> bool {
    a_min <= b_max && b_min <= a_max
}
