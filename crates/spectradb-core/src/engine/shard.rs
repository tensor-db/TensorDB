use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;

use crate::config::Config;
use crate::error::Result;
use crate::ledger::internal_key::{decode_internal_key, encode_internal_key, KIND_PUT};
use crate::ledger::record::{FactMetadata, FactValue, FactWrite};
use crate::native_bridge::Hasher;
use crate::storage::compaction::compact_l0;
use crate::storage::manifest::{Manifest, ManifestShardState};
use crate::storage::memtable::Memtable;
use crate::storage::sstable::{build_sstable, SsTableReader};
use crate::storage::wal::Wal;

#[derive(Debug, Clone)]
pub struct GetResult {
    pub value: Option<Vec<u8>>,
    pub bloom_hit: Option<bool>,
    pub sstable_block: Option<usize>,
    pub commit_ts_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixScanResultRow {
    pub user_key: Vec<u8>,
    pub doc: Vec<u8>,
    pub commit_ts: u64,
}

#[derive(Debug)]
pub enum ShardCommand {
    Put {
        user_key: Vec<u8>,
        doc: Vec<u8>,
        valid_from: u64,
        valid_to: u64,
        schema_version: Option<u64>,
        resp: Sender<Result<u64>>,
    },
    Get {
        user_key: Vec<u8>,
        as_of: Option<u64>,
        valid_at: Option<u64>,
        resp: Sender<Result<GetResult>>,
    },
    ScanPrefix {
        user_key_prefix: Vec<u8>,
        as_of: Option<u64>,
        valid_at: Option<u64>,
        limit: Option<usize>,
        resp: Sender<Result<Vec<PrefixScanResultRow>>>,
    },
    Stats {
        resp: Sender<ShardStats>,
    },
    Shutdown,
}

#[derive(Debug, Clone, Default)]
pub struct ShardStats {
    pub puts: u64,
    pub gets: u64,
    pub flushes: u64,
    pub compactions: u64,
    pub bloom_negatives: u64,
    pub mmap_block_reads: u64,
}

pub struct ShardRuntime {
    shard_id: usize,
    shard_dir: PathBuf,
    config: Config,
    hasher: Arc<dyn Hasher + Send + Sync>,
    manifest: Arc<Mutex<Manifest>>,
    wal: Wal,
    memtable: Memtable,
    immutable_memtables: Vec<Memtable>,
    sstables: Vec<SsTableReader>,
    commit_counter: u64,
    stats: ShardStats,
}

impl ShardRuntime {
    pub fn open(
        shard_id: usize,
        root: &Path,
        config: Config,
        hasher: Arc<dyn Hasher + Send + Sync>,
        manifest: Arc<Mutex<Manifest>>,
        shard_state: ManifestShardState,
    ) -> Result<Self> {
        let shard_dir = root.join(format!("shard-{shard_id}"));
        fs::create_dir_all(&shard_dir)?;
        let wal_path = shard_dir.join(&shard_state.wal_file);
        if !wal_path.exists() {
            let _ = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&wal_path)?;
        }

        let mut commit_counter = shard_state.commit_ts_high_watermark;

        let mut sstables = Vec::new();
        for file in &shard_state.l0_files {
            let p = shard_dir.join(file);
            if p.exists() {
                sstables.push(SsTableReader::open(&p)?);
            }
        }

        let mut memtable = Memtable::new();
        for write in Wal::replay(&wal_path)? {
            if let Ok(decoded) = decode_internal_key(&write.internal_key) {
                if decoded.commit_ts > commit_counter {
                    commit_counter = decoded.commit_ts;
                }
            }
            memtable.insert(write.internal_key, write.fact);
        }

        let wal = Wal::open(wal_path, config.wal_fsync_every_n_records)?;

        Ok(Self {
            shard_id,
            shard_dir,
            config,
            hasher,
            manifest,
            wal,
            memtable,
            immutable_memtables: Vec::new(),
            sstables,
            commit_counter,
            stats: ShardStats::default(),
        })
    }

    pub fn run(mut self, rx: Receiver<ShardCommand>) {
        while let Ok(cmd) = rx.recv() {
            match cmd {
                ShardCommand::Put {
                    user_key,
                    doc,
                    valid_from,
                    valid_to,
                    schema_version,
                    resp,
                } => {
                    let res = self.handle_put(&user_key, doc, valid_from, valid_to, schema_version);
                    let _ = resp.send(res);
                }
                ShardCommand::Get {
                    user_key,
                    as_of,
                    valid_at,
                    resp,
                } => {
                    let res = self.handle_get(&user_key, as_of, valid_at);
                    let _ = resp.send(res);
                }
                ShardCommand::ScanPrefix {
                    user_key_prefix,
                    as_of,
                    valid_at,
                    limit,
                    resp,
                } => {
                    let res = self.handle_scan_prefix(&user_key_prefix, as_of, valid_at, limit);
                    let _ = resp.send(res);
                }
                ShardCommand::Stats { resp } => {
                    let _ = resp.send(self.stats.clone());
                }
                ShardCommand::Shutdown => {
                    let _ = self.wal.sync();
                    let _ = self.persist_manifest_state();
                    break;
                }
            }
        }
    }

    fn handle_put(
        &mut self,
        user_key: &[u8],
        doc: Vec<u8>,
        valid_from: u64,
        valid_to: u64,
        schema_version: Option<u64>,
    ) -> Result<u64> {
        self.commit_counter = self.commit_counter.saturating_add(1);
        let commit_ts = self.commit_counter;

        let internal_key = encode_internal_key(user_key, commit_ts, KIND_PUT);
        let fact = FactValue {
            doc,
            valid_from,
            valid_to,
        }
        .encode();
        let write = FactWrite {
            internal_key: internal_key.clone(),
            fact: fact.clone(),
            metadata: FactMetadata {
                source_id: None,
                schema_version,
            },
        };

        self.wal.append(&write)?;
        self.memtable.insert(internal_key, fact);
        self.stats.puts += 1;

        if self.memtable.approx_bytes() >= self.config.memtable_max_bytes {
            self.flush_active_memtable()?;
        } else {
            self.persist_manifest_watermark_in_memory();
        }

        Ok(commit_ts)
    }

    fn handle_get(
        &mut self,
        user_key: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
    ) -> Result<GetResult> {
        self.stats.gets += 1;
        let as_of_ts = as_of.unwrap_or(self.commit_counter);

        let mut best: Option<(u64, Vec<u8>)> = None;
        if let Some((ts, doc)) = self.memtable.visible_get(user_key, as_of_ts, valid_at)? {
            best = Some((ts, doc));
        }

        for imm in &self.immutable_memtables {
            if let Some((ts, doc)) = imm.visible_get(user_key, as_of_ts, valid_at)? {
                match &best {
                    Some((best_ts, _)) if *best_ts >= ts => {}
                    _ => best = Some((ts, doc)),
                }
            }
        }

        let mut explain_bloom = None;
        let mut explain_block = None;
        for reader in self.sstables.iter().rev() {
            let lookup = reader.get_visible(user_key, as_of_ts, valid_at, self.hasher.as_ref())?;
            if explain_bloom.is_none() {
                explain_bloom = Some(lookup.bloom_hit);
                explain_block = lookup.block_read;
            }
            if !lookup.bloom_hit {
                self.stats.bloom_negatives += 1;
            }
            if lookup.block_read.is_some() {
                self.stats.mmap_block_reads += 1;
            }
            if let (Some(ts), Some(doc)) = (lookup.commit_ts, lookup.value) {
                match &best {
                    Some((best_ts, _)) if *best_ts >= ts => {}
                    _ => best = Some((ts, doc)),
                }
            }
        }

        Ok(GetResult {
            value: best.map(|(_, d)| d),
            bloom_hit: explain_bloom,
            sstable_block: explain_block,
            commit_ts_used: as_of_ts,
        })
    }

    fn handle_scan_prefix(
        &mut self,
        user_key_prefix: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<PrefixScanResultRow>> {
        if matches!(limit, Some(0)) {
            return Ok(Vec::new());
        }

        // Stage 1 metric integration: treat prefix scans as read operations.
        self.stats.gets += 1;
        let as_of_ts = as_of.unwrap_or(self.commit_counter);

        let mut best: BTreeMap<Vec<u8>, (u64, Vec<u8>)> = BTreeMap::new();
        for (key, value) in self.memtable.iter() {
            Self::update_prefix_best(&mut best, key, value, user_key_prefix, as_of_ts, valid_at)?;
        }
        for imm in &self.immutable_memtables {
            for (key, value) in imm.iter() {
                Self::update_prefix_best(
                    &mut best,
                    key,
                    value,
                    user_key_prefix,
                    as_of_ts,
                    valid_at,
                )?;
            }
        }
        for reader in &self.sstables {
            for (key, value) in reader.iter_all_entries()? {
                Self::update_prefix_best(
                    &mut best,
                    &key,
                    &value,
                    user_key_prefix,
                    as_of_ts,
                    valid_at,
                )?;
            }
        }

        let mut out = best
            .into_iter()
            .map(|(user_key, (commit_ts, doc))| PrefixScanResultRow {
                user_key,
                doc,
                commit_ts,
            })
            .collect::<Vec<_>>();
        if let Some(limit) = limit {
            out.truncate(limit);
        }
        Ok(out)
    }

    fn update_prefix_best(
        best: &mut BTreeMap<Vec<u8>, (u64, Vec<u8>)>,
        internal_key: &[u8],
        value: &[u8],
        user_key_prefix: &[u8],
        as_of_ts: u64,
        valid_at: Option<u64>,
    ) -> Result<()> {
        let decoded = decode_internal_key(internal_key)?;
        if decoded.commit_ts > as_of_ts {
            return Ok(());
        }
        if !decoded.user_key.starts_with(user_key_prefix) {
            return Ok(());
        }

        let fact = FactValue::decode(value)?;
        if let Some(valid_ts) = valid_at {
            if !(fact.valid_from <= valid_ts && valid_ts < fact.valid_to) {
                return Ok(());
            }
        }

        match best.get(decoded.user_key.as_slice()) {
            Some((best_ts, _)) if *best_ts >= decoded.commit_ts => {}
            _ => {
                best.insert(decoded.user_key, (decoded.commit_ts, fact.doc));
            }
        }
        Ok(())
    }

    fn persist_manifest_watermark_in_memory(&self) {
        let mut manifest = self.manifest.lock();
        if let Some(shard) = manifest
            .state
            .shards
            .iter_mut()
            .find(|s| s.shard_id == self.shard_id)
        {
            shard.commit_ts_high_watermark = self.commit_counter;
        }
    }

    fn persist_manifest_state(&self) -> Result<()> {
        let mut manifest = self.manifest.lock();
        if let Some(shard) = manifest
            .state
            .shards
            .iter_mut()
            .find(|s| s.shard_id == self.shard_id)
        {
            shard.commit_ts_high_watermark = self.commit_counter;
            let mut files = Vec::new();
            for r in &self.sstables {
                if let Some(name) = r.path.file_name().and_then(|n| n.to_str()) {
                    files.push(name.to_string());
                }
            }
            shard.l0_files = files;
        }
        manifest.save()?;
        Ok(())
    }

    fn flush_active_memtable(&mut self) -> Result<()> {
        self.wal.sync()?;
        if self.memtable.is_empty() {
            return Ok(());
        }

        let old = std::mem::replace(&mut self.memtable, Memtable::new());
        let mut entries: Vec<(Vec<u8>, Vec<u8>)> =
            old.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let file_name = {
            let mut manifest = self.manifest.lock();
            let id = manifest.state.next_file_id;
            manifest.state.next_file_id += 1;
            format!("l0-{id}.sst")
        };

        let sst_path = self.shard_dir.join(&file_name);
        build_sstable(
            &sst_path,
            &entries,
            self.config.sstable_block_bytes,
            self.config.bloom_bits_per_key,
            self.hasher.as_ref(),
        )?;
        // Fsync shard directory so the new SSTable's directory entry is durable.
        File::open(&self.shard_dir)?.sync_all()?;

        self.sstables.push(SsTableReader::open(&sst_path)?);
        self.stats.flushes += 1;
        self.persist_manifest_state()?;

        // Truncate active WAL after successful flush+manifest persistence.
        self.wal.truncate()?;

        if self.sstables.len() > self.config.compaction_l0_threshold {
            self.compact_l0()?;
        }

        Ok(())
    }

    fn compact_l0(&mut self) -> Result<()> {
        if self.sstables.len() <= 1 {
            return Ok(());
        }
        let old_files: HashSet<PathBuf> = self.sstables.iter().map(|r| r.path.clone()).collect();

        let compact_file_name = {
            let mut manifest = self.manifest.lock();
            let id = manifest.state.next_file_id;
            manifest.state.next_file_id += 1;
            format!("compact-{id}.sst")
        };
        let compact_path = self.shard_dir.join(compact_file_name);

        compact_l0(
            &self.sstables,
            &compact_path,
            self.config.sstable_block_bytes,
            self.config.bloom_bits_per_key,
            self.hasher.as_ref(),
        )?;
        // Fsync shard directory so the compacted SSTable's directory entry is durable.
        File::open(&self.shard_dir)?.sync_all()?;

        let new_reader = SsTableReader::open(&compact_path)?;
        self.sstables = vec![new_reader];

        for old in old_files {
            let _ = fs::remove_file(old);
        }

        self.stats.compactions += 1;
        self.persist_manifest_state()?;
        Ok(())
    }
}
