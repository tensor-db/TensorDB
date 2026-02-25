use std::collections::BTreeMap;
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
use crate::storage::cache::{BlockCache, IndexCache};
use crate::storage::levels::LevelManager;
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

#[derive(Debug, Clone)]
pub struct WriteBatchItem {
    pub user_key: Vec<u8>,
    pub doc: Vec<u8>,
    pub valid_from: u64,
    pub valid_to: u64,
    pub schema_version: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ChangeEvent {
    pub user_key: Vec<u8>,
    pub doc: Vec<u8>,
    pub commit_ts: u64,
    pub valid_from: u64,
    pub valid_to: u64,
}

pub enum ShardCommand {
    Put {
        user_key: Vec<u8>,
        doc: Vec<u8>,
        valid_from: u64,
        valid_to: u64,
        schema_version: Option<u64>,
        resp: Sender<Result<u64>>,
    },
    WriteBatch {
        entries: Vec<WriteBatchItem>,
        resp: Sender<Result<Vec<u64>>>,
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
    Subscribe {
        prefix: Vec<u8>,
        sender: crossbeam_channel::Sender<ChangeEvent>,
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

struct Subscriber {
    prefix: Vec<u8>,
    sender: crossbeam_channel::Sender<ChangeEvent>,
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
    levels: LevelManager,
    commit_counter: u64,
    stats: ShardStats,
    subscribers: Vec<Subscriber>,
    _block_cache: Arc<BlockCache>,
    _index_cache: Arc<IndexCache>,
}

impl ShardRuntime {
    pub fn open(
        shard_id: usize,
        root: &Path,
        config: Config,
        hasher: Arc<dyn Hasher + Send + Sync>,
        manifest: Arc<Mutex<Manifest>>,
        shard_state: ManifestShardState,
        block_cache: Arc<BlockCache>,
        index_cache: Arc<IndexCache>,
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

        let levels = LevelManager::from_sstables(
            shard_dir.clone(),
            sstables,
            config.compaction_max_levels,
            config.compaction_l1_target_bytes,
            config.compaction_size_ratio,
            config.sstable_max_file_bytes,
        );

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
            levels,
            commit_counter,
            stats: ShardStats::default(),
            subscribers: Vec::new(),
            _block_cache: block_cache,
            _index_cache: index_cache,
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
                ShardCommand::WriteBatch { entries, resp } => {
                    let res = self.handle_write_batch(entries);
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
                ShardCommand::Subscribe { prefix, sender } => {
                    self.subscribers.push(Subscriber { prefix, sender });
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

        // Notify subscribers
        self.emit_change(user_key, &write.fact, commit_ts, valid_from, valid_to);

        if self.memtable.approx_bytes() >= self.config.memtable_max_bytes {
            self.flush_active_memtable()?;
        } else {
            self.persist_manifest_watermark_in_memory();
        }

        Ok(commit_ts)
    }

    fn emit_change(
        &mut self,
        user_key: &[u8],
        _fact_bytes: &[u8],
        commit_ts: u64,
        valid_from: u64,
        valid_to: u64,
    ) {
        if self.subscribers.is_empty() {
            return;
        }
        // Remove dead subscribers and notify matching ones
        self.subscribers.retain(|sub| {
            if !user_key.starts_with(&sub.prefix) {
                return true; // Keep but don't notify
            }
            sub.sender
                .send(ChangeEvent {
                    user_key: user_key.to_vec(),
                    doc: Vec::new(), // We don't decode the fact bytes here for perf
                    commit_ts,
                    valid_from,
                    valid_to,
                })
                .is_ok() // Drop if receiver is gone
        });
    }

    fn handle_write_batch(&mut self, entries: Vec<WriteBatchItem>) -> Result<Vec<u64>> {
        let mut timestamps = Vec::with_capacity(entries.len());
        let mut wal_writes = Vec::with_capacity(entries.len());

        for entry in &entries {
            self.commit_counter = self.commit_counter.saturating_add(1);
            let commit_ts = self.commit_counter;
            timestamps.push(commit_ts);

            let internal_key = encode_internal_key(&entry.user_key, commit_ts, KIND_PUT);
            let fact = FactValue {
                doc: entry.doc.clone(),
                valid_from: entry.valid_from,
                valid_to: entry.valid_to,
            }
            .encode();

            wal_writes.push(FactWrite {
                internal_key: internal_key.clone(),
                fact: fact.clone(),
                metadata: FactMetadata {
                    source_id: None,
                    schema_version: entry.schema_version,
                },
            });

            self.memtable.insert(internal_key, fact);
        }

        // Single WAL frame for the batch
        self.wal.append_batch(&wal_writes)?;
        self.stats.puts += entries.len() as u64;

        if self.memtable.approx_bytes() >= self.config.memtable_max_bytes {
            self.flush_active_memtable()?;
        } else {
            self.persist_manifest_watermark_in_memory();
        }

        Ok(timestamps)
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
        for reader in self.levels.all_readers().iter().rev() {
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
        for reader in self.levels.all_readers() {
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
            shard.l0_files = self.levels.file_names();
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
        File::open(&self.shard_dir)?.sync_all()?;

        self.levels.add_l0_file(SsTableReader::open(&sst_path)?);
        self.stats.flushes += 1;
        self.persist_manifest_state()?;

        self.wal.truncate()?;

        // Check for compaction using leveled strategy
        self.maybe_compact()?;

        Ok(())
    }

    fn maybe_compact(&mut self) -> Result<()> {
        // Use leveled compaction strategy from LevelManager
        let mut next_file_id = {
            let manifest = self.manifest.lock();
            manifest.state.next_file_id
        };

        while let Some(task) = self
            .levels
            .needs_compaction(self.config.compaction_l0_threshold)
        {
            let result = self.levels.execute_compaction(
                &task,
                self.config.sstable_block_bytes,
                self.config.bloom_bits_per_key,
                self.hasher.as_ref(),
                &mut next_file_id,
            )?;

            self.stats.compactions += 1;
            tracing::info!(
                "shard {}: compacted L{}→L{}: removed {} files, created {}",
                self.shard_id,
                task.source_level,
                task.target_level,
                result.files_removed,
                result.files_created,
            );
        }

        // Update manifest with new file id
        {
            let mut manifest = self.manifest.lock();
            manifest.state.next_file_id = next_file_id;
        }
        self.persist_manifest_state()?;
        Ok(())
    }
}
