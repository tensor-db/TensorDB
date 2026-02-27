use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};
use parking_lot::{Mutex, RwLock};

use crate::ai::{hex_encode, is_internal_ai_key, quick_risk_score};
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
    pub ai_risk_score: Option<f64>,
    pub ai_tags: Option<Vec<String>>,
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

/// Shard commands — writes and control flow go through the channel.
/// Reads bypass the channel entirely via ShardReadHandle.
/// Fast writes bypass the channel entirely via FastWritePath.
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
    /// Request from the fast write path to flush the active memtable.
    FlushRequest,
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

// ---------------------------------------------------------------------------
// ShardShared — read state shared between writer thread and concurrent readers
// ---------------------------------------------------------------------------

/// Shared read state accessible by both the shard writer thread and concurrent readers.
/// The writer takes brief write locks for memtable inserts, flush, and compaction.
/// Readers take read locks for get() and scan_prefix() — no channel round-trip.
/// FastWritePath directly accesses active_memtable and commit_counter.
pub struct ShardShared {
    pub(crate) active_memtable: RwLock<Memtable>,
    pub(crate) immutable_memtables: RwLock<Vec<Memtable>>,
    pub(crate) levels: RwLock<LevelManager>,
    pub(crate) commit_counter: AtomicU64,
    pub(crate) hasher: Arc<dyn Hasher + Send + Sync>,
    pub(crate) block_cache: Arc<BlockCache>,
    pub(crate) index_cache: Arc<IndexCache>,
    // Atomic read stats
    pub(crate) stats_gets: AtomicU64,
    pub(crate) stats_bloom_negatives: AtomicU64,
    pub(crate) stats_mmap_block_reads: AtomicU64,
    // AI config
    pub(crate) ai_annotate_reads: bool,
    /// True when the shard has active change-feed subscribers.
    /// The fast write path checks this and falls back to the channel path
    /// so that change events are properly emitted.
    pub(crate) has_subscribers: AtomicBool,
}

// ---------------------------------------------------------------------------
// ShardReadHandle — lightweight, cloneable handle for direct reads
// ---------------------------------------------------------------------------

/// A lightweight, cloneable handle for direct reads, bypassing the shard actor channel.
#[derive(Clone)]
pub struct ShardReadHandle {
    shared: Arc<ShardShared>,
}

impl ShardReadHandle {
    /// Direct point read — no channel, no context switch.
    pub fn get(
        &self,
        user_key: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
    ) -> Result<GetResult> {
        self.shared.stats_gets.fetch_add(1, Ordering::Relaxed);
        let as_of_ts = as_of.unwrap_or_else(|| self.shared.commit_counter.load(Ordering::Acquire));

        let mut best: Option<(u64, Vec<u8>)> = None;

        {
            let memtable = self.shared.active_memtable.read();
            if let Some((ts, doc)) = memtable.visible_get(user_key, as_of_ts, valid_at)? {
                best = Some((ts, doc));
            }
        }

        {
            let imm_tables = self.shared.immutable_memtables.read();
            for imm in imm_tables.iter() {
                if let Some((ts, doc)) = imm.visible_get(user_key, as_of_ts, valid_at)? {
                    match &best {
                        Some((best_ts, _)) if *best_ts >= ts => {}
                        _ => best = Some((ts, doc)),
                    }
                }
            }
        }

        let mut explain_bloom = None;
        let mut explain_block = None;
        {
            let levels = self.shared.levels.read();
            for reader in levels.all_readers().iter().rev() {
                let lookup = reader.get_visible(
                    user_key,
                    as_of_ts,
                    valid_at,
                    self.shared.hasher.as_ref(),
                    Some(&*self.shared.block_cache),
                    Some(&*self.shared.index_cache),
                )?;
                if explain_bloom.is_none() {
                    explain_bloom = Some(lookup.bloom_hit);
                    explain_block = lookup.block_read;
                }
                if !lookup.bloom_hit {
                    self.shared
                        .stats_bloom_negatives
                        .fetch_add(1, Ordering::Relaxed);
                }
                if lookup.block_read.is_some() {
                    self.shared
                        .stats_mmap_block_reads
                        .fetch_add(1, Ordering::Relaxed);
                }
                if let (Some(ts), Some(doc)) = (lookup.commit_ts, lookup.value) {
                    match &best {
                        Some((best_ts, _)) if *best_ts >= ts => {}
                        _ => best = Some((ts, doc)),
                    }
                }
            }
        }

        let (ai_risk_score, ai_tags) =
            if self.shared.ai_annotate_reads && !is_internal_ai_key(user_key) {
                self.lookup_ai_annotations(user_key)
            } else {
                (None, None)
            };

        Ok(GetResult {
            value: best.map(|(_, d)| d),
            bloom_hit: explain_bloom,
            sstable_block: explain_block,
            commit_ts_used: as_of_ts,
            ai_risk_score,
            ai_tags,
        })
    }

    /// Direct prefix scan — no channel, no context switch.
    pub fn scan_prefix(
        &self,
        user_key_prefix: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<PrefixScanResultRow>> {
        if matches!(limit, Some(0)) {
            return Ok(Vec::new());
        }

        self.shared.stats_gets.fetch_add(1, Ordering::Relaxed);
        let as_of_ts = as_of.unwrap_or_else(|| self.shared.commit_counter.load(Ordering::Acquire));

        let mut best: BTreeMap<Vec<u8>, (u64, Vec<u8>)> = BTreeMap::new();

        {
            let memtable = self.shared.active_memtable.read();
            merge_prefix_visible(&memtable, user_key_prefix, as_of_ts, valid_at, &mut best)?;
        }

        {
            let imm_tables = self.shared.immutable_memtables.read();
            for imm in imm_tables.iter() {
                merge_prefix_visible(imm, user_key_prefix, as_of_ts, valid_at, &mut best)?;
            }
        }

        {
            let levels = self.shared.levels.read();
            for reader in levels.all_readers() {
                for (key, value) in reader.iter_all_entries()? {
                    update_prefix_best(
                        &mut best,
                        &key,
                        &value,
                        user_key_prefix,
                        as_of_ts,
                        valid_at,
                    )?;
                }
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

    fn lookup_ai_annotations(&self, user_key: &[u8]) -> (Option<f64>, Option<Vec<String>>) {
        let risk_prefix = format!("__ai/risk/{}/", hex_encode(user_key)).into_bytes();
        let memtable = self.shared.active_memtable.read();
        if let Ok(results) = memtable.scan_prefix_visible(&risk_prefix, u64::MAX, None) {
            if let Some((_, (_, doc))) = results.iter().next_back() {
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(doc) {
                    let risk = v.get("risk").and_then(|r| r.as_f64());
                    return (risk, None);
                }
            }
        }
        (None, None)
    }

    pub fn commit_counter(&self) -> u64 {
        self.shared.commit_counter.load(Ordering::Acquire)
    }

    /// Set the has_subscribers flag on the shared state.
    /// Used by Database::subscribe() to eagerly disable the fast write path.
    pub fn set_has_subscribers(&self, value: bool) {
        self.shared.has_subscribers.store(value, Ordering::Release);
    }

    pub fn reader_stats(&self) -> ShardStats {
        ShardStats {
            gets: self.shared.stats_gets.load(Ordering::Relaxed),
            bloom_negatives: self.shared.stats_bloom_negatives.load(Ordering::Relaxed),
            mmap_block_reads: self.shared.stats_mmap_block_reads.load(Ordering::Relaxed),
            ..ShardStats::default()
        }
    }
}

// ---------------------------------------------------------------------------
// ShardRuntime — the single-writer shard actor thread
// ---------------------------------------------------------------------------

/// Parameters for opening a shard (avoids too-many-arguments lint).
pub struct ShardOpenParams {
    pub shard_id: usize,
    pub root: PathBuf,
    pub config: Config,
    pub hasher: Arc<dyn Hasher + Send + Sync>,
    pub manifest: Arc<Mutex<Manifest>>,
    pub shard_state: ManifestShardState,
    pub block_cache: Arc<BlockCache>,
    pub index_cache: Arc<IndexCache>,
}

pub struct ShardRuntime {
    shard_id: usize,
    shard_dir: PathBuf,
    config: Config,
    manifest: Arc<Mutex<Manifest>>,
    wal: Wal,
    local_commit_counter: u64,
    stats: ShardStats,
    subscribers: Vec<Subscriber>,
    shared: Arc<ShardShared>,
}

impl ShardRuntime {
    /// Get the Arc<ShardShared> for direct access by the fast write path.
    pub fn shared(&self) -> Arc<ShardShared> {
        self.shared.clone()
    }

    /// Get the WAL file path for the group commit durability thread.
    pub fn wal_path(&self) -> PathBuf {
        self.wal.path().to_path_buf()
    }

    pub fn open(params: ShardOpenParams) -> Result<(Self, ShardReadHandle)> {
        let ShardOpenParams {
            shard_id,
            root,
            config,
            hasher,
            manifest,
            shard_state,
            block_cache,
            index_cache,
        } = params;

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

        let levels = if let Some(level_info) = &shard_state.level_files {
            LevelManager::from_manifest_levels(
                shard_dir.clone(),
                level_info,
                config.compaction_max_levels,
                config.compaction_l1_target_bytes,
                config.compaction_size_ratio,
                config.sstable_max_file_bytes,
            )?
        } else {
            let mut sstables = Vec::new();
            for file in &shard_state.l0_files {
                let p = shard_dir.join(file);
                if p.exists() {
                    sstables.push(SsTableReader::open(&p)?);
                }
            }
            LevelManager::from_sstables(
                shard_dir.clone(),
                sstables,
                config.compaction_max_levels,
                config.compaction_l1_target_bytes,
                config.compaction_size_ratio,
                config.sstable_max_file_bytes,
            )
        };

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

        let ai_annotate_reads = config.ai_annotate_reads;
        let shared = Arc::new(ShardShared {
            active_memtable: RwLock::new(memtable),
            immutable_memtables: RwLock::new(Vec::new()),
            levels: RwLock::new(levels),
            commit_counter: AtomicU64::new(commit_counter),
            hasher,
            block_cache,
            index_cache,
            stats_gets: AtomicU64::new(0),
            stats_bloom_negatives: AtomicU64::new(0),
            stats_mmap_block_reads: AtomicU64::new(0),
            ai_annotate_reads,
            has_subscribers: AtomicBool::new(false),
        });

        let read_handle = ShardReadHandle {
            shared: shared.clone(),
        };

        let runtime = Self {
            shard_id,
            shard_dir,
            config,
            manifest,
            wal,
            local_commit_counter: commit_counter,
            stats: ShardStats::default(),
            subscribers: Vec::new(),
            shared,
        };

        Ok((runtime, read_handle))
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
                ShardCommand::FlushRequest => {
                    // Sync local_commit_counter from the atomic (fast path may have advanced it)
                    let current = self.shared.commit_counter.load(Ordering::Acquire);
                    if current > self.local_commit_counter {
                        self.local_commit_counter = current;
                    }
                    let should_flush = {
                        let memtable = self.shared.active_memtable.read();
                        memtable.approx_bytes() >= self.config.memtable_max_bytes
                    };
                    if should_flush {
                        let _ = self.flush_active_memtable();
                    }
                }
                ShardCommand::Subscribe { prefix, sender } => {
                    self.subscribers.push(Subscriber { prefix, sender });
                    self.shared.has_subscribers.store(true, Ordering::Release);
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
        // Sync from atomic — the fast write path may have advanced commit_counter
        let atomic_ts = self.shared.commit_counter.load(Ordering::Acquire);
        if atomic_ts > self.local_commit_counter {
            self.local_commit_counter = atomic_ts;
        }
        self.local_commit_counter = self.local_commit_counter.saturating_add(1);
        let commit_ts = self.local_commit_counter;

        let internal_key = encode_internal_key(user_key, commit_ts, KIND_PUT);
        let doc_clone = doc.clone();
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

        {
            let mut memtable = self.shared.active_memtable.write();
            memtable.insert(internal_key, fact);
        }

        // Inline AI risk assessment (ultra-fast, < 500ns)
        if self.config.ai_inline_risk_assessment && !is_internal_ai_key(user_key) {
            let risk = quick_risk_score(&doc_clone);
            if risk >= 0.75 {
                self.write_ai_risk_annotation(user_key, commit_ts, risk, valid_from, valid_to);
            }
        }

        self.shared
            .commit_counter
            .store(commit_ts, Ordering::Release);

        self.stats.puts += 1;
        self.emit_change(user_key, doc_clone, commit_ts, valid_from, valid_to);

        let should_flush = {
            let memtable = self.shared.active_memtable.read();
            memtable.approx_bytes() >= self.config.memtable_max_bytes
        };

        if should_flush {
            self.flush_active_memtable()?;
        } else {
            self.persist_manifest_watermark_in_memory();
        }

        Ok(commit_ts)
    }

    fn emit_change(
        &mut self,
        user_key: &[u8],
        doc: Vec<u8>,
        commit_ts: u64,
        valid_from: u64,
        valid_to: u64,
    ) {
        if self.subscribers.is_empty() {
            return;
        }
        self.subscribers.retain(|sub| {
            if !user_key.starts_with(&sub.prefix) {
                return true;
            }
            sub.sender
                .send(ChangeEvent {
                    user_key: user_key.to_vec(),
                    doc: doc.clone(),
                    commit_ts,
                    valid_from,
                    valid_to,
                })
                .is_ok()
        });
    }

    fn handle_write_batch(&mut self, entries: Vec<WriteBatchItem>) -> Result<Vec<u64>> {
        // Sync from atomic — the fast write path may have advanced commit_counter
        let atomic_ts = self.shared.commit_counter.load(Ordering::Acquire);
        if atomic_ts > self.local_commit_counter {
            self.local_commit_counter = atomic_ts;
        }

        let mut timestamps = Vec::with_capacity(entries.len());
        let mut wal_writes = Vec::with_capacity(entries.len());
        let mut memtable_entries = Vec::with_capacity(entries.len());

        for entry in &entries {
            self.local_commit_counter = self.local_commit_counter.saturating_add(1);
            let commit_ts = self.local_commit_counter;
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

            memtable_entries.push((internal_key, fact));
        }

        self.wal.append_batch(&wal_writes)?;

        {
            let mut memtable = self.shared.active_memtable.write();
            for (key, value) in memtable_entries {
                memtable.insert(key, value);
            }
        }

        self.shared
            .commit_counter
            .store(self.local_commit_counter, Ordering::Release);

        self.stats.puts += entries.len() as u64;

        let should_flush = {
            let memtable = self.shared.active_memtable.read();
            memtable.approx_bytes() >= self.config.memtable_max_bytes
        };

        if should_flush {
            self.flush_active_memtable()?;
        } else {
            self.persist_manifest_watermark_in_memory();
        }

        Ok(timestamps)
    }

    fn write_ai_risk_annotation(
        &mut self,
        user_key: &[u8],
        commit_ts: u64,
        risk: f64,
        valid_from: u64,
        valid_to: u64,
    ) {
        let annotation_key =
            format!("__ai/risk/{}/{}", hex_encode(user_key), commit_ts).into_bytes();
        let annotation_doc =
            format!("{{\"risk\":{risk:.4},\"source_ts\":{commit_ts}}}").into_bytes();

        self.local_commit_counter = self.local_commit_counter.saturating_add(1);
        let annotation_ts = self.local_commit_counter;
        let internal_key = encode_internal_key(&annotation_key, annotation_ts, KIND_PUT);
        let fact = FactValue {
            doc: annotation_doc,
            valid_from,
            valid_to,
        }
        .encode();

        let mut memtable = self.shared.active_memtable.write();
        memtable.insert(internal_key, fact);
    }

    fn persist_manifest_watermark_in_memory(&self) {
        let mut manifest = self.manifest.lock();
        if let Some(shard) = manifest
            .state
            .shards
            .iter_mut()
            .find(|s| s.shard_id == self.shard_id)
        {
            shard.commit_ts_high_watermark = self.local_commit_counter;
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
            shard.commit_ts_high_watermark = self.local_commit_counter;
            let levels = self.shared.levels.read();
            shard.l0_files = levels.file_names();
            shard.level_files = Some(levels.to_manifest_levels());
        }
        manifest.save()?;
        Ok(())
    }

    fn flush_active_memtable(&mut self) -> Result<()> {
        self.wal.sync()?;

        let entries: Vec<(Vec<u8>, Vec<u8>)> = {
            let mut active = self.shared.active_memtable.write();
            if active.is_empty() {
                return Ok(());
            }
            let old = std::mem::replace(&mut *active, Memtable::new());
            let mut imm = self.shared.immutable_memtables.write();
            imm.push(old);
            let last = imm.last().unwrap();
            let mut entries: Vec<(Vec<u8>, Vec<u8>)> =
                last.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries
        };

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
            self.shared.hasher.as_ref(),
        )?;
        File::open(&self.shard_dir)?.sync_all()?;

        {
            let mut imm = self.shared.immutable_memtables.write();
            imm.pop();
        }
        {
            let mut levels = self.shared.levels.write();
            levels.add_l0_file(SsTableReader::open(&sst_path)?);
        }

        self.stats.flushes += 1;
        self.persist_manifest_state()?;
        self.wal.truncate()?;
        self.maybe_compact()?;

        Ok(())
    }

    fn maybe_compact(&mut self) -> Result<()> {
        let mut next_file_id = {
            let manifest = self.manifest.lock();
            manifest.state.next_file_id
        };

        loop {
            let task = {
                let levels = self.shared.levels.read();
                levels.needs_compaction(self.config.compaction_l0_threshold)
            };

            let Some(task) = task else { break };

            let result = {
                let mut levels = self.shared.levels.write();
                levels.execute_compaction(
                    &task,
                    self.config.sstable_block_bytes,
                    self.config.bloom_bits_per_key,
                    self.shared.hasher.as_ref(),
                    &mut next_file_id,
                )?
            };

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

        {
            let mut manifest = self.manifest.lock();
            manifest.state.next_file_id = next_file_id;
        }
        self.persist_manifest_state()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers for prefix scan
// ---------------------------------------------------------------------------

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

fn merge_prefix_visible(
    memtable: &Memtable,
    prefix: &[u8],
    as_of: u64,
    valid_at: Option<u64>,
    best: &mut BTreeMap<Vec<u8>, (u64, Vec<u8>)>,
) -> Result<()> {
    let prefix_results = memtable.scan_prefix_visible(prefix, as_of, valid_at)?;
    for (user_key, (commit_ts, doc)) in prefix_results {
        match best.get(&user_key) {
            Some((best_ts, _)) if *best_ts >= commit_ts => {}
            _ => {
                best.insert(user_key, (commit_ts, doc));
            }
        }
    }
    Ok(())
}
