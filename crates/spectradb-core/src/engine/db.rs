use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crossbeam_channel::{bounded, unbounded, Sender};
use parking_lot::Mutex;

use crate::config::Config;
use crate::engine::shard::{
    GetResult, PrefixScanResultRow, ShardCommand, ShardRuntime, ShardStats,
};
use crate::error::{Result, SpectraError};
use crate::native_bridge::{build_hasher, Hasher};
use crate::sql::exec::{execute_sql, SqlResult};
use crate::storage::manifest::Manifest;

#[derive(Debug, Clone)]
pub struct ExplainRow {
    pub shard_id: usize,
    pub bloom_hit: Option<bool>,
    pub sstable_block: Option<usize>,
    pub commit_ts_used: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixScanRow {
    pub user_key: Vec<u8>,
    pub doc: Vec<u8>,
    pub commit_ts: u64,
}

#[derive(Debug, Clone, Default)]
pub struct DbStats {
    pub shard_count: usize,
    pub puts: u64,
    pub gets: u64,
    pub flushes: u64,
    pub compactions: u64,
    pub bloom_negatives: u64,
    pub mmap_block_reads: u64,
}

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub write_ops: usize,
    pub read_ops: usize,
    pub keyspace: usize,
    pub read_miss_ratio: f64,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            write_ops: 50_000,
            read_ops: 25_000,
            keyspace: 10_000,
            read_miss_ratio: 0.10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchReport {
    pub write_ops_per_sec: f64,
    pub fsync_every_n_records: usize,
    pub read_p50_us: u64,
    pub read_p95_us: u64,
    pub read_p99_us: u64,
    pub requested_read_miss_ratio: f64,
    pub observed_read_miss_ratio: f64,
    pub bloom_miss_rate: f64,
    pub mmap_reads: u64,
    pub hasher_impl: String,
}

pub struct Database {
    root: PathBuf,
    config: Config,
    manifest: Arc<Mutex<Manifest>>,
    hasher: Arc<dyn Hasher + Send + Sync>,
    shard_senders: Vec<Sender<ShardCommand>>,
    shard_handles: Vec<JoinHandle<()>>,
}

impl Database {
    pub fn open(path: impl AsRef<Path>, config: Config) -> Result<Self> {
        config.validate()?;
        let root = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;

        let manifest = Manifest::load_or_create(&root, config.shard_count)?;
        if manifest.state.shards.len() != config.shard_count {
            return Err(SpectraError::ManifestFormat(format!(
                "manifest shard count {} != config shard count {}",
                manifest.state.shards.len(),
                config.shard_count
            )));
        }

        let manifest = Arc::new(Mutex::new(manifest));
        let hasher = build_hasher();

        let mut shard_senders = Vec::with_capacity(config.shard_count);
        let mut shard_handles = Vec::with_capacity(config.shard_count);

        for shard_id in 0..config.shard_count {
            let (tx, rx) = unbounded();
            let shard_state = {
                let guard = manifest.lock();
                guard.state.shards[shard_id].clone()
            };

            let runtime = ShardRuntime::open(
                shard_id,
                &root,
                config.clone(),
                hasher.clone(),
                manifest.clone(),
                shard_state,
            )?;

            let handle = thread::Builder::new()
                .name(format!("spectradb-shard-{shard_id}"))
                .spawn(move || runtime.run(rx))?;

            shard_senders.push(tx);
            shard_handles.push(handle);
        }

        Ok(Self {
            root,
            config,
            manifest,
            hasher,
            shard_senders,
            shard_handles,
        })
    }

    pub fn put(
        &self,
        user_key: &[u8],
        doc: Vec<u8>,
        valid_from: u64,
        valid_to: u64,
        schema_version: Option<u64>,
    ) -> Result<u64> {
        let shard_id = self.shard_for(user_key);
        let (tx, rx) = bounded(1);
        self.shard_senders[shard_id]
            .send(ShardCommand::Put {
                user_key: user_key.to_vec(),
                doc,
                valid_from,
                valid_to,
                schema_version,
                resp: tx,
            })
            .map_err(|_| SpectraError::ChannelClosed)?;
        rx.recv().map_err(|_| SpectraError::ChannelClosed)?
    }

    pub fn get(
        &self,
        user_key: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
    ) -> Result<Option<Vec<u8>>> {
        Ok(self.get_with_trace(user_key, as_of, valid_at)?.0)
    }

    pub fn explain_get(
        &self,
        user_key: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
    ) -> Result<ExplainRow> {
        let shard_id = self.shard_for(user_key);
        let (_, trace) = self.get_with_trace(user_key, as_of, valid_at)?;
        Ok(ExplainRow {
            shard_id,
            bloom_hit: trace.bloom_hit,
            sstable_block: trace.sstable_block,
            commit_ts_used: trace.commit_ts_used,
        })
    }

    fn get_with_trace(
        &self,
        user_key: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
    ) -> Result<(Option<Vec<u8>>, GetResult)> {
        let shard_id = self.shard_for(user_key);
        let (tx, rx) = bounded(1);
        self.shard_senders[shard_id]
            .send(ShardCommand::Get {
                user_key: user_key.to_vec(),
                as_of,
                valid_at,
                resp: tx,
            })
            .map_err(|_| SpectraError::ChannelClosed)?;
        let r = rx.recv().map_err(|_| SpectraError::ChannelClosed)??;
        Ok((r.value.clone(), r))
    }

    pub fn scan_prefix(
        &self,
        user_key_prefix: &[u8],
        as_of: Option<u64>,
        valid_at: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<PrefixScanRow>> {
        if matches!(limit, Some(0)) {
            return Ok(Vec::new());
        }

        let mut receivers = Vec::with_capacity(self.shard_senders.len());
        for shard_tx in &self.shard_senders {
            let (tx, rx) = bounded(1);
            shard_tx
                .send(ShardCommand::ScanPrefix {
                    user_key_prefix: user_key_prefix.to_vec(),
                    as_of,
                    valid_at,
                    limit,
                    resp: tx,
                })
                .map_err(|_| SpectraError::ChannelClosed)?;
            receivers.push(rx);
        }

        let mut merged = Vec::new();
        for rx in receivers {
            let rows: Vec<PrefixScanResultRow> =
                rx.recv().map_err(|_| SpectraError::ChannelClosed)??;
            merged.extend(rows.into_iter().map(|row| PrefixScanRow {
                user_key: row.user_key,
                doc: row.doc,
                commit_ts: row.commit_ts,
            }));
        }

        merged.sort_by(|a, b| a.user_key.cmp(&b.user_key));
        if let Some(limit) = limit {
            merged.truncate(limit);
        }
        Ok(merged)
    }

    pub fn sql(&self, query: &str) -> Result<SqlResult> {
        execute_sql(self, query)
    }

    pub fn stats(&self) -> Result<DbStats> {
        let mut stats = DbStats {
            shard_count: self.config.shard_count,
            ..DbStats::default()
        };

        for tx in &self.shard_senders {
            let (stx, srx) = bounded(1);
            tx.send(ShardCommand::Stats { resp: stx })
                .map_err(|_| SpectraError::ChannelClosed)?;
            let shard_stats: ShardStats = srx.recv().map_err(|_| SpectraError::ChannelClosed)?;
            stats.puts += shard_stats.puts;
            stats.gets += shard_stats.gets;
            stats.flushes += shard_stats.flushes;
            stats.compactions += shard_stats.compactions;
            stats.bloom_negatives += shard_stats.bloom_negatives;
            stats.mmap_block_reads += shard_stats.mmap_block_reads;
        }

        Ok(stats)
    }

    pub fn bench(&self, opts: BenchOptions) -> Result<BenchReport> {
        let read_miss_ratio = opts.read_miss_ratio.clamp(0.0, 1.0);
        let start = Instant::now();
        for i in 0..opts.write_ops {
            let key = format!("bench/{:08}", i % opts.keyspace);
            let value = format!("{{\"n\":{i}}}").into_bytes();
            let _ = self.put(key.as_bytes(), value, 0, u64::MAX, Some(1))?;
        }
        let write_elapsed = start.elapsed().as_secs_f64().max(0.000_001);
        let write_ops_per_sec = opts.write_ops as f64 / write_elapsed;

        let mut samples = Vec::with_capacity(opts.read_ops);
        let stats_before = self.stats()?;
        let mut read_misses = 0usize;

        for _ in 0..opts.read_ops {
            let miss = fastrand::f64() < read_miss_ratio;
            let n = fastrand::usize(..opts.keyspace);
            let key = if miss {
                read_misses += 1;
                format!("bench-miss/{:08}", n + opts.keyspace)
            } else {
                format!("bench/{:08}", n)
            };
            let t0 = Instant::now();
            let _ = self.get(key.as_bytes(), None, None)?;
            let dt = t0.elapsed();
            samples.push(dt.as_micros() as u64);
        }

        samples.sort_unstable();
        let p50 = percentile(&samples, 0.50);
        let p95 = percentile(&samples, 0.95);
        let p99 = percentile(&samples, 0.99);

        let stats_after = self.stats()?;
        let gets_delta = (stats_after.gets.saturating_sub(stats_before.gets)).max(1);
        let bloom_delta = stats_after
            .bloom_negatives
            .saturating_sub(stats_before.bloom_negatives);
        let mmap_delta = stats_after
            .mmap_block_reads
            .saturating_sub(stats_before.mmap_block_reads);

        Ok(BenchReport {
            write_ops_per_sec,
            fsync_every_n_records: self.config.wal_fsync_every_n_records,
            read_p50_us: p50,
            read_p95_us: p95,
            read_p99_us: p99,
            requested_read_miss_ratio: read_miss_ratio,
            observed_read_miss_ratio: read_misses as f64 / opts.read_ops.max(1) as f64,
            bloom_miss_rate: bloom_delta as f64 / gets_delta as f64,
            mmap_reads: mmap_delta,
            hasher_impl: self.hasher.name().to_string(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.manifest.lock().path().to_path_buf()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub(crate) fn shard_for(&self, key: &[u8]) -> usize {
        (self.hasher.hash64(key) as usize) % self.config.shard_count
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        for tx in &self.shard_senders {
            let _ = tx.send(ShardCommand::Shutdown);
        }
        while let Some(handle) = self.shard_handles.pop() {
            let _ = handle.join();
        }
    }
}

fn percentile(samples: &[u64], p: f64) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let idx = ((samples.len() - 1) as f64 * p).round() as usize;
    samples[idx.min(samples.len() - 1)]
}
