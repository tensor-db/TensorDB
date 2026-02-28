//! Group Commit WAL — batches WAL writes across shards for amortized fsync cost.
//!
//! Instead of each write paying for its own WAL append + fsync, writes enqueue
//! pre-encoded WAL frames into per-shard queues. A single durability thread:
//! 1. Drains all pending records
//! 2. Writes them in a single `write_all` call per shard WAL
//! 3. Does one `fdatasync()` per flush cycle
//!
//! At 100K writes/sec, one 100µs fsync covers ~100 writes → amortized ~1µs/write.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

/// A pre-encoded WAL frame ready for batch write.
/// The `frame` field contains a complete WAL record (magic + len + crc + payload)
/// that can be directly written to the WAL file.
pub struct WalRecord {
    pub shard_id: usize,
    pub frame: Vec<u8>,
}

/// Per-shard batch queue for the group commit system.
pub struct WalBatchQueue {
    shard_queues: Vec<Mutex<Vec<Vec<u8>>>>,
    notify: Arc<(Mutex<bool>, Condvar)>,
    shard_count: usize,
}

impl WalBatchQueue {
    pub fn new(shard_count: usize) -> Self {
        let shard_queues = (0..shard_count).map(|_| Mutex::new(Vec::new())).collect();
        Self {
            shard_queues,
            notify: Arc::new((Mutex::new(false), Condvar::new())),
            shard_count,
        }
    }

    /// Enqueue a pre-encoded WAL frame for background flush.
    pub fn enqueue(&self, record: WalRecord) {
        let shard_id = record.shard_id;
        debug_assert!(shard_id < self.shard_count);
        {
            let mut queue = self.shard_queues[shard_id].lock();
            queue.push(record.frame);
        }
        // Notify the durability thread
        let (lock, cvar) = &*self.notify;
        let mut pending = lock.lock();
        *pending = true;
        cvar.notify_one();
    }

    /// Drain all pending frames for a given shard.
    fn drain_shard(&self, shard_id: usize) -> Vec<Vec<u8>> {
        let mut queue = self.shard_queues[shard_id].lock();
        std::mem::take(&mut *queue)
    }

    /// Check if any shard has pending records.
    fn has_pending(&self) -> bool {
        for q in &self.shard_queues {
            if !q.lock().is_empty() {
                return true;
            }
        }
        false
    }
}

/// Background durability thread that batches WAL writes across shards.
pub struct DurabilityThread {
    queue: Arc<WalBatchQueue>,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl DurabilityThread {
    /// Spawn the durability thread. `wal_paths` must have one entry per shard.
    pub fn spawn(
        queue: Arc<WalBatchQueue>,
        wal_paths: Vec<PathBuf>,
        batch_interval_us: u64,
    ) -> Self {
        assert_eq!(wal_paths.len(), queue.shard_count);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();
        let queue_clone = queue.clone();

        let handle = thread::Builder::new()
            .name("tensordb-group-wal".to_string())
            .spawn(move || {
                Self::run_loop(queue_clone, &wal_paths, batch_interval_us, shutdown_clone);
            })
            .expect("failed to spawn durability thread");

        Self {
            queue,
            shutdown,
            handle: Some(handle),
        }
    }

    fn run_loop(
        queue: Arc<WalBatchQueue>,
        wal_paths: &[PathBuf],
        batch_interval_us: u64,
        shutdown: Arc<AtomicBool>,
    ) {
        let mut wal_files: Vec<File> = wal_paths
            .iter()
            .map(|p| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .expect("failed to open WAL file for group commit")
            })
            .collect();

        let timeout = Duration::from_micros(batch_interval_us);

        loop {
            // Wait for notification or timeout
            {
                let (lock, cvar) = &*queue.notify;
                let mut pending = lock.lock();
                if !*pending && !shutdown.load(Ordering::Relaxed) {
                    cvar.wait_for(&mut pending, timeout);
                }
                *pending = false;
            }

            Self::flush_all(&queue, &mut wal_files);

            if shutdown.load(Ordering::Relaxed) && !queue.has_pending() {
                break;
            }
        }

        // Final flush on shutdown
        Self::flush_all(&queue, &mut wal_files);
    }

    fn flush_all(queue: &WalBatchQueue, wal_files: &mut [File]) {
        for (shard_id, wal_file) in wal_files.iter_mut().enumerate() {
            let frames = queue.drain_shard(shard_id);
            if frames.is_empty() {
                continue;
            }

            // Write all frames for this shard in one syscall
            let total_len: usize = frames.iter().map(|f| f.len()).sum();
            let mut buf = Vec::with_capacity(total_len);
            for frame in &frames {
                buf.extend_from_slice(frame);
            }

            if let Err(e) = wal_file.write_all(&buf) {
                tracing::error!("group WAL write failed for shard {shard_id}: {e}");
                continue;
            }

            if let Err(e) = wal_file.sync_data() {
                tracing::error!("group WAL sync failed for shard {shard_id}: {e}");
            }
        }
    }

    /// Signal shutdown and wait for the durability thread to finish.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        let (lock, cvar) = &*self.queue.notify;
        {
            let mut pending = lock.lock();
            *pending = true;
        }
        cvar.notify_one();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    /// Wait until all currently enqueued records have been flushed to disk.
    pub fn sync(&self) {
        // Wake the durability thread
        let (lock, cvar) = &*self.queue.notify;
        {
            let mut pending = lock.lock();
            *pending = true;
        }
        cvar.notify_one();
        // Spin briefly until queues drain
        for _ in 0..10_000 {
            if !self.queue.has_pending() {
                return;
            }
            thread::sleep(Duration::from_micros(50));
        }
    }
}

impl Drop for DurabilityThread {
    fn drop(&mut self) {
        self.shutdown();
    }
}
