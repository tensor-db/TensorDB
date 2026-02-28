use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

/// Global metrics registry for TensorDB.
pub struct MetricsRegistry {
    counters: Mutex<Vec<(String, Arc<AtomicU64>)>>,
    gauges: Mutex<Vec<(String, Arc<AtomicU64>)>>,
    histograms: Mutex<Vec<(String, Arc<Histogram>)>>,
    slow_query_log: Arc<SlowQueryLog>,
}

impl MetricsRegistry {
    pub fn new(slow_query_threshold_us: u64) -> Self {
        MetricsRegistry {
            counters: Mutex::new(Vec::new()),
            gauges: Mutex::new(Vec::new()),
            histograms: Mutex::new(Vec::new()),
            slow_query_log: Arc::new(SlowQueryLog::new(slow_query_threshold_us)),
        }
    }

    /// Register a counter metric.
    pub fn counter(&self, name: &str) -> Arc<AtomicU64> {
        let mut counters = self.counters.lock();
        for (n, c) in counters.iter() {
            if n == name {
                return Arc::clone(c);
            }
        }
        let c = Arc::new(AtomicU64::new(0));
        counters.push((name.to_string(), Arc::clone(&c)));
        c
    }

    /// Register a gauge metric (can go up and down).
    pub fn gauge(&self, name: &str) -> Arc<AtomicU64> {
        let mut gauges = self.gauges.lock();
        for (n, g) in gauges.iter() {
            if n == name {
                return Arc::clone(g);
            }
        }
        let g = Arc::new(AtomicU64::new(0));
        gauges.push((name.to_string(), Arc::clone(&g)));
        g
    }

    /// Register a histogram metric.
    pub fn histogram(&self, name: &str) -> Arc<Histogram> {
        let mut histograms = self.histograms.lock();
        for (n, h) in histograms.iter() {
            if n == name {
                return Arc::clone(h);
            }
        }
        let h = Arc::new(Histogram::new());
        histograms.push((name.to_string(), Arc::clone(&h)));
        h
    }

    /// Get the slow query log.
    pub fn slow_query_log(&self) -> &Arc<SlowQueryLog> {
        &self.slow_query_log
    }

    /// Snapshot all metrics as a JSON-compatible structure.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let counters: Vec<(String, u64)> = self
            .counters
            .lock()
            .iter()
            .map(|(n, c)| (n.clone(), c.load(Ordering::Relaxed)))
            .collect();

        let gauges: Vec<(String, u64)> = self
            .gauges
            .lock()
            .iter()
            .map(|(n, g)| (n.clone(), g.load(Ordering::Relaxed)))
            .collect();

        let histograms: Vec<(String, HistogramSnapshot)> = self
            .histograms
            .lock()
            .iter()
            .map(|(n, h)| (n.clone(), h.snapshot()))
            .collect();

        MetricsSnapshot {
            counters,
            gauges,
            histograms,
            slow_queries: self.slow_query_log.recent(20),
        }
    }
}

/// A snapshot of all metrics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MetricsSnapshot {
    pub counters: Vec<(String, u64)>,
    pub gauges: Vec<(String, u64)>,
    pub histograms: Vec<(String, HistogramSnapshot)>,
    pub slow_queries: Vec<SlowQueryEntry>,
}

/// A histogram for tracking latency distributions.
pub struct Histogram {
    inner: Mutex<HistogramInner>,
}

struct HistogramInner {
    count: u64,
    sum: u64,
    min: u64,
    max: u64,
    /// HDR-style buckets: [0-1µs, 1-2µs, 2-4µs, 4-8µs, ..., 512ms-1s, 1s+]
    buckets: [u64; 32],
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

impl Histogram {
    pub fn new() -> Self {
        Histogram {
            inner: Mutex::new(HistogramInner {
                count: 0,
                sum: 0,
                min: u64::MAX,
                max: 0,
                buckets: [0; 32],
            }),
        }
    }

    /// Record a value in microseconds.
    pub fn record(&self, value_us: u64) {
        let mut inner = self.inner.lock();
        inner.count += 1;
        inner.sum += value_us;
        inner.min = inner.min.min(value_us);
        inner.max = inner.max.max(value_us);

        let bucket = if value_us == 0 {
            0
        } else {
            (64 - value_us.leading_zeros()).min(31) as usize
        };
        inner.buckets[bucket] += 1;
    }

    /// Get a snapshot of the histogram.
    pub fn snapshot(&self) -> HistogramSnapshot {
        let inner = self.inner.lock();
        let avg = if inner.count > 0 {
            inner.sum / inner.count
        } else {
            0
        };

        // Approximate p50 and p99 from buckets
        let p50 = percentile_from_buckets(&inner.buckets, inner.count, 0.5);
        let p99 = percentile_from_buckets(&inner.buckets, inner.count, 0.99);

        HistogramSnapshot {
            count: inner.count,
            sum_us: inner.sum,
            min_us: if inner.min == u64::MAX { 0 } else { inner.min },
            max_us: inner.max,
            avg_us: avg,
            p50_us: p50,
            p99_us: p99,
        }
    }
}

fn percentile_from_buckets(buckets: &[u64; 32], total: u64, percentile: f64) -> u64 {
    if total == 0 {
        return 0;
    }
    let target = (total as f64 * percentile) as u64;
    let mut cumulative = 0u64;
    for (i, &count) in buckets.iter().enumerate() {
        cumulative += count;
        if cumulative >= target {
            return 1u64 << i;
        }
    }
    1u64 << 31
}

/// Snapshot of histogram data.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HistogramSnapshot {
    pub count: u64,
    pub sum_us: u64,
    pub min_us: u64,
    pub max_us: u64,
    pub avg_us: u64,
    pub p50_us: u64,
    pub p99_us: u64,
}

/// A slow query log entry.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlowQueryEntry {
    pub query: String,
    pub duration_us: u64,
    pub timestamp: u64,
    pub rows_returned: Option<u64>,
}

/// Slow query log — ring buffer of recent slow queries.
pub struct SlowQueryLog {
    threshold_us: u64,
    entries: Mutex<VecDeque<SlowQueryEntry>>,
    total_slow: AtomicU64,
    max_entries: usize,
}

impl SlowQueryLog {
    pub fn new(threshold_us: u64) -> Self {
        SlowQueryLog {
            threshold_us,
            entries: Mutex::new(VecDeque::with_capacity(256)),
            total_slow: AtomicU64::new(0),
            max_entries: 256,
        }
    }

    /// Record a query execution. Only logs if duration exceeds threshold.
    pub fn record(&self, query: &str, duration_us: u64, rows_returned: Option<u64>) {
        if duration_us < self.threshold_us {
            return;
        }

        self.total_slow.fetch_add(1, Ordering::Relaxed);

        let entry = SlowQueryEntry {
            query: if query.len() > 500 {
                format!("{}...", &query[..500])
            } else {
                query.to_string()
            },
            duration_us,
            timestamp: current_timestamp_ms(),
            rows_returned,
        };

        let mut entries = self.entries.lock();
        if entries.len() >= self.max_entries {
            entries.pop_front();
        }
        entries.push_back(entry);
    }

    /// Get recent slow queries.
    pub fn recent(&self, limit: usize) -> Vec<SlowQueryEntry> {
        let entries = self.entries.lock();
        entries.iter().rev().take(limit).cloned().collect()
    }

    /// Total number of slow queries since startup.
    pub fn total_count(&self) -> u64 {
        self.total_slow.load(Ordering::Relaxed)
    }

    /// Get the threshold in microseconds.
    pub fn threshold_us(&self) -> u64 {
        self.threshold_us
    }
}

/// System stats collected at a point in time.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemStats {
    pub uptime_ms: u64,
    pub total_reads: u64,
    pub total_writes: u64,
    pub total_scans: u64,
    pub total_sql_queries: u64,
    pub cache_hit_rate: f64,
    pub memtable_bytes: u64,
    pub sstable_count: u64,
    pub wal_bytes: u64,
    pub shard_count: usize,
}

/// A timer guard that records elapsed time on drop.
pub struct TimerGuard {
    start: std::time::Instant,
    histogram: Arc<Histogram>,
}

impl TimerGuard {
    pub fn new(histogram: Arc<Histogram>) -> Self {
        TimerGuard {
            start: std::time::Instant::now(),
            histogram,
        }
    }

    /// Get elapsed time in microseconds without stopping.
    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }
}

impl Drop for TimerGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_micros() as u64;
        self.histogram.record(elapsed);
    }
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let reg = MetricsRegistry::new(1000);
        let c = reg.counter("queries_total");
        c.fetch_add(1, Ordering::Relaxed);
        c.fetch_add(1, Ordering::Relaxed);

        let snap = reg.snapshot();
        let val = snap
            .counters
            .iter()
            .find(|(n, _)| n == "queries_total")
            .unwrap()
            .1;
        assert_eq!(val, 2);
    }

    #[test]
    fn test_gauge() {
        let reg = MetricsRegistry::new(1000);
        let g = reg.gauge("active_connections");
        g.store(5, Ordering::Relaxed);
        g.store(3, Ordering::Relaxed);

        let snap = reg.snapshot();
        let val = snap
            .gauges
            .iter()
            .find(|(n, _)| n == "active_connections")
            .unwrap()
            .1;
        assert_eq!(val, 3);
    }

    #[test]
    fn test_histogram() {
        let h = Histogram::new();
        h.record(10);
        h.record(20);
        h.record(30);
        h.record(100);

        let snap = h.snapshot();
        assert_eq!(snap.count, 4);
        assert_eq!(snap.min_us, 10);
        assert_eq!(snap.max_us, 100);
        assert_eq!(snap.avg_us, 40);
    }

    #[test]
    fn test_histogram_empty() {
        let h = Histogram::new();
        let snap = h.snapshot();
        assert_eq!(snap.count, 0);
        assert_eq!(snap.min_us, 0);
        assert_eq!(snap.max_us, 0);
        assert_eq!(snap.p50_us, 0);
    }

    #[test]
    fn test_slow_query_log_below_threshold() {
        let log = SlowQueryLog::new(1000); // 1ms threshold
        log.record("SELECT 1", 500, None); // 500µs < 1000µs
        assert_eq!(log.total_count(), 0);
        assert!(log.recent(10).is_empty());
    }

    #[test]
    fn test_slow_query_log_above_threshold() {
        let log = SlowQueryLog::new(1000);
        log.record("SELECT * FROM big_table", 5000, Some(100));
        assert_eq!(log.total_count(), 1);

        let entries = log.recent(10);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].duration_us, 5000);
        assert_eq!(entries[0].rows_returned, Some(100));
    }

    #[test]
    fn test_slow_query_log_truncates_long_queries() {
        let log = SlowQueryLog::new(0); // Log everything
        let long_query = "x".repeat(1000);
        log.record(&long_query, 100, None);

        let entries = log.recent(1);
        assert!(entries[0].query.len() < 600);
        assert!(entries[0].query.ends_with("..."));
    }

    #[test]
    fn test_slow_query_log_ring_buffer() {
        let log = SlowQueryLog::new(0);
        for i in 0..300 {
            log.record(&format!("query_{i}"), i as u64, None);
        }
        assert_eq!(log.total_count(), 300);
        // Ring buffer capped at 256
        let entries = log.recent(300);
        assert!(entries.len() <= 256);
    }

    #[test]
    fn test_timer_guard() {
        let h = Arc::new(Histogram::new());
        {
            let _guard = TimerGuard::new(Arc::clone(&h));
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let snap = h.snapshot();
        assert_eq!(snap.count, 1);
        assert!(snap.min_us >= 500); // At least 500µs (sleep might be imprecise)
    }

    #[test]
    fn test_metrics_registry_same_name_returns_same_counter() {
        let reg = MetricsRegistry::new(1000);
        let c1 = reg.counter("test");
        let c2 = reg.counter("test");
        c1.fetch_add(5, Ordering::Relaxed);
        assert_eq!(c2.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_metrics_snapshot() {
        let reg = MetricsRegistry::new(0);
        reg.counter("reads").fetch_add(100, Ordering::Relaxed);
        reg.gauge("mem_bytes").store(1024, Ordering::Relaxed);
        reg.histogram("latency").record(50);
        reg.slow_query_log().record("SELECT 1", 100, None);

        let snap = reg.snapshot();
        assert_eq!(snap.counters.len(), 1);
        assert_eq!(snap.gauges.len(), 1);
        assert_eq!(snap.histograms.len(), 1);
        assert_eq!(snap.slow_queries.len(), 1);
    }
}
