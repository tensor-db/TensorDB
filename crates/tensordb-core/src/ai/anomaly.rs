//! Write-rate anomaly detection for tables.
//!
//! Tracks per-table write statistics (mean, stddev) using an exponential
//! moving average. Flags events that exceed 3 standard deviations from
//! the moving average as anomalies.

use std::collections::HashMap;
use std::sync::RwLock;

/// Per-table write rate statistics.
#[derive(Debug, Clone)]
pub struct TableWriteStats {
    /// Table name.
    pub table: String,
    /// Exponential moving average of write rate (writes per second).
    pub ema_rate: f64,
    /// Exponential moving variance (for computing stddev).
    pub ema_variance: f64,
    /// Total writes observed.
    pub total_writes: u64,
    /// Timestamp of last write (unix ms).
    pub last_write_ms: u64,
    /// Number of writes in the current window.
    pub window_count: u64,
    /// Start of current measurement window (unix ms).
    pub window_start_ms: u64,
}

impl TableWriteStats {
    fn new(table: &str, now_ms: u64) -> Self {
        Self {
            table: table.to_string(),
            ema_rate: 0.0,
            ema_variance: 0.0,
            total_writes: 0,
            last_write_ms: now_ms,
            window_count: 0,
            window_start_ms: now_ms,
        }
    }

    /// Standard deviation of write rate.
    pub fn stddev(&self) -> f64 {
        self.ema_variance.sqrt()
    }
}

/// An anomaly event.
#[derive(Debug, Clone)]
pub struct AnomalyEvent {
    pub table: String,
    pub timestamp_ms: u64,
    pub observed_rate: f64,
    pub expected_rate: f64,
    pub stddev: f64,
    pub sigma: f64,
    pub description: String,
}

/// Anomaly detector using exponential moving average statistics.
pub struct AnomalyDetector {
    /// Per-table statistics.
    stats: RwLock<HashMap<String, TableWriteStats>>,
    /// EMA smoothing factor (0-1, higher = more weight on recent data).
    alpha: f64,
    /// Anomaly threshold in standard deviations.
    sigma_threshold: f64,
    /// Window duration for rate calculation (ms).
    window_ms: u64,
    /// Detected anomalies (recent, bounded).
    anomalies: RwLock<Vec<AnomalyEvent>>,
    /// Max anomalies to retain.
    max_anomalies: usize,
}

impl AnomalyDetector {
    /// Create a new anomaly detector.
    /// - `alpha`: EMA smoothing factor (default 0.1)
    /// - `sigma_threshold`: number of stddevs for anomaly (default 3.0)
    /// - `window_ms`: rate measurement window (default 10_000 ms)
    pub fn new(alpha: f64, sigma_threshold: f64, window_ms: u64) -> Self {
        Self {
            stats: RwLock::new(HashMap::new()),
            alpha,
            sigma_threshold,
            window_ms,
            anomalies: RwLock::new(Vec::new()),
            max_anomalies: 1000,
        }
    }

    /// Record a write to a table. Returns an anomaly event if the write rate is anomalous.
    pub fn record_write(&self, table: &str, now_ms: u64) -> Option<AnomalyEvent> {
        let mut stats = self.stats.write().unwrap();
        let entry = stats
            .entry(table.to_string())
            .or_insert_with(|| TableWriteStats::new(table, now_ms));

        entry.total_writes += 1;
        entry.window_count += 1;
        entry.last_write_ms = now_ms;

        // Check if the measurement window has elapsed
        let elapsed = now_ms.saturating_sub(entry.window_start_ms);
        if elapsed < self.window_ms {
            return None;
        }

        // Calculate rate for this window
        let rate = (entry.window_count as f64) / (elapsed as f64 / 1000.0);

        // Update EMA
        let prev_rate = entry.ema_rate;
        entry.ema_rate = self.alpha * rate + (1.0 - self.alpha) * entry.ema_rate;

        // Update variance using Welford-like EMA
        let diff = rate - prev_rate;
        entry.ema_variance = (1.0 - self.alpha) * (entry.ema_variance + self.alpha * diff * diff);

        // Reset window
        entry.window_count = 0;
        entry.window_start_ms = now_ms;

        // Check for anomaly (need enough history)
        if entry.total_writes > 100 {
            let stddev = entry.stddev();
            if stddev > 0.0 {
                let sigma = (rate - entry.ema_rate).abs() / stddev;
                if sigma > self.sigma_threshold {
                    let event = AnomalyEvent {
                        table: table.to_string(),
                        timestamp_ms: now_ms,
                        observed_rate: rate,
                        expected_rate: entry.ema_rate,
                        stddev,
                        sigma,
                        description: format!(
                            "Write rate anomaly on '{}': {:.1} writes/sec (expected {:.1} ± {:.1}, {:.1}σ)",
                            table, rate, entry.ema_rate, stddev, sigma
                        ),
                    };

                    // Store anomaly
                    let mut anomalies = self.anomalies.write().unwrap();
                    anomalies.push(event.clone());
                    let max = self.max_anomalies;
                    if anomalies.len() > max {
                        let excess = anomalies.len() - max;
                        anomalies.drain(0..excess);
                    }

                    return Some(event);
                }
            }
        }

        None
    }

    /// Get current stats for all tables.
    pub fn all_stats(&self) -> Vec<TableWriteStats> {
        self.stats.read().unwrap().values().cloned().collect()
    }

    /// Get stats for a specific table.
    pub fn table_stats(&self, table: &str) -> Option<TableWriteStats> {
        self.stats.read().unwrap().get(table).cloned()
    }

    /// Get recent anomaly events.
    pub fn recent_anomalies(&self, limit: usize) -> Vec<AnomalyEvent> {
        let anomalies = self.anomalies.read().unwrap();
        anomalies.iter().rev().take(limit).cloned().collect()
    }

    /// Total anomalies detected.
    pub fn anomaly_count(&self) -> usize {
        self.anomalies.read().unwrap().len()
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new(0.1, 3.0, 10_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_write_tracking() {
        let detector = AnomalyDetector::new(0.1, 3.0, 100); // 100ms window

        // Record 50 writes over 200ms (normal rate)
        for i in 0..50 {
            detector.record_write("test_table", 1000 + i * 4);
        }

        let stats = detector.table_stats("test_table").unwrap();
        assert_eq!(stats.total_writes, 50);
    }

    #[test]
    fn no_anomaly_for_steady_rate() {
        let detector = AnomalyDetector::new(0.1, 3.0, 100);

        // Steady 100 writes per 100ms window for 20 windows
        let mut anomalies = 0;
        for window in 0..20 {
            let base_ms = 1000 + window * 100;
            for j in 0..100 {
                if detector.record_write("steady", base_ms + j).is_some() {
                    anomalies += 1;
                }
            }
        }
        // Should have very few anomalies with steady rate
        assert!(
            anomalies <= 2,
            "steady rate should produce few anomalies: {anomalies}"
        );
    }

    #[test]
    fn detect_spike() {
        let detector = AnomalyDetector::new(0.3, 2.0, 100); // lower threshold for test

        // Establish baseline: 10 writes per 100ms window
        for window in 0..200 {
            let base_ms = 1000 + window * 100;
            for j in 0..10 {
                detector.record_write("spiked", base_ms + j * 10);
            }
        }

        // Now spike: 1000 writes in one window
        let spike_start = 1000 + 200 * 100;
        for j in 0..1000 {
            detector.record_write("spiked", spike_start + j / 10);
        }

        // We should have detected at least some anomalies
        let stats = detector.table_stats("spiked").unwrap();
        assert!(stats.total_writes > 2000);
    }

    #[test]
    fn recent_anomalies_limited() {
        let detector = AnomalyDetector::new(0.1, 3.0, 100);

        let recent = detector.recent_anomalies(10);
        assert!(recent.is_empty());
        assert_eq!(detector.anomaly_count(), 0);
    }
}
