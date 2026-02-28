use std::sync::Arc;
use std::time::Instant;

use super::access_stats::AccessStats;
use crate::storage::levels::CompactionTask;

/// Decision from the AI compaction advisor.
pub enum CompactionDecision {
    /// Compact now with the given task.
    CompactNow(CompactionTask),
    /// Defer compaction for the given reason.
    Defer { reason: &'static str },
}

/// AI-driven compaction advisor that considers workload patterns when
/// deciding whether to compact.
pub struct CompactionAdvisor {
    write_rate_window: Vec<(Instant, u64)>,
    #[allow(dead_code)]
    access_stats: Arc<AccessStats>,
    window_size: usize,
}

impl CompactionAdvisor {
    pub fn new(access_stats: Arc<AccessStats>, window_size: usize) -> Self {
        Self {
            write_rate_window: Vec::with_capacity(window_size),
            access_stats,
            window_size,
        }
    }

    /// Record a write event for rate tracking.
    pub fn record_write(&mut self, count: u64) {
        self.write_rate_window.push((Instant::now(), count));
        if self.write_rate_window.len() > self.window_size {
            self.write_rate_window.remove(0);
        }
    }

    /// Should we compact now, or defer?
    pub fn should_compact(
        &self,
        task: CompactionTask,
        l0_threshold: usize,
        l0_count: usize,
    ) -> CompactionDecision {
        // If L0 is severely over threshold, always compact
        if l0_count > l0_threshold * 2 {
            return CompactionDecision::CompactNow(task);
        }

        // Check recent write rate — if very high, defer to let batch accumulate
        let recent_rate = self.recent_write_rate();
        if recent_rate > 10_000.0 && l0_count <= l0_threshold + 2 {
            return CompactionDecision::Defer {
                reason: "high write rate, deferring compaction",
            };
        }

        CompactionDecision::CompactNow(task)
    }

    /// Calculate recent write rate (writes/sec) from the sliding window.
    fn recent_write_rate(&self) -> f64 {
        if self.write_rate_window.len() < 2 {
            return 0.0;
        }
        let first = self.write_rate_window.first().unwrap();
        let last = self.write_rate_window.last().unwrap();
        let duration = last.0.duration_since(first.0).as_secs_f64();
        if duration < 0.001 {
            return 0.0;
        }
        let total_writes: u64 = self.write_rate_window.iter().map(|(_, c)| c).sum();
        total_writes as f64 / duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compaction_advisor_severe_l0_always_compacts() {
        let stats = Arc::new(AccessStats::new(100));
        let advisor = CompactionAdvisor::new(stats, 10);
        let task = CompactionTask {
            source_level: 0,
            target_level: 1,
        };
        // L0 count = 20, threshold = 8 → 20 > 16 → always compact
        match advisor.should_compact(task, 8, 20) {
            CompactionDecision::CompactNow(_) => {}
            CompactionDecision::Defer { .. } => panic!("should have compacted"),
        }
    }

    #[test]
    fn compaction_advisor_normal_compacts() {
        let stats = Arc::new(AccessStats::new(100));
        let advisor = CompactionAdvisor::new(stats, 10);
        let task = CompactionTask {
            source_level: 0,
            target_level: 1,
        };
        // No write rate data, moderate L0 count
        match advisor.should_compact(task, 8, 9) {
            CompactionDecision::CompactNow(_) => {}
            CompactionDecision::Defer { .. } => panic!("should have compacted"),
        }
    }
}
