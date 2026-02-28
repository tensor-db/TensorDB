use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::access_stats::AccessStats;

/// Hint for the SQL executor on which access path to use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessPathHint {
    /// Single key lookup (most efficient)
    PointRead,
    /// Prefix-bounded scan
    PrefixScan,
    /// Full table scan (least efficient)
    FullScan,
}

/// AI-driven query advisor that uses access pattern statistics to inform
/// query execution decisions.
pub struct QueryAdvisor {
    access_stats: Arc<AccessStats>,
}

impl QueryAdvisor {
    pub fn new(access_stats: Arc<AccessStats>) -> Self {
        Self { access_stats }
    }

    /// Returns estimated selectivity for a prefix scan (0.0 = rare, 1.0 = frequent).
    pub fn estimate_scan_selectivity(&self, prefix: &[u8]) -> f64 {
        let total_scans = self.access_stats.total_scans.load(Ordering::Relaxed).max(1);
        let prefix_count = self.access_stats.prefix_counts.lock().get_count(prefix);
        prefix_count as f64 / total_scans as f64
    }

    /// Suggests which access path to use for a query.
    pub fn recommend_access_path(&self, table: &str, has_pk_filter: bool) -> AccessPathHint {
        if has_pk_filter {
            return AccessPathHint::PointRead;
        }
        let prefix = format!("table/{table}/").into_bytes();
        let selectivity = self.estimate_scan_selectivity(&prefix);
        if selectivity > 0.5 {
            // Hot prefix — full scan may be faster than index lookup
            AccessPathHint::FullScan
        } else {
            AccessPathHint::PrefixScan
        }
    }

    /// Returns the top-N hottest keys for cache warming.
    pub fn hot_keys(&self, limit: usize) -> Vec<(Vec<u8>, u64)> {
        self.access_stats.hot_keys.lock().top_n(limit)
    }

    /// Returns the top-N hottest scan prefixes.
    pub fn hot_prefixes(&self, limit: usize) -> Vec<(Vec<u8>, u64)> {
        self.access_stats.prefix_counts.lock().top_n(limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_advisor_point_read_hint() {
        let stats = Arc::new(AccessStats::new(100));
        let advisor = QueryAdvisor::new(stats);
        assert_eq!(
            advisor.recommend_access_path("users", true),
            AccessPathHint::PointRead
        );
    }

    #[test]
    fn query_advisor_selectivity() {
        let stats = Arc::new(AccessStats::new(100));
        for _ in 0..10 {
            stats.record_scan(b"table/orders/");
        }
        for _ in 0..10 {
            stats.record_scan(b"table/users/");
        }
        let advisor = QueryAdvisor::new(stats);
        let sel = advisor.estimate_scan_selectivity(b"table/orders/");
        assert!(sel > 0.0);
    }
}
