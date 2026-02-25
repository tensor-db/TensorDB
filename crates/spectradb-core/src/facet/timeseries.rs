//! Time-series facet.
//!
//! Provides optimized storage for high-cardinality timestamp-keyed data.
//!
//! Storage layout:
//!   `__ts/{table}/{metric}/{timestamp_bucket}` → packed values
//!
//! Features:
//! - Bucketed storage for efficient range queries
//! - Downsampling support (avg, min, max per bucket)

use crate::error::{Result, SpectraError};

const TS_PREFIX: &str = "__ts";
const DEFAULT_BUCKET_SIZE: u64 = 3600; // 1 hour in seconds

/// Generate the storage key for a time-series bucket.
pub fn ts_bucket_key(table: &str, metric: &str, bucket_id: u64) -> Vec<u8> {
    format!("{TS_PREFIX}/{table}/{metric}/{bucket_id:016}").into_bytes()
}

/// Compute the bucket ID for a given timestamp.
pub fn bucket_id(timestamp: u64, bucket_size: u64) -> u64 {
    timestamp / bucket_size
}

/// Compute which buckets overlap with a time range [start, end].
pub fn range_bucket_ids(start: u64, end: u64, bucket_size: u64) -> Vec<u64> {
    let first = bucket_id(start, bucket_size);
    let last = bucket_id(end, bucket_size);
    (first..=last).collect()
}

/// A single time-series data point.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DataPoint {
    pub timestamp: u64,
    pub value: f64,
}

/// A packed bucket of data points.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Bucket {
    pub points: Vec<DataPoint>,
}

impl Bucket {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
        }
    }

    pub fn add(&mut self, timestamp: u64, value: f64) {
        self.points.push(DataPoint { timestamp, value });
    }

    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn decode(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(|e| SpectraError::SqlExec(e.to_string()))
    }

    pub fn merge(&mut self, other: &Bucket) {
        self.points.extend(other.points.iter().cloned());
        self.points.sort_by_key(|p| p.timestamp);
    }

    /// Downsample: return one aggregated data point for the bucket.
    pub fn downsample(&self) -> Option<DownsampleResult> {
        if self.points.is_empty() {
            return None;
        }
        let count = self.points.len() as f64;
        let sum: f64 = self.points.iter().map(|p| p.value).sum();
        let min = self
            .points
            .iter()
            .map(|p| p.value)
            .fold(f64::INFINITY, f64::min);
        let max = self
            .points
            .iter()
            .map(|p| p.value)
            .fold(f64::NEG_INFINITY, f64::max);
        Some(DownsampleResult {
            count: count as u64,
            sum,
            avg: sum / count,
            min,
            max,
            first_ts: self.points.first().map(|p| p.timestamp).unwrap_or(0),
            last_ts: self.points.last().map(|p| p.timestamp).unwrap_or(0),
        })
    }

    /// Filter points within a time range [start, end].
    pub fn range(&self, start: u64, end: u64) -> Vec<&DataPoint> {
        self.points
            .iter()
            .filter(|p| p.timestamp >= start && p.timestamp <= end)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct DownsampleResult {
    pub count: u64,
    pub sum: f64,
    pub avg: f64,
    pub min: f64,
    pub max: f64,
    pub first_ts: u64,
    pub last_ts: u64,
}

/// Configuration for time-series storage.
pub struct TimeSeriesConfig {
    pub bucket_size: u64,
}

impl Default for TimeSeriesConfig {
    fn default() -> Self {
        Self {
            bucket_size: DEFAULT_BUCKET_SIZE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_key_format() {
        let key = ts_bucket_key("metrics", "cpu", 42);
        assert_eq!(
            std::str::from_utf8(&key).unwrap(),
            "__ts/metrics/cpu/0000000000000042"
        );
    }

    #[test]
    fn bucket_id_computation() {
        assert_eq!(bucket_id(3600, 3600), 1);
        assert_eq!(bucket_id(7199, 3600), 1);
        assert_eq!(bucket_id(7200, 3600), 2);
    }

    #[test]
    fn range_bucket_ids_computation() {
        let ids = range_bucket_ids(3600, 10800, 3600);
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn bucket_operations() {
        let mut bucket = Bucket::new();
        bucket.add(1000, 10.0);
        bucket.add(1001, 20.0);
        bucket.add(1002, 30.0);

        let ds = bucket.downsample().unwrap();
        assert_eq!(ds.count, 3);
        assert!((ds.avg - 20.0).abs() < 0.001);
        assert!((ds.min - 10.0).abs() < 0.001);
        assert!((ds.max - 30.0).abs() < 0.001);
    }

    #[test]
    fn bucket_roundtrip() {
        let mut bucket = Bucket::new();
        bucket.add(100, 1.5);
        bucket.add(200, 2.5);
        let encoded = bucket.encode();
        let decoded = Bucket::decode(&encoded).unwrap();
        assert_eq!(decoded.points.len(), 2);
    }

    #[test]
    fn bucket_range_filter() {
        let mut bucket = Bucket::new();
        bucket.add(100, 1.0);
        bucket.add(200, 2.0);
        bucket.add(300, 3.0);
        let in_range = bucket.range(150, 250);
        assert_eq!(in_range.len(), 1);
        assert_eq!(in_range[0].timestamp, 200);
    }
}
