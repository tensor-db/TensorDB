//! Phi-accrual failure detector for cluster health monitoring.
//!
//! Uses heartbeat inter-arrival times to compute a suspicion level (phi).
//! When phi exceeds a threshold, the node is considered failed.

use std::collections::HashMap;

use parking_lot::RwLock;

/// Phi-accrual failure detector.
pub struct HealthChecker {
    heartbeats: RwLock<HashMap<String, Vec<u64>>>,
    phi_threshold: f64,
    window_size: usize,
}

impl HealthChecker {
    /// Create a new health checker with the given phi threshold.
    pub fn new(phi_threshold: f64) -> Self {
        Self {
            heartbeats: RwLock::new(HashMap::new()),
            phi_threshold,
            window_size: 100,
        }
    }

    /// Record a heartbeat from a node.
    pub fn record_heartbeat(&self, node_id: &str) {
        let now = current_time_ms();
        let mut heartbeats = self.heartbeats.write();
        let times = heartbeats.entry(node_id.to_string()).or_default();
        times.push(now);
        if times.len() > self.window_size {
            times.remove(0);
        }
    }

    /// Compute the phi (suspicion) value for a node.
    pub fn phi(&self, node_id: &str) -> f64 {
        let now = current_time_ms();
        let heartbeats = self.heartbeats.read();
        let times = match heartbeats.get(node_id) {
            Some(t) if t.len() >= 2 => t,
            _ => return 0.0, // Not enough data
        };

        // Compute mean and std dev of inter-arrival times
        let intervals: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]) as f64).collect();

        let n = intervals.len() as f64;
        let mean = intervals.iter().sum::<f64>() / n;
        let variance = intervals
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / n;
        let std_dev = variance.sqrt().max(1.0); // Avoid division by zero

        let last = *times.last().unwrap();
        let elapsed = (now - last) as f64;

        // Phi = -log10(1 - CDF(elapsed))
        // Using normal distribution approximation
        let y = (elapsed - mean) / std_dev;
        let p = 1.0 / (1.0 + (-1.5976 * y * (1.0 + 0.04417 * y * y)).exp());
        -p.log10()
    }

    /// Check if a node is considered healthy.
    pub fn is_healthy(&self, node_id: &str) -> bool {
        self.phi(node_id) < self.phi_threshold
    }

    /// Get all nodes with their health status.
    pub fn all_status(&self) -> Vec<(String, bool, f64)> {
        let heartbeats = self.heartbeats.read();
        heartbeats
            .keys()
            .map(|id| {
                let phi = self.phi(id);
                (id.clone(), phi < self.phi_threshold, phi)
            })
            .collect()
    }
}

fn current_time_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_healthy_node() {
        let checker = HealthChecker::new(8.0);
        // Record several heartbeats
        for _ in 0..5 {
            checker.record_heartbeat("node-1");
        }
        // Just recorded, should be healthy
        assert!(checker.is_healthy("node-1"));
    }

    #[test]
    fn test_unknown_node() {
        let checker = HealthChecker::new(8.0);
        // No heartbeats recorded — phi is 0 (healthy by default)
        assert!(checker.is_healthy("unknown"));
    }

    #[test]
    fn test_all_status() {
        let checker = HealthChecker::new(8.0);
        checker.record_heartbeat("node-1");
        checker.record_heartbeat("node-1");
        let status = checker.all_status();
        assert_eq!(status.len(), 1);
    }
}
