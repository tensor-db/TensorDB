//! Rate limiter for migration traffic.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Token bucket rate limiter for migration bandwidth.
pub struct MigrationThrottle {
    rate_bytes_per_sec: u64,
    tokens: AtomicU64,
    last_refill: parking_lot::Mutex<Instant>,
}

impl MigrationThrottle {
    /// Create a new throttle with the given rate limit.
    pub fn new(rate_bytes_per_sec: u64) -> Self {
        Self {
            rate_bytes_per_sec,
            tokens: AtomicU64::new(rate_bytes_per_sec),
            last_refill: parking_lot::Mutex::new(Instant::now()),
        }
    }

    /// Try to consume `bytes` tokens. Returns true if allowed.
    pub fn try_acquire(&self, bytes: u64) -> bool {
        self.refill();
        let current = self.tokens.load(Ordering::Relaxed);
        if current >= bytes {
            self.tokens.fetch_sub(bytes, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(&self) {
        let mut last = self.last_refill.lock();
        let elapsed = last.elapsed();
        let new_tokens = (elapsed.as_secs_f64() * self.rate_bytes_per_sec as f64) as u64;
        if new_tokens > 0 {
            let current = self.tokens.load(Ordering::Relaxed);
            let max = self.rate_bytes_per_sec;
            self.tokens
                .store((current + new_tokens).min(max), Ordering::Relaxed);
            *last = Instant::now();
        }
    }

    /// Get current available tokens.
    pub fn available(&self) -> u64 {
        self.refill();
        self.tokens.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throttle_basic() {
        let throttle = MigrationThrottle::new(1_000_000); // 1 MB/s
        assert!(throttle.try_acquire(500_000)); // 500 KB
        assert!(throttle.try_acquire(500_000)); // 500 KB
        assert!(!throttle.try_acquire(500_000)); // Over limit
    }

    #[test]
    fn test_throttle_available() {
        let throttle = MigrationThrottle::new(1_000_000);
        assert!(throttle.available() > 0);
    }
}
