use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

/// Configuration for a connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum number of concurrent connections.
    pub max_connections: usize,
    /// Minimum number of idle connections to keep warm.
    pub min_idle: usize,
    /// Connection idle timeout in milliseconds.
    pub idle_timeout_ms: u64,
    /// Maximum time to wait for a connection in milliseconds.
    pub acquire_timeout_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            max_connections: 100,
            min_idle: 5,
            idle_timeout_ms: 300_000,   // 5 minutes
            acquire_timeout_ms: 30_000, // 30 seconds
        }
    }
}

/// Statistics for the connection pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    pub active_connections: usize,
    pub idle_connections: usize,
    pub total_connections: usize,
    pub total_acquired: u64,
    pub total_released: u64,
    pub total_timeouts: u64,
    pub total_created: u64,
}

/// A connection slot in the pool.
#[derive(Debug)]
pub struct ConnectionSlot {
    pub id: u64,
    pub created_at: u64,
    pub last_used_at: u64,
    pub queries_executed: u64,
}

impl ConnectionSlot {
    fn new(id: u64) -> Self {
        let now = current_timestamp_ms();
        ConnectionSlot {
            id,
            created_at: now,
            last_used_at: now,
            queries_executed: 0,
        }
    }

    fn touch(&mut self) {
        self.last_used_at = current_timestamp_ms();
    }

    fn is_expired(&self, idle_timeout_ms: u64) -> bool {
        let now = current_timestamp_ms();
        now.saturating_sub(self.last_used_at) > idle_timeout_ms
    }
}

/// A connection pool for managing database connections.
pub struct ConnectionPool {
    config: PoolConfig,
    idle: Mutex<VecDeque<ConnectionSlot>>,
    active_count: AtomicUsize,
    total_count: AtomicUsize,
    next_id: AtomicU64,
    stats_acquired: AtomicU64,
    stats_released: AtomicU64,
    stats_timeouts: AtomicU64,
    stats_created: AtomicU64,
}

impl ConnectionPool {
    /// Create a new connection pool with the given configuration.
    pub fn new(config: PoolConfig) -> Self {
        let pool = ConnectionPool {
            config,
            idle: Mutex::new(VecDeque::new()),
            active_count: AtomicUsize::new(0),
            total_count: AtomicUsize::new(0),
            next_id: AtomicU64::new(1),
            stats_acquired: AtomicU64::new(0),
            stats_released: AtomicU64::new(0),
            stats_timeouts: AtomicU64::new(0),
            stats_created: AtomicU64::new(0),
        };
        pool.warm_up();
        pool
    }

    /// Pre-create min_idle connections.
    fn warm_up(&self) {
        let mut idle = self.idle.lock();
        for _ in 0..self.config.min_idle {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            idle.push_back(ConnectionSlot::new(id));
            self.total_count.fetch_add(1, Ordering::Relaxed);
            self.stats_created.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Acquire a connection from the pool.
    /// Returns None if pool is full and no idle connections are available.
    pub fn acquire(&self) -> Option<ConnectionSlot> {
        // Try to get an idle connection
        {
            let mut idle = self.idle.lock();
            // Remove expired connections first
            while let Some(front) = idle.front() {
                if front.is_expired(self.config.idle_timeout_ms) {
                    idle.pop_front();
                    self.total_count.fetch_sub(1, Ordering::Relaxed);
                } else {
                    break;
                }
            }

            if let Some(mut conn) = idle.pop_front() {
                conn.touch();
                self.active_count.fetch_add(1, Ordering::Relaxed);
                self.stats_acquired.fetch_add(1, Ordering::Relaxed);
                return Some(conn);
            }
        }

        // No idle connections — create a new one if under limit
        let current = self.total_count.load(Ordering::Relaxed);
        if current < self.config.max_connections {
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            self.total_count.fetch_add(1, Ordering::Relaxed);
            self.active_count.fetch_add(1, Ordering::Relaxed);
            self.stats_acquired.fetch_add(1, Ordering::Relaxed);
            self.stats_created.fetch_add(1, Ordering::Relaxed);
            return Some(ConnectionSlot::new(id));
        }

        // Pool is full
        self.stats_timeouts.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Release a connection back to the pool.
    pub fn release(&self, mut conn: ConnectionSlot) {
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        self.stats_released.fetch_add(1, Ordering::Relaxed);

        conn.touch();
        let mut idle = self.idle.lock();

        // Only keep up to max_connections in idle pool
        if idle.len() < self.config.max_connections {
            idle.push_back(conn);
        } else {
            self.total_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Close a connection (don't return to pool).
    pub fn close(&self, _conn: ConnectionSlot) {
        self.active_count.fetch_sub(1, Ordering::Relaxed);
        self.total_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current pool statistics.
    pub fn stats(&self) -> PoolStats {
        let idle_count = self.idle.lock().len();
        PoolStats {
            active_connections: self.active_count.load(Ordering::Relaxed),
            idle_connections: idle_count,
            total_connections: self.total_count.load(Ordering::Relaxed),
            total_acquired: self.stats_acquired.load(Ordering::Relaxed),
            total_released: self.stats_released.load(Ordering::Relaxed),
            total_timeouts: self.stats_timeouts.load(Ordering::Relaxed),
            total_created: self.stats_created.load(Ordering::Relaxed),
        }
    }

    /// Evict expired idle connections.
    pub fn evict_expired(&self) -> usize {
        let mut idle = self.idle.lock();
        let before = idle.len();
        idle.retain(|c| !c.is_expired(self.config.idle_timeout_ms));
        let evicted = before - idle.len();
        self.total_count.fetch_sub(evicted, Ordering::Relaxed);
        evicted
    }

    /// Get the pool configuration.
    pub fn config(&self) -> &PoolConfig {
        &self.config
    }
}

/// A guard that automatically returns a connection to the pool on drop.
pub struct PooledConnection {
    conn: Option<ConnectionSlot>,
    pool: Arc<ConnectionPool>,
}

impl PooledConnection {
    pub fn new(conn: ConnectionSlot, pool: Arc<ConnectionPool>) -> Self {
        PooledConnection {
            conn: Some(conn),
            pool,
        }
    }

    pub fn id(&self) -> u64 {
        self.conn.as_ref().map(|c| c.id).unwrap_or(0)
    }

    pub fn record_query(&mut self) {
        if let Some(ref mut conn) = self.conn {
            conn.queries_executed += 1;
        }
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.release(conn);
        }
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
    fn test_pool_creation_with_warmup() {
        let config = PoolConfig {
            max_connections: 10,
            min_idle: 3,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let stats = pool.stats();
        assert_eq!(stats.idle_connections, 3);
        assert_eq!(stats.total_connections, 3);
        assert_eq!(stats.active_connections, 0);
    }

    #[test]
    fn test_acquire_and_release() {
        let pool = ConnectionPool::new(PoolConfig {
            max_connections: 5,
            min_idle: 0,
            ..Default::default()
        });

        let conn = pool.acquire().unwrap();
        assert_eq!(pool.stats().active_connections, 1);

        pool.release(conn);
        assert_eq!(pool.stats().active_connections, 0);
        assert_eq!(pool.stats().idle_connections, 1);
    }

    #[test]
    fn test_pool_exhaustion() {
        let pool = ConnectionPool::new(PoolConfig {
            max_connections: 2,
            min_idle: 0,
            ..Default::default()
        });

        let c1 = pool.acquire().unwrap();
        let c2 = pool.acquire().unwrap();
        let c3 = pool.acquire();

        assert!(c3.is_none()); // Pool exhausted
        assert_eq!(pool.stats().total_timeouts, 1);

        pool.release(c1);
        let c3 = pool.acquire();
        assert!(c3.is_some()); // Now available

        pool.release(c2);
        pool.release(c3.unwrap());
    }

    #[test]
    fn test_connection_reuse() {
        let pool = ConnectionPool::new(PoolConfig {
            max_connections: 5,
            min_idle: 0,
            ..Default::default()
        });

        let conn = pool.acquire().unwrap();
        let conn_id = conn.id;
        pool.release(conn);

        let conn2 = pool.acquire().unwrap();
        assert_eq!(conn2.id, conn_id); // Reused same connection
        pool.release(conn2);
    }

    #[test]
    fn test_expired_connections_evicted() {
        let pool = ConnectionPool::new(PoolConfig {
            max_connections: 5,
            min_idle: 0,
            idle_timeout_ms: 0, // Immediate expiry
            ..Default::default()
        });

        let conn = pool.acquire().unwrap();
        pool.release(conn);

        std::thread::sleep(std::time::Duration::from_millis(1));
        let evicted = pool.evict_expired();
        assert_eq!(evicted, 1);
        assert_eq!(pool.stats().idle_connections, 0);
    }

    #[test]
    fn test_pooled_connection_auto_release() {
        let pool = Arc::new(ConnectionPool::new(PoolConfig {
            max_connections: 5,
            min_idle: 0,
            ..Default::default()
        }));

        {
            let conn = pool.acquire().unwrap();
            let _guard = PooledConnection::new(conn, Arc::clone(&pool));
            assert_eq!(pool.stats().active_connections, 1);
            // _guard drops here
        }

        assert_eq!(pool.stats().active_connections, 0);
        assert_eq!(pool.stats().idle_connections, 1);
    }

    #[test]
    fn test_pool_stats() {
        let pool = ConnectionPool::new(PoolConfig {
            max_connections: 10,
            min_idle: 2,
            ..Default::default()
        });

        let c1 = pool.acquire().unwrap();
        let c2 = pool.acquire().unwrap();
        let _c3 = pool.acquire().unwrap();

        pool.release(c1);
        pool.release(c2);

        let stats = pool.stats();
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.total_acquired, 3);
        assert_eq!(stats.total_released, 2);
    }

    #[test]
    fn test_close_connection() {
        let pool = ConnectionPool::new(PoolConfig {
            max_connections: 5,
            min_idle: 0,
            ..Default::default()
        });

        let conn = pool.acquire().unwrap();
        pool.close(conn);

        assert_eq!(pool.stats().active_connections, 0);
        assert_eq!(pool.stats().total_connections, 0);
    }
}
