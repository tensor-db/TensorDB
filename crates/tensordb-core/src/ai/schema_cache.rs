//! TTL-based schema context cache with DDL invalidation.
//!
//! Caches the schema context string and its tokenized form so that repeated
//! `ask()` calls don't re-run `SHOW TABLES` + `DESCRIBE` on every invocation.
//! DDL statements (CREATE/DROP/ALTER TABLE) invalidate the cache.

use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Cached schema context with TTL-based expiry.
pub struct SchemaCache {
    inner: RwLock<Option<CachedSchema>>,
    ttl: Duration,
}

struct CachedSchema {
    text: String,
    token_ids: Vec<u32>,
    created_at: Instant,
}

impl SchemaCache {
    /// Create a new schema cache with the given TTL.
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            inner: RwLock::new(None),
            ttl: Duration::from_secs(ttl_secs),
        }
    }

    /// Get the cached schema text, or compute it using the provided closure.
    ///
    /// The `tokenize_fn` is called to tokenize the schema text when the cache
    /// is populated or refreshed.
    pub fn get_or_compute<F, T>(
        &self,
        gather_fn: F,
        tokenize_fn: T,
    ) -> crate::error::Result<(String, Vec<u32>)>
    where
        F: FnOnce() -> crate::error::Result<String>,
        T: FnOnce(&str) -> Vec<u32>,
    {
        // Fast path: check read lock
        {
            let guard = self.inner.read();
            if let Some(ref cached) = *guard {
                if cached.created_at.elapsed() < self.ttl {
                    return Ok((cached.text.clone(), cached.token_ids.clone()));
                }
            }
        }

        // Slow path: recompute and store
        let text = gather_fn()?;
        let token_ids = tokenize_fn(&text);

        let mut guard = self.inner.write();
        // Double-check: another thread may have populated the cache
        if let Some(ref cached) = *guard {
            if cached.created_at.elapsed() < self.ttl {
                return Ok((cached.text.clone(), cached.token_ids.clone()));
            }
        }

        let result = (text.clone(), token_ids.clone());
        *guard = Some(CachedSchema {
            text,
            token_ids,
            created_at: Instant::now(),
        });

        Ok(result)
    }

    /// Invalidate the cache. Called after DDL statements.
    pub fn invalidate(&self) {
        let mut guard = self.inner.write();
        *guard = None;
    }

    /// Check if the cache currently holds a valid entry.
    pub fn is_valid(&self) -> bool {
        let guard = self.inner.read();
        guard
            .as_ref()
            .map(|c| c.created_at.elapsed() < self.ttl)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hit_avoids_recomputation() {
        let cache = SchemaCache::new(60);
        let mut call_count = 0u32;

        // First call: should compute
        let (text, tokens) = cache
            .get_or_compute(
                || {
                    call_count += 1;
                    Ok("Table: users\n  id INTEGER\n  name TEXT\n".to_string())
                },
                |s| vec![s.len() as u32],
            )
            .unwrap();
        assert_eq!(call_count, 1);
        assert!(text.contains("users"));
        assert_eq!(tokens.len(), 1);

        // Second call: should return cached
        let (text2, _) = cache
            .get_or_compute(
                || {
                    call_count += 1;
                    Ok("should not be called".to_string())
                },
                |s| vec![s.len() as u32],
            )
            .unwrap();
        assert_eq!(call_count, 1); // gather_fn not called again
        assert_eq!(text, text2);
    }

    #[test]
    fn invalidate_forces_recomputation() {
        let cache = SchemaCache::new(60);
        let mut call_count = 0u32;

        // Populate cache
        cache
            .get_or_compute(
                || {
                    call_count += 1;
                    Ok("v1".to_string())
                },
                |_| vec![],
            )
            .unwrap();
        assert_eq!(call_count, 1);

        // Invalidate
        cache.invalidate();
        assert!(!cache.is_valid());

        // Should recompute
        let (text, _) = cache
            .get_or_compute(
                || {
                    call_count += 1;
                    Ok("v2".to_string())
                },
                |_| vec![],
            )
            .unwrap();
        assert_eq!(call_count, 2);
        assert_eq!(text, "v2");
    }

    #[test]
    fn expired_cache_recomputes() {
        let cache = SchemaCache::new(0); // TTL = 0 → always expired

        let (text, _) = cache
            .get_or_compute(|| Ok("first".to_string()), |_| vec![])
            .unwrap();
        assert_eq!(text, "first");

        // Even immediately after, TTL=0 means it's expired
        let (text2, _) = cache
            .get_or_compute(|| Ok("second".to_string()), |_| vec![])
            .unwrap();
        assert_eq!(text2, "second");
    }
}
