//! Object store abstraction for pluggable storage backends.
//!
//! Provides a trait for SSTable storage that can be backed by local filesystem
//! or remote object stores (S3, GCS, Azure Blob).

use crate::error::{Result, TensorError};
use std::path::{Path, PathBuf};

/// Trait for object storage backends.
/// All operations are synchronous; async wrappers can be added per backend.
pub trait ObjectStore: Send + Sync {
    /// Put an object at the given key.
    fn put(&self, key: &str, data: &[u8]) -> Result<()>;

    /// Get an object by key. Returns None if not found.
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Delete an object by key.
    fn delete(&self, key: &str) -> Result<()>;

    /// List objects with the given prefix.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Check if an object exists.
    fn exists(&self, key: &str) -> Result<bool>;

    /// Get the size of an object in bytes.
    fn size(&self, key: &str) -> Result<Option<u64>>;

    /// Get a range of bytes from an object (for block-level reads).
    fn get_range(&self, key: &str, offset: u64, length: u64) -> Result<Option<Vec<u8>>>;

    /// Backend name for diagnostics.
    fn backend_name(&self) -> &str;
}

/// Local filesystem object store (default backend).
pub struct LocalObjectStore {
    root: PathBuf,
}

impl LocalObjectStore {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn key_to_path(&self, key: &str) -> PathBuf {
        self.root.join(key)
    }
}

impl ObjectStore for LocalObjectStore {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        let path = self.key_to_path(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, data)?;
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let path = self.key_to_path(key);
        if path.exists() {
            Ok(Some(std::fs::read(&path)?))
        } else {
            Ok(None)
        }
    }

    fn delete(&self, key: &str) -> Result<()> {
        let path = self.key_to_path(key);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let prefix_path = self.key_to_path(prefix);
        let search_dir = if prefix_path.is_dir() {
            prefix_path
        } else {
            prefix_path.parent().unwrap_or(&self.root).to_path_buf()
        };

        if !search_dir.exists() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        collect_files(&search_dir, &self.root, prefix, &mut results)?;
        Ok(results)
    }

    fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.key_to_path(key).exists())
    }

    fn size(&self, key: &str) -> Result<Option<u64>> {
        let path = self.key_to_path(key);
        if path.exists() {
            Ok(Some(std::fs::metadata(&path)?.len()))
        } else {
            Ok(None)
        }
    }

    fn get_range(&self, key: &str, offset: u64, length: u64) -> Result<Option<Vec<u8>>> {
        use std::io::{Read, Seek, SeekFrom};
        let path = self.key_to_path(key);
        if !path.exists() {
            return Ok(None);
        }
        let mut file = std::fs::File::open(&path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; length as usize];
        let n = file.read(&mut buf)?;
        buf.truncate(n);
        Ok(Some(buf))
    }

    fn backend_name(&self) -> &str {
        "local"
    }
}

/// Recursively collect files matching a prefix.
fn collect_files(dir: &Path, root: &Path, prefix: &str, results: &mut Vec<String>) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(TensorError::Io)?;
    for entry in entries {
        let entry = entry.map_err(TensorError::Io)?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, root, prefix, results)?;
        } else if let Ok(relative) = path.strip_prefix(root) {
            let key = relative.to_string_lossy().to_string();
            if key.starts_with(prefix) {
                results.push(key);
            }
        }
    }
    Ok(())
}

/// In-memory object store for testing.
pub struct MemoryObjectStore {
    data: parking_lot::RwLock<std::collections::HashMap<String, Vec<u8>>>,
}

impl MemoryObjectStore {
    pub fn new() -> Self {
        Self {
            data: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }
}

impl Default for MemoryObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectStore for MemoryObjectStore {
    fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        self.data.write().insert(key.to_string(), data.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.data.read().get(key).cloned())
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.data.write().remove(key);
        Ok(())
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        Ok(self
            .data
            .read()
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        Ok(self.data.read().contains_key(key))
    }

    fn size(&self, key: &str) -> Result<Option<u64>> {
        Ok(self.data.read().get(key).map(|d| d.len() as u64))
    }

    fn get_range(&self, key: &str, offset: u64, length: u64) -> Result<Option<Vec<u8>>> {
        Ok(self.data.read().get(key).map(|d| {
            let start = offset as usize;
            let end = (offset + length) as usize;
            d[start..end.min(d.len())].to_vec()
        }))
    }

    fn backend_name(&self) -> &str {
        "memory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_object_store_crud() {
        let dir = tempfile::tempdir().unwrap();
        let store = LocalObjectStore::new(dir.path()).unwrap();

        // Put
        store.put("test/file1.sst", b"hello world").unwrap();
        assert!(store.exists("test/file1.sst").unwrap());

        // Get
        let data = store.get("test/file1.sst").unwrap().unwrap();
        assert_eq!(data, b"hello world");

        // Size
        assert_eq!(store.size("test/file1.sst").unwrap(), Some(11));

        // Get range
        let range = store.get_range("test/file1.sst", 6, 5).unwrap().unwrap();
        assert_eq!(range, b"world");

        // List
        store.put("test/file2.sst", b"second").unwrap();
        let files = store.list("test/").unwrap();
        assert_eq!(files.len(), 2);

        // Delete
        store.delete("test/file1.sst").unwrap();
        assert!(!store.exists("test/file1.sst").unwrap());
    }

    #[test]
    fn memory_object_store_crud() {
        let store = MemoryObjectStore::new();

        store.put("a/b", b"data").unwrap();
        assert!(store.exists("a/b").unwrap());
        assert_eq!(store.get("a/b").unwrap().unwrap(), b"data");
        assert_eq!(store.size("a/b").unwrap(), Some(4));

        store.delete("a/b").unwrap();
        assert!(!store.exists("a/b").unwrap());
        assert!(store.get("a/b").unwrap().is_none());
    }

    #[test]
    fn get_nonexistent_returns_none() {
        let store = MemoryObjectStore::new();
        assert!(store.get("missing").unwrap().is_none());
        assert_eq!(store.size("missing").unwrap(), None);
    }
}
