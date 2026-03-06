//! Encryption key rotation manager.
//!
//! Manages multiple encryption key versions for seamless key rotation.
//! New writes use the active key version; reads look up the key version
//! stored in the block header to select the correct decryption key.

use std::collections::BTreeMap;
use std::sync::RwLock;

use crate::error::{Result, TensorError};
use crate::storage::encryption::EncryptionKey;

/// Manages versioned encryption keys for key rotation.
pub struct KeyManager {
    /// Map from key version to encryption key.
    keys: RwLock<BTreeMap<u32, EncryptionKey>>,
    /// Active key version used for new writes.
    active_version: RwLock<u32>,
}

impl KeyManager {
    /// Create a new key manager with no keys (encryption disabled).
    pub fn new() -> Self {
        Self {
            keys: RwLock::new(BTreeMap::new()),
            active_version: RwLock::new(0),
        }
    }

    /// Create a key manager with an initial key at version 1.
    pub fn with_key(key: EncryptionKey) -> Self {
        let mut keys = BTreeMap::new();
        keys.insert(1, key);
        Self {
            keys: RwLock::new(keys),
            active_version: RwLock::new(1),
        }
    }

    /// Add a new key version and set it as the active key for new writes.
    pub fn rotate_key(&self, new_key: EncryptionKey) -> u32 {
        let mut keys = self.keys.write().unwrap();
        let mut active = self.active_version.write().unwrap();
        let new_version = keys.keys().last().copied().unwrap_or(0) + 1;
        keys.insert(new_version, new_key);
        *active = new_version;
        new_version
    }

    /// Get the active key version for new writes.
    pub fn active_version(&self) -> u32 {
        *self.active_version.read().unwrap()
    }

    /// Get the active encryption key for new writes.
    pub fn active_key(&self) -> Option<EncryptionKey> {
        let keys = self.keys.read().unwrap();
        let version = *self.active_version.read().unwrap();
        keys.get(&version).cloned()
    }

    /// Look up a key by version (for decryption of existing data).
    pub fn get_key(&self, version: u32) -> Option<EncryptionKey> {
        self.keys.read().unwrap().get(&version).cloned()
    }

    /// Returns true if encryption is enabled (at least one key loaded).
    pub fn is_enabled(&self) -> bool {
        !self.keys.read().unwrap().is_empty()
    }

    /// List all key versions.
    pub fn versions(&self) -> Vec<u32> {
        self.keys.read().unwrap().keys().copied().collect()
    }

    /// Encrypt data using the active key. Returns `(version, encrypted_data)`.
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<(u32, Vec<u8>)> {
        let version = self.active_version();
        let key = self
            .active_key()
            .ok_or_else(|| TensorError::Config("no active encryption key".to_string()))?;
        let encrypted = crate::storage::encryption::encrypt_block(&key, plaintext)?;
        Ok((version, encrypted))
    }

    /// Decrypt data using the key version specified.
    pub fn decrypt(&self, version: u32, encrypted: &[u8]) -> Result<Vec<u8>> {
        let key = self.get_key(version).ok_or_else(|| {
            TensorError::Config(format!("encryption key version {version} not found"))
        })?;
        crate::storage::encryption::decrypt_block(&key, encrypted)
    }
}

impl Default for KeyManager {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for KeyManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyManager")
            .field("active_version", &self.active_version())
            .field("num_keys", &self.keys.read().unwrap().len())
            .finish()
    }
}

/// Derive a column-specific encryption key from a master key using HKDF-like
/// construction: SHA-256(master_key || table || column).
pub fn derive_column_key(master_key: &EncryptionKey, table: &str, column: &str) -> EncryptionKey {
    use crate::storage::encryption::EncryptionKey as EK;
    // HKDF-expand-like: hash(key || context)
    let context = format!("tensordb:column:{}:{}", table, column);
    let mut input = Vec::with_capacity(32 + context.len());
    input.extend_from_slice(master_key.as_bytes());
    input.extend_from_slice(context.as_bytes());
    // Use the SHA-256 from the encryption module via passphrase trick
    // We construct a deterministic key from the concatenation
    EK::from_passphrase(&hex_encode(&input))
}

fn hex_encode(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(data.len() * 2);
    for &b in data {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_rotation_increments_version() {
        let km = KeyManager::new();
        assert!(!km.is_enabled());
        assert_eq!(km.active_version(), 0);

        let k1 = EncryptionKey::from_passphrase("key1");
        let v1 = km.rotate_key(k1);
        assert_eq!(v1, 1);
        assert!(km.is_enabled());
        assert_eq!(km.active_version(), 1);

        let k2 = EncryptionKey::from_passphrase("key2");
        let v2 = km.rotate_key(k2);
        assert_eq!(v2, 2);
        assert_eq!(km.active_version(), 2);
        assert_eq!(km.versions(), vec![1, 2]);
    }

    #[test]
    fn old_keys_still_accessible() {
        let km = KeyManager::new();
        let k1 = EncryptionKey::from_passphrase("key1");
        km.rotate_key(k1.clone());
        let k2 = EncryptionKey::from_passphrase("key2");
        km.rotate_key(k2);

        // Can still retrieve version 1
        let retrieved = km.get_key(1).unwrap();
        assert_eq!(retrieved.as_bytes(), k1.as_bytes());
    }

    #[test]
    fn derive_column_key_is_deterministic() {
        let master = EncryptionKey::from_passphrase("master");
        let ck1 = derive_column_key(&master, "users", "email");
        let ck2 = derive_column_key(&master, "users", "email");
        assert_eq!(ck1.as_bytes(), ck2.as_bytes());
    }

    #[test]
    fn derive_column_key_differs_per_column() {
        let master = EncryptionKey::from_passphrase("master");
        let ck1 = derive_column_key(&master, "users", "email");
        let ck2 = derive_column_key(&master, "users", "phone");
        assert_ne!(ck1.as_bytes(), ck2.as_bytes());
    }

    #[test]
    fn with_key_starts_at_version_1() {
        let key = EncryptionKey::from_passphrase("test");
        let km = KeyManager::with_key(key);
        assert!(km.is_enabled());
        assert_eq!(km.active_version(), 1);
        assert!(km.get_key(1).is_some());
    }
}
