//! Encryption at rest using AES-256-GCM.
//!
//! Provides transparent block-level encryption for SSTable data blocks and WAL frames.
//! The encryption key is derived from a user-supplied passphrase or loaded from a key file.
//!
//! Design:
//! - Each encrypted block is: [12-byte nonce][ciphertext][16-byte auth tag]
//! - Nonces are generated randomly per encryption call (AES-GCM is safe with random nonces
//!   up to 2^32 encryptions per key, which far exceeds typical database usage)
//! - The encryption layer is opt-in via the `encryption` feature flag

use crate::error::{Result, TensorError};

/// An encryption key for AES-256-GCM (32 bytes).
#[derive(Clone)]
pub struct EncryptionKey {
    key_bytes: [u8; 32],
}

impl std::fmt::Debug for EncryptionKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptionKey")
            .field("key_bytes", &"[REDACTED]")
            .finish()
    }
}

impl EncryptionKey {
    /// Create a key from raw 32 bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { key_bytes: bytes }
    }

    /// Derive a key from a passphrase using SHA-256 (simple KDF).
    /// For production, use a proper KDF like Argon2 or PBKDF2.
    pub fn from_passphrase(passphrase: &str) -> Self {
        // Simple SHA-256 hash as KDF — sufficient for embedded use
        let mut hasher = Sha256::new();
        hasher.update(passphrase.as_bytes());
        let hash = hasher.finalize();
        Self { key_bytes: hash }
    }

    /// Load a key from a file (expects exactly 32 bytes or 64 hex chars).
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| TensorError::Config(format!("failed to read encryption key file: {e}")))?;
        let trimmed = contents.trim();

        // Try hex decode first
        if trimmed.len() == 64 {
            if let Ok(bytes) = hex_decode(trimmed) {
                if bytes.len() == 32 {
                    let mut key = [0u8; 32];
                    key.copy_from_slice(&bytes);
                    return Ok(Self { key_bytes: key });
                }
            }
        }

        // Fall back to passphrase derivation
        Ok(Self::from_passphrase(trimmed))
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.key_bytes
    }
}

/// Minimal SHA-256 for key derivation (avoids extra dependency).
struct Sha256 {
    state: [u32; 8],
    buffer: Vec<u8>,
    total_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buffer: Vec::new(),
            total_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        self.total_len += data.len() as u64;

        while self.buffer.len() >= 64 {
            let block: [u8; 64] = self.buffer[..64].try_into().unwrap();
            self.buffer.drain(..64);
            self.compress(&block);
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total_len * 8;
        self.buffer.push(0x80);
        while self.buffer.len() % 64 != 56 {
            self.buffer.push(0);
        }
        self.buffer.extend_from_slice(&bit_len.to_be_bytes());

        let padded = std::mem::take(&mut self.buffer);
        for chunk in padded.chunks(64) {
            let block: [u8; 64] = chunk.try_into().unwrap();
            self.compress(&block);
        }

        let mut result = [0u8; 32];
        for (i, &s) in self.state.iter().enumerate() {
            result[i * 4..(i + 1) * 4].copy_from_slice(&s.to_be_bytes());
        }
        result
    }

    fn compress(&mut self, block: &[u8; 64]) {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

fn hex_decode(s: &str) -> std::result::Result<Vec<u8>, ()> {
    if !s.len().is_multiple_of(2) {
        return Err(());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ()))
        .collect()
}

/// Encrypt a plaintext block using AES-256-GCM.
/// Returns: [12-byte nonce][ciphertext + 16-byte tag]
#[cfg(feature = "encryption")]
pub fn encrypt_block(key: &EncryptionKey, plaintext: &[u8]) -> Result<Vec<u8>> {
    use aes_gcm::{
        aead::{Aead, KeyInit, OsRng},
        AeadCore, Aes256Gcm,
    };

    let cipher = Aes256Gcm::new_from_slice(key.as_bytes())
        .map_err(|e| TensorError::Io(std::io::Error::other(format!("cipher init: {e}"))))?;
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|e| TensorError::Io(std::io::Error::other(format!("encrypt: {e}"))))?;

    let mut output = Vec::with_capacity(12 + ciphertext.len());
    output.extend_from_slice(&nonce);
    output.extend_from_slice(&ciphertext);
    Ok(output)
}

/// Decrypt a block encrypted with `encrypt_block`.
/// Input format: [12-byte nonce][ciphertext + 16-byte tag]
#[cfg(feature = "encryption")]
pub fn decrypt_block(key: &EncryptionKey, encrypted: &[u8]) -> Result<Vec<u8>> {
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };

    if encrypted.len() < 12 + 16 {
        return Err(TensorError::Io(std::io::Error::other(
            "encrypted block too short",
        )));
    }

    let nonce = Nonce::from_slice(&encrypted[..12]);
    let ciphertext = &encrypted[12..];

    let cipher = Aes256Gcm::new_from_slice(key.as_bytes())
        .map_err(|e| TensorError::Io(std::io::Error::other(format!("cipher init: {e}"))))?;

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| TensorError::Io(std::io::Error::other(format!("decrypt: {e}"))))
}

/// No-op encrypt when encryption feature is disabled.
#[cfg(not(feature = "encryption"))]
pub fn encrypt_block(_key: &EncryptionKey, plaintext: &[u8]) -> Result<Vec<u8>> {
    Ok(plaintext.to_vec())
}

/// No-op decrypt when encryption feature is disabled.
#[cfg(not(feature = "encryption"))]
pub fn decrypt_block(_key: &EncryptionKey, encrypted: &[u8]) -> Result<Vec<u8>> {
    Ok(encrypted.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_from_passphrase_is_deterministic() {
        let k1 = EncryptionKey::from_passphrase("my-secret-key");
        let k2 = EncryptionKey::from_passphrase("my-secret-key");
        assert_eq!(k1.as_bytes(), k2.as_bytes());
    }

    #[test]
    fn key_from_different_passphrases_differ() {
        let k1 = EncryptionKey::from_passphrase("key-a");
        let k2 = EncryptionKey::from_passphrase("key-b");
        assert_ne!(k1.as_bytes(), k2.as_bytes());
    }

    #[test]
    fn sha256_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let mut h = Sha256::new();
        h.update(b"");
        let result = h.finalize();
        assert_eq!(
            hex_encode(&result),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_abc() {
        let mut h = Sha256::new();
        h.update(b"abc");
        let result = h.finalize();
        assert_eq!(
            hex_encode(&result),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn encrypt_decrypt_roundtrip() {
        let key = EncryptionKey::from_passphrase("test-key");
        let plaintext = b"Hello, TensorDB encryption!";
        let encrypted = encrypt_block(&key, plaintext).unwrap();
        assert_ne!(&encrypted[..], plaintext);
        assert!(encrypted.len() > plaintext.len()); // nonce + tag overhead
        let decrypted = decrypt_block(&key, &encrypted).unwrap();
        assert_eq!(&decrypted, plaintext);
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn wrong_key_fails_decrypt() {
        let key1 = EncryptionKey::from_passphrase("correct-key");
        let key2 = EncryptionKey::from_passphrase("wrong-key");
        let encrypted = encrypt_block(&key1, b"secret data").unwrap();
        assert!(decrypt_block(&key2, &encrypted).is_err());
    }

    #[cfg(feature = "encryption")]
    #[test]
    fn tampered_data_fails_decrypt() {
        let key = EncryptionKey::from_passphrase("test-key");
        let mut encrypted = encrypt_block(&key, b"important data").unwrap();
        // Flip a bit in the ciphertext
        let last = encrypted.len() - 1;
        encrypted[last] ^= 0x01;
        assert!(decrypt_block(&key, &encrypted).is_err());
    }

    fn hex_encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{b:02x}")).collect()
    }
}
