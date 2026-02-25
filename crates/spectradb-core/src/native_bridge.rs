use std::sync::Arc;

use crate::storage::bloom::BloomFilter;

pub trait Hasher: Send + Sync {
    fn hash64(&self, bytes: &[u8]) -> u64;
    fn name(&self) -> &'static str;
}

pub trait Compressor: Send + Sync {
    fn compress(&self, input: &[u8]) -> Vec<u8>;
    fn decompress(&self, input: &[u8], _out_len: usize) -> Vec<u8>;
}

pub trait BloomProbe: Send + Sync {
    fn may_contain(&self, filter: &[u8], key: &[u8]) -> bool;
}

#[derive(Default)]
pub struct RustHasher;

impl Hasher for RustHasher {
    fn hash64(&self, bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= h >> 32;
        }
        h
    }

    fn name(&self) -> &'static str {
        "rust-fnv64-mix"
    }
}

pub struct NoopCompressor;

impl Compressor for NoopCompressor {
    fn compress(&self, input: &[u8]) -> Vec<u8> {
        input.to_vec()
    }

    fn decompress(&self, input: &[u8], _out_len: usize) -> Vec<u8> {
        input.to_vec()
    }
}

pub struct RustBloomProbe;

impl BloomProbe for RustBloomProbe {
    fn may_contain(&self, filter: &[u8], key: &[u8]) -> bool {
        let hasher = RustHasher;
        BloomFilter::decode(filter)
            .map(|f| f.may_contain(key, &hasher))
            .unwrap_or(false)
    }
}

#[cfg(feature = "native")]
pub struct NativeHasher;

#[cfg(feature = "native")]
impl Hasher for NativeHasher {
    fn hash64(&self, bytes: &[u8]) -> u64 {
        spectradb_native::hash64(bytes)
    }

    fn name(&self) -> &'static str {
        "native-demo64"
    }
}

pub fn build_hasher() -> Arc<dyn Hasher + Send + Sync> {
    #[cfg(feature = "native")]
    {
        return Arc::new(NativeHasher);
    }

    #[cfg(not(feature = "native"))]
    {
        Arc::new(RustHasher)
    }
}

#[cfg(feature = "native")]
pub fn native_hash_call_count() -> u64 {
    spectradb_native::hash_call_count()
}

#[cfg(not(feature = "native"))]
pub fn native_hash_call_count() -> u64 {
    0
}

#[cfg(feature = "native")]
pub fn reset_native_hash_call_count() {
    spectradb_native::reset_hash_call_count()
}

#[cfg(not(feature = "native"))]
pub fn reset_native_hash_call_count() {}
