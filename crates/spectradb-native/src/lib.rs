#[cxx::bridge]
mod ffi {
    #[namespace = "spectradb_native"]
    unsafe extern "C++" {
        include!("spectradb_native.h");

        fn native_hash64(bytes: &[u8]) -> u64;
        fn native_hash_call_count() -> u64;
        fn native_hash_reset_count();
    }
}

pub fn hash64(bytes: &[u8]) -> u64 {
    ffi::native_hash64(bytes)
}

pub fn hash_call_count() -> u64 {
    ffi::native_hash_call_count()
}

pub fn reset_hash_call_count() {
    ffi::native_hash_reset_count();
}
