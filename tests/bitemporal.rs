use tempfile::tempdir;

use tensordb::config::Config;
use tensordb::Database;

#[test]
fn valid_at_filters_versions() {
    let dir = tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();

    let key = b"policy/42";
    let _ = db
        .put(key, br#"{"tier":"bronze"}"#.to_vec(), 0, 100, Some(1))
        .unwrap();
    let _ = db
        .put(key, br#"{"tier":"silver"}"#.to_vec(), 100, 200, Some(1))
        .unwrap();
    let _ = db
        .put(key, br#"{"tier":"gold"}"#.to_vec(), 200, u64::MAX, Some(1))
        .unwrap();

    let at_50 = db.get(key, None, Some(50)).unwrap().unwrap();
    let at_150 = db.get(key, None, Some(150)).unwrap().unwrap();
    let at_250 = db.get(key, None, Some(250)).unwrap().unwrap();

    assert_eq!(at_50, br#"{"tier":"bronze"}"#.to_vec());
    assert_eq!(at_150, br#"{"tier":"silver"}"#.to_vec());
    assert_eq!(at_250, br#"{"tier":"gold"}"#.to_vec());

    let none = db.get(key, None, Some(u64::MAX)).unwrap();
    assert!(none.is_none());
}

#[cfg(feature = "native")]
#[test]
fn native_hasher_path_is_invoked_and_matches_rust() {
    use tensordb::native_bridge::{
        native_hash_call_count, reset_native_hash_call_count, Hasher, RustHasher,
    };

    reset_native_hash_call_count();
    let native = tensordb::native_bridge::build_hasher();
    let rust = RustHasher;

    let inputs: [&[u8]; 4] = [b"", b"abc", b"spectra", b"0123456789abcdef"];
    for i in inputs {
        assert_eq!(native.hash64(i), rust.hash64(i));
    }
    assert!(native_hash_call_count() >= 4);
}
