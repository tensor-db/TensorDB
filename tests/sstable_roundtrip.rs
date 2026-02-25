use tempfile::tempdir;

use spectradb::ledger::internal_key::{encode_internal_key, KIND_PUT};
use spectradb::ledger::record::FactValue;
use spectradb::native_bridge::build_hasher;
use spectradb::storage::sstable::{build_sstable, SsTableReader};

#[test]
fn sstable_roundtrip_point_reads_and_bloom() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("roundtrip.sst");
    let hasher = build_hasher();

    let mut entries = Vec::new();
    for i in 0..200u64 {
        let user = format!("k{i:04}");
        let key = encode_internal_key(user.as_bytes(), i + 1, KIND_PUT);
        let value = FactValue {
            doc: format!("v{i}").into_bytes(),
            valid_from: 0,
            valid_to: u64::MAX,
        }
        .encode();
        entries.push((key, value));
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    build_sstable(&path, &entries, 1024, 10, hasher.as_ref()).unwrap();
    let reader = SsTableReader::open(&path).unwrap();

    let hit = reader
        .get_visible(b"k0007", u64::MAX, None, hasher.as_ref())
        .unwrap();
    assert!(hit.value.is_some());
    assert!(hit.bloom_hit);

    let miss = reader
        .get_visible(b"missing-key", u64::MAX, None, hasher.as_ref())
        .unwrap();
    assert!(miss.value.is_none());
    assert!(!miss.bloom_hit);
}
