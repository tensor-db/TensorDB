use std::fs;

use tempfile::tempdir;

use tensordb::ledger::internal_key::{encode_internal_key, KIND_PUT};
use tensordb::ledger::record::{FactMetadata, FactValue, FactWrite};
use tensordb::storage::wal::Wal;

fn mk_write(k: &str, commit_ts: u64, v: &str) -> FactWrite {
    FactWrite {
        internal_key: encode_internal_key(k.as_bytes(), commit_ts, KIND_PUT),
        fact: FactValue {
            doc: v.as_bytes().to_vec(),
            valid_from: 0,
            valid_to: u64::MAX,
        }
        .encode(),
        metadata: FactMetadata::default(),
        epoch: 0,
        txn_id: 0,
    }
}

#[test]
fn wal_replay_stops_on_torn_tail_record() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("fault.wal");

    let mut wal = Wal::open(&wal_path, 1).unwrap();
    wal.append(&mk_write("k1", 1, "v1")).unwrap();
    wal.append(&mk_write("k2", 2, "v2")).unwrap();
    wal.sync().unwrap();

    let mut bytes = fs::read(&wal_path).unwrap();
    bytes.truncate(bytes.len() - 1);
    fs::write(&wal_path, bytes).unwrap();

    let replayed = Wal::replay(&wal_path).unwrap();
    assert_eq!(replayed.len(), 1);
}

#[test]
fn wal_replay_stops_on_crc_mismatch() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("fault_crc.wal");

    let mut wal = Wal::open(&wal_path, 1).unwrap();
    wal.append(&mk_write("k1", 1, "value")).unwrap();
    wal.sync().unwrap();

    let mut bytes = fs::read(&wal_path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    fs::write(&wal_path, bytes).unwrap();

    // CRC mismatch now stops replay instead of returning an error.
    let replayed = Wal::replay(&wal_path).unwrap();
    assert_eq!(
        replayed.len(),
        0,
        "corrupted single record should yield zero recovered records"
    );
}

#[test]
fn wal_replay_stops_at_mid_file_crc_corruption() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Write 5 records
    {
        let mut wal = Wal::open(&wal_path, 1).unwrap();
        for i in 0..5u8 {
            let write = FactWrite {
                internal_key: vec![b'k', i, 0, 0, 0, 0, 0, 0, 0, i + 1, 0],
                fact: vec![b'v', i],
                metadata: FactMetadata::default(),
                epoch: 0,
                txn_id: 0,
            };
            wal.append(&write).unwrap();
        }
    }

    // Corrupt the payload of the 3rd record (index 2).
    // Each record: 12-byte header + variable-length payload.
    // We need to find the start of record 3 and corrupt its payload.
    let mut bytes = fs::read(&wal_path).unwrap();
    let original_len = bytes.len();

    // Walk records to find record index 2
    let mut offset = 0usize;
    for _ in 0..2 {
        // skip header (12 bytes) + payload (len from bytes 4..8)
        let len = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 12 + len;
    }
    // Now offset points to record 2's header. Corrupt its payload.
    let payload_start = offset + 12;
    // Flip a byte in the payload to cause CRC mismatch
    bytes[payload_start] ^= 0xFF;

    fs::write(&wal_path, &bytes).unwrap();
    assert_eq!(fs::read(&wal_path).unwrap().len(), original_len);

    // Replay should recover exactly 2 records (stop at corrupted 3rd)
    let recovered = Wal::replay(&wal_path).unwrap();
    assert_eq!(
        recovered.len(),
        2,
        "should recover exactly the 2 records before corruption"
    );
}
