use tempfile::tempdir;

use spectradb::config::Config;
use spectradb::ledger::internal_key::{encode_internal_key, KIND_PUT};
use spectradb::ledger::record::{FactMetadata, FactValue, FactWrite};
use spectradb::storage::wal::Wal;
use spectradb::Database;

#[test]
fn wal_replay_recovers_records() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("recover.wal");

    let mut wal = Wal::open(&wal_path, 4).unwrap();
    for i in 0..10u64 {
        let internal = encode_internal_key(format!("k{i}").as_bytes(), i + 1, KIND_PUT);
        let fact = FactValue {
            doc: format!("v{i}").into_bytes(),
            valid_from: 0,
            valid_to: u64::MAX,
        }
        .encode();
        wal.append(&FactWrite {
            internal_key: internal,
            fact,
            metadata: FactMetadata::default(),
        })
        .unwrap();
    }
    wal.sync().unwrap();

    let replayed = Wal::replay(&wal_path).unwrap();
    assert_eq!(replayed.len(), 10);
}

#[test]
fn db_reopens_and_recovers_from_wal() {
    let dir = tempdir().unwrap();
    let cfg = Config {
        memtable_max_bytes: 16 * 1024 * 1024,
        ..Config::default()
    };

    {
        let db = Database::open(dir.path(), cfg.clone()).unwrap();
        for i in 0..128 {
            let key = format!("u/{i}");
            let value = format!("{{\"v\":{i}}}").into_bytes();
            db.put(key.as_bytes(), value, 0, u64::MAX, Some(1)).unwrap();
        }
    }

    let db = Database::open(dir.path(), cfg).unwrap();
    for i in 0..128 {
        let key = format!("u/{i}");
        let got = db.get(key.as_bytes(), None, None).unwrap();
        assert!(got.is_some());
    }
}
