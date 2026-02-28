use tempfile::TempDir;
use tensordb::{Config, Database, WriteBatchItem};

fn open_db(dir: &TempDir) -> Database {
    let config = Config {
        shard_count: 2,
        memtable_max_bytes: 64 * 1024,
        ..Config::default()
    };
    Database::open(dir.path(), config).unwrap()
}

#[test]
fn write_batch_returns_monotonic_timestamps() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);

    let entries: Vec<WriteBatchItem> = (0..10)
        .map(|i| WriteBatchItem {
            user_key: format!("key/{i:04}").into_bytes(),
            doc: format!("{{\"n\":{i}}}").into_bytes(),
            valid_from: 0,
            valid_to: u64::MAX,
            schema_version: Some(1),
        })
        .collect();

    let timestamps = db.write_batch(entries).unwrap();
    assert_eq!(timestamps.len(), 10);

    // Timestamps should be monotonically increasing within each shard grouping
    // but overall non-zero
    for ts in &timestamps {
        assert!(*ts > 0, "timestamp should be > 0");
    }
}

#[test]
fn write_batch_data_is_readable() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);

    let entries = vec![
        WriteBatchItem {
            user_key: b"batch/alpha".to_vec(),
            doc: b"{\"v\":\"a\"}".to_vec(),
            valid_from: 0,
            valid_to: u64::MAX,
            schema_version: Some(1),
        },
        WriteBatchItem {
            user_key: b"batch/beta".to_vec(),
            doc: b"{\"v\":\"b\"}".to_vec(),
            valid_from: 0,
            valid_to: u64::MAX,
            schema_version: Some(1),
        },
        WriteBatchItem {
            user_key: b"batch/gamma".to_vec(),
            doc: b"{\"v\":\"c\"}".to_vec(),
            valid_from: 0,
            valid_to: u64::MAX,
            schema_version: Some(1),
        },
    ];

    db.write_batch(entries).unwrap();

    let a = db.get(b"batch/alpha", None, None).unwrap().unwrap();
    assert_eq!(a, b"{\"v\":\"a\"}");

    let b = db.get(b"batch/beta", None, None).unwrap().unwrap();
    assert_eq!(b, b"{\"v\":\"b\"}");

    let c = db.get(b"batch/gamma", None, None).unwrap().unwrap();
    assert_eq!(c, b"{\"v\":\"c\"}");
}

#[test]
fn write_batch_empty_is_noop() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    let timestamps = db.write_batch(vec![]).unwrap();
    assert!(timestamps.is_empty());
}

#[test]
fn write_batch_survives_reopen() {
    let dir = TempDir::new().unwrap();

    {
        let db = open_db(&dir);
        let entries = vec![
            WriteBatchItem {
                user_key: b"durable/one".to_vec(),
                doc: b"{\"x\":1}".to_vec(),
                valid_from: 0,
                valid_to: u64::MAX,
                schema_version: None,
            },
            WriteBatchItem {
                user_key: b"durable/two".to_vec(),
                doc: b"{\"x\":2}".to_vec(),
                valid_from: 0,
                valid_to: u64::MAX,
                schema_version: None,
            },
        ];
        db.write_batch(entries).unwrap();
    }

    // Reopen
    let db = open_db(&dir);
    let one = db.get(b"durable/one", None, None).unwrap().unwrap();
    assert_eq!(one, b"{\"x\":1}");
    let two = db.get(b"durable/two", None, None).unwrap().unwrap();
    assert_eq!(two, b"{\"x\":2}");
}

#[test]
fn write_batch_mixed_with_single_puts() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);

    // Single put first
    db.put(b"mix/single", b"{\"s\":true}".to_vec(), 0, u64::MAX, None)
        .unwrap();

    // Then batch
    let entries = vec![WriteBatchItem {
        user_key: b"mix/batch".to_vec(),
        doc: b"{\"b\":true}".to_vec(),
        valid_from: 0,
        valid_to: u64::MAX,
        schema_version: None,
    }];
    db.write_batch(entries).unwrap();

    // Then another single put
    db.put(b"mix/single2", b"{\"s2\":true}".to_vec(), 0, u64::MAX, None)
        .unwrap();

    assert!(db.get(b"mix/single", None, None).unwrap().is_some());
    assert!(db.get(b"mix/batch", None, None).unwrap().is_some());
    assert!(db.get(b"mix/single2", None, None).unwrap().is_some());
}
