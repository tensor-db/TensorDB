use tempfile::TempDir;
use tensordb::{Config, Database};

fn padding(ch: char, len: usize) -> String {
    std::iter::repeat_n(ch, len).collect()
}

#[test]
fn leveled_compaction_triggers_on_l0_threshold() {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 1,
        memtable_max_bytes: 1024,
        compaction_l0_threshold: 3,
        compaction_max_levels: 4,
        compaction_l1_target_bytes: 10 * 1024,
        compaction_size_ratio: 10,
        sstable_max_file_bytes: 64 * 1024 * 1024,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    let pad = padding('x', 100);
    for i in 0..200 {
        let key = format!("compact/{i:06}");
        let doc = format!("{{\"n\":{i},\"padding\":\"{pad}\"}}");
        db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1))
            .unwrap();
    }

    for i in 0..200 {
        let key = format!("compact/{i:06}");
        let val = db.get(key.as_bytes(), None, None).unwrap();
        assert!(val.is_some(), "key {key} should exist after compaction");
    }

    let stats = db.stats().unwrap();
    assert!(
        stats.compactions > 0,
        "expected at least 1 compaction, got {}",
        stats.compactions
    );
}

#[test]
fn compaction_preserves_temporal_versions() {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 1,
        memtable_max_bytes: 1024,
        compaction_l0_threshold: 2,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    let ts1 = db
        .put(b"temporal/key", b"{\"v\":1}".to_vec(), 100, 200, Some(1))
        .unwrap();
    let ts2 = db
        .put(b"temporal/key", b"{\"v\":2}".to_vec(), 200, 300, Some(1))
        .unwrap();

    let pad = padding('y', 50);
    for i in 0..100 {
        let key = format!("filler/{i:06}");
        let doc = format!("{{\"pad\":\"{pad}\"}}");
        db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1))
            .unwrap();
    }

    let ts3 = db
        .put(
            b"temporal/key",
            b"{\"v\":3}".to_vec(),
            300,
            u64::MAX,
            Some(1),
        )
        .unwrap();

    let latest = db.get(b"temporal/key", None, None).unwrap().unwrap();
    assert_eq!(latest, b"{\"v\":3}");

    let v1 = db.get(b"temporal/key", Some(ts1), None).unwrap().unwrap();
    assert_eq!(v1, b"{\"v\":1}");

    let v2 = db.get(b"temporal/key", Some(ts2), None).unwrap().unwrap();
    assert_eq!(v2, b"{\"v\":2}");

    let vat = db
        .get(b"temporal/key", Some(ts3), Some(150))
        .unwrap()
        .unwrap();
    assert_eq!(vat, b"{\"v\":1}");

    let vat2 = db
        .get(b"temporal/key", Some(ts3), Some(250))
        .unwrap()
        .unwrap();
    assert_eq!(vat2, b"{\"v\":2}");
}

#[test]
fn compaction_data_survives_reopen() {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 1,
        memtable_max_bytes: 1024,
        compaction_l0_threshold: 2,
        ..Config::default()
    };

    {
        let db = Database::open(dir.path(), config.clone()).unwrap();
        let pad = padding('z', 80);
        for i in 0..100 {
            let key = format!("reopen/{i:04}");
            let doc = format!("{{\"n\":{i},\"data\":\"{pad}\"}}");
            db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1))
                .unwrap();
        }
    }

    let db = Database::open(dir.path(), config).unwrap();
    for i in 0..100 {
        let key = format!("reopen/{i:04}");
        let val = db.get(key.as_bytes(), None, None).unwrap();
        assert!(
            val.is_some(),
            "key {key} should survive reopen after compaction"
        );
    }
}

#[test]
fn scan_prefix_works_after_compaction() {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 1,
        memtable_max_bytes: 1024,
        compaction_l0_threshold: 2,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    let pad_a = padding('a', 100);
    for i in 0..50 {
        let key = format!("scan/{i:04}");
        let doc = format!("{{\"i\":{i},\"fill\":\"{pad_a}\"}}");
        db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1))
            .unwrap();
    }

    let pad_b = padding('b', 100);
    for i in 0..50 {
        let key = format!("other/{i:04}");
        let doc = format!("{{\"o\":{i},\"fill\":\"{pad_b}\"}}");
        db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1))
            .unwrap();
    }

    let rows = db.scan_prefix(b"scan/", None, None, None).unwrap();
    assert_eq!(rows.len(), 50, "scan should return all 50 keys");
}
