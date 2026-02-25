use tempfile::tempdir;

use spectradb::config::Config;
use spectradb::Database;

#[test]
fn prefix_scan_respects_mvcc_and_valid_at_filters() {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 1,
        ..Config::default()
    };
    let db = Database::open(dir.path(), cfg).unwrap();

    let t_a1 = db
        .put(b"acct/1", br#"{"v":1}"#.to_vec(), 0, 100, Some(1))
        .unwrap();
    let t_a2 = db
        .put(b"acct/1", br#"{"v":2}"#.to_vec(), 100, u64::MAX, Some(1))
        .unwrap();
    let t_b1 = db
        .put(b"acct/2", br#"{"v":10}"#.to_vec(), 0, u64::MAX, Some(1))
        .unwrap();
    let t_b2 = db
        .put(b"acct/2", br#"{"v":20}"#.to_vec(), 0, u64::MAX, Some(1))
        .unwrap();
    let t_c1 = db
        .put(b"acct/3", br#"{"v":30}"#.to_vec(), 0, 50, Some(1))
        .unwrap();
    let _ = db
        .put(b"other/1", br#"{"v":999}"#.to_vec(), 0, u64::MAX, Some(1))
        .unwrap();

    let rows = db.scan_prefix(b"acct/", None, Some(25), None).unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].user_key, b"acct/1".to_vec());
    assert_eq!(rows[0].doc, br#"{"v":1}"#.to_vec());
    assert_eq!(rows[0].commit_ts, t_a1);
    assert_eq!(rows[1].user_key, b"acct/2".to_vec());
    assert_eq!(rows[1].doc, br#"{"v":20}"#.to_vec());
    assert_eq!(rows[1].commit_ts, t_b2);
    assert_eq!(rows[2].user_key, b"acct/3".to_vec());
    assert_eq!(rows[2].doc, br#"{"v":30}"#.to_vec());
    assert_eq!(rows[2].commit_ts, t_c1);

    let rows_as_of = db
        .scan_prefix(b"acct/", Some(t_b1), Some(150), None)
        .unwrap();
    assert_eq!(rows_as_of.len(), 2);
    assert_eq!(rows_as_of[0].user_key, b"acct/1".to_vec());
    assert_eq!(rows_as_of[0].doc, br#"{"v":2}"#.to_vec());
    assert_eq!(rows_as_of[0].commit_ts, t_a2);
    assert_eq!(rows_as_of[1].user_key, b"acct/2".to_vec());
    assert_eq!(rows_as_of[1].doc, br#"{"v":10}"#.to_vec());
    assert_eq!(rows_as_of[1].commit_ts, t_b1);

    let none = db
        .scan_prefix(b"acct/", Some(t_a1), Some(150), None)
        .unwrap();
    assert!(none.is_empty());
}

#[test]
fn prefix_scan_merges_globally_sorted_and_applies_limit_after_merge() {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 8,
        ..Config::default()
    };
    let db = Database::open(dir.path(), cfg.clone()).unwrap();

    for (i, key) in ["user/d", "user/a", "user/e", "user/b", "user/c"]
        .into_iter()
        .enumerate()
    {
        let value = format!("{{\"v\":{i}}}").into_bytes();
        let _ = db.put(key.as_bytes(), value, 0, u64::MAX, Some(1)).unwrap();
    }
    let commit_b_new = db
        .put(b"user/b", br#"{"v":99}"#.to_vec(), 0, u64::MAX, Some(1))
        .unwrap();

    let gets_before = db.stats().unwrap().gets;
    let rows = db.scan_prefix(b"user/", None, None, Some(3)).unwrap();
    let gets_after = db.stats().unwrap().gets;

    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].user_key, b"user/a".to_vec());
    assert_eq!(rows[1].user_key, b"user/b".to_vec());
    assert_eq!(rows[2].user_key, b"user/c".to_vec());
    assert_eq!(rows[1].doc, br#"{"v":99}"#.to_vec());
    assert_eq!(rows[1].commit_ts, commit_b_new);

    // One fan-out scan should register at least one read in shard metrics.
    assert!(gets_after > gets_before);
    assert!(gets_after >= gets_before + cfg.shard_count as u64);
}
