use tempfile::tempdir;

use spectradb::config::Config;
use spectradb::Database;

#[test]
fn temporal_correctness_across_flush_compaction_and_reopen() {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 1,
        memtable_max_bytes: 1024,
        compaction_l0_threshold: 3,
        ..Config::default()
    };

    let key = b"acct/temporal";
    let mut commits = Vec::new();
    {
        let db = Database::open(dir.path(), cfg.clone()).unwrap();

        for i in 0u64..120 {
            let ts = db
                .put(
                    key,
                    format!("{{\"v\":{i}}}").into_bytes(),
                    i * 10,
                    (i + 1) * 10,
                    Some(1),
                )
                .unwrap();
            commits.push(ts);
        }

        for i in (0u64..120).step_by(13) {
            let got = db
                .get(key, Some(commits[i as usize]), Some(i * 10 + 1))
                .unwrap();
            assert_eq!(got, Some(format!("{{\"v\":{i}}}").into_bytes()));
        }
    }

    let db = Database::open(dir.path(), cfg).unwrap();

    for i in (0u64..120).step_by(11) {
        let got = db.get(key, None, Some(i * 10 + 1)).unwrap();
        assert_eq!(got, Some(format!("{{\"v\":{i}}}").into_bytes()));
    }

    let none = db.get(key, None, Some(120 * 10 + 1)).unwrap();
    assert!(none.is_none());
}
