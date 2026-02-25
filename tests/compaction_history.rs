use std::collections::HashMap;

use tempfile::tempdir;

use spectradb::config::Config;
use spectradb::Database;

#[test]
fn compaction_preserves_multiversion_history() {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 1,
        memtable_max_bytes: 2048,
        compaction_l0_threshold: 2,
        ..Config::default()
    };

    let mut history: HashMap<String, Vec<(u64, u64)>> = HashMap::new();

    {
        let db = Database::open(dir.path(), cfg.clone()).unwrap();

        for ver in 0u64..40 {
            for key_idx in 0u64..5 {
                let key = format!("acct/{key_idx}");
                let payload = format!("{{\"k\":{key_idx},\"v\":{ver}}}").into_bytes();
                let commit_ts = db
                    .put(key.as_bytes(), payload, ver * 100, (ver + 1) * 100, Some(1))
                    .unwrap();
                history.entry(key).or_default().push((commit_ts, ver));
            }
        }

        for key_idx in 0u64..5 {
            let key = format!("acct/{key_idx}");
            for ver in (0u64..40).step_by(7) {
                let valid_at = ver * 100 + 1;
                let got = db
                    .get(key.as_bytes(), None, Some(valid_at))
                    .unwrap()
                    .unwrap();
                assert_eq!(got, format!("{{\"k\":{key_idx},\"v\":{ver}}}").into_bytes());
            }

            let entries = history.get(&key).unwrap();
            for (idx, (commit_ts, ver)) in entries.iter().enumerate().step_by(9) {
                let valid_at = *ver * 100 + 1;
                let got = db
                    .get(key.as_bytes(), Some(*commit_ts), Some(valid_at))
                    .unwrap()
                    .unwrap();
                assert_eq!(got, format!("{{\"k\":{key_idx},\"v\":{ver}}}").into_bytes());

                if idx > 0 {
                    let prev_ver = entries[idx - 1].1;
                    let got_prev = db
                        .get(
                            key.as_bytes(),
                            Some(*commit_ts - 1),
                            Some(prev_ver * 100 + 1),
                        )
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        got_prev,
                        format!("{{\"k\":{key_idx},\"v\":{prev_ver}}}").into_bytes()
                    );
                }
            }
        }
    }

    let db = Database::open(dir.path(), cfg).unwrap();
    for key_idx in 0u64..5 {
        let key = format!("acct/{key_idx}");
        for ver in (0u64..40).step_by(8) {
            let valid_at = ver * 100 + 1;
            let got = db
                .get(key.as_bytes(), None, Some(valid_at))
                .unwrap()
                .unwrap();
            assert_eq!(got, format!("{{\"k\":{key_idx},\"v\":{ver}}}").into_bytes());
        }
    }
}
