use tempfile::tempdir;

use spectradb::config::Config;
use spectradb::Database;

#[test]
fn repeated_reopen_cycles_preserve_latest_state() {
    let dir = tempdir().unwrap();
    let cfg = Config {
        shard_count: 4,
        memtable_max_bytes: 32768,
        compaction_l0_threshold: 3,
        ..Config::default()
    };

    const CYCLES: u64 = 20;
    const KEYS: u64 = 64;

    for cycle in 0..CYCLES {
        {
            let db = Database::open(dir.path(), cfg.clone()).unwrap();
            for k in 0..KEYS {
                let key = format!("acct/{k:03}");
                let payload = format!("{{\"cycle\":{cycle},\"k\":{k}}}").into_bytes();
                db.put(
                    key.as_bytes(),
                    payload,
                    cycle * 100,
                    (cycle + 1) * 100,
                    Some(1),
                )
                .unwrap();
            }

            for k in (0..KEYS).step_by(9) {
                let key = format!("acct/{k:03}");
                let got = db.get(key.as_bytes(), None, Some(cycle * 100 + 1)).unwrap();
                assert_eq!(
                    got,
                    Some(format!("{{\"cycle\":{cycle},\"k\":{k}}}").into_bytes())
                );
            }
        }

        let db = Database::open(dir.path(), cfg.clone()).unwrap();
        for k in (0..KEYS).step_by(11) {
            let key = format!("acct/{k:03}");
            let got = db.get(key.as_bytes(), None, Some(cycle * 100 + 1)).unwrap();
            assert_eq!(
                got,
                Some(format!("{{\"cycle\":{cycle},\"k\":{k}}}").into_bytes())
            );
        }
    }
}
