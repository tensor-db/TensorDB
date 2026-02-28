use tempfile::tempdir;

use tensordb::config::Config;
use tensordb::Database;

#[test]
fn reads_respect_as_of_commit_ts() {
    let dir = tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();

    let k = b"acct/1";
    let t1 = db
        .put(k, br#"{"balance":10}"#.to_vec(), 0, u64::MAX, Some(1))
        .unwrap();
    let t2 = db
        .put(k, br#"{"balance":20}"#.to_vec(), 0, u64::MAX, Some(1))
        .unwrap();
    let t3 = db
        .put(k, br#"{"balance":30}"#.to_vec(), 0, u64::MAX, Some(1))
        .unwrap();

    assert!(t1 < t2 && t2 < t3);

    let v1 = db.get(k, Some(t1), None).unwrap().unwrap();
    let v2 = db.get(k, Some(t2), None).unwrap().unwrap();
    let v3 = db.get(k, Some(t3), None).unwrap().unwrap();

    assert_eq!(v1, br#"{"balance":10}"#.to_vec());
    assert_eq!(v2, br#"{"balance":20}"#.to_vec());
    assert_eq!(v3, br#"{"balance":30}"#.to_vec());
}
