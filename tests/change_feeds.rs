use spectradb::{Config, Database};
use std::time::Duration;
use tempfile::TempDir;

#[test]
fn subscribe_receives_matching_events() {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 1,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    let rx = db.subscribe(b"feed/");

    // Write some matching data
    db.put(b"feed/key1", b"{\"v\":1}".to_vec(), 0, u64::MAX, None)
        .unwrap();
    db.put(b"feed/key2", b"{\"v\":2}".to_vec(), 0, u64::MAX, None)
        .unwrap();
    // Write non-matching
    db.put(
        b"other/key3",
        b"{\"v\":3}".to_vec(),
        0,
        u64::MAX,
        None,
    )
    .unwrap();

    // Should receive 2 events (matching prefix)
    let mut events = Vec::new();
    while let Ok(evt) = rx.recv_timeout(Duration::from_millis(100)) {
        events.push(evt);
    }
    assert_eq!(
        events.len(),
        2,
        "expected 2 events for prefix 'feed/', got {}",
        events.len()
    );
    assert_eq!(events[0].user_key, b"feed/key1");
    assert_eq!(events[1].user_key, b"feed/key2");
}

#[test]
fn subscribe_empty_prefix_gets_all() {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 1,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    let rx = db.subscribe(b"");

    db.put(b"a/1", b"{}".to_vec(), 0, u64::MAX, None).unwrap();
    db.put(b"b/2", b"{}".to_vec(), 0, u64::MAX, None).unwrap();

    let mut events = Vec::new();
    while let Ok(evt) = rx.recv_timeout(Duration::from_millis(100)) {
        events.push(evt);
    }
    assert_eq!(events.len(), 2);
}

#[test]
fn subscribe_no_events_when_no_match() {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 1,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    let rx = db.subscribe(b"nonexistent/");

    db.put(b"other/key", b"{}".to_vec(), 0, u64::MAX, None)
        .unwrap();

    let evt = rx.recv_timeout(Duration::from_millis(100));
    assert!(evt.is_err(), "should not receive event for non-matching prefix");
}
