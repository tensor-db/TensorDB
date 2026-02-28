use std::time::Duration;

use tempfile::TempDir;
use tensordb::{Config, Database};

#[test]
fn readme_quickstart_sql_examples_execute() {
    let dir = TempDir::new().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();

    db.sql("CREATE TABLE events (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("CREATE TABLE accounts (id INTEGER PRIMARY KEY, name TEXT NOT NULL, balance REAL);")
        .unwrap();

    db.sql("INSERT INTO events (pk, doc) VALUES ('evt-1', '{\"type\":\"signup\",\"user\":\"alice\"}');")
        .unwrap();
    db.sql("INSERT INTO events (pk, doc) VALUES ('evt-2', '{\"type\":\"purchase\",\"user\":\"bob\",\"amount\":49.99}');")
        .unwrap();
    db.sql("INSERT INTO accounts (id, name, balance) VALUES (1, 'alice', 1000.0);")
        .unwrap();
    db.sql("INSERT INTO accounts (id, name, balance) VALUES (2, 'bob', 500.0);")
        .unwrap();

    db.sql("SELECT pk, doc FROM events ORDER BY pk LIMIT 10;")
        .unwrap();
    db.sql("SELECT pk, doc FROM events JOIN accounts ON events.pk=accounts.pk ORDER BY pk ASC;")
        .unwrap();
    db.sql("UPDATE events SET doc = '{\"type\":\"refund\",\"user\":\"bob\"}' WHERE pk = 'evt-2';")
        .unwrap();
    db.sql("DELETE FROM events WHERE pk = 'evt-1';").unwrap();
    db.sql(
        "BEGIN;
         INSERT INTO events (pk, doc) VALUES ('evt-3', '{\"type\":\"refund\",\"user\":\"bob\"}');
         COMMIT;",
    )
    .unwrap();
}

#[test]
fn readme_ai_example_generates_insight() {
    let dir = TempDir::new().unwrap();
    let cfg = Config {
        ai_auto_insights: true,
        shard_count: 1,
        ..Config::default()
    };
    let db = Database::open(dir.path(), cfg).unwrap();

    db.sql("CREATE TABLE events (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("INSERT INTO events (pk, doc) VALUES ('evt-3', '{\"type\":\"refund\",\"note\":\"chargeback dispute\"}');")
        .unwrap();

    let mut insights = Vec::new();
    for _ in 0..80 {
        insights = db
            .ai_insights_for_key(b"table/events/evt-3", Some(5))
            .unwrap();
        if !insights.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    assert!(
        !insights.is_empty(),
        "README AI example should produce insights"
    );
}
