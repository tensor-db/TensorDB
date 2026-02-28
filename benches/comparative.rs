use criterion::{criterion_group, criterion_main, Criterion};
use tempfile::TempDir;
use tensordb::{Config, Database};

fn bench_tensordb_point_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 4,
        fast_write_enabled: true,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    let mut i = 0u64;
    c.bench_function("tensordb_point_write", |b| {
        b.iter(|| {
            let key = format!("bench/{i:08}");
            let doc = format!("{{\"n\":{i}}}");
            let _ = db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1));
            i += 1;
        })
    });
}

fn bench_tensordb_point_write_channel(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 4,
        fast_write_enabled: false,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    let mut i = 0u64;
    c.bench_function("tensordb_point_write_channel", |b| {
        b.iter(|| {
            let key = format!("bench/{i:08}");
            let doc = format!("{{\"n\":{i}}}");
            let _ = db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1));
            i += 1;
        })
    });
}

fn bench_tensordb_point_read(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 4,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    // Pre-populate
    for i in 0..10_000u64 {
        let key = format!("bench/{i:08}");
        let doc = format!("{{\"n\":{i}}}");
        let _ = db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1));
    }

    let mut i = 0u64;
    c.bench_function("tensordb_point_read", |b| {
        b.iter(|| {
            let key = format!("bench/{:08}", i % 10_000);
            let _ = db.get(key.as_bytes(), None, None);
            i += 1;
        })
    });
}

fn bench_tensordb_scan(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 4,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    // Pre-populate
    for i in 0..1_000u64 {
        let key = format!("scan/{i:06}");
        let doc = format!("{{\"n\":{i}}}");
        let _ = db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1));
    }

    c.bench_function("tensordb_prefix_scan_1000", |b| {
        b.iter(|| {
            let _ = db.scan_prefix(b"scan/", None, None, None);
        })
    });
}

fn bench_tensordb_sql_select(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 2,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    db.sql("CREATE TABLE bench (pk TEXT PRIMARY KEY)").unwrap();
    for i in 0..100 {
        let q = format!("INSERT INTO bench (pk, doc) VALUES ('k{i}', '{{\"val\":{i}}}')");
        db.sql(&q).unwrap();
    }

    c.bench_function("tensordb_sql_select_100", |b| {
        b.iter(|| {
            let _ = db.sql("SELECT pk, doc FROM bench");
        })
    });
}

// --- SQLite comparison benchmarks ---

fn bench_sqlite_point_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("sqlite.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         CREATE TABLE bench (key TEXT PRIMARY KEY, doc TEXT);",
    )
    .unwrap();

    let mut i = 0u64;
    c.bench_function("sqlite_point_write", |b| {
        b.iter(|| {
            let key = format!("bench/{i:08}");
            let doc = format!("{{\"n\":{i}}}");
            conn.execute(
                "INSERT OR REPLACE INTO bench (key, doc) VALUES (?1, ?2)",
                [&key, &doc],
            )
            .unwrap();
            i += 1;
        })
    });
}

fn bench_sqlite_point_read(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("sqlite.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         CREATE TABLE bench (key TEXT PRIMARY KEY, doc TEXT);",
    )
    .unwrap();

    // Pre-populate
    for i in 0..10_000u64 {
        let key = format!("bench/{i:08}");
        let doc = format!("{{\"n\":{i}}}");
        conn.execute("INSERT INTO bench (key, doc) VALUES (?1, ?2)", [&key, &doc])
            .unwrap();
    }

    let mut stmt = conn
        .prepare("SELECT doc FROM bench WHERE key = ?1")
        .unwrap();
    let mut i = 0u64;
    c.bench_function("sqlite_point_read", |b| {
        b.iter(|| {
            let key = format!("bench/{:08}", i % 10_000);
            let _: Option<String> = stmt.query_row([&key], |row| row.get(0)).ok();
            i += 1;
        })
    });
}

fn bench_sqlite_scan(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("sqlite.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         CREATE TABLE bench (key TEXT PRIMARY KEY, doc TEXT);",
    )
    .unwrap();

    for i in 0..1_000u64 {
        let key = format!("scan/{i:06}");
        let doc = format!("{{\"n\":{i}}}");
        conn.execute("INSERT INTO bench (key, doc) VALUES (?1, ?2)", [&key, &doc])
            .unwrap();
    }

    c.bench_function("sqlite_prefix_scan_1000", |b| {
        b.iter(|| {
            let mut stmt = conn
                .prepare("SELECT key, doc FROM bench WHERE key LIKE 'scan/%'")
                .unwrap();
            let rows: Vec<(String, String)> = stmt
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect();
            std::hint::black_box(rows);
        })
    });
}

fn bench_tensordb_mixed_workload(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 4,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    // Pre-populate
    for i in 0..5_000u64 {
        let key = format!("mixed/{i:08}");
        let doc = format!("{{\"n\":{i}}}");
        let _ = db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1));
    }

    let mut i = 5_000u64;
    c.bench_function("tensordb_mixed_80read_20write", |b| {
        b.iter(|| {
            if i.is_multiple_of(5) {
                // 20% writes
                let key = format!("mixed/{i:08}");
                let doc = format!("{{\"n\":{i}}}");
                let _ = db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1));
            } else {
                // 80% reads
                let key = format!("mixed/{:08}", i % 5_000);
                let _ = db.get(key.as_bytes(), None, None);
            }
            i += 1;
        })
    });
}

fn bench_sqlite_mixed_workload(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("sqlite.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         CREATE TABLE bench (key TEXT PRIMARY KEY, doc TEXT);",
    )
    .unwrap();

    for i in 0..5_000u64 {
        let key = format!("mixed/{i:08}");
        let doc = format!("{{\"n\":{i}}}");
        conn.execute("INSERT INTO bench (key, doc) VALUES (?1, ?2)", [&key, &doc])
            .unwrap();
    }

    let mut i = 5_000u64;
    c.bench_function("sqlite_mixed_80read_20write", |b| {
        b.iter(|| {
            if i.is_multiple_of(5) {
                let key = format!("mixed/{i:08}");
                let doc = format!("{{\"n\":{i}}}");
                conn.execute(
                    "INSERT OR REPLACE INTO bench (key, doc) VALUES (?1, ?2)",
                    [&key, &doc],
                )
                .unwrap();
            } else {
                let key = format!("mixed/{:08}", i % 5_000);
                let mut stmt = conn
                    .prepare_cached("SELECT doc FROM bench WHERE key = ?1")
                    .unwrap();
                let _: Option<String> = stmt.query_row([&key], |row| row.get(0)).ok();
            }
            i += 1;
        })
    });
}

criterion_group!(
    benches,
    bench_tensordb_point_write,
    bench_tensordb_point_write_channel,
    bench_sqlite_point_write,
    bench_tensordb_point_read,
    bench_sqlite_point_read,
    bench_tensordb_scan,
    bench_sqlite_scan,
    bench_tensordb_sql_select,
    bench_tensordb_mixed_workload,
    bench_sqlite_mixed_workload,
);
criterion_main!(benches);
