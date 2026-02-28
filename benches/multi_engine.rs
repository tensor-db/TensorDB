//! Comprehensive benchmark suite: TensorDB vs SQLite vs sled vs redb
//!
//! Covers: point writes, point reads, batch writes, prefix scans, mixed workloads.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tempfile::TempDir;
use tensordb::{Config, Database};

const POPULATION: u64 = 10_000;
const BATCH_SIZE: u64 = 100;

// ---------------------------------------------------------------------------
// TensorDB benchmarks
// ---------------------------------------------------------------------------

fn tensordb_point_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = Database::open(
        dir.path(),
        Config {
            shard_count: 4,
            ..Config::default()
        },
    )
    .unwrap();

    let mut i = 0u64;
    c.bench_function("tensordb/point_write", |b| {
        b.iter(|| {
            let key = format!("bench/{i:08}");
            let doc = format!("{{\"n\":{i}}}").into_bytes();
            let _ = db.put(key.as_bytes(), doc, 0, u64::MAX, Some(1));
            i += 1;
        })
    });
}

fn tensordb_point_read(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = Database::open(
        dir.path(),
        Config {
            shard_count: 4,
            ..Config::default()
        },
    )
    .unwrap();

    for i in 0..POPULATION {
        let key = format!("bench/{i:08}");
        let doc = format!("{{\"n\":{i}}}").into_bytes();
        let _ = db.put(key.as_bytes(), doc, 0, u64::MAX, Some(1));
    }

    let mut i = 0u64;
    c.bench_function("tensordb/point_read", |b| {
        b.iter(|| {
            let key = format!("bench/{:08}", i % POPULATION);
            let _ = db.get(key.as_bytes(), None, None);
            i += 1;
        })
    });
}

fn tensordb_batch_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = Database::open(
        dir.path(),
        Config {
            shard_count: 4,
            ..Config::default()
        },
    )
    .unwrap();

    let mut base = 0u64;
    c.bench_function("tensordb/batch_write_100", |b| {
        b.iter(|| {
            let entries: Vec<tensordb::WriteBatchItem> = (0..BATCH_SIZE)
                .map(|j| {
                    let key = format!("batch/{:08}", base + j);
                    tensordb::WriteBatchItem {
                        user_key: key.into_bytes(),
                        doc: format!("{{\"n\":{}}}", base + j).into_bytes(),
                        valid_from: 0,
                        valid_to: u64::MAX,
                        schema_version: Some(1),
                    }
                })
                .collect();
            let _ = db.write_batch(entries);
            base += BATCH_SIZE;
        })
    });
}

fn tensordb_prefix_scan(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = Database::open(
        dir.path(),
        Config {
            shard_count: 4,
            ..Config::default()
        },
    )
    .unwrap();

    for i in 0..1_000u64 {
        let key = format!("scan/{i:06}");
        let doc = format!("{{\"n\":{i}}}").into_bytes();
        let _ = db.put(key.as_bytes(), doc, 0, u64::MAX, Some(1));
    }

    c.bench_function("tensordb/prefix_scan_1000", |b| {
        b.iter(|| {
            let _ = db.scan_prefix(b"scan/", None, None, None);
        })
    });
}

fn tensordb_mixed_workload(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = Database::open(
        dir.path(),
        Config {
            shard_count: 4,
            ..Config::default()
        },
    )
    .unwrap();

    for i in 0..5_000u64 {
        let key = format!("mixed/{i:08}");
        let doc = format!("{{\"n\":{i}}}").into_bytes();
        let _ = db.put(key.as_bytes(), doc, 0, u64::MAX, Some(1));
    }

    let mut i = 5_000u64;
    c.bench_function("tensordb/mixed_80r_20w", |b| {
        b.iter(|| {
            if i.is_multiple_of(5) {
                let key = format!("mixed/{i:08}");
                let doc = format!("{{\"n\":{i}}}").into_bytes();
                let _ = db.put(key.as_bytes(), doc, 0, u64::MAX, Some(1));
            } else {
                let key = format!("mixed/{:08}", i % 5_000);
                let _ = db.get(key.as_bytes(), None, None);
            }
            i += 1;
        })
    });
}

// ---------------------------------------------------------------------------
// SQLite benchmarks
// ---------------------------------------------------------------------------

fn sqlite_setup(dir: &TempDir) -> rusqlite::Connection {
    let path = dir.path().join("sqlite.db");
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         CREATE TABLE bench (key TEXT PRIMARY KEY, doc TEXT);",
    )
    .unwrap();
    conn
}

fn sqlite_point_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let conn = sqlite_setup(&dir);

    let mut i = 0u64;
    c.bench_function("sqlite/point_write", |b| {
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

fn sqlite_point_read(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let conn = sqlite_setup(&dir);

    for i in 0..POPULATION {
        let key = format!("bench/{i:08}");
        let doc = format!("{{\"n\":{i}}}");
        conn.execute("INSERT INTO bench (key, doc) VALUES (?1, ?2)", [&key, &doc])
            .unwrap();
    }

    let mut stmt = conn
        .prepare("SELECT doc FROM bench WHERE key = ?1")
        .unwrap();
    let mut i = 0u64;
    c.bench_function("sqlite/point_read", |b| {
        b.iter(|| {
            let key = format!("bench/{:08}", i % POPULATION);
            let _: Option<String> = stmt.query_row([&key], |row| row.get(0)).ok();
            i += 1;
        })
    });
}

fn sqlite_batch_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let conn = sqlite_setup(&dir);

    let mut base = 0u64;
    c.bench_function("sqlite/batch_write_100", |b| {
        b.iter(|| {
            conn.execute_batch("BEGIN;").unwrap();
            for j in 0..BATCH_SIZE {
                let key = format!("batch/{:08}", base + j);
                let doc = format!("{{\"n\":{}}}", base + j);
                conn.execute(
                    "INSERT OR REPLACE INTO bench (key, doc) VALUES (?1, ?2)",
                    [&key, &doc],
                )
                .unwrap();
            }
            conn.execute_batch("COMMIT;").unwrap();
            base += BATCH_SIZE;
        })
    });
}

fn sqlite_prefix_scan(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let conn = sqlite_setup(&dir);

    for i in 0..1_000u64 {
        let key = format!("scan/{i:06}");
        let doc = format!("{{\"n\":{i}}}");
        conn.execute("INSERT INTO bench (key, doc) VALUES (?1, ?2)", [&key, &doc])
            .unwrap();
    }

    c.bench_function("sqlite/prefix_scan_1000", |b| {
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

fn sqlite_mixed_workload(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let conn = sqlite_setup(&dir);

    for i in 0..5_000u64 {
        let key = format!("mixed/{i:08}");
        let doc = format!("{{\"n\":{i}}}");
        conn.execute("INSERT INTO bench (key, doc) VALUES (?1, ?2)", [&key, &doc])
            .unwrap();
    }

    let mut i = 5_000u64;
    c.bench_function("sqlite/mixed_80r_20w", |b| {
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

// ---------------------------------------------------------------------------
// sled benchmarks
// ---------------------------------------------------------------------------

fn sled_point_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = sled::open(dir.path()).unwrap();

    let mut i = 0u64;
    c.bench_function("sled/point_write", |b| {
        b.iter(|| {
            let key = format!("bench/{i:08}");
            let doc = format!("{{\"n\":{i}}}");
            db.insert(key.as_bytes(), doc.as_bytes()).unwrap();
            i += 1;
        })
    });
}

fn sled_point_read(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = sled::open(dir.path()).unwrap();

    for i in 0..POPULATION {
        let key = format!("bench/{i:08}");
        let doc = format!("{{\"n\":{i}}}");
        db.insert(key.as_bytes(), doc.as_bytes()).unwrap();
    }

    let mut i = 0u64;
    c.bench_function("sled/point_read", |b| {
        b.iter(|| {
            let key = format!("bench/{:08}", i % POPULATION);
            let _ = db.get(key.as_bytes());
            i += 1;
        })
    });
}

fn sled_batch_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = sled::open(dir.path()).unwrap();

    let mut base = 0u64;
    c.bench_function("sled/batch_write_100", |b| {
        b.iter(|| {
            let mut batch = sled::Batch::default();
            for j in 0..BATCH_SIZE {
                let key = format!("batch/{:08}", base + j);
                let doc = format!("{{\"n\":{}}}", base + j);
                batch.insert(key.as_bytes(), doc.as_bytes());
            }
            db.apply_batch(batch).unwrap();
            base += BATCH_SIZE;
        })
    });
}

fn sled_prefix_scan(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = sled::open(dir.path()).unwrap();

    for i in 0..1_000u64 {
        let key = format!("scan/{i:06}");
        let doc = format!("{{\"n\":{i}}}");
        db.insert(key.as_bytes(), doc.as_bytes()).unwrap();
    }

    c.bench_function("sled/prefix_scan_1000", |b| {
        b.iter(|| {
            let rows: Vec<_> = db.scan_prefix(b"scan/").filter_map(|r| r.ok()).collect();
            std::hint::black_box(rows);
        })
    });
}

fn sled_mixed_workload(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = sled::open(dir.path()).unwrap();

    for i in 0..5_000u64 {
        let key = format!("mixed/{i:08}");
        let doc = format!("{{\"n\":{i}}}");
        db.insert(key.as_bytes(), doc.as_bytes()).unwrap();
    }

    let mut i = 5_000u64;
    c.bench_function("sled/mixed_80r_20w", |b| {
        b.iter(|| {
            if i.is_multiple_of(5) {
                let key = format!("mixed/{i:08}");
                let doc = format!("{{\"n\":{i}}}");
                db.insert(key.as_bytes(), doc.as_bytes()).unwrap();
            } else {
                let key = format!("mixed/{:08}", i % 5_000);
                let _ = db.get(key.as_bytes());
            }
            i += 1;
        })
    });
}

// ---------------------------------------------------------------------------
// redb benchmarks
// ---------------------------------------------------------------------------

const REDB_TABLE: redb::TableDefinition<&[u8], &[u8]> = redb::TableDefinition::new("bench");

fn redb_point_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = redb::Database::create(dir.path().join("redb.db")).unwrap();

    let mut i = 0u64;
    c.bench_function("redb/point_write", |b| {
        b.iter(|| {
            let key = format!("bench/{i:08}");
            let doc = format!("{{\"n\":{i}}}");
            let txn = db.begin_write().unwrap();
            {
                let mut table = txn.open_table(REDB_TABLE).unwrap();
                table.insert(key.as_bytes(), doc.as_bytes()).unwrap();
            }
            txn.commit().unwrap();
            i += 1;
        })
    });
}

fn redb_point_read(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = redb::Database::create(dir.path().join("redb.db")).unwrap();

    // Pre-populate
    {
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(REDB_TABLE).unwrap();
            for i in 0..POPULATION {
                let key = format!("bench/{i:08}");
                let doc = format!("{{\"n\":{i}}}");
                table.insert(key.as_bytes(), doc.as_bytes()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    let mut i = 0u64;
    c.bench_function("redb/point_read", |b| {
        b.iter(|| {
            let key = format!("bench/{:08}", i % POPULATION);
            let txn = db.begin_read().unwrap();
            let table = txn.open_table(REDB_TABLE).unwrap();
            let _ = table.get(key.as_bytes());
            i += 1;
        })
    });
}

fn redb_batch_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = redb::Database::create(dir.path().join("redb.db")).unwrap();

    let mut base = 0u64;
    c.bench_function("redb/batch_write_100", |b| {
        b.iter(|| {
            let txn = db.begin_write().unwrap();
            {
                let mut table = txn.open_table(REDB_TABLE).unwrap();
                for j in 0..BATCH_SIZE {
                    let key = format!("batch/{:08}", base + j);
                    let doc = format!("{{\"n\":{}}}", base + j);
                    table.insert(key.as_bytes(), doc.as_bytes()).unwrap();
                }
            }
            txn.commit().unwrap();
            base += BATCH_SIZE;
        })
    });
}

fn redb_prefix_scan(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = redb::Database::create(dir.path().join("redb.db")).unwrap();

    {
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(REDB_TABLE).unwrap();
            for i in 0..1_000u64 {
                let key = format!("scan/{i:06}");
                let doc = format!("{{\"n\":{i}}}");
                table.insert(key.as_bytes(), doc.as_bytes()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    c.bench_function("redb/prefix_scan_1000", |b| {
        b.iter(|| {
            let txn = db.begin_read().unwrap();
            let table = txn.open_table(REDB_TABLE).unwrap();
            let rows: Vec<_> = table
                .range("scan/".as_bytes().."scan0".as_bytes())
                .unwrap()
                .filter_map(|r| r.ok())
                .collect();
            std::hint::black_box(rows);
        })
    });
}

fn redb_mixed_workload(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let db = redb::Database::create(dir.path().join("redb.db")).unwrap();

    {
        let txn = db.begin_write().unwrap();
        {
            let mut table = txn.open_table(REDB_TABLE).unwrap();
            for i in 0..5_000u64 {
                let key = format!("mixed/{i:08}");
                let doc = format!("{{\"n\":{i}}}");
                table.insert(key.as_bytes(), doc.as_bytes()).unwrap();
            }
        }
        txn.commit().unwrap();
    }

    let mut i = 5_000u64;
    c.bench_function("redb/mixed_80r_20w", |b| {
        b.iter(|| {
            if i.is_multiple_of(5) {
                let key = format!("mixed/{i:08}");
                let doc = format!("{{\"n\":{i}}}");
                let txn = db.begin_write().unwrap();
                {
                    let mut table = txn.open_table(REDB_TABLE).unwrap();
                    table.insert(key.as_bytes(), doc.as_bytes()).unwrap();
                }
                txn.commit().unwrap();
            } else {
                let key = format!("mixed/{:08}", i % 5_000);
                let txn = db.begin_read().unwrap();
                let table = txn.open_table(REDB_TABLE).unwrap();
                let _ = table.get(key.as_bytes());
            }
            i += 1;
        })
    });
}

// ---------------------------------------------------------------------------
// Throughput benchmark group (parameterized by dataset size)
// ---------------------------------------------------------------------------

fn throughput_reads(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_point_read");

    for size in [1_000u64, 10_000, 50_000] {
        // TensorDB
        let dir = TempDir::new().unwrap();
        let db = Database::open(
            dir.path(),
            Config {
                shard_count: 4,
                ..Config::default()
            },
        )
        .unwrap();
        for i in 0..size {
            let key = format!("bench/{i:08}");
            let doc = format!("{{\"n\":{i}}}").into_bytes();
            let _ = db.put(key.as_bytes(), doc, 0, u64::MAX, Some(1));
        }

        group.throughput(Throughput::Elements(1));
        let mut i = 0u64;
        group.bench_with_input(BenchmarkId::new("tensordb", size), &size, |b, &_size| {
            b.iter(|| {
                let key = format!("bench/{:08}", i % size);
                let _ = db.get(key.as_bytes(), None, None);
                i += 1;
            })
        });

        // sled
        let sled_dir = TempDir::new().unwrap();
        let sled_db = sled::open(sled_dir.path()).unwrap();
        for i in 0..size {
            let key = format!("bench/{i:08}");
            let doc = format!("{{\"n\":{i}}}");
            sled_db.insert(key.as_bytes(), doc.as_bytes()).unwrap();
        }

        let mut si = 0u64;
        group.bench_with_input(BenchmarkId::new("sled", size), &size, |b, &_size| {
            b.iter(|| {
                let key = format!("bench/{:08}", si % size);
                let _ = sled_db.get(key.as_bytes());
                si += 1;
            })
        });
    }

    group.finish();
}

criterion_group!(
    single_op,
    tensordb_point_write,
    sqlite_point_write,
    sled_point_write,
    redb_point_write,
    tensordb_point_read,
    sqlite_point_read,
    sled_point_read,
    redb_point_read,
);

criterion_group!(
    batch_and_scan,
    tensordb_batch_write,
    sqlite_batch_write,
    sled_batch_write,
    redb_batch_write,
    tensordb_prefix_scan,
    sqlite_prefix_scan,
    sled_prefix_scan,
    redb_prefix_scan,
);

criterion_group!(
    mixed,
    tensordb_mixed_workload,
    sqlite_mixed_workload,
    sled_mixed_workload,
    redb_mixed_workload,
);

criterion_group!(throughput, throughput_reads,);

criterion_main!(single_op, batch_and_scan, mixed, throughput);
