use criterion::{criterion_group, criterion_main, Criterion};
use spectradb::{Config, Database};
use tempfile::TempDir;

fn bench_spectradb_point_write(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 4,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    let mut i = 0u64;
    c.bench_function("spectradb_point_write", |b| {
        b.iter(|| {
            let key = format!("bench/{i:08}");
            let doc = format!("{{\"n\":{i}}}");
            let _ = db.put(key.as_bytes(), doc.into_bytes(), 0, u64::MAX, Some(1));
            i += 1;
        })
    });
}

fn bench_spectradb_point_read(c: &mut Criterion) {
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
    c.bench_function("spectradb_point_read", |b| {
        b.iter(|| {
            let key = format!("bench/{:08}", i % 10_000);
            let _ = db.get(key.as_bytes(), None, None);
            i += 1;
        })
    });
}

fn bench_spectradb_scan(c: &mut Criterion) {
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

    c.bench_function("spectradb_prefix_scan_1000", |b| {
        b.iter(|| {
            let _ = db.scan_prefix(b"scan/", None, None, None);
        })
    });
}

fn bench_spectradb_sql_select(c: &mut Criterion) {
    let dir = TempDir::new().unwrap();
    let config = Config {
        shard_count: 2,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();

    db.sql("CREATE TABLE bench (pk TEXT PRIMARY KEY)").unwrap();
    for i in 0..100 {
        let q = format!(
            "INSERT INTO bench (pk, doc) VALUES ('k{i}', '{{\"val\":{i}}}')"
        );
        db.sql(&q).unwrap();
    }

    c.bench_function("spectradb_sql_select_100", |b| {
        b.iter(|| {
            let _ = db.sql("SELECT pk, doc FROM bench");
        })
    });
}

criterion_group!(
    benches,
    bench_spectradb_point_write,
    bench_spectradb_point_read,
    bench_spectradb_scan,
    bench_spectradb_sql_select,
);
criterion_main!(benches);
