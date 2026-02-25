use criterion::{criterion_group, criterion_main, Criterion};
use tempfile::tempdir;

use spectradb::{BenchOptions, Config, Database};

fn bench_basic(c: &mut Criterion) {
    let dir = tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();

    c.bench_function("spectradb_cli_like_bench", |b| {
        b.iter(|| {
            let _ = db
                .bench(BenchOptions {
                    write_ops: 2_000,
                    read_ops: 1_000,
                    keyspace: 500,
                    read_miss_ratio: 0.20,
                })
                .unwrap();
        })
    });
}

criterion_group!(benches, bench_basic);
criterion_main!(benches);
