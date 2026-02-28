//! TensorDB Quickstart Example
//!
//! Demonstrates the core features of TensorDB: creating tables, inserting data,
//! querying with SQL, time-travel reads, and AI insights.
//!
//! Run with: `cargo run --example quickstart`

use tensordb::{Config, Database};

fn main() -> tensordb::Result<()> {
    // Create a temporary database directory
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let db = Database::open(
        dir.path(),
        Config {
            shard_count: 4,
            ..Config::default()
        },
    )?;

    println!("=== TensorDB Quickstart ===\n");

    // --- 1. Create Tables ---
    println!("1. Creating tables...");
    db.sql("CREATE TABLE events (pk TEXT PRIMARY KEY);")?;
    db.sql("CREATE TABLE accounts (id INTEGER PRIMARY KEY, name TEXT, balance REAL);")?;
    println!("   Created: events (legacy JSON), accounts (typed columns)\n");

    // --- 2. Insert Data ---
    println!("2. Inserting data...");
    db.sql("INSERT INTO events (pk, doc) VALUES ('evt-1', '{\"type\":\"signup\",\"user\":\"alice\"}');")?;
    db.sql("INSERT INTO events (pk, doc) VALUES ('evt-2', '{\"type\":\"purchase\",\"user\":\"bob\",\"amount\":49.99}');")?;
    db.sql("INSERT INTO events (pk, doc) VALUES ('evt-3', '{\"type\":\"purchase\",\"user\":\"alice\",\"amount\":120.00}');")?;

    db.sql("INSERT INTO accounts (id, name, balance) VALUES (1, 'alice', 1000.0);")?;
    db.sql("INSERT INTO accounts (id, name, balance) VALUES (2, 'bob', 500.0);")?;
    println!("   Inserted 3 events and 2 accounts\n");

    // --- 3. Query with SQL ---
    println!("3. Querying with SQL...");

    // Simple SELECT
    let result = db.sql("SELECT doc FROM events ORDER BY pk ASC;")?;
    println!("   All events:");
    print_rows(&result);

    // Aggregate
    let result = db.sql("SELECT count(*) FROM events;")?;
    println!("   Event count:");
    print_rows(&result);

    // Typed table query
    let result = db.sql("SELECT name, balance FROM accounts ORDER BY id;")?;
    println!("   Accounts:");
    print_rows(&result);

    // WHERE filter
    let result = db.sql("SELECT name FROM accounts WHERE balance > 600;")?;
    println!("   Accounts with balance > 600:");
    print_rows(&result);

    // --- 4. Time Travel ---
    println!("4. Time travel...");

    // Get the current commit timestamp by inserting and checking
    let ct1 = db.put(b"__demo/marker", b"v1".to_vec(), 0, u64::MAX, None)?;
    let ct2 = db.put(b"__demo/marker", b"v2".to_vec(), 0, u64::MAX, None)?;

    let v1 = db.get(b"__demo/marker", Some(ct1), None)?;
    let v2 = db.get(b"__demo/marker", Some(ct2), None)?;
    println!(
        "   Value at commit {ct1}: {:?}",
        v1.map(|v| String::from_utf8_lossy(&v).to_string())
    );
    println!(
        "   Value at commit {ct2}: {:?}\n",
        v2.map(|v| String::from_utf8_lossy(&v).to_string())
    );

    // --- 5. Prepared Statements ---
    println!("5. Prepared statements...");
    let prepared = db.prepare("SELECT name FROM accounts WHERE balance > $1;")?;
    println!("   Prepared: SELECT name FROM accounts WHERE balance > $1");
    println!("   Param count: {}", prepared.param_count());

    let result = prepared.execute_with_params(&db, &["600"])?;
    println!("   With $1 = 600:");
    print_rows(&result);

    let result = prepared.execute_with_params(&db, &["100"])?;
    println!("   With $1 = 100:");
    print_rows(&result);

    // --- 6. EXPLAIN & ANALYZE ---
    println!("6. Query planning...");
    let result = db.sql("EXPLAIN SELECT name FROM accounts WHERE balance > 500;")?;
    println!("   EXPLAIN output:");
    print_explain(&result);

    db.sql("ANALYZE accounts;")?;
    println!("   Table statistics collected for 'accounts'\n");

    // --- 7. Write Batch ---
    println!("7. Write batch...");
    let items: Vec<tensordb::WriteBatchItem> = (100..110)
        .map(|i| tensordb::WriteBatchItem {
            user_key: format!("batch/key-{i}").into_bytes(),
            doc: format!("{{\"batch_id\":{i}}}").into_bytes(),
            valid_from: 0,
            valid_to: u64::MAX,
            schema_version: Some(1),
        })
        .collect();
    let results = db.write_batch(items)?;
    println!("   Wrote 10 keys in a single batch");
    println!("   Commit timestamps: {:?}\n", &results[..3]);

    // --- 8. Prefix Scan ---
    println!("8. Prefix scan...");
    let scanned = db.scan_prefix(b"batch/", None, None, None)?;
    println!("   Found {} keys with prefix 'batch/'", scanned.len());
    for row in scanned.iter().take(3) {
        println!(
            "   {} -> {}",
            String::from_utf8_lossy(&row.user_key),
            String::from_utf8_lossy(&row.doc)
        );
    }
    println!("   ...\n");

    // --- 9. Database Stats ---
    println!("9. Database stats...");
    let stats = db.stats()?;
    println!("   Shards: {}", stats.shard_count);
    println!("   Total puts: {}", stats.puts);
    println!("   Total gets: {}", stats.gets);
    println!("   Bloom filter negatives: {}", stats.bloom_negatives);

    println!("\n=== Done! TensorDB is ready. ===");
    Ok(())
}

fn print_rows(result: &tensordb::sql::exec::SqlResult) {
    match result {
        tensordb::sql::exec::SqlResult::Rows(rows) => {
            for row in rows {
                println!("     {}", String::from_utf8_lossy(row));
            }
        }
        tensordb::sql::exec::SqlResult::Affected { message, .. } => {
            println!("     {message}");
        }
        tensordb::sql::exec::SqlResult::Explain(text) => {
            println!("     {text}");
        }
    }
    println!();
}

fn print_explain(result: &tensordb::sql::exec::SqlResult) {
    if let tensordb::sql::exec::SqlResult::Explain(text) = result {
        for line in text.lines() {
            println!("     {line}");
        }
    }
    println!();
}
