//! AI-Native Database Example
//!
//! Demonstrates TensorDB's AI runtime features: automatic insight synthesis,
//! change feed monitoring, and query advisors.
//!
//! Run with: `cargo run --example ai_native`

use tensordb::{Config, Database};

fn main() -> tensordb::Result<()> {
    let dir = tempfile::tempdir().expect("failed to create temp dir");

    // Enable AI features
    let config = Config {
        shard_count: 2,
        ai_auto_insights: true,
        ai_batch_window_ms: 10,
        ai_batch_max_events: 4,
        ..Config::default()
    };
    let db = Database::open(dir.path(), config)?;

    println!("=== AI-Native Database Example ===\n");

    // --- 1. AI auto-insights on writes ---
    println!("1. Writing events with AI monitoring...\n");

    db.sql("CREATE TABLE audit_log (pk TEXT PRIMARY KEY);")?;

    // Normal events
    db.sql("INSERT INTO audit_log (pk, doc) VALUES ('evt-1', '{\"action\":\"login\",\"user\":\"alice\",\"status\":\"success\"}');")?;
    db.sql("INSERT INTO audit_log (pk, doc) VALUES ('evt-2', '{\"action\":\"upload\",\"user\":\"bob\",\"file\":\"report.pdf\"}');")?;

    // Suspicious events (AI runtime detects risk keywords)
    db.sql("INSERT INTO audit_log (pk, doc) VALUES ('evt-3', '{\"action\":\"login\",\"user\":\"admin\",\"status\":\"failed\",\"error\":\"unauthorized access attempt\"}');")?;
    db.sql("INSERT INTO audit_log (pk, doc) VALUES ('evt-4', '{\"action\":\"delete\",\"user\":\"unknown\",\"critical\":true,\"error\":\"timeout during critical operation\"}');")?;

    println!("   Wrote 4 audit events (2 normal, 2 suspicious)\n");

    // Give AI runtime a moment to process
    std::thread::sleep(std::time::Duration::from_millis(50));

    // --- 2. Check AI insights ---
    println!("2. Checking AI insights...\n");

    let insights = db.scan_prefix(b"__ai/", None, None, None)?;
    if insights.is_empty() {
        println!("   No AI insights generated yet (AI synthesis may be async)");
    } else {
        println!("   {} AI insights generated:", insights.len());
        for insight in &insights {
            println!(
                "     {} -> {}",
                String::from_utf8_lossy(&insight.user_key),
                String::from_utf8_lossy(&insight.doc)
            );
        }
    }
    println!();

    // --- 3. Query Advisor ---
    println!("3. Query advisor (access pattern analysis)...\n");

    // Do several reads to build up access patterns
    for _ in 0..10 {
        let _ = db.get(b"table/audit_log/evt-3", None, None);
        let _ = db.get(b"table/audit_log/evt-4", None, None);
    }
    let _ = db.get(b"table/audit_log/evt-1", None, None);

    let stats = db.stats()?;
    println!("   Total reads: {}", stats.gets);
    println!("   Total writes: {}", stats.puts);
    println!("   Bloom filter negatives: {}", stats.bloom_negatives);
    println!();

    // --- 4. EXPLAIN ANALYZE with cost model ---
    println!("4. Cost-based query planning...\n");

    let result = db.sql("EXPLAIN SELECT doc FROM audit_log;")?;
    println!("   Query plan for full scan:");
    if let tensordb::sql::exec::SqlResult::Explain(text) = &result {
        for line in text.lines() {
            println!("     {line}");
        }
    }
    println!();

    let result = db.sql("EXPLAIN SELECT doc FROM audit_log WHERE pk='evt-3';")?;
    println!("   Query plan for point lookup:");
    if let tensordb::sql::exec::SqlResult::Explain(text) = &result {
        for line in text.lines() {
            println!("     {line}");
        }
    }
    println!();

    // --- 5. EXPLAIN ANALYZE with runtime metrics ---
    let result = db.sql("EXPLAIN ANALYZE SELECT doc FROM audit_log;")?;
    println!("   EXPLAIN ANALYZE output:");
    if let tensordb::sql::exec::SqlResult::Explain(text) = &result {
        for line in text.lines() {
            println!("     {line}");
        }
    }
    println!();

    // --- 6. Collect table statistics ---
    println!("5. Table statistics...\n");
    let result = db.sql("ANALYZE audit_log;")?;
    if let tensordb::sql::exec::SqlResult::Affected { message, .. } = &result {
        println!("   {message}");
    }

    println!("\n=== Key Takeaway ===");
    println!("TensorDB is AI-native: the AI runtime participates in");
    println!("query planning, risk assessment, and insight synthesis.");
    println!("Every operation feeds the AI — no external integration needed.\n");

    Ok(())
}
