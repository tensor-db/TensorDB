//! Bitemporal Ledger Example
//!
//! Shows how TensorDB tracks both system time (when data was recorded) and
//! business time (when data was valid), enabling regulatory compliance,
//! audit trails, and "what did we know and when" queries.
//!
//! Run with: `cargo run --example bitemporal`

use tensordb::{Config, Database};

fn main() -> tensordb::Result<()> {
    let dir = tempfile::tempdir().expect("failed to create temp dir");
    let db = Database::open(dir.path(), Config::default())?;

    println!("=== Bitemporal Ledger Example ===\n");

    // Scenario: Employee salary tracking with corrections
    //
    // Facts are immutable. Every write records:
    //   - commit_ts: system time (auto-assigned, monotonic)
    //   - valid_from/valid_to: business time interval
    //
    // This means we can ask:
    //   "What was Alice's salary effective Jan 1?" (business time query)
    //   "What did we believe on commit #5?" (system time query)
    //   "What did we believe about Jan 1 salaries as of commit #5?" (bi-temporal)

    println!("1. Recording salary facts with business time...\n");

    // Alice's salary: $80k effective from day 0 to day 365
    let ct1 = db.put(
        b"employee/alice/salary",
        br#"{"amount":80000,"currency":"USD"}"#.to_vec(),
        0,   // valid_from: day 0
        365, // valid_to: day 365
        None,
    )?;
    println!("   Commit {ct1}: Alice salary $80k (valid days 0-365)");

    // Bob's salary: $90k effective from day 0 to day 365
    let ct2 = db.put(
        b"employee/bob/salary",
        br#"{"amount":90000,"currency":"USD"}"#.to_vec(),
        0,
        365,
        None,
    )?;
    println!("   Commit {ct2}: Bob salary $90k (valid days 0-365)");

    // Correction: Alice got a raise on day 180 — her salary is now $95k
    // We don't UPDATE — we record a new fact with a different validity interval
    let ct3 = db.put(
        b"employee/alice/salary",
        br#"{"amount":95000,"currency":"USD"}"#.to_vec(),
        180, // valid_from: day 180 (when raise took effect)
        365, // valid_to: day 365
        None,
    )?;
    println!("   Commit {ct3}: Alice salary corrected to $95k (valid days 180-365)");

    println!("\n2. Querying by business time (VALID AT)...\n");

    // What was Alice's salary valid at day 100? (before the raise)
    let val_100 = db.get(b"employee/alice/salary", None, Some(100))?;
    println!(
        "   Alice's salary at day 100: {}",
        val_100
            .map(|v| String::from_utf8_lossy(&v).to_string())
            .unwrap_or_else(|| "N/A".to_string())
    );

    // What was Alice's salary valid at day 200? (after the raise)
    let val_200 = db.get(b"employee/alice/salary", None, Some(200))?;
    println!(
        "   Alice's salary at day 200: {}",
        val_200
            .map(|v| String::from_utf8_lossy(&v).to_string())
            .unwrap_or_else(|| "N/A".to_string())
    );

    println!("\n3. Querying by system time (AS OF commit)...\n");

    // What did we believe about Alice's salary before the correction?
    // AS OF commit ct2 (before ct3 was recorded)
    let before_correction = db.get(b"employee/alice/salary", Some(ct2), None)?;
    println!(
        "   Alice's salary as of commit {ct2} (before correction): {}",
        before_correction
            .map(|v| String::from_utf8_lossy(&v).to_string())
            .unwrap_or_else(|| "N/A".to_string())
    );

    // What do we currently believe?
    let current = db.get(b"employee/alice/salary", None, None)?;
    println!(
        "   Alice's salary now (latest commit): {}",
        current
            .map(|v| String::from_utf8_lossy(&v).to_string())
            .unwrap_or_else(|| "N/A".to_string())
    );

    println!("\n4. Bi-temporal query (AS OF + VALID AT)...\n");

    // What did we believe about Alice's salary at day 200, as of commit ct2?
    // (Before the correction was recorded, what did the system say was valid at day 200?)
    let bitemp = db.get(b"employee/alice/salary", Some(ct2), Some(200))?;
    println!(
        "   Alice salary at day 200, as-of commit {ct2}: {}",
        bitemp
            .map(|v| String::from_utf8_lossy(&v).to_string())
            .unwrap_or_else(|| "N/A".to_string())
    );

    // Same question, but as-of commit ct3 (after correction)
    let bitemp2 = db.get(b"employee/alice/salary", Some(ct3), Some(200))?;
    println!(
        "   Alice salary at day 200, as-of commit {ct3}: {}",
        bitemp2
            .map(|v| String::from_utf8_lossy(&v).to_string())
            .unwrap_or_else(|| "N/A".to_string())
    );

    println!("\n5. Audit trail — all versions preserved...\n");

    // Scan all salary facts for Alice
    let all_facts = db.scan_prefix(b"employee/alice/salary", None, None, None)?;
    println!("   All salary facts for Alice:");
    for fact in &all_facts {
        println!(
            "     commit_ts={} -> {}",
            fact.commit_ts,
            String::from_utf8_lossy(&fact.doc),
        );
    }

    println!("\n=== Key Takeaway ===");
    println!("Every fact is immutable. Corrections create new facts with");
    println!("different validity intervals. You can always answer:");
    println!("  - What is true now?");
    println!("  - What was true at time T?");
    println!("  - What did we believe at system time S?");
    println!("  - What did we believe about time T at system time S?\n");

    Ok(())
}
