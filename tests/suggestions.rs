// Phase 2: "Did you mean?" suggestion tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::error::levenshtein;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_levenshtein_distance() {
    assert_eq!(levenshtein("kitten", "sitting"), 3);
    assert_eq!(levenshtein("abc", "abc"), 0);
    assert_eq!(levenshtein("", "abc"), 3);
    assert_eq!(levenshtein("abc", ""), 3);
    assert_eq!(levenshtein("users", "usrs"), 1);
}

#[test]
fn test_levenshtein_case_insensitive() {
    // Our implementation is case-insensitive
    assert_eq!(levenshtein("Users", "users"), 0);
    assert_eq!(levenshtein("USERS", "users"), 0);
}

#[test]
fn test_suggest_closest() {
    use tensordb_core::error::suggest_closest;
    let candidates = &["users", "orders", "products"];
    assert_eq!(
        suggest_closest("usrs", candidates, 3),
        Some("users".to_string())
    );
    assert_eq!(
        suggest_closest("ordrs", candidates, 3),
        Some("orders".to_string())
    );
    // Exact match is filtered out (distance = 0), but close candidates may still match
    // With max_distance=0, nothing matches since exact is excluded
    assert_eq!(suggest_closest("users", candidates, 0), None);
    // Too far away
    assert_eq!(suggest_closest("zzzzz", candidates, 2), None);
}

#[test]
fn test_misspelled_table_errors() {
    let (db, _dir) = setup();
    // Create a table
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        .unwrap();
    // Query a misspelled version — should error
    let err = db.sql("SELECT * FROM usrs").unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("does not exist"), "Should error: {msg}");
}
