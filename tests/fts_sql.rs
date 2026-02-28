//! Integration tests for v0.13 Full-Text Search SQL Integration
//!
//! Tests CREATE FULLTEXT INDEX, MATCH(), HIGHLIGHT(), and FTS index maintenance.

use tempfile::tempdir;

use tensordb::config::Config;
use tensordb::sql::exec::SqlResult;
use tensordb::Database;

fn setup_db() -> (tempfile::TempDir, Database) {
    let dir = tempdir().unwrap();
    let db = Database::open(
        dir.path(),
        Config {
            shard_count: 1,
            ..Config::default()
        },
    )
    .unwrap();
    (dir, db)
}

fn row_strings(result: &SqlResult) -> Vec<String> {
    match result {
        SqlResult::Rows(rows) => rows
            .iter()
            .map(|r| String::from_utf8_lossy(r).to_string())
            .collect(),
        _ => panic!("expected Rows, got {result:?}"),
    }
}

// ---------- CREATE FULLTEXT INDEX ----------

#[test]
fn create_fulltext_index_on_doc_column() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE articles (pk TEXT PRIMARY KEY);")
        .unwrap();

    let result = db
        .sql("CREATE FULLTEXT INDEX idx_articles_fts ON articles (doc);")
        .unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("created fulltext index"));
        assert!(message.contains("idx_articles_fts"));
    } else {
        panic!("expected Affected result");
    }
}

#[test]
fn create_fulltext_index_on_typed_column() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT, body TEXT);")
        .unwrap();

    let result = db
        .sql("CREATE FULLTEXT INDEX idx_posts_fts ON posts (title, body);")
        .unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("created fulltext index"));
        assert!(message.contains("title, body"));
    } else {
        panic!("expected Affected result");
    }
}

#[test]
fn create_fulltext_index_duplicate_fails() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("CREATE FULLTEXT INDEX idx ON t (doc);").unwrap();

    let result = db.sql("CREATE FULLTEXT INDEX idx ON t (doc);");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}

#[test]
fn create_fulltext_index_invalid_column() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    let result = db.sql("CREATE FULLTEXT INDEX idx ON t (nonexistent);");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("does not exist"));
}

// ---------- DROP FULLTEXT INDEX ----------

#[test]
fn drop_fulltext_index() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("CREATE FULLTEXT INDEX idx ON t (doc);").unwrap();

    let result = db.sql("DROP FULLTEXT INDEX idx ON t;").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("dropped fulltext index"));
    } else {
        panic!("expected Affected result");
    }
}

#[test]
fn drop_fulltext_index_nonexistent() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();

    let result = db.sql("DROP FULLTEXT INDEX nonexistent ON t;");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("does not exist"));
}

// ---------- FTS Index Maintenance ----------

#[test]
fn fts_index_backfills_existing_data() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE articles (pk TEXT PRIMARY KEY);")
        .unwrap();

    // Insert data before creating the FTS index
    db.sql("INSERT INTO articles (pk, doc) VALUES ('a1', '{\"title\":\"Rust programming language\"}');")
        .unwrap();
    db.sql(
        "INSERT INTO articles (pk, doc) VALUES ('a2', '{\"title\":\"Python for data science\"}');",
    )
    .unwrap();

    // Create FTS index — should backfill
    let result = db
        .sql("CREATE FULLTEXT INDEX idx ON articles (doc);")
        .unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("indexed 2 rows"));
    } else {
        panic!("expected Affected result");
    }
}

#[test]
fn fts_index_maintained_on_insert() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE articles (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("CREATE FULLTEXT INDEX idx ON articles (doc);")
        .unwrap();

    // Insert after creating FTS index
    db.sql("INSERT INTO articles (pk, doc) VALUES ('a1', '{\"title\":\"Rust programming language\"}');")
        .unwrap();
    db.sql(
        "INSERT INTO articles (pk, doc) VALUES ('a2', '{\"title\":\"Python for data science\"}');",
    )
    .unwrap();

    // MATCH should find the inserted data
    let result = db
        .sql("SELECT doc FROM articles WHERE MATCH(doc, 'rust');")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("Rust"));
}

// ---------- MATCH() queries ----------

#[test]
fn match_single_term() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE articles (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("CREATE FULLTEXT INDEX idx ON articles (doc);")
        .unwrap();

    db.sql("INSERT INTO articles (pk, doc) VALUES ('a1', '{\"title\":\"Rust programming\"}');")
        .unwrap();
    db.sql("INSERT INTO articles (pk, doc) VALUES ('a2', '{\"title\":\"Python programming\"}');")
        .unwrap();
    db.sql("INSERT INTO articles (pk, doc) VALUES ('a3', '{\"title\":\"Go language\"}');")
        .unwrap();

    let result = db
        .sql("SELECT doc FROM articles WHERE MATCH(doc, 'programming');")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 2);
}

#[test]
fn match_multiple_terms_intersection() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE articles (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("CREATE FULLTEXT INDEX idx ON articles (doc);")
        .unwrap();

    db.sql(
        "INSERT INTO articles (pk, doc) VALUES ('a1', '{\"text\":\"rust programming language\"}');",
    )
    .unwrap();
    db.sql("INSERT INTO articles (pk, doc) VALUES ('a2', '{\"text\":\"python programming language\"}');")
        .unwrap();
    db.sql("INSERT INTO articles (pk, doc) VALUES ('a3', '{\"text\":\"rust systems\"}');")
        .unwrap();

    // "rust programming" should match only a1 (both terms)
    let result = db
        .sql("SELECT doc FROM articles WHERE MATCH(doc, 'rust programming');")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("rust programming"));
}

#[test]
fn match_no_results() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE articles (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("CREATE FULLTEXT INDEX idx ON articles (doc);")
        .unwrap();

    db.sql("INSERT INTO articles (pk, doc) VALUES ('a1', '{\"text\":\"hello world\"}');")
        .unwrap();

    let result = db
        .sql("SELECT doc FROM articles WHERE MATCH(doc, 'nonexistent');")
        .unwrap();
    let rows = row_strings(&result);
    assert!(rows.is_empty());
}

#[test]
fn match_with_stemming() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE articles (pk TEXT PRIMARY KEY);")
        .unwrap();
    db.sql("CREATE FULLTEXT INDEX idx ON articles (doc);")
        .unwrap();

    db.sql("INSERT INTO articles (pk, doc) VALUES ('a1', '{\"text\":\"running fast\"}');")
        .unwrap();

    // "running" stems to "runn", "run" also stems to "run"
    // With our basic stemmer, "running" → "runn", so searching "running" should match
    let result = db
        .sql("SELECT doc FROM articles WHERE MATCH(doc, 'running');")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
}

// ---------- HIGHLIGHT() function ----------

#[test]
fn highlight_marks_matching_terms() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY, content TEXT);")
        .unwrap();
    db.sql("INSERT INTO t (id, content) VALUES (1, 'The quick brown fox jumps');")
        .unwrap();

    let result = db
        .sql("SELECT HIGHLIGHT(content, 'quick fox') FROM t;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("<<quick>>"));
    assert!(rows[0].contains("<<fox>>"));
    assert!(rows[0].contains("brown")); // not highlighted
}

// ---------- FTS with typed columns ----------

#[test]
fn fts_on_typed_columns() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE posts (id INTEGER PRIMARY KEY, title TEXT, body TEXT);")
        .unwrap();
    db.sql("CREATE FULLTEXT INDEX idx ON posts (title, body);")
        .unwrap();

    db.sql("INSERT INTO posts (id, title, body) VALUES (1, 'Introduction to Rust', 'Rust is a systems programming language');")
        .unwrap();
    db.sql("INSERT INTO posts (id, title, body) VALUES (2, 'Python Basics', 'Python is great for data science');")
        .unwrap();

    // Search across both title and body columns
    let result = db
        .sql("SELECT title FROM posts WHERE MATCH(title, 'rust');")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("Rust"));
}
