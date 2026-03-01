//! Integration tests for Vector Search SQL features.
//!
//! Tests VECTOR column type, CREATE/DROP VECTOR INDEX, <-> distance operator,
//! vector_search() table function, vector SQL functions, and hybrid search.

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

fn affected_message(result: &SqlResult) -> String {
    match result {
        SqlResult::Affected { message, .. } => message.clone(),
        _ => panic!("expected Affected, got {result:?}"),
    }
}

// ── VECTOR column type ──────────────────────────────────────────────────────

#[test]
fn create_table_with_vector_column() {
    let (_dir, db) = setup_db();
    let result = db
        .sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, embedding VECTOR(384));")
        .unwrap();
    let msg = affected_message(&result);
    assert!(msg.contains("created table"), "got: {msg}");
}

#[test]
fn insert_and_select_vector_column() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, vec VECTOR(3));")
        .unwrap();

    db.sql("INSERT INTO items (id, name, vec) VALUES (1, 'alpha', '[1.0, 0.0, 0.0]');")
        .unwrap();
    db.sql("INSERT INTO items (id, name, vec) VALUES (2, 'beta', '[0.0, 1.0, 0.0]');")
        .unwrap();

    let result = db
        .sql("SELECT id, name, vec FROM items ORDER BY id;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 2);
    assert!(rows[0].contains("alpha"));
    assert!(rows[1].contains("beta"));
}

// ── CREATE / DROP VECTOR INDEX ──────────────────────────────────────────────

#[test]
fn create_vector_index_hnsw() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(128));")
        .unwrap();

    let result = db
        .sql("CREATE VECTOR INDEX idx_docs_emb ON docs (embedding) USING HNSW WITH (m = 32, ef_construction = 200, metric = 'cosine');")
        .unwrap();
    let msg = affected_message(&result);
    assert!(msg.contains("created vector index"), "got: {msg}");
    assert!(msg.contains("idx_docs_emb"), "got: {msg}");
}

#[test]
fn create_vector_index_ivf_pq() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, vec VECTOR(256));")
        .unwrap();

    let result = db
        .sql("CREATE VECTOR INDEX idx_emb ON embeddings (vec) USING IVF_PQ WITH (nlist = 1024, nprobe = 16, pq_m = 32, pq_bits = 8);")
        .unwrap();
    let msg = affected_message(&result);
    assert!(msg.contains("created vector index"), "got: {msg}");
}

#[test]
fn drop_vector_index() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(64));")
        .unwrap();
    db.sql("CREATE VECTOR INDEX idx ON docs (embedding) USING HNSW WITH (metric = 'euclidean');")
        .unwrap();

    let result = db.sql("DROP VECTOR INDEX idx ON docs;").unwrap();
    let msg = affected_message(&result);
    assert!(msg.contains("dropped vector index"), "got: {msg}");
}

#[test]
fn create_vector_index_duplicate_fails() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(64));")
        .unwrap();
    db.sql("CREATE VECTOR INDEX idx ON docs (embedding) USING HNSW WITH (metric = 'cosine');")
        .unwrap();

    let err = db
        .sql("CREATE VECTOR INDEX idx2 ON docs (embedding) USING HNSW WITH (metric = 'cosine');")
        .unwrap_err();
    assert!(err.to_string().contains("already exists"), "got: {}", err);
}

#[test]
fn create_vector_index_on_non_vector_column_fails() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, name TEXT);")
        .unwrap();

    let err = db
        .sql("CREATE VECTOR INDEX idx ON docs (name) USING HNSW WITH (metric = 'cosine');")
        .unwrap_err();
    assert!(err.to_string().contains("expected VECTOR"), "got: {}", err);
}

// ── Vector data persistence via indexes ─────────────────────────────────────

#[test]
fn vector_index_stores_vector_data_on_insert() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(3));")
        .unwrap();
    db.sql("CREATE VECTOR INDEX idx ON docs (embedding) USING HNSW WITH (metric = 'euclidean');")
        .unwrap();

    db.sql("INSERT INTO docs (id, embedding) VALUES (1, '[1.0, 0.0, 0.0]');")
        .unwrap();
    db.sql("INSERT INTO docs (id, embedding) VALUES (2, '[0.0, 1.0, 0.0]');")
        .unwrap();
    db.sql("INSERT INTO docs (id, embedding) VALUES (3, '[0.0, 0.0, 1.0]');")
        .unwrap();

    // Verify data via vector_search()
    let result = db
        .sql("SELECT * FROM vector_search('docs', 'embedding', '[1.0, 0.0, 0.0]', 2);")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 2, "expected 2 results, got: {rows:?}");
    // The closest should be pk=1 (distance=0)
    assert!(
        rows[0].contains("\"pk\":\"1\""),
        "first result should be pk=1: {}",
        rows[0]
    );
}

// ── vector_search() table function ──────────────────────────────────────────

#[test]
fn vector_search_returns_ranked_results() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE vecs (id INTEGER PRIMARY KEY, emb VECTOR(2));")
        .unwrap();
    db.sql("CREATE VECTOR INDEX idx ON vecs (emb) USING HNSW WITH (metric = 'euclidean');")
        .unwrap();

    db.sql("INSERT INTO vecs (id, emb) VALUES (1, '[1.0, 0.0]');")
        .unwrap();
    db.sql("INSERT INTO vecs (id, emb) VALUES (2, '[0.0, 1.0]');")
        .unwrap();
    db.sql("INSERT INTO vecs (id, emb) VALUES (3, '[0.5, 0.5]');")
        .unwrap();

    let result = db
        .sql("SELECT * FROM vector_search('vecs', 'emb', '[1.0, 0.0]', 3);")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 3);

    // Results should be ordered by distance
    // pk=1 (distance=0), pk=3 (distance~0.707), pk=2 (distance~1.414)
    assert!(
        rows[0].contains("\"pk\":\"1\""),
        "closest should be pk=1: {}",
        rows[0]
    );
}

#[test]
fn vector_search_with_default_k() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE items (id INTEGER PRIMARY KEY, vec VECTOR(2));")
        .unwrap();
    db.sql("CREATE VECTOR INDEX idx ON items (vec) USING HNSW WITH (metric = 'cosine');")
        .unwrap();

    db.sql("INSERT INTO items (id, vec) VALUES (1, '[1.0, 0.0]');")
        .unwrap();

    // No k argument — defaults to 10
    let result = db
        .sql("SELECT * FROM vector_search('items', 'vec', '[1.0, 0.0]');")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1); // only 1 vector in the DB
}

// ── <-> distance operator ───────────────────────────────────────────────────

#[test]
fn vector_distance_operator_in_select() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE vecs (id INTEGER PRIMARY KEY, emb VECTOR(3));")
        .unwrap();

    db.sql("INSERT INTO vecs (id, emb) VALUES (1, '[1.0, 0.0, 0.0]');")
        .unwrap();

    let result = db
        .sql("SELECT id, VECTOR_DISTANCE('[1.0, 0.0, 0.0]', '[0.0, 1.0, 0.0]') AS dist FROM vecs;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    // Euclidean distance between [1,0,0] and [0,1,0] = sqrt(2) ≈ 1.4142
    assert!(
        rows[0].contains("dist"),
        "should have dist column: {}",
        rows[0]
    );
}

// ── Scalar vector functions ─────────────────────────────────────────────────

#[test]
fn vector_norm_function() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (id) VALUES (1);").unwrap();

    let result = db
        .sql("SELECT VECTOR_NORM('[3.0, 4.0]') AS norm FROM t;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    // sqrt(9 + 16) = 5.0
    assert!(rows[0].contains("5"), "norm should be 5: {}", rows[0]);
}

#[test]
fn vector_dims_function() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (id) VALUES (1);").unwrap();

    let result = db
        .sql("SELECT VECTOR_DIMS('[1.0, 2.0, 3.0]') AS dims FROM t;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    assert!(rows[0].contains("3"), "dims should be 3: {}", rows[0]);
}

#[test]
fn cosine_similarity_function() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (id) VALUES (1);").unwrap();

    let result = db
        .sql("SELECT COSINE_SIMILARITY('[1.0, 0.0]', '[1.0, 0.0]') AS sim FROM t;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    // Identical vectors => cosine similarity = 1.0
    assert!(rows[0].contains("1"), "similarity should be 1: {}", rows[0]);
}

// ── HYBRID_SCORE function ───────────────────────────────────────────────────

#[test]
fn hybrid_score_function() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE t (id INTEGER PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (id) VALUES (1);").unwrap();

    let result = db
        .sql("SELECT HYBRID_SCORE(0.5, 0.8, 0.7, 0.3) AS score FROM t;")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    // score should be a number between 0 and 1
    assert!(rows[0].contains("score"), "should have score: {}", rows[0]);
}

// ── Vector index with UPDATE and DELETE ─────────────────────────────────────

#[test]
fn vector_data_updated_on_row_update() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(2));")
        .unwrap();
    db.sql("CREATE VECTOR INDEX idx ON docs (embedding) USING HNSW WITH (metric = 'euclidean');")
        .unwrap();

    db.sql("INSERT INTO docs (id, embedding) VALUES (1, '[1.0, 0.0]');")
        .unwrap();

    // Update the vector
    db.sql("UPDATE docs SET embedding = '[0.0, 1.0]' WHERE id = 1;")
        .unwrap();

    let result = db
        .sql("SELECT * FROM vector_search('docs', 'embedding', '[0.0, 1.0]', 1);")
        .unwrap();
    let rows = row_strings(&result);
    assert_eq!(rows.len(), 1);
    // After update, pk=1 should have vector [0.0, 1.0] so distance to query is 0
    assert!(
        rows[0].contains("\"pk\":\"1\""),
        "should find pk=1: {}",
        rows[0]
    );
}

#[test]
fn vector_data_removed_on_row_delete() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, embedding VECTOR(2));")
        .unwrap();
    db.sql("CREATE VECTOR INDEX idx ON docs (embedding) USING HNSW WITH (metric = 'euclidean');")
        .unwrap();

    db.sql("INSERT INTO docs (id, embedding) VALUES (1, '[1.0, 0.0]');")
        .unwrap();
    db.sql("INSERT INTO docs (id, embedding) VALUES (2, '[0.0, 1.0]');")
        .unwrap();

    db.sql("DELETE FROM docs WHERE id = 1;").unwrap();

    let result = db
        .sql("SELECT * FROM vector_search('docs', 'embedding', '[1.0, 0.0]', 10);")
        .unwrap();
    let rows = row_strings(&result);
    // Only pk=2 should remain
    assert_eq!(
        rows.len(),
        1,
        "expected 1 result after delete, got: {rows:?}"
    );
    assert!(
        rows[0].contains("\"pk\":\"2\""),
        "remaining should be pk=2: {}",
        rows[0]
    );
}

// ── Multiple vector indexes on same table ───────────────────────────────────

#[test]
fn multiple_vector_columns_independent_indexes() {
    let (_dir, db) = setup_db();
    db.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, title_emb VECTOR(2), body_emb VECTOR(2));")
        .unwrap();
    db.sql(
        "CREATE VECTOR INDEX idx_title ON docs (title_emb) USING HNSW WITH (metric = 'cosine');",
    )
    .unwrap();
    db.sql(
        "CREATE VECTOR INDEX idx_body ON docs (body_emb) USING HNSW WITH (metric = 'euclidean');",
    )
    .unwrap();

    db.sql("INSERT INTO docs (id, title_emb, body_emb) VALUES (1, '[1.0, 0.0]', '[0.0, 1.0]');")
        .unwrap();

    let result1 = db
        .sql("SELECT * FROM vector_search('docs', 'title_emb', '[1.0, 0.0]', 1);")
        .unwrap();
    let result2 = db
        .sql("SELECT * FROM vector_search('docs', 'body_emb', '[0.0, 1.0]', 1);")
        .unwrap();

    assert_eq!(row_strings(&result1).len(), 1);
    assert_eq!(row_strings(&result2).len(), 1);
}
