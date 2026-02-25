use spectradb::{Config, Database};
use tempfile::TempDir;

fn open_db(dir: &TempDir) -> Database {
    Database::open(
        dir.path(),
        Config {
            shard_count: 1,
            ..Config::default()
        },
    )
    .unwrap()
}

fn setup_data(db: &Database) {
    db.sql("CREATE TABLE sales (pk TEXT PRIMARY KEY)").unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s1', '{\"region\":\"east\",\"amount\":100}')")
        .unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s2', '{\"region\":\"east\",\"amount\":200}')")
        .unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s3', '{\"region\":\"west\",\"amount\":150}')")
        .unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s4', '{\"region\":\"west\",\"amount\":300}')")
        .unwrap();
    db.sql("INSERT INTO sales (pk, doc) VALUES ('s5', '{\"region\":\"east\",\"amount\":50}')")
        .unwrap();
}

#[test]
fn row_number_over_order_by() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_data(&db);

    let result = db
        .sql("SELECT pk, ROW_NUMBER() OVER (ORDER BY doc.amount ASC) AS rn FROM sales ORDER BY doc.amount ASC")
        .unwrap();
    match result {
        spectradb_core::sql::exec::SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 5);
            // Check that row numbers are 1..5
            for (i, row) in rows.iter().enumerate() {
                let v: serde_json::Value = serde_json::from_slice(row).unwrap();
                assert_eq!(v["rn"], (i + 1) as f64);
            }
        }
        _ => panic!("expected rows"),
    }
}

#[test]
fn row_number_with_partition() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    setup_data(&db);

    let result = db
        .sql("SELECT pk, ROW_NUMBER() OVER (PARTITION BY doc.region ORDER BY doc.amount ASC) AS rn FROM sales")
        .unwrap();
    match result {
        spectradb_core::sql::exec::SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 5);
            // Each partition should have sequential row numbers
            for row in &rows {
                let v: serde_json::Value = serde_json::from_slice(row).unwrap();
                let rn = v["rn"].as_f64().unwrap();
                assert!(rn >= 1.0, "row_number should be >= 1, got {rn}");
            }
        }
        _ => panic!("expected rows"),
    }
}

#[test]
fn rank_with_ties() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE scores (pk TEXT PRIMARY KEY)").unwrap();
    db.sql("INSERT INTO scores (pk, doc) VALUES ('a', '{\"score\":100}')")
        .unwrap();
    db.sql("INSERT INTO scores (pk, doc) VALUES ('b', '{\"score\":200}')")
        .unwrap();
    db.sql("INSERT INTO scores (pk, doc) VALUES ('c', '{\"score\":200}')")
        .unwrap();
    db.sql("INSERT INTO scores (pk, doc) VALUES ('d', '{\"score\":300}')")
        .unwrap();

    let result = db
        .sql("SELECT pk, RANK() OVER (ORDER BY doc.score ASC) AS rnk FROM scores ORDER BY doc.score ASC")
        .unwrap();
    match result {
        spectradb_core::sql::exec::SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 4);
            let vals: Vec<serde_json::Value> = rows
                .iter()
                .map(|r| serde_json::from_slice(r).unwrap())
                .collect();
            // score=100 → rank 1, score=200 → rank 2 (tied), score=300 → rank 4
            assert_eq!(vals[0]["rnk"], 1.0);
            assert_eq!(vals[1]["rnk"], 2.0);
            assert_eq!(vals[2]["rnk"], 2.0);
            assert_eq!(vals[3]["rnk"], 4.0);
        }
        _ => panic!("expected rows"),
    }
}

#[test]
fn dense_rank() {
    let dir = TempDir::new().unwrap();
    let db = open_db(&dir);
    db.sql("CREATE TABLE scores (pk TEXT PRIMARY KEY)").unwrap();
    db.sql("INSERT INTO scores (pk, doc) VALUES ('a', '{\"score\":100}')")
        .unwrap();
    db.sql("INSERT INTO scores (pk, doc) VALUES ('b', '{\"score\":200}')")
        .unwrap();
    db.sql("INSERT INTO scores (pk, doc) VALUES ('c', '{\"score\":200}')")
        .unwrap();
    db.sql("INSERT INTO scores (pk, doc) VALUES ('d', '{\"score\":300}')")
        .unwrap();

    let result = db
        .sql("SELECT pk, DENSE_RANK() OVER (ORDER BY doc.score ASC) AS drnk FROM scores ORDER BY doc.score ASC")
        .unwrap();
    match result {
        spectradb_core::sql::exec::SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 4);
            let vals: Vec<serde_json::Value> = rows
                .iter()
                .map(|r| serde_json::from_slice(r).unwrap())
                .collect();
            assert_eq!(vals[0]["drnk"], 1.0);
            assert_eq!(vals[1]["drnk"], 2.0);
            assert_eq!(vals[2]["drnk"], 2.0);
            assert_eq!(vals[3]["drnk"], 3.0); // dense rank: 3, not 4
        }
        _ => panic!("expected rows"),
    }
}
