/// Integration tests for v0.17 SQL Completeness features.
use tensordb_core::config::Config;
use tensordb_core::Database;

fn test_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

fn sql(db: &Database, query: &str) -> Vec<serde_json::Value> {
    match db.sql(query).unwrap() {
        tensordb_core::sql::exec::SqlResult::Rows(rows) => rows
            .into_iter()
            .map(|r| {
                serde_json::from_slice(&r).unwrap_or_else(|_| {
                    // Fallback: treat as plain text value
                    let s = String::from_utf8_lossy(&r);
                    serde_json::json!({ "result": s.as_ref() })
                })
            })
            .collect(),
        tensordb_core::sql::exec::SqlResult::Affected { message, .. } => {
            vec![serde_json::json!({ "message": message })]
        }
        tensordb_core::sql::exec::SqlResult::Explain(text) => {
            vec![serde_json::json!({ "explain": text })]
        }
    }
}

fn sql_affected(db: &Database, query: &str) -> String {
    match db.sql(query).unwrap() {
        tensordb_core::sql::exec::SqlResult::Affected { message, .. } => message,
        other => panic!("expected Affected, got: {other:?}"),
    }
}

fn setup_test_table(db: &Database) {
    db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, city TEXT);")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age, city) VALUES (1, 'Alice', 30, 'New York');")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age, city) VALUES (2, 'Bob', 25, 'San Francisco');")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age, city) VALUES (3, 'Charlie', 35, 'New York');")
        .unwrap();
    db.sql("INSERT INTO users (id, name, age, city) VALUES (4, 'Diana', 28, 'Chicago');")
        .unwrap();
}

// ========== String Functions ==========

#[test]
fn test_substr() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT SUBSTR(name, 1, 3) AS s FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["s"], "Ali");
}

#[test]
fn test_substr_no_length() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT SUBSTR(name, 3) AS s FROM users WHERE id = 1;");
    assert_eq!(rows[0]["s"], "ice");
}

#[test]
fn test_trim() {
    let (db, _dir) = test_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"val\":\"  hello  \"}');")
        .unwrap();
    let rows = sql(&db, "SELECT TRIM(val) AS v FROM t;");
    assert_eq!(rows[0]["v"], "hello");
}

#[test]
fn test_replace() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT REPLACE(name, 'li', 'LI') AS r FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["r"], "ALIce");
}

#[test]
fn test_concat() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT CONCAT(name, ' from ', city) AS full FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["full"], "Alice from New York");
}

#[test]
fn test_concat_ws() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT CONCAT_WS(', ', name, city) AS full FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["full"], "Alice, New York");
}

#[test]
fn test_left_right() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT LEFT(name, 3) AS l, RIGHT(name, 3) AS r FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["l"], "Ali");
    assert_eq!(rows[0]["r"], "ice");
}

#[test]
fn test_lpad_rpad() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT LPAD(name, 8, '*') AS l, RPAD(name, 8, '*') AS r FROM users WHERE id = 2;",
    );
    assert_eq!(rows[0]["l"], "*****Bob");
    // RPAD: "Bob" + "*****" = "Bob*****"
    assert_eq!(rows[0]["r"], "Bob*****");
}

#[test]
fn test_reverse() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT REVERSE(name) AS r FROM users WHERE id = 2;");
    assert_eq!(rows[0]["r"], "boB");
}

#[test]
fn test_split_part() {
    let (db, _dir) = test_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"email\":\"user@example.com\"}');")
        .unwrap();
    let rows = sql(
        &db,
        "SELECT SPLIT_PART(email, '@', 1) AS user_part, SPLIT_PART(email, '@', 2) AS domain FROM t;",
    );
    assert_eq!(rows[0]["user_part"], "user");
    assert_eq!(rows[0]["domain"], "example.com");
}

#[test]
fn test_repeat() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT REPEAT(name, 2) AS r FROM users WHERE id = 2;");
    assert_eq!(rows[0]["r"], "BobBob");
}

#[test]
fn test_position() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT POSITION(name, 'li') AS p FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["p"], 2.0);
}

#[test]
fn test_initcap() {
    let (db, _dir) = test_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"val\":\"hello world\"}');")
        .unwrap();
    let rows = sql(&db, "SELECT INITCAP(val) AS r FROM t;");
    assert_eq!(rows[0]["r"], "Hello World");
}

// ========== Math Functions ==========

#[test]
fn test_round() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT ROUND(3.14159, 2) AS r FROM users WHERE id = 1;",
    );
    #[allow(clippy::approx_constant)]
    let expected = 3.14;
    assert!((rows[0]["r"].as_f64().unwrap() - expected).abs() < 0.001);
}

#[test]
fn test_ceil_floor() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT CEIL(3.2) AS c, FLOOR(3.8) AS f FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["c"], 4.0);
    assert_eq!(rows[0]["f"], 3.0);
}

#[test]
fn test_power_sqrt() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT POWER(2, 10) AS p, SQRT(144) AS s FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["p"], 1024.0);
    assert_eq!(rows[0]["s"], 12.0);
}

#[test]
fn test_log_ln() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT ROUND(LOG(100), 0) AS l, ROUND(LN(1), 0) AS n FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["l"], 2.0);
    assert_eq!(rows[0]["n"], 0.0);
}

#[test]
fn test_mod_function() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT MOD(10, 3) AS m FROM users WHERE id = 1;");
    assert_eq!(rows[0]["m"], 1.0);
}

#[test]
fn test_sign() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT SIGN(5) AS pos, SIGN(0) AS zero FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["pos"], 1.0);
    assert_eq!(rows[0]["zero"], 0.0);
}

#[test]
fn test_random() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT RANDOM() AS r FROM users WHERE id = 1;");
    let r = rows[0]["r"].as_f64().unwrap();
    assert!((0.0..1.0).contains(&r));
}

#[test]
fn test_pi() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT PI() AS p FROM users WHERE id = 1;");
    let p = rows[0]["p"].as_f64().unwrap();
    assert!((p - std::f64::consts::PI).abs() < 1e-10);
}

// ========== Utility Functions ==========

#[test]
fn test_nullif() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT NULLIF(age, 30) AS r FROM users WHERE id = 1;");
    assert!(rows[0]["r"].is_null());
    let rows2 = sql(&db, "SELECT NULLIF(age, 99) AS r FROM users WHERE id = 1;");
    assert_eq!(rows2[0]["r"], 30.0);
}

#[test]
fn test_greatest_least() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT GREATEST(1, 5, 3) AS g, LEAST(1, 5, 3) AS l FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["g"], 5.0);
    assert_eq!(rows[0]["l"], 1.0);
}

#[test]
fn test_if_function() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT IIF(age > 30, 'senior', 'junior') AS cat FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["cat"], "junior");
    let rows2 = sql(
        &db,
        "SELECT IIF(age > 30, 'senior', 'junior') AS cat FROM users WHERE id = 3;",
    );
    assert_eq!(rows2[0]["cat"], "senior");
}

// ========== Date/Time Functions ==========

#[test]
fn test_now() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT NOW() AS ts FROM users WHERE id = 1;");
    let ts = rows[0]["ts"].as_f64().unwrap();
    // Should be a Unix epoch timestamp > 2020
    assert!(ts > 1577836800.0); // 2020-01-01
}

#[test]
fn test_extract() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    // 2023-01-15 12:30:45 UTC = 1673785845
    let rows = sql(
        &db,
        "SELECT EXTRACT('HOUR', 1673785845) AS h, EXTRACT('MINUTE', 1673785845) AS m, EXTRACT('YEAR', 1673785845) AS y FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["h"], 12.0);
    assert_eq!(rows[0]["m"], 30.0);
    assert_eq!(rows[0]["y"], 2023.0);
}

#[test]
fn test_date_trunc() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    // 2023-01-15 12:30:45 UTC = 1673785845
    // Truncate to day: 2023-01-15 00:00:00 = 1673740800
    let rows = sql(
        &db,
        "SELECT DATE_TRUNC('DAY', 1673785845) AS d FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["d"], 1673740800.0);
}

// ========== ILIKE ==========

#[test]
fn test_ilike() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT name FROM users WHERE name ILIKE 'ALICE';");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], "Alice");
}

#[test]
fn test_ilike_pattern() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT name FROM users WHERE name ILIKE '%LI%';");
    assert_eq!(rows.len(), 2); // Alice and Charlie
}

#[test]
fn test_not_ilike() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT name FROM users WHERE name NOT ILIKE 'alice';");
    assert_eq!(rows.len(), 3); // Bob, Charlie, Diana
}

// ========== UNION / INTERSECT / EXCEPT ==========

#[test]
fn test_union_all() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT name FROM users WHERE city = 'New York' UNION ALL SELECT name FROM users WHERE city = 'Chicago';",
    );
    assert_eq!(rows.len(), 3); // Alice, Charlie, Diana
}

#[test]
fn test_union_dedup() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    // Both queries return Alice
    let rows = sql(
        &db,
        "SELECT name FROM users WHERE id = 1 UNION SELECT name FROM users WHERE id = 1;",
    );
    assert_eq!(rows.len(), 1); // Deduplicated
}

#[test]
fn test_intersect() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT name FROM users WHERE city = 'New York' INTERSECT SELECT name FROM users WHERE age > 30;",
    );
    assert_eq!(rows.len(), 1); // Only Charlie (New York AND age > 30)
    assert_eq!(rows[0]["name"], "Charlie");
}

#[test]
fn test_except() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT name FROM users WHERE city = 'New York' EXCEPT SELECT name FROM users WHERE age > 30;",
    );
    assert_eq!(rows.len(), 1); // Alice (New York but NOT age > 30)
    assert_eq!(rows[0]["name"], "Alice");
}

// ========== INSERT ... RETURNING ==========

#[test]
fn test_insert_returning() {
    let (db, _dir) = test_db();
    db.sql("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL);")
        .unwrap();
    let rows = sql(
        &db,
        "INSERT INTO items (id, name, price) VALUES (1, 'Widget', 9.99) RETURNING *;",
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], "Widget");
    assert_eq!(rows[0]["price"], 9.99);
}

#[test]
fn test_insert_returning_specific_columns() {
    let (db, _dir) = test_db();
    db.sql("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, price REAL);")
        .unwrap();
    let rows = sql(
        &db,
        "INSERT INTO items (id, name, price) VALUES (1, 'Widget', 9.99) RETURNING name;",
    );
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], "Widget");
}

// ========== CREATE TABLE AS SELECT ==========

#[test]
fn test_create_table_as_select() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    sql_affected(
        &db,
        "CREATE TABLE ny_users AS SELECT id, name, city FROM users WHERE city = 'New York';",
    );
    let rows = sql(&db, "SELECT name, city FROM ny_users;");
    assert_eq!(rows.len(), 2); // Alice and Charlie
}

// ========== Additional Aggregates ==========

#[test]
fn test_string_agg() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT STRING_AGG(name, ', ') AS names FROM users WHERE city = 'New York';",
    );
    let names = rows[0]["names"].as_str().unwrap();
    assert!(names.contains("Alice"));
    assert!(names.contains("Charlie"));
}

#[test]
fn test_stddev() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT ROUND(STDDEV_POP(age), 2) AS sd FROM users;");
    let sd = rows[0]["sd"].as_f64().unwrap();
    // ages: 30, 25, 35, 28 → mean=29.5, variance=13.25, stddev≈3.64
    assert!((sd - 3.64).abs() < 0.1);
}

#[test]
fn test_variance() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT ROUND(VAR_POP(age), 2) AS v FROM users;");
    let v = rows[0]["v"].as_f64().unwrap();
    // ages: 30, 25, 35, 28 → mean=29.5, variance=13.25
    assert!((v - 13.25).abs() < 0.1);
}

// ========== Existing Features Still Work ==========

#[test]
fn test_case_still_works() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT CASE WHEN age > 30 THEN 'senior' ELSE 'junior' END AS cat FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["cat"], "junior");
}

#[test]
fn test_cast_still_works() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT CAST(age AS TEXT) AS a FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["a"], "30");
}

#[test]
fn test_coalesce_still_works() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT COALESCE(NULL, name) AS c FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["c"], "Alice");
}

#[test]
fn test_between_still_works() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT name FROM users WHERE age BETWEEN 26 AND 32;");
    assert_eq!(rows.len(), 2); // Alice (30), Diana (28)
}

// ========== Combined Features ==========

#[test]
fn test_functions_in_where() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT name FROM users WHERE LENGTH(name) > 4;");
    assert_eq!(rows.len(), 3); // Alice (5), Charlie (7), Diana (5)
}

#[test]
fn test_math_in_select() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(
        &db,
        "SELECT name, ROUND(age * 1.1, 1) AS adj_age FROM users WHERE id = 1;",
    );
    assert_eq!(rows[0]["name"], "Alice");
    assert!((rows[0]["adj_age"].as_f64().unwrap() - 33.0).abs() < 0.01);
}

#[test]
fn test_exp() {
    let (db, _dir) = test_db();
    setup_test_table(&db);
    let rows = sql(&db, "SELECT ROUND(EXP(1), 2) AS e FROM users WHERE id = 1;");
    let e = rows[0]["e"].as_f64().unwrap();
    assert!((e - 2.72).abs() < 0.01);
}

#[test]
fn test_ltrim_rtrim() {
    let (db, _dir) = test_db();
    db.sql("CREATE TABLE t (pk TEXT PRIMARY KEY);").unwrap();
    db.sql("INSERT INTO t (pk, doc) VALUES ('k1', '{\"val\":\"  hello  \"}');")
        .unwrap();
    let rows = sql(&db, "SELECT LTRIM(val) AS l, RTRIM(val) AS r FROM t;");
    assert_eq!(rows[0]["l"], "hello  ");
    assert_eq!(rows[0]["r"], "  hello");
}
