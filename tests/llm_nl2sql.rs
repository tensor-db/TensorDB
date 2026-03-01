/// Comprehensive LLM NL->SQL tests: easy -> hard.
///
/// These tests require the `llm` feature and a GGUF model file.
/// Set `LLM_MODEL_PATH` env var to override the model, otherwise
/// the default Qwen3-0.6B-Q8_0 is used.
///
/// Run:
///   cargo test --test llm_nl2sql -- --test-threads=1
///
/// With a custom model:
///   LLM_MODEL_PATH=/path/to/model.gguf cargo test --test llm_nl2sql -- --test-threads=1
#[cfg(feature = "llm")]
mod tests {
    use tensordb_core::config::Config;
    use tensordb_core::engine::db::Database;

    fn open_test_db(name: &str) -> Database {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        let _ = std::fs::create_dir_all(&path);

        let config = Config {
            shard_count: 1,
            ai_auto_insights: false,
            llm_model_path: std::env::var("LLM_MODEL_PATH").ok(),
            ..Config::default()
        };

        let db = Database::open(path, config).expect("open db");

        // Set up a realistic schema
        db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT, role TEXT, created_at INTEGER);").unwrap();
        db.sql("INSERT INTO users (id, name, email, role, created_at) VALUES (1, 'alice', 'alice@example.com', 'admin', 1700000000);").unwrap();
        db.sql("INSERT INTO users (id, name, email, role, created_at) VALUES (2, 'bob', 'bob@example.com', 'user', 1700100000);").unwrap();
        db.sql("INSERT INTO users (id, name, email, role, created_at) VALUES (3, 'carol', 'carol@example.com', 'admin', 1700200000);").unwrap();
        db.sql("INSERT INTO users (id, name, email, role, created_at) VALUES (4, 'dave', 'dave@example.com', 'user', 1700300000);").unwrap();
        db.sql("INSERT INTO users (id, name, email, role, created_at) VALUES (5, 'eve', 'eve@example.com', 'moderator', 1700400000);").unwrap();

        db.sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, product TEXT, amount REAL, status TEXT, ordered_at INTEGER);").unwrap();
        db.sql("INSERT INTO orders (id, user_id, product, amount, status, ordered_at) VALUES (101, 1, 'Widget A', 29.99, 'shipped', 1700500000);").unwrap();
        db.sql("INSERT INTO orders (id, user_id, product, amount, status, ordered_at) VALUES (102, 2, 'Widget B', 49.99, 'pending', 1700600000);").unwrap();
        db.sql("INSERT INTO orders (id, user_id, product, amount, status, ordered_at) VALUES (103, 1, 'Widget C', 15.00, 'delivered', 1700700000);").unwrap();
        db.sql("INSERT INTO orders (id, user_id, product, amount, status, ordered_at) VALUES (104, 3, 'Widget A', 29.99, 'cancelled', 1700800000);").unwrap();
        db.sql("INSERT INTO orders (id, user_id, product, amount, status, ordered_at) VALUES (105, 4, 'Widget D', 99.99, 'shipped', 1700900000);").unwrap();
        db.sql("INSERT INTO orders (id, user_id, product, amount, status, ordered_at) VALUES (106, 2, 'Widget A', 29.99, 'delivered', 1701000000);").unwrap();

        db.sql("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, category TEXT, stock INTEGER);").unwrap();
        db.sql("INSERT INTO products (id, name, price, category, stock) VALUES (1, 'Widget A', 29.99, 'electronics', 100);").unwrap();
        db.sql("INSERT INTO products (id, name, price, category, stock) VALUES (2, 'Widget B', 49.99, 'electronics', 50);").unwrap();
        db.sql("INSERT INTO products (id, name, price, category, stock) VALUES (3, 'Widget C', 15.00, 'accessories', 200);").unwrap();
        db.sql("INSERT INTO products (id, name, price, category, stock) VALUES (4, 'Widget D', 99.99, 'premium', 25);").unwrap();

        // Leak tempdir so it doesn't get deleted
        std::mem::forget(dir);
        db
    }

    /// Helper: generate SQL from question, print it, and optionally execute.
    fn ask_and_report(
        db: &Database,
        question: &str,
    ) -> (String, std::result::Result<String, String>) {
        eprintln!("\n======================================================================");
        eprintln!("Q: {question}");

        match db.ask_sql(question) {
            Ok(sql) => {
                eprintln!("  SQL: {sql}");
                match db.sql(&sql) {
                    Ok(result) => {
                        let result_str = format!("{result:?}");
                        let display = if result_str.len() > 200 {
                            format!("{}...", &result_str[..200])
                        } else {
                            result_str.clone()
                        };
                        eprintln!("  Result: {display}");
                        (sql, Ok(result_str))
                    }
                    Err(e) => {
                        eprintln!("  EXEC ERROR: {e}");
                        (sql, Err(e.to_string()))
                    }
                }
            }
            Err(e) => {
                eprintln!("  GEN ERROR: {e}");
                (String::new(), Err(e.to_string()))
            }
        }
    }

    /// Checks that generated SQL contains expected keywords (case-insensitive).
    fn sql_contains(sql: &str, keywords: &[&str]) -> bool {
        let upper = sql.to_uppercase();
        keywords.iter().all(|kw| upper.contains(&kw.to_uppercase()))
    }

    // =========================================================================
    // TIER 1: TRIVIAL
    // =========================================================================

    #[test]
    fn t1_show_tables() {
        let db = open_test_db("t1_show");
        let (sql, result) = ask_and_report(&db, "What tables exist?");
        assert!(
            sql_contains(&sql, &["SHOW"]) || sql_contains(&sql, &["TABLE"]),
            "Expected SHOW TABLES or similar, got: {sql}"
        );
        assert!(result.is_ok(), "Should execute successfully");
    }

    #[test]
    fn t1_count_all_users() {
        let db = open_test_db("t1_count");
        let (sql, result) = ask_and_report(&db, "How many users are there?");
        assert!(
            sql_contains(&sql, &["SELECT", "COUNT", "users"]),
            "Expected SELECT COUNT(*) FROM users, got: {sql}"
        );
        let res = result.unwrap();
        assert!(res.contains('5'), "Should return 5 users, got: {res}");
    }

    #[test]
    fn t1_select_all_products() {
        let db = open_test_db("t1_products");
        let (sql, result) = ask_and_report(&db, "Show me all products");
        assert!(
            sql_contains(&sql, &["SELECT", "products"]),
            "Expected SELECT ... FROM products, got: {sql}"
        );
        assert!(result.is_ok(), "Should execute successfully");
    }

    // =========================================================================
    // TIER 2: EASY
    // =========================================================================

    #[test]
    fn t2_filter_by_role() {
        let db = open_test_db("t2_role");
        let (sql, result) = ask_and_report(&db, "List all admin users");
        assert!(
            sql_contains(&sql, &["SELECT", "users", "admin"]),
            "Expected WHERE role = 'admin', got: {sql}"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn t2_order_by() {
        let db = open_test_db("t2_order");
        let (sql, result) =
            ask_and_report(&db, "Show products sorted by price from highest to lowest");
        assert!(
            sql_contains(&sql, &["SELECT", "products", "ORDER", "price"]),
            "Expected ORDER BY price DESC, got: {sql}"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn t2_total_revenue() {
        let db = open_test_db("t2_revenue");
        let (sql, result) = ask_and_report(&db, "What is the total amount of all orders?");
        assert!(
            sql_contains(&sql, &["SELECT", "SUM", "amount", "orders"]),
            "Expected SUM(amount) FROM orders, got: {sql}"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn t2_limit() {
        let db = open_test_db("t2_limit");
        let (sql, result) = ask_and_report(&db, "Show the top 3 most expensive products");
        assert!(
            sql_contains(&sql, &["SELECT", "products", "ORDER", "LIMIT"]),
            "Expected ORDER BY price DESC LIMIT 3, got: {sql}"
        );
        assert!(result.is_ok());
    }

    // =========================================================================
    // TIER 3: MEDIUM
    // =========================================================================

    #[test]
    fn t3_join_users_orders() {
        let db = open_test_db("t3_join");
        let (sql, result) = ask_and_report(&db, "Show each user's name and their order products");
        assert!(
            sql_contains(&sql, &["SELECT", "JOIN", "users", "orders"]),
            "Expected a JOIN between users and orders, got: {sql}"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn t3_group_by_with_count() {
        let db = open_test_db("t3_groupby");
        let (sql, result) = ask_and_report(&db, "How many orders does each user have?");
        assert!(
            sql_contains(&sql, &["SELECT", "COUNT", "GROUP BY"]),
            "Expected GROUP BY with COUNT, got: {sql}"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn t3_multi_condition() {
        let db = open_test_db("t3_multi");
        let (sql, result) = ask_and_report(
            &db,
            "Find orders that are shipped and have amount greater than 20",
        );
        assert!(
            sql_contains(&sql, &["SELECT", "orders", "shipped"]),
            "Expected filter on status='shipped' AND amount>20, got: {sql}"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn t3_aggregate_with_join() {
        let db = open_test_db("t3_agg_join");
        let (sql, result) = ask_and_report(
            &db,
            "What is the total order amount per user? Show user names.",
        );
        assert!(
            sql_contains(&sql, &["SELECT", "SUM", "JOIN", "GROUP BY"]),
            "Expected JOIN + SUM + GROUP BY, got: {sql}"
        );
        assert!(result.is_ok());
    }

    // =========================================================================
    // TIER 4: HARD
    // =========================================================================

    #[test]
    fn t4_having_clause() {
        let db = open_test_db("t4_having");
        let (sql, result) = ask_and_report(&db, "Which users have placed more than 1 order?");
        assert!(
            sql_contains(&sql, &["SELECT", "COUNT", "GROUP BY"]),
            "Expected GROUP BY ... HAVING COUNT > 1, got: {sql}"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn t4_subquery() {
        let db = open_test_db("t4_subquery");
        let (sql, _result) = ask_and_report(&db, "Find users who have never placed an order");
        assert!(
            sql_contains(&sql, &["SELECT", "users"]),
            "Expected query referencing users and orders, got: {sql}"
        );
    }

    #[test]
    fn t4_case_expression() {
        let db = open_test_db("t4_case");
        let (sql, _result) = ask_and_report(
            &db,
            "Categorize orders: amounts under 30 as 'cheap', 30-50 as 'mid', over 50 as 'expensive'",
        );
        assert!(
            sql_contains(&sql, &["SELECT", "CASE"]) || sql_contains(&sql, &["SELECT", "IIF"]),
            "Expected CASE WHEN or IIF expression, got: {sql}"
        );
    }

    // =========================================================================
    // TIER 5: VERY HARD
    // =========================================================================

    #[test]
    fn t5_window_function() {
        let db = open_test_db("t5_window");
        let (sql, _result) = ask_and_report(
            &db,
            "Rank users by their total order spend, showing their name and rank",
        );
        assert!(
            sql_contains(&sql, &["SELECT", "users"]),
            "Expected a query involving users and orders, got: {sql}"
        );
    }

    #[test]
    fn t5_cte() {
        let db = open_test_db("t5_cte");
        let (sql, _result) = ask_and_report(
            &db,
            "Using a CTE, first calculate each user's total spend, then show only users who spent more than 30",
        );
        if sql_contains(&sql, &["WITH"]) {
            eprintln!("  Model used CTE correctly");
        } else {
            eprintln!("  Note: Model didn't use CTE -- may have used subquery instead");
        }
    }

    #[test]
    fn t5_temporal_query() {
        let db = open_test_db("t5_temporal");
        let (sql, _result) = ask_and_report(
            &db,
            "Show the state of the users table as it was at commit timestamp 1",
        );
        assert!(
            sql_contains(&sql, &["SELECT", "AS OF"])
                || sql_contains(&sql, &["SELECT", "SYSTEM_TIME"]),
            "Expected AS OF or SYSTEM_TIME temporal clause, got: {sql}"
        );
    }

    #[test]
    fn t5_describe_table() {
        let db = open_test_db("t5_describe");
        let (sql, result) = ask_and_report(&db, "What columns does the orders table have?");
        assert!(
            sql_contains(&sql, &["DESCRIBE", "orders"]) || sql_contains(&sql, &["SHOW", "orders"]),
            "Expected DESCRIBE orders, got: {sql}"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn t5_complex_multi_join() {
        let db = open_test_db("t5_complex");
        let (sql, _result) = ask_and_report(
            &db,
            "Show each user's name, their orders, and the product category for each order. Join users, orders, and products.",
        );
        assert!(
            sql_contains(&sql, &["SELECT", "JOIN"]),
            "Expected multi-table JOIN, got: {sql}"
        );
    }

    // =========================================================================
    // TIER 6: EXTREME
    // =========================================================================

    #[test]
    fn t6_ambiguous_natural_language() {
        let db = open_test_db("t6_ambiguous");
        let (sql, _result) = ask_and_report(&db, "Who spent the most?");
        assert!(
            !sql.is_empty(),
            "Model should generate something, got empty"
        );
    }

    #[test]
    fn t6_insert_generation() {
        let db = open_test_db("t6_insert");
        let (sql, result) = ask_and_report(
            &db,
            "Add a new user named 'frank' with email 'frank@example.com' and role 'user'",
        );
        assert!(
            sql_contains(&sql, &["INSERT", "users", "frank"]),
            "Expected INSERT INTO users, got: {sql}"
        );
        if result.is_ok() {
            let check = db
                .sql("SELECT COUNT(*) FROM users WHERE name = 'frank';")
                .unwrap();
            eprintln!("  Verification: {check:?}");
        }
    }

    #[test]
    fn t6_update_generation() {
        let db = open_test_db("t6_update");
        let (sql, _result) = ask_and_report(&db, "Change bob's role to 'admin'");
        assert!(
            sql_contains(&sql, &["UPDATE", "users", "admin", "bob"]),
            "Expected UPDATE users SET role='admin' WHERE name='bob', got: {sql}"
        );
    }

    // =========================================================================
    // SCORECARD: runs all difficulty levels and prints summary
    // =========================================================================

    #[test]
    fn scorecard() {
        let db = open_test_db("scorecard");

        let questions: Vec<(&str, &str, Vec<&str>)> = vec![
            ("TRIVIAL", "What tables are available?", vec!["SHOW"]),
            ("TRIVIAL", "How many users?", vec!["SELECT", "COUNT"]),
            ("EASY", "Show all admin users", vec!["SELECT", "admin"]),
            (
                "EASY",
                "Total order amount",
                vec!["SELECT", "SUM", "amount"],
            ),
            (
                "MEDIUM",
                "Show user names with their orders",
                vec!["SELECT", "JOIN"],
            ),
            (
                "MEDIUM",
                "Orders per user count",
                vec!["SELECT", "COUNT", "GROUP BY"],
            ),
            (
                "HARD",
                "Users with more than 1 order",
                vec!["SELECT", "COUNT"],
            ),
            ("HARD", "Describe the orders table", vec!["DESCRIBE"]),
            (
                "VERY HARD",
                "User table state at commit 1",
                vec!["SELECT", "AS OF"],
            ),
            ("EXTREME", "Who spent the most money?", vec!["SELECT"]),
        ];

        let mut pass = 0;
        let mut fail = 0;
        let mut errors = 0;

        eprintln!("\n======================================================================");
        eprintln!("LLM NL->SQL SCORECARD");
        eprintln!("======================================================================");

        for (difficulty, question, keywords) in &questions {
            let (sql, result) = ask_and_report(&db, question);

            let sql_ok = if sql.is_empty() {
                false
            } else {
                sql_contains(&sql, keywords)
            };

            let exec_ok = result.is_ok();

            let status = if sql_ok && exec_ok {
                pass += 1;
                "PASS"
            } else if sql_ok {
                errors += 1;
                "SQL OK, EXEC FAIL"
            } else {
                fail += 1;
                "FAIL"
            };

            eprintln!("  [{difficulty:>10}] {status:>16} | {question}");
            if !sql.is_empty() {
                eprintln!("               SQL: {sql}");
            }
        }

        eprintln!("\n======================================================================");
        eprintln!(
            "RESULTS: {} passed, {} failed, {} exec errors out of {}",
            pass,
            fail,
            errors,
            questions.len()
        );
        eprintln!("======================================================================\n");

        // A small 0.6B model should get at least trivial + easy right
        assert!(
            pass >= 3,
            "Model should pass at least 3/10 questions, got {pass}"
        );
    }
}
