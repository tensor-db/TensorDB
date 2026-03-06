//! End-to-end NL2SQL test with medium-to-hard queries.
//! Run: LLM_MODEL_PATH=~/.tensordb/models/Qwen3-0.6B-Q8_0.gguf \
//!   cargo test --release --test test_nl2sql_e2e -- --nocapture --test-threads=1
#[cfg(feature = "llm")]
mod tests {
    use tensordb_core::config::Config;
    use tensordb_core::engine::db::Database;

    fn setup_db() -> (Database, tempfile::TempDir) {
        let home = std::env::var("HOME").unwrap_or_default();
        let model = std::env::var("LLM_MODEL_PATH")
            .unwrap_or_else(|_| format!("{home}/.tensordb/models/Qwen3-0.6B-Q8_0.gguf"));
        if !std::path::Path::new(&model).exists() {
            panic!("Model not found at {model}");
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_e2e");
        let _ = std::fs::create_dir_all(&path);

        let config = Config {
            shard_count: 1,
            ai_auto_insights: false,
            llm_model_path: Some(model),
            ..Config::default()
        };

        let db = Database::open(path, config).expect("open db");

        // -- Schema: e-commerce with 4 tables --
        db.sql("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, country TEXT, tier TEXT);").unwrap();
        db.sql("INSERT INTO customers (id, name, email, country, tier) VALUES (1, 'Alice Johnson', 'alice@example.com', 'US', 'gold');").unwrap();
        db.sql("INSERT INTO customers (id, name, email, country, tier) VALUES (2, 'Bob Smith', 'bob@example.com', 'UK', 'silver');").unwrap();
        db.sql("INSERT INTO customers (id, name, email, country, tier) VALUES (3, 'Carol Lee', 'carol@example.com', 'US', 'gold');").unwrap();
        db.sql("INSERT INTO customers (id, name, email, country, tier) VALUES (4, 'Dave Brown', 'dave@example.com', 'DE', 'bronze');").unwrap();
        db.sql("INSERT INTO customers (id, name, email, country, tier) VALUES (5, 'Eve Wilson', 'eve@example.com', 'US', 'silver');").unwrap();

        db.sql("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER);").unwrap();
        db.sql("INSERT INTO products (id, name, category, price, stock) VALUES (1, 'Laptop Pro', 'electronics', 1299.99, 45);").unwrap();
        db.sql("INSERT INTO products (id, name, category, price, stock) VALUES (2, 'Wireless Mouse', 'electronics', 29.99, 200);").unwrap();
        db.sql("INSERT INTO products (id, name, category, price, stock) VALUES (3, 'Standing Desk', 'furniture', 549.00, 30);").unwrap();
        db.sql("INSERT INTO products (id, name, category, price, stock) VALUES (4, 'Monitor 4K', 'electronics', 399.99, 75);").unwrap();
        db.sql("INSERT INTO products (id, name, category, price, stock) VALUES (5, 'Keyboard', 'electronics', 79.99, 150);").unwrap();
        db.sql("INSERT INTO products (id, name, category, price, stock) VALUES (6, 'Desk Lamp', 'furniture', 45.00, 100);").unwrap();

        db.sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, quantity INTEGER, total REAL, status TEXT, order_date TEXT);").unwrap();
        db.sql("INSERT INTO orders (id, customer_id, product_id, quantity, total, status, order_date) VALUES (1, 1, 1, 1, 1299.99, 'shipped', '2024-01-15');").unwrap();
        db.sql("INSERT INTO orders (id, customer_id, product_id, quantity, total, status, order_date) VALUES (2, 1, 2, 2, 59.98, 'delivered', '2024-01-20');").unwrap();
        db.sql("INSERT INTO orders (id, customer_id, product_id, quantity, total, status, order_date) VALUES (3, 2, 3, 1, 549.00, 'pending', '2024-02-01');").unwrap();
        db.sql("INSERT INTO orders (id, customer_id, product_id, quantity, total, status, order_date) VALUES (4, 3, 4, 2, 799.98, 'shipped', '2024-02-10');").unwrap();
        db.sql("INSERT INTO orders (id, customer_id, product_id, quantity, total, status, order_date) VALUES (5, 3, 5, 3, 239.97, 'delivered', '2024-02-15');").unwrap();
        db.sql("INSERT INTO orders (id, customer_id, product_id, quantity, total, status, order_date) VALUES (6, 4, 1, 1, 1299.99, 'cancelled', '2024-03-01');").unwrap();
        db.sql("INSERT INTO orders (id, customer_id, product_id, quantity, total, status, order_date) VALUES (7, 5, 6, 2, 90.00, 'delivered', '2024-03-05');").unwrap();
        db.sql("INSERT INTO orders (id, customer_id, product_id, quantity, total, status, order_date) VALUES (8, 2, 2, 5, 149.95, 'shipped', '2024-03-10');").unwrap();
        db.sql("INSERT INTO orders (id, customer_id, product_id, quantity, total, status, order_date) VALUES (9, 1, 4, 1, 399.99, 'delivered', '2024-03-15');").unwrap();

        db.sql("CREATE TABLE reviews (id INTEGER PRIMARY KEY, product_id INTEGER, customer_id INTEGER, rating INTEGER, comment TEXT);").unwrap();
        db.sql("INSERT INTO reviews (id, product_id, customer_id, rating, comment) VALUES (1, 1, 1, 5, 'Excellent laptop');").unwrap();
        db.sql("INSERT INTO reviews (id, product_id, customer_id, rating, comment) VALUES (2, 2, 1, 4, 'Good mouse');").unwrap();
        db.sql("INSERT INTO reviews (id, product_id, customer_id, rating, comment) VALUES (3, 3, 2, 3, 'Decent desk');").unwrap();
        db.sql("INSERT INTO reviews (id, product_id, customer_id, rating, comment) VALUES (4, 4, 3, 5, 'Great monitor');").unwrap();
        db.sql("INSERT INTO reviews (id, product_id, customer_id, rating, comment) VALUES (5, 5, 3, 4, 'Nice keyboard');").unwrap();

        (db, dir)
    }

    struct TestCase {
        question: &'static str,
        difficulty: &'static str,
        /// Keywords that should appear in a correct SQL (case-insensitive)
        expect_keywords: Vec<&'static str>,
        /// If true, we also try executing the SQL (it should not error)
        expect_executable: bool,
    }

    #[test]
    fn nl2sql_medium_to_hard() {
        let (db, _dir) = setup_db();

        let cases = vec![
            // --- EASY (warm up) ---
            TestCase {
                question: "Show all customers",
                difficulty: "easy",
                expect_keywords: vec!["SELECT", "customers"],
                expect_executable: true,
            },
            TestCase {
                question: "How many products are there?",
                difficulty: "easy",
                expect_keywords: vec!["SELECT", "COUNT", "products"],
                expect_executable: true,
            },
            // --- MEDIUM ---
            TestCase {
                question: "List all electronics products sorted by price descending",
                difficulty: "medium",
                expect_keywords: vec!["SELECT", "products", "ORDER BY", "price"],
                expect_executable: true,
            },
            TestCase {
                question: "Which customers are from the US?",
                difficulty: "medium",
                expect_keywords: vec!["SELECT", "customers", "WHERE"],
                expect_executable: true,
            },
            TestCase {
                question: "Show all delivered orders",
                difficulty: "medium",
                expect_keywords: vec!["SELECT", "orders", "delivered"],
                expect_executable: true,
            },
            TestCase {
                question: "What is the total revenue from all orders?",
                difficulty: "medium",
                expect_keywords: vec!["SELECT", "SUM", "total"],
                expect_executable: true,
            },
            TestCase {
                question: "Show products with stock less than 50",
                difficulty: "medium",
                expect_keywords: vec!["SELECT", "products", "stock"],
                expect_executable: true,
            },
            // --- HARD ---
            TestCase {
                question: "What is the average order total?",
                difficulty: "hard",
                expect_keywords: vec!["SELECT", "AVG", "total"],
                expect_executable: true,
            },
            TestCase {
                question: "Show the most expensive product",
                difficulty: "hard",
                expect_keywords: vec!["SELECT", "products", "price"],
                expect_executable: true,
            },
            TestCase {
                question: "How many orders does each customer have?",
                difficulty: "hard",
                expect_keywords: vec!["SELECT", "COUNT", "GROUP BY"],
                expect_executable: true,
            },
            TestCase {
                question: "What is the average product rating?",
                difficulty: "hard",
                expect_keywords: vec!["SELECT", "AVG", "rating"],
                expect_executable: true,
            },
            TestCase {
                question: "Show all shipped orders with their total amount",
                difficulty: "hard",
                expect_keywords: vec!["SELECT", "orders", "shipped"],
                expect_executable: true,
            },
        ];

        eprintln!("\n{}", "=".repeat(70));
        eprintln!("  NL2SQL END-TO-END TEST — Qwen3 0.6B");
        eprintln!("  {} test cases (easy → hard)  ", cases.len());
        eprintln!("{}\n", "=".repeat(70));

        let mut pass = 0;
        let mut fail = 0;
        let mut exec_pass = 0;
        let mut exec_fail = 0;

        for (i, tc) in cases.iter().enumerate() {
            let start = std::time::Instant::now();
            eprintln!(
                "[{}/{}] [{}] Q: {}",
                i + 1,
                cases.len(),
                tc.difficulty,
                tc.question
            );

            match db.ask_sql(tc.question) {
                Ok(sql) => {
                    let elapsed = start.elapsed();
                    let sql_upper = sql.to_uppercase();
                    let keywords_ok = tc
                        .expect_keywords
                        .iter()
                        .all(|kw| sql_upper.contains(&kw.to_uppercase()));

                    let status = if keywords_ok { "PASS" } else { "PARTIAL" };
                    if keywords_ok {
                        pass += 1;
                    } else {
                        fail += 1;
                    }

                    eprintln!("  SQL: {sql}");
                    eprintln!("  Keywords: {status} (expected: {:?})", tc.expect_keywords);
                    eprintln!("  Time: {:.1}s", elapsed.as_secs_f64());

                    if tc.expect_executable {
                        match db.sql(&sql) {
                            Ok(result) => {
                                exec_pass += 1;
                                eprintln!("  Exec: OK — {result:?}");
                            }
                            Err(e) => {
                                exec_fail += 1;
                                eprintln!("  Exec: FAIL — {e}");
                            }
                        }
                    }
                }
                Err(e) => {
                    fail += 1;
                    exec_fail += 1;
                    eprintln!("  ERROR: {e}");
                }
            }
            eprintln!();
        }

        eprintln!("{}", "=".repeat(70));
        eprintln!("  SCORECARD");
        eprintln!("  SQL generation: {pass}/{} pass", pass + fail);
        eprintln!(
            "  SQL execution:  {exec_pass}/{} pass",
            exec_pass + exec_fail
        );
        eprintln!(
            "  Pass rate:      {:.0}%",
            (pass as f64 / (pass + fail) as f64) * 100.0
        );
        eprintln!("{}\n", "=".repeat(70));

        std::mem::forget(_dir);
    }
}
