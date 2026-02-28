#!/usr/bin/env python3
"""Generate training data for TensorDB NL→SQL fine-tuning.

Combines:
  1. ~10K subsampled examples from b-mc2/sql-create-context (HuggingFace)
  2. ~300 hand-crafted TensorDB-specific examples

Output: training_data.jsonl and eval_data.jsonl (95/5 split)
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

# ---------------------------------------------------------------------------
# System prompt — must match SYSTEM_PROMPT in llm.rs exactly
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a SQL translator for TensorDB (a bitemporal database). "
    "TensorDB SQL supports: SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, "
    "SHOW TABLES, DESCRIBE <table>, time-travel (AS OF <timestamp>), "
    "aggregates (count, sum, avg, min, max), JOINs, CTEs, window functions. "
    "IMPORTANT: TensorDB does NOT have information_schema or pg_catalog. "
    "To list tables use SHOW TABLES. To describe a table use DESCRIBE <table>. "
    "Table names are plain identifiers — never use schema-qualified names like schema.table. "
    "Output ONLY a single SQL statement, nothing else — no explanation, no markdown. /no_think"
)

SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Part 1: Generic SQL from b-mc2/sql-create-context
# ---------------------------------------------------------------------------
def load_generic_sql(n: int = 10_000, seed: int = 42) -> list[dict]:
    """Download and subsample the sql-create-context dataset."""
    print(f"Loading b-mc2/sql-create-context from HuggingFace (subsample {n})...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    subset = ds.select(indices)

    examples = []
    for row in subset:
        question = row["question"].strip()
        context = row["context"].strip()
        answer = row["answer"].strip()

        # Skip examples with empty fields
        if not question or not answer:
            continue

        # Remove trailing semicolons from answer (our model should output without)
        answer = answer.rstrip(";").strip()

        # Build user message matching the format in llm.rs nl_to_sql()
        if context:
            user_msg = f"Schema:\n{context}\n\nQuestion: {question}"
        else:
            user_msg = f"Question: {question}"

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer},
            ]
        })

    print(f"  Loaded {len(examples)} generic SQL examples")
    return examples


# ---------------------------------------------------------------------------
# Part 2: TensorDB-specific examples
# ---------------------------------------------------------------------------

# Sample schema contexts matching gather_schema_context() format
SCHEMA_USERS = "Tables: users\nSchema for users: id (INTEGER PK), name (TEXT), email (TEXT), balance (REAL), active (BOOLEAN)"
SCHEMA_ORDERS = "Tables: users, orders\nSchema for users: id (INTEGER PK), name (TEXT), email (TEXT), balance (REAL)\nSchema for orders: id (INTEGER PK), user_id (INTEGER), amount (REAL), status (TEXT), created_at (INTEGER)"
SCHEMA_PRODUCTS = "Tables: products\nSchema for products: id (INTEGER PK), name (TEXT), price (REAL), category (TEXT), stock (INTEGER)"
SCHEMA_EVENTS = "Tables: events\nSchema for events: id (INTEGER PK), type (TEXT), payload (JSON), ts (INTEGER)"
SCHEMA_LOGS = "Tables: logs\nSchema for logs: id (INTEGER PK), level (TEXT), message (TEXT), timestamp (INTEGER)"
SCHEMA_METRICS = "Tables: metrics\nSchema for metrics: ts (INTEGER), cpu (REAL), memory (REAL), disk (REAL)"
SCHEMA_EMPLOYEES = "Tables: employees, departments\nSchema for employees: id (INTEGER PK), name (TEXT), dept_id (INTEGER), salary (REAL), hire_date (INTEGER)\nSchema for departments: id (INTEGER PK), name (TEXT), budget (REAL)"
SCHEMA_NONE = ""  # No schema context


def generate_tensordb_examples() -> list[dict]:
    """Generate hand-crafted TensorDB-specific training examples."""
    examples = []

    def add(question: str, sql: str, schema: str = SCHEMA_NONE):
        if schema:
            user_msg = f"Schema:\n{schema}\n\nQuestion: {question}"
        else:
            user_msg = f"Question: {question}"
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": sql},
            ]
        })

    # ----- SHOW TABLES (30 variants) -----
    show_tables_questions = [
        "list all tables",
        "show me all the tables",
        "what tables exist",
        "what tables are in the database",
        "list tables",
        "show tables",
        "what tables do we have",
        "display all tables",
        "which tables are available",
        "list all available tables",
        "get all table names",
        "show database tables",
        "what tables are there",
        "enumerate all tables",
        "can you list the tables",
        "give me a list of tables",
        "show all the tables we have",
        "what tables have been created",
        "list every table",
        "show me the tables",
        "what are the tables",
        "print all tables",
        "get tables",
        "return all table names",
        "how many tables do we have",  # edge case — SHOW TABLES is more helpful
        "tell me the tables",
        "I want to see all tables",
        "which tables exist in the database",
        "show me all database tables",
        "what tables are stored",
    ]
    for q in show_tables_questions:
        add(q, "SHOW TABLES")

    # ----- DESCRIBE (25 variants) -----
    describe_variants = [
        ("describe the users table", "DESCRIBE users"),
        ("what columns does users have", "DESCRIBE users"),
        ("show me the schema of users", "DESCRIBE users"),
        ("what is the structure of the users table", "DESCRIBE users"),
        ("describe users", "DESCRIBE users"),
        ("show columns of users", "DESCRIBE users"),
        ("what fields does users have", "DESCRIBE users"),
        ("tell me about the users table", "DESCRIBE users"),
        ("users table schema", "DESCRIBE users"),
        ("what does the users table look like", "DESCRIBE users"),
        ("describe the orders table", "DESCRIBE orders"),
        ("what columns are in orders", "DESCRIBE orders"),
        ("show the schema for orders", "DESCRIBE orders"),
        ("describe orders", "DESCRIBE orders"),
        ("tell me about the orders table structure", "DESCRIBE orders"),
        ("describe the products table", "DESCRIBE products"),
        ("what is the products schema", "DESCRIBE products"),
        ("show me the products table structure", "DESCRIBE products"),
        ("describe products", "DESCRIBE products"),
        ("what columns does the events table have", "DESCRIBE events"),
        ("describe events", "DESCRIBE events"),
        ("describe the logs table", "DESCRIBE logs"),
        ("what are the columns in logs", "DESCRIBE logs"),
        ("describe employees", "DESCRIBE employees"),
        ("show the employee table schema", "DESCRIBE employees"),
    ]
    for q, sql in describe_variants:
        add(q, sql)

    # ----- Temporal: FOR SYSTEM_TIME (20 variants) -----
    temporal_system = [
        ("show users as of timestamp 1000", "SELECT * FROM users FOR SYSTEM_TIME AS OF 1000", SCHEMA_USERS),
        ("get users at system time 5000", "SELECT * FROM users FOR SYSTEM_TIME AS OF 5000", SCHEMA_USERS),
        ("show the state of users at time 2000", "SELECT * FROM users FOR SYSTEM_TIME AS OF 2000", SCHEMA_USERS),
        ("users history from 1000 to 5000", "SELECT * FROM users FOR SYSTEM_TIME FROM 1000 TO 5000", SCHEMA_USERS),
        ("show users between system time 1000 and 5000", "SELECT * FROM users FOR SYSTEM_TIME BETWEEN 1000 AND 5000", SCHEMA_USERS),
        ("all historical versions of users", "SELECT * FROM users FOR SYSTEM_TIME ALL", SCHEMA_USERS),
        ("show complete history of users table", "SELECT * FROM users FOR SYSTEM_TIME ALL", SCHEMA_USERS),
        ("get orders as they were at timestamp 3000", "SELECT * FROM orders FOR SYSTEM_TIME AS OF 3000", SCHEMA_ORDERS),
        ("time travel to timestamp 1500 for users", "SELECT * FROM users FOR SYSTEM_TIME AS OF 1500", SCHEMA_USERS),
        ("what did the users table look like at time 2500", "SELECT * FROM users FOR SYSTEM_TIME AS OF 2500", SCHEMA_USERS),
        ("show all versions of products between 100 and 900", "SELECT * FROM products FOR SYSTEM_TIME BETWEEN 100 AND 900", SCHEMA_PRODUCTS),
        ("users history from timestamp 0 to 10000", "SELECT * FROM users FOR SYSTEM_TIME FROM 0 TO 10000", SCHEMA_USERS),
        ("get the products table at system time 500", "SELECT * FROM products FOR SYSTEM_TIME AS OF 500", SCHEMA_PRODUCTS),
        ("show orders from time 1000 to 2000", "SELECT * FROM orders FOR SYSTEM_TIME FROM 1000 TO 2000", SCHEMA_ORDERS),
        ("full audit trail for users", "SELECT * FROM users FOR SYSTEM_TIME ALL", SCHEMA_USERS),
        ("show me the users at point in time 7500", "SELECT * FROM users FOR SYSTEM_TIME AS OF 7500", SCHEMA_USERS),
        ("list all changes to orders between 2000 and 8000", "SELECT * FROM orders FOR SYSTEM_TIME BETWEEN 2000 AND 8000", SCHEMA_ORDERS),
        ("retrieve orders system time history from 500 to 1500", "SELECT * FROM orders FOR SYSTEM_TIME FROM 500 TO 1500", SCHEMA_ORDERS),
        ("show users at time 100", "SELECT * FROM users FOR SYSTEM_TIME AS OF 100", SCHEMA_USERS),
        ("get historical data for employees at timestamp 3000", "SELECT * FROM employees FOR SYSTEM_TIME AS OF 3000", SCHEMA_EMPLOYEES),
    ]
    for q, sql, schema in temporal_system:
        add(q, sql, schema)

    # ----- Temporal: FOR APPLICATION_TIME (20 variants) -----
    temporal_app = [
        ("show users valid at application time 1000", "SELECT * FROM users FOR APPLICATION_TIME AS OF 1000", SCHEMA_USERS),
        ("get users with application time as of 5000", "SELECT * FROM users FOR APPLICATION_TIME AS OF 5000", SCHEMA_USERS),
        ("users for application time from 1000 to 5000", "SELECT * FROM users FOR APPLICATION_TIME FROM 1000 TO 5000", SCHEMA_USERS),
        ("show users for application time between 100 and 900", "SELECT * FROM users FOR APPLICATION_TIME BETWEEN 100 AND 900", SCHEMA_USERS),
        ("get orders at application time 3000", "SELECT * FROM orders FOR APPLICATION_TIME AS OF 3000", SCHEMA_ORDERS),
        ("application time view of products at 2000", "SELECT * FROM products FOR APPLICATION_TIME AS OF 2000", SCHEMA_PRODUCTS),
        ("show users valid as of 7500", "SELECT * FROM users FOR APPLICATION_TIME AS OF 7500", SCHEMA_USERS),
        ("get employees at business time 4000", "SELECT * FROM employees FOR APPLICATION_TIME AS OF 4000", SCHEMA_EMPLOYEES),
        ("orders for application time from 0 to 10000", "SELECT * FROM orders FOR APPLICATION_TIME FROM 0 TO 10000", SCHEMA_ORDERS),
        ("show products valid between 500 and 2000", "SELECT * FROM products FOR APPLICATION_TIME BETWEEN 500 AND 2000", SCHEMA_PRODUCTS),
        ("application time history of users from 100 to 500", "SELECT * FROM users FOR APPLICATION_TIME FROM 100 TO 500", SCHEMA_USERS),
        ("show users at valid time 300", "SELECT * FROM users FOR APPLICATION_TIME AS OF 300", SCHEMA_USERS),
        ("get orders valid at time 6000", "SELECT * FROM orders FOR APPLICATION_TIME AS OF 6000", SCHEMA_ORDERS),
        ("users between application time 2000 and 4000", "SELECT * FROM users FOR APPLICATION_TIME BETWEEN 2000 AND 4000", SCHEMA_USERS),
        ("what were the products at application time 800", "SELECT * FROM products FOR APPLICATION_TIME AS OF 800", SCHEMA_PRODUCTS),
        ("show employees valid from 1000 to 3000", "SELECT * FROM employees FOR APPLICATION_TIME FROM 1000 TO 3000", SCHEMA_EMPLOYEES),
        ("get users at business time 9000", "SELECT * FROM users FOR APPLICATION_TIME AS OF 9000", SCHEMA_USERS),
        ("show orders between application time 5000 and 7000", "SELECT * FROM orders FOR APPLICATION_TIME BETWEEN 5000 AND 7000", SCHEMA_ORDERS),
        ("products valid at application time 1500", "SELECT * FROM products FOR APPLICATION_TIME AS OF 1500", SCHEMA_PRODUCTS),
        ("retrieve users for application time as of 250", "SELECT * FROM users FOR APPLICATION_TIME AS OF 250", SCHEMA_USERS),
    ]
    for q, sql, schema in temporal_app:
        add(q, sql, schema)

    # ----- AS OF / VALID AT legacy temporal syntax (10 variants) -----
    temporal_legacy = [
        ("show user with id 1 as of time 1000", "SELECT * FROM users AS OF 1000 WHERE id = 1", SCHEMA_USERS),
        ("count users at time 2000", "SELECT COUNT(*) FROM users AS OF 2000", SCHEMA_USERS),
        ("users valid at 5000", "SELECT * FROM users VALID AT 5000", SCHEMA_USERS),
        ("get products as of 3000", "SELECT * FROM products AS OF 3000", SCHEMA_PRODUCTS),
        ("count orders as of timestamp 1500", "SELECT COUNT(*) FROM orders AS OF 1500", SCHEMA_ORDERS),
        ("users as of 100 where balance > 500", "SELECT * FROM users AS OF 100 WHERE balance > 500", SCHEMA_USERS),
        ("show orders at system time 4000", "SELECT * FROM orders AS OF 4000", SCHEMA_ORDERS),
        ("get employees valid at time 2000", "SELECT * FROM employees VALID AT 2000", SCHEMA_EMPLOYEES),
        ("count products as of 7000", "SELECT COUNT(*) FROM products AS OF 7000", SCHEMA_PRODUCTS),
        ("users valid at 3000 where active = true", "SELECT * FROM users VALID AT 3000 WHERE active = true", SCHEMA_USERS),
    ]
    for q, sql, schema in temporal_legacy:
        add(q, sql, schema)

    # ----- Anti-patterns: information_schema → TensorDB syntax (30 variants) -----
    anti_patterns = [
        ("select from information_schema.tables", "SHOW TABLES"),
        ("query information_schema to get tables", "SHOW TABLES"),
        ("select table_name from information_schema.tables", "SHOW TABLES"),
        ("show me the information schema", "SHOW TABLES"),
        ("query pg_catalog for table list", "SHOW TABLES"),
        ("select from pg_tables", "SHOW TABLES"),
        ("select * from pg_catalog.pg_tables", "SHOW TABLES"),
        ("list tables from information_schema", "SHOW TABLES"),
        ("get table info from system catalog", "SHOW TABLES"),
        ("query the database schema", "SHOW TABLES"),
        ("select column_name from information_schema.columns where table_name = 'users'", "DESCRIBE users"),
        ("get columns of users from information_schema", "DESCRIBE users"),
        ("select * from information_schema.columns where table_name = 'orders'", "DESCRIBE orders"),
        ("query system tables for column info on products", "DESCRIBE products"),
        ("get schema info for users table from pg_catalog", "DESCRIBE users"),
        ("select column_name, data_type from information_schema.columns for users", "DESCRIBE users"),
        ("show columns from pg_catalog for events", "DESCRIBE events"),
        ("query metadata for users table", "DESCRIBE users"),
        ("select from pg_attribute for users", "DESCRIBE users"),
        ("get table definition for orders", "DESCRIBE orders"),
        ("sys.tables query", "SHOW TABLES"),
        ("select from sys.columns for users", "DESCRIBE users"),
        ("sqlite_master tables", "SHOW TABLES"),
        ("select name from sqlite_master where type='table'", "SHOW TABLES"),
        ("show all schemas", "SHOW TABLES"),
        ("list all schemas in the database", "SHOW TABLES"),
        ("query pg_stat_all_tables", "SHOW TABLES"),
        ("what system views are available", "SHOW TABLES"),
        ("select from all_tables", "SHOW TABLES"),
        ("dba_tables query", "SHOW TABLES"),
    ]
    for q, sql in anti_patterns:
        add(q, sql)

    # ----- EXPLAIN / EXPLAIN ANALYZE (15 variants) -----
    explain_variants = [
        ("explain the query select * from users", "EXPLAIN SELECT * FROM users", SCHEMA_USERS),
        ("show query plan for select * from users where balance > 100", "EXPLAIN SELECT * FROM users WHERE balance > 100", SCHEMA_USERS),
        ("analyze select count(*) from orders", "EXPLAIN ANALYZE SELECT COUNT(*) FROM orders", SCHEMA_ORDERS),
        ("explain analyze select * from users where id = 1", "EXPLAIN ANALYZE SELECT * FROM users WHERE id = 1", SCHEMA_USERS),
        ("show execution plan for counting products", "EXPLAIN SELECT COUNT(*) FROM products", SCHEMA_PRODUCTS),
        ("how does the query for active users execute", "EXPLAIN SELECT * FROM users WHERE active = true", SCHEMA_USERS),
        ("run explain analyze on select avg(balance) from users", "EXPLAIN ANALYZE SELECT AVG(balance) FROM users", SCHEMA_USERS),
        ("show query plan for orders join users", "EXPLAIN SELECT * FROM orders JOIN users ON orders.user_id = users.id", SCHEMA_ORDERS),
        ("explain how a full table scan of products works", "EXPLAIN SELECT * FROM products", SCHEMA_PRODUCTS),
        ("profile select * from events where type = 'click'", "EXPLAIN ANALYZE SELECT * FROM events WHERE type = 'click'", SCHEMA_EVENTS),
        ("explain select name from users order by balance desc limit 10", "EXPLAIN SELECT name FROM users ORDER BY balance DESC LIMIT 10", SCHEMA_USERS),
        ("analyze query performance for orders", "EXPLAIN ANALYZE SELECT * FROM orders", SCHEMA_ORDERS),
        ("run explain on select sum(amount) from orders group by user_id", "EXPLAIN SELECT SUM(amount) FROM orders GROUP BY user_id", SCHEMA_ORDERS),
        ("show me the plan for select * from logs where level = 'ERROR'", "EXPLAIN SELECT * FROM logs WHERE level = 'ERROR'", SCHEMA_LOGS),
        ("profile the users count query", "EXPLAIN ANALYZE SELECT COUNT(*) FROM users", SCHEMA_USERS),
    ]
    for q, sql, schema in explain_variants:
        add(q, sql, schema)

    # ----- COPY TO/FROM (10 variants) -----
    copy_variants = [
        ("export users to csv", "COPY users TO 'users.csv' FORMAT CSV", SCHEMA_USERS),
        ("export orders to json file", "COPY orders TO 'orders.json' FORMAT JSON", SCHEMA_ORDERS),
        ("save products to parquet", "COPY products TO 'products.parquet' FORMAT PARQUET", SCHEMA_PRODUCTS),
        ("export users table as csv to users_export.csv", "COPY users TO 'users_export.csv' FORMAT CSV", SCHEMA_USERS),
        ("import data from data.csv into users", "COPY users FROM 'data.csv' FORMAT CSV", SCHEMA_USERS),
        ("load orders from orders.json", "COPY orders FROM 'orders.json' FORMAT JSON", SCHEMA_ORDERS),
        ("import parquet file into products", "COPY products FROM 'products.parquet' FORMAT PARQUET", SCHEMA_PRODUCTS),
        ("export events to ndjson", "COPY events TO 'events.ndjson' FORMAT NDJSON", SCHEMA_EVENTS),
        ("save logs to logs.csv as csv", "COPY logs TO 'logs.csv' FORMAT CSV", SCHEMA_LOGS),
        ("load csv data into employees", "COPY employees FROM 'employees.csv' FORMAT CSV", SCHEMA_EMPLOYEES),
    ]
    for q, sql, schema in copy_variants:
        add(q, sql, schema)

    # ----- CREATE TIMESERIES TABLE (10 variants) -----
    timeseries_variants = [
        ("create a timeseries table for cpu metrics with 1 minute buckets",
         "CREATE TIMESERIES TABLE cpu_metrics (ts INTEGER, cpu REAL) WITH (bucket_size = '1m')", ""),
        ("make a timeseries table called sensor_data with 5 second buckets",
         "CREATE TIMESERIES TABLE sensor_data (ts INTEGER, value REAL) WITH (bucket_size = '5s')", ""),
        ("create timeseries table temperature with 1 hour intervals",
         "CREATE TIMESERIES TABLE temperature (ts INTEGER, temp REAL, location TEXT) WITH (bucket_size = '1h')", ""),
        ("set up a timeseries table for stock prices with 1 minute buckets",
         "CREATE TIMESERIES TABLE stock_prices (ts INTEGER, price REAL, symbol TEXT) WITH (bucket_size = '1m')", ""),
        ("create a time series table for network traffic with 10 second buckets",
         "CREATE TIMESERIES TABLE network_traffic (ts INTEGER, bytes_in REAL, bytes_out REAL) WITH (bucket_size = '10s')", ""),
        ("make a timeseries table called heartbeats with 1s intervals",
         "CREATE TIMESERIES TABLE heartbeats (ts INTEGER, bpm INTEGER) WITH (bucket_size = '1s')", ""),
        ("create timeseries table for IoT sensor readings every 30 seconds",
         "CREATE TIMESERIES TABLE iot_readings (ts INTEGER, sensor_id TEXT, value REAL) WITH (bucket_size = '30s')", ""),
        ("set up a timeseries for request latency with 5 minute buckets",
         "CREATE TIMESERIES TABLE request_latency (ts INTEGER, latency_ms REAL, endpoint TEXT) WITH (bucket_size = '5m')", ""),
        ("create timeseries metrics table with 1h bucket size",
         "CREATE TIMESERIES TABLE metrics (ts INTEGER, cpu REAL, memory REAL, disk REAL) WITH (bucket_size = '1h')", ""),
        ("make a time series table for log counts per minute",
         "CREATE TIMESERIES TABLE log_counts (ts INTEGER, count INTEGER, level TEXT) WITH (bucket_size = '1m')", ""),
    ]
    for q, sql, schema in timeseries_variants:
        add(q, sql, schema)

    # ----- FULLTEXT INDEX (10 variants) -----
    fts_variants = [
        ("create a fulltext index on users name column", "CREATE FULLTEXT INDEX idx_users_name ON users (name)", SCHEMA_USERS),
        ("add fulltext search to the message column of logs", "CREATE FULLTEXT INDEX idx_logs_message ON logs (message)", SCHEMA_LOGS),
        ("create fulltext index on products name and category", "CREATE FULLTEXT INDEX idx_products_search ON products (name, category)", SCHEMA_PRODUCTS),
        ("enable full text search on events payload", "CREATE FULLTEXT INDEX idx_events_payload ON events (payload)", SCHEMA_EVENTS),
        ("add text search index on users email", "CREATE FULLTEXT INDEX idx_users_email ON users (email)", SCHEMA_USERS),
        ("drop the fulltext index idx_users_name on users", "DROP FULLTEXT INDEX idx_users_name ON users", SCHEMA_USERS),
        ("remove fulltext index idx_logs_message from logs", "DROP FULLTEXT INDEX idx_logs_message ON logs", SCHEMA_LOGS),
        ("delete the fulltext search index on products", "DROP FULLTEXT INDEX idx_products_search ON products", SCHEMA_PRODUCTS),
        ("create a full text search index on orders status", "CREATE FULLTEXT INDEX idx_orders_status ON orders (status)", SCHEMA_ORDERS),
        ("add fulltext index for searching employee names", "CREATE FULLTEXT INDEX idx_employees_name ON employees (name)", SCHEMA_EMPLOYEES),
    ]
    for q, sql, schema in fts_variants:
        add(q, sql, schema)

    # ----- Table functions: read_csv, read_json, read_parquet (10 variants) -----
    table_fn_variants = [
        ("query csv file data.csv", "SELECT * FROM read_csv('data.csv')", ""),
        ("read the json file users.json", "SELECT * FROM read_json('users.json')", ""),
        ("load and query parquet file orders.parquet", "SELECT * FROM read_parquet('orders.parquet')", ""),
        ("count rows in the csv file data.csv", "SELECT COUNT(*) FROM read_csv('data.csv')", ""),
        ("show first 10 rows of products.csv", "SELECT * FROM read_csv('products.csv') LIMIT 10", ""),
        ("filter rows from events.json where type is click", "SELECT * FROM read_json('events.json') WHERE type = 'click'", ""),
        ("get average price from products.parquet", "SELECT AVG(price) FROM read_parquet('products.parquet')", ""),
        ("read metrics from metrics.csv and filter cpu > 80", "SELECT * FROM read_csv('metrics.csv') WHERE cpu > 80", ""),
        ("join users table with data from extra.csv", "SELECT * FROM users JOIN read_csv('extra.csv') AS extra ON users.id = extra.id", SCHEMA_USERS),
        ("import and count rows in sales.parquet", "SELECT COUNT(*) FROM read_parquet('sales.parquet')", ""),
    ]
    for q, sql, schema in table_fn_variants:
        add(q, sql, schema)

    # ----- Transactions: BEGIN, COMMIT, ROLLBACK (10 variants) -----
    tx_variants = [
        ("start a transaction", "BEGIN"),
        ("begin a transaction", "BEGIN"),
        ("open a new transaction", "BEGIN"),
        ("commit the current transaction", "COMMIT"),
        ("save the transaction", "COMMIT"),
        ("commit changes", "COMMIT"),
        ("rollback the transaction", "ROLLBACK"),
        ("undo the current transaction", "ROLLBACK"),
        ("abort the transaction", "ROLLBACK"),
        ("cancel all pending changes", "ROLLBACK"),
    ]
    for q, sql in tx_variants:
        add(q, sql)

    # ----- ANALYZE table (10 variants) -----
    analyze_variants = [
        ("analyze the users table", "ANALYZE users", SCHEMA_USERS),
        ("collect statistics on orders", "ANALYZE orders", SCHEMA_ORDERS),
        ("update statistics for products", "ANALYZE products", SCHEMA_PRODUCTS),
        ("analyze events table", "ANALYZE events", SCHEMA_EVENTS),
        ("run analyze on logs", "ANALYZE logs", SCHEMA_LOGS),
        ("gather table statistics for employees", "ANALYZE employees", SCHEMA_EMPLOYEES),
        ("analyze metrics table", "ANALYZE metrics", SCHEMA_METRICS),
        ("compute statistics for users", "ANALYZE users", SCHEMA_USERS),
        ("analyze the orders table for query planning", "ANALYZE orders", SCHEMA_ORDERS),
        ("refresh statistics on products", "ANALYZE products", SCHEMA_PRODUCTS),
    ]
    for q, sql, schema in analyze_variants:
        add(q, sql, schema)

    # ----- ALTER TABLE ADD COLUMN (10 variants) -----
    alter_variants = [
        ("add an age column to users", "ALTER TABLE users ADD COLUMN age INTEGER", SCHEMA_USERS),
        ("add a phone column of type text to users", "ALTER TABLE users ADD COLUMN phone TEXT", SCHEMA_USERS),
        ("add a discount column to products", "ALTER TABLE products ADD COLUMN discount REAL", SCHEMA_PRODUCTS),
        ("add a notes text column to orders", "ALTER TABLE orders ADD COLUMN notes TEXT", SCHEMA_ORDERS),
        ("add a severity column to logs", "ALTER TABLE logs ADD COLUMN severity INTEGER", SCHEMA_LOGS),
        ("add a location text column to employees", "ALTER TABLE employees ADD COLUMN location TEXT", SCHEMA_EMPLOYEES),
        ("add a boolean verified column to users", "ALTER TABLE users ADD COLUMN verified BOOLEAN", SCHEMA_USERS),
        ("add a quantity integer column to orders", "ALTER TABLE orders ADD COLUMN quantity INTEGER", SCHEMA_ORDERS),
        ("add a weight real column to products", "ALTER TABLE products ADD COLUMN weight REAL", SCHEMA_PRODUCTS),
        ("add a source text column to events", "ALTER TABLE events ADD COLUMN source TEXT", SCHEMA_EVENTS),
    ]
    for q, sql, schema in alter_variants:
        add(q, sql, schema)

    # ----- CREATE/DROP TABLE, VIEW, INDEX (15 variants) -----
    ddl_variants = [
        ("create a table called tasks with id, title, and done columns",
         "CREATE TABLE tasks (id INTEGER PRIMARY KEY, title TEXT, done BOOLEAN)", ""),
        ("create a users table with id name and email",
         "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)", ""),
        ("create a table for storing products with id name price and category",
         "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, category TEXT)", ""),
        ("drop the users table", "DROP TABLE users", SCHEMA_USERS),
        ("delete the orders table", "DROP TABLE orders", SCHEMA_ORDERS),
        ("remove the products table", "DROP TABLE products", SCHEMA_PRODUCTS),
        ("create an index on users email", "CREATE INDEX idx_users_email ON users (email)", SCHEMA_USERS),
        ("add an index on orders user_id", "CREATE INDEX idx_orders_user_id ON orders (user_id)", SCHEMA_ORDERS),
        ("create an index on products category", "CREATE INDEX idx_products_category ON products (category)", SCHEMA_PRODUCTS),
        ("drop index idx_users_email on users", "DROP INDEX idx_users_email ON users", SCHEMA_USERS),
        ("create a view of active users",
         "CREATE VIEW active_users AS SELECT * FROM users WHERE active = true", SCHEMA_USERS),
        ("drop the active_users view", "DROP VIEW active_users", SCHEMA_USERS),
        ("create table sessions with id user_id and token",
         "CREATE TABLE sessions (id INTEGER PRIMARY KEY, user_id INTEGER, token TEXT)", ""),
        ("create a table for audit logs",
         "CREATE TABLE audit_logs (id INTEGER PRIMARY KEY, action TEXT, user_id INTEGER, timestamp INTEGER)", ""),
        ("make a new table called categories with id and name",
         "CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT)", ""),
    ]
    for q, sql, schema in ddl_variants:
        add(q, sql, schema)

    # ----- Common SELECT queries with schema context (30 variants) -----
    select_variants = [
        ("how many users are there", "SELECT COUNT(*) FROM users", SCHEMA_USERS),
        ("count all users", "SELECT COUNT(*) FROM users", SCHEMA_USERS),
        ("what is the total number of orders", "SELECT COUNT(*) FROM orders", SCHEMA_ORDERS),
        ("show all users", "SELECT * FROM users", SCHEMA_USERS),
        ("get all products", "SELECT * FROM products", SCHEMA_PRODUCTS),
        ("list all orders", "SELECT * FROM orders", SCHEMA_ORDERS),
        ("find users with balance greater than 1000", "SELECT * FROM users WHERE balance > 1000", SCHEMA_USERS),
        ("show users where active is true", "SELECT * FROM users WHERE active = true", SCHEMA_USERS),
        ("get the average balance of all users", "SELECT AVG(balance) FROM users", SCHEMA_USERS),
        ("what is the total amount of all orders", "SELECT SUM(amount) FROM orders", SCHEMA_ORDERS),
        ("show top 5 users by balance", "SELECT * FROM users ORDER BY balance DESC LIMIT 5", SCHEMA_USERS),
        ("get the user with the highest balance", "SELECT * FROM users ORDER BY balance DESC LIMIT 1", SCHEMA_USERS),
        ("find orders with amount over 500", "SELECT * FROM orders WHERE amount > 500", SCHEMA_ORDERS),
        ("count users grouped by active status", "SELECT active, COUNT(*) FROM users GROUP BY active", SCHEMA_USERS),
        ("total order amount per user", "SELECT user_id, SUM(amount) FROM orders GROUP BY user_id", SCHEMA_ORDERS),
        ("average price by category", "SELECT category, AVG(price) FROM products GROUP BY category", SCHEMA_PRODUCTS),
        ("show users whose name starts with A", "SELECT * FROM users WHERE name LIKE 'A%'", SCHEMA_USERS),
        ("get orders sorted by amount descending", "SELECT * FROM orders ORDER BY amount DESC", SCHEMA_ORDERS),
        ("find products cheaper than 50", "SELECT * FROM products WHERE price < 50", SCHEMA_PRODUCTS),
        ("show the minimum and maximum balance", "SELECT MIN(balance), MAX(balance) FROM users", SCHEMA_USERS),
        ("count orders by status", "SELECT status, COUNT(*) FROM orders GROUP BY status", SCHEMA_ORDERS),
        ("list products in the electronics category", "SELECT * FROM products WHERE category = 'electronics'", SCHEMA_PRODUCTS),
        ("show users with balance between 100 and 500", "SELECT * FROM users WHERE balance BETWEEN 100 AND 500", SCHEMA_USERS),
        ("get distinct categories from products", "SELECT DISTINCT category FROM products", SCHEMA_PRODUCTS),
        ("count products per category having more than 5", "SELECT category, COUNT(*) FROM products GROUP BY category HAVING COUNT(*) > 5", SCHEMA_PRODUCTS),
        ("show the first 10 orders", "SELECT * FROM orders LIMIT 10", SCHEMA_ORDERS),
        ("find users named Alice", "SELECT * FROM users WHERE name = 'Alice'", SCHEMA_USERS),
        ("get orders where status is pending", "SELECT * FROM orders WHERE status = 'pending'", SCHEMA_ORDERS),
        ("sum of salaries by department", "SELECT dept_id, SUM(salary) FROM employees GROUP BY dept_id", SCHEMA_EMPLOYEES),
        ("find employees earning more than 100000", "SELECT * FROM employees WHERE salary > 100000", SCHEMA_EMPLOYEES),
    ]
    for q, sql, schema in select_variants:
        add(q, sql, schema)

    # ----- JOINs (10 variants) -----
    join_variants = [
        ("show orders with user names", "SELECT orders.id, users.name, orders.amount FROM orders JOIN users ON orders.user_id = users.id", SCHEMA_ORDERS),
        ("list all orders with their user's email", "SELECT orders.id, users.email, orders.amount FROM orders JOIN users ON orders.user_id = users.id", SCHEMA_ORDERS),
        ("join employees with departments", "SELECT employees.name, departments.name FROM employees JOIN departments ON employees.dept_id = departments.id", SCHEMA_EMPLOYEES),
        ("show employees and their department names", "SELECT employees.name, departments.name FROM employees JOIN departments ON employees.dept_id = departments.id", SCHEMA_EMPLOYEES),
        ("get users who have orders", "SELECT DISTINCT users.name FROM users JOIN orders ON users.id = orders.user_id", SCHEMA_ORDERS),
        ("count orders per user with user name", "SELECT users.name, COUNT(*) FROM orders JOIN users ON orders.user_id = users.id GROUP BY users.name", SCHEMA_ORDERS),
        ("total order amount per user name", "SELECT users.name, SUM(orders.amount) FROM orders JOIN users ON orders.user_id = users.id GROUP BY users.name", SCHEMA_ORDERS),
        ("left join users with orders to show users without orders too", "SELECT users.name, orders.amount FROM users LEFT JOIN orders ON users.id = orders.user_id", SCHEMA_ORDERS),
        ("show department budgets with employee counts", "SELECT departments.name, departments.budget, COUNT(employees.id) FROM departments LEFT JOIN employees ON departments.id = employees.dept_id GROUP BY departments.name, departments.budget", SCHEMA_EMPLOYEES),
        ("cross join users and products", "SELECT users.name, products.name FROM users CROSS JOIN products", SCHEMA_USERS),
    ]
    for q, sql, schema in join_variants:
        add(q, sql, schema)

    # ----- Window functions (10 variants) -----
    window_variants = [
        ("rank users by balance", "SELECT name, balance, RANK() OVER (ORDER BY balance DESC) FROM users", SCHEMA_USERS),
        ("number the rows of users ordered by name", "SELECT name, ROW_NUMBER() OVER (ORDER BY name) FROM users", SCHEMA_USERS),
        ("dense rank products by price", "SELECT name, price, DENSE_RANK() OVER (ORDER BY price DESC) FROM products", SCHEMA_PRODUCTS),
        ("show each order with the previous order amount", "SELECT id, amount, LAG(amount) OVER (ORDER BY id) FROM orders", SCHEMA_ORDERS),
        ("show each order with the next order amount", "SELECT id, amount, LEAD(amount) OVER (ORDER BY id) FROM orders", SCHEMA_ORDERS),
        ("rank employees by salary within each department", "SELECT name, salary, RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) FROM employees", SCHEMA_EMPLOYEES),
        ("row number for orders partitioned by status", "SELECT id, status, ROW_NUMBER() OVER (PARTITION BY status ORDER BY amount DESC) FROM orders", SCHEMA_ORDERS),
        ("dense rank employees by salary", "SELECT name, salary, DENSE_RANK() OVER (ORDER BY salary DESC) FROM employees", SCHEMA_EMPLOYEES),
        ("show user balances with running rank", "SELECT name, balance, RANK() OVER (ORDER BY balance) FROM users", SCHEMA_USERS),
        ("show product rank within each category", "SELECT name, category, price, RANK() OVER (PARTITION BY category ORDER BY price DESC) FROM products", SCHEMA_PRODUCTS),
    ]
    for q, sql, schema in window_variants:
        add(q, sql, schema)

    # ----- CTEs (10 variants) -----
    cte_variants = [
        ("using a CTE, find users with above average balance",
         "WITH avg_bal AS (SELECT AVG(balance) AS avg_balance FROM users) SELECT * FROM users WHERE balance > (SELECT avg_balance FROM avg_bal)", SCHEMA_USERS),
        ("with a CTE count orders per user then show top 5",
         "WITH order_counts AS (SELECT user_id, COUNT(*) AS cnt FROM orders GROUP BY user_id) SELECT * FROM order_counts ORDER BY cnt DESC LIMIT 5", SCHEMA_ORDERS),
        ("use a common table expression to find the most expensive product per category",
         "WITH ranked AS (SELECT name, category, price, RANK() OVER (PARTITION BY category ORDER BY price DESC) AS rnk FROM products) SELECT * FROM ranked WHERE rnk = 1", SCHEMA_PRODUCTS),
        ("with CTE get department total salaries",
         "WITH dept_salaries AS (SELECT dept_id, SUM(salary) AS total FROM employees GROUP BY dept_id) SELECT * FROM dept_salaries ORDER BY total DESC", SCHEMA_EMPLOYEES),
        ("CTE to find users who have spent more than 1000 total",
         "WITH user_totals AS (SELECT user_id, SUM(amount) AS total FROM orders GROUP BY user_id) SELECT users.name, user_totals.total FROM user_totals JOIN users ON user_totals.user_id = users.id WHERE user_totals.total > 1000", SCHEMA_ORDERS),
    ]
    for q, sql, schema in cte_variants:
        add(q, sql, schema)

    # ----- INSERT / UPDATE / DELETE (15 variants) -----
    dml_variants = [
        ("insert a user named Alice with balance 100", "INSERT INTO users (name, balance) VALUES ('Alice', 100)", SCHEMA_USERS),
        ("add a new product called Widget priced at 29.99 in tools category", "INSERT INTO products (name, price, category) VALUES ('Widget', 29.99, 'tools')", SCHEMA_PRODUCTS),
        ("insert an order for user 1 with amount 50", "INSERT INTO orders (user_id, amount, status) VALUES (1, 50, 'pending')", SCHEMA_ORDERS),
        ("update user 1 balance to 500", "UPDATE users SET balance = 500 WHERE id = 1", SCHEMA_USERS),
        ("set all users active to false", "UPDATE users SET active = false", SCHEMA_USERS),
        ("update order status to shipped where id is 5", "UPDATE orders SET status = 'shipped' WHERE id = 5", SCHEMA_ORDERS),
        ("increase all product prices by 10%", "UPDATE products SET price = price * 1.10", SCHEMA_PRODUCTS),
        ("delete user with id 3", "DELETE FROM users WHERE id = 3", SCHEMA_USERS),
        ("remove all orders with status cancelled", "DELETE FROM orders WHERE status = 'cancelled'", SCHEMA_ORDERS),
        ("delete products with stock 0", "DELETE FROM products WHERE stock = 0", SCHEMA_PRODUCTS),
        ("insert employee John in department 2 with salary 75000", "INSERT INTO employees (name, dept_id, salary) VALUES ('John', 2, 75000)", SCHEMA_EMPLOYEES),
        ("update employee salary to 80000 where name is Alice", "UPDATE employees SET salary = 80000 WHERE name = 'Alice'", SCHEMA_EMPLOYEES),
        ("delete logs older than timestamp 1000", "DELETE FROM logs WHERE timestamp < 1000", SCHEMA_LOGS),
        ("insert a log entry with level ERROR", "INSERT INTO logs (level, message) VALUES ('ERROR', 'Something went wrong')", SCHEMA_LOGS),
        ("update products set category to misc where category is null", "UPDATE products SET category = 'misc' WHERE category IS NULL", SCHEMA_PRODUCTS),
    ]
    for q, sql, schema in dml_variants:
        add(q, sql, schema)

    # ----- UNION / INTERSECT / EXCEPT (5 variants) -----
    setop_variants = [
        ("combine users and employees names", "SELECT name FROM users UNION SELECT name FROM employees", SCHEMA_USERS),
        ("all names from users and employees including duplicates", "SELECT name FROM users UNION ALL SELECT name FROM employees", SCHEMA_USERS),
        ("products that are in both electronics and sale categories", "SELECT name FROM products WHERE category = 'electronics' INTERSECT SELECT name FROM products WHERE price < 20", SCHEMA_PRODUCTS),
        ("users who have no orders", "SELECT id FROM users EXCEPT SELECT DISTINCT user_id FROM orders", SCHEMA_ORDERS),
        ("combine error and warning log messages", "SELECT message FROM logs WHERE level = 'ERROR' UNION SELECT message FROM logs WHERE level = 'WARNING'", SCHEMA_LOGS),
    ]
    for q, sql, schema in setop_variants:
        add(q, sql, schema)

    # ----- Subqueries (5 variants) -----
    subquery_variants = [
        ("users with balance above average", "SELECT * FROM users WHERE balance > (SELECT AVG(balance) FROM users)", SCHEMA_USERS),
        ("products more expensive than the average", "SELECT * FROM products WHERE price > (SELECT AVG(price) FROM products)", SCHEMA_PRODUCTS),
        ("orders from the top spending user", "SELECT * FROM orders WHERE user_id = (SELECT user_id FROM orders GROUP BY user_id ORDER BY SUM(amount) DESC LIMIT 1)", SCHEMA_ORDERS),
        ("employees earning more than department average", "SELECT * FROM (SELECT * FROM employees WHERE salary > 50000) AS high_earners", SCHEMA_EMPLOYEES),
        ("users who placed orders over 500", "SELECT * FROM users WHERE id IN (SELECT DISTINCT user_id FROM orders WHERE amount > 500)", SCHEMA_ORDERS),
    ]
    for q, sql, schema in subquery_variants:
        add(q, sql, schema)

    # ----- Parameter placeholders $1, $2 (10 variants) -----
    param_variants = [
        ("find user by id parameter", "SELECT * FROM users WHERE id = $1", SCHEMA_USERS),
        ("get orders for a specific user id", "SELECT * FROM orders WHERE user_id = $1", SCHEMA_ORDERS),
        ("find products in a given category under a price limit", "SELECT * FROM products WHERE category = $1 AND price < $2", SCHEMA_PRODUCTS),
        ("update user balance by id", "UPDATE users SET balance = $2 WHERE id = $1", SCHEMA_USERS),
        ("delete order by id", "DELETE FROM orders WHERE id = $1", SCHEMA_ORDERS),
        ("insert a user with parameterized name and balance", "INSERT INTO users (name, balance) VALUES ($1, $2)", SCHEMA_USERS),
        ("get logs between two timestamps", "SELECT * FROM logs WHERE timestamp BETWEEN $1 AND $2", SCHEMA_LOGS),
        ("find employees in a department with minimum salary", "SELECT * FROM employees WHERE dept_id = $1 AND salary >= $2", SCHEMA_EMPLOYEES),
        ("update product price by id", "UPDATE products SET price = $2 WHERE id = $1", SCHEMA_PRODUCTS),
        ("select events by type parameter", "SELECT * FROM events WHERE type = $1", SCHEMA_EVENTS),
    ]
    for q, sql, schema in param_variants:
        add(q, sql, schema)

    print(f"  Generated {len(examples)} TensorDB-specific examples")
    return examples


# ---------------------------------------------------------------------------
# Main: combine, shuffle, split, save
# ---------------------------------------------------------------------------
def main():
    random.seed(42)

    generic = load_generic_sql(n=10_000, seed=42)
    tensordb = generate_tensordb_examples()

    all_examples = generic + tensordb
    random.shuffle(all_examples)

    total = len(all_examples)
    split_idx = int(total * 0.95)
    train = all_examples[:split_idx]
    eval_ = all_examples[split_idx:]

    train_path = SCRIPT_DIR / "training_data.jsonl"
    eval_path = SCRIPT_DIR / "eval_data.jsonl"

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(eval_path, "w") as f:
        for ex in eval_:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTotal examples: {total}")
    print(f"  Train: {len(train)} → {train_path}")
    print(f"  Eval:  {len(eval_)} → {eval_path}")


if __name__ == "__main__":
    main()
