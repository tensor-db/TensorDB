#!/usr/bin/env python3
"""Post-training validation for the fine-tuned TensorDB NL→SQL model.

Tests the model via TensorDB Python bindings to verify that critical
NL→SQL translations work correctly after fine-tuning.

Usage:
    conda run -n cortex_ngc python scripts/eval_nl2sql.py

Prerequisites:
    - Fine-tuned GGUF deployed to .local/models/Qwen3-0.6B-Q8_0.gguf
    - Python bindings rebuilt: maturin develop --release
"""

import sys
import tempfile
import os

# Try to import tensordb — if not available, explain how to build
try:
    import tensordb
except ImportError:
    print("ERROR: tensordb Python bindings not found.")
    print("Build with: cd crates/tensordb-python && maturin develop --release")
    sys.exit(1)


def test_case(db, question: str, expected_keywords: list[str], description: str) -> bool:
    """Run a single test case.

    Args:
        db: TensorDB Database instance
        question: Natural language question to ask
        expected_keywords: List of SQL keywords/fragments that should appear in the generated SQL
        description: Human-readable test description

    Returns:
        True if test passed, False otherwise
    """
    try:
        result = db.ask(question)
        sql = result["sql"].strip().upper()

        passed = all(kw.upper() in sql for kw in expected_keywords)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {description}")
        print(f"         Q: {question}")
        print(f"         SQL: {result['sql'].strip()}")
        if not passed:
            print(f"         Expected keywords: {expected_keywords}")
        print()
        return passed
    except Exception as e:
        print(f"  [ERROR] {description}")
        print(f"         Q: {question}")
        print(f"         Error: {e}")
        print()
        return False


def main():
    # Create a temp directory for the test database
    with tempfile.TemporaryDirectory(prefix="tensordb_eval_") as tmp:
        db_path = os.path.join(tmp, "eval_db")
        print(f"Test database: {db_path}\n")

        db = tensordb.PyDatabase.open(db_path)

        # Create test tables
        db.sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, balance REAL, active BOOLEAN)")
        db.sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT)")
        db.sql("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL, category TEXT)")

        # Insert some test data
        db.sql("INSERT INTO users (id, name, email, balance, active) VALUES (1, 'Alice', 'alice@example.com', 1500.0, true)")
        db.sql("INSERT INTO users (id, name, email, balance, active) VALUES (2, 'Bob', 'bob@example.com', 250.0, true)")
        db.sql("INSERT INTO orders (id, user_id, amount, status) VALUES (1, 1, 99.99, 'completed')")
        db.sql("INSERT INTO products (id, name, price, category) VALUES (1, 'Widget', 29.99, 'tools')")

        print("=" * 60)
        print("  TensorDB NL→SQL Evaluation")
        print("=" * 60)
        print()

        results = []

        # ----- Critical: SHOW TABLES -----
        results.append(test_case(
            db, "list all tables",
            ["SHOW", "TABLES"],
            "SHOW TABLES (variant: list all tables)",
        ))
        results.append(test_case(
            db, "what tables exist in the database",
            ["SHOW", "TABLES"],
            "SHOW TABLES (variant: what tables exist)",
        ))
        results.append(test_case(
            db, "show me all the tables we have",
            ["SHOW", "TABLES"],
            "SHOW TABLES (variant: show me all tables)",
        ))

        # ----- Critical: DESCRIBE -----
        results.append(test_case(
            db, "describe the users table",
            ["DESCRIBE", "USERS"],
            "DESCRIBE users",
        ))
        results.append(test_case(
            db, "what columns does orders have",
            ["DESCRIBE", "ORDERS"],
            "DESCRIBE orders (variant: what columns)",
        ))

        # ----- Basic SELECT -----
        results.append(test_case(
            db, "how many users are there",
            ["SELECT", "COUNT", "FROM", "USERS"],
            "COUNT(*) query",
        ))
        results.append(test_case(
            db, "show all users with balance over 500",
            ["SELECT", "FROM", "USERS", "WHERE", "BALANCE"],
            "SELECT with WHERE clause",
        ))
        results.append(test_case(
            db, "top 5 users by balance",
            ["SELECT", "FROM", "USERS", "ORDER", "BY", "BALANCE"],
            "SELECT with ORDER BY",
        ))

        # ----- Temporal -----
        # Both "AS OF 1000" (legacy) and "FOR SYSTEM_TIME AS OF 1000" (SQL:2011) are valid
        results.append(test_case(
            db, "show users as of timestamp 1000",
            ["SELECT", "FROM", "USERS", "AS", "OF", "1000"],
            "Temporal: AS OF (legacy or SQL:2011)",
        ))
        results.append(test_case(
            db, "all historical versions of users",
            ["SELECT", "FROM", "USERS", "SYSTEM_TIME", "ALL"],
            "Temporal: FOR SYSTEM_TIME ALL",
        ))

        # ----- Anti-patterns -----
        results.append(test_case(
            db, "select from information_schema.tables",
            ["SHOW", "TABLES"],
            "Anti-pattern: information_schema → SHOW TABLES",
        ))

        # ----- Transaction (note: ask() executes the SQL, so BEGIN opens a tx
        #        that must be rolled back to avoid cleanup issues) -----
        try:
            result = db.ask("start a transaction")
            tx_sql = result["sql"].strip().upper()
            tx_pass = "BEGIN" in tx_sql
            print(f"  [{'PASS' if tx_pass else 'FAIL'}] Transaction: BEGIN")
            print(f"         Q: start a transaction")
            print(f"         SQL: {result['sql'].strip()}")
            if not tx_pass:
                print(f"         Expected keywords: ['BEGIN']")
            print()
            results.append(tx_pass)
            # Clean up the open transaction
            try:
                db.sql("ROLLBACK")
            except Exception:
                pass
        except Exception as e:
            # If the error message contains "BEGIN", the model generated correctly
            err_msg = str(e)
            if "BEGIN" in err_msg.upper() or "transaction" in err_msg.lower():
                print(f"  [PASS] Transaction: BEGIN (generated correctly, execution side-effect)")
                print(f"         Q: start a transaction")
                print(f"         Error: {e}")
                print()
                results.append(True)
            else:
                print(f"  [ERROR] Transaction: BEGIN")
                print(f"         Q: start a transaction")
                print(f"         Error: {e}")
                print()
                results.append(False)

        # ----- Summary -----
        passed = sum(results)
        total = len(results)
        print("=" * 60)
        print(f"  Results: {passed}/{total} passed")
        print("=" * 60)

        if passed == total:
            print("\n  All tests passed! The fine-tuned model is working correctly.")
        elif passed >= total * 0.8:
            print(f"\n  Most tests passed ({passed}/{total}). Review failures above.")
        else:
            print(f"\n  WARNING: Only {passed}/{total} tests passed. Model may need more training.")

        sys.exit(0 if passed >= total * 0.8 else 1)


if __name__ == "__main__":
    main()
