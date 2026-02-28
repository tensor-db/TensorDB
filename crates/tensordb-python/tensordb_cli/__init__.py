"""TensorDB Interactive Shell — pip install tensordb && tensordb"""
import sys
import os
import json
import readline
import atexit

def main():
    from tensordb import PyDatabase

    path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--path" and i < len(sys.argv) - 1:
            path = sys.argv[i + 1]
            break
        elif not arg.startswith("-"):
            path = arg
            break

    if path is None:
        path = os.path.join(os.getcwd(), "tensordb_data")

    os.makedirs(path, exist_ok=True)
    db = PyDatabase.open(path)

    # readline history
    history_file = os.path.expanduser("~/.tensordb_history")
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, history_file)

    print("TensorDB Interactive Shell")
    print(f"Database: {path}")
    print("Type SQL statements ending with ';' or .help for commands.\n")

    buf = []
    while True:
        try:
            prompt = "tensordb> " if not buf else "     ...> "
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = line.strip()

        # Natural language query: # how many users?
        if not buf and stripped.startswith("#"):
            question = stripped[1:].strip()
            if not question:
                continue
            try:
                response = db.ask(question)
                sql = response["sql"]
                print(f"\n  Generated SQL: {sql}")
                confirm = input("  Execute? [Y/n] ").strip().lower()
                if confirm in ("", "y", "yes"):
                    result = response["result"]
                    if isinstance(result, list):
                        for row in result:
                            if isinstance(row, dict):
                                print(json.dumps(row))
                            else:
                                print(row)
                    elif isinstance(result, dict):
                        msg = result.get("message", "")
                        rows = result.get("rows", 0)
                        print(f"{msg} ({rows} row(s))")
                    else:
                        print(result)
                else:
                    print("  Skipped.")
            except AttributeError:
                print("  Error: NL→SQL not available (model not found or llm feature disabled)")
            except Exception as e:
                print(f"  Error: {e}")
            continue

        if not buf and stripped.startswith("."):
            if stripped == ".help":
                print("  .help       Show this message")
                print("  .tables     List all tables")
                print("  .quit       Exit the shell")
                print("  # <question> Ask in natural language (AI translates to SQL)")
                print("  SQL;        Execute SQL (end with semicolon)")
            elif stripped == ".tables":
                try:
                    result = db.sql("SHOW TABLES")
                    if isinstance(result, list):
                        for row in result:
                            print(row)
                    else:
                        print(result)
                except Exception as e:
                    print(f"Error: {e}")
            elif stripped in (".quit", ".exit"):
                break
            else:
                print(f"Unknown command: {stripped}")
            continue

        buf.append(line)
        full = " ".join(buf).strip()

        if full.endswith(";"):
            query = full[:-1].strip()
            if query:
                try:
                    result = db.sql(query)
                    if isinstance(result, list):
                        for row in result:
                            if isinstance(row, dict):
                                print(json.dumps(row))
                            else:
                                print(row)
                    elif isinstance(result, dict):
                        msg = result.get("message", "")
                        rows = result.get("rows", 0)
                        print(f"{msg} ({rows} row(s))")
                    else:
                        print(result)
                except Exception as e:
                    print(f"Error: {e}")
            buf.clear()


if __name__ == "__main__":
    main()
