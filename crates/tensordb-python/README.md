# tensordb

Python bindings for [TensorDB](https://github.com/tensor-db/tensorDB) — an AI-native, bitemporal ledger database with MVCC, SQL, and LSM storage.

## Install

```bash
pip install tensordb
```

## Quick Start

```python
from tensordb import PyDatabase
import json

# Open (or create) a database
db = PyDatabase.open("/tmp/mydb")

# Insert a document
doc = json.dumps({"name": "Alice", "age": 30}).encode()
db.put(b"user:1", doc, 0, 2**63 - 1)

# Point read
data = db.get(b"user:1")
print(json.loads(data))  # {'name': 'Alice', 'age': 30}

# SQL query
db.sql("CREATE TABLE users (id INT, name TEXT, age INT)")
db.sql("INSERT INTO users VALUES (1, 'Alice', 30)")
rows = db.sql("SELECT * FROM users WHERE age > 25")
print(rows)
```

## API

### `PyDatabase.open(path, shard_count=None)`
Open or create a database at `path`.

### `db.put(key, doc, valid_from, valid_to)`
Insert a document. Returns the commit timestamp.

### `db.get(key, as_of=None, valid_at=None)`
Read a document by key. Supports bitemporal queries.

### `db.sql(query)`
Execute a SQL statement. Returns rows or affected-row metadata.

## License

PolyForm Noncommercial 1.0.0
