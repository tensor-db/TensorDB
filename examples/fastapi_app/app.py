"""FastAPI example application using TensorDB.

Run:
    pip install tensordb fastapi uvicorn
    uvicorn app:app --reload

Endpoints:
    POST /items       - Create an item
    GET  /items/{id}  - Get item by ID
    GET  /items       - List items (SQL query)
    GET  /search      - Full-text search
"""

import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensordb

app = FastAPI(title="TensorDB FastAPI Example")

# Open database (creates if not exists)
db = tensordb.PyDatabase.open("./fastapi_data", shard_count=2)

# Initialize schema
db.sql("CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, name TEXT, price REAL, category TEXT);")


class Item(BaseModel):
    id: str
    name: str
    price: float
    category: str = "general"


@app.post("/items")
def create_item(item: Item):
    """Insert a new item."""
    try:
        result = db.sql(
            f"INSERT INTO items (id, name, price, category) "
            f"VALUES ('{item.id}', '{item.name}', {item.price}, '{item.category}');"
        )
        return {"status": "created", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/items/{item_id}")
def get_item(item_id: str):
    """Get an item by ID."""
    result = db.sql(f"SELECT * FROM items WHERE id = '{item_id}';")
    rows = json.loads(result) if result else []
    if not rows:
        raise HTTPException(status_code=404, detail="Item not found")
    return rows[0] if isinstance(rows, list) else rows


@app.get("/items")
def list_items(category: str = None, limit: int = 100):
    """List items, optionally filtered by category."""
    if category:
        query = f"SELECT * FROM items WHERE category = '{category}' LIMIT {limit};"
    else:
        query = f"SELECT * FROM items LIMIT {limit};"
    result = db.sql(query)
    return json.loads(result) if result else []


@app.get("/search")
def search_items(q: str, limit: int = 10):
    """Search items by name (requires FTS index)."""
    try:
        result = db.sql(
            f"SELECT * FROM items WHERE name LIKE '%{q}%' LIMIT {limit};"
        )
        return json.loads(result) if result else []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def database_stats():
    """Get database statistics."""
    result = db.sql("SELECT COUNT(*) as count FROM items;")
    return json.loads(result) if result else {"count": 0}
