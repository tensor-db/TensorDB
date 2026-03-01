/**
 * Express.js example application using TensorDB.
 *
 * Run:
 *   npm install tensordb express
 *   node index.js
 *
 * Endpoints:
 *   POST /items       - Create an item
 *   GET  /items/:id   - Get item by ID
 *   GET  /items       - List items
 *   GET  /stats       - Database statistics
 */

const express = require('express');
const { JsDatabase } = require('tensordb');

const app = express();
app.use(express.json());

// Open database
const db = JsDatabase.open('./express_data', 2);

// Initialize schema
db.sql("CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, name TEXT, price REAL, category TEXT);");

// Create item
app.post('/items', (req, res) => {
  try {
    const { id, name, price, category } = req.body;
    const result = db.sql(
      `INSERT INTO items (id, name, price, category) VALUES ('${id}', '${name}', ${price}, '${category || 'general'}');`
    );
    res.json({ status: 'created', result });
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// Get item by ID
app.get('/items/:id', (req, res) => {
  try {
    const result = db.sql(`SELECT * FROM items WHERE id = '${req.params.id}';`);
    const rows = JSON.parse(result || '[]');
    if (rows.length === 0) {
      return res.status(404).json({ error: 'Item not found' });
    }
    res.json(rows[0]);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// List items
app.get('/items', (req, res) => {
  try {
    const limit = req.query.limit || 100;
    const category = req.query.category;
    let query;
    if (category) {
      query = `SELECT * FROM items WHERE category = '${category}' LIMIT ${limit};`;
    } else {
      query = `SELECT * FROM items LIMIT ${limit};`;
    }
    const result = db.sql(query);
    res.json(JSON.parse(result || '[]'));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Database stats
app.get('/stats', (req, res) => {
  try {
    const result = db.sql("SELECT COUNT(*) as count FROM items;");
    res.json(JSON.parse(result || '{"count": 0}'));
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`TensorDB Express app listening on port ${PORT}`);
});
