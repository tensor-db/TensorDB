use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use spectradb_core::sql::exec::SqlResult;
use spectradb_core::{Config, Database};

#[pyclass]
struct PyDatabase {
    db: Database,
}

#[pymethods]
impl PyDatabase {
    #[staticmethod]
    #[pyo3(signature = (path, shard_count=None))]
    fn open(path: String, shard_count: Option<usize>) -> PyResult<Self> {
        let mut config = Config::default();
        if let Some(sc) = shard_count {
            config.shard_count = sc;
        }
        let db =
            Database::open(&path, config).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { db })
    }

    fn put(&self, key: &[u8], doc: Vec<u8>, valid_from: u64, valid_to: u64) -> PyResult<u64> {
        self.db
            .put(key, doc, valid_from, valid_to, None)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(signature = (key, as_of=None, valid_at=None))]
    fn get(&self, key: &[u8], as_of: Option<u64>, valid_at: Option<u64>) -> PyResult<Option<Vec<u8>>> {
        self.db
            .get(key, as_of, valid_at)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn sql(&self, py: Python<'_>, query: &str) -> PyResult<PyObject> {
        let result = self
            .db
            .sql(query)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        match result {
            SqlResult::Rows(rows) => {
                let py_rows: Vec<PyObject> = rows
                    .into_iter()
                    .map(|row| {
                        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&row) {
                            json_to_py(py, &val)
                        } else {
                            let s = String::from_utf8_lossy(&row).to_string();
                            s.to_object(py)
                        }
                    })
                    .collect();
                Ok(pyo3::types::PyList::new_bound(py, &py_rows).to_object(py))
            }
            SqlResult::Affected {
                rows,
                commit_ts,
                message,
            } => {
                let dict = pyo3::types::PyDict::new_bound(py);
                dict.set_item("rows", rows)?;
                dict.set_item("commit_ts", commit_ts)?;
                dict.set_item("message", message)?;
                Ok(dict.to_object(py))
            }
            SqlResult::Explain(text) => Ok(text.to_object(py)),
        }
    }
}

fn json_to_py(py: Python<'_>, val: &serde_json::Value) -> PyObject {
    match val {
        serde_json::Value::Null => py.None(),
        serde_json::Value::Bool(b) => b.to_object(py),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_object(py)
            } else {
                n.as_f64().unwrap_or(0.0).to_object(py)
            }
        }
        serde_json::Value::String(s) => s.to_object(py),
        serde_json::Value::Array(arr) => {
            let items: Vec<PyObject> = arr.iter().map(|v| json_to_py(py, v)).collect();
            pyo3::types::PyList::new_bound(py, &items).to_object(py)
        }
        serde_json::Value::Object(map) => {
            let dict = pyo3::types::PyDict::new_bound(py);
            for (k, v) in map {
                let _ = dict.set_item(k, json_to_py(py, v));
            }
            dict.to_object(py)
        }
    }
}

#[pymodule]
fn spectradb_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatabase>()?;
    Ok(())
}
