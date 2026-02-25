use napi::bindgen_prelude::*;
use napi_derive::napi;

use spectradb_core::sql::exec::SqlResult;
use spectradb_core::{Config, Database};

#[napi]
pub struct JsDatabase {
    db: Database,
}

#[napi]
impl JsDatabase {
    #[napi(factory)]
    pub fn open(path: String, shard_count: Option<u32>) -> Result<Self> {
        let mut config = Config::default();
        if let Some(sc) = shard_count {
            config.shard_count = sc as usize;
        }
        let db =
            Database::open(&path, config).map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(Self { db })
    }

    #[napi]
    pub fn put(
        &self,
        key: Buffer,
        doc: Buffer,
        valid_from: i64,
        valid_to: i64,
    ) -> Result<i64> {
        let ts = self
            .db
            .put(
                key.as_ref(),
                doc.to_vec(),
                valid_from as u64,
                valid_to as u64,
                None,
            )
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(ts as i64)
    }

    #[napi]
    pub fn get(
        &self,
        key: Buffer,
        as_of: Option<i64>,
        valid_at: Option<i64>,
    ) -> Result<Option<Buffer>> {
        let result = self
            .db
            .get(
                key.as_ref(),
                as_of.map(|v| v as u64),
                valid_at.map(|v| v as u64),
            )
            .map_err(|e| Error::from_reason(e.to_string()))?;
        Ok(result.map(|v| v.into()))
    }

    /// Execute SQL and return JSON string result.
    #[napi]
    pub fn sql(&self, query: String) -> Result<String> {
        let result = self
            .db
            .sql(&query)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        match result {
            SqlResult::Rows(rows) => {
                let parsed: Vec<serde_json::Value> = rows
                    .into_iter()
                    .map(|row| {
                        serde_json::from_slice(&row).unwrap_or_else(|_| {
                            serde_json::Value::String(String::from_utf8_lossy(&row).to_string())
                        })
                    })
                    .collect();
                let out = serde_json::json!({ "rows": parsed });
                serde_json::to_string(&out).map_err(|e| Error::from_reason(e.to_string()))
            }
            SqlResult::Affected {
                rows,
                commit_ts,
                message,
            } => {
                let out = serde_json::json!({
                    "rows": rows,
                    "commit_ts": commit_ts,
                    "message": message,
                });
                serde_json::to_string(&out).map_err(|e| Error::from_reason(e.to_string()))
            }
            SqlResult::Explain(text) => {
                let out = serde_json::json!({ "explain": text });
                serde_json::to_string(&out).map_err(|e| Error::from_reason(e.to_string()))
            }
        }
    }
}
