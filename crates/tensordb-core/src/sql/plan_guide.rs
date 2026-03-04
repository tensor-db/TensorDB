//! Plan guide storage and matching for plan stability.
//!
//! Stored under `__meta/plan_guide/{name}`.

use serde::{Deserialize, Serialize};

use crate::engine::db::Database;
use crate::error::{sql_exec_err, Result};

const PLAN_GUIDE_PREFIX: &str = "__meta/plan_guide/";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanGuide {
    pub name: String,
    pub sql_pattern: String,
    pub hints: String,
    pub created_at: u64,
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub struct PlanGuideManager;

impl PlanGuideManager {
    pub fn create(db: &Database, name: &str, sql_pattern: &str, hints: &str) -> Result<()> {
        let key = format!("{PLAN_GUIDE_PREFIX}{name}").into_bytes();
        if db.get(&key, None, None)?.is_some() {
            return Err(sql_exec_err(format!("plan guide '{name}' already exists")));
        }
        let guide = PlanGuide {
            name: name.to_string(),
            sql_pattern: sql_pattern.to_string(),
            hints: hints.to_string(),
            created_at: current_timestamp_ms(),
        };
        let value = serde_json::to_vec(&guide)?;
        db.put(&key, value, 0, u64::MAX, None)?;
        Ok(())
    }

    pub fn drop(db: &Database, name: &str) -> Result<()> {
        let key = format!("{PLAN_GUIDE_PREFIX}{name}").into_bytes();
        if db.get(&key, None, None)?.is_none() {
            return Err(sql_exec_err(format!("plan guide '{name}' does not exist")));
        }
        db.put(&key, Vec::new(), 0, u64::MAX, None)?;
        Ok(())
    }

    pub fn list(db: &Database) -> Result<Vec<PlanGuide>> {
        let prefix = PLAN_GUIDE_PREFIX.as_bytes();
        let rows = db.scan_prefix(prefix, None, None, None)?;
        let mut guides = Vec::new();
        for row in rows {
            if row.doc.is_empty() {
                continue;
            }
            if let Ok(guide) = serde_json::from_slice::<PlanGuide>(&row.doc) {
                guides.push(guide);
            }
        }
        Ok(guides)
    }

    /// Find a plan guide matching the given SQL (normalized comparison).
    pub fn find_matching(db: &Database, sql: &str) -> Result<Option<PlanGuide>> {
        let normalized = normalize_sql(sql);
        let guides = Self::list(db)?;
        for guide in guides {
            if normalize_sql(&guide.sql_pattern) == normalized {
                return Ok(Some(guide));
            }
        }
        Ok(None)
    }
}

/// Normalize SQL for matching: lowercase, collapse whitespace.
fn normalize_sql(sql: &str) -> String {
    sql.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}
