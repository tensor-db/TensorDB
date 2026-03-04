//! Row-Level Security (RLS) policies stored under `__meta/policy/` prefix.

use serde::{Deserialize, Serialize};

use crate::engine::db::Database;
use crate::error::{sql_exec_err, Result};

const POLICY_PREFIX: &str = "__meta/policy/";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyOperation {
    Select,
    Insert,
    Update,
    Delete,
    All,
}

impl PolicyOperation {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_uppercase().as_str() {
            "SELECT" => Ok(Self::Select),
            "INSERT" => Ok(Self::Insert),
            "UPDATE" => Ok(Self::Update),
            "DELETE" => Ok(Self::Delete),
            "ALL" => Ok(Self::All),
            _ => Err(sql_exec_err(format!(
                "invalid policy operation: {s}, expected SELECT/INSERT/UPDATE/DELETE/ALL"
            ))),
        }
    }

    pub fn matches(&self, op: PolicyOperation) -> bool {
        *self == PolicyOperation::All || *self == op
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowPolicy {
    pub name: String,
    pub table: String,
    pub operation: PolicyOperation,
    pub using_expr: String,
    pub roles: Vec<String>,
    pub created_at: u64,
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub struct PolicyManager;

impl PolicyManager {
    pub fn create_policy(db: &Database, policy: &RowPolicy) -> Result<()> {
        let key = format!("{POLICY_PREFIX}{}/{}", policy.table, policy.name).into_bytes();
        if db.get(&key, None, None)?.is_some() {
            return Err(sql_exec_err(format!(
                "policy {} already exists on table {}",
                policy.name, policy.table
            )));
        }
        let value = serde_json::to_vec(policy)?;
        db.put(&key, value, 0, u64::MAX, None)?;
        Ok(())
    }

    pub fn drop_policy(db: &Database, table: &str, name: &str) -> Result<()> {
        let key = format!("{POLICY_PREFIX}{table}/{name}").into_bytes();
        if db.get(&key, None, None)?.is_none() {
            return Err(sql_exec_err(format!(
                "policy {name} does not exist on table {table}"
            )));
        }
        db.put(&key, Vec::new(), 0, u64::MAX, None)?;
        Ok(())
    }

    pub fn get_policies_for_table(db: &Database, table: &str) -> Result<Vec<RowPolicy>> {
        let prefix = format!("{POLICY_PREFIX}{table}/").into_bytes();
        let rows = db.scan_prefix(&prefix, None, None, None)?;
        let mut policies = Vec::new();
        for row in rows {
            if row.doc.is_empty() {
                continue;
            }
            if let Ok(policy) = serde_json::from_slice::<RowPolicy>(&row.doc) {
                policies.push(policy);
            }
        }
        Ok(policies)
    }

    pub fn get_applicable(
        db: &Database,
        table: &str,
        op: PolicyOperation,
        roles: &[String],
    ) -> Result<Vec<RowPolicy>> {
        let all = Self::get_policies_for_table(db, table)?;
        Ok(all
            .into_iter()
            .filter(|p| {
                // Operation must match
                if !p.operation.matches(op) {
                    return false;
                }
                // If policy has roles, current user must have at least one
                if p.roles.is_empty() {
                    return true;
                }
                roles.iter().any(|r| p.roles.contains(r))
            })
            .collect())
    }

    pub fn new_policy(
        name: String,
        table: String,
        operation: PolicyOperation,
        using_expr: String,
        roles: Vec<String>,
    ) -> RowPolicy {
        RowPolicy {
            name,
            table,
            operation,
            using_expr,
            roles,
            created_at: current_timestamp_ms(),
        }
    }
}
