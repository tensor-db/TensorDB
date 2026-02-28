use crate::engine::db::Database;
use crate::error::{Result, TensorError};

/// Key prefix for schema migration history.
const MIGRATION_PREFIX: &str = "__schema/migration/";
/// Key prefix for schema version registry.
const SCHEMA_VERSION_PREFIX: &str = "__schema/version/";

/// A schema migration record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Migration {
    pub version: u64,
    pub name: String,
    pub description: String,
    pub up_sql: Vec<String>,
    pub down_sql: Vec<String>,
    pub applied_at: Option<u64>,
    pub checksum: String,
}

/// Schema version entry for a specific table.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SchemaVersion {
    pub table_name: String,
    pub version: u64,
    pub columns: Vec<ColumnDefinition>,
    pub created_at: u64,
    pub migration_id: Option<u64>,
}

/// Column definition within a schema version.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColumnDefinition {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
    pub default_value: Option<String>,
}

/// Result of applying a migration.
#[derive(Debug, Clone)]
pub struct MigrationResult {
    pub version: u64,
    pub name: String,
    pub statements_executed: usize,
    pub duration_us: u64,
}

/// Schema migration manager.
pub struct MigrationManager;

impl MigrationManager {
    /// Register a migration (without applying it).
    pub fn register(db: &Database, migration: &Migration) -> Result<()> {
        let key = format!("{}{:020}", MIGRATION_PREFIX, migration.version);
        if db.get(key.as_bytes(), None, None)?.is_some() {
            return Err(TensorError::SqlExec(format!(
                "migration version {} already exists",
                migration.version
            )));
        }
        let value = serde_json::to_vec(migration)
            .map_err(|e| TensorError::SqlExec(format!("failed to serialize migration: {e}")))?;
        db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        Ok(())
    }

    /// Apply a migration — execute its up_sql statements.
    pub fn apply(db: &Database, version: u64) -> Result<MigrationResult> {
        let key = format!("{}{:020}", MIGRATION_PREFIX, version);
        let bytes = db.get(key.as_bytes(), None, None)?.ok_or_else(|| {
            TensorError::SqlExec(format!("migration version {version} not found"))
        })?;
        let mut migration: Migration = serde_json::from_slice(&bytes)
            .map_err(|e| TensorError::SqlExec(format!("failed to parse migration: {e}")))?;

        if migration.applied_at.is_some() {
            return Err(TensorError::SqlExec(format!(
                "migration {} already applied",
                version
            )));
        }

        let start = std::time::Instant::now();
        let mut executed = 0;

        for sql in &migration.up_sql {
            db.sql(sql)?;
            executed += 1;
        }

        let duration_us = start.elapsed().as_micros() as u64;

        // Mark as applied
        migration.applied_at = Some(current_timestamp_ms());
        let value = serde_json::to_vec(&migration)
            .map_err(|e| TensorError::SqlExec(format!("failed to serialize migration: {e}")))?;
        db.put(key.as_bytes(), value, 0, u64::MAX, None)?;

        Ok(MigrationResult {
            version,
            name: migration.name.clone(),
            statements_executed: executed,
            duration_us,
        })
    }

    /// Rollback a migration — execute its down_sql statements.
    pub fn rollback(db: &Database, version: u64) -> Result<MigrationResult> {
        let key = format!("{}{:020}", MIGRATION_PREFIX, version);
        let bytes = db.get(key.as_bytes(), None, None)?.ok_or_else(|| {
            TensorError::SqlExec(format!("migration version {version} not found"))
        })?;
        let mut migration: Migration = serde_json::from_slice(&bytes)
            .map_err(|e| TensorError::SqlExec(format!("failed to parse migration: {e}")))?;

        if migration.applied_at.is_none() {
            return Err(TensorError::SqlExec(format!(
                "migration {} was not applied",
                version
            )));
        }

        let start = std::time::Instant::now();
        let mut executed = 0;

        for sql in &migration.down_sql {
            db.sql(sql)?;
            executed += 1;
        }

        let duration_us = start.elapsed().as_micros() as u64;

        // Mark as un-applied
        migration.applied_at = None;
        let value = serde_json::to_vec(&migration)
            .map_err(|e| TensorError::SqlExec(format!("failed to serialize migration: {e}")))?;
        db.put(key.as_bytes(), value, 0, u64::MAX, None)?;

        Ok(MigrationResult {
            version,
            name: migration.name.clone(),
            statements_executed: executed,
            duration_us,
        })
    }

    /// Get the current migration version (highest applied).
    pub fn current_version(db: &Database) -> Result<Option<u64>> {
        let rows = db.scan_prefix(MIGRATION_PREFIX.as_bytes(), None, None, None)?;
        let mut max_applied: Option<u64> = None;
        for row in rows {
            if let Ok(m) = serde_json::from_slice::<Migration>(&row.doc) {
                if m.applied_at.is_some() {
                    max_applied = Some(max_applied.map_or(m.version, |v: u64| v.max(m.version)));
                }
            }
        }
        Ok(max_applied)
    }

    /// List all migrations (applied and pending).
    pub fn list(db: &Database) -> Result<Vec<Migration>> {
        let rows = db.scan_prefix(MIGRATION_PREFIX.as_bytes(), None, None, None)?;
        let mut migrations = Vec::new();
        for row in rows {
            if let Ok(m) = serde_json::from_slice::<Migration>(&row.doc) {
                migrations.push(m);
            }
        }
        migrations.sort_by_key(|m| m.version);
        Ok(migrations)
    }

    /// Get pending (unapplied) migrations.
    pub fn pending(db: &Database) -> Result<Vec<Migration>> {
        let all = Self::list(db)?;
        Ok(all.into_iter().filter(|m| m.applied_at.is_none()).collect())
    }

    /// Apply all pending migrations in order.
    pub fn apply_all(db: &Database) -> Result<Vec<MigrationResult>> {
        let pending = Self::pending(db)?;
        let mut results = Vec::new();
        for m in pending {
            results.push(Self::apply(db, m.version)?);
        }
        Ok(results)
    }
}

/// Schema version registry — track table schema changes over time.
pub struct SchemaRegistry;

impl SchemaRegistry {
    /// Record a new schema version for a table.
    pub fn record_version(db: &Database, schema: &SchemaVersion) -> Result<()> {
        let key = format!(
            "{}{}/{:020}",
            SCHEMA_VERSION_PREFIX, schema.table_name, schema.version
        );
        let value = serde_json::to_vec(schema)
            .map_err(|e| TensorError::SqlExec(format!("failed to serialize schema: {e}")))?;
        db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        Ok(())
    }

    /// Get the latest schema version for a table.
    pub fn latest(db: &Database, table_name: &str) -> Result<Option<SchemaVersion>> {
        let prefix = format!("{}{}/", SCHEMA_VERSION_PREFIX, table_name);
        let rows = db.scan_prefix(prefix.as_bytes(), None, None, None)?;
        let mut latest: Option<SchemaVersion> = None;
        for row in rows {
            if let Ok(sv) = serde_json::from_slice::<SchemaVersion>(&row.doc) {
                if latest.as_ref().is_none_or(|l| sv.version > l.version) {
                    latest = Some(sv);
                }
            }
        }
        Ok(latest)
    }

    /// Get all schema versions for a table.
    pub fn history(db: &Database, table_name: &str) -> Result<Vec<SchemaVersion>> {
        let prefix = format!("{}{}/", SCHEMA_VERSION_PREFIX, table_name);
        let rows = db.scan_prefix(prefix.as_bytes(), None, None, None)?;
        let mut versions = Vec::new();
        for row in rows {
            if let Ok(sv) = serde_json::from_slice::<SchemaVersion>(&row.doc) {
                versions.push(sv);
            }
        }
        versions.sort_by_key(|v| v.version);
        Ok(versions)
    }

    /// Compute a diff between two schema versions.
    pub fn diff(old: &SchemaVersion, new: &SchemaVersion) -> SchemaDiff {
        let old_cols: std::collections::HashMap<&str, &ColumnDefinition> =
            old.columns.iter().map(|c| (c.name.as_str(), c)).collect();
        let new_cols: std::collections::HashMap<&str, &ColumnDefinition> =
            new.columns.iter().map(|c| (c.name.as_str(), c)).collect();

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        for (name, col) in &new_cols {
            if let Some(old_col) = old_cols.get(name) {
                if old_col.data_type != col.data_type
                    || old_col.nullable != col.nullable
                    || old_col.default_value != col.default_value
                {
                    modified.push(ColumnChange {
                        name: name.to_string(),
                        old_type: Some(old_col.data_type.clone()),
                        new_type: Some(col.data_type.clone()),
                        old_nullable: Some(old_col.nullable),
                        new_nullable: Some(col.nullable),
                    });
                }
            } else {
                added.push((*col).clone());
            }
        }

        for name in old_cols.keys() {
            if !new_cols.contains_key(name) {
                removed.push(name.to_string());
            }
        }

        SchemaDiff {
            from_version: old.version,
            to_version: new.version,
            added_columns: added,
            removed_columns: removed,
            modified_columns: modified,
        }
    }
}

/// Diff between two schema versions.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SchemaDiff {
    pub from_version: u64,
    pub to_version: u64,
    pub added_columns: Vec<ColumnDefinition>,
    pub removed_columns: Vec<String>,
    pub modified_columns: Vec<ColumnChange>,
}

/// A change to a column.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ColumnChange {
    pub name: String,
    pub old_type: Option<String>,
    pub new_type: Option<String>,
    pub old_nullable: Option<bool>,
    pub new_nullable: Option<bool>,
}

/// Compute a simple checksum for SQL statements.
pub fn compute_checksum(statements: &[String]) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    for s in statements {
        s.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn setup() -> (Database, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path(), Config::default()).unwrap();
        (db, dir)
    }

    #[test]
    fn test_register_migration() {
        let (db, _dir) = setup();
        let m = Migration {
            version: 1,
            name: "create_users".to_string(),
            description: "Create users table".to_string(),
            up_sql: vec!["CREATE TABLE users (id INT PRIMARY KEY, name TEXT);".to_string()],
            down_sql: vec!["DROP TABLE users;".to_string()],
            applied_at: None,
            checksum: compute_checksum(&[
                "CREATE TABLE users (id INT PRIMARY KEY, name TEXT);".to_string()
            ]),
        };
        MigrationManager::register(&db, &m).unwrap();

        let migrations = MigrationManager::list(&db).unwrap();
        assert_eq!(migrations.len(), 1);
        assert_eq!(migrations[0].name, "create_users");
    }

    #[test]
    fn test_duplicate_migration_rejected() {
        let (db, _dir) = setup();
        let m = Migration {
            version: 1,
            name: "init".to_string(),
            description: String::new(),
            up_sql: vec![],
            down_sql: vec![],
            applied_at: None,
            checksum: String::new(),
        };
        MigrationManager::register(&db, &m).unwrap();
        assert!(MigrationManager::register(&db, &m).is_err());
    }

    #[test]
    fn test_apply_migration() {
        let (db, _dir) = setup();
        let m = Migration {
            version: 1,
            name: "create_orders".to_string(),
            description: "Create orders table".to_string(),
            up_sql: vec!["CREATE TABLE orders (id INT PRIMARY KEY, total FLOAT);".to_string()],
            down_sql: vec!["DROP TABLE orders;".to_string()],
            applied_at: None,
            checksum: String::new(),
        };
        MigrationManager::register(&db, &m).unwrap();

        let result = MigrationManager::apply(&db, 1).unwrap();
        assert_eq!(result.version, 1);
        assert_eq!(result.statements_executed, 1);

        // Verify migration was applied
        assert_eq!(MigrationManager::current_version(&db).unwrap(), Some(1));
        assert!(MigrationManager::pending(&db).unwrap().is_empty());
    }

    #[test]
    fn test_apply_already_applied() {
        let (db, _dir) = setup();
        let m = Migration {
            version: 1,
            name: "init".to_string(),
            description: String::new(),
            up_sql: vec!["CREATE TABLE t1 (id INT PRIMARY KEY);".to_string()],
            down_sql: vec![],
            applied_at: None,
            checksum: String::new(),
        };
        MigrationManager::register(&db, &m).unwrap();
        MigrationManager::apply(&db, 1).unwrap();
        assert!(MigrationManager::apply(&db, 1).is_err());
    }

    #[test]
    fn test_current_version() {
        let (db, _dir) = setup();
        assert!(MigrationManager::current_version(&db).unwrap().is_none());

        let m1 = Migration {
            version: 1,
            name: "v1".to_string(),
            description: String::new(),
            up_sql: vec!["CREATE TABLE t1 (id INT PRIMARY KEY);".to_string()],
            down_sql: vec![],
            applied_at: None,
            checksum: String::new(),
        };
        let m2 = Migration {
            version: 2,
            name: "v2".to_string(),
            description: String::new(),
            up_sql: vec!["CREATE TABLE t2 (id INT PRIMARY KEY);".to_string()],
            down_sql: vec![],
            applied_at: None,
            checksum: String::new(),
        };
        MigrationManager::register(&db, &m1).unwrap();
        MigrationManager::register(&db, &m2).unwrap();
        MigrationManager::apply(&db, 1).unwrap();

        assert_eq!(MigrationManager::current_version(&db).unwrap(), Some(1));
    }

    #[test]
    fn test_pending_migrations() {
        let (db, _dir) = setup();
        let m1 = Migration {
            version: 1,
            name: "v1".to_string(),
            description: String::new(),
            up_sql: vec!["CREATE TABLE t1 (id INT PRIMARY KEY);".to_string()],
            down_sql: vec![],
            applied_at: None,
            checksum: String::new(),
        };
        let m2 = Migration {
            version: 2,
            name: "v2".to_string(),
            description: String::new(),
            up_sql: vec!["CREATE TABLE t2 (id INT PRIMARY KEY);".to_string()],
            down_sql: vec![],
            applied_at: None,
            checksum: String::new(),
        };
        MigrationManager::register(&db, &m1).unwrap();
        MigrationManager::register(&db, &m2).unwrap();
        MigrationManager::apply(&db, 1).unwrap();

        let pending = MigrationManager::pending(&db).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].version, 2);
    }

    #[test]
    fn test_rollback_migration() {
        let (db, _dir) = setup();
        let m = Migration {
            version: 1,
            name: "create_items".to_string(),
            description: String::new(),
            up_sql: vec!["CREATE TABLE items (id INT PRIMARY KEY, name TEXT);".to_string()],
            down_sql: vec!["DROP TABLE items;".to_string()],
            applied_at: None,
            checksum: String::new(),
        };
        MigrationManager::register(&db, &m).unwrap();
        MigrationManager::apply(&db, 1).unwrap();

        let result = MigrationManager::rollback(&db, 1).unwrap();
        assert_eq!(result.statements_executed, 1);

        // Table should be gone
        assert!(MigrationManager::current_version(&db).unwrap().is_none());
    }

    #[test]
    fn test_schema_version_registry() {
        let (db, _dir) = setup();
        let v1 = SchemaVersion {
            table_name: "users".to_string(),
            version: 1,
            columns: vec![
                ColumnDefinition {
                    name: "id".to_string(),
                    data_type: "INT".to_string(),
                    nullable: false,
                    default_value: None,
                },
                ColumnDefinition {
                    name: "name".to_string(),
                    data_type: "TEXT".to_string(),
                    nullable: false,
                    default_value: None,
                },
            ],
            created_at: 1000,
            migration_id: Some(1),
        };
        SchemaRegistry::record_version(&db, &v1).unwrap();

        let latest = SchemaRegistry::latest(&db, "users").unwrap().unwrap();
        assert_eq!(latest.version, 1);
        assert_eq!(latest.columns.len(), 2);
    }

    #[test]
    fn test_schema_diff() {
        let v1 = SchemaVersion {
            table_name: "users".to_string(),
            version: 1,
            columns: vec![
                ColumnDefinition {
                    name: "id".to_string(),
                    data_type: "INT".to_string(),
                    nullable: false,
                    default_value: None,
                },
                ColumnDefinition {
                    name: "name".to_string(),
                    data_type: "TEXT".to_string(),
                    nullable: false,
                    default_value: None,
                },
            ],
            created_at: 1000,
            migration_id: None,
        };
        let v2 = SchemaVersion {
            table_name: "users".to_string(),
            version: 2,
            columns: vec![
                ColumnDefinition {
                    name: "id".to_string(),
                    data_type: "BIGINT".to_string(), // Changed type
                    nullable: false,
                    default_value: None,
                },
                ColumnDefinition {
                    name: "email".to_string(), // New column
                    data_type: "TEXT".to_string(),
                    nullable: true,
                    default_value: None,
                },
                // "name" removed
            ],
            created_at: 2000,
            migration_id: None,
        };

        let diff = SchemaRegistry::diff(&v1, &v2);
        assert_eq!(diff.added_columns.len(), 1);
        assert_eq!(diff.added_columns[0].name, "email");
        assert_eq!(diff.removed_columns, vec!["name"]);
        assert_eq!(diff.modified_columns.len(), 1);
        assert_eq!(diff.modified_columns[0].name, "id");
    }

    #[test]
    fn test_schema_history() {
        let (db, _dir) = setup();
        for v in 1..=3 {
            let sv = SchemaVersion {
                table_name: "products".to_string(),
                version: v,
                columns: vec![ColumnDefinition {
                    name: "id".to_string(),
                    data_type: "INT".to_string(),
                    nullable: false,
                    default_value: None,
                }],
                created_at: v * 1000,
                migration_id: None,
            };
            SchemaRegistry::record_version(&db, &sv).unwrap();
        }

        let history = SchemaRegistry::history(&db, "products").unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].version, 1);
        assert_eq!(history[2].version, 3);
    }

    #[test]
    fn test_compute_checksum() {
        let stmts = vec!["CREATE TABLE t1 (id INT PRIMARY KEY);".to_string()];
        let cs1 = compute_checksum(&stmts);
        let cs2 = compute_checksum(&stmts);
        assert_eq!(cs1, cs2);

        let stmts2 = vec!["CREATE TABLE t2 (id INT PRIMARY KEY);".to_string()];
        let cs3 = compute_checksum(&stmts2);
        assert_ne!(cs1, cs3);
    }

    #[test]
    fn test_apply_all_migrations() {
        let (db, _dir) = setup();
        for i in 1..=3 {
            let m = Migration {
                version: i,
                name: format!("v{i}"),
                description: String::new(),
                up_sql: vec![format!("CREATE TABLE t{i} (id INT PRIMARY KEY);")],
                down_sql: vec![format!("DROP TABLE t{i};")],
                applied_at: None,
                checksum: String::new(),
            };
            MigrationManager::register(&db, &m).unwrap();
        }

        let results = MigrationManager::apply_all(&db).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(MigrationManager::current_version(&db).unwrap(), Some(3));
        assert!(MigrationManager::pending(&db).unwrap().is_empty());
    }
}
