// Integration tests for v0.26 Schema Evolution
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::facet::schema_evolution::*;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_migration_lifecycle() {
    let (db, _dir) = setup();

    // Register migrations
    let m1 = Migration {
        version: 1,
        name: "create_products".to_string(),
        description: "Initial products table".to_string(),
        up_sql: vec![
            "CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price FLOAT);".to_string(),
        ],
        down_sql: vec!["DROP TABLE products;".to_string()],
        applied_at: None,
        checksum: compute_checksum(&[
            "CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price FLOAT);".to_string(),
        ]),
    };

    let m2 = Migration {
        version: 2,
        name: "create_categories".to_string(),
        description: "Categories table".to_string(),
        up_sql: vec!["CREATE TABLE categories (id INT PRIMARY KEY, name TEXT);".to_string()],
        down_sql: vec!["DROP TABLE categories;".to_string()],
        applied_at: None,
        checksum: String::new(),
    };

    MigrationManager::register(&db, &m1).unwrap();
    MigrationManager::register(&db, &m2).unwrap();

    // Check pending
    let pending = MigrationManager::pending(&db).unwrap();
    assert_eq!(pending.len(), 2);

    // Apply all
    let results = MigrationManager::apply_all(&db).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].version, 1);
    assert_eq!(results[1].version, 2);

    // Verify current version
    assert_eq!(MigrationManager::current_version(&db).unwrap(), Some(2));
    assert!(MigrationManager::pending(&db).unwrap().is_empty());

    // Rollback version 2
    let rb = MigrationManager::rollback(&db, 2).unwrap();
    assert_eq!(rb.statements_executed, 1);
    assert_eq!(MigrationManager::current_version(&db).unwrap(), Some(1));
}

#[test]
fn test_schema_version_tracking() {
    let (db, _dir) = setup();

    // Record version 1
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

    // Record version 2 — add email, change id type
    let v2 = SchemaVersion {
        table_name: "users".to_string(),
        version: 2,
        columns: vec![
            ColumnDefinition {
                name: "id".to_string(),
                data_type: "BIGINT".to_string(),
                nullable: false,
                default_value: None,
            },
            ColumnDefinition {
                name: "name".to_string(),
                data_type: "TEXT".to_string(),
                nullable: false,
                default_value: None,
            },
            ColumnDefinition {
                name: "email".to_string(),
                data_type: "TEXT".to_string(),
                nullable: true,
                default_value: None,
            },
        ],
        created_at: 2000,
        migration_id: Some(2),
    };
    SchemaRegistry::record_version(&db, &v2).unwrap();

    // Check latest
    let latest = SchemaRegistry::latest(&db, "users").unwrap().unwrap();
    assert_eq!(latest.version, 2);
    assert_eq!(latest.columns.len(), 3);

    // Check history
    let history = SchemaRegistry::history(&db, "users").unwrap();
    assert_eq!(history.len(), 2);

    // Check diff
    let diff = SchemaRegistry::diff(&v1, &v2);
    assert_eq!(diff.added_columns.len(), 1);
    assert_eq!(diff.added_columns[0].name, "email");
    assert_eq!(diff.modified_columns.len(), 1);
    assert_eq!(diff.modified_columns[0].name, "id");
    assert!(diff.removed_columns.is_empty());
}
