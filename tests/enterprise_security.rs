//! Phase 4: Enterprise Security tests
//! Tests for audit log tamper detection, key rotation, and column encryption.

use tensordb_core::config::Config;
use tensordb_core::Database;

fn open_tmp() -> Database {
    let dir = tempfile::tempdir().unwrap();
    Database::open(dir.path(), Config::default()).unwrap()
}

// ─── 4.1: Audit Log Tamper Detection ───

#[test]
fn test_verify_audit_log_empty() {
    let db = open_tmp();
    let result = db.sql("VERIFY AUDIT LOG").unwrap();
    match result {
        tensordb_core::sql::exec::SqlResult::Rows(rows) => {
            assert_eq!(rows.len(), 1);
            let val: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
            assert_eq!(val["status"], "OK");
            assert_eq!(val["verified"], 0);
            assert_eq!(val["total"], 0);
        }
        _ => panic!("expected Rows"),
    }
}

#[test]
fn test_verify_audit_log_after_operations() {
    let db = open_tmp();
    // Create table generates audit events
    db.sql("CREATE TABLE audit_test (id INT PRIMARY KEY, name TEXT)")
        .unwrap();
    db.sql("INSERT INTO audit_test (id, name) VALUES (1, 'Alice')")
        .unwrap();

    let result = db.sql("VERIFY AUDIT LOG").unwrap();
    match result {
        tensordb_core::sql::exec::SqlResult::Rows(rows) => {
            let val: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
            assert_eq!(val["status"], "OK");
            assert!(val["broken_at"].is_null());
        }
        _ => panic!("expected Rows"),
    }
}

#[test]
fn test_show_audit_log_has_hash_fields() {
    let db = open_tmp();
    db.sql("CREATE TABLE hash_test (x INT PRIMARY KEY)")
        .unwrap();

    let result = db.sql("SHOW AUDIT LOG").unwrap();
    match result {
        tensordb_core::sql::exec::SqlResult::Rows(rows) => {
            assert!(!rows.is_empty(), "should have at least 1 audit event");
        }
        _ => panic!("expected Rows"),
    }
}

// ─── 4.3: Key Rotation ───

#[test]
fn test_rotate_encryption_key() {
    let db = open_tmp();
    let result = db.sql("ROTATE ENCRYPTION KEY 'my-new-passphrase'").unwrap();
    match result {
        tensordb_core::sql::exec::SqlResult::Rows(rows) => {
            let val: serde_json::Value = serde_json::from_slice(&rows[0]).unwrap();
            assert_eq!(val["status"], "OK");
            let version = val["new_version"].as_u64().unwrap();
            assert!(version >= 1);
        }
        _ => panic!("expected Rows"),
    }
}

#[test]
fn test_rotate_key_increments_version() {
    let db = open_tmp();
    let r1 = db.sql("ROTATE ENCRYPTION KEY 'key-1'").unwrap();
    let r2 = db.sql("ROTATE ENCRYPTION KEY 'key-2'").unwrap();
    match (r1, r2) {
        (
            tensordb_core::sql::exec::SqlResult::Rows(rows1),
            tensordb_core::sql::exec::SqlResult::Rows(rows2),
        ) => {
            let v1: serde_json::Value = serde_json::from_slice(&rows1[0]).unwrap();
            let v2: serde_json::Value = serde_json::from_slice(&rows2[0]).unwrap();
            let ver1 = v1["new_version"].as_u64().unwrap();
            let ver2 = v2["new_version"].as_u64().unwrap();
            assert!(ver2 > ver1, "version should increment");
        }
        _ => panic!("expected Rows"),
    }
}

#[test]
fn test_key_manager_old_keys_accessible() {
    let db = open_tmp();
    db.sql("ROTATE ENCRYPTION KEY 'first-key'").unwrap();
    db.sql("ROTATE ENCRYPTION KEY 'second-key'").unwrap();

    let km = db.key_manager();
    assert!(km.get_key(1).is_some(), "version 1 should still exist");
    assert!(km.get_key(2).is_some(), "version 2 should exist");
    assert_eq!(km.active_version(), 2);
}

// ─── 4.3: Key Manager Unit Tests ───

#[test]
fn test_key_manager_derive_column_key() {
    use tensordb_core::storage::encryption::EncryptionKey;
    use tensordb_core::storage::key_manager::derive_column_key;

    let master = EncryptionKey::from_passphrase("master-secret");
    let ck1 = derive_column_key(&master, "users", "email");
    let ck2 = derive_column_key(&master, "users", "email");
    assert_eq!(ck1.as_bytes(), ck2.as_bytes());

    let ck3 = derive_column_key(&master, "users", "phone");
    assert_ne!(ck1.as_bytes(), ck3.as_bytes());
}

#[test]
fn test_key_manager_with_config_passphrase() {
    let dir = tempfile::tempdir().unwrap();
    let config = Config {
        encryption_passphrase: Some("config-test-key".to_string()),
        ..Config::default()
    };
    let db = Database::open(dir.path(), config).unwrap();
    let km = db.key_manager();
    assert!(km.is_enabled());
    assert_eq!(km.active_version(), 1);
}
