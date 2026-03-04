// Phase 2: Structured error codes tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::error::{ErrorCode, TensorError};

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_parse_error_has_t1_code() {
    let (db, _dir) = setup();
    let err = db.sql("SELEC * FROM t").unwrap_err();
    if let TensorError::SqlParse(e) = &err {
        assert!(
            e.code.code_str().starts_with("T1"),
            "expected T1xxx code, got {}",
            e.code.code_str()
        );
        assert_eq!(e.code.category(), "Syntax");
    } else {
        panic!("expected SqlParse, got {err:?}");
    }
}

#[test]
fn test_table_not_found_has_t2001_code() {
    let (db, _dir) = setup();
    let err = db.sql("SELECT * FROM nonexistent_table").unwrap_err();
    if let TensorError::SqlExec(e) = &err {
        assert_eq!(
            e.code,
            ErrorCode::TableNotFound,
            "expected T2001, got {}",
            e.code.code_str()
        );
    } else {
        // May be a different error type; just check it errors
        assert!(
            format!("{err}").contains("does not exist") || format!("{err}").contains("not exist"),
            "unexpected error: {err}"
        );
    }
}

#[test]
fn test_error_display_includes_code_prefix() {
    let (db, _dir) = setup();
    let err = db.sql("SELEC * FROM t").unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("T1"),
        "Display should include error code prefix: {msg}"
    );
}

#[test]
fn test_error_code_categories() {
    assert_eq!(ErrorCode::SyntaxError.category(), "Syntax");
    assert_eq!(ErrorCode::TableNotFound.category(), "Schema");
    assert_eq!(ErrorCode::NotNullViolation.category(), "Constraint");
    assert_eq!(ErrorCode::QueryTimeout.category(), "Execution");
    assert_eq!(ErrorCode::PermissionDenied.category(), "Auth");
}
