// Phase 4: Compaction scheduling tests
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;
use tensordb_core::sql::exec::SqlResult;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_set_compaction_window() {
    let (db, _dir) = setup();
    let result = db.sql("SET COMPACTION_WINDOW = '02:00-06:00'").unwrap();
    if let SqlResult::Affected { message, .. } = result {
        assert!(message.contains("compaction_window"), "got: {message}");
    }
    // Verify the window was set on the database
    let window = db.compaction_window();
    assert_eq!(window, Some((2, 6)));
}

#[test]
fn test_compaction_window_format_validation() {
    let (db, _dir) = setup();
    let err = db.sql("SET COMPACTION_WINDOW = 'invalid'").unwrap_err();
    assert!(
        format!("{err}").contains("format"),
        "Should report format error: {err}"
    );
}
