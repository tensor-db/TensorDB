// Integration tests for v0.23 Authentication & RBAC
use tensordb_core::auth::rbac::*;
use tensordb_core::auth::session::*;
use tensordb_core::config::Config;
use tensordb_core::engine::db::Database;

fn setup() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path(), Config::default()).unwrap();
    (db, dir)
}

#[test]
fn test_full_auth_workflow() {
    let (db, _dir) = setup();

    // Create default roles
    RoleManager::create_default_roles(&db).unwrap();

    // Create users with different roles
    UserManager::create_user(&db, "admin_alice", "admin_pass!", vec!["admin".to_string()]).unwrap();
    UserManager::create_user(&db, "analyst_bob", "bob_pass!", vec!["reader".to_string()]).unwrap();
    UserManager::create_user(
        &db,
        "writer_carol",
        "carol_pass!",
        vec!["writer".to_string()],
    )
    .unwrap();
    UserManager::create_user(&db, "guest_dave", "dave_pass!", vec![]).unwrap();

    // Authenticate users
    assert!(UserManager::authenticate(&db, "admin_alice", "admin_pass!")
        .unwrap()
        .is_some());
    assert!(UserManager::authenticate(&db, "analyst_bob", "wrong_pass")
        .unwrap()
        .is_none());
    assert!(UserManager::authenticate(&db, "nonexistent", "pass")
        .unwrap()
        .is_none());

    // Check privileges
    // Admin can do everything
    assert!(
        UserManager::check_privilege(&db, "admin_alice", Privilege::Drop, Some("any_table"))
            .unwrap()
    );
    assert!(UserManager::check_privilege(&db, "admin_alice", Privilege::Admin, None).unwrap());

    // Reader can only SELECT
    assert!(
        UserManager::check_privilege(&db, "analyst_bob", Privilege::Select, Some("orders"))
            .unwrap()
    );
    assert!(
        !UserManager::check_privilege(&db, "analyst_bob", Privilege::Insert, Some("orders"))
            .unwrap()
    );

    // Writer can SELECT, INSERT, UPDATE
    assert!(
        UserManager::check_privilege(&db, "writer_carol", Privilege::Select, Some("orders"))
            .unwrap()
    );
    assert!(
        UserManager::check_privilege(&db, "writer_carol", Privilege::Insert, Some("orders"))
            .unwrap()
    );
    assert!(
        !UserManager::check_privilege(&db, "writer_carol", Privilege::Drop, Some("orders"))
            .unwrap()
    );

    // Guest has no permissions
    assert!(
        !UserManager::check_privilege(&db, "guest_dave", Privilege::Select, Some("orders"))
            .unwrap()
    );
}

#[test]
fn test_direct_permissions() {
    let (db, _dir) = setup();

    UserManager::create_user(&db, "eve", "pass", vec![]).unwrap();

    // No access initially
    assert!(!UserManager::check_privilege(&db, "eve", Privilege::Select, Some("logs")).unwrap());

    // Grant specific table permission
    UserManager::grant_permission(&db, "eve", Permission::new(Privilege::Select, Some("logs")))
        .unwrap();

    // Can read logs
    assert!(UserManager::check_privilege(&db, "eve", Privilege::Select, Some("logs")).unwrap());
    // But not other tables
    assert!(!UserManager::check_privilege(&db, "eve", Privilege::Select, Some("users")).unwrap());
}

#[test]
fn test_session_workflow() {
    let (db, _dir) = setup();
    RoleManager::create_default_roles(&db).unwrap();

    let user =
        UserManager::create_user(&db, "frank", "pass123", vec!["reader".to_string()]).unwrap();

    // Resolve role permissions for session creation
    let reader_role = RoleManager::get_role(&db, "reader").unwrap().unwrap();
    let role_perms: Vec<Permission> = reader_role.permissions.into_iter().collect();
    let ctx_with_roles = AuthContext::from_user_with_roles(&user, role_perms);

    // Create session with resolved roles
    let store = SessionStore::new(3_600_000);
    let token = store.create_session_with_context(ctx_with_roles);

    // Validate session
    let ctx = store.get_session(&token).unwrap();
    assert_eq!(ctx.username, "frank");
    assert!(!ctx.is_superuser);

    // Check permissions via context
    assert!(ctx.has_privilege(Privilege::Select, Some("any_table")));
    assert!(!ctx.has_privilege(Privilege::Insert, Some("any_table")));

    // Revoke session
    store.revoke_session(&token);
    assert!(store.get_session(&token).is_none());
}

#[test]
fn test_password_change() {
    let (db, _dir) = setup();

    UserManager::create_user(&db, "grace", "old_pw", vec![]).unwrap();

    // Authenticate with old password
    assert!(UserManager::authenticate(&db, "grace", "old_pw")
        .unwrap()
        .is_some());

    // Change password
    UserManager::change_password(&db, "grace", "new_pw").unwrap();

    // Old password no longer works
    assert!(UserManager::authenticate(&db, "grace", "old_pw")
        .unwrap()
        .is_none());
    // New password works
    assert!(UserManager::authenticate(&db, "grace", "new_pw")
        .unwrap()
        .is_some());
}

#[test]
fn test_user_disable() {
    let (db, _dir) = setup();

    UserManager::create_user(&db, "hank", "pass", vec![]).unwrap();
    UserManager::disable_user(&db, "hank").unwrap();

    // Authentication should fail with disabled error
    let result = UserManager::authenticate(&db, "hank", "pass");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("disabled"));
}

#[test]
fn test_role_management() {
    let (db, _dir) = setup();

    // Create custom role
    RoleManager::create_role(&db, "auditor").unwrap();
    RoleManager::grant_to_role(&db, "auditor", Permission::new(Privilege::Select, None)).unwrap();

    // Assign to user
    UserManager::create_user(&db, "iris", "pass", vec![]).unwrap();
    UserManager::grant_role(&db, "iris", "auditor").unwrap();

    // Check privilege via role
    assert!(
        UserManager::check_privilege(&db, "iris", Privilege::Select, Some("any_table")).unwrap()
    );

    // Revoke role
    UserManager::revoke_role(&db, "iris", "auditor").unwrap();
    assert!(
        !UserManager::check_privilege(&db, "iris", Privilege::Select, Some("any_table")).unwrap()
    );
}
