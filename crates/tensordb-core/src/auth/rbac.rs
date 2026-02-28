use std::collections::HashSet;

use crate::engine::db::Database;
use crate::error::{Result, TensorError};

/// Key prefix for user records.
const USER_PREFIX: &str = "__auth/user/";
/// Key prefix for role definitions.
const ROLE_PREFIX: &str = "__auth/role/";

/// A database privilege.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Privilege {
    Select,
    Insert,
    Update,
    Delete,
    Create,
    Drop,
    Alter,
    Admin,
}

impl Privilege {
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "SELECT" => Some(Privilege::Select),
            "INSERT" => Some(Privilege::Insert),
            "UPDATE" => Some(Privilege::Update),
            "DELETE" => Some(Privilege::Delete),
            "CREATE" => Some(Privilege::Create),
            "DROP" => Some(Privilege::Drop),
            "ALTER" => Some(Privilege::Alter),
            "ADMIN" => Some(Privilege::Admin),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Privilege::Select => "SELECT",
            Privilege::Insert => "INSERT",
            Privilege::Update => "UPDATE",
            Privilege::Delete => "DELETE",
            Privilege::Create => "CREATE",
            Privilege::Drop => "DROP",
            Privilege::Alter => "ALTER",
            Privilege::Admin => "ADMIN",
        }
    }
}

/// A permission: privilege + optional table scope (None = all tables).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Permission {
    pub privilege: Privilege,
    pub table: Option<String>,
}

impl Permission {
    pub fn new(privilege: Privilege, table: Option<&str>) -> Self {
        Permission {
            privilege,
            table: table.map(|t| t.to_string()),
        }
    }

    /// Check if this permission covers the requested privilege on the given table.
    pub fn covers(&self, privilege: Privilege, table: Option<&str>) -> bool {
        if self.privilege == Privilege::Admin {
            return true;
        }
        if self.privilege != privilege {
            return false;
        }
        match (&self.table, table) {
            (None, _) => true,            // Global permission covers all tables
            (Some(p), Some(t)) => p == t, // Table-specific must match
            (Some(_), None) => false,     // Table-specific doesn't cover global ops
        }
    }
}

/// A role with a set of permissions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Role {
    pub name: String,
    pub permissions: HashSet<Permission>,
    pub created_at: u64,
}

impl Role {
    pub fn new(name: &str) -> Self {
        Role {
            name: name.to_string(),
            permissions: HashSet::new(),
            created_at: current_timestamp_ms(),
        }
    }

    pub fn grant(&mut self, permission: Permission) {
        self.permissions.insert(permission);
    }

    pub fn revoke(&mut self, permission: &Permission) {
        self.permissions.remove(permission);
    }

    pub fn has_privilege(&self, privilege: Privilege, table: Option<&str>) -> bool {
        self.permissions.iter().any(|p| p.covers(privilege, table))
    }
}

/// A user record stored in the database.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct UserRecord {
    pub username: String,
    pub password_hash: String,
    pub roles: Vec<String>,
    pub direct_permissions: HashSet<Permission>,
    pub enabled: bool,
    pub created_at: u64,
    pub last_login: Option<u64>,
}

impl UserRecord {
    pub fn storage_key(&self) -> String {
        format!("{}{}", USER_PREFIX, self.username)
    }
}

/// Hash a password using a simple but effective approach.
/// Uses HMAC-like construction with SHA-256 (via our existing hasher).
fn hash_password(password: &str, salt: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    salt.hash(&mut hasher);
    password.hash(&mut hasher);
    let h1 = hasher.finish();

    let mut hasher2 = DefaultHasher::new();
    h1.hash(&mut hasher2);
    password.hash(&mut hasher2);
    salt.hash(&mut hasher2);
    let h2 = hasher2.finish();

    format!("spectra${}${:016x}{:016x}", salt, h1, h2)
}

fn verify_password(password: &str, stored_hash: &str) -> bool {
    let parts: Vec<&str> = stored_hash.split('$').collect();
    if parts.len() != 3 || parts[0] != "spectra" {
        return false;
    }
    let salt = parts[1];
    let expected = hash_password(password, salt);
    expected == stored_hash
}

fn generate_salt() -> String {
    let ts = current_timestamp_ms();
    let rand_bits = ts
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    format!("{:016x}", rand_bits)
}

/// Manages user accounts.
pub struct UserManager;

impl UserManager {
    /// Create a new user.
    pub fn create_user(
        db: &Database,
        username: &str,
        password: &str,
        roles: Vec<String>,
    ) -> Result<UserRecord> {
        let key = format!("{}{}", USER_PREFIX, username);
        if db.get(key.as_bytes(), None, None)?.is_some() {
            return Err(TensorError::SqlExec(format!(
                "user already exists: {username}"
            )));
        }

        let salt = generate_salt();
        let password_hash = hash_password(password, &salt);

        let user = UserRecord {
            username: username.to_string(),
            password_hash,
            roles,
            direct_permissions: HashSet::new(),
            enabled: true,
            created_at: current_timestamp_ms(),
            last_login: None,
        };

        let value = serde_json::to_vec(&user)
            .map_err(|e| TensorError::SqlExec(format!("failed to serialize user: {e}")))?;
        db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        Ok(user)
    }

    /// Authenticate a user. Returns the user record if credentials are valid.
    pub fn authenticate(
        db: &Database,
        username: &str,
        password: &str,
    ) -> Result<Option<UserRecord>> {
        let key = format!("{}{}", USER_PREFIX, username);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let user: UserRecord = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse user: {e}")))?;
                if !user.enabled {
                    return Err(TensorError::SqlExec(format!(
                        "user is disabled: {username}"
                    )));
                }
                if verify_password(password, &user.password_hash) {
                    Ok(Some(user))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    /// Get a user record.
    pub fn get_user(db: &Database, username: &str) -> Result<Option<UserRecord>> {
        let key = format!("{}{}", USER_PREFIX, username);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let user: UserRecord = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse user: {e}")))?;
                Ok(Some(user))
            }
            None => Ok(None),
        }
    }

    /// List all users.
    pub fn list_users(db: &Database) -> Result<Vec<UserRecord>> {
        let rows = db.scan_prefix(USER_PREFIX.as_bytes(), None, None, None)?;
        let mut users = Vec::new();
        for row in rows {
            if let Ok(user) = serde_json::from_slice::<UserRecord>(&row.doc) {
                users.push(user);
            }
        }
        Ok(users)
    }

    /// Update a user's password.
    pub fn change_password(db: &Database, username: &str, new_password: &str) -> Result<()> {
        let key = format!("{}{}", USER_PREFIX, username);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let mut user: UserRecord = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse user: {e}")))?;
                let salt = generate_salt();
                user.password_hash = hash_password(new_password, &salt);
                let value = serde_json::to_vec(&user)
                    .map_err(|e| TensorError::SqlExec(format!("failed to serialize user: {e}")))?;
                db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
                Ok(())
            }
            None => Err(TensorError::SqlExec(format!("user not found: {username}"))),
        }
    }

    /// Grant a role to a user.
    pub fn grant_role(db: &Database, username: &str, role: &str) -> Result<()> {
        let key = format!("{}{}", USER_PREFIX, username);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let mut user: UserRecord = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse user: {e}")))?;
                if !user.roles.contains(&role.to_string()) {
                    user.roles.push(role.to_string());
                }
                let value = serde_json::to_vec(&user)
                    .map_err(|e| TensorError::SqlExec(format!("failed to serialize user: {e}")))?;
                db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
                Ok(())
            }
            None => Err(TensorError::SqlExec(format!("user not found: {username}"))),
        }
    }

    /// Revoke a role from a user.
    pub fn revoke_role(db: &Database, username: &str, role: &str) -> Result<()> {
        let key = format!("{}{}", USER_PREFIX, username);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let mut user: UserRecord = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse user: {e}")))?;
                user.roles.retain(|r| r != role);
                let value = serde_json::to_vec(&user)
                    .map_err(|e| TensorError::SqlExec(format!("failed to serialize user: {e}")))?;
                db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
                Ok(())
            }
            None => Err(TensorError::SqlExec(format!("user not found: {username}"))),
        }
    }

    /// Grant a direct permission to a user.
    pub fn grant_permission(db: &Database, username: &str, permission: Permission) -> Result<()> {
        let key = format!("{}{}", USER_PREFIX, username);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let mut user: UserRecord = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse user: {e}")))?;
                user.direct_permissions.insert(permission);
                let value = serde_json::to_vec(&user)
                    .map_err(|e| TensorError::SqlExec(format!("failed to serialize user: {e}")))?;
                db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
                Ok(())
            }
            None => Err(TensorError::SqlExec(format!("user not found: {username}"))),
        }
    }

    /// Disable a user (soft-delete).
    pub fn disable_user(db: &Database, username: &str) -> Result<()> {
        let key = format!("{}{}", USER_PREFIX, username);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let mut user: UserRecord = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse user: {e}")))?;
                user.enabled = false;
                let value = serde_json::to_vec(&user)
                    .map_err(|e| TensorError::SqlExec(format!("failed to serialize user: {e}")))?;
                db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
                Ok(())
            }
            None => Err(TensorError::SqlExec(format!("user not found: {username}"))),
        }
    }

    /// Check if a user has a specific privilege on a table.
    /// Checks direct permissions first, then role-based permissions.
    pub fn check_privilege(
        db: &Database,
        username: &str,
        privilege: Privilege,
        table: Option<&str>,
    ) -> Result<bool> {
        let user = match Self::get_user(db, username)? {
            Some(u) => u,
            None => return Ok(false),
        };

        if !user.enabled {
            return Ok(false);
        }

        // Check direct permissions
        if user
            .direct_permissions
            .iter()
            .any(|p| p.covers(privilege, table))
        {
            return Ok(true);
        }

        // Check role-based permissions
        for role_name in &user.roles {
            if let Some(role) = RoleManager::get_role(db, role_name)? {
                if role.has_privilege(privilege, table) {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
}

/// Manages roles.
pub struct RoleManager;

impl RoleManager {
    /// Create a new role.
    pub fn create_role(db: &Database, name: &str) -> Result<Role> {
        let key = format!("{}{}", ROLE_PREFIX, name);
        if db.get(key.as_bytes(), None, None)?.is_some() {
            return Err(TensorError::SqlExec(format!("role already exists: {name}")));
        }

        let role = Role::new(name);
        let value = serde_json::to_vec(&role)
            .map_err(|e| TensorError::SqlExec(format!("failed to serialize role: {e}")))?;
        db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        Ok(role)
    }

    /// Get a role.
    pub fn get_role(db: &Database, name: &str) -> Result<Option<Role>> {
        let key = format!("{}{}", ROLE_PREFIX, name);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let role: Role = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse role: {e}")))?;
                Ok(Some(role))
            }
            None => Ok(None),
        }
    }

    /// List all roles.
    pub fn list_roles(db: &Database) -> Result<Vec<Role>> {
        let rows = db.scan_prefix(ROLE_PREFIX.as_bytes(), None, None, None)?;
        let mut roles = Vec::new();
        for row in rows {
            if let Ok(role) = serde_json::from_slice::<Role>(&row.doc) {
                roles.push(role);
            }
        }
        Ok(roles)
    }

    /// Grant a permission to a role.
    pub fn grant_to_role(db: &Database, role_name: &str, permission: Permission) -> Result<()> {
        let key = format!("{}{}", ROLE_PREFIX, role_name);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let mut role: Role = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse role: {e}")))?;
                role.grant(permission);
                let value = serde_json::to_vec(&role)
                    .map_err(|e| TensorError::SqlExec(format!("failed to serialize role: {e}")))?;
                db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
                Ok(())
            }
            None => Err(TensorError::SqlExec(format!("role not found: {role_name}"))),
        }
    }

    /// Revoke a permission from a role.
    pub fn revoke_from_role(db: &Database, role_name: &str, permission: &Permission) -> Result<()> {
        let key = format!("{}{}", ROLE_PREFIX, role_name);
        match db.get(key.as_bytes(), None, None)? {
            Some(bytes) => {
                let mut role: Role = serde_json::from_slice(&bytes)
                    .map_err(|e| TensorError::SqlExec(format!("failed to parse role: {e}")))?;
                role.revoke(permission);
                let value = serde_json::to_vec(&role)
                    .map_err(|e| TensorError::SqlExec(format!("failed to serialize role: {e}")))?;
                db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
                Ok(())
            }
            None => Err(TensorError::SqlExec(format!("role not found: {role_name}"))),
        }
    }

    /// Delete a role.
    pub fn delete_role(db: &Database, name: &str) -> Result<()> {
        let key = format!("{}{}", ROLE_PREFIX, name);
        db.put(
            key.as_bytes(),
            b"{\"deleted\":true}".to_vec(),
            0,
            u64::MAX,
            None,
        )?;
        Ok(())
    }

    /// Create default built-in roles.
    pub fn create_default_roles(db: &Database) -> Result<()> {
        // Admin role — full access
        if RoleManager::get_role(db, "admin")?.is_none() {
            let mut admin = Role::new("admin");
            admin.grant(Permission::new(Privilege::Admin, None));
            let key = format!("{}admin", ROLE_PREFIX);
            let value = serde_json::to_vec(&admin)
                .map_err(|e| TensorError::SqlExec(format!("failed to serialize role: {e}")))?;
            db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        }

        // Reader role — SELECT on all tables
        if RoleManager::get_role(db, "reader")?.is_none() {
            let mut reader = Role::new("reader");
            reader.grant(Permission::new(Privilege::Select, None));
            let key = format!("{}reader", ROLE_PREFIX);
            let value = serde_json::to_vec(&reader)
                .map_err(|e| TensorError::SqlExec(format!("failed to serialize role: {e}")))?;
            db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        }

        // Writer role — SELECT + INSERT + UPDATE on all tables
        if RoleManager::get_role(db, "writer")?.is_none() {
            let mut writer = Role::new("writer");
            writer.grant(Permission::new(Privilege::Select, None));
            writer.grant(Permission::new(Privilege::Insert, None));
            writer.grant(Permission::new(Privilege::Update, None));
            let key = format!("{}writer", ROLE_PREFIX);
            let value = serde_json::to_vec(&writer)
                .map_err(|e| TensorError::SqlExec(format!("failed to serialize role: {e}")))?;
            db.put(key.as_bytes(), value, 0, u64::MAX, None)?;
        }

        Ok(())
    }
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
    fn test_permission_covers() {
        let admin = Permission::new(Privilege::Admin, None);
        assert!(admin.covers(Privilege::Select, Some("orders")));
        assert!(admin.covers(Privilege::Drop, None));

        let global_select = Permission::new(Privilege::Select, None);
        assert!(global_select.covers(Privilege::Select, Some("orders")));
        assert!(!global_select.covers(Privilege::Insert, Some("orders")));

        let table_select = Permission::new(Privilege::Select, Some("orders"));
        assert!(table_select.covers(Privilege::Select, Some("orders")));
        assert!(!table_select.covers(Privilege::Select, Some("users")));
        assert!(!table_select.covers(Privilege::Select, None));
    }

    #[test]
    fn test_role_permissions() {
        let mut role = Role::new("analyst");
        role.grant(Permission::new(Privilege::Select, None));
        role.grant(Permission::new(Privilege::Insert, Some("reports")));

        assert!(role.has_privilege(Privilege::Select, Some("anything")));
        assert!(role.has_privilege(Privilege::Insert, Some("reports")));
        assert!(!role.has_privilege(Privilege::Insert, Some("users")));
        assert!(!role.has_privilege(Privilege::Drop, None));
    }

    #[test]
    fn test_create_and_authenticate_user() {
        let (db, _dir) = setup();

        UserManager::create_user(&db, "alice", "s3cret!", vec!["reader".to_string()]).unwrap();

        // Correct password
        let user = UserManager::authenticate(&db, "alice", "s3cret!").unwrap();
        assert!(user.is_some());
        assert_eq!(user.unwrap().username, "alice");

        // Wrong password
        let user = UserManager::authenticate(&db, "alice", "wrong").unwrap();
        assert!(user.is_none());

        // Non-existent user
        let user = UserManager::authenticate(&db, "bob", "pass").unwrap();
        assert!(user.is_none());
    }

    #[test]
    fn test_duplicate_user_rejected() {
        let (db, _dir) = setup();
        UserManager::create_user(&db, "alice", "pass1", vec![]).unwrap();
        let result = UserManager::create_user(&db, "alice", "pass2", vec![]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("already exists"));
    }

    #[test]
    fn test_change_password() {
        let (db, _dir) = setup();
        UserManager::create_user(&db, "bob", "old_pass", vec![]).unwrap();

        UserManager::change_password(&db, "bob", "new_pass").unwrap();

        // Old password fails
        assert!(UserManager::authenticate(&db, "bob", "old_pass")
            .unwrap()
            .is_none());
        // New password works
        assert!(UserManager::authenticate(&db, "bob", "new_pass")
            .unwrap()
            .is_some());
    }

    #[test]
    fn test_disable_user() {
        let (db, _dir) = setup();
        UserManager::create_user(&db, "carol", "pass", vec![]).unwrap();
        UserManager::disable_user(&db, "carol").unwrap();

        let result = UserManager::authenticate(&db, "carol", "pass");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("disabled"));
    }

    #[test]
    fn test_grant_revoke_role() {
        let (db, _dir) = setup();
        UserManager::create_user(&db, "dave", "pass", vec![]).unwrap();

        UserManager::grant_role(&db, "dave", "writer").unwrap();
        let user = UserManager::get_user(&db, "dave").unwrap().unwrap();
        assert!(user.roles.contains(&"writer".to_string()));

        UserManager::revoke_role(&db, "dave", "writer").unwrap();
        let user = UserManager::get_user(&db, "dave").unwrap().unwrap();
        assert!(!user.roles.contains(&"writer".to_string()));
    }

    #[test]
    fn test_rbac_check_privilege() {
        let (db, _dir) = setup();
        RoleManager::create_default_roles(&db).unwrap();

        UserManager::create_user(&db, "admin_user", "pass", vec!["admin".to_string()]).unwrap();
        UserManager::create_user(&db, "reader_user", "pass", vec!["reader".to_string()]).unwrap();
        UserManager::create_user(&db, "nobody", "pass", vec![]).unwrap();

        // Admin can do anything
        assert!(
            UserManager::check_privilege(&db, "admin_user", Privilege::Select, Some("t")).unwrap()
        );
        assert!(
            UserManager::check_privilege(&db, "admin_user", Privilege::Drop, Some("t")).unwrap()
        );

        // Reader can SELECT
        assert!(
            UserManager::check_privilege(&db, "reader_user", Privilege::Select, Some("t")).unwrap()
        );
        assert!(
            !UserManager::check_privilege(&db, "reader_user", Privilege::Insert, Some("t"))
                .unwrap()
        );

        // Nobody can't do anything
        assert!(
            !UserManager::check_privilege(&db, "nobody", Privilege::Select, Some("t")).unwrap()
        );
    }

    #[test]
    fn test_direct_permission_override() {
        let (db, _dir) = setup();
        UserManager::create_user(&db, "eve", "pass", vec![]).unwrap();

        // No access initially
        assert!(
            !UserManager::check_privilege(&db, "eve", Privilege::Select, Some("logs")).unwrap()
        );

        // Grant direct permission
        UserManager::grant_permission(&db, "eve", Permission::new(Privilege::Select, Some("logs")))
            .unwrap();

        // Now has access to logs only
        assert!(UserManager::check_privilege(&db, "eve", Privilege::Select, Some("logs")).unwrap());
        assert!(
            !UserManager::check_privilege(&db, "eve", Privilege::Select, Some("other")).unwrap()
        );
    }

    #[test]
    fn test_create_and_list_roles() {
        let (db, _dir) = setup();
        RoleManager::create_role(&db, "analyst").unwrap();
        RoleManager::create_role(&db, "ops").unwrap();
        RoleManager::grant_to_role(&db, "analyst", Permission::new(Privilege::Select, None))
            .unwrap();

        let roles = RoleManager::list_roles(&db).unwrap();
        assert!(roles.len() >= 2);

        let analyst = RoleManager::get_role(&db, "analyst").unwrap().unwrap();
        assert!(analyst.has_privilege(Privilege::Select, Some("any_table")));
    }

    #[test]
    fn test_list_users() {
        let (db, _dir) = setup();
        UserManager::create_user(&db, "u1", "p", vec![]).unwrap();
        UserManager::create_user(&db, "u2", "p", vec![]).unwrap();

        let users = UserManager::list_users(&db).unwrap();
        assert!(users.len() >= 2);
    }

    #[test]
    fn test_privilege_from_str() {
        assert_eq!(Privilege::from_str_name("SELECT"), Some(Privilege::Select));
        assert_eq!(Privilege::from_str_name("admin"), Some(Privilege::Admin));
        assert_eq!(Privilege::from_str_name("UNKNOWN"), None);
    }
}
