use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::auth::rbac::{Permission, Privilege, UserRecord};
use crate::error::{Result, TensorError};

/// A session token for authenticated access.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionToken(pub String);

impl SessionToken {
    pub fn generate(username: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        let mut hasher = DefaultHasher::new();
        username.hash(&mut hasher);
        ts.hash(&mut hasher);
        let h1 = hasher.finish();

        let mut hasher2 = DefaultHasher::new();
        h1.hash(&mut hasher2);
        ts.wrapping_add(1).hash(&mut hasher2);
        let h2 = hasher2.finish();

        SessionToken(format!("sdb_{:016x}{:016x}", h1, h2))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Authentication context for the current session.
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub username: String,
    pub roles: Vec<String>,
    pub direct_permissions: Vec<Permission>,
    /// All effective permissions (direct + role-resolved).
    pub effective_permissions: Vec<Permission>,
    pub is_superuser: bool,
    pub created_at: u64,
}

impl AuthContext {
    /// Create an auth context from a user record.
    /// Note: this only includes direct permissions. Use `from_user_with_roles`
    /// to include role-resolved permissions.
    pub fn from_user(user: &UserRecord) -> Self {
        let is_superuser = user.roles.contains(&"admin".to_string())
            || user
                .direct_permissions
                .iter()
                .any(|p| p.privilege == Privilege::Admin);

        let direct: Vec<Permission> = user.direct_permissions.iter().cloned().collect();
        AuthContext {
            username: user.username.clone(),
            roles: user.roles.clone(),
            effective_permissions: direct.clone(),
            direct_permissions: direct,
            is_superuser,
            created_at: current_timestamp_ms(),
        }
    }

    /// Create an auth context with resolved role permissions.
    pub fn from_user_with_roles(user: &UserRecord, role_permissions: Vec<Permission>) -> Self {
        let is_superuser = user.roles.contains(&"admin".to_string())
            || user
                .direct_permissions
                .iter()
                .any(|p| p.privilege == Privilege::Admin)
            || role_permissions
                .iter()
                .any(|p| p.privilege == Privilege::Admin);

        let direct: Vec<Permission> = user.direct_permissions.iter().cloned().collect();
        let mut effective = direct.clone();
        effective.extend(role_permissions);

        AuthContext {
            username: user.username.clone(),
            roles: user.roles.clone(),
            direct_permissions: direct,
            effective_permissions: effective,
            is_superuser,
            created_at: current_timestamp_ms(),
        }
    }

    /// Create a superuser context (for bootstrap / internal use).
    pub fn superuser() -> Self {
        let perms = vec![Permission::new(Privilege::Admin, None)];
        AuthContext {
            username: "__system__".to_string(),
            roles: vec!["admin".to_string()],
            direct_permissions: perms.clone(),
            effective_permissions: perms,
            is_superuser: true,
            created_at: current_timestamp_ms(),
        }
    }

    /// Check if this context has a specific privilege.
    pub fn has_privilege(&self, privilege: Privilege, table: Option<&str>) -> bool {
        if self.is_superuser {
            return true;
        }
        self.effective_permissions
            .iter()
            .any(|p| p.covers(privilege, table))
    }

    /// Require a privilege or return an error.
    pub fn require_privilege(&self, privilege: Privilege, table: Option<&str>) -> Result<()> {
        if self.has_privilege(privilege, table) {
            Ok(())
        } else {
            Err(TensorError::SqlExec(format!(
                "permission denied: {} on {} for user {}",
                privilege.name(),
                table.unwrap_or("*"),
                self.username
            )))
        }
    }
}

/// In-memory session store for token management.
pub struct SessionStore {
    sessions: Arc<RwLock<HashMap<SessionToken, AuthContext>>>,
    ttl_ms: u64,
}

impl SessionStore {
    pub fn new(ttl_ms: u64) -> Self {
        SessionStore {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            ttl_ms,
        }
    }

    /// Create a new session for a user. Returns the session token.
    pub fn create_session(&self, user: &UserRecord) -> SessionToken {
        let token = SessionToken::generate(&user.username);
        let ctx = AuthContext::from_user(user);
        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(token.clone(), ctx);
        token
    }

    /// Create a session with a pre-built auth context (e.g., with resolved role permissions).
    pub fn create_session_with_context(&self, ctx: AuthContext) -> SessionToken {
        let token = SessionToken::generate(&ctx.username);
        let mut sessions = self.sessions.write().unwrap();
        sessions.insert(token.clone(), ctx);
        token
    }

    /// Look up a session by token.
    pub fn get_session(&self, token: &SessionToken) -> Option<AuthContext> {
        let sessions = self.sessions.read().unwrap();
        sessions.get(token).and_then(|ctx| {
            let now = current_timestamp_ms();
            if now - ctx.created_at > self.ttl_ms {
                None // Expired
            } else {
                Some(ctx.clone())
            }
        })
    }

    /// Revoke a session.
    pub fn revoke_session(&self, token: &SessionToken) -> bool {
        let mut sessions = self.sessions.write().unwrap();
        sessions.remove(token).is_some()
    }

    /// Revoke all sessions for a user.
    pub fn revoke_all_for_user(&self, username: &str) {
        let mut sessions = self.sessions.write().unwrap();
        sessions.retain(|_, ctx| ctx.username != username);
    }

    /// Clean up expired sessions.
    pub fn cleanup_expired(&self) -> usize {
        let now = current_timestamp_ms();
        let mut sessions = self.sessions.write().unwrap();
        let before = sessions.len();
        sessions.retain(|_, ctx| now - ctx.created_at <= self.ttl_ms);
        before - sessions.len()
    }

    /// Number of active sessions.
    pub fn active_count(&self) -> usize {
        self.sessions.read().unwrap().len()
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
    use std::collections::HashSet;

    fn mock_user(username: &str, roles: Vec<&str>) -> UserRecord {
        UserRecord {
            username: username.to_string(),
            password_hash: String::new(),
            roles: roles.into_iter().map(|r| r.to_string()).collect(),
            direct_permissions: HashSet::new(),
            enabled: true,
            created_at: current_timestamp_ms(),
            last_login: None,
        }
    }

    #[test]
    fn test_session_token_generation() {
        let t1 = SessionToken::generate("alice");
        let t2 = SessionToken::generate("alice");
        assert!(t1.as_str().starts_with("sdb_"));
        assert_ne!(t1, t2); // Unique tokens
    }

    #[test]
    fn test_auth_context_superuser() {
        let ctx = AuthContext::superuser();
        assert!(ctx.is_superuser);
        assert!(ctx.has_privilege(Privilege::Admin, None));
        assert!(ctx.has_privilege(Privilege::Drop, Some("any_table")));
    }

    #[test]
    fn test_auth_context_from_user() {
        let user = mock_user("bob", vec!["reader"]);
        let ctx = AuthContext::from_user(&user);
        assert_eq!(ctx.username, "bob");
        assert!(!ctx.is_superuser);
    }

    #[test]
    fn test_auth_context_admin_is_superuser() {
        let user = mock_user("root", vec!["admin"]);
        let ctx = AuthContext::from_user(&user);
        assert!(ctx.is_superuser);
        assert!(ctx.has_privilege(Privilege::Drop, Some("users")));
    }

    #[test]
    fn test_require_privilege_denied() {
        let user = mock_user("guest", vec![]);
        let ctx = AuthContext::from_user(&user);
        let result = ctx.require_privilege(Privilege::Select, Some("orders"));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("permission denied"));
    }

    #[test]
    fn test_session_store_create_and_get() {
        let store = SessionStore::new(3_600_000); // 1 hour TTL
        let user = mock_user("alice", vec!["reader"]);
        let token = store.create_session(&user);

        let ctx = store.get_session(&token);
        assert!(ctx.is_some());
        assert_eq!(ctx.unwrap().username, "alice");
    }

    #[test]
    fn test_session_store_revoke() {
        let store = SessionStore::new(3_600_000);
        let user = mock_user("bob", vec![]);
        let token = store.create_session(&user);

        assert!(store.revoke_session(&token));
        assert!(store.get_session(&token).is_none());
    }

    #[test]
    fn test_session_store_revoke_all_for_user() {
        let store = SessionStore::new(3_600_000);
        let user = mock_user("carol", vec![]);
        let _t1 = store.create_session(&user);
        let _t2 = store.create_session(&user);
        assert_eq!(store.active_count(), 2);

        store.revoke_all_for_user("carol");
        assert_eq!(store.active_count(), 0);
    }

    #[test]
    fn test_session_expiry() {
        let store = SessionStore::new(0); // Immediate expiry
        let user = mock_user("dave", vec![]);
        let token = store.create_session(&user);

        // Session should be expired already (TTL = 0)
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(store.get_session(&token).is_none());
    }
}
