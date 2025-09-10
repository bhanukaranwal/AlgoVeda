use serde::{Deserialize, Serialize};
use config::ConfigError;
use super::ConfigLoader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub require_tls: bool,
    pub tls_cert_path: String,
    pub tls_key_path: String,
    pub jwt_secret: String,
    pub jwt_expiry: u64,
    pub session_timeout: u64,
    pub max_login_attempts: u32,
    pub lockout_duration: u64,
    pub password_policy: PasswordPolicy,
    pub encryption: EncryptionConfig,
    pub authentication: AuthenticationConfig,
    pub authorization: AuthorizationConfig,
    pub audit: AuditConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PasswordPolicy {
    pub min_length: u32,
    pub require_uppercase: bool,
    pub require_lowercase: bool,
    pub require_numbers: bool,
    pub require_special_chars: bool,
    pub max_age_days: u32,
    pub history_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub algorithm: String,
    pub key_size: u32,
    pub key_rotation_interval: u64,
    pub at_rest_encryption: bool,
    pub in_transit_encryption: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    pub methods: Vec<String>,
    pub mfa_required: bool,
    pub mfa_methods: Vec<String>,
    pub oauth_providers: Vec<OAuthProvider>,
    pub ldap: Option<LdapConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthProvider {
    pub name: String,
    pub client_id: String,
    pub client_secret: String,
    pub redirect_url: String,
    pub scopes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdapConfig {
    pub server: String,
    pub port: u16,
    pub bind_dn: String,
    pub bind_password: String,
    pub user_base: String,
    pub user_filter: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationConfig {
    pub enable_rbac: bool,
    pub default_role: String,
    pub admin_users: Vec<String>,
    pub role_mappings: std::collections::HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enable_audit_logging: bool,
    pub log_all_requests: bool,
    pub log_sensitive_data: bool,
    pub retention_days: u32,
    pub storage_backend: String,
}

impl ConfigLoader for SecurityConfig {
    fn load(_path: Option<&str>) -> Result<Self, ConfigError> {
        Ok(Self::default())
    }
    
    fn validate(&self) -> Result<(), ConfigError> {
        if self.jwt_secret.len() < 32 {
            return Err(ConfigError::Message("JWT secret must be at least 32 characters".to_string()));
        }
        
        if self.jwt_expiry == 0 {
            return Err(ConfigError::Message("JWT expiry must be greater than 0".to_string()));
        }
        
        if self.password_policy.min_length < 8 {
            return Err(ConfigError::Message("Password minimum length should be at least 8".to_string()));
        }
        
        Ok(())
    }
    
    fn merge(&mut self, _other: Self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_tls: false,
            tls_cert_path: "/etc/ssl/certs/algoveda.crt".to_string(),
            tls_key_path: "/etc/ssl/private/algoveda.key".to_string(),
            jwt_secret: "your-super-secret-jwt-key-change-in-production".to_string(),
            jwt_expiry: 3600,
            session_timeout: 1800,
            max_login_attempts: 5,
            lockout_duration: 900,
            password_policy: PasswordPolicy::default(),
            encryption: EncryptionConfig::default(),
            authentication: AuthenticationConfig::default(),
            authorization: AuthorizationConfig::default(),
            audit: AuditConfig::default(),
        }
    }
}

impl Default for PasswordPolicy {
    fn default() -> Self {
        Self {
            min_length: 8,
            require_uppercase: true,
            require_lowercase: true,
            require_numbers: true,
            require_special_chars: true,
            max_age_days: 90,
            history_count: 5,
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: "AES-256-GCM".to_string(),
            key_size: 256,
            key_rotation_interval: 86400,
            at_rest_encryption: true,
            in_transit_encryption: true,
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            methods: vec!["password".to_string(), "jwt".to_string()],
            mfa_required: false,
            mfa_methods: vec!["totp".to_string()],
            oauth_providers: vec![],
            ldap: None,
        }
    }
}

impl Default for AuthorizationConfig {
    fn default() -> Self {
        let mut role_mappings = std::collections::HashMap::new();
        role_mappings.insert("admin".to_string(), vec!["*".to_string()]);
        role_mappings.insert("trader".to_string(), vec!["trading:*".to_string(), "portfolio:read".to_string()]);
        role_mappings.insert("viewer".to_string(), vec!["portfolio:read".to_string(), "reports:read".to_string()]);
        
        Self {
            enable_rbac: true,
            default_role: "viewer".to_string(),
            admin_users: vec![],
            role_mappings,
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enable_audit_logging: true,
            log_all_requests: false,
            log_sensitive_data: false,
            retention_days: 365,
            storage_backend: "database".to_string(),
        }
    }
}
