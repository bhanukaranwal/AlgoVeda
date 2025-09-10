/*!
 * Configuration Module for AlgoVeda Trading Platform
 * 
 * This module provides centralized configuration management for all
 * components of the AlgoVeda platform, supporting:
 * - Hierarchical configuration loading
 * - Environment variable overrides
 * - Runtime configuration validation
 * - Hot-reload capabilities
 */

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use config::{Config, ConfigError, Environment, File};
use tracing::{info, warn};

// Re-export configuration modules
pub mod app_config;
pub mod trading_config;
pub mod risk_config;
pub mod database_config;
pub mod monitoring_config;
pub mod security_config;

// Re-export main configuration types
pub use app_config::AppConfig;
pub use trading_config::TradingConfig;
pub use risk_config::RiskConfig;
pub use database_config::DatabaseConfig;
pub use monitoring_config::MonitoringConfig;
pub use security_config::SecurityConfig;

/// Configuration loader trait
pub trait ConfigLoader {
    fn load(&self, path: Option<&str>) -> Result<Self, ConfigError>
    where
        Self: Sized;
    
    fn validate(&self) -> Result<(), ConfigError>;
    
    fn merge(&mut self, other: Self) -> Result<(), ConfigError>;
}

/// Configuration watcher for hot-reload functionality
#[derive(Debug)]
pub struct ConfigWatcher {
    config_path: String,
    last_modified: std::time::SystemTime,
}

impl ConfigWatcher {
    pub fn new(config_path: String) -> Self {
        Self {
            config_path,
            last_modified: std::time::SystemTime::UNIX_EPOCH,
        }
    }

    pub fn check_for_changes(&mut self) -> bool {
        if let Ok(metadata) = std::fs::metadata(&self.config_path) {
            if let Ok(modified) = metadata.modified() {
                if modified > self.last_modified {
                    self.last_modified = modified;
                    info!("Configuration file changed: {}", self.config_path);
                    return true;
                }
            }
        }
        false
    }
}

/// Configuration manager for the platform
#[derive(Debug, Clone)]
pub struct ConfigManager {
    app_config: Arc<AppConfig>,
    config_watcher: Option<Arc<std::sync::Mutex<ConfigWatcher>>>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new(config_path: Option<&str>) -> Result<Self, ConfigError> {
        let app_config = Arc::new(AppConfig::load(config_path)?);
        
        let config_watcher = config_path.map(|path| {
            Arc::new(std::sync::Mutex::new(ConfigWatcher::new(path.to_string())))
        });

        Ok(Self {
            app_config,
            config_watcher,
        })
    }

    /// Get the current application configuration
    pub fn get_config(&self) -> Arc<AppConfig> {
        Arc::clone(&self.app_config)
    }

    /// Check for configuration changes and reload if necessary
    pub fn check_and_reload(&mut self) -> Result<bool, ConfigError> {
        if let Some(watcher) = &self.config_watcher {
            let mut guard = watcher.lock().unwrap();
            if guard.check_for_changes() {
                drop(guard);
                self.reload_config()?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Reload configuration from file
    fn reload_config(&mut self) -> Result<(), ConfigError> {
        if let Some(watcher) = &self.config_watcher {
            let guard = watcher.lock().unwrap();
            let config_path = &guard.config_path;
            let new_config = AppConfig::load(Some(config_path))?;
            self.app_config = Arc::new(new_config);
            info!("Configuration reloaded successfully from: {}", config_path);
        }
        Ok(())
    }
}

/// Utility function to load configuration from multiple sources
pub fn load_config_from_sources(
    default_config: &str,
    env_prefix: &str,
    config_file: Option<&str>,
) -> Result<Config, ConfigError> {
    let mut config = Config::builder();

    // Start with default configuration
    config = config.add_source(File::from_str(default_config, config::FileFormat::Toml));

    // Add configuration file if specified
    if let Some(file_path) = config_file {
        config = config.add_source(File::with_name(file_path).required(false));
    }

    // Add environment variable overrides
    config = config.add_source(
        Environment::with_prefix(env_prefix)
            .try_parsing(true)
            .separator("_")
            .list_separator(",")
    );

    config.build()
}

/// Validate all configuration components
pub fn validate_complete_config(config: &AppConfig) -> Result<(), ConfigError> {
    // Validate each configuration section
    config.trading.validate()?;
    config.risk.validate()?;
    config.database.validate()?;
    config.monitoring.validate()?;
    config.security.validate()?;

    // Cross-section validation
    validate_cross_sections(config)?;

    info!("All configuration sections validated successfully");
    Ok(())
}

/// Cross-section configuration validation
fn validate_cross_sections(config: &AppConfig) -> Result<(), ConfigError> {
    // Validate that risk limits are compatible with trading settings
    if config.risk.max_position_size > config.trading.max_order_size {
        warn!("Risk max_position_size ({}) exceeds trading max_order_size ({})", 
              config.risk.max_position_size, config.trading.max_order_size);
    }

    // Validate database connections match monitoring requirements
    if config.monitoring.enable_database_metrics && !config.database.enable_monitoring {
        return Err(ConfigError::Message(
            "Monitoring requires database monitoring to be enabled".to_string()
        ));
    }

    // Validate security settings are compatible
    if config.security.require_tls && config.database.connection_string.starts_with("http://") {
        return Err(ConfigError::Message(
            "TLS required but database connection uses unencrypted HTTP".to_string()
        ));
    }

    Ok(())
}

/// Configuration defaults for testing and development
pub mod defaults {
    use super::*;

    pub const DEFAULT_TOML_CONFIG: &str = r#"
[app]
name = "AlgoVeda"
version = "1.0.0"
environment = "development"
log_level = "info"

[trading]
max_order_size = 1000000.0
default_commission = 0.001
enable_paper_trading = true
enable_live_trading = false

[risk]
max_position_size = 500000.0
max_daily_loss = 10000.0
enable_real_time_risk = true
risk_check_interval = 1000

[database]
connection_string = "postgresql://localhost:5432/algoveda"
pool_size = 10
enable_monitoring = true

[monitoring]
enable_metrics = true
metrics_port = 9090
enable_database_metrics = true

[security]
require_tls = false
jwt_secret = "development-secret"
session_timeout = 3600
"#;

    /// Get default configuration for testing
    pub fn get_test_config() -> Result<AppConfig, ConfigError> {
        AppConfig::from_toml_str(DEFAULT_TOML_CONFIG)
    }

    /// Get minimal configuration for development
    pub fn get_dev_config() -> Result<AppConfig, ConfigError> {
        let mut config = get_test_config()?;
        config.app.environment = "development".to_string();
        config.app.log_level = "debug".to_string();
        config.trading.enable_paper_trading = true;
        config.trading.enable_live_trading = false;
        Ok(config)
    }

    /// Get production-ready configuration template
    pub fn get_prod_config_template() -> &'static str {
        r#"
[app]
name = "AlgoVeda"
version = "1.0.0"
environment = "production"
log_level = "warn"

[trading]
max_order_size = 10000000.0
default_commission = 0.0005
enable_paper_trading = false
enable_live_trading = true

[risk]
max_position_size = 5000000.0
max_daily_loss = 50000.0
enable_real_time_risk = true
risk_check_interval = 100

[database]
connection_string = "${DATABASE_URL}"
pool_size = 50
enable_monitoring = true

[monitoring]
enable_metrics = true
metrics_port = 9090
enable_database_metrics = true

[security]
require_tls = true
jwt_secret = "${JWT_SECRET}"
session_timeout = 1800
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_loads() {
        let config = defaults::get_test_config();
        assert!(config.is_ok());
    }

    #[test]
    fn test_config_validation() {
        let config = defaults::get_test_config().unwrap();
        assert!(validate_complete_config(&config).is_ok());
    }

    #[test]
    fn test_config_manager() {
        let manager = ConfigManager::new(None);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_cross_section_validation() {
        let mut config = defaults::get_test_config().unwrap();
        
        // Create invalid configuration
        config.risk.max_position_size = config.trading.max_order_size + 1.0;
        
        // Should still pass validation with warning
        assert!(validate_complete_config(&config).is_ok());
    }

    #[test]
    fn test_environment_override() {
        std::env::set_var("ALGOVEDA_TRADING__MAX_ORDER_SIZE", "2000000");
        
        let config_result = load_config_from_sources(
            defaults::DEFAULT_TOML_CONFIG,
            "ALGOVEDA",
            None,
        );
        
        assert!(config_result.is_ok());
        std::env::remove_var("ALGOVEDA_TRADING__MAX_ORDER_SIZE");
    }
}
