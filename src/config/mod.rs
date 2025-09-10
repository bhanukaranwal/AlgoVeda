/*!
 * Configuration Module
 * Configuration management for AlgoVeda trading platform
 */

use serde::{Deserialize, Serialize};
use std::{path::Path, fs};
use crate::error::{Result, AlgoVedaError};

pub mod app_config;
pub mod trading_config;
pub mod risk_config;
pub mod database_config;
pub mod monitoring_config;
pub mod security_config;

pub use app_config::AppConfig;
pub use trading_config::TradingConfig;
pub use risk_config::RiskConfig;
pub use database_config::DatabaseConfig;
pub use monitoring_config::MonitoringConfig;
pub use security_config::SecurityConfig;

/// Configuration trait for all config types
pub trait Configuration: Serialize + for<'de> Deserialize<'de> + Default {
    const CONFIG_NAME: &'static str;
    
    fn load() -> Result<Self> {
        Self::load_from_file(&format!("config/{}.toml", Self::CONFIG_NAME))
    }
    
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)
            .map_err(|e| AlgoVedaError::Config(format!("Failed to read config file: {}", e)))?;
        
        toml::from_str(&content)
            .map_err(|e| AlgoVedaError::Config(format!("Failed to parse config: {}", e)))
    }
    
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| AlgoVedaError::Config(format!("Failed to serialize config: {}", e)))?;
        
        fs::write(path, content)
            .map_err(|e| AlgoVedaError::Config(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
    
    fn validate(&self) -> Result<()>;
}

/// Environment-aware configuration loading
pub fn load_config_for_environment<T: Configuration>(env: &str) -> Result<T> {
    let config_path = format!("config/{}/{}.toml", env, T::CONFIG_NAME);
    
    if Path::new(&config_path).exists() {
        T::load_from_file(config_path)
    } else {
        T::load()
    }
}

/// Merge configurations with environment overrides
pub fn merge_with_env_overrides<T: Configuration>(mut config: T, env_prefix: &str) -> T {
    // This would typically use environment variables to override config values
    // Implementation would use reflection or macro-based field updates
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[derive(Serialize, Deserialize, Default)]
    struct TestConfig {
        value: i32,
        name: String,
    }

    impl Configuration for TestConfig {
        const CONFIG_NAME: &'static str = "test";
        
        fn validate(&self) -> Result<()> {
            if self.value < 0 {
                return Err(AlgoVedaError::Config("Value must be non-negative".to_string()));
            }
            Ok(())
        }
    }

    #[test]
    fn test_config_save_load() {
        let config = TestConfig {
            value: 42,
            name: "test".to_string(),
        };

        let temp_file = NamedTempFile::new().unwrap();
        config.save_to_file(temp_file.path()).unwrap();
        
        let loaded_config: TestConfig = TestConfig::load_from_file(temp_file.path()).unwrap();
        assert_eq!(loaded_config.value, 42);
        assert_eq!(loaded_config.name, "test");
    }
}
