/*!
 * Application Configuration for AlgoVeda Platform
 */

use serde::{Deserialize, Serialize};
use config::{Config, ConfigError};
use super::{TradingConfig, RiskConfig, DatabaseConfig, MonitoringConfig, SecurityConfig, ConfigLoader};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub app: AppInfo,
    pub trading: TradingConfig,
    pub risk: RiskConfig,
    pub database: DatabaseConfig,
    pub monitoring: MonitoringConfig,
    pub security: SecurityConfig,
    pub dhan: DhanConfig,
    pub market_data: MarketDataConfig,
    pub backtesting: BacktestingConfig,
    pub machine_learning: MLConfig,
    pub storage: StorageConfig,
    pub networking: NetworkingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppInfo {
    pub name: String,
    pub version: String,
    pub environment: String,
    pub log_level: String,
    pub max_workers: usize,
    pub shutdown_timeout: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DhanConfig {
    pub client_id: String,
    pub access_token: String,
    pub api_key: String,
    pub base_url: String,
    pub websocket_url: String,
    pub rate_limit: u32,
    pub timeout: u64,
    pub retry_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataConfig {
    pub primary_provider: String,
    pub backup_providers: Vec<String>,
    pub update_frequency: u64,
    pub history_retention: u32,
    pub enable_level2: bool,
    pub enable_level3: bool,
    pub compression_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestingConfig {
    pub engine: String,
    pub initial_capital: f64,
    pub commission_model: String,
    pub slippage_model: String,
    pub benchmark: String,
    pub risk_free_rate: f64,
    pub parallel_workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    pub default_model: String,
    pub feature_lookback: Vec<u32>,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub max_trials: u32,
    pub cv_folds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub engine: String,
    pub compression: String,
    pub retention_days: u32,
    pub backup_enabled: bool,
    pub replication_factor: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkingConfig {
    pub tcp_nodelay: bool,
    pub keepalive: bool,
    pub buffer_size: usize,
    pub connection_timeout: u64,
    pub max_connections: u32,
}

impl ConfigLoader for AppConfig {
    fn load(path: Option<&str>) -> Result<Self, ConfigError> {
        let mut config = Config::builder();
        
        if let Some(file_path) = path {
            config = config.add_source(config::File::with_name(file_path));
        }
        
        config = config.add_source(
            config::Environment::with_prefix("ALGOVEDA")
                .try_parsing(true)
                .separator("_")
        );
        
        config.build()?.try_deserialize()
    }
    
    fn validate(&self) -> Result<(), ConfigError> {
        if self.app.name.is_empty() {
            return Err(ConfigError::Message("App name cannot be empty".to_string()));
        }
        
        self.trading.validate()?;
        self.risk.validate()?;
        self.database.validate()?;
        self.monitoring.validate()?;
        self.security.validate()?;
        
        Ok(())
    }
    
    fn merge(&mut self, _other: Self) -> Result<(), ConfigError> {
        // Implementation for merging configurations
        Ok(())
    }
}

impl AppConfig {
    pub fn from_toml_str(toml_str: &str) -> Result<Self, ConfigError> {
        let config = Config::builder()
            .add_source(config::File::from_str(toml_str, config::FileFormat::Toml))
            .build()?;
        
        config.try_deserialize()
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            app: AppInfo {
                name: "AlgoVeda".to_string(),
                version: "1.0.0".to_string(),
                environment: "development".to_string(),
                log_level: "info".to_string(),
                max_workers: num_cpus::get(),
                shutdown_timeout: 30,
            },
            trading: TradingConfig::default(),
            risk: RiskConfig::default(),
            database: DatabaseConfig::default(),
            monitoring: MonitoringConfig::default(),
            security: SecurityConfig::default(),
            dhan: DhanConfig::default(),
            market_data: MarketDataConfig::default(),
            backtesting: BacktestingConfig::default(),
            machine_learning: MLConfig::default(),
            storage: StorageConfig::default(),
            networking: NetworkingConfig::default(),
        }
    }
}

impl Default for DhanConfig {
    fn default() -> Self {
        Self {
            client_id: "".to_string(),
            access_token: "".to_string(),
            api_key: "".to_string(),
            base_url: "https://api.dhan.co".to_string(),
            websocket_url: "wss://api-feed.dhan.co".to_string(),
            rate_limit: 100,
            timeout: 30,
            retry_attempts: 3,
        }
    }
}

impl Default for MarketDataConfig {
    fn default() -> Self {
        Self {
            primary_provider: "dhan".to_string(),
            backup_providers: vec!["yahoo".to_string()],
            update_frequency: 1000,
            history_retention: 730,
            enable_level2: true,
            enable_level3: false,
            compression_enabled: true,
        }
    }
}
