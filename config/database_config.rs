use serde::{Deserialize, Serialize};
use config::ConfigError;
use super::ConfigLoader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub connection_string: String,
    pub pool_size: u32,
    pub max_overflow: u32,
    pub connection_timeout: u64,
    pub idle_timeout: u64,
    pub max_lifetime: u64,
    pub enable_ssl: bool,
    pub enable_monitoring: bool,
    pub read_replicas: Vec<ReplicaConfig>,
    pub redis: RedisConfig,
    pub clickhouse: ClickhouseConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicaConfig {
    pub connection_string: String,
    pub weight: u32,
    pub lag_threshold: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub connection_string: String,
    pub pool_size: u32,
    pub timeout: u64,
    pub cluster_mode: bool,
    pub master_name: Option<String>,
    pub password: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClickhouseConfig {
    pub connection_string: String,
    pub pool_size: u32,
    pub compression: String,
    pub enable_async_inserts: bool,
    pub batch_size: u32,
}

impl ConfigLoader for DatabaseConfig {
    fn load(_path: Option<&str>) -> Result<Self, ConfigError> {
        Ok(Self::default())
    }
    
    fn validate(&self) -> Result<(), ConfigError> {
        if self.connection_string.is_empty() {
            return Err(ConfigError::Message("Database connection string cannot be empty".to_string()));
        }
        
        if self.pool_size == 0 {
            return Err(ConfigError::Message("Database pool size must be greater than 0".to_string()));
        }
        
        if self.redis.connection_string.is_empty() {
            return Err(ConfigError::Message("Redis connection string cannot be empty".to_string()));
        }
        
        Ok(())
    }
    
    fn merge(&mut self, _other: Self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            connection_string: "postgresql://localhost:5432/algoveda".to_string(),
            pool_size: 10,
            max_overflow: 20,
            connection_timeout: 30,
            idle_timeout: 300,
            max_lifetime: 3600,
            enable_ssl: false,
            enable_monitoring: true,
            read_replicas: vec![],
            redis: RedisConfig::default(),
            clickhouse: ClickhouseConfig::default(),
        }
    }
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            connection_string: "redis://localhost:6379".to_string(),
            pool_size: 10,
            timeout: 5,
            cluster_mode: false,
            master_name: None,
            password: None,
        }
    }
}

impl Default for ClickhouseConfig {
    fn default() -> Self {
        Self {
            connection_string: "http://localhost:8123".to_string(),
            pool_size: 5,
            compression: "lz4".to_string(),
            enable_async_inserts: true,
            batch_size: 10000,
        }
    }
}
