/*!
 * Application Configuration
 * Main application configuration settings
 */

use serde::{Deserialize, Serialize};
use std::time::Duration;
use crate::config::{Configuration, DatabaseConfig, MonitoringConfig, SecurityConfig};
use crate::error::{Result, AlgoVedaError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Application metadata
    pub app: AppMetadata,
    
    /// Server configuration
    pub server: ServerConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Performance settings
    pub performance: PerformanceConfig,
    
    /// Market data configuration
    pub market_data: MarketDataConfig,
    
    /// Storage configuration
    pub storage: StorageConfig,
    
    /// Network configuration
    pub network: NetworkConfig,
    
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub build_timestamp: String,
    pub git_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub worker_threads: usize,
    pub enable_http2: bool,
    pub enable_compression: bool,
    pub request_timeout_ms: u64,
    pub keep_alive_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub output: LogOutput,
    pub file_rotation: FileRotationConfig,
    pub structured_logging: bool,
    pub include_caller: bool,
    pub include_thread_id: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutput {
    Console,
    File { path: String },
    Both { path: String },
    Syslog,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRotationConfig {
    pub max_size_mb: u64,
    pub max_files: u32,
    pub compress: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub cpu_affinity: Option<Vec<usize>>,
    pub numa_node: Option<usize>,
    pub huge_pages: bool,
    pub memory_prefetch: bool,
    pub branch_prediction_hints: bool,
    pub simd_optimization: bool,
    pub lock_free_structures: bool,
    pub zero_copy_networking: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataConfig {
    pub providers: Vec<MarketDataProvider>,
    pub buffer_size: usize,
    pub max_symbols: usize,
    pub enable_level2: bool,
    pub enable_time_and_sales: bool,
    pub multicast_groups: Vec<String>,
    pub failover_enabled: bool,
    pub latency_monitoring: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataProvider {
    pub name: String,
    pub url: String,
    pub api_key: Option<String>,
    pub priority: u8,
    pub enabled: bool,
    pub rate_limit: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub engine: StorageEngine,
    pub data_directory: String,
    pub max_memory_usage_gb: u64,
    pub compression_enabled: bool,
    pub encryption_enabled: bool,
    pub backup_enabled: bool,
    pub backup_interval_hours: u64,
    pub retention_days: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageEngine {
    Memory,
    Disk,
    Hybrid,
    Distributed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub tcp_nodelay: bool,
    pub tcp_keepalive: bool,
    pub so_reuseaddr: bool,
    pub so_reuseport: bool,
    pub send_buffer_size: usize,
    pub recv_buffer_size: usize,
    pub max_frame_size: usize,
    pub connection_timeout_ms: u64,
    pub read_timeout_ms: u64,
    pub write_timeout_ms: u64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            app: AppMetadata {
                name: "AlgoVeda".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: "Ultra-High Performance Algorithmic Trading Platform".to_string(),
                author: "AlgoVeda Team".to_string(),
                build_timestamp: chrono::Utc::now().to_rfc3339(),
                git_hash: "unknown".to_string(),
            },
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                max_connections: 10000,
                worker_threads: num_cpus::get(),
                enable_http2: true,
                enable_compression: true,
                request_timeout_ms: 30000,
                keep_alive_timeout_ms: 60000,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "json".to_string(),
                output: LogOutput::Console,
                file_rotation: FileRotationConfig {
                    max_size_mb: 100,
                    max_files: 10,
                    compress: true,
                },
                structured_logging: true,
                include_caller: true,
                include_thread_id: true,
            },
            performance: PerformanceConfig {
                cpu_affinity: None,
                numa_node: None,
                huge_pages: false,
                memory_prefetch: true,
                branch_prediction_hints: true,
                simd_optimization: true,
                lock_free_structures: true,
                zero_copy_networking: true,
            },
            market_data: MarketDataConfig {
                providers: vec![],
                buffer_size: 1024 * 1024,
                max_symbols: 10000,
                enable_level2: true,
                enable_time_and_sales: true,
                multicast_groups: vec![],
                failover_enabled: true,
                latency_monitoring: true,
            },
            storage: StorageConfig {
                engine: StorageEngine::Hybrid,
                data_directory: "./data".to_string(),
                max_memory_usage_gb: 32,
                compression_enabled: true,
                encryption_enabled: false,
                backup_enabled: true,
                backup_interval_hours: 24,
                retention_days: 365,
            },
            network: NetworkConfig {
                tcp_nodelay: true,
                tcp_keepalive: true,
                so_reuseaddr: true,
                so_reuseport: true,
                send_buffer_size: 64 * 1024,
                recv_buffer_size: 64 * 1024,
                max_frame_size: 16 * 1024 * 1024,
                connection_timeout_ms: 5000,
                read_timeout_ms: 30000,
                write_timeout_ms: 30000,
            },
            database: DatabaseConfig::default(),
            monitoring: MonitoringConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

impl Configuration for AppConfig {
    const CONFIG_NAME: &'static str = "app";
    
    fn validate(&self) -> Result<()> {
        // Validate server configuration
        if self.server.port == 0 {
            return Err(AlgoVedaError::Config("Invalid server port".to_string()));
        }
        
        if self.server.max_connections == 0 {
            return Err(AlgoVedaError::Config("Max connections must be greater than 0".to_string()));
        }
        
        // Validate performance configuration
        if let Some(ref cpu_affinity) = self.performance.cpu_affinity {
            let max_cpus = num_cpus::get();
            for &cpu in cpu_affinity {
                if cpu >= max_cpus {
                    return Err(AlgoVedaError::Config(
                        format!("CPU affinity {} exceeds available CPUs ({})", cpu, max_cpus)
                    ));
                }
            }
        }
        
        // Validate storage configuration
        if self.storage.max_memory_usage_gb == 0 {
            return Err(AlgoVedaError::Config("Max memory usage must be greater than 0".to_string()));
        }
        
        // Validate market data configuration
        if self.market_data.buffer_size == 0 {
            return Err(AlgoVedaError::Config("Market data buffer size must be greater than 0".to_string()));
        }
        
        // Validate nested configurations
        self.database.validate()?;
        self.monitoring.validate()?;
        self.security.validate()?;
        
        Ok(())
    }
}

impl AppConfig {
    /// Get request timeout as Duration
    pub fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.server.request_timeout_ms)
    }
    
    /// Get connection timeout as Duration
    pub fn connection_timeout(&self) -> Duration {
        Duration::from_millis(self.network.connection_timeout_ms)
    }
    
    /// Check if running in development mode
    pub fn is_development(&self) -> bool {
        self.app.version.contains("dev") || 
        self.app.version.contains("alpha") || 
        self.app.version.contains("beta")
    }
    
    /// Get the number of worker threads to use
    pub fn worker_threads(&self) -> usize {
        if self.server.worker_threads == 0 {
            num_cpus::get()
        } else {
            self.server.worker_threads
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = AppConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_port_validation() {
        let mut config = AppConfig::default();
        config.server.port = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_worker_threads() {
        let config = AppConfig::default();
        assert!(config.worker_threads() > 0);
    }
}
