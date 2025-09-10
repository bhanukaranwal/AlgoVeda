use serde::{Deserialize, Serialize};
use config::ConfigError;
use super::ConfigLoader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub metrics_port: u16,
    pub metrics_path: String,
    pub enable_tracing: bool,
    pub enable_logging: bool,
    pub log_level: String,
    pub enable_database_metrics: bool,
    pub enable_trading_metrics: bool,
    pub enable_risk_metrics: bool,
    pub prometheus: PrometheusConfig,
    pub jaeger: JaegerConfig,
    pub alerting: AlertingConfig,
    pub profiling: ProfilingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusConfig {
    pub endpoint: String,
    pub scrape_interval: u64,
    pub retention: String,
    pub external_labels: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerConfig {
    pub endpoint: String,
    pub service_name: String,
    pub sampling_rate: f64,
    pub max_packet_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub enable_alerts: bool,
    pub webhook_url: String,
    pub email_recipients: Vec<String>,
    pub slack_webhook: Option<String>,
    pub alert_rules: Vec<AlertRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub condition: String,
    pub threshold: f64,
    pub duration: u64,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    pub enable_cpu_profiling: bool,
    pub enable_memory_profiling: bool,
    pub enable_heap_profiling: bool,
    pub profile_duration: u64,
    pub profile_interval: u64,
}

impl ConfigLoader for MonitoringConfig {
    fn load(_path: Option<&str>) -> Result<Self, ConfigError> {
        Ok(Self::default())
    }
    
    fn validate(&self) -> Result<(), ConfigError> {
        if self.metrics_port == 0 {
            return Err(ConfigError::Message("Metrics port must be specified".to_string()));
        }
        
        if self.jaeger.sampling_rate < 0.0 || self.jaeger.sampling_rate > 1.0 {
            return Err(ConfigError::Message("Jaeger sampling rate must be between 0 and 1".to_string()));
        }
        
        Ok(())
    }
    
    fn merge(&mut self, _other: Self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_port: 9090,
            metrics_path: "/metrics".to_string(),
            enable_tracing: true,
            enable_logging: true,
            log_level: "info".to_string(),
            enable_database_metrics: true,
            enable_trading_metrics: true,
            enable_risk_metrics: true,
            prometheus: PrometheusConfig::default(),
            jaeger: JaegerConfig::default(),
            alerting: AlertingConfig::default(),
            profiling: ProfilingConfig::default(),
        }
    }
}

impl Default for PrometheusConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:9090".to_string(),
            scrape_interval: 15,
            retention: "15d".to_string(),
            external_labels: std::collections::HashMap::new(),
        }
    }
}

impl Default for JaegerConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://localhost:14268/api/traces".to_string(),
            service_name: "algoveda".to_string(),
            sampling_rate: 0.01,
            max_packet_size: 65000,
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enable_alerts: true,
            webhook_url: "".to_string(),
            email_recipients: vec![],
            slack_webhook: None,
            alert_rules: vec![
                AlertRule {
                    name: "High CPU Usage".to_string(),
                    condition: "cpu_usage > 80".to_string(),
                    threshold: 80.0,
                    duration: 300,
                    severity: "warning".to_string(),
                },
                AlertRule {
                    name: "Trading System Down".to_string(),
                    condition: "trading_system_up == 0".to_string(),
                    threshold: 0.0,
                    duration: 60,
                    severity: "critical".to_string(),
                },
            ],
        }
    }
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enable_cpu_profiling: false,
            enable_memory_profiling: false,
            enable_heap_profiling: false,
            profile_duration: 60,
            profile_interval: 300,
        }
    }
}
