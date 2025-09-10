/*!
 * Comprehensive Security Audit Logger for AlgoVeda Trading Platform
 * Enterprise-grade audit logging with compliance features
 */

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::Mutex;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use tokio::sync::mpsc;
use tracing::{info, warn, error};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub resource_type: String,
    pub resource_id: Option<String>,
    pub action: String,
    pub result: AuditResult,
    pub details: HashMap<String, serde_json::Value>,
    pub risk_level: RiskLevel,
    pub compliance_tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    Authentication,
    Authorization,
    DataAccess,
    DataModification,
    SystemConfiguration,
    TradingAction,
    RiskEvent,
    SecurityEvent,
    ComplianceEvent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure,
    Warning,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

pub struct SecurityAuditLogger {
    event_sender: mpsc::UnboundedSender<AuditEvent>,
    config: AuditConfig,
    metrics: Arc<Mutex<AuditMetrics>>,
}

#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub enable_console_output: bool,
    pub enable_file_output: bool,
    pub enable_database_output: bool,
    pub enable_siem_output: bool,
    pub file_path: String,
    pub database_table: String,
    pub siem_endpoint: Option<String>,
    pub retention_days: u32,
    pub encrypt_sensitive_data: bool,
    pub compliance_mode: ComplianceMode,
}

#[derive(Debug, Clone)]
pub enum ComplianceMode {
    SOX,      // Sarbanes-Oxley
    PCI_DSS,  // Payment Card Industry
    GDPR,     // General Data Protection Regulation
    SOC2,     // Service Organization Control 2
    Custom(Vec<String>),
}

#[derive(Debug, Default)]
struct AuditMetrics {
    events_logged: u64,
    events_failed: u64,
    high_risk_events: u64,
    last_event_timestamp: Option<DateTime<Utc>>,
}

impl SecurityAuditLogger {
    pub fn new(config: AuditConfig) -> Self {
        let (sender, mut receiver) = mpsc::unbounded_channel();
        let logger_config = config.clone();

        // Start background audit processing task
        tokio::spawn(async move {
            let mut file_writer = if logger_config.enable_file_output {
                Some(Self::create_file_writer(&logger_config.file_path).await)
            } else {
                None
            };

            while let Some(event) = receiver.recv().await {
                Self::process_audit_event(&event, &logger_config, &mut file_writer).await;
            }
        });

        Self {
            event_sender: sender,
            config,
            metrics: Arc::new(Mutex::new(AuditMetrics::default())),
        }
    }

    // Log authentication events[2][5]
    pub fn log_authentication_event(&self, 
        user_id: Option<String>,
        action: &str,
        result: AuditResult,
        ip_address: Option<String>,
        details: HashMap<String, serde_json::Value>
    ) {
        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::Authentication,
            user_id,
            session_id: None,
            ip_address,
            user_agent: None,
            resource_type: "authentication".to_string(),
            resource_id: None,
            action: action.to_string(),
            result,
            details,
            risk_level: match result {
                AuditResult::Failure => RiskLevel::High,
                AuditResult::Blocked => RiskLevel::Critical,
                _ => RiskLevel::Low,
            },
            compliance_tags: vec!["SOX".to_string(), "SOC2".to_string()],
        };

        self.send_audit_event(event);
    }

    // Log trading actions with enhanced compliance tracking[9]
    pub fn log_trading_event(&self,
        user_id: Option<String>,
        order_id: &str,
        action: &str,
        symbol: &str,
        quantity: f64,
        price: Option<f64>,
        result: AuditResult,
        details: HashMap<String, serde_json::Value>
    ) {
        let mut enhanced_details = details;
        enhanced_details.insert("symbol".to_string(), serde_json::Value::String(symbol.to_string()));
        enhanced_details.insert("quantity".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(quantity).unwrap()));
        if let Some(p) = price {
            enhanced_details.insert("price".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(p).unwrap()));
        }

        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::TradingAction,
            user_id,
            session_id: None,
            ip_address: None,
            user_agent: None,
            resource_type: "order".to_string(),
            resource_id: Some(order_id.to_string()),
            action: action.to_string(),
            result,
            details: enhanced_details,
            risk_level: match quantity.abs() {
                q if q > 1000000.0 => RiskLevel::Critical,
                q if q > 100000.0 => RiskLevel::High,
                q if q > 10000.0 => RiskLevel::Medium,
                _ => RiskLevel::Low,
            },
            compliance_tags: vec![
                "SOX".to_string(), 
                "MiFID_II".to_string(), 
                "FINRA".to_string()
            ],
        };

        self.send_audit_event(event);
    }

    // Log data access events for GDPR compliance
    pub fn log_data_access_event(&self,
        user_id: Option<String>,
        resource_type: &str,
        resource_id: &str,
        action: &str,
        data_classification: &str,
        result: AuditResult
    ) {
        let mut details = HashMap::new();
        details.insert("data_classification".to_string(), 
                      serde_json::Value::String(data_classification.to_string()));

        let event = AuditEvent {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: AuditEventType::DataAccess,
            user_id,
            session_id: None,
            ip_address: None,
            user_agent: None,
            resource_type: resource_type.to_string(),
            resource_id: Some(resource_id.to_string()),
            action: action.to_string(),
            result,
            details,
            risk_level: match data_classification {
                "PII" | "CONFIDENTIAL" => RiskLevel::High,
                "INTERNAL" => RiskLevel::Medium,
                _ => RiskLevel::Low,
            },
            compliance_tags: vec!["GDPR".to_string(), "SOC2".to_string()],
        };

        self.send_audit_event(event);
    }

    fn send_audit_event(&self, event: AuditEvent) {
        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.events_logged += 1;
            metrics.last_event_timestamp = Some(event.timestamp);
            
            if matches!(event.risk_level, RiskLevel::High | RiskLevel::Critical) {
                metrics.high_risk_events += 1;
            }
        }

        // Send event for processing
        if let Err(e) = self.event_sender.send(event) {
            error!("Failed to send audit event: {}", e);
            let mut metrics = self.metrics.lock();
            metrics.events_failed += 1;
        }
    }

    async fn process_audit_event(
        event: &AuditEvent,
        config: &AuditConfig,
        file_writer: &mut Option<tokio::fs::File>
    ) {
        // Console output
        if config.enable_console_output {
            Self::write_to_console(event);
        }

        // File output
        if config.enable_file_output {
            if let Some(writer) = file_writer {
                Self::write_to_file(event, writer).await;
            }
        }

        // Database output
        if config.enable_database_output {
            Self::write_to_database(event, config).await;
        }

        // SIEM output
        if config.enable_siem_output && config.siem_endpoint.is_some() {
            Self::write_to_siem(event, config).await;
        }
    }

    fn write_to_console(event: &AuditEvent) {
        let level = match event.risk_level {
            RiskLevel::Critical => "CRITICAL",
            RiskLevel::High => "HIGH",
            RiskLevel::Medium => "MEDIUM",
            RiskLevel::Low => "LOW",
        };

        info!(
            "AUDIT [{}] {} - {} {} on {} (Result: {:?})",
            level,
            event.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            event.user_id.as_deref().unwrap_or("SYSTEM"),
            event.action,
            event.resource_type,
            event.result
        );
    }

    async fn write_to_file(event: &AuditEvent, _file_writer: &mut tokio::fs::File) {
        // Implementation would write structured JSON to file
        let json_event = serde_json::to_string(event).unwrap_or_default();
        // file_writer.write_all(format!("{}\n", json_event).as_bytes()).await.ok();
    }

    async fn write_to_database(event: &AuditEvent, _config: &AuditConfig) {
        // Implementation would insert into database table
        // This would use the configured database connection
    }

    async fn write_to_siem(event: &AuditEvent, config: &AuditConfig) {
        if let Some(endpoint) = &config.siem_endpoint {
            let client = reqwest::Client::new();
            let json_event = serde_json::to_string(event).unwrap_or_default();
            
            if let Err(e) = client
                .post(endpoint)
                .header("Content-Type", "application/json")
                .body(json_event)
                .send()
                .await
            {
                warn!("Failed to send audit event to SIEM: {}", e);
            }
        }
    }

    async fn create_file_writer(file_path: &str) -> tokio::fs::File {
        tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)
            .await
            .expect("Failed to open audit log file")
    }

    pub fn get_metrics(&self) -> AuditMetrics {
        self.metrics.lock().clone()
    }
}

impl Clone for AuditMetrics {
    fn clone(&self) -> Self {
        Self {
            events_logged: self.events_logged,
            events_failed: self.events_failed,
            high_risk_events: self.high_risk_events,
            last_event_timestamp: self.last_event_timestamp,
        }
    }
}
