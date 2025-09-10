/*!
 * Comprehensive Error Handling for AlgoVeda Trading Platform
 */

use std::fmt;
use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum AlgoVedaError {
    #[error("Trading error: {message}")]
    Trading { message: String, code: Option<String> },
    
    #[error("Market data error: {message}")]
    MarketData { message: String, source: Option<String> },
    
    #[error("Risk management error: {message}")]
    Risk { message: String, severity: RiskSeverity },
    
    #[error("Order error: {message}")]
    Order { message: String, order_id: Option<String> },
    
    #[error("Calculation error: {message}")]
    Calculation { message: String, symbol: Option<String> },
    
    #[error("Configuration error: {message}")]
    Configuration { message: String, field: Option<String> },
    
    #[error("Network error: {message}")]
    Network { message: String, retry_count: u32 },
    
    #[error("Database error: {message}")]
    Database { message: String, query: Option<String> },
    
    #[error("Authentication error: {message}")]
    Authentication { message: String },
    
    #[error("Authorization error: {message}")]
    Authorization { message: String, resource: Option<String> },
    
    #[error("Validation error: {message}")]
    Validation { message: String, field: Option<String> },
    
    #[error("System error: {message}")]
    System { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

pub type Result<T> = std::result::Result<T, AlgoVedaError>;

impl AlgoVedaError {
    pub fn trading(message: impl Into<String>) -> Self {
        Self::Trading {
            message: message.into(),
            code: None,
        }
    }

    pub fn risk_critical(message: impl Into<String>) -> Self {
        Self::Risk {
            message: message.into(),
            severity: RiskSeverity::Critical,
        }
    }

    pub fn validation(message: impl Into<String>, field: Option<String>) -> Self {
        Self::Validation {
            message: message.into(),
            field,
        }
    }
}
