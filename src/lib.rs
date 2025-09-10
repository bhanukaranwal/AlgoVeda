/*!
 * AlgoVeda Core Library
 * High-performance algorithmic trading platform library
 */

#![deny(unsafe_code)]
#![warn(
    missing_docs,
    rust_2018_idioms,
    unreachable_pub,
    clippy::all,
    clippy::pedantic,
    clippy::nursery
)]

//! # AlgoVeda Core
//!
//! AlgoVeda Core is an ultra-high performance algorithmic trading platform
//! designed for institutional-grade trading with sub-microsecond latency.
//!
//! ## Features
//!
//! - Ultra-low latency order execution
//! - Real-time risk management
//! - Multi-venue connectivity
//! - Advanced portfolio analytics
//! - GPU-accelerated calculations
//! - Machine learning integration
//!
//! ## Architecture
//!
//! The platform is built using a modular architecture with the following components:
//!
//! - **Trading Engine**: Core order management and execution
//! - **Risk Engine**: Real-time risk monitoring and controls
//! - **Market Data**: Ultra-fast market data processing
//! - **Portfolio**: Position and P&L management
//! - **Analytics**: Performance and risk analytics
//! - **ML Engine**: Machine learning models and predictions

pub mod config;
pub mod trading;
pub mod market_data;
pub mod portfolio;
pub mod risk_engine;
pub mod analytics;
pub mod machine_learning;
pub mod calculations;
pub mod storage;
pub mod networking;
pub mod monitoring;
pub mod security;
pub mod utils;
pub mod backtesting_engine;
pub mod visualization_engine;
pub mod dhan_integration;

// Re-export commonly used types
pub use config::{AppConfig, TradingConfig, RiskConfig};
pub use trading::{
    Order, OrderType, OrderSide, OrderStatus,
    OrderManager, ExecutionEngine, SmartOrderRouter,
};
pub use market_data::{
    MarketData, Tick, Quote, Trade,
    UltraFastFeedHandler, Level2Book,
};
pub use portfolio::{
    Position, Portfolio, PortfolioManager,
    MTMEngine, PnLCalculator,
};
pub use risk_engine::{
    RiskEngine, RiskCheck, RiskLimit,
    VaRCalculator, StressTester,
};

// Error handling
pub mod error {
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum AlgoVedaError {
        #[error("Configuration error: {0}")]
        Config(String),
        
        #[error("Trading error: {0}")]
        Trading(String),
        
        #[error("Market data error: {0}")]
        MarketData(String),
        
        #[error("Risk management error: {0}")]
        Risk(String),
        
        #[error("Storage error: {0}")]
        Storage(String),
        
        #[error("Network error: {0}")]
        Network(String),
        
        #[error("Calculation error: {0}")]
        Calculation(String),
        
        #[error("Security error: {0}")]
        Security(String),
        
        #[error("Internal error: {0}")]
        Internal(String),
    }
    
    pub type Result<T> = std::result::Result<T, AlgoVedaError>;
}

// Common traits
pub mod traits {
    use async_trait::async_trait;
    use crate::error::Result;

    #[async_trait]
    pub trait Service {
        async fn start(&self) -> Result<()>;
        async fn stop(&self) -> Result<()>;
        async fn health_check(&self) -> Result<bool>;
    }

    #[async_trait]
    pub trait DataProvider {
        type Data;
        async fn get_data(&self) -> Result<Self::Data>;
        async fn subscribe(&self) -> Result<()>;
        async fn unsubscribe(&self) -> Result<()>;
    }

    pub trait Calculator<T> {
        type Output;
        fn calculate(&self, input: T) -> Result<Self::Output>;
    }

    pub trait Validator<T> {
        fn validate(&self, item: &T) -> Result<()>;
    }
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT_HASH: &str = env!("GIT_HASH");
pub const BUILD_TIMESTAMP: &str = env!("BUILD_TIMESTAMP");

// Platform information
pub fn platform_info() -> String {
    format!(
        "AlgoVeda Core v{} ({}), built at {}",
        VERSION, GIT_HASH, BUILD_TIMESTAMP
    )
}

// Performance monitoring
use std::sync::atomic::{AtomicU64, Ordering};

static ORDERS_PROCESSED: AtomicU64 = AtomicU64::new(0);
static MESSAGES_PROCESSED: AtomicU64 = AtomicU64::new(0);
static CALCULATIONS_PERFORMED: AtomicU64 = AtomicU64::new(0);

pub fn increment_orders_processed() {
    ORDERS_PROCESSED.fetch_add(1, Ordering::Relaxed);
}

pub fn increment_messages_processed() {
    MESSAGES_PROCESSED.fetch_add(1, Ordering::Relaxed);
}

pub fn increment_calculations_performed() {
    CALCULATIONS_PERFORMED.fetch_add(1, Ordering::Relaxed);
}

pub fn get_performance_stats() -> (u64, u64, u64) {
    (
        ORDERS_PROCESSED.load(Ordering::Relaxed),
        MESSAGES_PROCESSED.load(Ordering::Relaxed),
        CALCULATIONS_PERFORMED.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert!(!platform_info().is_empty());
    }

    #[test]
    fn test_performance_counters() {
        let initial_stats = get_performance_stats();
        
        increment_orders_processed();
        increment_messages_processed();
        increment_calculations_performed();
        
        let updated_stats = get_performance_stats();
        
        assert_eq!(updated_stats.0, initial_stats.0 + 1);
        assert_eq!(updated_stats.1, initial_stats.1 + 1);
        assert_eq!(updated_stats.2, initial_stats.2 + 1);
    }
}
