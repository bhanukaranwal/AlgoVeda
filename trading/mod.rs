//! Trading system module exports

pub mod order_manager;
pub mod execution_engine;
pub mod smart_order_router;
pub mod liquidity_aggregator;
pub mod market_maker;
pub mod dark_pool_connector;
pub mod cross_connect_engine;
pub mod order_book_manager;
pub mod trade_matcher;
pub mod settlement_engine;
pub mod regulatory_compliance;

// Algorithmic execution submodules
pub mod algo_execution {
    pub mod twap_engine;
    pub mod vwap_engine;
    pub mod implementation_shortfall;
    pub mod participation_rate;
    pub mod iceberg_execution;
    pub mod sniper_execution;
    pub mod guerrilla_execution;
    pub mod stealth_execution;
    pub mod predatory_execution;
    pub mod adaptive_execution;
    pub mod arrival_price;
    pub mod close_price;
    pub mod pov_execution;
    pub mod custom_algo_framework;
}

// Risk engine submodules
pub mod risk_engine {
    pub mod pre_trade_risk;
    pub mod real_time_risk;
    pub mod post_trade_risk;
    pub mod portfolio_risk;
    pub mod concentration_limits;
    pub mod var_calculator;
    pub mod expected_shortfall;
    pub mod stress_tester;
    pub mod scenario_analyzer;
    pub mod tail_risk_analyzer;
    pub mod correlation_monitor;
    pub mod liquidity_risk;
    pub mod counterparty_risk;
    pub mod operational_risk;
    pub mod model_risk;
    pub mod credit_risk;
    pub mod market_risk;
    pub mod regulatory_capital;
    pub mod risk_reporting;
}

// Portfolio management submodules
pub mod portfolio {
    pub mod position_manager;
    pub mod mtm_engine;
    pub mod pnl_calculator;
    pub mod attribution_engine;
    pub mod performance_analytics;
    pub mod benchmark_comparison;
    pub mod style_analysis;
    pub mod factor_analysis;
    pub mod sector_allocation;
    pub mod geographic_exposure;
    pub mod currency_exposure;
    pub mod asset_allocation;
    pub mod rebalancing_engine;
    pub mod tax_optimization;
    pub mod cash_management;
    pub mod corporate_actions;
    pub mod esg_analytics;
}

// Analytics submodules
pub mod analytics {
    pub mod trade_cost_analysis;
    pub mod execution_quality;
    pub mod market_impact_model;
    pub mod liquidity_analysis;
    pub mod alpha_generation;
    pub mod beta_analysis;
    pub mod factor_modeling;
    pub mod regime_detection;
    pub mod volatility_forecasting;
    pub mod correlation_forecasting;
    pub mod return_prediction;
    pub mod momentum_analysis;
    pub mod mean_reversion_analysis;
    pub mod arbitrage_detector;
}

// Re-export commonly used types
pub use order_manager::{OrderManager, Order, OrderSide, OrderType, TimeInForce};
pub use execution_engine::{ExecutionEngine, ExecutionReport, Fill};
pub use risk_engine::real_time_risk::{RiskEngine, RiskCheck, RiskViolation};
pub use portfolio::position_manager::{PositionManager, Position};

// Common trading types and enums
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketSession {
    PreMarket,
    RegularHours,
    PostMarket,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrumentType {
    Equity,
    Option,
    Future,
    Bond,
    Forex,
    Crypto,
    Commodity,
}

#[derive(Debug, Clone)]
pub struct MarketData {
    pub symbol: String,
    pub price: f64,
    pub volume: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    pub bid_size: Option<u64>,
    pub ask_size: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TradingLimits {
    pub max_order_value: f64,
    pub max_position_size: f64,
    pub max_daily_volume: f64,
    pub max_trades_per_day: u32,
}

// Error types for trading operations
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Invalid order: {0}")]
    InvalidOrder(String),
    #[error("Risk check failed: {0}")]
    RiskCheckFailed(String),
    #[error("Insufficient funds: required {required}, available {available}")]
    InsufficientFunds { required: f64, available: f64 },
    #[error("Market closed for symbol: {0}")]
    MarketClosed(String),
    #[error("Position limit exceeded")]
    PositionLimitExceeded,
    #[error("Order execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Settlement failed: {0}")]
    SettlementFailed(String),
    #[error("Regulatory compliance violation: {0}")]
    ComplianceViolation(String),
}

pub type TradingResult<T> = std::result::Result<T, TradingError>;
