/*!
 * AlgoVeda Trading Platform - Core Library Exports
 * 
 * This module provides the main public API for the AlgoVeda trading platform,
 * exposing all major components and functionality for external integration.
 * 
 * Features:
 * - Comprehensive trading system components
 * - Real-time market data processing
 * - Advanced backtesting framework
 * - Machine learning integration
 * - GPU-accelerated calculations
 * - Multi-broker connectivity
 * - Advanced risk management
 */

#![deny(unsafe_code)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

// Core configuration system
pub mod config {
    pub mod app_config;
    pub mod trading_config;
    pub mod risk_config;
    pub mod database_config;
    pub mod monitoring_config;
    pub mod security_config;
}

// Complete trading system
pub mod trading {
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
    
    // Algorithmic execution strategies
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
    
    // Real-time risk engine
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
    
    // Portfolio management
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
    
    // Trading analytics
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
}

// Ultra-fast market data system
pub mod market_data {
    pub mod ultra_fast_feed_handler;
    pub mod multicast_receiver;
    pub mod level1_processor;
    pub mod level2_book;
    pub mod level3_book;
    pub mod tick_processor;
    pub mod bar_generator;
    pub mod options_chain_manager;
    pub mod futures_chain_manager;
    pub mod volatility_surface_builder;
    pub mod term_structure_builder;
    pub mod correlation_engine;
    pub mod market_microstructure;
    pub mod order_flow_analyzer;
    pub mod liquidity_tracker;
    pub mod market_regime_detector;
    pub mod anomaly_detector;
    pub mod news_impact_analyzer;
    pub mod economic_calendar;
    pub mod earnings_tracker;
    pub mod dividend_tracker;
    pub mod corporate_actions_tracker;
    
    // Alternative data sources
    pub mod alternative_data {
        pub mod news_processor;
        pub mod sentiment_analyzer;
        pub mod social_media_feed;
        pub mod satellite_data_processor;
        pub mod supply_chain_monitor;
        pub mod patent_data_analyzer;
        pub mod insider_trading_tracker;
        pub mod analyst_revision_tracker;
        pub mod credit_rating_monitor;
        pub mod esg_data_processor;
        pub mod weather_data_processor;
        pub mod economic_indicator_tracker;
        pub mod crypto_sentiment_analyzer;
    }
}

// Comprehensive backtesting framework
pub mod backtesting_engine {
    pub mod vectorized_backtester;
    pub mod event_driven_backtester;
    pub mod multi_asset_backtester;
    pub mod options_backtester;
    pub mod portfolio_backtester;
    pub mod walk_forward_analyzer;
    pub mod monte_carlo_simulator;
    pub mod bootstrap_simulator;
    pub mod sensitivity_analyzer;
    pub mod scenario_backtester;
    pub mod stress_backtester;
    
    // Parameter optimization
    pub mod parameter_optimizer {
        pub mod genetic_algorithm;
        pub mod particle_swarm;
        pub mod differential_evolution;
        pub mod bayesian_optimization;
        pub mod simulated_annealing;
        pub mod grid_search;
        pub mod random_search;
        pub mod hyperband;
        pub mod optuna_integration;
        pub mod multi_objective_optimizer;
    }
    
    // Performance analysis
    pub mod performance_analyzer {
        pub mod returns_analyzer;
        pub mod drawdown_analyzer;
        pub mod risk_metrics_analyzer;
        pub mod trade_analyzer;
        pub mod consistency_analyzer;
        pub mod benchmark_analyzer;
        pub mod factor_attribution;
        pub mod regime_analysis;
        pub mod seasonal_analysis;
        pub mod rolling_metrics;
    }
    
    // Validation framework
    pub mod validation {
        pub mod cross_validator;
        pub mod time_series_validator;
        pub mod overfitting_detector;
        pub mod data_snooping_detector;
        pub mod multiple_testing_correction;
        pub mod statistical_significance;
        pub mod white_reality_check;
        pub mod hansen_spa_test;
        pub mod bootstrap_validator;
        pub mod permutation_test;
    }
    
    // Reporting system
    pub mod reporting {
        pub mod performance_report;
        pub mod risk_report;
        pub mod trade_report;
        pub mod attribution_report;
        pub mod compliance_report;
        pub mod tear_sheet_generator;
        pub mod interactive_report;
        pub mod pdf_generator;
        pub mod excel_generator;
        pub mod web_report_server;
    }
}

// Real-time visualization
pub mod visualization_engine {
    pub mod real_time_aggregator;
    pub mod performance_calculator;
    pub mod risk_calculator;
    pub mod correlation_calculator;
    pub mod volatility_calculator;
    pub mod attribution_calculator;
    pub mod benchmark_calculator;
    pub mod drawdown_calculator;
    pub mod greeks_calculator;
    pub mod factor_calculator;
    pub mod sector_calculator;
    pub mod geographic_calculator;
    pub mod currency_calculator;
    pub mod liquidity_calculator;
    pub mod microstructure_calculator;
}

// Machine learning system
pub mod machine_learning {
    // Feature engineering
    pub mod feature_engineering {
        pub mod technical_features;
        pub mod fundamental_features;
        pub mod alternative_features;
        pub mod sentiment_features;
        pub mod macro_features;
        pub mod cross_asset_features;
        pub mod time_features;
        pub mod volatility_features;
        pub mod correlation_features;
        pub mod momentum_features;
        pub mod mean_reversion_features;
        pub mod microstructure_features;
        pub mod feature_selector;
    }
    
    // Model implementations
    pub mod models {
        pub mod linear_models;
        pub mod tree_models;
        pub mod ensemble_models;
        pub mod neural_networks;
        pub mod deep_learning;
        pub mod time_series_models;
        pub mod reinforcement_learning;
        pub mod clustering_models;
        pub mod dimensionality_reduction;
        pub mod anomaly_detection;
        pub mod causal_models;
        pub mod bayesian_models;
    }
    
    // Training framework
    pub mod training {
        pub mod model_trainer;
        pub mod hyperparameter_tuner;
        pub mod cross_validator;
        pub mod early_stopping;
        pub mod model_selector;
        pub mod ensemble_trainer;
        pub mod online_learning;
        pub mod transfer_learning;
        pub mod meta_learning;
        pub mod federated_learning;
    }
    
    // Prediction system
    pub mod prediction {
        pub mod prediction_engine;
        pub mod ensemble_predictor;
        pub mod uncertainty_quantifier;
        pub mod prediction_validator;
        pub mod confidence_intervals;
        pub mod prediction_calibration;
        pub mod multi_step_predictor;
        pub mod probabilistic_predictor;
        pub mod streaming_predictor;
    }
    
    // Model evaluation
    pub mod evaluation {
        pub mod model_evaluator;
        pub mod feature_importance;
        pub mod model_explainer;
        pub mod drift_detector;
        pub mod performance_monitor;
        pub mod a_b_tester;
        pub mod champion_challenger;
        pub mod model_governance;
    }
}

// DhanHQ integration system
pub mod dhan_integration {
    pub mod enhanced_api_client;
    pub mod connection_manager;
    pub mod rate_limiter;
    pub mod circuit_breaker;
    pub mod retry_handler;
    pub mod load_balancer;
    pub mod failover_manager;
    pub mod order_service_complete;
    pub mod super_order_service;
    pub mod forever_order_service;
    pub mod trade_service;
    pub mod portfolio_service;
    pub mod market_data_service;
    pub mod funds_service;
    pub mod edis_service;
    pub mod risk_service;
    pub mod statements_service;
    pub mod charts_service;
    pub mod websocket_handler;
    pub mod streaming_processor;
    pub mod authentication_manager;
    pub mod session_manager;
    pub mod error_handler;
    pub mod logging_interceptor;
    pub mod metrics_collector;
    pub mod health_checker;
}

// Financial calculations
pub mod calculations {
    // Options pricing
    pub mod options_pricing {
        pub mod black_scholes_gpu;
        pub mod black_scholes_merton;
        pub mod binomial_tree_cuda;
        pub mod trinomial_tree;
        pub mod monte_carlo_gpu;
        pub mod finite_difference;
        pub mod heston_model;
        pub mod sabr_model;
        pub mod local_volatility;
        pub mod stochastic_volatility;
        pub mod jump_diffusion;
        pub mod levy_processes;
        pub mod exotic_options;
        pub mod barrier_options;
        pub mod asian_options;
        pub mod lookback_options;
        pub mod american_options;
        pub mod bermudan_options;
        pub mod compound_options;
    }
    
    // Greeks calculations
    pub mod greeks {
        pub mod analytical_greeks;
        pub mod numerical_greeks;
        pub mod portfolio_greeks;
        pub mod cross_greeks;
        pub mod higher_order_greeks;
        pub mod gamma_scalping;
        pub mod vega_hedging;
        pub mod theta_analysis;
        pub mod rho_sensitivity;
        pub mod volga_calculations;
        pub mod vanna_calculations;
        pub mod charm_calculations;
        pub mod speed_calculations;
    }
    
    // Volatility models
    pub mod volatility {
        pub mod implied_volatility;
        pub mod historical_volatility;
        pub mod realized_volatility;
        pub mod garch_models;
        pub mod egarch_models;
        pub mod gjr_garch;
        pub mod figarch_models;
        pub mod stochastic_volatility;
        pub mod volatility_surface;
        pub mod vol_smile_calibration;
        pub mod vol_clustering;
        pub mod vol_forecasting;
        pub mod vol_regime_switching;
        pub mod vol_term_structure;
        pub mod vol_surface_dynamics;
    }
    
    // Risk metrics
    pub mod risk_metrics {
        pub mod var_models_advanced;
        pub mod expected_shortfall;
        pub mod conditional_var;
        pub mod spectral_risk_measures;
        pub mod coherent_risk_measures;
        pub mod maximum_drawdown;
        pub mod calmar_ratio;
        pub mod sterling_ratio;
        pub mod burke_ratio;
        pub mod ulcer_index;
        pub mod pain_index;
        pub mod tail_measures;
        pub mod extreme_value_theory;
        pub mod copula_risk_measures;
        pub mod stress_testing_metrics;
    }
    
    // Performance metrics
    pub mod performance_metrics {
        pub mod return_metrics;
        pub mod risk_adjusted_returns;
        pub mod sharpe_ratio_variants;
        pub mod sortino_ratio;
        pub mod treynor_ratio;
        pub mod information_ratio;
        pub mod omega_ratio;
        pub mod kappa_ratios;
        pub mod rachev_ratio;
        pub mod gain_loss_ratio;
        pub mod profit_factor;
        pub mod recovery_factor;
        pub mod payoff_ratio;
        pub mod consistency_metrics;
    }
    
    // Portfolio analytics
    pub mod portfolio_analytics {
        pub mod performance_attribution;
        pub mod brinson_attribution;
        pub mod factor_attribution;
        pub mod fama_french_analysis;
        pub mod carhart_four_factor;
        pub mod arbitrage_pricing_theory;
        pub mod capm_analysis;
        pub mod style_analysis;
        pub mod holdings_analysis;
        pub mod tracking_error;
        pub mod active_share;
        pub mod concentration_metrics;
        pub mod diversification_ratio;
        pub mod effective_bets;
    }
}

// High-performance storage
pub mod storage {
    pub mod ultra_fast_storage;
    pub mod memory_mapped_files;
    pub mod columnar_storage;
    pub mod compression_engine;
    pub mod distributed_cache;
    pub mod sharded_storage;
    pub mod replication_manager;
    pub mod consistency_manager;
    pub mod versioning_system;
    pub mod backup_manager;
    pub mod archival_system;
    pub mod data_lake;
    pub mod object_store;
    pub mod blockchain_storage;
}

// Ultra-low latency networking
pub mod networking {
    pub mod ultra_low_latency;
    pub mod kernel_bypass;
    pub mod rdma_transport;
    pub mod tcp_optimized;
    pub mod udp_multicast;
    pub mod websocket_server;
    pub mod message_router;
    pub mod load_balancer;
    pub mod connection_pooling;
    pub mod ssl_tls_handler;
    pub mod compression_handler;
    pub mod encryption_handler;
    pub mod monitoring_agent;
}

// Comprehensive monitoring
pub mod monitoring {
    pub mod metrics_collector;
    pub mod latency_tracker;
    pub mod throughput_monitor;
    pub mod resource_monitor;
    pub mod performance_profiler;
    pub mod memory_profiler;
    pub mod cpu_profiler;
    pub mod network_profiler;
    pub mod disk_profiler;
    pub mod alert_manager;
    pub mod anomaly_detector;
    pub mod health_checker;
    pub mod sla_monitor;
    pub mod dashboard_generator;
}

// Security system
pub mod security {
    pub mod encryption;
    pub mod authentication;
    pub mod authorization;
    pub mod audit_logger;
    pub mod access_control;
    pub mod key_management;
    pub mod certificate_manager;
    pub mod secure_communication;
    pub mod intrusion_detection;
    pub mod vulnerability_scanner;
    pub mod threat_detector;
    pub mod incident_response;
    pub mod compliance_monitor;
}

// Utility functions
pub mod utils {
    pub mod constants;
    pub mod formatting;
    pub mod calculations;
    pub mod validators;
    pub mod helpers;
    pub mod date_time;
    pub mod error_handling;
    pub mod logging;
    pub mod configuration;
    pub mod serialization;
    pub mod compression;
    pub mod hashing;
    pub mod random_number_generator;
}

// Re-export main types for convenience
pub use config::*;
pub use trading::{order_manager::*, execution_engine::*, risk_engine::*};
pub use market_data::{ultra_fast_feed_handler::*, level2_book::*};
pub use backtesting_engine::{vectorized_backtester::*, performance_analyzer::*};
pub use calculations::{options_pricing::*, greeks::*, volatility::*};
pub use dhan_integration::{enhanced_api_client::*, websocket_handler::*};
pub use machine_learning::{feature_engineering::*, models::*, prediction::*};
pub use utils::*;

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

// Feature flags
#[cfg(feature = "gpu")]
pub mod gpu {
    pub use crate::calculations::options_pricing::black_scholes_gpu;
    pub use crate::calculations::options_pricing::monte_carlo_gpu;
    pub use crate::calculations::options_pricing::binomial_tree_cuda;
}

#[cfg(feature = "machine_learning")]
pub use machine_learning::*;

#[cfg(feature = "backtesting")]
pub use backtesting_engine::*;

#[cfg(feature = "visualization")]
pub use visualization_engine::*;

// Platform information
pub fn platform_info() -> PlatformInfo {
    PlatformInfo {
        version: VERSION,
        build_date: env!("BUILD_DATE"),
        git_hash: env!("GIT_HASH"),
        features: get_enabled_features(),
        target_arch: std::env::consts::ARCH,
        target_os: std::env::consts::OS,
    }
}

#[derive(Debug, Clone)]
pub struct PlatformInfo {
    pub version: &'static str,
    pub build_date: &'static str,
    pub git_hash: &'static str,
    pub features: Vec<&'static str>,
    pub target_arch: &'static str,
    pub target_os: &'static str,
}

fn get_enabled_features() -> Vec<&'static str> {
    let mut features = vec!["core"];
    
    #[cfg(feature = "gpu")]
    features.push("gpu");
    
    #[cfg(feature = "machine_learning")]
    features.push("machine_learning");
    
    #[cfg(feature = "backtesting")]
    features.push("backtesting");
    
    #[cfg(feature = "visualization")]
    features.push("visualization");
    
    #[cfg(feature = "dhan_integration")]
    features.push("dhan_integration");
    
    features
}

// Error types
pub mod errors {
    use thiserror::Error;
    
    #[derive(Error, Debug)]
    pub enum AlgoVedaError {
        #[error("Trading error: {0}")]
        Trading(String),
        #[error("Market data error: {0}")]
        MarketData(String),
        #[error("Risk management error: {0}")]
        Risk(String),
        #[error("Calculation error: {0}")]
        Calculation(String),
        #[error("Configuration error: {0}")]
        Configuration(String),
        #[error("Network error: {0}")]
        Network(String),
        #[error("Storage error: {0}")]
        Storage(String),
        #[error("Security error: {0}")]
        Security(String),
        #[error("Internal error: {0}")]
        Internal(String),
    }
}

pub use errors::AlgoVedaError;

// Result type alias
pub type Result<T> = std::result::Result<T, AlgoVedaError>;

// Prelude module for common imports
pub mod prelude {
    pub use crate::{
        trading::{order_manager::*, execution_engine::*},
        market_data::ultra_fast_feed_handler::*,
        calculations::options_pricing::*,
        utils::*,
        Result,
        AlgoVedaError,
    };
}
