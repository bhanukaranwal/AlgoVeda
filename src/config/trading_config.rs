/*!
 * Trading Configuration
 * Configuration for trading engine and execution parameters
 */

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use rust_decimal::Decimal;
use crate::config::Configuration;
use crate::error::{Result, AlgoVedaError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    /// Global trading settings
    pub global: GlobalTradingConfig,
    
    /// Venue configurations
    pub venues: HashMap<String, VenueConfig>,
    
    /// Order routing configuration
    pub routing: OrderRoutingConfig,
    
    /// Execution algorithms configuration
    pub execution_algos: ExecutionAlgoConfig,
    
    /// Commission and fee configuration
    pub fees: FeeConfig,
    
    /// Position limits
    pub position_limits: PositionLimitsConfig,
    
    /// Trading hours configuration
    pub trading_hours: TradingHoursConfig,
    
    /// Market making configuration
    pub market_making: MarketMakingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalTradingConfig {
    pub enabled: bool,
    pub max_orders_per_second: u32,
    pub max_daily_orders: u32,
    pub max_order_value: Decimal,
    pub min_order_value: Decimal,
    pub default_currency: String,
    pub enable_short_selling: bool,
    pub enable_margin_trading: bool,
    pub enable_options_trading: bool,
    pub enable_futures_trading: bool,
    pub enable_crypto_trading: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueConfig {
    pub name: String,
    pub enabled: bool,
    pub priority: u8,
    pub connection_config: VenueConnectionConfig,
    pub order_types: Vec<String>,
    pub supported_instruments: Vec<String>,
    pub min_order_size: Decimal,
    pub max_order_size: Decimal,
    pub tick_size: Decimal,
    pub lot_size: u32,
    pub trading_fees: VenueFeeConfig,
    pub rate_limits: RateLimitsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueConnectionConfig {
    pub api_url: String,
    pub websocket_url: String,
    pub fix_config: Option<FixConfig>,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub sandbox_mode: bool,
    pub timeout_ms: u64,
    pub max_retries: u32,
    pub heartbeat_interval_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixConfig {
    pub host: String,
    pub port: u16,
    pub sender_comp_id: String,
    pub target_comp_id: String,
    pub version: String,
    pub ssl_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueFeeConfig {
    pub maker_fee_bps: u32,
    pub taker_fee_bps: u32,
    pub withdrawal_fee: Decimal,
    pub currency_conversion_fee_bps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitsConfig {
    pub orders_per_second: u32,
    pub requests_per_minute: u32,
    pub weight_per_order: u32,
    pub max_weight_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRoutingConfig {
    pub default_venue: String,
    pub routing_algorithm: RoutingAlgorithm,
    pub smart_routing_enabled: bool,
    pub latency_threshold_ms: u64,
    pub fill_ratio_threshold: f64,
    pub cost_threshold_bps: u32,
    pub dark_pool_preference: f64,
    pub fragmentation_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingAlgorithm {
    First,
    Round,
    Random,
    Weighted,
    Smart,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionAlgoConfig {
    pub twap: TwapConfig,
    pub vwap: VwapConfig,
    pub implementation_shortfall: ImplementationShortfallConfig,
    pub participation_rate: ParticipationRateConfig,
    pub iceberg: IcebergConfig,
    pub sniper: SniperConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapConfig {
    pub enabled: bool,
    pub default_duration_minutes: u32,
    pub min_slice_size: Decimal,
    pub max_slice_size: Decimal,
    pub randomization_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapConfig {
    pub enabled: bool,
    pub lookback_periods: u32,
    pub participation_rate: f64,
    pub volume_threshold: Decimal,
    pub adaptive_rate: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImplementationShortfallConfig {
    pub enabled: bool,
    pub risk_aversion: f64,
    pub market_impact_model: String,
    pub volatility_model: String,
    pub rebalance_frequency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipationRateConfig {
    pub enabled: bool,
    pub default_rate: f64,
    pub min_rate: f64,
    pub max_rate: f64,
    pub adaptive_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergConfig {
    pub enabled: bool,
    pub default_visible_ratio: f64,
    pub min_visible_size: Decimal,
    pub refresh_threshold: f64,
    pub randomization_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SniperConfig {
    pub enabled: bool,
    pub aggression_level: u8,
    pub timeout_ms: u64,
    pub retry_attempts: u32,
    pub price_improvement_threshold_bps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeConfig {
    pub default_commission_per_share: Decimal,
    pub default_commission_percentage: f64,
    pub minimum_commission: Decimal,
    pub maximum_commission: Decimal,
    pub currency_conversion_spread: f64,
    pub borrowing_rate_annual: f64,
    pub margin_interest_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionLimitsConfig {
    pub max_position_value: Decimal,
    pub max_daily_volume: Decimal,
    pub max_concentration_percentage: f64,
    pub max_sector_exposure: f64,
    pub max_single_stock_exposure: f64,
    pub max_leverage: f64,
    pub instrument_limits: HashMap<String, InstrumentLimits>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstrumentLimits {
    pub max_position_size: Decimal,
    pub max_order_size: Decimal,
    pub max_daily_turnover: Decimal,
    pub position_limit_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingHoursConfig {
    pub timezone: String,
    pub market_sessions: HashMap<String, MarketSession>,
    pub holidays: Vec<String>,
    pub early_close_days: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSession {
    pub name: String,
    pub start_time: String,
    pub end_time: String,
    pub days_of_week: Vec<u8>,
    pub trading_enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakingConfig {
    pub enabled: bool,
    pub target_spread_bps: u32,
    pub inventory_target: Decimal,
    pub max_inventory_deviation: f64,
    pub quote_refresh_interval_ms: u64,
    pub skew_adjustment_factor: f64,
    pub adverse_selection_protection: bool,
    pub minimum_quote_size: Decimal,
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            global: GlobalTradingConfig {
                enabled: true,
                max_orders_per_second: 100,
                max_daily_orders: 10000,
                max_order_value: Decimal::new(1_000_000, 0), // $1M
                min_order_value: Decimal::new(1, 0), // $1
                default_currency: "USD".to_string(),
                enable_short_selling: true,
                enable_margin_trading: true,
                enable_options_trading: true,
                enable_futures_trading: true,
                enable_crypto_trading: false,
            },
            venues: HashMap::new(),
            routing: OrderRoutingConfig {
                default_venue: "PRIMARY".to_string(),
                routing_algorithm: RoutingAlgorithm::Smart,
                smart_routing_enabled: true,
                latency_threshold_ms: 100,
                fill_ratio_threshold: 0.8,
                cost_threshold_bps: 5,
                dark_pool_preference: 0.3,
                fragmentation_threshold: 5,
            },
            execution_algos: ExecutionAlgoConfig::default(),
            fees: FeeConfig {
                default_commission_per_share: Decimal::new(1, 3), // $0.001
                default_commission_percentage: 0.0005, // 0.05%
                minimum_commission: Decimal::new(1, 0), // $1
                maximum_commission: Decimal::new(100, 0), // $100
                currency_conversion_spread: 0.001, // 0.1%
                borrowing_rate_annual: 0.03, // 3%
                margin_interest_rate: 0.05, // 5%
            },
            position_limits: PositionLimitsConfig {
                max_position_value: Decimal::new(10_000_000, 0), // $10M
                max_daily_volume: Decimal::new(50_000_000, 0), // $50M
                max_concentration_percentage: 0.1, // 10%
                max_sector_exposure: 0.25, // 25%
                max_single_stock_exposure: 0.05, // 5%
                max_leverage: 2.0, // 2:1
                instrument_limits: HashMap::new(),
            },
            trading_hours: TradingHoursConfig {
                timezone: "America/New_York".to_string(),
                market_sessions: HashMap::new(),
                holidays: vec![],
                early_close_days: HashMap::new(),
            },
            market_making: MarketMakingConfig {
                enabled: false,
                target_spread_bps: 5,
                inventory_target: Decimal::ZERO,
                max_inventory_deviation: 0.1, // 10%
                quote_refresh_interval_ms: 100,
                skew_adjustment_factor: 0.5,
                adverse_selection_protection: true,
                minimum_quote_size: Decimal::new(100, 0),
            },
        }
    }
}

impl Default for ExecutionAlgoConfig {
    fn default() -> Self {
        Self {
            twap: TwapConfig {
                enabled: true,
                default_duration_minutes: 30,
                min_slice_size: Decimal::new(100, 0),
                max_slice_size: Decimal::new(10000, 0),
                randomization_factor: 0.1,
            },
            vwap: VwapConfig {
                enabled: true,
                lookback_periods: 20,
                participation_rate: 0.1, // 10%
                volume_threshold: Decimal::new(1000, 0),
                adaptive_rate: true,
            },
            implementation_shortfall: ImplementationShortfallConfig {
                enabled: true,
                risk_aversion: 0.5,
                market_impact_model: "square_root".to_string(),
                volatility_model: "garch".to_string(),
                rebalance_frequency_ms: 1000,
            },
            participation_rate: ParticipationRateConfig {
                enabled: true,
                default_rate: 0.2, // 20%
                min_rate: 0.01, // 1%
                max_rate: 0.5, // 50%
                adaptive_enabled: true,
            },
            iceberg: IcebergConfig {
                enabled: true,
                default_visible_ratio: 0.1, // 10%
                min_visible_size: Decimal::new(100, 0),
                refresh_threshold: 0.5, // 50%
                randomization_enabled: true,
            },
            sniper: SniperConfig {
                enabled: true,
                aggression_level: 3,
                timeout_ms: 100,
                retry_attempts: 3,
                price_improvement_threshold_bps: 1,
            },
        }
    }
}

impl Configuration for TradingConfig {
    const CONFIG_NAME: &'static str = "trading";
    
    fn validate(&self) -> Result<()> {
        // Validate global settings
        if self.global.max_orders_per_second == 0 {
            return Err(AlgoVedaError::Config("Max orders per second must be greater than 0".to_string()));
        }
        
        if self.global.max_order_value <= self.global.min_order_value {
            return Err(AlgoVedaError::Config("Max order value must be greater than min order value".to_string()));
        }
        
        // Validate position limits
        if self.position_limits.max_leverage <= 0.0 {
            return Err(AlgoVedaError::Config("Max leverage must be greater than 0".to_string()));
        }
        
        if self.position_limits.max_concentration_percentage <= 0.0 || 
           self.position_limits.max_concentration_percentage > 1.0 {
            return Err(AlgoVedaError::Config("Max concentration percentage must be between 0 and 1".to_string()));
        }
        
        // Validate venue configurations
        for (venue_name, venue_config) in &self.venues {
            if venue_config.priority == 0 {
                return Err(AlgoVedaError::Config(
                    format!("Venue {} priority must be greater than 0", venue_name)
                ));
            }
            
            if venue_config.min_order_size >= venue_config.max_order_size {
                return Err(AlgoVedaError::Config(
                    format!("Venue {} min order size must be less than max order size", venue_name)
                ));
            }
        }
        
        // Validate execution algorithm configurations
        if self.execution_algos.participation_rate.default_rate <= 0.0 || 
           self.execution_algos.participation_rate.default_rate > 1.0 {
            return Err(AlgoVedaError::Config("Participation rate must be between 0 and 1".to_string()));
        }
        
        Ok(())
    }
}

impl TradingConfig {
    /// Get venue configuration by name
    pub fn get_venue(&self, name: &str) -> Option<&VenueConfig> {
        self.venues.get(name)
    }
    
    /// Get enabled venues sorted by priority
    pub fn get_enabled_venues(&self) -> Vec<&VenueConfig> {
        let mut venues: Vec<&VenueConfig> = self.venues
            .values()
            .filter(|v| v.enabled)
            .collect();
        
        venues.sort_by(|a, b| a.priority.cmp(&b.priority));
        venues
    }
    
    /// Check if trading is currently enabled
    pub fn is_trading_enabled(&self) -> bool {
        self.global.enabled
    }
    
    /// Get commission for a given order value
    pub fn calculate_commission(&self, order_value: Decimal, shares: u32) -> Decimal {
        let commission_by_value = order_value * Decimal::from_f64(self.fees.default_commission_percentage).unwrap();
        let commission_by_shares = Decimal::from(shares) * self.fees.default_commission_per_share;
        
        let commission = commission_by_value.max(commission_by_shares);
        commission.max(self.fees.minimum_commission).min(self.fees.maximum_commission)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_validation() {
        let config = TradingConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_commission_calculation() {
        let config = TradingConfig::default();
        let commission = config.calculate_commission(Decimal::new(10000, 0), 100);
        assert!(commission >= config.fees.minimum_commission);
    }

    #[test]
    fn test_enabled_venues() {
        let mut config = TradingConfig::default();
        
        // Add test venues
        config.venues.insert("VENUE1".to_string(), VenueConfig {
            name: "Venue 1".to_string(),
            enabled: true,
            priority: 1,
            connection_config: VenueConnectionConfig {
                api_url: "https://api.venue1.com".to_string(),
                websocket_url: "wss://ws.venue1.com".to_string(),
                fix_config: None,
                api_key: None,
                secret_key: None,
                sandbox_mode: true,
                timeout_ms: 5000,
                max_retries: 3,
                heartbeat_interval_ms: 30000,
            },
            order_types: vec!["LIMIT".to_string(), "MARKET".to_string()],
            supported_instruments: vec!["EQUITY".to_string()],
            min_order_size: Decimal::new(1, 0),
            max_order_size: Decimal::new(1000000, 0),
            tick_size: Decimal::new(1, 2), // $0.01
            lot_size: 1,
            trading_fees: VenueFeeConfig {
                maker_fee_bps: 5,
                taker_fee_bps: 10,
                withdrawal_fee: Decimal::ZERO,
                currency_conversion_fee_bps: 50,
            },
            rate_limits: RateLimitsConfig {
                orders_per_second: 10,
                requests_per_minute: 600,
                weight_per_order: 1,
                max_weight_per_minute: 1200,
            },
        });
        
        let enabled_venues = config.get_enabled_venues();
        assert_eq!(enabled_venues.len(), 1);
        assert_eq!(enabled_venues[0].name, "Venue 1");
    }
}
