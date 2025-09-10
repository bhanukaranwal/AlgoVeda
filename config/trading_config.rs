use serde::{Deserialize, Serialize};
use config::ConfigError;
use super::ConfigLoader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    pub max_order_size: f64,
    pub min_order_size: f64,
    pub default_commission: f64,
    pub default_slippage: f64,
    pub enable_paper_trading: bool,
    pub enable_live_trading: bool,
    pub order_timeout: u64,
    pub max_daily_orders: u32,
    pub supported_exchanges: Vec<String>,
    pub supported_instruments: Vec<String>,
    pub execution_algorithms: ExecutionAlgorithmConfig,
    pub order_routing: OrderRoutingConfig,
    pub market_making: MarketMakingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionAlgorithmConfig {
    pub enable_twap: bool,
    pub enable_vwap: bool,
    pub enable_implementation_shortfall: bool,
    pub enable_participation_rate: bool,
    pub enable_iceberg: bool,
    pub enable_sniper: bool,
    pub enable_guerrilla: bool,
    pub enable_stealth: bool,
    pub default_algorithm: String,
    pub slice_size_ratio: f64,
    pub max_participation_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRoutingConfig {
    pub enable_smart_routing: bool,
    pub enable_dark_pools: bool,
    pub venue_preferences: Vec<VenuePreference>,
    pub routing_algorithm: String,
    pub latency_threshold: u64,
    pub fill_ratio_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenuePreference {
    pub venue: String,
    pub priority: u32,
    pub min_size: f64,
    pub max_size: f64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMakingConfig {
    pub enable_market_making: bool,
    pub max_spread_bps: f64,
    pub min_spread_bps: f64,
    pub quote_size: u32,
    pub max_inventory: f64,
    pub skew_factor: f64,
    pub adverse_selection_threshold: f64,
}

impl ConfigLoader for TradingConfig {
    fn load(_path: Option<&str>) -> Result<Self, ConfigError> {
        Ok(Self::default())
    }
    
    fn validate(&self) -> Result<(), ConfigError> {
        if self.max_order_size <= 0.0 {
            return Err(ConfigError::Message("Max order size must be positive".to_string()));
        }
        
        if self.min_order_size <= 0.0 {
            return Err(ConfigError::Message("Min order size must be positive".to_string()));
        }
        
        if self.min_order_size > self.max_order_size {
            return Err(ConfigError::Message("Min order size cannot exceed max order size".to_string()));
        }
        
        if self.default_commission < 0.0 {
            return Err(ConfigError::Message("Commission cannot be negative".to_string()));
        }
        
        Ok(())
    }
    
    fn merge(&mut self, _other: Self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl Default for TradingConfig {
    fn default() -> Self {
        Self {
            max_order_size: 1000000.0,
            min_order_size: 1.0,
            default_commission: 0.001,
            default_slippage: 0.0005,
            enable_paper_trading: true,
            enable_live_trading: false,
            order_timeout: 30,
            max_daily_orders: 10000,
            supported_exchanges: vec!["NSE".to_string(), "BSE".to_string()],
            supported_instruments: vec!["equity".to_string(), "options".to_string(), "futures".to_string()],
            execution_algorithms: ExecutionAlgorithmConfig::default(),
            order_routing: OrderRoutingConfig::default(),
            market_making: MarketMakingConfig::default(),
        }
    }
}

impl Default for ExecutionAlgorithmConfig {
    fn default() -> Self {
        Self {
            enable_twap: true,
            enable_vwap: true,
            enable_implementation_shortfall: true,
            enable_participation_rate: true,
            enable_iceberg: true,
            enable_sniper: false,
            enable_guerrilla: false,
            enable_stealth: false,
            default_algorithm: "twap".to_string(),
            slice_size_ratio: 0.1,
            max_participation_rate: 0.3,
        }
    }
}

impl Default for OrderRoutingConfig {
    fn default() -> Self {
        Self {
            enable_smart_routing: true,
            enable_dark_pools: false,
            venue_preferences: vec![
                VenuePreference {
                    venue: "NSE".to_string(),
                    priority: 1,
                    min_size: 1.0,
                    max_size: 1000000.0,
                    enabled: true,
                },
            ],
            routing_algorithm: "smart".to_string(),
            latency_threshold: 1000,
            fill_ratio_threshold: 0.95,
        }
    }
}

impl Default for MarketMakingConfig {
    fn default() -> Self {
        Self {
            enable_market_making: false,
            max_spread_bps: 50.0,
            min_spread_bps: 5.0,
            quote_size: 100,
            max_inventory: 10000.0,
            skew_factor: 0.5,
            adverse_selection_threshold: 0.1,
        }
    }
}
