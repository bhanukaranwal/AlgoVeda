use serde::{Deserialize, Serialize};
use config::ConfigError;
use super::ConfigLoader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub max_position_size: f64,
    pub max_portfolio_value: f64,
    pub max_daily_loss: f64,
    pub max_drawdown: f64,
    pub max_leverage: f64,
    pub max_concentration: f64,
    pub max_sector_exposure: f64,
    pub max_correlation: f64,
    pub enable_real_time_risk: bool,
    pub risk_check_interval: u64,
    pub var_confidence_level: f64,
    pub var_horizon_days: u32,
    pub stress_scenarios: Vec<StressScenario>,
    pub position_limits: PositionLimits,
    pub exposure_limits: ExposureLimits,
    pub liquidity_limits: LiquidityLimits,
    pub options_limits: OptionsLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressScenario {
    pub name: String,
    pub market_shock: f64,
    pub volatility_shock: f64,
    pub correlation_shock: f64,
    pub liquidity_shock: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionLimits {
    pub max_single_position: f64,
    pub max_positions_per_symbol: u32,
    pub max_total_positions: u32,
    pub position_concentration_limit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureLimits {
    pub max_gross_exposure: f64,
    pub max_net_exposure: f64,
    pub max_sector_exposure: f64,
    pub max_country_exposure: f64,
    pub max_currency_exposure: f64,
    pub max_single_issuer: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidityLimits {
    pub min_liquidity_ratio: f64,
    pub max_illiquid_positions: f64,
    pub liquidity_horizon_days: u32,
    pub min_average_daily_volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsLimits {
    pub max_delta_exposure: f64,
    pub max_gamma_exposure: f64,
    pub max_vega_exposure: f64,
    pub max_theta_exposure: f64,
    pub max_rho_exposure: f64,
    pub max_options_concentration: f64,
}

impl ConfigLoader for RiskConfig {
    fn load(_path: Option<&str>) -> Result<Self, ConfigError> {
        Ok(Self::default())
    }
    
    fn validate(&self) -> Result<(), ConfigError> {
        if self.max_position_size <= 0.0 {
            return Err(ConfigError::Message("Max position size must be positive".to_string()));
        }
        
        if self.max_daily_loss <= 0.0 {
            return Err(ConfigError::Message("Max daily loss must be positive".to_string()));
        }
        
        if self.var_confidence_level <= 0.0 || self.var_confidence_level >= 1.0 {
            return Err(ConfigError::Message("VaR confidence level must be between 0 and 1".to_string()));
        }
        
        if self.max_leverage < 1.0 {
            return Err(ConfigError::Message("Max leverage cannot be less than 1".to_string()));
        }
        
        Ok(())
    }
    
    fn merge(&mut self, _other: Self) -> Result<(), ConfigError> {
        Ok(())
    }
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_size: 500000.0,
            max_portfolio_value: 10000000.0,
            max_daily_loss: 50000.0,
            max_drawdown: 0.15,
            max_leverage: 4.0,
            max_concentration: 0.20,
            max_sector_exposure: 0.30,
            max_correlation: 0.70,
            enable_real_time_risk: true,
            risk_check_interval: 1000,
            var_confidence_level: 0.95,
            var_horizon_days: 1,
            stress_scenarios: vec![
                StressScenario {
                    name: "Market Crash".to_string(),
                    market_shock: -0.20,
                    volatility_shock: 2.0,
                    correlation_shock: 0.5,
                    liquidity_shock: 0.5,
                },
                StressScenario {
                    name: "Volatility Spike".to_string(),
                    market_shock: -0.10,
                    volatility_shock: 3.0,
                    correlation_shock: 0.3,
                    liquidity_shock: 0.3,
                },
            ],
            position_limits: PositionLimits::default(),
            exposure_limits: ExposureLimits::default(),
            liquidity_limits: LiquidityLimits::default(),
            options_limits: OptionsLimits::default(),
        }
    }
}

impl Default for PositionLimits {
    fn default() -> Self {
        Self {
            max_single_position: 100000.0,
            max_positions_per_symbol: 5,
            max_total_positions: 100,
            position_concentration_limit: 0.1,
        }
    }
}

impl Default for ExposureLimits {
    fn default() -> Self {
        Self {
            max_gross_exposure: 2.0,
            max_net_exposure: 1.0,
            max_sector_exposure: 0.3,
            max_country_exposure: 0.8,
            max_currency_exposure: 0.2,
            max_single_issuer: 0.05,
        }
    }
}

impl Default for LiquidityLimits {
    fn default() -> Self {
        Self {
            min_liquidity_ratio: 0.1,
            max_illiquid_positions: 0.2,
            liquidity_horizon_days: 5,
            min_average_daily_volume: 100000.0,
        }
    }
}

impl Default for OptionsLimits {
    fn default() -> Self {
        Self {
            max_delta_exposure: 1000000.0,
            max_gamma_exposure: 100000.0,
            max_vega_exposure: 500000.0,
            max_theta_exposure: -10000.0,
            max_rho_exposure: 100000.0,
            max_options_concentration: 0.3,
        }
    }
}
