/*!
 * Performance Attribution Engine
 * Advanced portfolio performance analysis and factor attribution
 */

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use ndarray::{Array1, Array2, Axis};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration as ChronoDuration};

use crate::{
    error::{Result, AlgoVedaError},
    portfolio::{Portfolio, Position, Transaction},
    market_data::MarketData,
    utils::statistics::{regression, correlation},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionConfig {
    pub enable_brinson_attribution: bool,
    pub enable_factor_attribution: bool,
    pub enable_sector_attribution: bool,
    pub enable_currency_attribution: bool,
    pub benchmark_symbol: Option<String>,
    pub attribution_frequency: AttributionFrequency,
    pub lookback_days: u32,
    pub factor_models: Vec<FactorModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttributionFrequency {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorModel {
    CAPM,
    FamaFrench3Factor,
    FamaFrench5Factor,
    Carhart4Factor,
    Custom(Vec<String>), // Custom factor names
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionResult {
    pub total_return: f64,
    pub benchmark_return: f64,
    pub active_return: f64,
    pub attribution_components: HashMap<String, AttributionComponent>,
    pub factor_exposures: HashMap<String, f64>,
    pub risk_attribution: RiskAttribution,
    pub performance_metrics: PerformanceMetrics,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub calculation_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionComponent {
    pub allocation_effect: f64,
    pub selection_effect: f64,
    pub interaction_effect: f64,
    pub total_effect: f64,
    pub weight_portfolio: f64,
    pub weight_benchmark: f64,
    pub return_portfolio: f64,
    pub return_benchmark: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAttribution {
    pub total_risk: f64,
    pub systematic_risk: f64,
    pub specific_risk: f64,
    pub factor_risk_contributions: HashMap<String, f64>,
    pub asset_risk_contributions: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub sharpe_ratio: f64,
    pub information_ratio: f64,
    pub treynor_ratio: f64,
    pub jensen_alpha: f64,
    pub beta: f64,
    pub tracking_error: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub sortino_ratio: f64,
    pub upside_capture: f64,
    pub downside_capture: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoldingPeriodReturn {
    pub symbol: String,
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub start_price: f64,
    pub end_price: f64,
    pub dividends: f64,
    pub total_return: f64,
    pub weight: f64,
    pub contribution: f64,
}

pub struct PerformanceAttributionEngine {
    config: AttributionConfig,
    historical_prices: Arc<RwLock<HashMap<String, Vec<(DateTime<Utc>, f64)>>>>,
    factor_data: Arc<RwLock<HashMap<String, Vec<(DateTime<Utc>, f64)>>>>,
    benchmark_data: Arc<RwLock<Vec<(DateTime<Utc>, f64)>>>,
    sector_mappings: Arc<RwLock<HashMap<String, String>>>,
    currency_mappings: Arc<RwLock<HashMap<String, String>>>,
}

impl PerformanceAttributionEngine {
    pub fn new(config: AttributionConfig) -> Self {
        Self {
            config,
            historical_prices: Arc::new(RwLock::new(HashMap::new())),
            factor_data: Arc::new(RwLock::new(HashMap::new())),
            benchmark_data: Arc::new(RwLock::new(Vec::new())),
            sector_mappings: Arc::new(RwLock::new(HashMap::new())),
            currency_mappings: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Perform comprehensive performance attribution analysis
    pub fn analyze_performance(
        &self, 
        portfolio: &Portfolio, 
        start_date: DateTime<Utc>, 
        end_date: DateTime<Utc>
    ) -> Result<AttributionResult> {
        let start_time = Instant::now();

        // Calculate portfolio returns
        let portfolio_returns = self.calculate_portfolio_returns(portfolio, start_date, end_date)?;
        let total_return = portfolio_returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;

        // Calculate benchmark returns
        let benchmark_return = if let Some(benchmark) = &self.config.benchmark_symbol {
            self.calculate_benchmark_return(benchmark, start_date, end_date)?
        } else {
            0.0
        };

        let active_return = total_return - benchmark_return;

        // Perform different types of attribution
        let mut attribution_components = HashMap::new();

        if self.config.enable_brinson_attribution {
            let brinson_results = self.brinson_attribution(portfolio, start_date, end_date)?;
            attribution_components.extend(brinson_results);
        }

        if self.config.enable_sector_attribution {
            let sector_results = self.sector_attribution(portfolio, start_date, end_date)?;
            attribution_components.extend(sector_results);
        }

        if self.config.enable_factor_attribution {
            let factor_results = self.factor_attribution(portfolio, start_date, end_date)?;
            attribution_components.extend(factor_results);
        }

        // Calculate factor exposures
        let factor_exposures = self.calculate_factor_exposures(portfolio, start_date, end_date)?;

        // Calculate risk attribution
        let risk_attribution = self.calculate_risk_attribution(portfolio, start_date, end_date)?;

        // Calculate performance metrics
        let performance_metrics = self.calculate_performance_metrics(
            &portfolio_returns, 
            benchmark_return,
            start_date,
            end_date
        )?;

        Ok(AttributionResult {
            total_return,
            benchmark_return,
            active_return,
            attribution_components,
            factor_exposures,
            risk_attribution,
            performance_metrics,
            period_start: start_date,
            period_end: end_date,
            calculation_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Brinson-Hood-Beebower attribution model
    fn brinson_attribution(
        &self,
        portfolio: &Portfolio,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<HashMap<String, AttributionComponent>> {
        let mut attribution_components = HashMap::new();

        // Get sector classifications
        let sector_mappings = self.sector_mappings.read().unwrap();
        let mut sector_data: HashMap<String, Vec<&Position>> = HashMap::new();

        // Group positions by sector
        for position in portfolio.get_positions() {
            let sector = sector_mappings
                .get(&position.symbol)
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string());
            
            sector_data
                .entry(sector)
                .or_insert_with(Vec::new)
                .push(position);
        }

        // Calculate attribution for each sector
        for (sector, positions) in sector_data {
            let sector_component = self.calculate_sector_attribution(
                &positions,
                &sector,
                start_date,
                end_date,
            )?;
            
            attribution_components.insert(format!("Sector_{}", sector), sector_component);
        }

        Ok(attribution_components)
    }

    /// Calculate attribution for a specific sector
    fn calculate_sector_attribution(
        &self,
        positions: &[&Position],
        sector: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<AttributionComponent> {
        let historical_prices = self.historical_prices.read().unwrap();
        
        // Calculate portfolio sector weight and return
        let mut sector_value = 0.0;
        let mut sector_return_weighted = 0.0;
        let total_portfolio_value = positions.iter().map(|p| p.market_value).sum::<f64>();

        for position in positions {
            if let Some(price_history) = historical_prices.get(&position.symbol) {
                let start_price = self.get_price_at_date(price_history, start_date);
                let end_price = self.get_price_at_date(price_history, end_date);
                
                if start_price > 0.0 && end_price > 0.0 {
                    let return_rate = (end_price - start_price) / start_price;
                    let weight = position.market_value / total_portfolio_value;
                    
                    sector_value += position.market_value;
                    sector_return_weighted += weight * return_rate;
                }
            }
        }

        let weight_portfolio = sector_value / total_portfolio_value;
        let return_portfolio = if sector_value > 0.0 { 
            sector_return_weighted / weight_portfolio 
        } else { 
            0.0 
        };

        // For benchmark data, we'd need benchmark sector composition
        // Simplified: assume benchmark weight is equal weight across sectors
        let weight_benchmark = 0.1; // Placeholder - would be calculated from benchmark
        let return_benchmark = 0.05; // Placeholder - would be calculated from benchmark

        // Brinson attribution components
        let allocation_effect = (weight_portfolio - weight_benchmark) * return_benchmark;
        let selection_effect = weight_benchmark * (return_portfolio - return_benchmark);
        let interaction_effect = (weight_portfolio - weight_benchmark) * (return_portfolio - return_benchmark);
        let total_effect = allocation_effect + selection_effect + interaction_effect;

        Ok(AttributionComponent {
            allocation_effect,
            selection_effect,
            interaction_effect,
            total_effect,
            weight_portfolio,
            weight_benchmark,
            return_portfolio,
            return_benchmark,
        })
    }

    /// Factor-based attribution using multi-factor models
    fn factor_attribution(
        &self,
        portfolio: &Portfolio,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<HashMap<String, AttributionComponent>> {
        let mut factor_components = HashMap::new();

        for factor_model in &self.config.factor_models {
            match factor_model {
                FactorModel::CAPM => {
                    let capm_result = self.capm_attribution(portfolio, start_date, end_date)?;
                    factor_components.insert("Market_Beta".to_string(), capm_result);
                }
                FactorModel::FamaFrench3Factor => {
                    let ff3_results = self.fama_french_3factor_attribution(portfolio, start_date, end_date)?;
                    factor_components.extend(ff3_results);
                }
                FactorModel::FamaFrench5Factor => {
                    let ff5_results = self.fama_french_5factor_attribution(portfolio, start_date, end_date)?;
                    factor_components.extend(ff5_results);
                }
                FactorModel::Carhart4Factor => {
                    let carhart_results = self.carhart_4factor_attribution(portfolio, start_date, end_date)?;
                    factor_components.extend(carhart_results);
                }
                FactorModel::Custom(factors) => {
                    for factor in factors {
                        let custom_result = self.custom_factor_attribution(portfolio, factor, start_date, end_date)?;
                        factor_components.insert(factor.clone(), custom_result);
                    }
                }
            }
        }

        Ok(factor_components)
    }

    /// CAPM attribution
    fn capm_attribution(
        &self,
        portfolio: &Portfolio,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<AttributionComponent> {
        let portfolio_returns = self.calculate_portfolio_returns(portfolio, start_date, end_date)?;
        let market_returns = self.get_factor_returns("Market", start_date, end_date)?;

        if portfolio_returns.len() != market_returns.len() {
            return Err(AlgoVedaError::Risk("Mismatched return series lengths".to_string()));
        }

        // Calculate beta using regression
        let (alpha, beta, r_squared) = regression(&market_returns, &portfolio_returns)?;
        
        let portfolio_return = portfolio_returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;
        let market_return = market_returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;
        let risk_free_rate = 0.02; // Would fetch from market data

        // CAPM decomposition
        let systematic_return = beta * market_return;
        let alpha_return = portfolio_return - systematic_return - risk_free_rate;

        Ok(AttributionComponent {
            allocation_effect: 0.0, // Not applicable for factor models
            selection_effect: alpha_return,
            interaction_effect: systematic_return - beta * market_return,
            total_effect: alpha_return + systematic_return,
            weight_portfolio: 1.0, // Full portfolio
            weight_benchmark: 1.0,
            return_portfolio: portfolio_return,
            return_benchmark: market_return,
        })
    }

    /// Fama-French 3-factor attribution
    fn fama_french_3factor_attribution(
        &self,
        portfolio: &Portfolio,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<HashMap<String, AttributionComponent>> {
        let portfolio_returns = self.calculate_portfolio_returns(portfolio, start_date, end_date)?;
        
        // Get factor returns
        let market_returns = self.get_factor_returns("Market", start_date, end_date)?;
        let smb_returns = self.get_factor_returns("SMB", start_date, end_date)?; // Small minus Big
        let hml_returns = self.get_factor_returns("HML", start_date, end_date)?; // High minus Low

        // Multiple regression: R_p = α + β₁*MKT + β₂*SMB + β₃*HML + ε
        let factor_matrix = Array2::from_shape_vec(
            (portfolio_returns.len(), 3),
            market_returns.iter()
                .zip(smb_returns.iter())
                .zip(hml_returns.iter())
                .flat_map(|((mkt, smb), hml)| vec![*mkt, *smb, *hml])
                .collect()
        ).map_err(|_| AlgoVedaError::Risk("Failed to create factor matrix".to_string()))?;

        let (coefficients, alpha) = self.multiple_regression(&factor_matrix, &portfolio_returns)?;

        let mut components = HashMap::new();

        // Market factor
        components.insert("Market".to_string(), AttributionComponent {
            allocation_effect: 0.0,
            selection_effect: coefficients[0] * market_returns.iter().sum::<f64>(),
            interaction_effect: 0.0,
            total_effect: coefficients[0] * market_returns.iter().sum::<f64>(),
            weight_portfolio: coefficients[0],
            weight_benchmark: 1.0,
            return_portfolio: market_returns.iter().sum::<f64>(),
            return_benchmark: market_returns.iter().sum::<f64>(),
        });

        // Size factor (SMB)
        components.insert("Size".to_string(), AttributionComponent {
            allocation_effect: 0.0,
            selection_effect: coefficients[1] * smb_returns.iter().sum::<f64>(),
            interaction_effect: 0.0,
            total_effect: coefficients[1] * smb_returns.iter().sum::<f64>(),
            weight_portfolio: coefficients[1],
            weight_benchmark: 0.0,
            return_portfolio: smb_returns.iter().sum::<f64>(),
            return_benchmark: 0.0,
        });

        // Value factor (HML)
        components.insert("Value".to_string(), AttributionComponent {
            allocation_effect: 0.0,
            selection_effect: coefficients[2] * hml_returns.iter().sum::<f64>(),
            interaction_effect: 0.0,
            total_effect: coefficients[2] * hml_returns.iter().sum::<f64>(),
            weight_portfolio: coefficients[2],
            weight_benchmark: 0.0,
            return_portfolio: hml_returns.iter().sum::<f64>(),
            return_benchmark: 0.0,
        });

        // Alpha
        components.insert("Alpha".to_string(), AttributionComponent {
            allocation_effect: 0.0,
            selection_effect: alpha,
            interaction_effect: 0.0,
            total_effect: alpha,
            weight_portfolio: 1.0,
            weight_benchmark: 0.0,
            return_portfolio: alpha,
            return_benchmark: 0.0,
        });

        Ok(components)
    }

    /// Sector-based attribution
    fn sector_attribution(
        &self,
        portfolio: &Portfolio,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<HashMap<String, AttributionComponent>> {
        // This would be similar to Brinson attribution but focused on sectors
        self.brinson_attribution(portfolio, start_date, end_date)
    }

    /// Calculate factor exposures for the portfolio
    fn calculate_factor_exposures(
        &self,
        portfolio: &Portfolio,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<HashMap<String, f64>> {
        let mut factor_exposures = HashMap::new();
        let portfolio_returns = self.calculate_portfolio_returns(portfolio, start_date, end_date)?;

        // Calculate beta (market exposure)
        if let Ok(market_returns) = self.get_factor_returns("Market", start_date, end_date) {
            if portfolio_returns.len() == market_returns.len() {
                let (_, beta, _) = regression(&market_returns, &portfolio_returns)?;
                factor_exposures.insert("Beta".to_string(), beta);
            }
        }

        // Calculate other factor exposures based on portfolio composition
        let positions = portfolio.get_positions();
        let total_value = portfolio.get_total_value();
        
        // Size exposure (market cap weighted)
        let mut size_exposure = 0.0;
        for position in positions {
            // Would need market cap data - simplified calculation
            let weight = position.market_value / total_value;
            size_exposure += weight * 1.0; // Placeholder - would calculate actual size factor
        }
        factor_exposures.insert("Size".to_string(), size_exposure);

        // Value exposure (P/E, P/B weighted)
        let mut value_exposure = 0.0;
        for position in positions {
            let weight = position.market_value / total_value;
            value_exposure += weight * 0.5; // Placeholder - would calculate from fundamental data
        }
        factor_exposures.insert("Value".to_string(), value_exposure);

        Ok(factor_exposures)
    }

    /// Calculate risk attribution
    fn calculate_risk_attribution(
        &self,
        portfolio: &Portfolio,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<RiskAttribution> {
        let portfolio_returns = self.calculate_portfolio_returns(portfolio, start_date, end_date)?;
        let total_risk = self.calculate_volatility(&portfolio_returns);

        // Systematic vs specific risk decomposition
        let market_returns = self.get_factor_returns("Market", start_date, end_date)
            .unwrap_or_else(|_| vec![0.0; portfolio_returns.len()]);

        let (_, beta, r_squared) = if portfolio_returns.len() == market_returns.len() {
            regression(&market_returns, &portfolio_returns)?
        } else {
            (0.0, 1.0, 0.0)
        };

        let systematic_risk = (beta * beta * self.calculate_volatility(&market_returns).powi(2)).sqrt();
        let specific_risk = (total_risk.powi(2) - systematic_risk.powi(2)).max(0.0).sqrt();

        // Factor risk contributions (simplified)
        let mut factor_risk_contributions = HashMap::new();
        factor_risk_contributions.insert("Market".to_string(), systematic_risk);
        factor_risk_contributions.insert("Specific".to_string(), specific_risk);

        // Asset risk contributions
        let mut asset_risk_contributions = HashMap::new();
        let positions = portfolio.get_positions();
        let total_value = portfolio.get_total_value();

        for position in positions {
            let weight = position.market_value / total_value;
            let asset_contribution = weight * total_risk * 0.1; // Simplified - would need covariance matrix
            asset_risk_contributions.insert(position.symbol.clone(), asset_contribution);
        }

        Ok(RiskAttribution {
            total_risk,
            systematic_risk,
            specific_risk,
            factor_risk_contributions,
            asset_risk_contributions,
        })
    }

    /// Calculate comprehensive performance metrics
    fn calculate_performance_metrics(
        &self,
        portfolio_returns: &[f64],
        benchmark_return: f64,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<PerformanceMetrics> {
        let risk_free_rate = 0.02; // Would fetch from market data
        let portfolio_return = portfolio_returns.iter().fold(1.0, |acc, r| acc * (1.0 + r)) - 1.0;
        
        // Volatility and tracking error
        let volatility = self.calculate_volatility(portfolio_returns);
        let excess_returns: Vec<f64> = portfolio_returns.iter()
            .map(|r| r - benchmark_return / portfolio_returns.len() as f64)
            .collect();
        let tracking_error = self.calculate_volatility(&excess_returns);

        // Sharpe ratio
        let sharpe_ratio = if volatility > 0.0 {
            (portfolio_return - risk_free_rate) / volatility
        } else {
            0.0
        };

        // Information ratio
        let active_return = portfolio_return - benchmark_return;
        let information_ratio = if tracking_error > 0.0 {
            active_return / tracking_error
        } else {
            0.0
        };

        // Beta calculation
        let market_returns = self.get_factor_returns("Market", start_date, end_date)
            .unwrap_or_else(|_| vec![benchmark_return / portfolio_returns.len() as f64; portfolio_returns.len()]);
        
        let (jensen_alpha, beta, _) = if portfolio_returns.len() == market_returns.len() {
            regression(&market_returns, portfolio_returns)?
        } else {
            (0.0, 1.0, 0.0)
        };

        // Treynor ratio
        let treynor_ratio = if beta != 0.0 {
            (portfolio_return - risk_free_rate) / beta
        } else {
            0.0
        };

        // Max drawdown
        let mut cumulative_return = 1.0;
        let mut peak = 1.0;
        let mut max_drawdown = 0.0;

        for &ret in portfolio_returns {
            cumulative_return *= 1.0 + ret;
            if cumulative_return > peak {
                peak = cumulative_return;
            }
            let drawdown = (peak - cumulative_return) / peak;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        // Calmar ratio
        let calmar_ratio = if max_drawdown > 0.0 {
            portfolio_return / max_drawdown
        } else {
            0.0
        };

        // Sortino ratio (downside deviation)
        let downside_returns: Vec<f64> = portfolio_returns.iter()
            .filter(|&&r| r < 0.0)
            .cloned()
            .collect();
        let downside_deviation = self.calculate_volatility(&downside_returns);
        
        let sortino_ratio = if downside_deviation > 0.0 {
            (portfolio_return - risk_free_rate) / downside_deviation
        } else {
            0.0
        };

        // Upside/Downside capture ratios
        let up_market_returns: Vec<f64> = market_returns.iter()
            .zip(portfolio_returns.iter())
            .filter(|(m, _)| **m > 0.0)
            .map(|(_, p)| *p)
            .collect();
        
        let down_market_returns: Vec<f64> = market_returns.iter()
            .zip(portfolio_returns.iter())
            .filter(|(m, _)| **m < 0.0)
            .map(|(_, p)| *p)
            .collect();

        let upside_capture = if !up_market_returns.is_empty() {
            let up_portfolio = up_market_returns.iter().sum::<f64>() / up_market_returns.len() as f64;
            let up_market = market_returns.iter().filter(|&&r| r > 0.0).sum::<f64>() / 
                          market_returns.iter().filter(|&&r| r > 0.0).count() as f64;
            if up_market != 0.0 { up_portfolio / up_market } else { 0.0 }
        } else {
            0.0
        };

        let downside_capture = if !down_market_returns.is_empty() {
            let down_portfolio = down_market_returns.iter().sum::<f64>() / down_market_returns.len() as f64;
            let down_market = market_returns.iter().filter(|&&r| r < 0.0).sum::<f64>() / 
                            market_returns.iter().filter(|&&r| r < 0.0).count() as f64;
            if down_market != 0.0 { down_portfolio / down_market } else { 0.0 }
        } else {
            0.0
        };

        Ok(PerformanceMetrics {
            sharpe_ratio,
            information_ratio,
            treynor_ratio,
            jensen_alpha,
            beta,
            tracking_error,
            max_drawdown,
            calmar_ratio,
            sortino_ratio,
            upside_capture,
            downside_capture,
        })
    }

    /// Helper functions
    fn calculate_portfolio_returns(
        &self,
        portfolio: &Portfolio,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Vec<f64>> {
        let historical_prices = self.historical_prices.read().unwrap();
        let positions = portfolio.get_positions();
        
        if positions.is_empty() {
            return Ok(vec![]);
        }

        // Get the date range for daily returns
        let mut current_date = start_date;
        let mut portfolio_returns = Vec::new();
        
        while current_date < end_date {
            let mut daily_return = 0.0;
            let mut total_weight = 0.0;
            
            for position in positions {
                if let Some(price_history) = historical_prices.get(&position.symbol) {
                    let price_today = self.get_price_at_date(price_history, current_date);
                    let price_tomorrow = self.get_price_at_date(price_history, 
                        current_date + ChronoDuration::days(1));
                    
                    if price_today > 0.0 && price_tomorrow > 0.0 {
                        let asset_return = (price_tomorrow - price_today) / price_today;
                        let weight = position.market_value / portfolio.get_total_value();
                        daily_return += weight * asset_return;
                        total_weight += weight;
                    }
                }
            }
            
            if total_weight > 0.0 {
                portfolio_returns.push(daily_return);
            }
            
            current_date += ChronoDuration::days(1);
        }
        
        Ok(portfolio_returns)
    }

    fn calculate_benchmark_return(&self, benchmark: &str, start_date: DateTime<Utc>, end_date: DateTime<Utc>) -> Result<f64> {
        let benchmark_data = self.benchmark_data.read().unwrap();
        
        let start_price = self.get_price_at_date(&benchmark_data, start_date);
        let end_price = self.get_price_at_date(&benchmark_data, end_date);
        
        if start_price > 0.0 && end_price > 0.0 {
            Ok((end_price - start_price) / start_price)
        } else {
            Ok(0.0)
        }
    }

    fn get_factor_returns(&self, factor_name: &str, start_date: DateTime<Utc>, end_date: DateTime<Utc>) -> Result<Vec<f64>> {
        let factor_data = self.factor_data.read().unwrap();
        
        if let Some(factor_series) = factor_data.get(factor_name) {
            let mut returns = Vec::new();
            let mut current_date = start_date;
            
            while current_date < end_date {
                let return_value = self.get_price_at_date(factor_series, current_date);
                returns.push(return_value);
                current_date += ChronoDuration::days(1);
            }
            
            Ok(returns)
        } else {
            Err(AlgoVedaError::Risk(format!("Factor data not found: {}", factor_name)))
        }
    }

    fn get_price_at_date(&self, price_series: &[(DateTime<Utc>, f64)], date: DateTime<Utc>) -> f64 {
        // Find the closest price to the given date
        price_series
            .iter()
            .min_by_key(|(d, _)| {
                if *d >= date {
                    (*d - date).num_seconds().abs()
                } else {
                    (date - *d).num_seconds().abs()
                }
            })
            .map(|(_, price)| *price)
            .unwrap_or(0.0)
    }

    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        (variance * 252.0).sqrt() // Annualized
    }

    fn multiple_regression(&self, factors: &Array2<f64>, returns: &[f64]) -> Result<(Vec<f64>, f64)> {
        // Simplified multiple regression - would use proper linear algebra library
        // Returns (coefficients, alpha)
        let n_factors = factors.ncols();
        let mut coefficients = vec![0.0; n_factors];
        
        // For each factor, calculate simple correlation-based coefficient
        for i in 0..n_factors {
            let factor_returns = factors.column(i).to_vec();
            let (alpha, beta, _) = regression(&factor_returns, returns)?;
            coefficients[i] = beta;
        }
        
        let alpha = returns.iter().sum::<f64>() / returns.len() as f64 - 
                   coefficients.iter().enumerate()
                       .map(|(i, &coef)| coef * factors.column(i).mean().unwrap_or(0.0))
                       .sum::<f64>();
        
        Ok((coefficients, alpha))
    }

    // Placeholder implementations for other factor models
    fn fama_french_5factor_attribution(&self, _portfolio: &Portfolio, _start_date: DateTime<Utc>, _end_date: DateTime<Utc>) -> Result<HashMap<String, AttributionComponent>> {
        // Would implement 5-factor model (3-factor + profitability + investment)
        Ok(HashMap::new())
    }

    fn carhart_4factor_attribution(&self, _portfolio: &Portfolio, _start_date: DateTime<Utc>, _end_date: DateTime<Utc>) -> Result<HashMap<String, AttributionComponent>> {
        // Would implement 4-factor model (3-factor + momentum)
        Ok(HashMap::new())
    }

    fn custom_factor_attribution(&self, _portfolio: &Portfolio, _factor: &str, _start_date: DateTime<Utc>, _end_date: DateTime<Utc>) -> Result<AttributionComponent> {
        // Would implement custom factor attribution
        Ok(AttributionComponent {
            allocation_effect: 0.0,
            selection_effect: 0.0,
            interaction_effect: 0.0,
            total_effect: 0.0,
            weight_portfolio: 0.0,
            weight_benchmark: 0.0,
            return_portfolio: 0.0,
            return_benchmark: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_attribution_creation() {
        let config = AttributionConfig {
            enable_brinson_attribution: true,
            enable_factor_attribution: true,
            enable_sector_attribution: true,
            enable_currency_attribution: false,
            benchmark_symbol: Some("SPY".to_string()),
            attribution_frequency: AttributionFrequency::Daily,
            lookback_days: 252,
            factor_models: vec![FactorModel::CAPM, FactorModel::FamaFrench3Factor],
        };
        
        let engine = PerformanceAttributionEngine::new(config);
        assert_eq!(engine.config.lookback_days, 252);
    }

    #[test]
    fn test_volatility_calculation() {
        let config = AttributionConfig {
            enable_brinson_attribution: false,
            enable_factor_attribution: false,
            enable_sector_attribution: false,
            enable_currency_attribution: false,
            benchmark_symbol: None,
            attribution_frequency: AttributionFrequency::Daily,
            lookback_days: 30,
            factor_models: vec![],
        };
        
        let engine = PerformanceAttributionEngine::new(config);
        
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.005];
        let volatility = engine.calculate_volatility(&returns);
        
        assert!(volatility > 0.0);
        assert!(volatility < 1.0);
    }
}
