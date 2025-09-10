/*!
 * Value at Risk (VaR) Calculator
 * Advanced risk metrics calculation with multiple methodologies
 */

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use nalgebra::{DMatrix, DVector};

use crate::{
    error::{Result, AlgoVedaError},
    portfolio::{Portfolio, Position},
    market_data::MarketData,
    utils::statistics::{percentile, correlation_matrix},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VarMethod {
    HistoricalSimulation,
    ParametricNormal,
    ParametricTDistribution,
    MonteCarlo,
    ExtremeValue,
    EWMA,
    GARCH,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarConfig {
    pub confidence_level: f64,  // e.g., 0.95 for 95% VaR
    pub time_horizon_days: u32,  // e.g., 1 for 1-day VaR
    pub lookback_days: u32,     // Historical data window
    pub monte_carlo_simulations: usize,
    pub ewma_lambda: f64,       // EWMA decay factor
    pub enable_component_var: bool,
    pub enable_marginal_var: bool,
    pub enable_incremental_var: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarResult {
    pub var_absolute: f64,
    pub var_percentage: f64,
    pub expected_shortfall: f64,  // Conditional VaR
    pub component_var: HashMap<String, f64>,
    pub marginal_var: HashMap<String, f64>,
    pub diversification_ratio: f64,
    pub method: VarMethod,
    pub confidence_level: f64,
    pub time_horizon_days: u32,
    pub calculation_time_ms: u64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct RiskFactorShock {
    pub factor_name: String,
    pub shock_magnitude: f64,
    pub shock_type: ShockType,
}

#[derive(Debug, Clone)]
pub enum ShockType {
    Absolute,      // Add shock value
    Relative,      // Multiply by (1 + shock)
    StandardDeviations, // Shock in terms of standard deviations
}

pub struct VarCalculator {
    config: VarConfig,
    historical_returns: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    correlation_matrix: Arc<RwLock<Option<DMatrix<f64>>>>,
    volatility_models: HashMap<String, Box<dyn VolatilityModel + Send + Sync>>,
    last_calculation_time: Arc<RwLock<Option<Instant>>>,
}

trait VolatilityModel {
    fn forecast_volatility(&self, returns: &[f64], horizon: u32) -> f64;
    fn update(&mut self, returns: &[f64]);
}

struct EWMAModel {
    lambda: f64,
    current_variance: Option<f64>,
}

struct GARCHModel {
    alpha0: f64,  // Constant
    alpha1: f64,  // ARCH parameter
    beta1: f64,   // GARCH parameter
    current_variance: Option<f64>,
}

impl VarCalculator {
    pub fn new(config: VarConfig) -> Self {
        let mut volatility_models: HashMap<String, Box<dyn VolatilityModel + Send + Sync>> = HashMap::new();
        
        // Initialize EWMA model
        volatility_models.insert(
            "EWMA".to_string(),
            Box::new(EWMAModel::new(config.ewma_lambda))
        );
        
        // Initialize GARCH model
        volatility_models.insert(
            "GARCH".to_string(),
            Box::new(GARCHModel::new())
        );

        Self {
            config,
            historical_returns: Arc::new(RwLock::new(HashMap::new())),
            correlation_matrix: Arc::new(RwLock::new(None)),
            volatility_models,
            last_calculation_time: Arc::new(RwLock::new(None)),
        }
    }

    /// Calculate VaR for a portfolio using specified method
    pub fn calculate_var(&mut self, portfolio: &Portfolio, method: VarMethod) -> Result<VarResult> {
        let start_time = Instant::now();

        // Update historical data
        self.update_historical_data(portfolio)?;
        
        let portfolio_value = portfolio.get_total_value();
        if portfolio_value <= 0.0 {
            return Err(AlgoVedaError::Risk("Portfolio has no positive value".to_string()));
        }

        let var_result = match method {
            VarMethod::HistoricalSimulation => self.historical_simulation_var(portfolio)?,
            VarMethod::ParametricNormal => self.parametric_normal_var(portfolio)?,
            VarMethod::ParametricTDistribution => self.parametric_t_var(portfolio)?,
            VarMethod::MonteCarlo => self.monte_carlo_var(portfolio)?,
            VarMethod::ExtremeValue => self.extreme_value_var(portfolio)?,
            VarMethod::EWMA => self.ewma_var(portfolio)?,
            VarMethod::GARCH => self.garch_var(portfolio)?,
        };

        // Update calculation time
        *self.last_calculation_time.write().unwrap() = Some(start_time);

        Ok(VarResult {
            var_absolute: var_result.var_absolute,
            var_percentage: var_result.var_percentage,
            expected_shortfall: var_result.expected_shortfall,
            component_var: if self.config.enable_component_var { 
                self.calculate_component_var(portfolio, &var_result)? 
            } else { 
                HashMap::new() 
            },
            marginal_var: if self.config.enable_marginal_var { 
                self.calculate_marginal_var(portfolio)? 
            } else { 
                HashMap::new() 
            },
            diversification_ratio: self.calculate_diversification_ratio(portfolio)?,
            method,
            confidence_level: self.config.confidence_level,
            time_horizon_days: self.config.time_horizon_days,
            calculation_time_ms: start_time.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
        })
    }

    /// Historical Simulation VaR
    fn historical_simulation_var(&self, portfolio: &Portfolio) -> Result<VarResult> {
        let historical_returns = self.historical_returns.read().unwrap();
        let positions = portfolio.get_positions();
        
        if positions.is_empty() {
            return Err(AlgoVedaError::Risk("Portfolio has no positions".to_string()));
        }

        // Calculate portfolio returns for each historical period
        let mut portfolio_returns = Vec::new();
        let lookback = self.config.lookback_days as usize;
        
        for period in 0..lookback {
            let mut period_return = 0.0;
            let mut total_weight = 0.0;
            
            for position in positions {
                if let Some(symbol_returns) = historical_returns.get(&position.symbol) {
                    if period < symbol_returns.len() {
                        let weight = position.market_value / portfolio.get_total_value();
                        period_return += weight * symbol_returns[symbol_returns.len() - 1 - period];
                        total_weight += weight.abs();
                    }
                }
            }
            
            if total_weight > 0.0 {
                portfolio_returns.push(period_return);
            }
        }

        if portfolio_returns.is_empty() {
            return Err(AlgoVedaError::Risk("No historical returns data available".to_string()));
        }

        // Sort returns for percentile calculation
        portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Calculate VaR and Expected Shortfall
        let var_percentile = (1.0 - self.config.confidence_level) * 100.0;
        let var_return = percentile(&portfolio_returns, var_percentile);
        let var_absolute = -var_return * portfolio.get_total_value();
        let var_percentage = -var_return;

        // Calculate Expected Shortfall (average of losses beyond VaR)
        let tail_losses: Vec<f64> = portfolio_returns.iter()
            .filter(|&&ret| ret <= var_return)
            .cloned()
            .collect();
        
        let expected_shortfall = if !tail_losses.is_empty() {
            -tail_losses.iter().sum::<f64>() / tail_losses.len() as f64 * portfolio.get_total_value()
        } else {
            var_absolute
        };

        Ok(VarResult {
            var_absolute,
            var_percentage,
            expected_shortfall,
            component_var: HashMap::new(),
            marginal_var: HashMap::new(),
            diversification_ratio: 1.0,
            method: VarMethod::HistoricalSimulation,
            confidence_level: self.config.confidence_level,
            time_horizon_days: self.config.time_horizon_days,
            calculation_time_ms: 0,
            timestamp: Utc::now(),
        })
    }

    /// Parametric VaR assuming normal distribution
    fn parametric_normal_var(&self, portfolio: &Portfolio) -> Result<VarResult> {
        let positions = portfolio.get_positions();
        let n_assets = positions.len();
        
        if n_assets == 0 {
            return Err(AlgoVedaError::Risk("Portfolio has no positions".to_string()));
        }

        // Calculate portfolio weights
        let total_value = portfolio.get_total_value();
        let weights = positions.iter()
            .map(|pos| pos.market_value / total_value)
            .collect::<Vec<f64>>();

        // Get historical returns for variance calculation
        let historical_returns = self.historical_returns.read().unwrap();
        let mut asset_returns: Vec<Vec<f64>> = Vec::new();
        let mut asset_volatilities: Vec<f64> = Vec::new();

        for position in positions {
            if let Some(returns) = historical_returns.get(&position.symbol) {
                let recent_returns = returns.iter()
                    .rev()
                    .take(self.config.lookback_days as usize)
                    .cloned()
                    .collect::<Vec<f64>>();
                
                if !recent_returns.is_empty() {
                    let volatility = self.calculate_volatility(&recent_returns);
                    asset_returns.push(recent_returns);
                    asset_volatilities.push(volatility);
                } else {
                    return Err(AlgoVedaError::Risk(format!("No returns data for {}", position.symbol)));
                }
            } else {
                return Err(AlgoVedaError::Risk(format!("No historical data for {}", position.symbol)));
            }
        }

        // Calculate correlation matrix
        let correlation_matrix = self.calculate_correlation_matrix(&asset_returns)?;
        
        // Calculate portfolio variance
        let mut portfolio_variance = 0.0;
        for i in 0..n_assets {
            for j in 0..n_assets {
                portfolio_variance += weights[i] * weights[j] * 
                    asset_volatilities[i] * asset_volatilities[j] * 
                    correlation_matrix[(i, j)];
            }
        }

        let portfolio_volatility = portfolio_variance.sqrt();
        
        // Adjust for time horizon
        let adjusted_volatility = portfolio_volatility * (self.config.time_horizon_days as f64).sqrt();
        
        // Calculate VaR using normal distribution
        let z_score = self.inverse_normal_cdf(self.config.confidence_level);
        let var_percentage = z_score * adjusted_volatility;
        let var_absolute = var_percentage * total_value;

        // Expected Shortfall for normal distribution
        let expected_shortfall = self.normal_expected_shortfall(adjusted_volatility, self.config.confidence_level) * total_value;

        Ok(VarResult {
            var_absolute,
            var_percentage,
            expected_shortfall,
            component_var: HashMap::new(),
            marginal_var: HashMap::new(),
            diversification_ratio: 1.0,
            method: VarMethod::ParametricNormal,
            confidence_level: self.config.confidence_level,
            time_horizon_days: self.config.time_horizon_days,
            calculation_time_ms: 0,
            timestamp: Utc::now(),
        })
    }

    /// Monte Carlo VaR simulation
    fn monte_carlo_var(&self, portfolio: &Portfolio) -> Result<VarResult> {
        let positions = portfolio.get_positions();
        let n_assets = positions.len();
        let n_simulations = self.config.monte_carlo_simulations;
        
        if n_assets == 0 {
            return Err(AlgoVedaError::Risk("Portfolio has no positions".to_string()));
        }

        // Get asset parameters
        let historical_returns = self.historical_returns.read().unwrap();
        let mut asset_means: Vec<f64> = Vec::new();
        let mut asset_volatilities: Vec<f64> = Vec::new();
        let mut asset_returns_data: Vec<Vec<f64>> = Vec::new();

        for position in positions {
            if let Some(returns) = historical_returns.get(&position.symbol) {
                let recent_returns = returns.iter()
                    .rev()
                    .take(self.config.lookback_days as usize)
                    .cloned()
                    .collect::<Vec<f64>>();
                
                let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
                let volatility = self.calculate_volatility(&recent_returns);
                
                asset_means.push(mean);
                asset_volatilities.push(volatility);
                asset_returns_data.push(recent_returns);
            } else {
                return Err(AlgoVedaError::Risk(format!("No historical data for {}", position.symbol)));
            }
        }

        // Calculate correlation matrix and Cholesky decomposition
        let correlation_matrix = self.calculate_correlation_matrix(&asset_returns_data)?;
        let cholesky = self.cholesky_decomposition(&correlation_matrix)?;

        // Portfolio weights
        let total_value = portfolio.get_total_value();
        let weights: Vec<f64> = positions.iter()
            .map(|pos| pos.market_value / total_value)
            .collect();

        // Monte Carlo simulation
        let mut portfolio_returns: Vec<f64> = Vec::with_capacity(n_simulations);
        let mut rng = rand::thread_rng();

        for _ in 0..n_simulations {
            // Generate correlated random returns
            let mut random_returns = vec![0.0; n_assets];
            let independent_normals: Vec<f64> = (0..n_assets)
                .map(|_| rand_distr::StandardNormal.sample(&mut rng))
                .collect();

            // Apply Cholesky decomposition for correlation
            for i in 0..n_assets {
                for j in 0..=i {
                    random_returns[i] += cholesky[(i, j)] * independent_normals[j];
                }
                
                // Scale by volatility and add drift
                random_returns[i] = asset_means[i] * (self.config.time_horizon_days as f64 / 252.0) + 
                    asset_volatilities[i] * random_returns[i] * (self.config.time_horizon_days as f64 / 252.0).sqrt();
            }

            // Calculate portfolio return
            let portfolio_return: f64 = weights.iter()
                .zip(random_returns.iter())
                .map(|(w, r)| w * r)
                .sum();
            
            portfolio_returns.push(portfolio_return);
        }

        // Sort for percentile calculation
        portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate VaR and Expected Shortfall
        let var_index = ((1.0 - self.config.confidence_level) * n_simulations as f64) as usize;
        let var_percentage = -portfolio_returns[var_index];
        let var_absolute = var_percentage * total_value;

        // Expected Shortfall
        let tail_losses: Vec<f64> = portfolio_returns.iter()
            .take(var_index + 1)
            .cloned()
            .collect();
        
        let expected_shortfall = if !tail_losses.is_empty() {
            -tail_losses.iter().sum::<f64>() / tail_losses.len() as f64 * total_value
        } else {
            var_absolute
        };

        Ok(VarResult {
            var_absolute,
            var_percentage,
            expected_shortfall,
            component_var: HashMap::new(),
            marginal_var: HashMap::new(),
            diversification_ratio: 1.0,
            method: VarMethod::MonteCarlo,
            confidence_level: self.config.confidence_level,
            time_horizon_days: self.config.time_horizon_days,
            calculation_time_ms: 0,
            timestamp: Utc::now(),
        })
    }

    /// EWMA-based VaR calculation
    fn ewma_var(&mut self, portfolio: &Portfolio) -> Result<VarResult> {
        let positions = portfolio.get_positions();
        let historical_returns = self.historical_returns.read().unwrap();
        
        // Update EWMA models for each asset
        for position in positions {
            if let Some(returns) = historical_returns.get(&position.symbol) {
                if let Some(ewma_model) = self.volatility_models.get_mut("EWMA") {
                    ewma_model.update(returns);
                }
            }
        }

        // Calculate EWMA-based volatilities
        let mut ewma_volatilities: Vec<f64> = Vec::new();
        for position in positions {
            if let Some(returns) = historical_returns.get(&position.symbol) {
                if let Some(ewma_model) = self.volatility_models.get("EWMA") {
                    let volatility = ewma_model.forecast_volatility(returns, self.config.time_horizon_days);
                    ewma_volatilities.push(volatility);
                } else {
                    return Err(AlgoVedaError::Risk("EWMA model not found".to_string()));
                }
            }
        }

        // Use parametric approach with EWMA volatilities
        let total_value = portfolio.get_total_value();
        let weights: Vec<f64> = positions.iter()
            .map(|pos| pos.market_value / total_value)
            .collect();

        // Simplified portfolio volatility (assuming independence for EWMA)
        let portfolio_variance: f64 = weights.iter()
            .zip(ewma_volatilities.iter())
            .map(|(w, vol)| w * w * vol * vol)
            .sum();

        let portfolio_volatility = portfolio_variance.sqrt();
        let adjusted_volatility = portfolio_volatility * (self.config.time_horizon_days as f64).sqrt();

        // Calculate VaR
        let z_score = self.inverse_normal_cdf(self.config.confidence_level);
        let var_percentage = z_score * adjusted_volatility;
        let var_absolute = var_percentage * total_value;

        let expected_shortfall = self.normal_expected_shortfall(adjusted_volatility, self.config.confidence_level) * total_value;

        Ok(VarResult {
            var_absolute,
            var_percentage,
            expected_shortfall,
            component_var: HashMap::new(),
            marginal_var: HashMap::new(),
            diversification_ratio: 1.0,
            method: VarMethod::EWMA,
            confidence_level: self.config.confidence_level,
            time_horizon_days: self.config.time_horizon_days,
            calculation_time_ms: 0,
            timestamp: Utc::now(),
        })
    }

    /// Parametric VaR with t-distribution
    fn parametric_t_var(&self, portfolio: &Portfolio) -> Result<VarResult> {
        // Implementation similar to parametric_normal_var but using t-distribution
        // This is a simplified version - full implementation would estimate degrees of freedom
        let normal_result = self.parametric_normal_var(portfolio)?;
        
        // Adjust for t-distribution (assume 10 degrees of freedom for illustration)
        let df = 10.0;
        let t_multiplier = ((df - 2.0) / df).sqrt() * self.student_t_inverse(self.config.confidence_level, df);
        let normal_multiplier = self.inverse_normal_cdf(self.config.confidence_level);
        
        let adjustment_factor = t_multiplier / normal_multiplier;
        
        Ok(VarResult {
            var_absolute: normal_result.var_absolute * adjustment_factor,
            var_percentage: normal_result.var_percentage * adjustment_factor,
            expected_shortfall: normal_result.expected_shortfall * adjustment_factor,
            method: VarMethod::ParametricTDistribution,
            ..normal_result
        })
    }

    /// Extreme Value Theory VaR
    fn extreme_value_var(&self, portfolio: &Portfolio) -> Result<VarResult> {
        // Simplified EVT implementation - would normally fit Generalized Pareto Distribution
        let hist_result = self.historical_simulation_var(portfolio)?;
        
        // Apply EVT scaling (simplified)
        let evt_scaling = 1.2; // This would be calculated from GPD parameters
        
        Ok(VarResult {
            var_absolute: hist_result.var_absolute * evt_scaling,
            var_percentage: hist_result.var_percentage * evt_scaling,
            expected_shortfall: hist_result.expected_shortfall * evt_scaling,
            method: VarMethod::ExtremeValue,
            ..hist_result
        })
    }

    /// GARCH-based VaR
    fn garch_var(&mut self, portfolio: &Portfolio) -> Result<VarResult> {
        // Similar to EWMA but using GARCH model
        let positions = portfolio.get_positions();
        let historical_returns = self.historical_returns.read().unwrap();
        
        // Update GARCH models
        for position in positions {
            if let Some(returns) = historical_returns.get(&position.symbol) {
                if let Some(garch_model) = self.volatility_models.get_mut("GARCH") {
                    garch_model.update(returns);
                }
            }
        }

        // Calculate GARCH volatilities and proceed similar to EWMA
        // Implementation would be similar to ewma_var but using GARCH forecasts
        self.ewma_var(portfolio) // Simplified - use EWMA as placeholder
    }

    /// Calculate Component VaR
    fn calculate_component_var(&self, portfolio: &Portfolio, base_var: &VarResult) -> Result<HashMap<String, f64>> {
        let mut component_vars = HashMap::new();
        let positions = portfolio.get_positions();
        
        for position in positions {
            // Simplified component VaR calculation
            // Full implementation would require marginal VaR calculation
            let weight = position.market_value / portfolio.get_total_value();
            let component_var = weight * base_var.var_absolute * 0.8; // Simplified
            component_vars.insert(position.symbol.clone(), component_var);
        }
        
        Ok(component_vars)
    }

    /// Calculate Marginal VaR
    fn calculate_marginal_var(&self, portfolio: &Portfolio) -> Result<HashMap<String, f64>> {
        let mut marginal_vars = HashMap::new();
        let positions = portfolio.get_positions();
        let base_var = self.historical_simulation_var(portfolio)?;
        
        // Calculate marginal VaR by shocking each position
        for position in positions {
            let shock_amount = position.market_value * 0.01; // 1% shock
            
            // Create modified portfolio (this would require portfolio modification methods)
            // Simplified calculation
            let marginal_var = base_var.var_absolute * 0.01; // Placeholder
            marginal_vars.insert(position.symbol.clone(), marginal_var);
        }
        
        Ok(marginal_vars)
    }

    /// Calculate diversification ratio
    fn calculate_diversification_ratio(&self, portfolio: &Portfolio) -> Result<f64> {
        let positions = portfolio.get_positions();
        if positions.len() <= 1 {
            return Ok(1.0);
        }

        // Simplified diversification ratio
        // Would normally be sum of individual VaRs / portfolio VaR
        let n_assets = positions.len() as f64;
        Ok((n_assets.sqrt() / n_assets).max(0.1).min(1.0))
    }

    /// Stress test portfolio against specific shocks
    pub fn stress_test(&self, portfolio: &Portfolio, shocks: &[RiskFactorShock]) -> Result<HashMap<String, f64>> {
        let mut stress_results = HashMap::new();
        
        for shock in shocks {
            let stressed_value = match shock.shock_type {
                ShockType::Absolute => portfolio.get_total_value() + shock.shock_magnitude,
                ShockType::Relative => portfolio.get_total_value() * (1.0 + shock.shock_magnitude),
                ShockType::StandardDeviations => {
                    // Would need volatility calculation
                    portfolio.get_total_value() * (1.0 + shock.shock_magnitude * 0.02)
                },
            };
            
            let pnl = stressed_value - portfolio.get_total_value();
            stress_results.insert(shock.factor_name.clone(), pnl);
        }
        
        Ok(stress_results)
    }

    /// Update historical returns data
    fn update_historical_data(&self, portfolio: &Portfolio) -> Result<()> {
        // This would typically fetch latest market data
        // For now, assume data is already populated
        Ok(())
    }

    /// Helper functions
    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        (variance * 252.0).sqrt() // Annualized volatility
    }

    fn calculate_correlation_matrix(&self, returns_data: &[Vec<f64>]) -> Result<DMatrix<f64>> {
        let n_assets = returns_data.len();
        let mut correlation_matrix = DMatrix::<f64>::identity(n_assets, n_assets);
        
        for i in 0..n_assets {
            for j in i+1..n_assets {
                let correlation = self.correlation(&returns_data[i], &returns_data[j]);
                correlation_matrix[(i, j)] = correlation;
                correlation_matrix[(j, i)] = correlation;
            }
        }
        
        Ok(correlation_matrix)
    }

    fn correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }
        
        let mean_x = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        
        let covariance = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>() / (x.len() - 1) as f64;
        
        let var_x = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
        let var_y = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / (y.len() - 1) as f64;
        
        if var_x > 0.0 && var_y > 0.0 {
            covariance / (var_x.sqrt() * var_y.sqrt())
        } else {
            0.0
        }
    }

    fn cholesky_decomposition(&self, matrix: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        // Simplified Cholesky - would use proper linear algebra library
        match matrix.cholesky() {
            Some(chol) => Ok(chol.l().clone()),
            None => Err(AlgoVedaError::Risk("Matrix is not positive definite".to_string())),
        }
    }

    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        // Beasley-Springer-Moro algorithm approximation
        let a = [0.0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
        let b = [0.0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
        let c = [0.0, -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
        let d = [0.0, 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            return (((((c[6] * q + c[5]) * q + c[4]) * q + c[3]) * q + c[2]) * q + c[1]) * q + c[0]) /
                   ((((d[4] * q + d[3]) * q + d[2]) * q + d[1]) * q + 1.0);
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            return (((((a[6] * r + a[5]) * r + a[4]) * r + a[3]) * r + a[2]) * r + a[1]) * r + a[0]) * q /
                   (((((b[5] * r + b[4]) * r + b[3]) * r + b[2]) * r + b[1]) * r + 1.0);
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            return -(((((c[6] * q + c[5]) * q + c[4]) * q + c[3]) * q + c[2]) * q + c[1]) * q + c[0]) /
                    ((((d[4] * q + d[3]) * q + d[2]) * q + d[1]) * q + 1.0);
        }
    }

    fn student_t_inverse(&self, p: f64, df: f64) -> f64 {
        // Simplified t-distribution inverse - would use proper statistical library
        let normal_quantile = self.inverse_normal_cdf(p);
        normal_quantile * (df / (df - 2.0)).sqrt()
    }

    fn normal_expected_shortfall(&self, volatility: f64, confidence_level: f64) -> f64 {
        let alpha = 1.0 - confidence_level;
        let z_alpha = self.inverse_normal_cdf(confidence_level);
        
        volatility / alpha * (-0.5 * z_alpha * z_alpha).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}

// Volatility model implementations
impl EWMAModel {
    fn new(lambda: f64) -> Self {
        Self {
            lambda,
            current_variance: None,
        }
    }
}

impl VolatilityModel for EWMAModel {
    fn forecast_volatility(&self, returns: &[f64], horizon: u32) -> f64 {
        if let Some(variance) = self.current_variance {
            (variance * horizon as f64).sqrt()
        } else {
            // Initialize with sample variance
            if returns.len() < 2 {
                return 0.2; // Default 20% volatility
            }
            
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            
            (variance * horizon as f64).sqrt()
        }
    }

    fn update(&mut self, returns: &[f64]) {
        if returns.is_empty() {
            return;
        }

        let latest_return = returns[returns.len() - 1];
        
        if let Some(current_var) = self.current_variance {
            // EWMA update
            self.current_variance = Some(
                self.lambda * current_var + (1.0 - self.lambda) * latest_return.powi(2)
            );
        } else {
            // Initialize
            self.current_variance = Some(latest_return.powi(2));
        }
    }
}

impl GARCHModel {
    fn new() -> Self {
        Self {
            alpha0: 0.000001, // Long-run variance
            alpha1: 0.05,     // ARCH parameter
            beta1: 0.93,      // GARCH parameter
            current_variance: None,
        }
    }
}

impl VolatilityModel for GARCHModel {
    fn forecast_volatility(&self, returns: &[f64], horizon: u32) -> f64 {
        if let Some(variance) = self.current_variance {
            // For GARCH(1,1), long-run variance forecast
            let long_run_var = self.alpha0 / (1.0 - self.alpha1 - self.beta1);
            let persistence = self.alpha1 + self.beta1;
            
            let forecast_variance = long_run_var + 
                persistence.powi(horizon as i32) * (variance - long_run_var);
            
            forecast_variance.sqrt()
        } else {
            0.2 // Default volatility
        }
    }

    fn update(&mut self, returns: &[f64]) {
        if returns.is_empty() {
            return;
        }

        let latest_return = returns[returns.len() - 1];
        
        if let Some(current_var) = self.current_variance {
            // GARCH(1,1) update
            self.current_variance = Some(
                self.alpha0 + 
                self.alpha1 * latest_return.powi(2) + 
                self.beta1 * current_var
            );
        } else {
            // Initialize with squared return
            self.current_variance = Some(latest_return.powi(2));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_calculator_creation() {
        let config = VarConfig {
            confidence_level: 0.95,
            time_horizon_days: 1,
            lookback_days: 252,
            monte_carlo_simulations: 10000,
            ewma_lambda: 0.94,
            enable_component_var: true,
            enable_marginal_var: true,
            enable_incremental_var: false,
        };
        
        let calculator = VarCalculator::new(config);
        assert_eq!(calculator.config.confidence_level, 0.95);
    }

    #[test]
    fn test_correlation_calculation() {
        let calculator = VarCalculator::new(VarConfig {
            confidence_level: 0.95,
            time_horizon_days: 1,
            lookback_days: 252,
            monte_carlo_simulations: 1000,
            ewma_lambda: 0.94,
            enable_component_var: false,
            enable_marginal_var: false,
            enable_incremental_var: false,
        });

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let correlation = calculator.correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10); // Perfect correlation
    }

    #[test]
    fn test_volatility_calculation() {
        let calculator = VarCalculator::new(VarConfig {
            confidence_level: 0.95,
            time_horizon_days: 1,
            lookback_days: 252,
            monte_carlo_simulations: 1000,
            ewma_lambda: 0.94,
            enable_component_var: false,
            enable_marginal_var: false,
            enable_incremental_var: false,
        });

        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.005];
        let volatility = calculator.calculate_volatility(&returns);
        
        assert!(volatility > 0.0);
        assert!(volatility < 1.0); // Should be reasonable
    }
}
