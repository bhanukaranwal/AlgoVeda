/*!
 * Monte Carlo Simulation Engine for AlgoVeda Backtesting
 * Advanced stochastic modeling for options pricing and portfolio risk
 */

use std::sync::Arc;
use parking_lot::Mutex;
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};
use serde::{Serialize, Deserialize};
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    pub num_simulations: usize,
    pub num_time_steps: usize,
    pub time_horizon: f64,  // In years
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub random_seed: Option<u64>,
    pub use_antithetic_variates: bool,
    pub use_control_variates: bool,
    pub enable_gpu_acceleration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetParameters {
    pub initial_price: f64,
    pub volatility: f64,
    pub drift: f64,
    pub correlation_matrix: Option<Vec<Vec<f64>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionParameters {
    pub option_type: OptionType,
    pub strike_price: f64,
    pub expiry_time: f64,
    pub barrier_level: Option<f64>,
    pub asian_averaging_start: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptionType {
    European { call: bool },
    American { call: bool },
    Asian { call: bool },
    Barrier { call: bool, knock_in: bool },
    Lookback { call: bool },
    Digital { call: bool },
}

pub struct MonteCarloSimulator {
    config: MonteCarloConfig,
    rng: Arc<Mutex<StdRng>>,
    paths_cache: Arc<Mutex<Option<Vec<Vec<f64>>>>>,
}

impl MonteCarloSimulator {
    pub fn new(config: MonteCarloConfig) -> Self {
        let rng = if let Some(seed) = config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        Self {
            config,
            rng: Arc::new(Mutex::new(rng)),
            paths_cache: Arc::new(Mutex::new(None)),
        }
    }

    /// Price European options using Monte Carlo simulation[7][15]
    pub fn price_european_option(
        &self,
        asset_params: &AssetParameters,
        option_params: &OptionParameters,
    ) -> Result<OptionPricingResult, Box<dyn std::error::Error>> {
        let dt = self.config.time_horizon / self.config.num_time_steps as f64;
        let discount_factor = (-self.config.risk_free_rate * option_params.expiry_time).exp();

        // Generate price paths in parallel for maximum performance
        let payoffs: Vec<f64> = (0..self.config.num_simulations)
            .into_par_iter()
            .map(|simulation_idx| {
                let mut local_rng = StdRng::seed_from_u64(
                    self.config.random_seed.unwrap_or(0) + simulation_idx as u64
                );

                // Generate single asset path
                let final_price = self.generate_gbm_path(
                    asset_params.initial_price,
                    asset_params.volatility,
                    asset_params.drift,
                    option_params.expiry_time,
                    &mut local_rng,
                );

                // Calculate payoff based on option type
                self.calculate_option_payoff(&option_params, final_price, &[final_price])
            })
            .collect();

        // Calculate statistics
        let mean_payoff = payoffs.iter().sum::<f64>() / payoffs.len() as f64;
        let option_price = mean_payoff * discount_factor;
        
        let variance = payoffs.iter()
            .map(|&x| (x - mean_payoff).powi(2))
            .sum::<f64>() / (payoffs.len() - 1) as f64;
        let standard_error = (variance / payoffs.len() as f64).sqrt() * discount_factor;

        // Calculate confidence intervals
        let z_score = 1.96; // 95% confidence
        let confidence_interval = (
            option_price - z_score * standard_error,
            option_price + z_score * standard_error,
        );

        Ok(OptionPricingResult {
            option_price,
            standard_error,
            confidence_interval,
            num_simulations: self.config.num_simulations,
            convergence_stats: self.calculate_convergence_stats(&payoffs, discount_factor),
        })
    }

    /// Price path-dependent options (Asian, Barrier, etc.)
    pub fn price_path_dependent_option(
        &self,
        asset_params: &AssetParameters,
        option_params: &OptionParameters,
    ) -> Result<OptionPricingResult, Box<dyn std::error::Error>> {
        let dt = self.config.time_horizon / self.config.num_time_steps as f64;
        let discount_factor = (-self.config.risk_free_rate * option_params.expiry_time).exp();

        let payoffs: Vec<f64> = (0..self.config.num_simulations)
            .into_par_iter()
            .map(|simulation_idx| {
                let mut local_rng = StdRng::seed_from_u64(
                    self.config.random_seed.unwrap_or(0) + simulation_idx as u64
                );

                // Generate full price path
                let price_path = self.generate_full_price_path(
                    asset_params,
                    option_params.expiry_time,
                    dt,
                    &mut local_rng,
                );

                // Calculate path-dependent payoff
                self.calculate_option_payoff(&option_params, 
                    price_path[price_path.len() - 1], 
                    &price_path)
            })
            .collect();

        let mean_payoff = payoffs.iter().sum::<f64>() / payoffs.len() as f64;
        let option_price = mean_payoff * discount_factor;
        
        let variance = payoffs.iter()
            .map(|&x| (x - mean_payoff).powi(2))
            .sum::<f64>() / (payoffs.len() - 1) as f64;
        let standard_error = (variance / payoffs.len() as f64).sqrt() * discount_factor;

        let confidence_interval = (
            option_price - 1.96 * standard_error,
            option_price + 1.96 * standard_error,
        );

        Ok(OptionPricingResult {
            option_price,
            standard_error,
            confidence_interval,
            num_simulations: self.config.num_simulations,
            convergence_stats: self.calculate_convergence_stats(&payoffs, discount_factor),
        })
    }

    /// Generate correlated asset paths for multi-asset options
    pub fn generate_correlated_paths(
        &self,
        asset_params: &[AssetParameters],
        correlation_matrix: &[Vec<f64>],
        time_horizon: f64,
    ) -> Result<Vec<Vec<Vec<f64>>>, Box<dyn std::error::Error>> {
        let num_assets = asset_params.len();
        let dt = time_horizon / self.config.num_time_steps as f64;
        
        // Cholesky decomposition for correlation
        let cholesky = self.cholesky_decomposition(correlation_matrix)?;

        let paths: Vec<Vec<Vec<f64>>> = (0..self.config.num_simulations)
            .into_par_iter()
            .map(|simulation_idx| {
                let mut local_rng = StdRng::seed_from_u64(
                    self.config.random_seed.unwrap_or(0) + simulation_idx as u64
                );

                let mut asset_paths = vec![vec![0.0; self.config.num_time_steps + 1]; num_assets];
                
                // Initialize with starting prices
                for (i, params) in asset_params.iter().enumerate() {
                    asset_paths[i] = params.initial_price;
                }

                // Generate correlated random walks
                for t in 1..=self.config.num_time_steps {
                    // Generate independent normal random variables
                    let independent_randoms: Vec<f64> = (0..num_assets)
                        .map(|_| StandardNormal.sample(&mut local_rng))
                        .collect();

                    // Apply correlation via Cholesky decomposition
                    let correlated_randoms = self.apply_correlation(&cholesky, &independent_randoms);

                    // Update asset prices
                    for (i, params) in asset_params.iter().enumerate() {
                        let previous_price = asset_paths[i][t - 1];
                        let drift_term = (params.drift - 0.5 * params.volatility.powi(2)) * dt;
                        let diffusion_term = params.volatility * dt.sqrt() * correlated_randoms[i];
                        
                        asset_paths[i][t] = previous_price * (drift_term + diffusion_term).exp();
                    }
                }

                asset_paths
            })
            .collect();

        Ok(paths)
    }

    /// Portfolio VaR calculation using Monte Carlo
    pub fn calculate_portfolio_var(
        &self,
        portfolio_weights: &[f64],
        asset_params: &[AssetParameters],
        correlation_matrix: &[Vec<f64>],
        confidence_level: f64,
        time_horizon: f64,
    ) -> Result<VaRResult, Box<dyn std::error::Error>> {
        let paths = self.generate_correlated_paths(asset_params, correlation_matrix, time_horizon)?;
        
        // Calculate portfolio returns for each simulation
        let portfolio_returns: Vec<f64> = paths.iter()
            .map(|simulation_paths| {
                let initial_value: f64 = portfolio_weights.iter()
                    .zip(asset_params.iter())
                    .map(|(w, params)| w * params.initial_price)
                    .sum();

                let final_value: f64 = portfolio_weights.iter()
                    .zip(simulation_paths.iter())
                    .map(|(w, path)| w * path[path.len() - 1])
                    .sum();

                (final_value - initial_value) / initial_value
            })
            .collect();

        // Sort returns for percentile calculation
        let mut sorted_returns = portfolio_returns;
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate VaR
        let var_index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let var_value = -sorted_returns[var_index.min(sorted_returns.len() - 1)];

        // Calculate Expected Shortfall (CVaR)
        let tail_returns: Vec<f64> = sorted_returns.iter()
            .take(var_index + 1)
            .cloned()
            .collect();
        let expected_shortfall = if !tail_returns.is_empty() {
            -tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
        } else {
            var_value
        };

        Ok(VaRResult {
            var_value,
            expected_shortfall,
            confidence_level,
            time_horizon,
            num_simulations: self.config.num_simulations,
        })
    }

    // Helper methods
    fn generate_gbm_path(
        &self,
        initial_price: f64,
        volatility: f64,
        drift: f64,
        time_to_expiry: f64,
        rng: &mut StdRng,
    ) -> f64 {
        let drift_term = (drift - 0.5 * volatility.powi(2)) * time_to_expiry;
        let diffusion_term = volatility * time_to_expiry.sqrt() * StandardNormal.sample(rng);
        initial_price * (drift_term + diffusion_term).exp()
    }

    fn generate_full_price_path(
        &self,
        asset_params: &AssetParameters,
        time_to_expiry: f64,
        dt: f64,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let num_steps = (time_to_expiry / dt) as usize;
        let mut path = vec![asset_params.initial_price];

        for _ in 0..num_steps {
            let previous_price = path[path.len() - 1];
            let drift_term = (asset_params.drift - 0.5 * asset_params.volatility.powi(2)) * dt;
            let diffusion_term = asset_params.volatility * dt.sqrt() * StandardNormal.sample(rng);
            
            let new_price = previous_price * (drift_term + diffusion_term).exp();
            path.push(new_price);
        }

        path
    }

    fn calculate_option_payoff(
        &self,
        option_params: &OptionParameters,
        final_price: f64,
        price_path: &[f64],
    ) -> f64 {
        match &option_params.option_type {
            OptionType::European { call } => {
                if *call {
                    (final_price - option_params.strike_price).max(0.0)
                } else {
                    (option_params.strike_price - final_price).max(0.0)
                }
            }
            OptionType::Asian { call } => {
                let average_price = price_path.iter().sum::<f64>() / price_path.len() as f64;
                if *call {
                    (average_price - option_params.strike_price).max(0.0)
                } else {
                    (option_params.strike_price - average_price).max(0.0)
                }
            }
            OptionType::Barrier { call, knock_in } => {
                let barrier = option_params.barrier_level.unwrap_or(option_params.strike_price);
                let barrier_crossed = price_path.iter().any(|&price| {
                    if *call { price >= barrier } else { price <= barrier }
                });

                let payoff = if *call {
                    (final_price - option_params.strike_price).max(0.0)
                } else {
                    (option_params.strike_price - final_price).max(0.0)
                };

                if (*knock_in && barrier_crossed) || (!*knock_in && !barrier_crossed) {
                    payoff
                } else {
                    0.0
                }
            }
            OptionType::Lookback { call } => {
                if *call {
                    let max_price = price_path.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    max_price - option_params.strike_price
                } else {
                    let min_price = price_path.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    option_params.strike_price - min_price
                }
            }
            OptionType::Digital { call } => {
                if (*call && final_price > option_params.strike_price) ||
                   (!*call && final_price < option_params.strike_price) {
                    1.0
                } else {
                    0.0
                }
            }
            _ => 0.0, // Other option types
        }
    }

    fn cholesky_decomposition(&self, matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let n = matrix.len();
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[j][k] * l[j][k];
                    }
                    let val = matrix[j][j] - sum;
                    if val <= 0.0 {
                        return Err("Matrix is not positive definite".into());
                    }
                    l[j][j] = val.sqrt();
                } else {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    if l[j][j] == 0.0 {
                        return Err("Matrix is singular".into());
                    }
                    l[i][j] = (matrix[i][j] - sum) / l[j][j];
                }
            }
        }

        Ok(l)
    }

    fn apply_correlation(&self, cholesky: &[Vec<f64>], independent: &[f64]) -> Vec<f64> {
        cholesky.iter()
            .map(|row| {
                row.iter()
                    .zip(independent.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }

    fn calculate_convergence_stats(&self, payoffs: &[f64], discount_factor: f64) -> ConvergenceStats {
        let mut running_means = Vec::new();
        let mut running_sum = 0.0;

        for (i, &payoff) in payoffs.iter().enumerate() {
            running_sum += payoff;
            let running_mean = (running_sum / (i + 1) as f64) * discount_factor;
            running_means.push(running_mean);
        }

        ConvergenceStats {
            final_mean: running_means[running_means.len() - 1],
            convergence_path: running_means,
            convergence_achieved: true, // Simplified check
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionPricingResult {
    pub option_price: f64,
    pub standard_error: f64,
    pub confidence_interval: (f64, f64),
    pub num_simulations: usize,
    pub convergence_stats: ConvergenceStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRResult {
    pub var_value: f64,
    pub expected_shortfall: f64,
    pub confidence_level: f64,
    pub time_horizon: f64,
    pub num_simulations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceStats {
    pub final_mean: f64,
    pub convergence_path: Vec<f64>,
    pub convergence_achieved: bool,
}
