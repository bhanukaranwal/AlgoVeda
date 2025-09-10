/*!
 * Portfolio Value at Risk (VaR) Calculations for AlgoVeda
 * Advanced VaR models with Monte Carlo simulation integration
 */

use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};
use serde::{Serialize, Deserialize};
use statrs::distribution::{ContinuousCDF, Normal as StatNormal};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRCalculationInput {
    pub portfolio_weights: Vec<f64>,
    pub asset_returns: Vec<Vec<f64>>, // Historical returns matrix
    pub confidence_levels: Vec<f64>,   // e.g., [0.95, 0.99]
    pub holding_period: u32,           // Days
    pub portfolio_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaRResult {
    pub confidence_level: f64,
    pub var_absolute: f64,
    pub var_percentage: f64,
    pub expected_shortfall: f64,
    pub method: VaRMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VaRMethod {
    Parametric,
    Historical,
    MonteCarlo,
    CornishFisher,
}

pub struct PortfolioVaRCalculator {
    num_simulations: usize,
    random_seed: Option<u64>,
}

impl PortfolioVaRCalculator {
    pub fn new(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            random_seed: None,
        }
    }

    /// Calculate VaR using multiple methods based on research findings [2][3]
    pub fn calculate_portfolio_var(&self, input: &VaRCalculationInput) -> Vec<VaRResult> {
        let mut results = Vec::new();

        for &confidence_level in &input.confidence_levels {
            // Parametric VaR (Variance-Covariance Method)
            if let Some(parametric_var) = self.calculate_parametric_var(input, confidence_level) {
                results.push(parametric_var);
            }

            // Historical VaR
            if let Some(historical_var) = self.calculate_historical_var(input, confidence_level) {
                results.push(historical_var);
            }

            // Monte Carlo VaR
            if let Some(monte_carlo_var) = self.calculate_monte_carlo_var(input, confidence_level) {
                results.push(monte_carlo_var);
            }

            // Cornish-Fisher VaR (accounts for skewness and kurtosis)
            if let Some(cf_var) = self.calculate_cornish_fisher_var(input, confidence_level) {
                results.push(cf_var);
            }
        }

        results
    }

    /// Parametric VaR using variance-covariance method [3]
    fn calculate_parametric_var(&self, input: &VaRCalculationInput, confidence_level: f64) -> Option<VaRResult> {
        let n_assets = input.portfolio_weights.len();
        let n_observations = input.asset_returns.len();

        if n_assets == 0 || n_observations < 30 {
            return None;
        }

        // Calculate mean returns
        let mean_returns: Vec<f64> = input.asset_returns.iter()
            .map(|returns| returns.iter().sum::<f64>() / returns.len() as f64)
            .collect();

        // Calculate covariance matrix
        let covariance_matrix = self.calculate_covariance_matrix(&input.asset_returns, &mean_returns);

        // Portfolio expected return
        let portfolio_return: f64 = input.portfolio_weights.iter()
            .zip(mean_returns.iter())
            .map(|(w, r)| w * r)
            .sum();

        // Portfolio variance calculation: w^T * Σ * w
        let portfolio_variance = self.calculate_portfolio_variance(&input.portfolio_weights, &covariance_matrix);
        let portfolio_std = portfolio_variance.sqrt();

        // Adjust for holding period
        let holding_period_factor = (input.holding_period as f64).sqrt();
        let adjusted_return = portfolio_return * input.holding_period as f64;
        let adjusted_std = portfolio_std * holding_period_factor;

        // Calculate z-score for confidence level
        let normal = StatNormal::new(0.0, 1.0).unwrap();
        let z_score = normal.inverse_cdf(1.0 - confidence_level);

        // VaR calculation: VaR = -(μ + z*σ) * W
        let var_percentage = -(adjusted_return + z_score * adjusted_std);
        let var_absolute = var_percentage * input.portfolio_value;

        // Expected Shortfall (CVaR) - simplified calculation
        let expected_shortfall = self.calculate_expected_shortfall_parametric(
            adjusted_return, adjusted_std, confidence_level, input.portfolio_value
        );

        Some(VaRResult {
            confidence_level,
            var_absolute,
            var_percentage,
            expected_shortfall,
            method: VaRMethod::Parametric,
        })
    }

    /// Historical VaR using empirical distribution [4]
    fn calculate_historical_var(&self, input: &VaRCalculationInput, confidence_level: f64) -> Option<VaRResult> {
        if input.asset_returns.is_empty() || input.asset_returns.len() < 100 {
            return None;
        }

        let n_observations = input.asset_returns.len();
        let mut portfolio_returns = Vec::with_capacity(n_observations);

        // Calculate historical portfolio returns
        for i in 0..n_observations {
            let portfolio_return: f64 = input.portfolio_weights.iter()
                .zip(input.asset_returns.iter())
                .map(|(w, returns)| w * returns[i])
                .sum();
            portfolio_returns.push(portfolio_return);
        }

        // Sort returns in ascending order
        portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Find VaR percentile
        let percentile_index = ((1.0 - confidence_level) * portfolio_returns.len() as f64) as usize;
        let var_percentage = -portfolio_returns[percentile_index.min(portfolio_returns.len() - 1)];
        let var_absolute = var_percentage * input.portfolio_value;

        // Calculate Expected Shortfall
        let tail_returns: Vec<f64> = portfolio_returns.iter()
            .take(percentile_index + 1)
            .cloned()
            .collect();
        
        let expected_shortfall = if !tail_returns.is_empty() {
            let avg_tail_loss = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
            -avg_tail_loss * input.portfolio_value
        } else {
            var_absolute
        };

        Some(VaRResult {
            confidence_level,
            var_absolute,
            var_percentage,
            expected_shortfall,
            method: VaRMethod::Historical,
        })
    }

    /// Monte Carlo VaR simulation [8][14]
    fn calculate_monte_carlo_var(&self, input: &VaRCalculationInput, confidence_level: f64) -> Option<VaRResult> {
        let n_assets = input.portfolio_weights.len();
        if n_assets == 0 || input.asset_returns.len() < 30 {
            return None;
        }

        let mut rng = if let Some(seed) = self.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        // Calculate mean returns and covariance matrix
        let mean_returns: Vec<f64> = input.asset_returns.iter()
            .map(|returns| returns.iter().sum::<f64>() / returns.len() as f64)
            .collect();

        let covariance_matrix = self.calculate_covariance_matrix(&input.asset_returns, &mean_returns);
        let cholesky_decomp = self.cholesky_decomposition(&covariance_matrix)?;

        let mut simulated_returns = Vec::with_capacity(self.num_simulations);

        // Run Monte Carlo simulation
        for _ in 0..self.num_simulations {
            // Generate random normal variables
            let random_normals: Vec<f64> = (0..n_assets)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();

            // Apply Cholesky decomposition for correlation
            let correlated_returns = self.matrix_vector_multiply(&cholesky_decomp, &random_normals);

            // Calculate portfolio return for this simulation
            let portfolio_return: f64 = input.portfolio_weights.iter()
                .zip(mean_returns.iter())
                .zip(correlated_returns.iter())
                .map(|((w, mean), shock)| w * (mean + shock))
                .sum();

            // Adjust for holding period
            let adjusted_return = portfolio_return * (input.holding_period as f64).sqrt();
            simulated_returns.push(adjusted_return);
        }

        // Sort simulated returns
        simulated_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate VaR
        let percentile_index = ((1.0 - confidence_level) * self.num_simulations as f64) as usize;
        let var_percentage = -simulated_returns[percentile_index.min(self.num_simulations - 1)];
        let var_absolute = var_percentage * input.portfolio_value;

        // Calculate Expected Shortfall
        let tail_returns: Vec<f64> = simulated_returns.iter()
            .take(percentile_index + 1)
            .cloned()
            .collect();
        
        let expected_shortfall = if !tail_returns.is_empty() {
            let avg_tail_loss = tail_returns.iter().sum::<f64>() / tail_returns.len() as f64;
            -avg_tail_loss * input.portfolio_value
        } else {
            var_absolute
        };

        Some(VaRResult {
            confidence_level,
            var_absolute,
            var_percentage,
            expected_shortfall,
            method: VaRMethod::MonteCarlo,
        })
    }

    /// Cornish-Fisher VaR accounting for skewness and kurtosis
    fn calculate_cornish_fisher_var(&self, input: &VaRCalculationInput, confidence_level: f64) -> Option<VaRResult> {
        let n_observations = input.asset_returns.len();
        if n_observations < 50 {
            return None;
        }

        // Calculate portfolio returns
        let mut portfolio_returns = Vec::with_capacity(n_observations);
        for i in 0..n_observations {
            let portfolio_return: f64 = input.portfolio_weights.iter()
                .zip(input.asset_returns.iter())
                .map(|(w, returns)| w * returns[i])
                .sum();
            portfolio_returns.push(portfolio_return);
        }

        // Calculate moments
        let mean = portfolio_returns.iter().sum::<f64>() / portfolio_returns.len() as f64;
        let variance = portfolio_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (portfolio_returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        let skewness = portfolio_returns.iter()
            .map(|r| ((r - mean) / std_dev).powi(3))
            .sum::<f64>() / portfolio_returns.len() as f64;

        let kurtosis = portfolio_returns.iter()
            .map(|r| ((r - mean) / std_dev).powi(4))
            .sum::<f64>() / portfolio_returns.len() as f64 - 3.0;

        // Calculate Cornish-Fisher adjusted quantile
        let normal = StatNormal::new(0.0, 1.0).unwrap();
        let z = normal.inverse_cdf(1.0 - confidence_level);
        
        let z_cf = z + (z * z - 1.0) * skewness / 6.0 + 
                   (z * z * z - 3.0 * z) * kurtosis / 24.0 - 
                   (2.0 * z * z * z - 5.0 * z) * skewness * skewness / 36.0;

        // Adjust for holding period
        let holding_period_factor = (input.holding_period as f64).sqrt();
        let adjusted_mean = mean * input.holding_period as f64;
        let adjusted_std = std_dev * holding_period_factor;

        let var_percentage = -(adjusted_mean + z_cf * adjusted_std);
        let var_absolute = var_percentage * input.portfolio_value;

        let expected_shortfall = self.calculate_expected_shortfall_parametric(
            adjusted_mean, adjusted_std, confidence_level, input.portfolio_value
        );

        Some(VaRResult {
            confidence_level,
            var_absolute,
            var_percentage,
            expected_shortfall,
            method: VaRMethod::CornishFisher,
        })
    }

    // Helper methods
    fn calculate_covariance_matrix(&self, returns: &[Vec<f64>], means: &[f64]) -> Vec<Vec<f64>> {
        let n_assets = returns.len();
        let n_observations = returns.len();
        let mut cov_matrix = vec![vec![0.0; n_assets]; n_assets];

        for i in 0..n_assets {
            for j in 0..n_assets {
                let mut covariance = 0.0;
                for k in 0..n_observations {
                    covariance += (returns[i][k] - means[i]) * (returns[j][k] - means[j]);
                }
                cov_matrix[i][j] = covariance / (n_observations - 1) as f64;
            }
        }

        cov_matrix
    }

    fn calculate_portfolio_variance(&self, weights: &[f64], cov_matrix: &[Vec<f64>]) -> f64 {
        let n = weights.len();
        let mut variance = 0.0;

        for i in 0..n {
            for j in 0..n {
                variance += weights[i] * weights[j] * cov_matrix[i][j];
            }
        }

        variance
    }

    fn cholesky_decomposition(&self, matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
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
                        return None;
                    }
                    l[j][j] = val.sqrt();
                } else {
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[i][k] * l[j][k];
                    }
                    if l[j][j] == 0.0 {
                        return None;
                    }
                    l[i][j] = (matrix[i][j] - sum) / l[j][j];
                }
            }
        }

        Some(l)
    }

    fn matrix_vector_multiply(&self, matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        matrix.iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }

    fn calculate_expected_shortfall_parametric(&self, mean: f64, std_dev: f64, confidence_level: f64, portfolio_value: f64) -> f64 {
        let normal = StatNormal::new(0.0, 1.0).unwrap();
        let z = normal.inverse_cdf(1.0 - confidence_level);
        let pdf_at_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        
        let expected_shortfall_percentage = -(mean + std_dev * pdf_at_z / (1.0 - confidence_level));
        expected_shortfall_percentage * portfolio_value
    }
}
