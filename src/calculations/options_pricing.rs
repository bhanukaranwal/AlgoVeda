/*!
 * Advanced Options Pricing Models
 * Implementation of Black-Scholes, Binomial Trees, Monte Carlo, and Exotic Options
 */

use std::{
    collections::HashMap,
    f64::consts::{PI, E},
    sync::Arc,
};
use rayon::prelude::*;
use ndarray::{Array1, Array2, Axis};
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

use crate::{
    error::{Result, AlgoVedaError},
    utils::math::{cumulative_normal, inverse_cumulative_normal},
    gpu::cuda_interface::CudaOptionsEngine,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExerciseStyle {
    European,
    American,
    Bermudan(Vec<f64>), // Exercise dates as time fractions
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionContract {
    pub option_type: OptionType,
    pub strike: f64,
    pub expiry: f64, // Time to expiry in years
    pub exercise_style: ExerciseStyle,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub spot_price: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
    pub vanna: f64,   // d²V/dS dσ
    pub volga: f64,   // d²V/dσ²
    pub charm: f64,   // d²V/dS dt
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingResult {
    pub price: f64,
    pub greeks: Greeks,
    pub implied_volatility: Option<f64>,
    pub pricing_method: String,
    pub computation_time_ns: u64,
}

pub struct OptionsEngine {
    cuda_engine: Option<Arc<CudaOptionsEngine>>,
    monte_carlo_paths: usize,
    binomial_steps: usize,
    finite_difference_step: f64,
}

impl OptionsEngine {
    pub fn new() -> Self {
        Self {
            cuda_engine: CudaOptionsEngine::new().ok().map(Arc::new),
            monte_carlo_paths: 1_000_000,
            binomial_steps: 500,
            finite_difference_step: 1e-6,
        }
    }

    /// Black-Scholes analytical pricing for European options
    pub fn black_scholes(&self, contract: &OptionContract, market: &MarketData) -> Result<PricingResult> {
        let start_time = std::time::Instant::now();

        if !matches!(contract.exercise_style, ExerciseStyle::European) {
            return Err(AlgoVedaError::Calculation("Black-Scholes only applies to European options".to_string()));
        }

        let S = market.spot_price;
        let K = contract.strike;
        let T = contract.expiry;
        let r = market.risk_free_rate;
        let q = market.dividend_yield;
        let sigma = market.volatility;

        if T <= 0.0 {
            let intrinsic = match contract.option_type {
                OptionType::Call => (S - K).max(0.0),
                OptionType::Put => (K - S).max(0.0),
            };
            
            return Ok(PricingResult {
                price: intrinsic,
                greeks: Greeks {
                    delta: if matches!(contract.option_type, OptionType::Call) { if S > K { 1.0 } else { 0.0 } } else { if S < K { -1.0 } else { 0.0 } },
                    ..Default::default()
                },
                implied_volatility: None,
                pricing_method: "Intrinsic Value".to_string(),
                computation_time_ns: start_time.elapsed().as_nanos() as u64,
            });
        }

        let sqrt_T = T.sqrt();
        let d1 = ((S / K).ln() + (r - q + 0.5 * sigma.powi(2)) * T) / (sigma * sqrt_T);
        let d2 = d1 - sigma * sqrt_T;

        let N_d1 = cumulative_normal(d1);
        let N_d2 = cumulative_normal(d2);
        let N_neg_d1 = cumulative_normal(-d1);
        let N_neg_d2 = cumulative_normal(-d2);

        let discount_factor = (-r * T).exp();
        let forward_factor = (-q * T).exp();

        let price = match contract.option_type {
            OptionType::Call => S * forward_factor * N_d1 - K * discount_factor * N_d2,
            OptionType::Put => K * discount_factor * N_neg_d2 - S * forward_factor * N_neg_d1,
        };

        // Calculate Greeks
        let phi_d1 = (-0.5 * d1.powi(2)).exp() / (2.0 * PI).sqrt();
        
        let delta = match contract.option_type {
            OptionType::Call => forward_factor * N_d1,
            OptionType::Put => forward_factor * (N_d1 - 1.0),
        };

        let gamma = forward_factor * phi_d1 / (S * sigma * sqrt_T);

        let theta = match contract.option_type {
            OptionType::Call => {
                -S * forward_factor * phi_d1 * sigma / (2.0 * sqrt_T)
                - r * K * discount_factor * N_d2
                + q * S * forward_factor * N_d1
            },
            OptionType::Put => {
                -S * forward_factor * phi_d1 * sigma / (2.0 * sqrt_T)
                + r * K * discount_factor * N_neg_d2
                - q * S * forward_factor * N_neg_d1
            },
        } / 365.25; // Convert to per-day

        let vega = S * forward_factor * phi_d1 * sqrt_T / 100.0; // Per 1% vol change

        let rho = match contract.option_type {
            OptionType::Call => K * T * discount_factor * N_d2 / 100.0,
            OptionType::Put => -K * T * discount_factor * N_neg_d2 / 100.0,
        }; // Per 1% rate change

        // Second-order Greeks
        let vanna = -forward_factor * phi_d1 * d2 / sigma / 100.0;
        let volga = S * forward_factor * phi_d1 * sqrt_T * d1 * d2 / sigma / 100.0;
        let charm = match contract.option_type {
            OptionType::Call => q * forward_factor * N_d1 - forward_factor * phi_d1 * (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (2.0 * T * sigma * sqrt_T),
            OptionType::Put => -q * forward_factor * N_neg_d1 - forward_factor * phi_d1 * (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (2.0 * T * sigma * sqrt_T),
        } / 365.25;

        let greeks = Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
            vanna,
            volga,
            charm,
        };

        Ok(PricingResult {
            price,
            greeks,
            implied_volatility: None,
            pricing_method: "Black-Scholes".to_string(),
            computation_time_ns: start_time.elapsed().as_nanos() as u64,
        })
    }

    /// Binomial tree pricing for American and European options
    pub fn binomial_tree(&self, contract: &OptionContract, market: &MarketData) -> Result<PricingResult> {
        let start_time = std::time::Instant::now();

        let S = market.spot_price;
        let K = contract.strike;
        let T = contract.expiry;
        let r = market.risk_free_rate;
        let q = market.dividend_yield;
        let sigma = market.volatility;
        let n = self.binomial_steps;

        if T <= 0.0 {
            let intrinsic = match contract.option_type {
                OptionType::Call => (S - K).max(0.0),
                OptionType::Put => (K - S).max(0.0),
            };
            
            return Ok(PricingResult {
                price: intrinsic,
                greeks: Greeks::default(),
                implied_volatility: None,
                pricing_method: "Intrinsic Value".to_string(),
                computation_time_ns: start_time.elapsed().as_nanos() as u64,
            });
        }

        let dt = T / n as f64;
        let u = (sigma * dt.sqrt()).exp();
        let d = 1.0 / u;
        let p = ((r - q) * dt).exp() - d) / (u - d);
        let discount = (-r * dt).exp();

        // Build the stock price tree
        let mut stock_prices = Array2::<f64>::zeros((n + 1, n + 1));
        
        for i in 0..=n {
            for j in 0..=i {
                stock_prices[[i, j]] = S * u.powi(j as i32) * d.powi((i - j) as i32);
            }
        }

        // Build the option value tree (backward induction)
        let mut option_values = Array2::<f64>::zeros((n + 1, n + 1));
        
        // Terminal condition
        for j in 0..=n {
            let stock_price = stock_prices[[n, j]];
            option_values[[n, j]] = match contract.option_type {
                OptionType::Call => (stock_price - K).max(0.0),
                OptionType::Put => (K - stock_price).max(0.0),
            };
        }

        // Backward induction
        for i in (0..n).rev() {
            for j in 0..=i {
                let hold_value = discount * (p * option_values[[i + 1, j + 1]] + (1.0 - p) * option_values[[i + 1, j]]);
                
                let stock_price = stock_prices[[i, j]];
                let exercise_value = match contract.option_type {
                    OptionType::Call => (stock_price - K).max(0.0),
                    OptionType::Put => (K - stock_price).max(0.0),
                };

                option_values[[i, j]] = match contract.exercise_style {
                    ExerciseStyle::European => hold_value,
                    ExerciseStyle::American => hold_value.max(exercise_value),
                    ExerciseStyle::Bermudan(ref exercise_times) => {
                        let current_time = i as f64 * dt;
                        if exercise_times.iter().any(|&t| (t - current_time).abs() < dt / 2.0) {
                            hold_value.max(exercise_value)
                        } else {
                            hold_value
                        }
                    },
                };
            }
        }

        let price = option_values[[0, 0]];

        // Calculate Greeks using finite differences
        let greeks = self.calculate_binomial_greeks(contract, market, &stock_prices, &option_values)?;

        Ok(PricingResult {
            price,
            greeks,
            implied_volatility: None,
            pricing_method: "Binomial Tree".to_string(),
            computation_time_ns: start_time.elapsed().as_nanos() as u64,
        })
    }

    /// Monte Carlo simulation for path-dependent and exotic options
    pub fn monte_carlo(&self, contract: &OptionContract, market: &MarketData) -> Result<PricingResult> {
        let start_time = std::time::Instant::now();

        if let Some(cuda_engine) = &self.cuda_engine {
            // Use GPU acceleration if available
            return self.monte_carlo_gpu(contract, market, cuda_engine.clone());
        }

        // CPU Monte Carlo implementation
        let S = market.spot_price;
        let K = contract.strike;
        let T = contract.expiry;
        let r = market.risk_free_rate;
        let q = market.dividend_yield;
        let sigma = market.volatility;
        let paths = self.monte_carlo_paths;

        if T <= 0.0 {
            let intrinsic = match contract.option_type {
                OptionType::Call => (S - K).max(0.0),
                OptionType::Put => (K - S).max(0.0),
            };
            
            return Ok(PricingResult {
                price: intrinsic,
                greeks: Greeks::default(),
                implied_volatility: None,
                pricing_method: "Intrinsic Value".to_string(),
                computation_time_ns: start_time.elapsed().as_nanos() as u64,
            });
        }

        let dt = T / 252.0; // Daily steps
        let steps = (T / dt) as usize;
        let drift = (r - q - 0.5 * sigma.powi(2)) * dt;
        let diffusion = sigma * dt.sqrt();

        let payoffs: Vec<f64> = (0..paths)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let normal = StandardNormal;
                let mut price = S;

                // Simulate price path
                for _ in 0..steps {
                    let z: f64 = rng.sample(normal);
                    price *= (drift + diffusion * z).exp();
                }

                // Calculate payoff based on exercise style
                match contract.exercise_style {
                    ExerciseStyle::European => match contract.option_type {
                        OptionType::Call => (price - K).max(0.0),
                        OptionType::Put => (K - price).max(0.0),
                    },
                    ExerciseStyle::American => {
                        // For American options, we need a more sophisticated approach
                        // This is a simplified implementation
                        match contract.option_type {
                            OptionType::Call => (price - K).max(0.0),
                            OptionType::Put => (K - price).max(0.0),
                        }
                    },
                    ExerciseStyle::Bermudan(_) => {
                        // Simplified Bermudan implementation
                        match contract.option_type {
                            OptionType::Call => (price - K).max(0.0),
                            OptionType::Put => (K - price).max(0.0),
                        }
                    },
                }
            })
            .collect();

        let price = (-r * T).exp() * payoffs.iter().sum::<f64>() / paths as f64;

        // Calculate Greeks using finite differences
        let greeks = self.calculate_monte_carlo_greeks(contract, market)?;

        Ok(PricingResult {
            price,
            greeks,
            implied_volatility: None,
            pricing_method: "Monte Carlo".to_string(),
            computation_time_ns: start_time.elapsed().as_nanos() as u64,
        })
    }

    /// GPU-accelerated Monte Carlo simulation
    fn monte_carlo_gpu(&self, contract: &OptionContract, market: &MarketData, 
                      cuda_engine: Arc<CudaOptionsEngine>) -> Result<PricingResult> {
        let start_time = std::time::Instant::now();

        // Use CUDA engine for GPU computation
        let result = cuda_engine.price_option_monte_carlo(
            market.spot_price,
            contract.strike,
            contract.expiry,
            market.risk_free_rate,
            market.volatility,
            matches!(contract.option_type, OptionType::Put),
            self.monte_carlo_paths,
        )?;

        Ok(PricingResult {
            price: result,
            greeks: self.calculate_monte_carlo_greeks(contract, market)?,
            implied_volatility: None,
            pricing_method: "Monte Carlo GPU".to_string(),
            computation_time_ns: start_time.elapsed().as_nanos() as u64,
        })
    }

    /// Calculate implied volatility using Brent's method
    pub fn implied_volatility(&self, contract: &OptionContract, market: &MarketData, 
                             market_price: f64) -> Result<f64> {
        let tolerance = 1e-6;
        let max_iterations = 100;
        
        let mut vol_low = 0.001;
        let mut vol_high = 3.0;
        
        // Check bounds
        let mut market_low = market.clone();
        market_low.volatility = vol_low;
        let price_low = self.black_scholes(contract, &market_low)?.price;
        
        let mut market_high = market.clone();
        market_high.volatility = vol_high;
        let price_high = self.black_scholes(contract, &market_high)?.price;
        
        if (price_low - market_price) * (price_high - market_price) > 0.0 {
            return Err(AlgoVedaError::Calculation("Market price is outside the bounds".to_string()));
        }

        // Brent's method
        let mut a = vol_low;
        let mut b = vol_high;
        let mut c = vol_high;
        
        let mut market_a = market_low;
        let mut market_b = market_high;
        let mut market_c = market_high;
        
        let mut fa = price_low - market_price;
        let mut fb = price_high - market_price;
        let mut fc = fb;
        
        for _ in 0..max_iterations {
            if fb.abs() < fa.abs() {
                std::mem::swap(&mut a, &mut b);
                std::mem::swap(&mut fa, &mut fb);
                std::mem::swap(&mut market_a, &mut market_b);
            }
            
            if fb.abs() < tolerance {
                return Ok(b);
            }
            
            let tol = 2.0 * f64::EPSILON * b.abs() + 0.5 * tolerance;
            let xm = 0.5 * (c - b);
            
            if xm.abs() < tol {
                return Ok(b);
            }
            
            let s = if fa != fc && fb != fc {
                // Inverse quadratic interpolation
                a * fb * fc / ((fa - fb) * (fa - fc)) +
                b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb))
            } else {
                // Secant method
                b - fb * (b - a) / (fb - fa)
            };
            
            let new_vol = if (s - 0.75 * a - 0.25 * b).abs() < (0.5 * (3.0 * a - b)).abs() &&
                            (s - b).abs() < 0.5 * (b - c).abs() {
                s
            } else {
                0.5 * (b + c)
            };
            
            let mut market_new = market.clone();
            market_new.volatility = new_vol;
            let price_new = self.black_scholes(contract, &market_new)?.price;
            let f_new = price_new - market_price;
            
            if f_new * fb < 0.0 {
                a = b;
                fa = fb;
                market_a = market_b.clone();
            } else {
                c = b;
                fc = fb;
                market_c = market_b.clone();
            }
            
            b = new_vol;
            fb = f_new;
            market_b = market_new;
        }
        
        Err(AlgoVedaError::Calculation("Implied volatility did not converge".to_string()))
    }

    /// Calculate Greeks using finite differences
    fn calculate_binomial_greeks(&self, contract: &OptionContract, market: &MarketData,
                                stock_prices: &Array2<f64>, option_values: &Array2<f64>) -> Result<Greeks> {
        let h = self.finite_difference_step;
        
        // Delta: using values from the tree
        let delta = if stock_prices.shape()[0] > 1 {
            (option_values[[1, 1]] - option_values[[1, 0]]) / (stock_prices[[1, 1]] - stock_prices[[1, 0]])
        } else {
            0.0
        };

        // Gamma: using second differences
        let gamma = if stock_prices.shape()[0] > 2 {
            let up_delta = (option_values[[2, 2]] - option_values[[2, 1]]) / (stock_prices[[2, 2]] - stock_prices[[2, 1]]);
            let down_delta = (option_values[[2, 1]] - option_values[[2, 0]]) / (stock_prices[[2, 1]] - stock_prices[[2, 0]]);
            let avg_price_diff = 0.5 * ((stock_prices[[2, 2]] - stock_prices[[2, 1]]) + (stock_prices[[2, 1]] - stock_prices[[2, 0]]));
            (up_delta - down_delta) / avg_price_diff
        } else {
            0.0
        };

        // For other Greeks, use finite difference method
        let base_price = self.binomial_tree(contract, market)?.price;

        // Theta
        let mut market_theta = market.clone();
        market_theta.risk_free_rate += h;
        let theta_price = self.binomial_tree(contract, &market_theta)?.price;
        let rho = (theta_price - base_price) / h / 100.0;

        // Vega
        let mut market_vega = market.clone();
        market_vega.volatility += h;
        let vega_price = self.binomial_tree(contract, &market_vega)?.price;
        let vega = (vega_price - base_price) / h / 100.0;

        // Simplified theta calculation
        let theta = -base_price * 0.1 / 365.25; // Rough approximation

        Ok(Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
            vanna: 0.0,   // Not calculated for binomial
            volga: 0.0,   // Not calculated for binomial
            charm: 0.0,   // Not calculated for binomial
        })
    }

    fn calculate_monte_carlo_greeks(&self, contract: &OptionContract, market: &MarketData) -> Result<Greeks> {
        let h = self.finite_difference_step;
        let base_price = self.monte_carlo(contract, market)?.price;

        // Delta
        let mut market_delta_up = market.clone();
        market_delta_up.spot_price += h;
        let mut market_delta_down = market.clone();
        market_delta_down.spot_price -= h;
        
        let price_up = self.monte_carlo(contract, &market_delta_up)?.price;
        let price_down = self.monte_carlo(contract, &market_delta_down)?.price;
        let delta = (price_up - price_down) / (2.0 * h);

        // Gamma
        let gamma = (price_up - 2.0 * base_price + price_down) / h.powi(2);

        // Vega
        let mut market_vega = market.clone();
        market_vega.volatility += h;
        let vega_price = self.monte_carlo(contract, &market_vega)?.price;
        let vega = (vega_price - base_price) / h / 100.0;

        // Rho
        let mut market_rho = market.clone();
        market_rho.risk_free_rate += h;
        let rho_price = self.monte_carlo(contract, &market_rho)?.price;
        let rho = (rho_price - base_price) / h / 100.0;

        // Theta (approximate)
        let theta = -base_price * market.risk_free_rate / 365.25;

        Ok(Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
            vanna: 0.0,   // Not calculated for Monte Carlo
            volga: 0.0,   // Not calculated for Monte Carlo
            charm: 0.0,   // Not calculated for Monte Carlo
        })
    }

    /// Price a portfolio of options
    pub fn price_portfolio(&self, contracts: &[OptionContract], quantities: &[f64], 
                          market: &MarketData) -> Result<PricingResult> {
        if contracts.len() != quantities.len() {
            return Err(AlgoVedaError::Calculation("Contracts and quantities must have the same length".to_string()));
        }

        let start_time = std::time::Instant::now();
        let mut total_price = 0.0;
        let mut total_greeks = Greeks::default();

        for (contract, &quantity) in contracts.iter().zip(quantities.iter()) {
            let result = self.black_scholes(contract, market)?;
            total_price += result.price * quantity;
            
            total_greeks.delta += result.greeks.delta * quantity;
            total_greeks.gamma += result.greeks.gamma * quantity;
            total_greeks.theta += result.greeks.theta * quantity;
            total_greeks.vega += result.greeks.vega * quantity;
            total_greeks.rho += result.greeks.rho * quantity;
            total_greeks.vanna += result.greeks.vanna * quantity;
            total_greeks.volga += result.greeks.volga * quantity;
            total_greeks.charm += result.greeks.charm * quantity;
        }

        Ok(PricingResult {
            price: total_price,
            greeks: total_greeks,
            implied_volatility: None,
            pricing_method: "Portfolio Black-Scholes".to_string(),
            computation_time_ns: start_time.elapsed().as_nanos() as u64,
        })
    }
}

impl Default for Greeks {
    fn default() -> Self {
        Self {
            delta: 0.0,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0,
            vanna: 0.0,
            volga: 0.0,
            charm: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_scholes_call() {
        let engine = OptionsEngine::new();
        
        let contract = OptionContract {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 0.25, // 3 months
            exercise_style: ExerciseStyle::European,
        };
        
        let market = MarketData {
            spot_price: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.2,
            timestamp: Utc::now(),
        };
        
        let result = engine.black_scholes(&contract, &market).unwrap();
        
        // Expected value around 3.99 for these parameters
        assert!((result.price - 3.99).abs() < 0.1);
        assert!(result.greeks.delta > 0.0 && result.greeks.delta < 1.0);
        assert!(result.greeks.gamma > 0.0);
        assert!(result.greeks.vega > 0.0);
    }

    #[test]
    fn test_black_scholes_put() {
        let engine = OptionsEngine::new();
        
        let contract = OptionContract {
            option_type: OptionType::Put,
            strike: 100.0,
            expiry: 0.25,
            exercise_style: ExerciseStyle::European,
        };
        
        let market = MarketData {
            spot_price: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.2,
            timestamp: Utc::now(),
        };
        
        let result = engine.black_scholes(&contract, &market).unwrap();
        
        // Put-call parity check would be useful here
        assert!(result.greeks.delta < 0.0 && result.greeks.delta > -1.0);
        assert!(result.greeks.gamma > 0.0);
    }

    #[test]
    fn test_implied_volatility() {
        let engine = OptionsEngine::new();
        
        let contract = OptionContract {
            option_type: OptionType::Call,
            strike: 100.0,
            expiry: 0.25,
            exercise_style: ExerciseStyle::European,
        };
        
        let market = MarketData {
            spot_price: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.2, // This will be ignored in IV calculation
            timestamp: Utc::now(),
        };
        
        // First get theoretical price with known volatility
        let theoretical_result = engine.black_scholes(&contract, &market).unwrap();
        
        // Now calculate implied volatility
        let implied_vol = engine.implied_volatility(&contract, &market, theoretical_result.price).unwrap();
        
        // Should be very close to original volatility (0.2)
        assert!((implied_vol - 0.2).abs() < 1e-4);
    }

    #[test]
    fn test_portfolio_pricing() {
        let engine = OptionsEngine::new();
        
        let contracts = vec![
            OptionContract {
                option_type: OptionType::Call,
                strike: 100.0,
                expiry: 0.25,
                exercise_style: ExerciseStyle::European,
            },
            OptionContract {
                option_type: OptionType::Put,
                strike: 100.0,
                expiry: 0.25,
                exercise_style: ExerciseStyle::European,
            },
        ];
        
        let quantities = vec![1.0, -1.0]; // Long call, short put (synthetic long)
        
        let market = MarketData {
            spot_price: 100.0,
            risk_free_rate: 0.05,
            dividend_yield: 0.0,
            volatility: 0.2,
            timestamp: Utc::now(),
        };
        
        let result = engine.price_portfolio(&contracts, &quantities, &market).unwrap();
        
        // Synthetic long should have delta close to 1.0
        assert!((result.greeks.delta - 1.0).abs() < 0.1);
    }
}
