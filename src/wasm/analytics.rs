/*!
 * WebAssembly Analytics Module
 * High-performance statistical computations for web frontend
 */

use wasm_bindgen::prelude::*;
use js_sys::{Array, Object};
use std::collections::HashMap;

#[wasm_bindgen]
pub struct AnalyticsModule {
    data_cache: HashMap<String, Vec<f64>>,
}

#[wasm_bindgen]
impl AnalyticsModule {
    #[wasm_bindgen(constructor)]
    pub fn new() -> AnalyticsModule {
        AnalyticsModule {
            data_cache: HashMap::new(),
        }
    }

    #[wasm_bindgen]
    pub fn calculate_mean(&self, data: &[f64]) -> f64 {
        if data.is_empty() { return 0.0; }
        data.iter().sum::<f64>() / data.len() as f64
    }

    #[wasm_bindgen]
    pub fn calculate_variance(&self, data: &[f64], mean: f64) -> f64 {
        if data.is_empty() { return 0.0; }
        data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64
    }

    #[wasm_bindgen]
    pub fn calculate_standard_deviation(&self, data: &[f64]) -> f64 {
        let mean = self.calculate_mean(data);
        self.calculate_variance(data, mean).sqrt()
    }

    #[wasm_bindgen]
    pub fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() { return 0.0; }

        let mean_x = self.calculate_mean(x);
        let mean_y = self.calculate_mean(y);
        
        let covariance: f64 = x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - mean_x) * (b - mean_y))
            .sum();
        
        let std_x = self.calculate_standard_deviation(x);
        let std_y = self.calculate_standard_deviation(y);
        
        if std_x == 0.0 || std_y == 0.0 { return 0.0; }
        
        covariance / (x.len() as f64 * std_x * std_y)
    }

    #[wasm_bindgen]
    pub fn calculate_var(&self, data: &[f64], confidence: f64) -> f64 {
        if data.is_empty() { return 0.0; }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence) * sorted_data.len() as f64) as usize;
        let index = index.min(sorted_data.len() - 1);
        
        -sorted_data[index]
    }

    #[wasm_bindgen]
    pub fn calculate_cvar(&self, data: &[f64], confidence: f64) -> f64 {
        if data.is_empty() { return 0.0; }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let cutoff_index = ((1.0 - confidence) * sorted_data.len() as f64) as usize;
        let cutoff_index = cutoff_index.min(sorted_data.len() - 1);
        
        if cutoff_index == 0 { return -sorted_data[0]; }
        
        let tail_sum: f64 = sorted_data[0..=cutoff_index].iter().sum();
        -tail_sum / (cutoff_index + 1) as f64
    }

    #[wasm_bindgen]
    pub fn calculate_sharpe_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64 {
        if returns.is_empty() { return 0.0; }
        
        let mean_return = self.calculate_mean(returns);
        let std_dev = self.calculate_standard_deviation(returns);
        
        if std_dev == 0.0 { return 0.0; }
        
        (mean_return - risk_free_rate) / std_dev
    }

    #[wasm_bindgen]
    pub fn calculate_max_drawdown(&self, portfolio_values: &[f64]) -> f64 {
        if portfolio_values.is_empty() { return 0.0; }
        
        let mut max_value = portfolio_values[0];
        let mut max_drawdown = 0.0;
        
        for &value in portfolio_values {
            if value > max_value {
                max_value = value;
            }
            
            let drawdown = (max_value - value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        max_drawdown
    }

    #[wasm_bindgen]
    pub fn calculate_beta(&self, asset_returns: &[f64], market_returns: &[f64]) -> f64 {
        if asset_returns.len() != market_returns.len() || asset_returns.is_empty() {
            return 1.0; // Default beta
        }
        
        let correlation = self.calculate_correlation(asset_returns, market_returns);
        let asset_std = self.calculate_standard_deviation(asset_returns);
        let market_std = self.calculate_standard_deviation(market_returns);
        
        if market_std == 0.0 { return 1.0; }
        
        correlation * (asset_std / market_std)
    }

    #[wasm_bindgen]
    pub fn rolling_calculation(&self, data: &[f64], window_size: usize, calc_type: &str) -> Vec<f64> {
        if data.len() < window_size { return vec![]; }
        
        let mut results = Vec::new();
        
        for i in window_size..=data.len() {
            let window = &data[i - window_size..i];
            
            let result = match calc_type {
                "mean" => self.calculate_mean(window),
                "std" => self.calculate_standard_deviation(window),
                "var_95" => self.calculate_var(window, 0.95),
                _ => 0.0,
            };
            
            results.push(result);
        }
        
        results
    }

    #[wasm_bindgen]
    pub fn calculate_portfolio_metrics(&self, weights: &[f64], returns_matrix: &[f64], num_assets: usize) -> JsValue {
        if weights.len() != num_assets || returns_matrix.is_empty() {
            return JsValue::NULL;
        }
        
        let num_periods = returns_matrix.len() / num_assets;
        let mut portfolio_returns = Vec::new();
        
        // Calculate portfolio returns
        for period in 0..num_periods {
            let mut portfolio_return = 0.0;
            for asset in 0..num_assets {
                let return_index = period * num_assets + asset;
                if return_index < returns_matrix.len() {
                    portfolio_return += weights[asset] * returns_matrix[return_index];
                }
            }
            portfolio_returns.push(portfolio_return);
        }
        
        // Calculate portfolio metrics
        let mean_return = self.calculate_mean(&portfolio_returns);
        let volatility = self.calculate_standard_deviation(&portfolio_returns);
        let sharpe = self.calculate_sharpe_ratio(&portfolio_returns, 0.0);
        let var_95 = self.calculate_var(&portfolio_returns, 0.95);
        let cvar_95 = self.calculate_cvar(&portfolio_returns, 0.95);
        
        // Create result object
        let result = Object::new();
        js_sys::Reflect::set(&result, &"mean_return".into(), &mean_return.into()).unwrap();
        js_sys::Reflect::set(&result, &"volatility".into(), &volatility.into()).unwrap();
        js_sys::Reflect::set(&result, &"sharpe_ratio".into(), &sharpe.into()).unwrap();
        js_sys::Reflect::set(&result, &"var_95".into(), &var_95.into()).unwrap();
        js_sys::Reflect::set(&result, &"cvar_95".into(), &cvar_95.into()).unwrap();
        
        result.into()
    }
}

// Utility functions
#[wasm_bindgen]
pub fn monte_carlo_simulation(
    initial_value: f64,
    drift: f64,
    volatility: f64,
    time_horizon: f64,
    num_simulations: usize,
    num_steps: usize
) -> Vec<f64> {
    let dt = time_horizon / num_steps as f64;
    let mut final_values = Vec::new();
    
    for _ in 0..num_simulations {
        let mut value = initial_value;
        
        for _ in 0..num_steps {
            let random_shock = js_sys::Math::random() - 0.5; // Simplified random number
            let dw = random_shock * dt.sqrt();
            value = value * (1.0 + drift * dt + volatility * dw);
        }
        
        final_values.push(value);
    }
    
    final_values
}

#[wasm_bindgen]
pub fn optimization_gradient_descent(
    returns: &[f64],
    covariance_matrix: &[f64],
    num_assets: usize,
    learning_rate: f64,
    max_iterations: usize
) -> Vec<f64> {
    let mut weights = vec![1.0 / num_assets as f64; num_assets];
    
    for _ in 0..max_iterations {
        let mut gradient = vec![0.0; num_assets];
        
        // Calculate gradient (simplified)
        for i in 0..num_assets {
            let mut grad = -returns[i]; // Return component
            
            // Risk component
            for j in 0..num_assets {
                let cov_index = i * num_assets + j;
                if cov_index < covariance_matrix.len() {
                    grad += covariance_matrix[cov_index] * weights[j];
                }
            }
            
            gradient[i] = grad;
        }
        
        // Update weights
        for i in 0..num_assets {
            weights[i] -= learning_rate * gradient[i];
        }
        
        // Normalize weights
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for weight in &mut weights {
                *weight /= sum;
            }
        }
    }
    
    weights
}
