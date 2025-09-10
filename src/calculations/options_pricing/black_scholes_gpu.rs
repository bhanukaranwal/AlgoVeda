use cudarc::driver::{CudaDevice, ContextHandle, CudaStream};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsParameters {
    pub spot_prices: Vec<f64>,
    pub strike_prices: Vec<f64>,
    pub times_to_expiry: Vec<f64>,
    pub risk_free_rates: Vec<f64>,
    pub volatilities: Vec<f64>,
    pub dividend_yields: Vec<f64>,
    pub option_types: Vec<OptionType>, // 0 = Call, 1 = Put
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptionType {
    Call = 0,
    Put = 1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionsResults {
    pub option_prices: Vec<f64>,
    pub deltas: Vec<f64>,
    pub gammas: Vec<f64>,
    pub vegas: Vec<f64>,
    pub thetas: Vec<f64>,
    pub rhos: Vec<f64>,
}

pub struct BlackScholesGPU {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    kernel_name: String,
}

const BLACK_SCHOLES_KERNEL: &str = r#"
extern "C" __global__ void black_scholes_kernel(
    const double* spot_prices,
    const double* strike_prices,
    const double* times_to_expiry,
    const double* risk_free_rates,
    const double* volatilities,
    const double* dividend_yields,
    const int* option_types,
    double* option_prices,
    double* deltas,
    double* gammas,
    double* vegas,
    double* thetas,
    double* rhos,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double S = spot_prices[idx];
    double K = strike_prices[idx];
    double T = times_to_expiry[idx];
    double r = risk_free_rates[idx];
    double sigma = volatilities[idx];
    double q = dividend_yields[idx];
    int option_type = option_types[idx];

    // Calculate d1 and d2
    double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);

    // Standard normal CDF approximation
    auto norm_cdf = [](double x) -> double {
        return 0.5 * (1.0 + erf(x / sqrt(2.0)));
    };

    // Standard normal PDF
    auto norm_pdf = [](double x) -> double {
        return exp(-0.5 * x * x) / sqrt(2.0 * M_PI);
    };

    double Nd1 = norm_cdf(d1);
    double Nd2 = norm_cdf(d2);
    double nd1 = norm_pdf(d1);

    // Calculate option price
    double price;
    if (option_type == 0) { // Call
        price = S * exp(-q * T) * Nd1 - K * exp(-r * T) * Nd2;
    } else { // Put
        price = K * exp(-r * T) * (1.0 - Nd2) - S * exp(-q * T) * (1.0 - Nd1);
    }

    option_prices[idx] = price;

    // Calculate Greeks
    double discount_factor = exp(-q * T);
    double pv_strike = K * exp(-r * T);

    // Delta
    if (option_type == 0) { // Call
        deltas[idx] = discount_factor * Nd1;
    } else { // Put
        deltas[idx] = discount_factor * (Nd1 - 1.0);
    }

    // Gamma
    gammas[idx] = discount_factor * nd1 / (S * sigma * sqrt(T));

    // Vega
    vegas[idx] = S * discount_factor * nd1 * sqrt(T);

    // Theta
    double theta_term1 = -S * discount_factor * nd1 * sigma / (2.0 * sqrt(T));
    double theta_term2 = q * S * discount_factor;
    double theta_term3 = r * pv_strike;
    
    if (option_type == 0) { // Call
        thetas[idx] = theta_term1 - theta_term2 * Nd1 + theta_term3 * Nd2;
    } else { // Put
        thetas[idx] = theta_term1 + theta_term2 * (1.0 - Nd1) - theta_term3 * (1.0 - Nd2);
    }

    // Rho
    if (option_type == 0) { // Call
        rhos[idx] = pv_strike * T * Nd2;
    } else { // Put
        rhos[idx] = -pv_strike * T * (1.0 - Nd2);
    }
}
"#;

impl BlackScholesGPU {
    pub fn new() -> Result<Self> {
        let device = CudaDevice::new(0)?;
        let stream = device.fork_default_stream()?;
        
        Ok(Self {
            device,
            stream,
            kernel_name: "black_scholes_kernel".to_string(),
        })
    }

    pub async fn calculate_options(&self, params: &OptionsParameters) -> Result<OptionsResults> {
        let n = params.spot_prices.len();
        
        // Compile CUDA kernel
        let ptx = compile_ptx(BLACK_SCHOLES_KERNEL)?;
        self.device.load_ptx(ptx, "black_scholes", &[&self.kernel_name])?;

        // Allocate GPU memory
        let spot_gpu = self.device.htod_copy(params.spot_prices.clone())?;
        let strike_gpu = self.device.htod_copy(params.strike_prices.clone())?;
        let time_gpu = self.device.htod_copy(params.times_to_expiry.clone())?;
        let rate_gpu = self.device.htod_copy(params.risk_free_rates.clone())?;
        let vol_gpu = self.device.htod_copy(params.volatilities.clone())?;
        let div_gpu = self.device.htod_copy(params.dividend_yields.clone())?;
        
        let option_types_i32: Vec<i32> = params.option_types.iter()
            .map(|t| match t { OptionType::Call => 0, OptionType::Put => 1 })
            .collect();
        let types_gpu = self.device.htod_copy(option_types_i32)?;

        // Allocate output arrays
        let mut prices_gpu = self.device.alloc_zeros::<f64>(n)?;
        let mut deltas_gpu = self.device.alloc_zeros::<f64>(n)?;
        let mut gammas_gpu = self.device.alloc_zeros::<f64>(n)?;
        let mut vegas_gpu = self.device.alloc_zeros::<f64>(n)?;
        let mut thetas_gpu = self.device.alloc_zeros::<f64>(n)?;
        let mut rhos_gpu = self.device.alloc_zeros::<f64>(n)?;

        // Launch kernel
        let block_size = 256;
        let grid_size = (n + block_size - 1) / block_size;

        let f = self.device.get_func("black_scholes", &self.kernel_name)?;
        unsafe {
            f.launch_on_stream(
                &self.stream,
                (grid_size as u32, 1, 1),
                (block_size as u32, 1, 1),
                0,
                &[
                    &spot_gpu, &strike_gpu, &time_gpu, &rate_gpu, &vol_gpu, &div_gpu,
                    &types_gpu, &mut prices_gpu, &mut deltas_gpu, &mut gammas_gpu,
                    &mut vegas_gpu, &mut thetas_gpu, &mut rhos_gpu, &(n as i32),
                ],
            )?;
        }

        // Copy results back to host
        let option_prices = self.device.dtoh_sync_copy(&prices_gpu)?;
        let deltas = self.device.dtoh_sync_copy(&deltas_gpu)?;
        let gammas = self.device.dtoh_sync_copy(&gammas_gpu)?;
        let vegas = self.device.dtoh_sync_copy(&vegas_gpu)?;
        let thetas = self.device.dtoh_sync_copy(&thetas_gpu)?;
        let rhos = self.device.dtoh_sync_copy(&rhos_gpu)?;

        Ok(OptionsResults {
            option_prices,
            deltas,
            gammas,
            vegas,
            thetas,
            rhos,
        })
    }

    pub async fn calculate_implied_volatility(
        &self,
        market_prices: &[f64],
        params: &OptionsParameters,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<Vec<f64>> {
        let mut implied_vols = Vec::with_capacity(market_prices.len());
        
        for (i, &market_price) in market_prices.iter().enumerate() {
            let mut vol_low = 0.001;
            let mut vol_high = 5.0;
            let mut vol_mid = (vol_low + vol_high) / 2.0;
            
            for _ in 0..max_iterations {
                // Create temporary parameters with current volatility guess
                let mut temp_params = params.clone();
                temp_params.spot_prices = vec![params.spot_prices[i]];
                temp_params.strike_prices = vec![params.strike_prices[i]];
                temp_params.times_to_expiry = vec![params.times_to_expiry[i]];
                temp_params.risk_free_rates = vec![params.risk_free_rates[i]];
                temp_params.volatilities = vec![vol_mid];
                temp_params.dividend_yields = vec![params.dividend_yields[i]];
                temp_params.option_types = vec![params.option_types[i].clone()];
                
                let results = self.calculate_options(&temp_params).await?;
                let model_price = results.option_prices[0];
                
                let price_diff = model_price - market_price;
                
                if price_diff.abs() < tolerance {
                    break;
                }
                
                if price_diff > 0.0 {
                    vol_high = vol_mid;
                } else {
                    vol_low = vol_mid;
                }
                
                vol_mid = (vol_low + vol_high) / 2.0;
            }
            
            implied_vols.push(vol_mid);
        }
        
        Ok(implied_vols)
    }
}

// Benchmarking and testing utilities
pub mod benchmarks {
    use super::*;
    use std::time::Instant;

    pub async fn benchmark_black_scholes_gpu(num_options: usize) -> Result<()> {
        let gpu_engine = BlackScholesGPU::new()?;
        
        // Generate test data
        let params = OptionsParameters {
            spot_prices: vec![100.0; num_options],
            strike_prices: vec![105.0; num_options],
            times_to_expiry: vec![0.25; num_options],
            risk_free_rates: vec![0.05; num_options],
            volatilities: vec![0.20; num_options],
            dividend_yields: vec![0.0; num_options],
            option_types: vec![OptionType::Call; num_options],
        };

        // Warm-up run
        let _ = gpu_engine.calculate_options(&params).await?;

        // Benchmark run
        let start = Instant::now();
        let _results = gpu_engine.calculate_options(&params).await?;
        let duration = start.elapsed();

        let options_per_second = num_options as f64 / duration.as_secs_f64();
        println!("GPU Black-Scholes: {} options/sec", options_per_second as u64);
        println!("Total time: {:?} for {} options", duration, num_options);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[tokio::test]
    async fn test_black_scholes_call_option() {
        let gpu_engine = BlackScholesGPU::new().unwrap();
        
        let params = OptionsParameters {
            spot_prices: vec![100.0],
            strike_prices: vec![100.0],
            times_to_expiry: vec![1.0],
            risk_free_rates: vec![0.05],
            volatilities: vec![0.20],
            dividend_yields: vec![0.0],
            option_types: vec![OptionType::Call],
        };

        let results = gpu_engine.calculate_options(&params).await.unwrap();
        
        // Expected results for at-the-money call option
        // S=100, K=100, T=1, r=0.05, sigma=0.20, q=0
        assert_relative_eq!(results.option_prices[0], 10.4506, epsilon = 1e-3);
        assert_relative_eq!(results.deltas[0], 0.6368, epsilon = 1e-3);
        assert_relative_eq!(results.gammas[0], 0.0188, epsilon = 1e-3);
        assert_relative_eq!(results.vegas[0], 37.5240, epsilon = 1e-1);
    }

    #[tokio::test]
    async fn test_black_scholes_put_option() {
        let gpu_engine = BlackScholesGPU::new().unwrap();
        
        let params = OptionsParameters {
            spot_prices: vec![100.0],
            strike_prices: vec![100.0],
            times_to_expiry: vec![1.0],
            risk_free_rates: vec![0.05],
            volatilities: vec![0.20],
            dividend_yields: vec![0.0],
            option_types: vec![OptionType::Put],
        };

        let results = gpu_engine.calculate_options(&params).await.unwrap();
        
        // Expected results for at-the-money put option
        assert_relative_eq!(results.option_prices[0], 5.5735, epsilon = 1e-3);
        assert_relative_eq!(results.deltas[0], -0.3632, epsilon = 1e-3);
        assert_relative_eq!(results.gammas[0], 0.0188, epsilon = 1e-3);
    }

    #[tokio::test]
    async fn test_implied_volatility_calculation() {
        let gpu_engine = BlackScholesGPU::new().unwrap();
        
        let params = OptionsParameters {
            spot_prices: vec![100.0],
            strike_prices: vec![100.0],
            times_to_expiry: vec![1.0],
            risk_free_rates: vec![0.05],
            volatilities: vec![0.20], // This will be ignored in IV calculation
            dividend_yields: vec![0.0],
            option_types: vec![OptionType::Call],
        };

        // Use theoretical price as market price (should recover original volatility)
        let theoretical_results = gpu_engine.calculate_options(&params).await.unwrap();
        let market_prices = vec![theoretical_results.option_prices[0]];

        let implied_vols = gpu_engine.calculate_implied_volatility(
            &market_prices,
            &params,
            100,
            1e-6,
        ).await.unwrap();

        assert_relative_eq!(implied_vols[0], 0.20, epsilon = 1e-3);
    }
}
