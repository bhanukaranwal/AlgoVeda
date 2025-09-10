/**
 * GPU-accelerated Monte Carlo simulation kernels for AlgoVeda
 * Implements various Monte Carlo methods for options pricing,
 * risk calculations, and portfolio simulation with optimized memory access.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <device_launch_parameters.h>

// Device constants
__constant__ float d_pi = CUDART_PI_F;
__constant__ float d_sqrt2 = 1.4142135623730950488f;

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while(0)

/**
 * Box-Muller transformation for generating normal random numbers
 */
__device__ float2 box_muller(curandState* state) {
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * d_pi * u2;
    
    float2 result;
    result.x = r * cosf(theta);
    result.y = r * sinf(theta);
    
    return result;
}

/**
 * Geometric Brownian Motion path generator
 */
__device__ void generate_gbm_path(
    curandState* state,
    float* path,
    float spot_price,
    float drift,
    float volatility,
    float dt,
    int n_steps
) {
    path[0] = spot_price;
    
    for (int i = 1; i < n_steps; i++) {
        float2 normal = box_muller(state);
        float dW = sqrtf(dt) * normal.x;
        
        float log_return = (drift - 0.5f * volatility * volatility) * dt + volatility * dW;
        path[i] = path[i-1] * expf(log_return);
    }
}

/**
 * Heston stochastic volatility model path generator
 */
__device__ void generate_heston_path(
    curandState* state,
    float* spot_path,
    float* vol_path,
    float spot_price,
    float initial_vol,
    float risk_free_rate,
    float kappa,
    float theta,
    float sigma_v,
    float rho,
    float dt,
    int n_steps
) {
    spot_path[0] = spot_price;
    vol_path[0] = initial_vol;
    
    for (int i = 1; i < n_steps; i++) {
        float2 normal1 = box_muller(state);
        float2 normal2 = box_muller(state);
        
        // Correlated normal variables
        float dW1 = normal1.x;
        float dW2 = rho * normal1.x + sqrtf(1.0f - rho * rho) * normal1.y;
        
        // Volatility process (Feller condition applied)
        float vol_drift = kappa * (theta - fmaxf(vol_path[i-1], 0.0f)) * dt;
        float vol_diffusion = sigma_v * sqrtf(fmaxf(vol_path[i-1], 0.0f)) * sqrtf(dt) * dW2;
        vol_path[i] = fmaxf(vol_path[i-1] + vol_drift + vol_diffusion, 0.0001f);
        
        // Spot price process
        float spot_drift = risk_free_rate * dt;
        float spot_diffusion = sqrtf(fmaxf(vol_path[i-1], 0.0f)) * sqrtf(dt) * dW1;
        spot_path[i] = spot_path[i-1] * expf(spot_drift - 0.5f * vol_path[i-1] * dt + spot_diffusion);
    }
}

/**
 * European option Monte Carlo pricing kernel
 */
__global__ void monte_carlo_european_option(
    float* option_prices,
    float* spot_prices,
    float* strike_prices,
    float* times_to_expiry,
    float* risk_free_rates,
    float* volatilities,
    int* option_types,
    int n_options,
    int n_paths,
    int n_steps,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Initialize curand state
    curandState state;
    curand_init(seed + tid, 0, 0, &state);
    
    // Shared memory for path generation
    extern __shared__ float shared_mem[];
    float* path = &shared_mem[threadIdx.x * n_steps];
    
    for (int opt_idx = tid; opt_idx < n_options; opt_idx += stride) {
        float spot = spot_prices[opt_idx];
        float strike = strike_prices[opt_idx];
        float time_to_expiry = times_to_expiry[opt_idx];
        float risk_free_rate = risk_free_rates[opt_idx];
        float volatility = volatilities[opt_idx];
        int option_type = option_types[opt_idx];
        
        float dt = time_to_expiry / (n_steps - 1);
        float drift = risk_free_rate;
        
        float payoff_sum = 0.0f;
        
        // Monte Carlo simulation
        for (int path_idx = 0; path_idx < n_paths; path_idx++) {
            generate_gbm_path(&state, path, spot, drift, volatility, dt, n_steps);
            
            float final_price = path[n_steps - 1];
            float payoff;
            
            if (option_type == 0) { // Call option
                payoff = fmaxf(final_price - strike, 0.0f);
            } else { // Put option
                payoff = fmaxf(strike - final_price, 0.0f);
            }
            
            payoff_sum += payoff;
        }
        
        // Discount to present value
        float option_price = (payoff_sum / n_paths) * expf(-risk_free_rate * time_to_expiry);
        option_prices[opt_idx] = option_price;
    }
}

/**
 * Asian option Monte Carlo pricing kernel
 */
__global__ void monte_carlo_asian_option(
    float* option_prices,
    float* spot_prices,
    float* strike_prices,
    float* times_to_expiry,
    float* risk_free_rates,
    float* volatilities,
    int* option_types,
    int n_options,
    int n_paths,
    int n_steps,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    curandState state;
    curand_init(seed + tid, 0, 0, &state);
    
    extern __shared__ float shared_mem[];
    float* path = &shared_mem[threadIdx.x * n_steps];
    
    for (int opt_idx = tid; opt_idx < n_options; opt_idx += stride) {
        float spot = spot_prices[opt_idx];
        float strike = strike_prices[opt_idx];
        float time_to_expiry = times_to_expiry[opt_idx];
        float risk_free_rate = risk_free_rates[opt_idx];
        float volatility = volatilities[opt_idx];
        int option_type = option_types[opt_idx];
        
        float dt = time_to_expiry / (n_steps - 1);
        float drift = risk_free_rate;
        
        float payoff_sum = 0.0f;
        
        for (int path_idx = 0; path_idx < n_paths; path_idx++) {
            generate_gbm_path(&state, path, spot, drift, volatility, dt, n_steps);
            
            // Calculate arithmetic average
            float average_price = 0.0f;
            for (int i = 0; i < n_steps; i++) {
                average_price += path[i];
            }
            average_price /= n_steps;
            
            float payoff;
            if (option_type == 0) { // Call option
                payoff = fmaxf(average_price - strike, 0.0f);
            } else { // Put option
                payoff = fmaxf(strike - average_price, 0.0f);
            }
            
            payoff_sum += payoff;
        }
        
        float option_price = (payoff_sum / n_paths) * expf(-risk_free_rate * time_to_expiry);
        option_prices[opt_idx] = option_price;
    }
}

/**
 * Barrier option Monte Carlo pricing kernel
 */
__global__ void monte_carlo_barrier_option(
    float* option_prices,
    float* spot_prices,
    float* strike_prices,
    float* barrier_levels,
    float* times_to_expiry,
    float* risk_free_rates,
    float* volatilities,
    int* option_types,
    int* barrier_types,
    int n_options,
    int n_paths,
    int n_steps,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    curandState state;
    curand_init(seed + tid, 0, 0, &state);
    
    extern __shared__ float shared_mem[];
    float* path = &shared_mem[threadIdx.x * n_steps];
    
    for (int opt_idx = tid; opt_idx < n_options; opt_idx += stride) {
        float spot = spot_prices[opt_idx];
        float strike = strike_prices[opt_idx];
        float barrier = barrier_levels[opt_idx];
        float time_to_expiry = times_to_expiry[opt_idx];
        float risk_free_rate = risk_free_rates[opt_idx];
        float volatility = volatilities[opt_idx];
        int option_type = option_types[opt_idx];
        int barrier_type = barrier_types[opt_idx];
        
        float dt = time_to_expiry / (n_steps - 1);
        float drift = risk_free_rate;
        
        float payoff_sum = 0.0f;
        
        for (int path_idx = 0; path_idx < n_paths; path_idx++) {
            generate_gbm_path(&state, path, spot, drift, volatility, dt, n_steps);
            
            // Check barrier condition
            bool barrier_hit = false;
            for (int i = 0; i < n_steps; i++) {
                if ((barrier_type == 0 && path[i] >= barrier) || // Up-and-out
                    (barrier_type == 1 && path[i] <= barrier)) { // Down-and-out
                    barrier_hit = true;
                    break;
                }
            }
            
            float payoff = 0.0f;
            if ((barrier_type <= 1 && !barrier_hit) || // Knock-out not hit
                (barrier_type >= 2 && barrier_hit)) {   // Knock-in hit
                
                float final_price = path[n_steps - 1];
                if (option_type == 0) { // Call
                    payoff = fmaxf(final_price - strike, 0.0f);
                } else { // Put
                    payoff = fmaxf(strike - final_price, 0.0f);
                }
            }
            
            payoff_sum += payoff;
        }
        
        float option_price = (payoff_sum / n_paths) * expf(-risk_free_rate * time_to_expiry);
        option_prices[opt_idx] = option_price;
    }
}

/**
 * Portfolio VaR Monte Carlo calculation kernel
 */
__global__ void monte_carlo_portfolio_var(
    float* var_estimates,
    float* portfolio_values,
    float* asset_prices,
    float* portfolio_weights,
    float* correlation_matrix,
    float* volatilities,
    int n_assets,
    int n_portfolios,
    int n_scenarios,
    float confidence_level,
    float time_horizon,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    curandState state;
    curand_init(seed + tid, 0, 0, &state);
    
    extern __shared__ float shared_mem[];
    float* scenario_returns = &shared_mem[threadIdx.x * n_assets];
    float* correlated_returns = &scenario_returns[n_assets];
    
    for (int port_idx = tid; port_idx < n_portfolios; port_idx += stride) {
        float* weights = &portfolio_weights[port_idx * n_assets];
        float initial_value = portfolio_values[port_idx];
        
        // Array to store scenario P&Ls
        float* scenario_pnls = new float[n_scenarios];
        
        for (int scenario = 0; scenario < n_scenarios; scenario++) {
            // Generate uncorrelated normal returns
            for (int i = 0; i < n_assets; i++) {
                float2 normal = box_muller(&state);
                scenario_returns[i] = normal.x;
            }
            
            // Apply correlation using Cholesky decomposition
            for (int i = 0; i < n_assets; i++) {
                correlated_returns[i] = 0.0f;
                for (int j = 0; j <= i; j++) {
                    int idx = i * n_assets + j;
                    correlated_returns[i] += correlation_matrix[idx] * scenario_returns[j];
                }
                
                // Scale by volatility and time horizon
                correlated_returns[i] *= volatilities[i] * sqrtf(time_horizon);
            }
            
            // Calculate portfolio return
            float portfolio_return = 0.0f;
            for (int i = 0; i < n_assets; i++) {
                portfolio_return += weights[i] * correlated_returns[i];
            }
            
            // Calculate scenario P&L
            scenario_pnls[scenario] = initial_value * (expf(portfolio_return) - 1.0f);
        }
        
        // Sort scenario P&Ls to find VaR
        // Simple bubble sort for GPU (could be optimized with thrust::sort)
        for (int i = 0; i < n_scenarios - 1; i++) {
            for (int j = 0; j < n_scenarios - i - 1; j++) {
                if (scenario_pnls[j] > scenario_pnls[j + 1]) {
                    float temp = scenario_pnls[j];
                    scenario_pnls[j] = scenario_pnls[j + 1];
                    scenario_pnls[j + 1] = temp;
                }
            }
        }
        
        // Calculate VaR at confidence level
        int var_index = (int)((1.0f - confidence_level) * n_scenarios);
        var_estimates[port_idx] = -scenario_pnls[var_index]; // VaR is positive loss
        
        delete[] scenario_pnls;
    }
}

/**
 * Heston model Monte Carlo option pricing kernel
 */
__global__ void monte_carlo_heston_option(
    float* option_prices,
    float* spot_prices,
    float* strike_prices,
    float* times_to_expiry,
    float* risk_free_rates,
    float* initial_vols,
    float* kappas,
    float* thetas,
    float* sigma_vs,
    float* rhos,
    int* option_types,
    int n_options,
    int n_paths,
    int n_steps,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    curandState state;
    curand_init(seed + tid, 0, 0, &state);
    
    extern __shared__ float shared_mem[];
    float* spot_path = &shared_mem[threadIdx.x * n_steps * 2];
    float* vol_path = &spot_path[n_steps];
    
    for (int opt_idx = tid; opt_idx < n_options; opt_idx += stride) {
        float spot = spot_prices[opt_idx];
        float strike = strike_prices[opt_idx];
        float time_to_expiry = times_to_expiry[opt_idx];
        float risk_free_rate = risk_free_rates[opt_idx];
        float initial_vol = initial_vols[opt_idx];
        float kappa = kappas[opt_idx];
        float theta = thetas[opt_idx];
        float sigma_v = sigma_vs[opt_idx];
        float rho = rhos[opt_idx];
        int option_type = option_types[opt_idx];
        
        float dt = time_to_expiry / (n_steps - 1);
        float payoff_sum = 0.0f;
        
        for (int path_idx = 0; path_idx < n_paths; path_idx++) {
            generate_heston_path(&state, spot_path, vol_path, spot, initial_vol,
                               risk_free_rate, kappa, theta, sigma_v, rho, dt, n_steps);
            
            float final_price = spot_path[n_steps - 1];
            float payoff;
            
            if (option_type == 0) { // Call option
                payoff = fmaxf(final_price - strike, 0.0f);
            } else { // Put option
                payoff = fmaxf(strike - final_price, 0.0f);
            }
            
            payoff_sum += payoff;
        }
        
        float option_price = (payoff_sum / n_paths) * expf(-risk_free_rate * time_to_expiry);
        option_prices[opt_idx] = option_price;
    }
}

/**
 * Variance reduction using antithetic variates
 */
__global__ void monte_carlo_antithetic_option(
    float* option_prices,
    float* spot_prices,
    float* strike_prices,
    float* times_to_expiry,
    float* risk_free_rates,
    float* volatilities,
    int* option_types,
    int n_options,
    int n_paths,
    unsigned long long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    curandState state;
    curand_init(seed + tid, 0, 0, &state);
    
    for (int opt_idx = tid; opt_idx < n_options; opt_idx += stride) {
        float spot = spot_prices[opt_idx];
        float strike = strike_prices[opt_idx];
        float time_to_expiry = times_to_expiry[opt_idx];
        float risk_free_rate = risk_free_rates[opt_idx];
        float volatility = volatilities[opt_idx];
        int option_type = option_types[opt_idx];
        
        float drift = (risk_free_rate - 0.5f * volatility * volatility) * time_to_expiry;
        float diffusion = volatility * sqrtf(time_to_expiry);
        
        float payoff_sum = 0.0f;
        
        for (int path_idx = 0; path_idx < n_paths / 2; path_idx++) {
            float2 normal = box_muller(&state);
            
            // Regular path
            float z1 = normal.x;
            float s1 = spot * expf(drift + diffusion * z1);
            float payoff1;
            if (option_type == 0) {
                payoff1 = fmaxf(s1 - strike, 0.0f);
            } else {
                payoff1 = fmaxf(strike - s1, 0.0f);
            }
            
            // Antithetic path
            float z2 = -normal.x;
            float s2 = spot * expf(drift + diffusion * z2);
            float payoff2;
            if (option_type == 0) {
                payoff2 = fmaxf(s2 - strike, 0.0f);
            } else {
                payoff2 = fmaxf(strike - s2, 0.0f);
            }
            
            payoff_sum += (payoff1 + payoff2) / 2.0f;
        }
        
        float option_price = (payoff_sum / (n_paths / 2)) * expf(-risk_free_rate * time_to_expiry);
        option_prices[opt_idx] = option_price;
    }
}

// Host function to launch kernels (would be in a separate .cpp file)
extern "C" {
    
cudaError_t launch_european_option_kernel(
    float* d_option_prices,
    float* d_spot_prices,
    float* d_strike_prices,
    float* d_times_to_expiry,
    float* d_risk_free_rates,
    float* d_volatilities,
    int* d_option_types,
    int n_options,
    int n_paths,
    int n_steps,
    unsigned long long seed
) {
    int block_size = 256;
    int grid_size = (n_options + block_size - 1) / block_size;
    int shared_mem_size = block_size * n_steps * sizeof(float);
    
    monte_carlo_european_option<<<grid_size, block_size, shared_mem_size>>>(
        d_option_prices, d_spot_prices, d_strike_prices, d_times_to_expiry,
        d_risk_free_rates, d_volatilities, d_option_types,
        n_options, n_paths, n_steps, seed
    );
    
    return cudaGetLastError();
}

cudaError_t launch_portfolio_var_kernel(
    float* d_var_estimates,
    float* d_portfolio_values,
    float* d_asset_prices,
    float* d_portfolio_weights,
    float* d_correlation_matrix,
    float* d_volatilities,
    int n_assets,
    int n_portfolios,
    int n_scenarios,
    float confidence_level,
    float time_horizon,
    unsigned long long seed
) {
    int block_size = 256;
    int grid_size = (n_portfolios + block_size - 1) / block_size;
    int shared_mem_size = block_size * n_assets * 2 * sizeof(float);
    
    monte_carlo_portfolio_var<<<grid_size, block_size, shared_mem_size>>>(
        d_var_estimates, d_portfolio_values, d_asset_prices, d_portfolio_weights,
        d_correlation_matrix, d_volatilities, n_assets, n_portfolios,
        n_scenarios, confidence_level, time_horizon, seed
    );
    
    return cudaGetLastError();
}

} // extern "C"
