/*
 * CUDA Options Pricing Kernels
 * GPU-accelerated options pricing using Black-Scholes and Monte Carlo
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

// Constants
#define PI 3.14159265358979323846
#define THREADS_PER_BLOCK 256
#define MAX_PATHS 1048576  // 1M paths

// Black-Scholes GPU kernel
__device__ double black_scholes_call_gpu(
    double S,      // Stock price
    double K,      // Strike price
    double T,      // Time to maturity
    double r,      // Risk-free rate
    double sigma   // Volatility
) {
    if (T <= 0.0) return fmax(S - K, 0.0);
    
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    
    // Cumulative normal distribution approximation
    auto norm_cdf = [](double x) -> double {
        return 0.5 * (1.0 + erf(x / sqrt(2.0)));
    };
    
    return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
}

__device__ double black_scholes_put_gpu(
    double S, double K, double T, double r, double sigma
) {
    if (T <= 0.0) return fmax(K - S, 0.0);
    
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    
    auto norm_cdf = [](double x) -> double {
        return 0.5 * (1.0 + erf(x / sqrt(2.0)));
    };
    
    return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
}

// Greeks calculations
__device__ double calculate_delta_call(
    double S, double K, double T, double r, double sigma
) {
    if (T <= 0.0) return (S > K) ? 1.0 : 0.0;
    
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    return 0.5 * (1.0 + erf(d1 / sqrt(2.0)));
}

__device__ double calculate_gamma(
    double S, double K, double T, double r, double sigma
) {
    if (T <= 0.0) return 0.0;
    
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double phi_d1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * PI);
    
    return phi_d1 / (S * sigma * sqrt(T));
}

__device__ double calculate_theta_call(
    double S, double K, double T, double r, double sigma
) {
    if (T <= 0.0) return 0.0;
    
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    
    auto norm_cdf = [](double x) -> double {
        return 0.5 * (1.0 + erf(x / sqrt(2.0)));
    };
    
    double phi_d1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * PI);
    
    return -(S * phi_d1 * sigma) / (2.0 * sqrt(T)) - 
           r * K * exp(-r * T) * norm_cdf(d2);
}

__device__ double calculate_vega(
    double S, double K, double T, double r, double sigma
) {
    if (T <= 0.0) return 0.0;
    
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double phi_d1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * PI);
    
    return S * phi_d1 * sqrt(T);
}

// Vectorized Black-Scholes kernel
__global__ void black_scholes_kernel(
    double* stock_prices,
    double* strike_prices,
    double* times_to_maturity,
    double* risk_free_rates,
    double* volatilities,
    double* call_prices,
    double* put_prices,
    double* deltas,
    double* gammas,
    double* thetas,
    double* vegas,
    int num_options
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_options) {
        double S = stock_prices[idx];
        double K = strike_prices[idx];
        double T = times_to_maturity[idx];
        double r = risk_free_rates[idx];
        double sigma = volatilities[idx];
        
        // Calculate option prices
        call_prices[idx] = black_scholes_call_gpu(S, K, T, r, sigma);
        put_prices[idx] = black_scholes_put_gpu(S, K, T, r, sigma);
        
        // Calculate Greeks
        deltas[idx] = calculate_delta_call(S, K, T, r, sigma);
        gammas[idx] = calculate_gamma(S, K, T, r, sigma);
        thetas[idx] = calculate_theta_call(S, K, T, r, sigma);
        vegas[idx] = calculate_vega(S, K, T, r, sigma);
    }
}

// Monte Carlo option pricing kernel
__global__ void monte_carlo_option_kernel(
    double* stock_prices,
    double* strike_prices,
    double* times_to_maturity,
    double* risk_free_rates,
    double* volatilities,
    double* option_prices,
    int* option_types,  // 0 = call, 1 = put
    int num_options,
    int num_paths,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_options) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        
        double S = stock_prices[idx];
        double K = strike_prices[idx];
        double T = times_to_maturity[idx];
        double r = risk_free_rates[idx];
        double sigma = volatilities[idx];
        int is_put = option_types[idx];
        
        double payoff_sum = 0.0;
        double dt = T / 252.0; // Daily steps
        
        for (int path = 0; path < num_paths; path++) {
            double current_price = S;
            
            // Simulate price path
            for (int step = 0; step < 252; step++) {
                double z = curand_normal_double(&state);
                current_price *= exp((r - 0.5 * sigma * sigma) * dt + 
                                   sigma * sqrt(dt) * z);
            }
            
            // Calculate payoff
            double payoff;
            if (is_put) {
                payoff = fmax(K - current_price, 0.0);
            } else {
                payoff = fmax(current_price - K, 0.0);
            }
            
            payoff_sum += payoff;
        }
        
        // Discount to present value
        option_prices[idx] = exp(-r * T) * (payoff_sum / num_paths);
    }
}

// Asian option pricing kernel
__global__ void asian_option_kernel(
    double* stock_prices,
    double* strike_prices,
    double* times_to_maturity,
    double* risk_free_rates,
    double* volatilities,
    double* option_prices,
    int num_options,
    int num_paths,
    int num_steps,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_options) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        
        double S = stock_prices[idx];
        double K = strike_prices[idx];
        double T = times_to_maturity[idx];
        double r = risk_free_rates[idx];
        double sigma = volatilities[idx];
        
        double payoff_sum = 0.0;
        double dt = T / num_steps;
        
        for (int path = 0; path < num_paths; path++) {
            double current_price = S;
            double price_sum = 0.0;
            
            // Simulate price path and accumulate for average
            for (int step = 0; step < num_steps; step++) {
                double z = curand_normal_double(&state);
                current_price *= exp((r - 0.5 * sigma * sigma) * dt + 
                                   sigma * sqrt(dt) * z);
                price_sum += current_price;
            }
            
            double average_price = price_sum / num_steps;
            double payoff = fmax(average_price - K, 0.0);
            payoff_sum += payoff;
        }
        
        option_prices[idx] = exp(-r * T) * (payoff_sum / num_paths);
    }
}

// Barrier option pricing kernel
__global__ void barrier_option_kernel(
    double* stock_prices,
    double* strike_prices,
    double* barrier_levels,
    double* times_to_maturity,
    double* risk_free_rates,
    double* volatilities,
    double* option_prices,
    int* barrier_types,  // 0 = up-and-out, 1 = down-and-out, 2 = up-and-in, 3 = down-and-in
    int num_options,
    int num_paths,
    int num_steps,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_options) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        
        double S = stock_prices[idx];
        double K = strike_prices[idx];
        double B = barrier_levels[idx];
        double T = times_to_maturity[idx];
        double r = risk_free_rates[idx];
        double sigma = volatilities[idx];
        int barrier_type = barrier_types[idx];
        
        double payoff_sum = 0.0;
        double dt = T / num_steps;
        
        for (int path = 0; path < num_paths; path++) {
            double current_price = S;
            bool barrier_hit = false;
            
            // Simulate price path and check barrier
            for (int step = 0; step < num_steps; step++) {
                double z = curand_normal_double(&state);
                current_price *= exp((r - 0.5 * sigma * sigma) * dt + 
                                   sigma * sqrt(dt) * z);
                
                // Check barrier conditions
                if ((barrier_type == 0 && current_price >= B) ||  // up-and-out
                    (barrier_type == 1 && current_price <= B)) {  // down-and-out
                    barrier_hit = true;
                    break;
                } else if ((barrier_type == 2 && current_price >= B) ||  // up-and-in
                          (barrier_type == 3 && current_price <= B)) {   // down-and-in
                    barrier_hit = true;
                }
            }
            
            // Calculate payoff based on barrier type
            double payoff = 0.0;
            if ((barrier_type <= 1 && !barrier_hit) ||  // knock-out options
                (barrier_type >= 2 && barrier_hit)) {   // knock-in options
                payoff = fmax(current_price - K, 0.0);
            }
            
            payoff_sum += payoff;
        }
        
        option_prices[idx] = exp(-r * T) * (payoff_sum / num_paths);
    }
}

// Implied volatility calculation using Newton-Raphson
__global__ void implied_volatility_kernel(
    double* stock_prices,
    double* strike_prices,
    double* times_to_maturity,
    double* risk_free_rates,
    double* market_prices,
    double* implied_vols,
    int* option_types,
    int num_options,
    double tolerance,
    int max_iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_options) {
        double S = stock_prices[idx];
        double K = strike_prices[idx];
        double T = times_to_maturity[idx];
        double r = risk_free_rates[idx];
        double market_price = market_prices[idx];
        int is_put = option_types[idx];
        
        double sigma = 0.2; // Initial guess
        double price_diff, vega_val;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            // Calculate theoretical price and vega
            double theoretical_price;
            if (is_put) {
                theoretical_price = black_scholes_put_gpu(S, K, T, r, sigma);
            } else {
                theoretical_price = black_scholes_call_gpu(S, K, T, r, sigma);
            }
            
            price_diff = theoretical_price - market_price;
            
            if (fabs(price_diff) < tolerance) {
                break;
            }
            
            vega_val = calculate_vega(S, K, T, r, sigma);
            
            if (fabs(vega_val) < 1e-8) {
                break;
            }
            
            // Newton-Raphson update
            sigma = sigma - price_diff / vega_val;
            
            // Ensure volatility stays positive
            sigma = fmax(sigma, 0.001);
            sigma = fmin(sigma, 5.0);
        }
        
        implied_vols[idx] = sigma;
    }
}

// Portfolio Greeks calculation kernel
__global__ void portfolio_greeks_kernel(
    double* stock_prices,
    double* strike_prices,
    double* times_to_maturity,
    double* risk_free_rates,
    double* volatilities,
    double* position_sizes,
    double* portfolio_delta,
    double* portfolio_gamma,
    double* portfolio_theta,
    double* portfolio_vega,
    int num_positions
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory for reduction
    __shared__ double shared_delta[THREADS_PER_BLOCK];
    __shared__ double shared_gamma[THREADS_PER_BLOCK];
    __shared__ double shared_theta[THREADS_PER_BLOCK];
    __shared__ double shared_vega[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    
    // Initialize shared memory
    shared_delta[tid] = 0.0;
    shared_gamma[tid] = 0.0;
    shared_theta[tid] = 0.0;
    shared_vega[tid] = 0.0;
    
    // Calculate Greeks for this position
    if (idx < num_positions) {
        double S = stock_prices[idx];
        double K = strike_prices[idx];
        double T = times_to_maturity[idx];
        double r = risk_free_rates[idx];
        double sigma = volatilities[idx];
        double position = position_sizes[idx];
        
        shared_delta[tid] = position * calculate_delta_call(S, K, T, r, sigma);
        shared_gamma[tid] = position * calculate_gamma(S, K, T, r, sigma);
        shared_theta[tid] = position * calculate_theta_call(S, K, T, r, sigma);
        shared_vega[tid] = position * calculate_vega(S, K, T, r, sigma);
    }
    
    __syncthreads();
    
    // Reduction to sum up portfolio Greeks
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_delta[tid] += shared_delta[tid + stride];
            shared_gamma[tid] += shared_gamma[tid + stride];
            shared_theta[tid] += shared_theta[tid + stride];
            shared_vega[tid] += shared_vega[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(portfolio_delta, shared_delta[0]);
        atomicAdd(portfolio_gamma, shared_gamma[0]);
        atomicAdd(portfolio_theta, shared_theta[0]);
        atomicAdd(portfolio_vega, shared_vega[0]);
    }
}

// Host functions for kernel launches
extern "C" {
    void launch_black_scholes_kernel(
        double* d_stock_prices,
        double* d_strike_prices,
        double* d_times_to_maturity,
        double* d_risk_free_rates,
        double* d_volatilities,
        double* d_call_prices,
        double* d_put_prices,
        double* d_deltas,
        double* d_gammas,
        double* d_thetas,
        double* d_vegas,
        int num_options
    ) {
        int blocks = (num_options + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        black_scholes_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_stock_prices, d_strike_prices, d_times_to_maturity,
            d_risk_free_rates, d_volatilities,
            d_call_prices, d_put_prices,
            d_deltas, d_gammas, d_thetas, d_vegas,
            num_options
        );
        
        cudaDeviceSynchronize();
    }
    
    void launch_monte_carlo_option_kernel(
        double* d_stock_prices,
        double* d_strike_prices,
        double* d_times_to_maturity,
        double* d_risk_free_rates,
        double* d_volatilities,
        double* d_option_prices,
        int* d_option_types,
        int num_options,
        int num_paths,
        unsigned long long seed
    ) {
        int blocks = (num_options + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        monte_carlo_option_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_stock_prices, d_strike_prices, d_times_to_maturity,
            d_risk_free_rates, d_volatilities, d_option_prices,
            d_option_types, num_options, num_paths, seed
        );
        
        cudaDeviceSynchronize();
    }
    
    void launch_implied_volatility_kernel(
        double* d_stock_prices,
        double* d_strike_prices,
        double* d_times_to_maturity,
        double* d_risk_free_rates,
        double* d_market_prices,
        double* d_implied_vols,
        int* d_option_types,
        int num_options,
        double tolerance,
        int max_iterations
    ) {
        int blocks = (num_options + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        implied_volatility_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_stock_prices, d_strike_prices, d_times_to_maturity,
            d_risk_free_rates, d_market_prices, d_implied_vols,
            d_option_types, num_options, tolerance, max_iterations
        );
        
        cudaDeviceSynchronize();
    }
}
