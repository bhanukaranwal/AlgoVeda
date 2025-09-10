/*!
 * CUDA Kernels for GPU-Accelerated Backtesting
 * Ultra-high performance parallel backtesting computation
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

#define THREADS_PER_BLOCK 256
#define MAX_ASSETS 1000

// Kernel for parallel portfolio backtesting
extern "C" __global__ void parallel_backtest_kernel(
    const double* prices,           // [n_periods x n_assets]
    const double* signals,          // [n_periods x n_assets]  
    double* portfolio_values,       // [n_periods x n_strategies]
    double* positions,             // [n_periods x n_assets x n_strategies]
    const double* initial_capitals, // [n_strategies]
    double commission_rate,
    int n_periods,
    int n_assets,
    int n_strategies
) {
    int strategy_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (strategy_idx >= n_strategies) return;
    
    // Shared memory for current portfolio state
    __shared__ double shared_prices[MAX_ASSETS];
    __shared__ double shared_signals[MAX_ASSETS];
    
    double cash = initial_capitals[strategy_idx];
    double portfolio_value = cash;
    
    // Initialize portfolio
    portfolio_values[strategy_idx] = portfolio_value;
    
    // Main backtesting loop
    for (int t = 1; t < n_periods; t++) {
        // Load current period data to shared memory
        if (threadIdx.x < n_assets) {
            shared_prices[threadIdx.x] = prices[t * n_assets + threadIdx.x];
            shared_signals[threadIdx.x] = signals[t * n_assets + threadIdx.x];
        }
        __syncthreads();
        
        // Update portfolio value from existing positions
        portfolio_value = cash;
        for (int a = 0; a < n_assets; a++) {
            double position = positions[(t-1) * n_assets * n_strategies + 
                                     a * n_strategies + strategy_idx];
            portfolio_value += position * shared_prices[a];
        }
        
        // Process each asset
        for (int a = 0; a < n_assets; a++) {
            double current_position = positions[(t-1) * n_assets * n_strategies + 
                                              a * n_strategies + strategy_idx];
            double signal = shared_signals[a];
            double price = shared_prices[a];
            
            // Calculate target position based on signal
            double target_value = portfolio_value * signal;
            double target_position = target_value / price;
            
            // Calculate trade
            double trade_size = target_position - current_position;
            
            if (fabs(trade_size) > 1e-8) {
                double trade_cost = fabs(trade_size) * price;
                double commission = trade_cost * commission_rate;
                
                // Update cash
                cash -= trade_size * price + commission;
                
                // Update position
                positions[t * n_assets * n_strategies + 
                         a * n_strategies + strategy_idx] = target_position;
            } else {
                // No trade, carry forward position
                positions[t * n_assets * n_strategies + 
                         a * n_strategies + strategy_idx] = current_position;
            }
        }
        
        // Recalculate final portfolio value
        portfolio_value = cash;
        for (int a = 0; a < n_assets; a++) {
            double position = positions[t * n_assets * n_strategies + 
                                     a * n_strategies + strategy_idx];
            portfolio_value += position * shared_prices[a];
        }
        
        portfolio_values[t * n_strategies + strategy_idx] = portfolio_value;
    }
}

// Kernel for Monte Carlo option pricing
extern "C" __global__ void monte_carlo_option_pricing_kernel(
    double* option_prices,
    const double* strikes,
    const double* spot_prices,
    const double* volatilities,
    const double* time_to_expiry,
    double risk_free_rate,
    int n_options,
    int n_simulations,
    unsigned long long seed
) {
    int option_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (option_idx >= n_options) return;
    
    // Initialize random number generator
    curandState state;
    curand_init(seed + option_idx, 0, 0, &state);
    
    double S = spot_prices[option_idx];
    double K = strikes[option_idx];
    double T = time_to_expiry[option_idx];
    double sigma = volatilities[option_idx];
    double r = risk_free_rate;
    
    double payoff_sum = 0.0;
    
    // Monte Carlo simulation
    for (int i = 0; i < n_simulations; i++) {
        // Generate random normal variable
        double z = curand_normal_double(&state);
        
        // Calculate final stock price using geometric Brownian motion
        double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * z);
        
        // Calculate payoff (call option)
        double payoff = fmax(ST - K, 0.0);
        payoff_sum += payoff;
    }
    
    // Calculate discounted expected payoff
    double option_price = exp(-r * T) * payoff_sum / n_simulations;
    option_prices[option_idx] = option_price;
}

// Kernel for portfolio risk calculations (VaR)
extern "C" __global__ void portfolio_var_kernel(
    const double* returns,          // [n_assets x n_periods]
    const double* weights,          // [n_assets]
    double* portfolio_returns,      // [n_periods]
    double* var_estimates,          // [1]
    int n_assets,
    int n_periods,
    double confidence_level
) {
    int period_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (period_idx >= n_periods) return;
    
    // Calculate portfolio return for this period
    double portfolio_return = 0.0;
    for (int a = 0; a < n_assets; a++) {
        portfolio_return += weights[a] * returns[a * n_periods + period_idx];
    }
    
    portfolio_returns[period_idx] = portfolio_return;
    
    // Use cooperative groups to calculate VaR
    __syncthreads();
    
    // Only first thread calculates VaR
    if (period_idx == 0) {
        // Simple bubble sort for small arrays (for demo)
        // In practice, use thrust::sort or other GPU sorting
        for (int i = 0; i < n_periods - 1; i++) {
            for (int j = 0; j < n_periods - i - 1; j++) {
                if (portfolio_returns[j] > portfolio_returns[j + 1]) {
                    double temp = portfolio_returns[j];
                    portfolio_returns[j] = portfolio_returns[j + 1];
                    portfolio_returns[j + 1] = temp;
                }
            }
        }
        
        // Calculate VaR percentile
        int var_index = (int)((1.0 - confidence_level) * n_periods);
        var_estimates[0] = -portfolio_returns[var_index];
    }
}

// Kernel for technical indicator calculations
extern "C" __global__ void technical_indicators_kernel(
    const double* prices,           // [n_periods]
    double* sma,                   // [n_periods] Simple Moving Average
    double* ema,                   // [n_periods] Exponential Moving Average  
    double* rsi,                   // [n_periods] Relative Strength Index
    double* bollinger_upper,       // [n_periods]
    double* bollinger_lower,       // [n_periods]
    int n_periods,
    int sma_period,
    double ema_alpha,
    int rsi_period
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_periods) return;
    
    // Simple Moving Average
    if (idx >= sma_period - 1) {
        double sum = 0.0;
        for (int i = 0; i < sma_period; i++) {
            sum += prices[idx - i];
        }
        sma[idx] = sum / sma_period;
    }
    
    // Exponential Moving Average
    if (idx == 0) {
        ema[0] = prices[0];
    } else {
        ema[idx] = ema_alpha * prices[idx] + (1.0 - ema_alpha) * ema[idx - 1];
    }
    
    // RSI calculation (simplified)
    if (idx >= rsi_period) {
        double gains = 0.0, losses = 0.0;
        for (int i = 1; i <= rsi_period; i++) {
            double change = prices[idx - rsi_period + i] - prices[idx - rsi_period + i - 1];
            if (change > 0) gains += change;
            else losses -= change;
        }
        
        if (losses > 0) {
            double rs = gains / losses;
            rsi[idx] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            rsi[idx] = 100.0;
        }
    }
    
    // Bollinger Bands (based on SMA)
    if (idx >= sma_period - 1) {
        double variance = 0.0;
        double mean = sma[idx];
        
        for (int i = 0; i < sma_period; i++) {
            double diff = prices[idx - i] - mean;
            variance += diff * diff;
        }
        
        double std_dev = sqrt(variance / sma_period);
        bollinger_upper[idx] = mean + 2.0 * std_dev;
        bollinger_lower[idx] = mean - 2.0 * std_dev;
    }
}
