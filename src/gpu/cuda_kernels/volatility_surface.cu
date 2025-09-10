/*
 * CUDA Volatility Surface Construction Kernels
 * GPU-accelerated implied volatility surface calibration and interpolation
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include <cooperative_groups.h>

#define MAX_STRIKES 50
#define MAX_EXPIRIES 20
#define MAX_ITERATIONS 100
#define TOLERANCE 1e-6
#define THREADS_PER_BLOCK 256

// Volatility surface data structure
struct VolatilitySurface {
    double strikes[MAX_STRIKES];
    double expiries[MAX_EXPIRIES];
    double volatilities[MAX_STRIKES * MAX_EXPIRIES];
    int num_strikes;
    int num_expiries;
    double spot_price;
    double risk_free_rate;
};

// Market data point for calibration
struct MarketQuote {
    double strike;
    double expiry;
    double market_price;
    double bid;
    double ask;
    int option_type;  // 0 = call, 1 = put
    double weight;
};

// Cubic spline interpolation coefficients
struct SplineCoeffs {
    double a, b, c, d;
};

// Black-Scholes formula for GPU
__device__ double black_scholes_price_gpu(
    double S, double K, double T, double r, double sigma, int option_type
) {
    if (T <= 0.0) {
        return option_type == 0 ? fmax(S - K, 0.0) : fmax(K - S, 0.0);
    }
    
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    
    // Cumulative normal distribution
    auto norm_cdf = [](double x) -> double {
        return 0.5 * (1.0 + erf(x / sqrt(2.0)));
    };
    
    if (option_type == 0) {  // Call
        return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2);
    } else {  // Put
        return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1);
    }
}

// Vega calculation for GPU
__device__ double calculate_vega_gpu(
    double S, double K, double T, double r, double sigma
) {
    if (T <= 0.0) return 0.0;
    
    double d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double phi_d1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * M_PI);
    
    return S * phi_d1 * sqrt(T);
}

// Newton-Raphson implied volatility kernel
__global__ void implied_volatility_kernel(
    MarketQuote* quotes,
    double* implied_vols,
    int num_quotes,
    double spot_price,
    double risk_free_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_quotes) return;
    
    MarketQuote quote = quotes[idx];
    double market_price = (quote.bid + quote.ask) / 2.0;  // Mid price
    
    // Initial guess
    double sigma = 0.2;
    double price_diff, vega;
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        // Calculate theoretical price
        double theoretical_price = black_scholes_price_gpu(
            spot_price, quote.strike, quote.expiry, 
            risk_free_rate, sigma, quote.option_type
        );
        
        price_diff = theoretical_price - market_price;
        
        if (fabs(price_diff) < TOLERANCE) {
            break;
        }
        
        // Calculate vega
        vega = calculate_vega_gpu(
            spot_price, quote.strike, quote.expiry, 
            risk_free_rate, sigma
        );
        
        if (fabs(vega) < 1e-8) {
            sigma = 0.2;  // Reset to initial guess
            break;
        }
        
        // Newton-Raphson update
        double new_sigma = sigma - price_diff / vega;
        
        // Ensure volatility stays within reasonable bounds
        new_sigma = fmax(new_sigma, 0.001);
        new_sigma = fmin(new_sigma, 5.0);
        
        // Check for convergence
        if (fabs(new_sigma - sigma) < TOLERANCE) {
            sigma = new_sigma;
            break;
        }
        
        sigma = new_sigma;
    }
    
    implied_vols[idx] = sigma;
}

// Bilinear interpolation for volatility surface
__device__ double bilinear_interpolation(
    double x, double y,
    double x1, double x2, double y1, double y2,
    double q11, double q12, double q21, double q22
) {
    double x2x1 = x2 - x1;
    double y2y1 = y2 - y1;
    double x2x = x2 - x;
    double y2y = y2 - y;
    double yy1 = y - y1;
    double xx1 = x - x1;
    
    return 1.0 / (x2x1 * y2y1) * (
        q11 * x2x * y2y +
        q21 * xx1 * y2y +
        q12 * x2x * yy1 +
        q22 * xx1 * yy1
    );
}

// Volatility surface interpolation kernel
__global__ void interpolate_volatility_kernel(
    VolatilitySurface* surface,
    double* query_strikes,
    double* query_expiries,
    double* interpolated_vols,
    int num_queries
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_queries) return;
    
    double strike = query_strikes[idx];
    double expiry = query_expiries[idx];
    
    // Find surrounding grid points
    int strike_idx = 0;
    int expiry_idx = 0;
    
    // Find strike indices
    for (int i = 0; i < surface->num_strikes - 1; i++) {
        if (strike >= surface->strikes[i] && strike <= surface->strikes[i + 1]) {
            strike_idx = i;
            break;
        }
    }
    
    // Find expiry indices
    for (int i = 0; i < surface->num_expiries - 1; i++) {
        if (expiry >= surface->expiries[i] && expiry <= surface->expiries[i + 1]) {
            expiry_idx = i;
            break;
        }
    }
    
    // Get corner volatilities
    double vol_11 = surface->volatilities[expiry_idx * surface->num_strikes + strike_idx];
    double vol_12 = surface->volatilities[(expiry_idx + 1) * surface->num_strikes + strike_idx];
    double vol_21 = surface->volatilities[expiry_idx * surface->num_strikes + strike_idx + 1];
    double vol_22 = surface->volatilities[(expiry_idx + 1) * surface->num_strikes + strike_idx + 1];
    
    // Perform bilinear interpolation
    double interpolated_vol = bilinear_interpolation(
        strike, expiry,
        surface->strikes[strike_idx], surface->strikes[strike_idx + 1],
        surface->expiries[expiry_idx], surface->expiries[expiry_idx + 1],
        vol_11, vol_12, vol_21, vol_22
    );
    
    interpolated_vols[idx] = interpolated_vol;
}

// SVI (Stochastic Volatility Inspired) parameterization kernel
__global__ void svi_calibration_kernel(
    double* strikes,
    double* market_vols,
    double* svi_params,  // [a, b, rho, m, sigma]
    int num_strikes,
    double forward_price,
    double time_to_expiry
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx != 0) return;  // Single thread for optimization
    
    // SVI parameters
    double a = svi_params[0];
    double b = svi_params[1];
    double rho = svi_params[2];
    double m = svi_params[3];
    double sigma = svi_params[4];
    
    // Objective function: minimize sum of squared errors
    double total_error = 0.0;
    
    for (int i = 0; i < num_strikes; i++) {
        double k = log(strikes[i] / forward_price);  // Log-moneyness
        
        // SVI formula: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
        double k_m = k - m;
        double svi_var = a + b * (rho * k_m + sqrt(k_m * k_m + sigma * sigma));
        double svi_vol = sqrt(fmax(svi_var / time_to_expiry, 1e-8));
        
        double error = svi_vol - market_vols[i];
        total_error += error * error;
    }
    
    // Store the error for external optimizer
    svi_params[5] = total_error;
}

// Heston model calibration kernel
__global__ void heston_calibration_kernel(
    MarketQuote* quotes,
    double* heston_params,  // [v0, kappa, theta, sigma, rho]
    int num_quotes,
    double spot_price,
    double risk_free_rate,
    int num_time_steps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_quotes) return;
    
    MarketQuote quote = quotes[idx];
    
    // Heston parameters
    double v0 = heston_params[0];      // Initial variance
    double kappa = heston_params[1];   // Mean reversion speed
    double theta = heston_params[2];   // Long-term variance
    double sigma = heston_params[3];   // Vol of vol
    double rho = heston_params[4];     // Correlation
    
    // Monte Carlo simulation for Heston model
    curandState state;
    curand_init(idx, 0, 0, &state);
    
    double dt = quote.expiry / num_time_steps;
    double sqrt_dt = sqrt(dt);
    
    int num_paths = 10000;
    double payoff_sum = 0.0;
    
    for (int path = 0; path < num_paths; path++) {
        double S = spot_price;
        double v = v0;
        
        for (int step = 0; step < num_time_steps; step++) {
            // Generate correlated random numbers
            double z1 = curand_normal_double(&state);
            double z2 = curand_normal_double(&state);
            double w1 = z1;
            double w2 = rho * z1 + sqrt(1 - rho * rho) * z2;
            
            // Update variance (Feller condition)
            double v_new = v + kappa * (theta - v) * dt + sigma * sqrt(fmax(v, 0.0)) * sqrt_dt * w2;
            v = fmax(v_new, 0.0);
            
            // Update stock price
            S = S * exp((risk_free_rate - 0.5 * v) * dt + sqrt(fmax(v, 0.0)) * sqrt_dt * w1);
        }
        
        // Calculate payoff
        double payoff;
        if (quote.option_type == 0) {  // Call
            payoff = fmax(S - quote.strike, 0.0);
        } else {  // Put
            payoff = fmax(quote.strike - S, 0.0);
        }
        
        payoff_sum += payoff;
    }
    
    // Discount to present value
    double model_price = exp(-risk_free_rate * quote.expiry) * (payoff_sum / num_paths);
    double market_price = (quote.bid + quote.ask) / 2.0;
    
    // Store pricing error for calibration
    double error = (model_price - market_price) / market_price;
    
    // This would typically be accumulated for optimization
    // For simplicity, we store individual errors
    quotes[idx].weight = error * error;  // Use weight field to store error
}

// Volatility smile fitting using cubic splines
__global__ void cubic_spline_volatility_kernel(
    double* strikes,
    double* volatilities,
    SplineCoeffs* coeffs,
    int num_points,
    double expiry
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_points - 1) return;
    
    // Calculate cubic spline coefficients for segment [idx, idx+1]
    double h = strikes[idx + 1] - strikes[idx];
    double dy = volatilities[idx + 1] - volatilities[idx];
    
    // For natural spline (simplified)
    double m0 = 0.0, m1 = 0.0;  // Second derivatives at endpoints
    
    if (idx > 0) {
        double h_prev = strikes[idx] - strikes[idx - 1];
        double dy_prev = volatilities[idx] - volatilities[idx - 1];
        m0 = 6.0 * (dy / h - dy_prev / h_prev) / (h + h_prev);
    }
    
    if (idx < num_points - 2) {
        double h_next = strikes[idx + 2] - strikes[idx + 1];
        double dy_next = volatilities[idx + 2] - volatilities[idx + 1];
        m1 = 6.0 * (dy_next / h_next - dy / h) / (h + h_next);
    }
    
    // Cubic spline coefficients
    coeffs[idx].a = volatilities[idx];
    coeffs[idx].b = dy / h - h * (2 * m0 + m1) / 6.0;
    coeffs[idx].c = m0 / 2.0;
    coeffs[idx].d = (m1 - m0) / (6.0 * h);
}

// Arbitrage-free volatility surface smoothing
__global__ void arbitrage_free_smoothing_kernel(
    VolatilitySurface* surface,
    double* smoothed_vols,
    double butterfly_threshold,
    double calendar_threshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = surface->num_strikes * surface->num_expiries;
    
    if (idx >= total_points) return;
    
    int strike_idx = idx % surface->num_strikes;
    int expiry_idx = idx / surface->num_strikes;
    
    double current_vol = surface->volatilities[idx];
    
    // Check butterfly arbitrage (convexity in strike)
    if (strike_idx > 0 && strike_idx < surface->num_strikes - 1) {
        double vol_left = surface->volatilities[expiry_idx * surface->num_strikes + strike_idx - 1];
        double vol_right = surface->volatilities[expiry_idx * surface->num_strikes + strike_idx + 1];
        
        double butterfly = vol_left - 2 * current_vol + vol_right;
        
        if (butterfly < -butterfly_threshold) {
            // Smooth to remove arbitrage
            current_vol = (vol_left + vol_right) / 2.0;
        }
    }
    
    // Check calendar arbitrage (monotonicity in time)
    if (expiry_idx > 0) {
        double vol_prev = surface->volatilities[(expiry_idx - 1) * surface->num_strikes + strike_idx];
        double variance_current = current_vol * current_vol * surface->expiries[expiry_idx];
        double variance_prev = vol_prev * vol_prev * surface->expiries[expiry_idx - 1];
        
        if (variance_current < variance_prev - calendar_threshold) {
            // Adjust to maintain calendar spread
            double min_variance = variance_prev + calendar_threshold;
            current_vol = sqrt(min_variance / surface->expiries[expiry_idx]);
        }
    }
    
    smoothed_vols[idx] = current_vol;
}

// Local volatility calculation from implied volatility surface
__global__ void local_volatility_kernel(
    VolatilitySurface* surface,
    double* local_vols,
    double dividend_yield
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = surface->num_strikes * surface->num_expiries;
    
    if (idx >= total_points) return;
    
    int strike_idx = idx % surface->num_strikes;
    int expiry_idx = idx / surface->num_strikes;
    
    if (strike_idx == 0 || strike_idx == surface->num_strikes - 1 ||
        expiry_idx == 0 || expiry_idx == surface->num_expiries - 1) {
        // Boundary conditions - use implied volatility
        local_vols[idx] = surface->volatilities[idx];
        return;
    }
    
    double K = surface->strikes[strike_idx];
    double T = surface->expiries[expiry_idx];
    double r = surface->risk_free_rate;
    double q = dividend_yield;
    
    // Get implied volatility and derivatives
    double sigma = surface->volatilities[idx];
    
    // Finite difference derivatives
    double dK = surface->strikes[strike_idx + 1] - surface->strikes[strike_idx - 1];
    double dT = surface->expiries[expiry_idx + 1] - surface->expiries[expiry_idx - 1];
    
    // dσ/dK
    double sigma_right = surface->volatilities[expiry_idx * surface->num_strikes + strike_idx + 1];
    double sigma_left = surface->volatilities[expiry_idx * surface->num_strikes + strike_idx - 1];
    double dsigma_dK = (sigma_right - sigma_left) / dK;
    
    // d²σ/dK²
    double d2sigma_dK2 = (sigma_right - 2 * sigma + sigma_left) / (dK * dK / 4);
    
    // dσ/dT
    double sigma_next = surface->volatilities[(expiry_idx + 1) * surface->num_strikes + strike_idx];
    double sigma_prev = surface->volatilities[(expiry_idx - 1) * surface->num_strikes + strike_idx];
    double dsigma_dT = (sigma_next - sigma_prev) / dT;
    
    // Dupire formula components
    double d1 = (log(surface->spot_price / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    
    double numerator = dsigma_dT + (r - q) * K * dsigma_dK + 0.5 * sigma * sigma * K * K * d2sigma_dK2;
    double denominator = (1 + K * d1 * sqrt(T) * dsigma_dK) * (1 + K * d1 * sqrt(T) * dsigma_dK) + 
                        K * K * T * sigma * (d2sigma_dK2 - d1 * sqrt(T) * dsigma_dK * dsigma_dK);
    
    double local_var = 2 * sigma * numerator / denominator;
    
    // Ensure positive variance
    local_vols[idx] = sqrt(fmax(local_var, 0.001));
}

// Host functions for kernel launches
extern "C" {
    void launch_implied_volatility_calibration(
        MarketQuote* d_quotes,
        double* d_implied_vols,
        int num_quotes,
        double spot_price,
        double risk_free_rate
    ) {
        int blocks = (num_quotes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        implied_volatility_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_quotes, d_implied_vols, num_quotes, spot_price, risk_free_rate
        );
        
        cudaDeviceSynchronize();
    }
    
    void launch_volatility_surface_interpolation(
        VolatilitySurface* d_surface,
        double* d_query_strikes,
        double* d_query_expiries,
        double* d_interpolated_vols,
        int num_queries
    ) {
        int blocks = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        interpolate_volatility_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_surface, d_query_strikes, d_query_expiries, d_interpolated_vols, num_queries
        );
        
        cudaDeviceSynchronize();
    }
    
    void launch_arbitrage_free_smoothing(
        VolatilitySurface* d_surface,
        double* d_smoothed_vols,
        double butterfly_threshold,
        double calendar_threshold
    ) {
        int total_points = d_surface->num_strikes * d_surface->num_expiries;
        int blocks = (total_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        arbitrage_free_smoothing_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_surface, d_smoothed_vols, butterfly_threshold, calendar_threshold
        );
        
        cudaDeviceSynchronize();
    }
    
    void launch_local_volatility_calculation(
        VolatilitySurface* d_surface,
        double* d_local_vols,
        double dividend_yield
    ) {
        int total_points = d_surface->num_strikes * d_surface->num_expiries;
        int blocks = (total_points + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        
        local_volatility_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_surface, d_local_vols, dividend_yield
        );
        
        cudaDeviceSynchronize();
    }
}
