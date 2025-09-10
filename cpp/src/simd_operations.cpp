/*!
 * SIMD Operations Implementation
 * High-performance vectorized operations for trading calculations
 */

#include "algoveda/core.hpp"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace algoveda {
namespace core {

// SIMD utility functions
class SIMDOperations {
public:
    // Vector addition with AVX-512
    static void vector_add_f32_avx512(const float* a, const float* b, float* result, size_t size) {
        size_t simd_size = size & ~15;  // Process in chunks of 16
        
        #if defined(__AVX512F__)
        for (size_t i = 0; i < simd_size; i += 16) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            __m512 vr = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(&result[i], vr);
        }
        #else
        // Fallback to AVX2
        simd_size = size & ~7;
        for (size_t i = 0; i < simd_size; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vr = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&result[i], vr);
        }
        #endif
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    
    // Dot product with FMA
    static double dot_product_fma(const double* a, const double* b, size_t size) {
        __m256d sum = _mm256_setzero_pd();
        size_t simd_size = size & ~3;  // Process in chunks of 4
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            sum = _mm256_fmadd_pd(va, vb, sum);
        }
        
        // Horizontal sum
        double result[4];
        _mm256_storeu_pd(result, sum);
        double total = result[0] + result[1] + result[2] + result[3];
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            total += a[i] * b[i];
        }
        
        return total;
    }
    
    // Moving average with SIMD
    static void moving_average_avx2(const float* prices, float* ma, size_t size, size_t window) {
        if (size < window) return;
        
        // Calculate initial sum
        __m256 sum = _mm256_setzero_ps();
        float scalar_sum = 0.0f;
        
        size_t simd_window = window & ~7;
        for (size_t i = 0; i < simd_window; i += 8) {
            __m256 v = _mm256_loadu_ps(&prices[i]);
            sum = _mm256_add_ps(sum, v);
        }
        
        // Extract sum components
        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum);
        for (int i = 0; i < 8; ++i) {
            scalar_sum += sum_array[i];
        }
        
        // Handle remaining elements
        for (size_t i = simd_window; i < window; ++i) {
            scalar_sum += prices[i];
        }
        
        ma[window - 1] = scalar_sum / window;
        
        // Sliding window
        for (size_t i = window; i < size; ++i) {
            scalar_sum = scalar_sum - prices[i - window] + prices[i];
            ma[i] = scalar_sum / window;
        }
    }
    
    // Exponential moving average
    static void ema_avx2(const float* prices, float* ema, size_t size, float alpha) {
        if (size == 0) return;
        
        ema[0] = prices[0];
        __m256 valpha = _mm256_set1_ps(alpha);
        __m256 vone_minus_alpha = _mm256_set1_ps(1.0f - alpha);
        
        size_t simd_start = 1;
        size_t simd_size = ((size - 1) & ~7) + 1;
        
        for (size_t i = simd_start; i < simd_size; i += 8) {
            size_t actual_size = std::min(8UL, size - i);
            
            // Load previous EMA values
            __m256 prev_ema;
            if (actual_size == 8) {
                prev_ema = _mm256_loadu_ps(&ema[i - 1]);
            } else {
                float temp[8] = {0};
                for (size_t j = 0; j < actual_size; ++j) {
                    temp[j] = ema[i + j - 1];
                }
                prev_ema = _mm256_loadu_ps(temp);
            }
            
            // Load current prices
            __m256 curr_prices;
            if (actual_size == 8) {
                curr_prices = _mm256_loadu_ps(&prices[i]);
            } else {
                float temp[8] = {0};
                for (size_t j = 0; j < actual_size; ++j) {
                    temp[j] = prices[i + j];
                }
                curr_prices = _mm256_loadu_ps(temp);
            }
            
            // EMA calculation: alpha * price + (1 - alpha) * prev_ema
            __m256 result = _mm256_fmadd_ps(valpha, curr_prices, 
                                          _mm256_mul_ps(vone_minus_alpha, prev_ema));
            
            // Store results
            if (actual_size == 8) {
                _mm256_storeu_ps(&ema[i], result);
            } else {
                float temp[8];
                _mm256_storeu_ps(temp, result);
                for (size_t j = 0; j < actual_size; ++j) {
                    ema[i + j] = temp[j];
                }
            }
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            ema[i] = alpha * prices[i] + (1.0f - alpha) * ema[i - 1];
        }
    }
    
    // Standard deviation calculation
    static double standard_deviation_avx2(const double* data, size_t size) {
        if (size < 2) return 0.0;
        
        // Calculate mean
        __m256d sum = _mm256_setzero_pd();
        size_t simd_size = size & ~3;
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d v = _mm256_loadu_pd(&data[i]);
            sum = _mm256_add_pd(sum, v);
        }
        
        double sum_array[4];
        _mm256_storeu_pd(sum_array, sum);
        double mean = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
        
        for (size_t i = simd_size; i < size; ++i) {
            mean += data[i];
        }
        mean /= size;
        
        // Calculate variance
        __m256d vmean = _mm256_set1_pd(mean);
        __m256d var_sum = _mm256_setzero_pd();
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d v = _mm256_loadu_pd(&data[i]);
            __m256d diff = _mm256_sub_pd(v, vmean);
            var_sum = _mm256_fmadd_pd(diff, diff, var_sum);
        }
        
        double var_array[4];
        _mm256_storeu_pd(var_array, var_sum);
        double variance = var_array[0] + var_array[1] + var_array[2] + var_array[3];
        
        for (size_t i = simd_size; i < size; ++i) {
            double diff = data[i] - mean;
            variance += diff * diff;
        }
        
        return std::sqrt(variance / (size - 1));
    }
    
    // Correlation calculation
    static double correlation_avx2(const double* x, const double* y, size_t size) {
        if (size < 2) return 0.0;
        
        // Calculate means
        double mean_x = 0.0, mean_y = 0.0;
        __m256d sum_x = _mm256_setzero_pd();
        __m256d sum_y = _mm256_setzero_pd();
        size_t simd_size = size & ~3;
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d vx = _mm256_loadu_pd(&x[i]);
            __m256d vy = _mm256_loadu_pd(&y[i]);
            sum_x = _mm256_add_pd(sum_x, vx);
            sum_y = _mm256_add_pd(sum_y, vy);
        }
        
        double sum_x_array[4], sum_y_array[4];
        _mm256_storeu_pd(sum_x_array, sum_x);
        _mm256_storeu_pd(sum_y_array, sum_y);
        
        for (int i = 0; i < 4; ++i) {
            mean_x += sum_x_array[i];
            mean_y += sum_y_array[i];
        }
        
        for (size_t i = simd_size; i < size; ++i) {
            mean_x += x[i];
            mean_y += y[i];
        }
        
        mean_x /= size;
        mean_y /= size;
        
        // Calculate covariance and variances
        __m256d vmean_x = _mm256_set1_pd(mean_x);
        __m256d vmean_y = _mm256_set1_pd(mean_y);
        __m256d cov_sum = _mm256_setzero_pd();
        __m256d var_x_sum = _mm256_setzero_pd();
        __m256d var_y_sum = _mm256_setzero_pd();
        
        for (size_t i = 0; i < simd_size; i += 4) {
            __m256d vx = _mm256_loadu_pd(&x[i]);
            __m256d vy = _mm256_loadu_pd(&y[i]);
            __m256d dx = _mm256_sub_pd(vx, vmean_x);
            __m256d dy = _mm256_sub_pd(vy, vmean_y);
            
            cov_sum = _mm256_fmadd_pd(dx, dy, cov_sum);
            var_x_sum = _mm256_fmadd_pd(dx, dx, var_x_sum);
            var_y_sum = _mm256_fmadd_pd(dy, dy, var_y_sum);
        }
        
        double cov_array[4], var_x_array[4], var_y_array[4];
        _mm256_storeu_pd(cov_array, cov_sum);
        _mm256_storeu_pd(var_x_array, var_x_sum);
        _mm256_storeu_pd(var_y_array, var_y_sum);
        
        double covariance = 0.0, variance_x = 0.0, variance_y = 0.0;
        for (int i = 0; i < 4; ++i) {
            covariance += cov_array[i];
            variance_x += var_x_array[i];
            variance_y += var_y_array[i];
        }
        
        for (size_t i = simd_size; i < size; ++i) {
            double dx = x[i] - mean_x;
            double dy = y[i] - mean_y;
            covariance += dx * dy;
            variance_x += dx * dx;
            variance_y += dy * dy;
        }
        
        double std_x = std::sqrt(variance_x);
        double std_y = std::sqrt(variance_y);
        
        if (std_x == 0.0 || std_y == 0.0) return 0.0;
        
        return covariance / (std_x * std_y);
    }
    
    // Matrix multiplication with cache blocking
    static void matrix_multiply_blocked(const float* A, const float* B, float* C, 
                                      size_t M, size_t N, size_t K, size_t block_size = 64) {
        // Clear result matrix
        std::fill_n(C, M * N, 0.0f);
        
        for (size_t bi = 0; bi < M; bi += block_size) {
            for (size_t bj = 0; bj < N; bj += block_size) {
                for (size_t bk = 0; bk < K; bk += block_size) {
                    size_t end_i = std::min(bi + block_size, M);
                    size_t end_j = std::min(bj + block_size, N);
                    size_t end_k = std::min(bk + block_size, K);
                    
                    // Block multiplication
                    for (size_t i = bi; i < end_i; ++i) {
                        for (size_t k = bk; k < end_k; ++k) {
                            __m256 va = _mm256_set1_ps(A[i * K + k]);
                            
                            size_t j = bj;
                            for (; j + 7 < end_j; j += 8) {
                                __m256 vb = _mm256_loadu_ps(&B[k * N + j]);
                                __m256 vc = _mm256_loadu_ps(&C[i * N + j]);
                                vc = _mm256_fmadd_ps(va, vb, vc);
                                _mm256_storeu_ps(&C[i * N + j], vc);
                            }
                            
                            // Handle remaining elements
                            for (; j < end_j; ++j) {
                                C[i * N + j] += A[i * K + k] * B[k * N + j];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Fast logarithm approximation
    static void fast_log_avx2(const float* input, float* output, size_t size) {
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 ln2 = _mm256_set1_ps(0.693147180559945f);
        
        size_t simd_size = size & ~7;
        
        for (size_t i = 0; i < simd_size; i += 8) {
            __m256 x = _mm256_loadu_ps(&input[i]);
            
            // Extract exponent
            __m256i exp_int = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
            exp_int = _mm256_sub_epi32(exp_int, _mm256_set1_epi32(127));
            __m256 exp_float = _mm256_cvtepi32_ps(exp_int);
            
            // Extract mantissa
            __m256i mantissa_int = _mm256_and_si256(_mm256_castps_si256(x), 
                                                  _mm256_set1_epi32(0x007FFFFF));
            mantissa_int = _mm256_or_si256(mantissa_int, _mm256_set1_epi32(0x3F800000));
            __m256 mantissa = _mm256_castsi256_ps(mantissa_int);
            
            // Polynomial approximation for ln(mantissa)
            __m256 t = _mm256_sub_ps(mantissa, one);
            __m256 result = _mm256_set1_ps(-0.333333333f);
            result = _mm256_fmadd_ps(result, t, _mm256_set1_ps(0.5f));
            result = _mm256_fmadd_ps(result, t, _mm256_set1_ps(-1.0f));
            result = _mm256_mul_ps(result, t);
            
            // Combine: ln(x) = exp * ln(2) + ln(mantissa)
            result = _mm256_fmadd_ps(exp_float, ln2, result);
            
            _mm256_storeu_ps(&output[i], result);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            output[i] = std::log(input[i]);
        }
    }
    
    // Fast exponential approximation
    static void fast_exp_avx2(const float* input, float* output, size_t size) {
        const __m256 ln2_inv = _mm256_set1_ps(1.44269504088896f);
        const __m256 ln2 = _mm256_set1_ps(0.693147180559945f);
        const __m256 one = _mm256_set1_ps(1.0f);
        
        size_t simd_size = size & ~7;
        
        for (size_t i = 0; i < simd_size; i += 8) {
            __m256 x = _mm256_loadu_ps(&input[i]);
            
            // Decompose: x = n * ln(2) + r, where |r| < ln(2)/2
            __m256 n_float = _mm256_round_ps(_mm256_mul_ps(x, ln2_inv), 
                                           _MM_FROUND_TO_NEAREST_INT);
            __m256 r = _mm256_fnmadd_ps(n_float, ln2, x);
            
            // Polynomial approximation for exp(r)
            __m256 result = _mm256_set1_ps(0.166666667f);
            result = _mm256_fmadd_ps(result, r, _mm256_set1_ps(0.5f));
            result = _mm256_fmadd_ps(result, r, one);
            result = _mm256_fmadd_ps(result, r, one);
            
            // Scale by 2^n
            __m256i n_int = _mm256_cvtps_epi32(n_float);
            n_int = _mm256_add_epi32(n_int, _mm256_set1_epi32(127));
            n_int = _mm256_slli_epi32(n_int, 23);
            __m256 scale = _mm256_castsi256_ps(n_int);
            
            result = _mm256_mul_ps(result, scale);
            
            _mm256_storeu_ps(&output[i], result);
        }
        
        // Handle remaining elements
        for (size_t i = simd_size; i < size; ++i) {
            output[i] = std::exp(input[i]);
        }
    }
    
    // Black-Scholes vectorized calculation
    static void black_scholes_vectorized(const float* S, const float* K, const float* T,
                                       const float* r, const float* sigma,
                                       float* call_prices, float* put_prices, size_t size) {
        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 sqrt_2pi_inv = _mm256_set1_ps(0.39894228040143f);
        
        size_t simd_size = size & ~7;
        
        for (size_t i = 0; i < simd_size; i += 8) {
            __m256 vS = _mm256_loadu_ps(&S[i]);
            __m256 vK = _mm256_loadu_ps(&K[i]);
            __m256 vT = _mm256_loadu_ps(&T[i]);
            __m256 vr = _mm256_loadu_ps(&r[i]);
            __m256 vsigma = _mm256_loadu_ps(&sigma[i]);
            
            // Calculate d1 and d2
            __m256 sigma_sqrt_T = _mm256_mul_ps(vsigma, _mm256_sqrt_ps(vT));
            __m256 log_S_K[8];
            float S_K_ratio[8], temp_log[8];
            
            _mm256_storeu_ps(S_K_ratio, _mm256_div_ps(vS, vK));
            fast_log_avx2(S_K_ratio, temp_log, 8);
            __m256 vlog_S_K = _mm256_loadu_ps(temp_log);
            
            __m256 d1_num = _mm256_add_ps(vlog_S_K, 
                                        _mm256_mul_ps(vT, _mm256_add_ps(vr, 
                                        _mm256_mul_ps(half, _mm256_mul_ps(vsigma, vsigma)))));
            __m256 d1 = _mm256_div_ps(d1_num, sigma_sqrt_T);
            __m256 d2 = _mm256_sub_ps(d1, sigma_sqrt_T);
            
            // Calculate N(d1) and N(d2) using approximation
            auto norm_cdf_approx = [](const __m256& x) -> __m256 {
                const __m256 a1 = _mm256_set1_ps(0.254829592f);
                const __m256 a2 = _mm256_set1_ps(-0.284496736f);
                const __m256 a3 = _mm256_set1_ps(1.421413741f);
                const __m256 a4 = _mm256_set1_ps(-1.453152027f);
                const __m256 a5 = _mm256_set1_ps(1.061405429f);
                const __m256 p = _mm256_set1_ps(0.3275911f);
                const __m256 one = _mm256_set1_ps(1.0f);
                const __m256 half = _mm256_set1_ps(0.5f);
                
                __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
                __m256 t = _mm256_div_ps(one, _mm256_fmadd_ps(p, abs_x, one));
                
                __m256 erf_approx = a1;
                erf_approx = _mm256_fmadd_ps(erf_approx, t, a2);
                erf_approx = _mm256_fmadd_ps(erf_approx, t, a3);
                erf_approx = _mm256_fmadd_ps(erf_approx, t, a4);
                erf_approx = _mm256_fmadd_ps(erf_approx, t, a5);
                erf_approx = _mm256_mul_ps(erf_approx, t);
                
                // exp(-x^2)
                __m256 x_squared = _mm256_mul_ps(x, x);
                __m256 neg_x_squared = _mm256_sub_ps(_mm256_setzero_ps(), x_squared);
                float exp_input[8], exp_output[8];
                _mm256_storeu_ps(exp_input, neg_x_squared);
                fast_exp_avx2(exp_input, exp_output, 8);
                __m256 exp_neg_x_sq = _mm256_loadu_ps(exp_output);
                
                erf_approx = _mm256_mul_ps(erf_approx, exp_neg_x_sq);
                __m256 erf_result = _mm256_sub_ps(one, erf_approx);
                
                // Handle negative x
                __m256 sign_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);
                erf_result = _mm256_blendv_ps(erf_result, _mm256_sub_ps(_mm256_setzero_ps(), erf_result), sign_mask);
                
                return _mm256_fmadd_ps(half, erf_result, half);
            };
            
            __m256 N_d1 = norm_cdf_approx(d1);
            __m256 N_d2 = norm_cdf_approx(d2);
            __m256 N_neg_d1 = _mm256_sub_ps(_mm256_set1_ps(1.0f), N_d1);
            __m256 N_neg_d2 = _mm256_sub_ps(_mm256_set1_ps(1.0f), N_d2);
            
            // Calculate option prices
            __m256 discount = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), vr), vT);
            float disc_array[8], disc_exp[8];
            _mm256_storeu_ps(disc_array, discount);
            fast_exp_avx2(disc_array, disc_exp, 8);
            __m256 exp_neg_rT = _mm256_loadu_ps(disc_exp);
            
            __m256 discounted_K = _mm256_mul_ps(vK, exp_neg_rT);
            
            // Call price: S * N(d1) - K * exp(-rT) * N(d2)
            __m256 call_price = _mm256_sub_ps(_mm256_mul_ps(vS, N_d1), 
                                            _mm256_mul_ps(discounted_K, N_d2));
            
            // Put price: K * exp(-rT) * N(-d2) - S * N(-d1)
            __m256 put_price = _mm256_sub_ps(_mm256_mul_ps(discounted_K, N_neg_d2), 
                                           _mm256_mul_ps(vS, N_neg_d1));
            
            _mm256_storeu_ps(&call_prices[i], call_price);
            _mm256_storeu_ps(&put_prices[i], put_price);
        }
        
        // Handle remaining elements with scalar code
        for (size_t i = simd_size; i < size; ++i) {
            float d1 = (std::log(S[i] / K[i]) + (r[i] + 0.5f * sigma[i] * sigma[i]) * T[i]) / 
                      (sigma[i] * std::sqrt(T[i]));
            float d2 = d1 - sigma[i] * std::sqrt(T[i]);
            
            auto norm_cdf = [](float x) -> float {
                return 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
            };
            
            float N_d1 = norm_cdf(d1);
            float N_d2 = norm_cdf(d2);
            float N_neg_d1 = 1.0f - N_d1;
            float N_neg_d2 = 1.0f - N_d2;
            
            float discount_factor = std::exp(-r[i] * T[i]);
            
            call_prices[i] = S[i] * N_d1 - K[i] * discount_factor * N_d2;
            put_prices[i] = K[i] * discount_factor * N_neg_d2 - S[i] * N_neg_d1;
        }
    }
};

// Performance benchmarking utilities
class SIMDBenchmark {
public:
    static void benchmark_vector_operations(size_t size = 1000000) {
        std::vector<float> a(size), b(size), result(size);
        
        // Initialize with random data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-100.0f, 100.0f);
        
        for (size_t i = 0; i < size; ++i) {
            a[i] = dis(gen);
            b[i] = dis(gen);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        SIMDOperations::vector_add_f32_avx512(a.data(), b.data(), result.data(), size);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "SIMD Vector Add (" << size << " elements): " 
                  << duration.count() << " microseconds" << std::endl;
        std::cout << "Throughput: " << (size * sizeof(float) * 2) / (duration.count() / 1000.0) / 1024 / 1024 
                  << " MB/s" << std::endl;
    }
    
    static void benchmark_black_scholes(size_t size = 100000) {
        std::vector<float> S(size), K(size), T(size), r(size), sigma(size);
        std::vector<float> call_prices(size), put_prices(size);
        
        // Initialize with realistic option parameters
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> stock_dis(50.0f, 200.0f);
        std::uniform_real_distribution<float> strike_dis(50.0f, 200.0f);
        std::uniform_real_distribution<float> time_dis(0.01f, 2.0f);
        std::uniform_real_distribution<float> rate_dis(0.0f, 0.1f);
        std::uniform_real_distribution<float> vol_dis(0.1f, 0.8f);
        
        for (size_t i = 0; i < size; ++i) {
            S[i] = stock_dis(gen);
            K[i] = strike_dis(gen);
            T[i] = time_dis(gen);
            r[i] = rate_dis(gen);
            sigma[i] = vol_dis(gen);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        SIMDOperations::black_scholes_vectorized(
            S.data(), K.data(), T.data(), r.data(), sigma.data(),
            call_prices.data(), put_prices.data(), size
        );
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "SIMD Black-Scholes (" << size << " options): " 
                  << duration.count() << " microseconds" << std::endl;
        std::cout << "Options per second: " << (size * 1000000.0) / duration.count() << std::endl;
    }
};

} // namespace core
} // namespace algoveda

// C API for external usage
extern "C" {
    void algoveda_vector_add_f32(const float* a, const float* b, float* result, size_t size) {
        algoveda::core::SIMDOperations::vector_add_f32_avx512(a, b, result, size);
    }
    
    double algoveda_dot_product_f64(const double* a, const double* b, size_t size) {
        return algoveda::core::SIMDOperations::dot_product_fma(a, b, size);
    }
    
    void algoveda_moving_average_f32(const float* prices, float* ma, size_t size, size_t window) {
        algoveda::core::SIMDOperations::moving_average_avx2(prices, ma, size, window);
    }
    
    void algoveda_black_scholes_vectorized(const float* S, const float* K, const float* T,
                                         const float* r, const float* sigma,
                                         float* call_prices, float* put_prices, size_t size) {
        algoveda::core::SIMDOperations::black_scholes_vectorized(S, K, T, r, sigma, call_prices, put_prices, size);
    }
}
