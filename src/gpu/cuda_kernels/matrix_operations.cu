/*
 * CUDA Matrix Operations Kernels
 * High-performance matrix multiplication and linear algebra operations
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cufft.h>
#include <cusolverDn.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32
#define MAX_SHARED_MEMORY 48000

// Matrix multiplication kernel with shared memory optimization
__global__ void matrix_multiply_shared(
    const double* A, const double* B, double* C,
    int M, int N, int K
) {
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    double sum = 0.0;
    
    for (int k = 0; k < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
        // Load tiles into shared memory
        if (row < M && k * BLOCK_SIZE + tx < K) {
            As[ty][tx] = A[row * K + k * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        
        if (col < N && k * BLOCK_SIZE + ty < K) {
            Bs[ty][tx] = B[(k * BLOCK_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Tensor-core optimized matrix multiplication (for compute capability 7.0+)
__global__ void matrix_multiply_tensorcore(
    const half* A, const half* B, half* C,
    int M, int N, int K
) {
    // Use wmma (Warp Matrix Multiply-Accumulate) API for tensor cores
    #if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_row = warp_id / (N / 16);
    const int warp_col = warp_id % (N / 16);
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + warp_row * 16 * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + warp_col * 16, N);
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    wmma::store_matrix_sync(C + warp_row * 16 * N + warp_col * 16, c_frag, N, wmma::mem_row_major);
    #endif
}

// Cholesky decomposition kernel
__global__ void cholesky_decomposition(double* A, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int k = 0; k < n; k++) {
        // Diagonal element
        if (tid == k) {
            double sum = 0.0;
            for (int j = 0; j < k; j++) {
                sum += A[k * n + j] * A[k * n + j];
            }
            A[k * n + k] = sqrt(A[k * n + k] - sum);
        }
        
        __syncthreads();
        
        // Lower triangular elements
        if (tid > k && tid < n) {
            double sum = 0.0;
            for (int j = 0; j < k; j++) {
                sum += A[tid * n + j] * A[k * n + j];
            }
            A[tid * n + k] = (A[tid * n + k] - sum) / A[k * n + k];
        }
        
        __syncthreads();
    }
}

// LU decomposition with partial pivoting
__global__ void lu_decomposition_kernel(
    double* A, int* pivot, int n, int step
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (step >= n) return;
    
    // Find pivot
    if (tid == step) {
        int max_idx = step;
        double max_val = fabs(A[step * n + step]);
        
        for (int i = step + 1; i < n; i++) {
            double val = fabs(A[i * n + step]);
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        
        pivot[step] = max_idx;
        
        // Swap rows if needed
        if (max_idx != step) {
            for (int j = 0; j < n; j++) {
                double temp = A[step * n + j];
                A[step * n + j] = A[max_idx * n + j];
                A[max_idx * n + j] = temp;
            }
        }
    }
    
    __syncthreads();
    
    // Gaussian elimination
    if (tid > step && tid < n) {
        double factor = A[tid * n + step] / A[step * n + step];
        A[tid * n + step] = factor;
        
        for (int j = step + 1; j < n; j++) {
            A[tid * n + j] -= factor * A[step * n + j];
        }
    }
}

// QR decomposition using Householder reflections
__global__ void qr_decomposition_kernel(
    double* A, double* Q, double* R, int m, int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize Q as identity matrix
    if (tid < m * m) {
        int i = tid / m;
        int j = tid % m;
        Q[tid] = (i == j) ? 1.0 : 0.0;
    }
    
    // Copy A to R
    if (tid < m * n) {
        R[tid] = A[tid];
    }
    
    __syncthreads();
    
    for (int k = 0; k < n; k++) {
        __shared__ double v[256]; // Householder vector
        __shared__ double beta;
        
        // Compute Householder vector
        if (tid < m - k) {
            int idx = k + tid;
            v[tid] = R[idx * n + k];
            
            if (tid == 0) {
                double norm = 0.0;
                for (int i = 0; i < m - k; i++) {
                    norm += v[i] * v[i];
                }
                norm = sqrt(norm);
                
                if (v[0] >= 0) norm = -norm;
                v[0] -= norm;
                
                double v_norm = 0.0;
                for (int i = 0; i < m - k; i++) {
                    v_norm += v[i] * v[i];
                }
                beta = 2.0 / v_norm;
            }
        }
        
        __syncthreads();
        
        // Apply Householder reflection to R
        if (tid < (m - k) * (n - k)) {
            int i = tid / (n - k);
            int j = tid % (n - k);
            
            double sum = 0.0;
            for (int l = 0; l < m - k; l++) {
                sum += v[l] * R[(k + l) * n + (k + j)];
            }
            
            R[(k + i) * n + (k + j)] -= beta * v[i] * sum;
        }
        
        // Apply Householder reflection to Q
        if (tid < m * (m - k)) {
            int i = tid / (m - k);
            int j = tid % (m - k);
            
            double sum = 0.0;
            for (int l = 0; l < m - k; l++) {
                sum += Q[i * m + (k + l)] * v[l];
            }
            
            Q[i * m + (k + j)] -= beta * sum * v[j];
        }
        
        __syncthreads();
    }
}

// Singular Value Decomposition (simplified version)
__global__ void svd_jacobi_kernel(
    double* A, double* U, double* S, double* V,
    int m, int n, int max_iterations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize U and V as identity matrices
    if (tid < m * m) {
        int i = tid / m;
        int j = tid % m;
        U[tid] = (i == j) ? 1.0 : 0.0;
    }
    
    if (tid < n * n) {
        int i = tid / n;
        int j = tid % n;
        V[tid] = (i == j) ? 1.0 : 0.0;
    }
    
    __syncthreads();
    
    // Jacobi iterations for bidiagonal matrix
    for (int iter = 0; iter < max_iterations; iter++) {
        bool converged = true;
        
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (tid == 0) {
                    // Compute rotation angle
                    double a11 = A[i * n + i];
                    double a22 = A[j * n + j];
                    double a12 = A[i * n + j];
                    double a21 = A[j * n + i];
                    
                    if (fabs(a12) > 1e-10 || fabs(a21) > 1e-10) {
                        converged = false;
                        
                        double theta = 0.5 * atan2(2.0 * a12, a11 - a22);
                        double c = cos(theta);
                        double s = sin(theta);
                        
                        // Apply rotation to A
                        for (int k = 0; k < n; k++) {
                            double temp1 = A[i * n + k];
                            double temp2 = A[j * n + k];
                            A[i * n + k] = c * temp1 + s * temp2;
                            A[j * n + k] = -s * temp1 + c * temp2;
                        }
                        
                        for (int k = 0; k < m; k++) {
                            double temp1 = A[k * n + i];
                            double temp2 = A[k * n + j];
                            A[k * n + i] = c * temp1 + s * temp2;
                            A[k * n + j] = -s * temp1 + c * temp2;
                        }
                    }
                }
                
                __syncthreads();
            }
        }
        
        if (converged) break;
    }
    
    // Extract singular values
    if (tid < n) {
        S[tid] = A[tid * n + tid];
    }
}

// Eigenvalue computation using power iteration
__global__ void power_iteration_kernel(
    const double* A, double* eigenvector, double* eigenvalue,
    int n, int max_iterations, double tolerance
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ double lambda;
    __shared__ double norm;
    
    // Initialize random eigenvector
    if (tid < n) {
        eigenvector[tid] = 1.0 / sqrt(n);
    }
    
    __syncthreads();
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Matrix-vector multiplication: v = A * v
        if (tid < n) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) {
                sum += A[tid * n + j] * eigenvector[j];
            }
            eigenvector[tid] = sum;
        }
        
        __syncthreads();
        
        // Compute norm
        if (tid == 0) {
            norm = 0.0;
            for (int i = 0; i < n; i++) {
                norm += eigenvector[i] * eigenvector[i];
            }
            norm = sqrt(norm);
        }
        
        __syncthreads();
        
        // Normalize eigenvector
        if (tid < n) {
            eigenvector[tid] /= norm;
        }
        
        // Compute eigenvalue (Rayleigh quotient)
        if (tid == 0) {
            lambda = 0.0;
            for (int i = 0; i < n; i++) {
                double sum = 0.0;
                for (int j = 0; j < n; j++) {
                    sum += A[i * n + j] * eigenvector[j];
                }
                lambda += eigenvector[i] * sum;
            }
        }
        
        __syncthreads();
        
        // Check convergence
        if (iter > 0 && fabs(*eigenvalue - lambda) < tolerance) {
            break;
        }
        
        if (tid == 0) {
            *eigenvalue = lambda;
        }
        
        __syncthreads();
    }
}

// Matrix inverse using Gauss-Jordan elimination
__global__ void matrix_inverse_kernel(double* A, double* inv, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize inverse as identity matrix
    if (tid < n * n) {
        int i = tid / n;
        int j = tid % n;
        inv[tid] = (i == j) ? 1.0 : 0.0;
    }
    
    __syncthreads();
    
    // Gauss-Jordan elimination
    for (int k = 0; k < n; k++) {
        // Find pivot
        __shared__ int pivot_row;
        __shared__ double pivot_val;
        
        if (tid == 0) {
            pivot_row = k;
            pivot_val = fabs(A[k * n + k]);
            
            for (int i = k + 1; i < n; i++) {
                double val = fabs(A[i * n + k]);
                if (val > pivot_val) {
                    pivot_val = val;
                    pivot_row = i;
                }
            }
        }
        
        __syncthreads();
        
        // Swap rows
        if (pivot_row != k && tid < n) {
            double temp_a = A[k * n + tid];
            double temp_inv = inv[k * n + tid];
            
            A[k * n + tid] = A[pivot_row * n + tid];
            inv[k * n + tid] = inv[pivot_row * n + tid];
            
            A[pivot_row * n + tid] = temp_a;
            inv[pivot_row * n + tid] = temp_inv;
        }
        
        __syncthreads();
        
        // Scale pivot row
        if (tid < n) {
            double pivot = A[k * n + k];
            A[k * n + tid] /= pivot;
            inv[k * n + tid] /= pivot;
        }
        
        __syncthreads();
        
        // Eliminate column
        if (tid < n && tid != k) {
            double factor = A[tid * n + k];
            
            for (int j = 0; j < n; j++) {
                A[tid * n + j] -= factor * A[k * n + j];
                inv[tid * n + j] -= factor * inv[k * n + j];
            }
        }
        
        __syncthreads();
    }
}

// Host functions for kernel launches
extern "C" {
    void launch_matrix_multiply(
        const double* d_A, const double* d_B, double* d_C,
        int M, int N, int K
    ) {
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                     (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        matrix_multiply_shared<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        cudaDeviceSynchronize();
    }
    
    void launch_cholesky_decomposition(double* d_A, int n) {
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        
        cholesky_decomposition<<<gridSize, blockSize>>>(d_A, n);
        cudaDeviceSynchronize();
    }
    
    void launch_matrix_inverse(double* d_A, double* d_inv, int n) {
        int blockSize = 256;
        int gridSize = (n * n + blockSize - 1) / blockSize;
        
        matrix_inverse_kernel<<<gridSize, blockSize>>>(d_A, d_inv, n);
        cudaDeviceSynchronize();
    }
}
