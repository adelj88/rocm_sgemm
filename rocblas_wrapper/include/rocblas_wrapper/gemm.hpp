#ifndef ROCBLAS_WRAPPER_GEMM_HPP
#define ROCBLAS_WRAPPER_GEMM_HPP

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdexcept>

/**
 * @brief rocBLAS wrapper for matrix multiplication
 *
 * Provides a simplified C++ interface to rocBLAS GEMM with handle management
 * and exception-based error handling.
 */
namespace rocblas_wrapper
{

/// Global rocBLAS handle (singleton pattern)
static rocblas_handle handle = nullptr;

/**
 * @brief Initialize rocBLAS library
 *
 * Creates a rocBLAS handle for subsequent operations. Safe to call multiple times.
 *
 * @return true if successful, false on failure
 */
bool init_rocblas();

/**
 * @brief Clean up rocBLAS resources
 *
 * Destroys the rocBLAS handle and releases resources. Safe to call multiple times.
 */
void cleanup_rocblas();

/**
 * @brief Matrix multiplication: C = A * B
 *
 * Performs single-precision matrix multiplication using rocBLAS.
 * All matrices must be in row-major order and allocated in GPU memory.
 *
 * @param C      Output matrix (M x N)
 * @param A      Input matrix A (M x K)
 * @param B      Input matrix B (K x N)
 * @param M      Number of rows in A and C
 * @param N      Number of columns in B and C
 * @param K      Number of columns in A and rows in B
 * @param stream HIP stream for execution
 *
 * @throws std::runtime_error if rocBLAS not initialized or operation fails
 *
 * @pre init_rocblas() must be called first
 * @pre All matrices must be allocated in GPU memory
 */
template<bool TRANSPOSE_A, bool TRANSPOSE_B>
__host__ void gemm(float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void
    gemm<true, true>(float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void
    gemm<true, false>(float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void
    gemm<false, true>(float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void gemm<false, false>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

} // namespace rocblas_wrapper

#endif // ROCBLAS_WRAPPER_GEMM_HPP
