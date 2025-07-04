#include <rocblas_wrapper/gemm.hpp>

namespace rocblas_wrapper
{

bool init_rocblas()
{
    if(handle != nullptr)
    {
        return true; // Already initialized
    }

    rocblas_status status = rocblas_create_handle(&handle);
    return (status == rocblas_status_success);
}

void cleanup_rocblas()
{
    if(handle != nullptr)
    {
        rocblas_destroy_handle(handle);
        handle = nullptr;
    }
}

template<bool TRANSPOSE_A, bool TRANSPOSE_B>
__host__ void gemm(float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    if(handle == nullptr)
    {
        throw std::runtime_error("rocBLAS not initialized. Call init_rocblas() first.");
    }

    // Set execution stream
    rocblas_status status = rocblas_set_stream(handle, stream);
    if(status != rocblas_status_success)
    {
        throw std::runtime_error("Failed to set rocBLAS stream");
    }

    // Perform C = A * B (alpha=1.0, beta=0.0)
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    status = rocblas_sgemm(handle,
                           (TRANSPOSE_A) ? rocblas_operation_transpose : rocblas_operation_none,
                           (TRANSPOSE_B) ? rocblas_operation_transpose : rocblas_operation_none,
                           M,
                           N,
                           K,
                           &alpha,
                           A,
                           (TRANSPOSE_A) ? K : M, // lda
                           B,
                           (TRANSPOSE_B) ? N : K, // ldb
                           &beta,
                           C,
                           M); // Leading dimension of C

    if(status != rocblas_status_success)
    {
        throw std::runtime_error("rocBLAS SGEMM failed");
    }
}

template __host__ void gemm<true, true>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<true, false>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<false, true>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

template __host__ void gemm<false, false>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

} // namespace rocblas_wrapper
