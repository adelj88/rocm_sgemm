#include <rocm_sgemm/gemm.hpp>
#include <rocm_sgemm/kernel_launcher.hpp>

namespace rocm_sgemm
{

/**
 * Implementation of gemm() interface using the kernel launcher with generated configurations.
 */
template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T>
__host__ void gemm(T* C, T* A, T* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    // Get optimal parameters from generated configuration
    auto params = get_gemm_params(M, N, K, layout_C, layout_A, layout_B);

    int block_m = params.block_m;
    int block_n = params.block_n;

    // Calculate grid dimensions
    int grid_m = (M + block_m - 1) / block_m;
    int grid_n = (N + block_n - 1) / block_n;

    dim3 grid_dim(grid_n * grid_m);
    dim3 block_dim(params.block_size);

    // Launch kernel using the template launcher
    kernel_launcher<T, layout_C, layout_A, layout_B>::launch(params,
                                                             C,
                                                             A,
                                                             B,
                                                             M,
                                                             N,
                                                             K,
                                                             grid_dim,
                                                             block_dim,
                                                             stream);
}

// Macro to instantiate all layout combinations for a type
#define INSTANTIATE_GEMM_FOR_TYPE(T)                                                   \
    template void gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                            \
        T*,                                                                            \
        T*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                            \
        T*,                                                                            \
        T*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                            \
        T*,                                                                            \
        T*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                            \
        T*,                                                                            \
        T*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                            \
        T*,                                                                            \
        T*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                            \
        T*,                                                                            \
        T*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                            \
        T*,                                                                            \
        T*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);                                                                 \
    template void gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                            \
        T*,                                                                            \
        T*,                                                                            \
        size_t,                                                                        \
        size_t,                                                                        \
        size_t,                                                                        \
        hipStream_t&);

INSTANTIATE_GEMM_FOR_TYPE(float)

} // namespace rocm_sgemm
