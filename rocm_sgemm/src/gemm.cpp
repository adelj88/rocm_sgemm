#include <hip/hip_runtime.h>
#include <rocm_sgemm/gemm.hpp>
#include <rocm_sgemm/kernel_launcher.hpp>

namespace rocm_sgemm
{

template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T>
__host__ void gemm(T* C, T* A, T* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    // Find the best config index for this problem size and layout
    size_t config_idx = detail::find_best_config(M, N, K, layout_A, layout_B, layout_C);

    // Get the params for grid/block calculation
    const auto& config     = detail::kernel_configs[config_idx];
    int         block_size = std::get<0>(config);
    int         block_m    = std::get<1>(config);
    int         block_n    = std::get<2>(config);

    // Calculate grid dimensions
    int grid_m = (M + block_m - 1) / block_m;
    int grid_n = (N + block_n - 1) / block_n;

    dim3 grid_dim(grid_n * grid_m);
    dim3 block_dim(block_size);

    // Launch via kernel lookup table
    // Pass block_m and block_n so launcher can check alignment at runtime
    kernel_launcher<T, layout_C, layout_A, layout_B>::launch(config_idx,
                                                             C,
                                                             A,
                                                             B,
                                                             M,
                                                             N,
                                                             K,
                                                             block_m,
                                                             block_n,
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

// Instantiate for float type
INSTANTIATE_GEMM_FOR_TYPE(float)

} // namespace rocm_sgemm
