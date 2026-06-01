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

    // Derive the block tile from the config generators (mirrors the kernel's compile-time
    // derivation): block_m/block_n from the warp grid x per-warp tile, block_size from warp count.
    const auto& config            = detail::kernel_configs[config_idx];
    const int   warps_m           = std::get<0>(config);
    const int   warps_n           = std::get<1>(config);
    const int   warp_tile_m_count = std::get<2>(config);
    const int   warp_tile_n_count = std::get<3>(config);
    const int   thread_tile_m     = std::get<4>(config);
    const int   thread_tile_n     = std::get<5>(config);
    const int   threads_n         = std::get<6>(config);

    const int threads_m  = warp_size / threads_n;
    const int block_m    = warps_m * warp_tile_m_count * thread_tile_m * threads_m;
    const int block_n    = warps_n * warp_tile_n_count * thread_tile_n * threads_n;
    const int block_size = warps_m * warps_n * warp_size;

    int grid_m = (M + block_m - 1) / block_m;
    int grid_n = (N + block_n - 1) / block_n;

    dim3 grid_dim(grid_n * grid_m);
    dim3 block_dim(block_size);

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
        T*, T*, T*, size_t, size_t, size_t, hipStream_t&);                             \
    template void gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>( \
        T*, T*, T*, size_t, size_t, size_t, hipStream_t&);                             \
    template void gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>( \
        T*, T*, T*, size_t, size_t, size_t, hipStream_t&);                             \
    template void gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>( \
        T*, T*, T*, size_t, size_t, size_t, hipStream_t&);                             \
    template void gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>( \
        T*, T*, T*, size_t, size_t, size_t, hipStream_t&);                             \
    template void gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>( \
        T*, T*, T*, size_t, size_t, size_t, hipStream_t&);                             \
    template void gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>( \
        T*, T*, T*, size_t, size_t, size_t, hipStream_t&);                             \
    template void gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>( \
        T*, T*, T*, size_t, size_t, size_t, hipStream_t&);

// Instantiate for float type
INSTANTIATE_GEMM_FOR_TYPE(float)

} // namespace rocm_sgemm

// =============================================================================
// C ABI entry point for runtime arch-library loading (used by kernel_loader).
// layout_a/b/c: 0 = row_major, 1 = col_major.
// =============================================================================

#define DISPATCH_LAYOUT(lc, la, lb)                                            \
    rocm_sgemm::gemm<rocm_sgemm::m_layout::lc,                                 \
                     rocm_sgemm::m_layout::la,                                 \
                     rocm_sgemm::m_layout::lb>(static_cast<float*>(C),         \
                                               static_cast<float*>(A),         \
                                               static_cast<float*>(B),         \
                                               M,                             \
                                               N,                             \
                                               K,                             \
                                               *static_cast<hipStream_t*>(stream))

extern "C" void rocm_sgemm_f32(void*  C,
                               void*  A,
                               void*  B,
                               size_t M,
                               size_t N,
                               size_t K,
                               int    layout_c,
                               int    layout_a,
                               int    layout_b,
                               void*  stream)
{
    const int key = layout_c << 2 | layout_a << 1 | layout_b;
    switch(key)
    {
        case 0: DISPATCH_LAYOUT(row_major, row_major, row_major); break;
        case 1: DISPATCH_LAYOUT(row_major, row_major, col_major); break;
        case 2: DISPATCH_LAYOUT(row_major, col_major, row_major); break;
        case 3: DISPATCH_LAYOUT(row_major, col_major, col_major); break;
        case 4: DISPATCH_LAYOUT(col_major, row_major, row_major); break;
        case 5: DISPATCH_LAYOUT(col_major, row_major, col_major); break;
        case 6: DISPATCH_LAYOUT(col_major, col_major, row_major); break;
        case 7: DISPATCH_LAYOUT(col_major, col_major, col_major); break;
        default: break;
    }
}
