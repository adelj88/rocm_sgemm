/*
 * MIT License
 *
 * Copyright (c) 2024 Adel Johar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ROCM_SGEMM_KERNEL_HPP
#define ROCM_SGEMM_KERNEL_HPP

#include "common.hpp"
#include "fragment.hpp"
#include "load.hpp"
#include "mapping.hpp"

namespace rocm_sgemm
{

/**
 * @brief GEMM kernel with hierarchical tiling: block -> warp -> thread levels.
 *
 * Computes C = A * B using three-level tiling with double-buffered shared memory.
 * Uses configurable warp tile counts and thread tile sizes.
 *
 * Algorithm overview:
 * - Each thread block processes a block_m x block_n tile of the output matrix.
 * - Within each block, warps process smaller warp-level tiles.
 * - Each warp tile is divided into configurable sub-tiles (warp_tile_m_count x warp_tile_n_count).
 * - Individual threads process thread_tile_m x thread_tile_n elements.
 * - Double buffering overlaps computation with memory transfers.
 * - Vectorized loads maximize memory bandwidth utilization.
 *
 * @tparam warps_m            Number of warps along the M dimension.
 * @tparam warps_n            Number of warps along the N dimension.
 * @tparam warp_tile_m_count  Number of sub-tiles per warp in M dimension.
 * @tparam warp_tile_n_count  Number of sub-tiles per warp in N dimension.
 * @tparam thread_tile_m      Elements per thread in M dimension.
 * @tparam thread_tile_n      Elements per thread in N dimension.
 * @tparam threads_n          Threads along N within a warp (threads_m = warp_size / threads_n).
 * @tparam block_k            K-dimension tile size for shared memory blocking.
 * @tparam single_buffer      1 = single-buffered LDS, 0 = double-buffered.
 * @tparam swizzle            Swizzle value for mapping.
 *
 * @param C  Output matrix C pointer.
 * @param A  Input matrix A pointer.
 * @param B  Input matrix B pointer.
 * @param M  Number of rows of matrices C and A.
 * @param N  Number of columns of matrices C and B.
 * @param K  Number of columns of matrix A and rows of matrix B.
 */
template<class T,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m_count,
         int      warp_tile_n_count,
         int      thread_tile_m,
         int      thread_tile_n,
         int      threads_n,
         int      block_k,
         int      single_buffer,
         int      swizzle,
         int      is_aligned>
__device__ __forceinline__ void gemm_impl(
    T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B, int M, int N, int K)
{
    // Thread arrangement within warp
    constexpr int threads_m = warp_size / threads_n;

    // Sub-tile dimensions (elements per warp sub-tile)
    constexpr int sub_tile_m = thread_tile_m * threads_m;
    constexpr int sub_tile_n = thread_tile_n * threads_n;

    // Warp tile dimensions
    constexpr int warp_m = warp_tile_m_count * sub_tile_m;
    constexpr int warp_n = warp_tile_n_count * sub_tile_n;

    // Block tile is the warp grid times the per-warp tile; block_size follows from the warp count.
    constexpr int num_warps  = warps_m * warps_n;
    constexpr int block_m    = warps_m * warp_m;
    constexpr int block_n    = warps_n * warp_n;
    constexpr int block_size = num_warps * warp_size;

    // Bank-conflict avoidance via fixed LDS padding on the transposed staging cases
    // (row-major A, col-major B), where consecutive K-rows would otherwise collide on
    // the commit store. The padded row stride staggers them. Native cases pad 0.
    constexpr int pad_a    = (LAYOUT_A == m_layout::row_major) ? lds_pad : 0;
    constexpr int pad_b    = (LAYOUT_B == m_layout::col_major) ? lds_pad : 0;
    constexpr int stride_a = block_m + pad_a;
    constexpr int stride_b = block_n + pad_b;

    constexpr int lds_size = (block_k * stride_a) + (block_k * stride_b);

    // Single buffering uses one LDS tile instead of the double-buffered pair (half the LDS,
    // higher occupancy) at the cost of a read-sync-write barrier per K iteration. The next
    // tile is still staged in registers by prefetch_fragment, so only the LDS commit differs.
    constexpr bool use_single_buffer = (single_buffer != 0);
    constexpr int  lds_buffers       = use_single_buffer ? 1 : 2;

    // Block coordinates
    const int grid_m  = (M + block_m - 1) / block_m;
    const int grid_n  = (N + block_n - 1) / block_n;
    const int tile_id = blockIdx.x;

    using mapper = tile_mapper<block_m, block_n, LAYOUT_A, LAYOUT_B, swizzle>;

    // Get block coordinates
    int block_row, block_col;
    mapper().map_tile(tile_id, grid_m, grid_n, &block_row, &block_col);

    // Shared memory allocation
    __shared__ T lds_mem[lds_buffers * lds_size];

    // Buffer partitioning. In single-buffer mode the "1" tiles alias the "0" tiles.
    T* a_tiles_0 = lds_mem;
    T* a_tiles_1 = use_single_buffer ? a_tiles_0 : (lds_mem + lds_size);
    T* b_tiles_0 = lds_mem + (block_k * stride_a);
    T* b_tiles_1 = use_single_buffer ? b_tiles_0 : (lds_mem + lds_size + (block_k * stride_a));

    // Thread and warp identification
    const int tid      = threadIdx.x;
    const int warp_id  = tid / warp_size;
    const int lane_id  = tid % warp_size;
    const int warp_row = warp_id / warps_n;
    const int warp_col = warp_id % warps_n;

    const int thread_row_in_warp = lane_id / threads_n;
    const int thread_col_in_warp = lane_id % threads_n;

    const int warp_base_row = warp_row * warp_m;
    const int warp_base_col = warp_col * warp_n;

    // Register fragments
    fragment<T, thread_tile_m * thread_tile_n> c_frag[warp_tile_m_count][warp_tile_n_count];
    fragment<T, thread_tile_m>                 a_frag[warp_tile_m_count];
    fragment<T, thread_tile_n>                 b_frag[warp_tile_n_count];

    // Loading setup
    const T* A_base     = A + block_row * ((LAYOUT_A == m_layout::col_major) ? 1 : K);
    const T* B_base     = B + block_col * ((LAYOUT_B == m_layout::col_major) ? K : 1);
    const T* A_tile_ptr = A_base;
    const T* B_tile_ptr = B_base;

    prefetch_fragment<m_input::matrix_a, LAYOUT_A, block_size, block_m, block_k, pad_a, T> pf_a(
        A,
        static_cast<unsigned>(M) * static_cast<unsigned>(K));
    prefetch_fragment<m_input::matrix_b, LAYOUT_B, block_size, block_k, block_n, pad_b, T> pf_b(
        B,
        static_cast<unsigned>(K) * static_cast<unsigned>(N));

    // Initial tile loading
    pf_a.prefetch(A_tile_ptr, M, K, tid);
    pf_b.prefetch(B_tile_ptr, K, N, tid);
    pf_a.commit(a_tiles_0, tid);
    pf_b.commit(b_tiles_0, tid);

    __syncthreads();

    // Double buffer pointers
    T* current_a = a_tiles_0;
    T* current_b = b_tiles_0;
    T* next_a    = a_tiles_1;
    T* next_b    = b_tiles_1;

    const int global_mult_A = block_k * ((LAYOUT_A == m_layout::col_major) ? M : 1);
    const int global_mult_B = block_k * ((LAYOUT_B == m_layout::col_major) ? 1 : N);

    const int thread_a_base = warp_base_row + thread_row_in_warp * thread_tile_m;
    const int thread_b_base = warp_base_col + thread_col_in_warp * thread_tile_n;

    // Flattened extents of the two compute loops: warp-tile sub-tiles per warp, and elements per
    // thread micro-tile. Both iterate in row-major order.
    constexpr size_t num_combos        = warp_tile_m_count * warp_tile_n_count;
    constexpr size_t thread_num_combos = thread_tile_m * thread_tile_n;

    // Main computation loop
    for(int k_tile = 0; k_tile < K - block_k; k_tile += block_k)
    {
        // Prefetch next tiles from global to registers
        const T* next_A = A_tile_ptr + global_mult_A;
        const T* next_B = B_tile_ptr + global_mult_B;

        // One k-slice per step: issue the next-tile global prefetch chunk, read all A then all B
        // fragments from LDS, then run the FMA burst under raised wave priority. The priority
        // bracket keeps the VALU-heavy burst from being preempted by other waves' memory traffic;
        // the grouped reads reuse the full a_frag/b_frag arrays and cost no extra registers.
        [&]<size_t... k_offset_idx>(std::index_sequence<k_offset_idx...>)
        {
            (
                [&]()
                {
                    constexpr size_t k_offset = k_offset_idx;
                    const T*         a_ptr    = current_a + k_offset * stride_a;
                    const T*         b_ptr    = current_b + k_offset * stride_b;

                    // Next-tile prefetch chunk for this k-slice.
                    pf_a.template partial_prefetch<k_offset, block_k>(next_A, M, K, tid);
                    pf_b.template partial_prefetch<k_offset, block_k>(next_B, K, N, tid);

                    // Read all A fragments, then all B fragments, from LDS.
                    [&]<size_t... wm>(std::index_sequence<wm...>) {
                        ((load_matrix(a_frag[wm],
                                      a_ptr + thread_a_base + wm * sub_tile_m,
                                      block_m,
                                      block_k)),
                         ...);
                    }(std::make_index_sequence<warp_tile_m_count>{});
                    [&]<size_t... wn>(std::index_sequence<wn...>) {
                        ((load_matrix(b_frag[wn],
                                      b_ptr + thread_b_base + wn * sub_tile_n,
                                      block_k,
                                      block_n)),
                         ...);
                    }(std::make_index_sequence<warp_tile_n_count>{});

                    // FMA burst over all warp-tile sub-tiles under raised wave priority.
                    __builtin_amdgcn_s_setprio(1);
                    [&]<size_t... i>(std::index_sequence<i...>)
                    {
                        (
                            [&]()
                            {
                                constexpr size_t wm       = i / warp_tile_n_count;
                                constexpr size_t wn       = i % warp_tile_n_count;
                                auto&            dest_ref = c_frag[wm][wn].get();
                                auto&            a_ref    = a_frag[wm].get();
                                auto&            b_ref    = b_frag[wn].get();

                                [&]<size_t... j>(std::index_sequence<j...>)
                                {
                                    (
                                        [&]()
                                        {
                                            constexpr size_t tm     = j / thread_tile_n;
                                            constexpr size_t tn     = j % thread_tile_n;
                                            const int        offset = tm * thread_tile_n;
                                            dest_ref[offset + tn] += a_ref[tm] * b_ref[tn];
                                        }(),
                                        ...);
                                }(std::make_index_sequence<thread_num_combos>{});
                            }(),
                            ...);
                    }(std::make_index_sequence<num_combos>{});
                    __builtin_amdgcn_s_setprio(0);
                }(),
                ...);
        }(std::make_index_sequence<block_k>{});

        // Advance global tile pointers
        A_tile_ptr += global_mult_A;
        B_tile_ptr += global_mult_B;

        if constexpr(use_single_buffer)
        {
            // Read-sync-write: the commit overwrites the same buffer the compute above just
            // read, so every load_matrix must retire before the commit stores begin.
            __syncthreads();
            pf_a.commit(current_a, tid);
            pf_b.commit(current_b, tid);
            __syncthreads();
        }
        else
        {
            // Double buffer: commit the next tile into the free half while the current half
            // is still being read, then swap. One barrier suffices.
            pf_a.commit(next_a, tid);
            pf_b.commit(next_b, tid);

            T* temp_a = current_a;
            T* temp_b = current_b;
            current_a = next_a;
            current_b = next_b;
            next_a    = temp_a;
            next_b    = temp_b;

            __syncthreads();
        }
    }

    // Epilogue for the final K tile — same grouped-read + priority-bracketed burst, no prefetch.
    [&]<size_t... k_offset_idx>(std::index_sequence<k_offset_idx...>)
    {
        (
            [&]()
            {
                constexpr size_t k_offset = k_offset_idx;
                const T*         a_ptr    = current_a + k_offset * stride_a;
                const T*         b_ptr    = current_b + k_offset * stride_b;

                [&]<size_t... wm>(std::index_sequence<wm...>) {
                    ((load_matrix(a_frag[wm],
                                  a_ptr + thread_a_base + wm * sub_tile_m,
                                  block_m,
                                  block_k)),
                     ...);
                }(std::make_index_sequence<warp_tile_m_count>{});
                [&]<size_t... wn>(std::index_sequence<wn...>) {
                    ((load_matrix(b_frag[wn],
                                  b_ptr + thread_b_base + wn * sub_tile_n,
                                  block_k,
                                  block_n)),
                     ...);
                }(std::make_index_sequence<warp_tile_n_count>{});

                __builtin_amdgcn_s_setprio(1);
                [&]<size_t... i>(std::index_sequence<i...>)
                {
                    (
                        [&]()
                        {
                            constexpr size_t wm       = i / warp_tile_n_count;
                            constexpr size_t wn       = i % warp_tile_n_count;
                            auto&            dest_ref = c_frag[wm][wn].get();
                            auto&            a_ref    = a_frag[wm].get();
                            auto&            b_ref    = b_frag[wn].get();

                            [&]<size_t... j>(std::index_sequence<j...>)
                            {
                                (
                                    [&]()
                                    {
                                        constexpr size_t tm     = j / thread_tile_n;
                                        constexpr size_t tn     = j % thread_tile_n;
                                        const int        offset = tm * thread_tile_n;
                                        dest_ref[offset + tn] += a_ref[tm] * b_ref[tn];
                                    }(),
                                    ...);
                            }(std::make_index_sequence<thread_num_combos>{});
                        }(),
                        ...);
                }(std::make_index_sequence<num_combos>{});
                __builtin_amdgcn_s_setprio(0);
            }(),
            ...);
    }(std::make_index_sequence<block_k>{});

    const int row_offset = block_row + warp_base_row + thread_row_in_warp * thread_tile_m;
    const int col_offset = block_col + warp_base_col + thread_col_in_warp * thread_tile_n;

    [&]<size_t... i>(std::index_sequence<i...>)
    {
        (
            [&]()
            {
                constexpr size_t wm            = i / warp_tile_n_count;
                constexpr size_t wn            = i % warp_tile_n_count;
                constexpr int    tile_row_base = wm * sub_tile_m;
                constexpr int    tile_col_base = wn * sub_tile_n;
                auto&            dest_ptr      = c_frag[wm][wn].get();

                [&]<size_t... j>(std::index_sequence<j...>)
                {
                    (
                        [&]()
                        {
                            constexpr size_t tm         = j / thread_tile_n;
                            constexpr size_t tn         = j % thread_tile_n;
                            constexpr int    offset     = tm * thread_tile_n;
                            const int        global_row = row_offset + tile_row_base + tm;
                            const int        global_col = col_offset + tile_col_base + tn;

                            // Aligned tiles sit fully in-bounds, so the per-element bounds branch
                            // is skipped; ragged tiles keep it to drop the overhanging lanes.
                            if constexpr(is_aligned)
                            {
                                if constexpr(LAYOUT_C == m_layout::col_major)
                                {
                                    C[global_col * M + global_row] = dest_ptr[offset + tn];
                                }
                                else
                                {
                                    C[global_row * N + global_col] = dest_ptr[offset + tn];
                                }
                            }
                            else
                            {
                                if(global_row < M && global_col < N)
                                {
                                    if constexpr(LAYOUT_C == m_layout::col_major)
                                    {
                                        C[global_col * M + global_row] = dest_ptr[offset + tn];
                                    }
                                    else
                                    {
                                        C[global_row * N + global_col] = dest_ptr[offset + tn];
                                    }
                                }
                            }
                        }(),
                        ...);
                }(std::make_index_sequence<thread_num_combos>{});
            }(),
            ...);
    }(std::make_index_sequence<num_combos>{});
}

/**
 * @brief Functor wrapping the GEMM kernel launch.
 *
 * The __global__ entry point lives on the struct's static run() so its launch_bounds is a
 * fixed compile-time attribute of a named symbol (the generated instantiation files take
 * &kernel_gemm_impl<...>::run). This matches the shipped-library build exactly and avoids the
 * free-function + per-file #define rename scheme.
 *
 * @tparam T Data type of the matrices.
 * @tparam LAYOUT_C Memory layout of C.
 * @tparam LAYOUT_A Memory layout of A.
 * @tparam LAYOUT_B Memory layout of B.
 * @tparam warps_m Number of warps mapped to the M dimension.
 * @tparam warps_n Number of warps mapped to the N dimension.
 * @tparam warp_tile_m_count Number of sub-tiles per warp in the M dimension.
 * @tparam warp_tile_n_count Number of sub-tiles per warp in the N dimension.
 * @tparam thread_tile_m Elements per thread in the M dimension.
 * @tparam thread_tile_n Elements per thread in the N dimension.
 * @tparam threads_n Threads along N within a warp (threads_m = warp_size / threads_n).
 * @tparam block_k K-dimension tile size for shared memory blocking.
 * @tparam single_buffer 1 = single LDS buffer (half LDS, read-sync-write) for higher
 *                       occupancy; 0 = double buffer (software-pipelined).
 * @tparam swizzle Swizzle size for the block mapping.
 * @tparam is_aligned True if the global matrix dimensions are strictly aligned to the block tiles.
 */
template<class T,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      warps_m,
         int      warps_n,
         int      warp_tile_m_count,
         int      warp_tile_n_count,
         int      thread_tile_m,
         int      thread_tile_n,
         int      threads_n,
         int      block_k,
         int      single_buffer,
         int      swizzle,
         int      is_aligned>
struct kernel_gemm_impl
{
    /**
     * @brief The global entry point for the GEMM kernel.
     *
     * @param C Output matrix C pointer.
     * @param A Input matrix A pointer.
     * @param B Input matrix B pointer.
     * @param M Number of rows in matrices A and C.
     * @param N Number of columns in matrices B and C.
     * @param K Number of columns in A and rows in B.
     */
    __global__ __launch_bounds__(warp_size* warps_m* warps_n) static void run(
        T* __restrict__ C, const T* __restrict__ A, const T* __restrict__ B, int M, int N, int K)
    {
        gemm_impl<T,
                  LAYOUT_C,
                  LAYOUT_A,
                  LAYOUT_B,
                  warps_m,
                  warps_n,
                  warp_tile_m_count,
                  warp_tile_n_count,
                  thread_tile_m,
                  thread_tile_n,
                  threads_n,
                  block_k,
                  single_buffer,
                  swizzle,
                  is_aligned>(C, A, B, M, N, K);
    }
};

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_KERNEL_HPP
