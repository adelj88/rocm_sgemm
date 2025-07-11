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
 * @brief GEMM kernel with hierarchical tiling: block -> warp -> thread levels
 *
 * Computes C = A * B using three-level tiling with double-buffered shared memory.
 * Uses configurable warp tile counts and thread tile sizes.
 *
 * Algorithm overview:
 * - Each thread block processes a block_m x block_n tile of the output matrix
 * - Within each block, warps process smaller warp-level tiles
 * - Each warp tile is divided into configurable sub-tiles (warp_tile_m_count x warp_tile_n_count)
 * - Individual threads process thread_tile_m x thread_tile_n elements
 * - Double buffering overlaps computation with memory transfers
 * - Vectorized loads maximize memory bandwidth utilization
 *
 * @tparam block_size         Total threads per block (must be multiple of warp_size)
 * @tparam block_m            Block tile size in M dimension
 * @tparam block_n            Block tile size in N dimension
 * @tparam block_k            K-dimension tile size for shared memory blocking
 * @tparam warp_tile_m_count  Number of sub-tiles per warp in M dimension
 * @tparam warp_tile_n_count  Number of sub-tiles per warp in N dimension
 * @tparam thread_tile_m      Elements per thread in M dimension
 * @tparam thread_tile_n      Elements per thread in N dimension
 *
 * @param C  Output matrix (M x N), row-major
 * @param A  Input matrix A (M x K), row-major
 * @param B  Input matrix B (K x N), row-major
 * @param M  Number of rows in A and C
 * @param N  Number of columns in B and C
 * @param K  Number of columns in A and rows in B
 */
template<class T,
         m_layout LAYOUT_C,
         m_layout LAYOUT_A,
         m_layout LAYOUT_B,
         int      block_size,
         int      block_m,
         int      block_n,
         int      block_k,
         int      warp_tile_m_count,
         int      warp_tile_n_count,
         int      thread_tile_m,
         int      thread_tile_n,
         int      threads_n,
         int      is_aligned>
__global__ __launch_bounds__(block_size) void kernel_gemm(
    T* C, const T* A, const T* B, int M, int N, int K)
{
    // Thread arrangement within warp
    constexpr int threads_m = warp_size / threads_n; // 4

    // Sub-tile dimensions (elements per warp sub-tile)
    constexpr int sub_tile_m = thread_tile_m * threads_m;
    constexpr int sub_tile_n = thread_tile_n * threads_n;

    // Warp tile dimensions
    constexpr int warp_m = warp_tile_m_count * sub_tile_m;
    constexpr int warp_n = warp_tile_n_count * sub_tile_n;

    // Warp arrangement
    constexpr int warps_m   = block_m / warp_m;
    constexpr int warps_n   = block_n / warp_n;
    constexpr int num_warps = warps_m * warps_n;

    static_assert(block_m % warp_m == 0);
    static_assert(block_n % warp_n == 0);
    static_assert(num_warps == block_size / warp_size);

    constexpr int lds_size = (block_m * block_k) + (block_k * block_n);

    // Block coordinates
    const int grid_m  = (M + block_m - 1) / block_m;
    const int grid_n  = (N + block_n - 1) / block_n;
    const int tile_id = blockIdx.x;

    constexpr bool use_hilbert
        = (LAYOUT_A == m_layout::row_major && LAYOUT_B == m_layout::col_major)
          || (LAYOUT_A == m_layout::col_major && LAYOUT_B == m_layout::col_major);

    int block_row, block_col;
    if constexpr(use_hilbert)
    {
        hilbert_tile_mapping<block_m, block_n>(tile_id, grid_m, grid_n, &block_row, &block_col);
    }
    else
    {
        snake_tile_mapping<block_m, block_n>(tile_id, grid_m, grid_n, &block_row, &block_col);
    }

    // Shared memory allocation
    __shared__ T lds_mem[2 * lds_size];

    // Double buffer partitioning
    T* a_tiles_0 = lds_mem;
    T* a_tiles_1 = lds_mem + lds_size;
    T* b_tiles_0 = lds_mem + (block_m * block_k);
    T* b_tiles_1 = lds_mem + lds_size + (block_m * block_k);

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
    constexpr int half_block = block_size / 2;
    const int     cid        = tid % half_block;

    const T* A_base     = A + block_row * ((LAYOUT_A == m_layout::col_major) ? 1 : K);
    const T* B_base     = B + block_col * ((LAYOUT_B == m_layout::col_major) ? K : 1);
    const T* A_tile_ptr = A_base;
    const T* B_tile_ptr = B_base;

    // Initial tile loading
    if(tid < half_block)
    {
        load_to_shared<m_input::matrix_a, LAYOUT_A, half_block, block_m, block_k>(a_tiles_0,
                                                                                  A_tile_ptr,
                                                                                  M,
                                                                                  K,
                                                                                  cid);
    }
    else
    {
        load_to_shared<m_input::matrix_b, LAYOUT_B, half_block, block_k, block_n>(b_tiles_0,
                                                                                  B_tile_ptr,
                                                                                  K,
                                                                                  N,
                                                                                  cid);
    }

    __syncthreads();

    // Double buffer pointers
    T* current_a = a_tiles_0;
    T* current_b = b_tiles_0;
    T* next_a    = a_tiles_1;
    T* next_b    = b_tiles_1;

    const int global_mult_A = (LAYOUT_A == m_layout::col_major) ? M : 1;
    const int global_mult_B = (LAYOUT_B == m_layout::col_major) ? 1 : N;

    // Main computation loop
    for(int k_tile = 0; k_tile < K; k_tile += block_k)
    {
        // Prefetch next tiles
        if(k_tile + block_k < K && tid >= half_block)
        {
            const T* next_A = A_tile_ptr + block_k * global_mult_A;
            load_to_shared<m_input::matrix_a, LAYOUT_A, half_block, block_m, block_k>(next_a,
                                                                                      next_A,
                                                                                      M,
                                                                                      K,
                                                                                      cid);
        }

        // Compute on current tile
        for(int k_offset = 0; k_offset < block_k; ++k_offset)
        {
            const int thread_a_base = warp_base_row + thread_row_in_warp * thread_tile_m;
            const int thread_b_base = warp_base_col + thread_col_in_warp * thread_tile_n;

            const T* a_ptr = current_a + k_offset * block_m + thread_a_base;
            const T* b_ptr = current_b + k_offset * block_n + thread_b_base;

            if constexpr(warp_tile_m_count < warp_tile_n_count)
            {
                for(int wn = 0; wn < warp_tile_n_count; ++wn)
                {
                    if(wn < warp_tile_m_count)
                    {
                        load_matrix(a_frag[wn], a_ptr, block_m, block_k);
                        a_ptr += sub_tile_m;
                    }
                    load_matrix(b_frag[wn], b_ptr, block_k, block_n);
                    b_ptr += sub_tile_n;
                }
            }
            else if constexpr(warp_tile_m_count > warp_tile_n_count)
            {
                for(int wm = 0; wm < warp_tile_m_count; ++wm)
                {
                    load_matrix(a_frag[wm], a_ptr, block_m, block_k);
                    if(wm < warp_tile_n_count)
                    {
                        load_matrix(b_frag[wm], b_ptr, block_k, block_n);
                        b_ptr += sub_tile_n;
                    }
                    a_ptr += sub_tile_m;
                }
            }
            else if constexpr(warp_tile_m_count == warp_tile_n_count)
            {
                for(int wm = 0; wm < warp_tile_m_count; ++wm)
                {
                    load_matrix(a_frag[wm], a_ptr, block_m, block_k);
                    load_matrix(b_frag[wm], b_ptr, block_k, block_n);
                    a_ptr += sub_tile_m;
                    b_ptr += sub_tile_n;
                }
            }

            // Compute outer products
            for(int wm = 0; wm < warp_tile_m_count; ++wm)
            {
                for(int wn = 0; wn < warp_tile_n_count; ++wn)
                {
                    auto& dest_ptr = c_frag[wm][wn].get();
                    for(int tm = 0; tm < thread_tile_m; ++tm)
                    {
                        T         a_val  = a_frag[wm][tm];
                        const int offset = tm * thread_tile_n;

                        for(int tn = 0; tn < thread_tile_n; ++tn)
                        {
                            dest_ptr[offset + tn] += a_val * b_frag[wn][tn];
                        }
                    }
                }
            }
        }

        if(k_tile + block_k < K && tid < half_block)
        {
            const T* next_B = B_tile_ptr + block_k * global_mult_B;
            load_to_shared<m_input::matrix_b, LAYOUT_B, half_block, block_k, block_n>(next_b,
                                                                                      next_B,
                                                                                      K,
                                                                                      N,
                                                                                      cid);
        }

        // Advance pointers and swap buffers
        A_tile_ptr += block_k * global_mult_A;
        B_tile_ptr += block_k * global_mult_B;

        T* temp_a = current_a;
        T* temp_b = current_b;
        current_a = next_a;
        current_b = next_b;
        next_a    = temp_a;
        next_b    = temp_b;

        __syncthreads();
    }

    const int row_offset = block_row + warp_base_row + thread_row_in_warp * thread_tile_m;
    const int col_offset = block_col + warp_base_col + thread_col_in_warp * thread_tile_n;

    if constexpr(is_aligned)
    {
        // Fast path: matrix dimensions are perfectly aligned, no bounds checking needed
        for(int wm = 0; wm < warp_tile_m_count; ++wm)
        {
            const int tile_row_base = wm * sub_tile_m;
            for(int wn = 0; wn < warp_tile_n_count; ++wn)
            {
                const int tile_col_base = wn * sub_tile_n;
                for(int tm = 0; tm < thread_tile_m; ++tm)
                {
                    const int offset     = tm * thread_tile_n;
                    const int global_row = row_offset + tile_row_base + tm;
                    for(int tn = 0; tn < thread_tile_n; ++tn)
                    {
                        const int global_col = col_offset + tile_col_base + tn;
                        if constexpr(LAYOUT_C == m_layout::col_major)
                        {
                            C[global_col * M + global_row] = c_frag[wm][wn][offset + tn];
                        }
                        else
                        {
                            C[global_row * N + global_col] = c_frag[wm][wn][offset + tn];
                        }
                    }
                }
            }
        }
    }
    else
    {
        // Standard path: use bounds checking
        for(int wm = 0; wm < warp_tile_m_count; ++wm)
        {
            const int tile_row_base = wm * sub_tile_m;
            for(int wn = 0; wn < warp_tile_n_count; ++wn)
            {
                const int tile_col_base = wn * sub_tile_n;
                for(int tm = 0; tm < thread_tile_m; ++tm)
                {
                    const int offset     = tm * thread_tile_n;
                    const int global_row = row_offset + tile_row_base + tm;
                    for(int tn = 0; tn < thread_tile_n; ++tn)
                    {
                        const int global_col = col_offset + tile_col_base + tn;
                        if(global_row < M && global_col < N)
                        {
                            if constexpr(LAYOUT_C == m_layout::col_major)
                            {
                                C[global_col * M + global_row] = c_frag[wm][wn][offset + tn];
                            }
                            else
                            {
                                C[global_row * N + global_col] = c_frag[wm][wn][offset + tn];
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_KERNEL_HPP
