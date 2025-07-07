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

#ifndef ROCM_SGEMM_LOAD_HPP
#define ROCM_SGEMM_LOAD_HPP

namespace rocm_sgemm
{

// Matrix A: Always store in [K][M] order in shared memory
template<m_input MATRIX, m_layout ACCESS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<(MATRIX == m_input::matrix_a && ACCESS == m_layout::col_major),
                            void>::type
{
    // A col_major global -> [K][M] shared (no transpose needed)
    constexpr int max_load_width    = 8;
    constexpr int min_block_dim     = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes   = min_block_dim * sizeof(T);
    constexpr int actual_load_width = (min_block_bytes >= 32)   ? max_load_width
                                      : (min_block_bytes >= 16) ? 4
                                      : (min_block_bytes >= 8)  ? 2
                                                                : 1;

    using vector_type          = float __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));
    constexpr int vectors_per_thread
        = (((BLOCK_M * BLOCK_N) / vector_width) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < vectors_per_thread; ++i)
    {
        const int vec_idx = (tid * vector_width) + (i * BLOCK_SIZE * vector_width);

        if(vec_idx < (BLOCK_M * BLOCK_N))
        {
            // Target: [K][M] layout in shared memory
            const int k = vec_idx / BLOCK_M;
            const int m = vec_idx % BLOCK_M;

            // Source: col_major global [K][M] -> gload = k * M + m
            const int gload = k * M + m;

            // Store at vec_idx position (which is already [K][M] order)
            *reinterpret_cast<vector_type*>(output + vec_idx)
                = *reinterpret_cast<const vector_type*>(input + gload);
        }
    }
}

template<m_input MATRIX, m_layout ACCESS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<(MATRIX == m_input::matrix_a && ACCESS == m_layout::row_major),
                            void>::type
{
    // A row_major global -> [K][M] shared (transpose during load)
    constexpr int elements_per_thread = ((BLOCK_M * BLOCK_N) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < elements_per_thread; ++i)
    {
        const int linear_idx = tid + (i * BLOCK_SIZE);

        if(linear_idx < (BLOCK_M * BLOCK_N))
        {
            // Target shared memory: [K][M] layout
            // linear_idx represents position in [K][M]: k * BLOCK_M + m
            const int k = linear_idx / BLOCK_M;
            const int m = linear_idx % BLOCK_M;

            // Source global memory: row_major [M][K] layout
            // We want global element at position [m][k]
            const int gload = m * N + k;

            output[linear_idx] = input[gload];
        }
    }
}

// Matrix B: Always store in [K][N] order in shared memory
template<m_input MATRIX, m_layout ACCESS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<(MATRIX == m_input::matrix_b && ACCESS == m_layout::row_major),
                            void>::type
{
    // B row_major global -> [K][N] shared (no transpose needed)
    constexpr int max_load_width    = 8;
    constexpr int min_block_dim     = (BLOCK_M < BLOCK_N) ? BLOCK_M : BLOCK_N;
    constexpr int min_block_bytes   = min_block_dim * sizeof(T);
    constexpr int actual_load_width = (min_block_bytes >= 32)   ? max_load_width
                                      : (min_block_bytes >= 16) ? 4
                                      : (min_block_bytes >= 8)  ? 2
                                                                : 1;

    using vector_type          = float __attribute__((ext_vector_type(actual_load_width)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));
    constexpr int vectors_per_thread
        = (((BLOCK_M * BLOCK_N) / vector_width) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < vectors_per_thread; ++i)
    {
        const int vec_idx = (tid * vector_width) + (i * BLOCK_SIZE * vector_width);

        if(vec_idx < (BLOCK_M * BLOCK_N))
        {
            // Target: [K][N] layout in shared memory
            const int k = vec_idx / BLOCK_N;
            const int n = vec_idx % BLOCK_N;

            // Source: row_major global [K][N] -> gload = k * N + n
            const int gload = k * N + n;

            // Store at vec_idx position ([K][N] order)
            *reinterpret_cast<vector_type*>(output + vec_idx)
                = *reinterpret_cast<const vector_type*>(input + gload);
        }
    }
}

template<m_input MATRIX, m_layout ACCESS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<(MATRIX == m_input::matrix_b && ACCESS == m_layout::col_major),
                            void>::type
{
    // B col_major global -> [K][N] shared (transpose during load)
    constexpr int elements_per_thread = ((BLOCK_M * BLOCK_N) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < elements_per_thread; ++i)
    {
        const int linear_idx = tid + (i * BLOCK_SIZE);

        if(linear_idx < (BLOCK_M * BLOCK_N))
        {
            // Target shared memory: [K][N] layout
            // linear_idx represents position in [K][N]: k * BLOCK_N + n
            const int k = linear_idx / BLOCK_N;
            const int n = linear_idx % BLOCK_N;

            // Source global memory: col_major [N][K] layout
            // We want global element at position [n][k]
            const int gload = n * M + k;

            output[linear_idx] = input[gload];
        }
    }
}

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_LOAD_HPP
