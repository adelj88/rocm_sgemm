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

template<m_layout ACCESS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::col_major, void>::type
{
    using vector_type          = float __attribute__((ext_vector_type(8)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));
    constexpr int vectors_per_thread
        = (((BLOCK_M * BLOCK_N) / vector_width) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < vectors_per_thread; ++i)
    {
        const int idx = (tid * vector_width) + (i * BLOCK_SIZE * vector_width);

        if(idx < (BLOCK_M * BLOCK_N))
        {
            const int col   = idx / BLOCK_M;
            const int row   = idx % BLOCK_M;
            const int gload = col * M + row;

            *reinterpret_cast<vector_type*>(output + idx)
                = *reinterpret_cast<const vector_type*>(input + gload);
        }
    }
}

template<m_layout ACCESS, int BLOCK_SIZE, int BLOCK_M, int BLOCK_N, class T>
__device__ __forceinline__ auto load_to_shared(T* output, const T* input, int M, int N, int tid) ->
    typename std::enable_if<ACCESS == m_layout::row_major, void>::type
{
    using vector_type          = float __attribute__((ext_vector_type(8)));
    constexpr int vector_width = (sizeof(vector_type) / sizeof(T));
    constexpr int vectors_per_thread
        = (((BLOCK_M * BLOCK_N) / vector_width) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < vectors_per_thread; ++i)
    {
        const int idx = (tid * vector_width) + (i * BLOCK_SIZE * vector_width);

        if(idx < (BLOCK_M * BLOCK_N))
        {
            const int row   = idx / BLOCK_N;
            const int col   = idx % BLOCK_N;
            const int gload = row * N + col;

            *reinterpret_cast<vector_type*>(output + idx)
                = *reinterpret_cast<const vector_type*>(input + gload);
        }
    }
}

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_LOAD_HPP
