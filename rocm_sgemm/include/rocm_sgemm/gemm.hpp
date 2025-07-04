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

#ifndef ROCM_SGEMM_GEMM_HPP
#define ROCM_SGEMM_GEMM_HPP

#include "kernel/kernel.hpp"

namespace rocm_sgemm
{

/**
 * @brief GPU matrix multiplication: C = A * B
 *
 * Launches a tiled GEMM kernel that computes C = A * B using hierarchical
 * blocking and shared memory optimization.
 *
 * @param C      Output matrix (M x N), row-major order
 * @param A      Input matrix A (M x K), row-major order
 * @param B      Input matrix B (K x N), row-major order
 * @param M      Number of rows in A and C
 * @param N      Number of columns in B and C
 * @param K      Number of columns in A and rows in B
 * @param stream HIP stream for kernel execution
 *
 * @pre All matrices must be allocated in GPU memory
 * @pre Matrix dimensions must be positive
 * @post C contains the result of A * B (previous contents overwritten)
 */
template<m_layout layout_C, m_layout layout_A, m_layout layout_B>
__host__ void gemm(float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

extern template __host__ void gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>(
    float* C, float* A, float* B, size_t M, size_t N, size_t K, hipStream_t& stream);

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_GEMM_HPP
