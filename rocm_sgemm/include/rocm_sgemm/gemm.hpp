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
 * @brief GPU matrix multiplication: C = A * B.
 *
 * Launches a tiled GEMM kernel that computes C = A * B using hierarchical
 * blocking and shared memory optimization.
 *
 * @tparam layout_C The memory layout of the output matrix C (row_major or col_major).
 * @tparam layout_A The memory layout of the input matrix A (row_major or col_major).
 * @tparam layout_B The memory layout of the input matrix B (row_major or col_major).
 * @tparam T The data type of the matrices.
 *
 * @param C Pointer to the output matrix C in device memory.
 * @param A Pointer to the input matrix A in device memory.
 * @param B Pointer to the input matrix B in device memory.
 * @param M Number of rows of matrices C and A.
 * @param N Number of columns of matrices C and B.
 * @param K Number of columns of matrix A and rows of matrix B.
 * @param stream The HIP stream to execute the kernel on.
 *
 * @pre All matrices must be allocated in GPU memory.
 * @pre Matrix dimensions must be positive.
 * @post C contains the result of A * B (previous contents overwritten).
 */
template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T>
__host__ void gemm(T* C, T* A, T* B, size_t M, size_t N, size_t K, hipStream_t& stream);

// Macro to declare all layout combinations for a type
#define DECLARE_GEMM_FOR_TYPE(T)                                                              \
    extern template void gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                                   \
        T*,                                                                                   \
        T*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                                   \
        T*,                                                                                   \
        T*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                                   \
        T*,                                                                                   \
        T*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::row_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                                   \
        T*,                                                                                   \
        T*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::row_major, m_layout::row_major>( \
        T*,                                                                                   \
        T*,                                                                                   \
        T*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::row_major, m_layout::col_major>( \
        T*,                                                                                   \
        T*,                                                                                   \
        T*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::col_major, m_layout::row_major>( \
        T*,                                                                                   \
        T*,                                                                                   \
        T*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);                                                                        \
    extern template void gemm<m_layout::col_major, m_layout::col_major, m_layout::col_major>( \
        T*,                                                                                   \
        T*,                                                                                   \
        T*,                                                                                   \
        size_t,                                                                               \
        size_t,                                                                               \
        size_t,                                                                               \
        hipStream_t&);

DECLARE_GEMM_FOR_TYPE(float)

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_GEMM_HPP
