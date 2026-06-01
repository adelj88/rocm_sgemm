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

#ifndef ROCM_SGEMM_COMMON_HPP
#define ROCM_SGEMM_COMMON_HPP

#include <hip/hip_runtime.h>

namespace rocm_sgemm
{

/**
 * @brief Enum class defining matrix memory layout options.
 */
enum class m_layout
{
    row_major, ///< Row-major layout (elements consecutive in memory by row)
    col_major ///< Column-major layout (elements consecutive in memory by column)
};

/**
 * @brief Enum to specify which input matrix is being accessed.
 */
enum class m_input
{
    matrix_a, ///< Refers to input matrix A
    matrix_b ///< Refers to input matrix B
};

/** @brief The number of threads in a warp for AMD GPUs (Wave32). */
constexpr int warp_size = 32;

/**
 * @brief LDS inner-dimension padding (in floats) for the transposed staging tiles.
 *
 * A tile is stored transposed in LDS as [block_k][block_m] (A) or [block_k][block_n] (B),
 * so consecutive K-rows sit block_m/block_n apart. When that stride is a multiple of the
 * 32 LDS banks, the per-K commit stores collide; padding the row stride staggers them.
 * The pad must be a multiple of the widest single LDS access (ds_read_b128 = 4 floats);
 * larger thread tiles read as multiple b128s, each of which stays 16-byte aligned at pad=4.
 */
constexpr int lds_pad = 4;

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_COMMON_HPP
