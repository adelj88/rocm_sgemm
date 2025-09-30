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

#ifndef ROCM_SGEMM_KERNEL_LAUNCHER_HPP
#define ROCM_SGEMM_KERNEL_LAUNCHER_HPP

#include <rocm_sgemm/kernel/config_generated.hpp>
#include <rocm_sgemm/kernel_lookup.hpp>
#include <stdexcept>

namespace rocm_sgemm
{

template<class T, m_layout layout_C, m_layout layout_A, m_layout layout_B>
struct kernel_launcher
{
    using kernel_func_ptr = void (*)(T*, const T*, const T*, int, int, int);

    static void launch(size_t       config_idx,
                       T*           C,
                       const T*     A,
                       const T*     B,
                       size_t       M,
                       size_t       N,
                       size_t       K,
                       int          block_m,
                       int          block_n,
                       dim3         grid_dim,
                       dim3         block_dim,
                       hipStream_t& stream)
    {
        // Compute layout index (0-7) based on (A,B,C) layout combination
        constexpr size_t layout_idx
            = (layout_A == m_layout::row_major && layout_B == m_layout::row_major
               && layout_C == m_layout::row_major)
                  ? 0 // rrr
              : (layout_A == m_layout::row_major && layout_B == m_layout::row_major
                 && layout_C == m_layout::col_major)
                  ? 1 // rrc
              : (layout_A == m_layout::row_major && layout_B == m_layout::col_major
                 && layout_C == m_layout::row_major)
                  ? 2 // rcr
              : (layout_A == m_layout::row_major && layout_B == m_layout::col_major
                 && layout_C == m_layout::col_major)
                  ? 3 // rcc
              : (layout_A == m_layout::col_major && layout_B == m_layout::row_major
                 && layout_C == m_layout::row_major)
                  ? 4 // crr
              : (layout_A == m_layout::col_major && layout_B == m_layout::row_major
                 && layout_C == m_layout::col_major)
                  ? 5 // crc
              : (layout_A == m_layout::col_major && layout_B == m_layout::col_major
                 && layout_C == m_layout::row_major)
                  ? 6 // ccr
              : (layout_A == m_layout::col_major && layout_B == m_layout::col_major
                 && layout_C == m_layout::col_major)
                  ? 7 // ccc
                  : 0;

        // Runtime alignment check (like the old code)
        bool   is_aligned    = (M % block_m == 0 && N % block_n == 0);
        size_t alignment_idx = is_aligned ? 1 : 0;

        // Lookup kernel function pointer from static table
        void* kernel_ptr = lookup_kernel(config_idx, layout_idx, alignment_idx);

        // Check for null pointer (kernel not available for this config/layout/alignment combination)
        if(kernel_ptr == nullptr)
        {
            // This should not happen if config generation is working correctly
            throw std::runtime_error(
                "No kernel available for the requested configuration, layout, and alignment");
        }

        // Cast to correct function pointer type and launch
        auto kernel_func = reinterpret_cast<kernel_func_ptr>(kernel_ptr);
        kernel_func<<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
    }
};

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_KERNEL_LAUNCHER_HPP
