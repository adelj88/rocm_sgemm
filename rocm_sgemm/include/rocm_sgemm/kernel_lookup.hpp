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

#ifndef ROCM_SGEMM_KERNEL_LOOKUP_HPP
#define ROCM_SGEMM_KERNEL_LOOKUP_HPP

#include <cstddef>

namespace rocm_sgemm
{

// Lookup function: returns kernel function pointer from static 3D table
// Parameters:
//   config_idx: Configuration index (from find_best_config)
//   layout_idx: Layout combination index (0-7 for ABC combinations)
//               0=rrr, 1=rrc, 2=rcr, 3=rcc, 4=crr, 5=crc, 6=ccr, 7=ccc
//   alignment_idx: Alignment index (0=unaligned, 1=aligned)
void* lookup_kernel(size_t config_idx, size_t layout_idx, size_t alignment_idx);

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_KERNEL_LOOKUP_HPP
