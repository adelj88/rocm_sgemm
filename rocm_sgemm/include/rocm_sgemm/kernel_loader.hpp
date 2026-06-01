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

#ifndef ROCM_SGEMM_KERNEL_LOADER_HPP
#define ROCM_SGEMM_KERNEL_LOADER_HPP

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <rocm_sgemm/kernel/common.hpp>
#include <stdexcept>
#include <string>

namespace rocm_sgemm
{

/**
 * @brief Runtime arch-library loader.
 *
 * Detects the current GPU architecture, opens the matching per-arch shared
 * library (librocm_sgemm_<arch>.so), and dispatches through its C ABI entry
 * point. A single application binary supports multiple GPU architectures without
 * recompiling, as long as the appropriate arch library is present at runtime.
 *
 * Usage:
 *   rocm_sgemm::loader loader;
 *   loader.gemm<m_layout::row_major, m_layout::row_major, m_layout::row_major>(
 *       C, A, B, M, N, K, stream);
 */
class loader
{
    using dispatch_fn = void (*)(void*, void*, void*, size_t, size_t, size_t, int, int, int, void*);

    void*       handle_  = nullptr;
    dispatch_fn fn_f32_  = nullptr;

    static dispatch_fn resolve(void* handle, const char* name)
    {
        auto fn = reinterpret_cast<dispatch_fn>(dlsym(handle, name));
        if(!fn)
        {
            throw std::runtime_error(std::string("rocm_sgemm::loader: symbol not found: ") + name
                                     + " (" + dlerror() + ")");
        }
        return fn;
    }

public:
    loader()
    {
        hipDeviceProp_t props{};
        if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        {
            throw std::runtime_error("rocm_sgemm::loader: hipGetDeviceProperties failed");
        }

        // Strip feature flags (e.g. "gfx1100:xnack-" -> "gfx1100").
        std::string arch  = props.gcnArchName;
        const auto  colon = arch.find(':');
        if(colon != std::string::npos)
        {
            arch = arch.substr(0, colon);
        }

        const std::string libname = "librocm_sgemm_" + arch + ".so";

#ifdef ROCM_SGEMM_LIB_DIR
        handle_ = dlopen((std::string(ROCM_SGEMM_LIB_DIR "/") + libname).c_str(),
                         RTLD_NOW | RTLD_LOCAL);
#endif
        if(!handle_)
        {
            handle_ = dlopen(libname.c_str(), RTLD_NOW | RTLD_LOCAL);
        }

        if(!handle_)
        {
            throw std::runtime_error("rocm_sgemm::loader: could not open " + libname + ": "
                                     + dlerror());
        }

        fn_f32_ = resolve(handle_, "rocm_sgemm_f32");
    }

    ~loader()
    {
        if(handle_)
        {
            dlclose(handle_);
        }
    }

    loader(const loader&)            = delete;
    loader& operator=(const loader&) = delete;
    loader(loader&&)                 = delete;
    loader& operator=(loader&&)      = delete;

    /**
     * @brief Execute a GEMM via the arch-specific library.
     */
    template<m_layout layout_C, m_layout layout_A, m_layout layout_B, class T>
    void gemm(T* C, T* A, T* B, size_t M, size_t N, size_t K, hipStream_t& stream)
    {
        static_assert(std::is_same_v<T, float>, "rocm_sgemm::loader supports float only");
        fn_f32_(C,
                A,
                B,
                M,
                N,
                K,
                layout_C == m_layout::col_major ? 1 : 0,
                layout_A == m_layout::col_major ? 1 : 0,
                layout_B == m_layout::col_major ? 1 : 0,
                &stream);
    }
};

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_KERNEL_LOADER_HPP
