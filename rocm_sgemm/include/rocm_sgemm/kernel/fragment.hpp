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

#ifndef ROCM_SGEMM_FRAGMENT_HPP
#define ROCM_SGEMM_FRAGMENT_HPP

namespace rocm_sgemm
{

/**
 * @brief Type selector struct for mapping high-level types to internal storage types.
 *
 * The default mapping is the identity (T -> T); specializations may remap a public
 * type to a narrower storage representation without changing fragment semantics.
 *
 * @tparam T The high-level data type (e.g., float).
 */
template<class T>
struct type_selector
{
    using type = T; ///< The mapped internal type.
};

/**
 * @brief Represents a register-resident fragment for holding a tile of matrix data.
 *
 * Elements are stored in a single packed vector (ext_vector_type) so whole fragments
 * move between registers and LDS with wide instructions, while the iterator/proxy pair
 * provides element-wise access for scalar producers and consumers.
 *
 * @tparam T The high-level data type of the matrix elements.
 * @tparam TILE The number of elements held by the fragment.
 */
template<class T, int TILE>
class fragment
{
public:
    using underlying_type = T;
    using type            = typename type_selector<T>::type;
    using frag_vec        = type __attribute__((ext_vector_type(TILE)));
    using value_type      = type;

private:
    frag_vec _fragment = {};

public:
    /**
     * @brief Proxy class for accessing elements of the fragment.
     */
    class proxy
    {
        frag_vec& vec_ref;
        int       index;

        friend class iterator;

    public:
        __device__ __forceinline__ proxy(frag_vec& v, int i) : vec_ref(v), index(i) {}

        template<typename U = type>
        __device__ __forceinline__ auto operator=(type value) ->
            typename std::enable_if<!std::is_same<U, T>::value, proxy&>::type
        {
            vec_ref[index] = value;
            return *this;
        }

        // This operator handles the T type and also serves as fallback when type == T
        __device__ __forceinline__ proxy& operator=(const T& value)
        {
            vec_ref[index] = static_cast<type>(value);
            return *this;
        }

        __device__ __forceinline__ operator type() const
        {
            return vec_ref[index];
        }

        proxy*       operator&()       = delete;
        const proxy* operator&() const = delete;
        proxy(const proxy&)            = delete;
    };

    /**
     * @brief Iterator class for traversing elements of the fragment.
     */
    class iterator
    {
        frag_vec& vec_ref;
        int       current_index;

        friend class fragment<T, TILE>;

        __device__ __forceinline__ iterator(frag_vec& v, int i) : vec_ref(v), current_index(i) {}

    public:
        __device__ __forceinline__ proxy operator*() const
        {
            return proxy(vec_ref, current_index);
        }

        __device__ __forceinline__ iterator& operator++()
        {
            ++current_index;
            return *this;
        }

        __device__ __forceinline__ iterator& operator+=(int n)
        {
            current_index += n;
            return *this;
        }

        __device__ __forceinline__ iterator operator+(int n) const
        {
            iterator temp = *this;
            temp += n;
            return temp;
        }

        __device__ __forceinline__ bool operator!=(const iterator& other) const
        {
            return current_index != other.current_index;
        }
    };

public:
    __device__ __forceinline__ iterator begin()
    {
        return iterator(_fragment, 0);
    }

    __device__ __forceinline__ iterator end()
    {
        return iterator(_fragment, TILE);
    }

    __device__ __forceinline__ frag_vec& get()
    {
        return _fragment;
    }

    __device__ __forceinline__ const frag_vec& get() const
    {
        return _fragment;
    }

    __device__ __forceinline__ type operator[](int i) const
    {
        return _fragment[i];
    }
};

/**
 * @brief Loads a matrix tile into a fragment with a single vectorized copy.
 *
 * The TILE elements starting at `data` must be contiguous in memory (shared or global),
 * so the entire fragment is fetched as one wide load. This matches the native access
 * pattern: rows of Matrix A and columns of Matrix B are the contiguous axes.
 *
 * @tparam T The element data type.
 * @tparam TILE The dimension of the tile.
 *
 * @param frag The destination fragment to populate.
 * @param data Pointer to the starting position in shared or global memory.
 * @param M Unused (kept for interface compatibility with strided load variants).
 * @param N Unused (kept for interface compatibility with strided load variants).
 */
template<class T, int TILE>
__device__ __forceinline__ void load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N)
{
    using type        = typename type_selector<T>::type;
    using vector_type = type __attribute__((ext_vector_type(TILE)));

    const vector_type* src_ptr  = reinterpret_cast<const vector_type*>(data);
    vector_type*       dest_ptr = reinterpret_cast<vector_type*>(&frag.get());

    *dest_ptr = *src_ptr;
}

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_FRAGMENT_HPP
