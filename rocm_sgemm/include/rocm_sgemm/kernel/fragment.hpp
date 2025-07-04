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

template<class T>
struct type_selector
{
    using type = T;
};

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

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_a && ACCESS == m_layout::row_major),
                            void>::type
{
    const T* tmp = reinterpret_cast<const T*>(data);
    for(auto it = frag.begin(); it != frag.end(); ++it)
    {
        *it = *tmp;
        tmp += N;
    }
}

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_a && ACCESS == m_layout::col_major),
                            void>::type
{
    auto& tmp = frag.get();
    for(int i = 0; i < TILE; ++i)
    {
        tmp[i] = data[i];
    }
}

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_b && ACCESS == m_layout::row_major),
                            void>::type
{
    auto& tmp = frag.get();
    for(int i = 0; i < TILE; ++i)
    {
        tmp[i] = data[i];
    }
}

template<m_input MATRIX, m_layout ACCESS, class T, int TILE>
__device__ __forceinline__ auto load_matrix(fragment<T, TILE>& frag, const T* data, int M, int N) ->
    typename std::enable_if<(MATRIX == m_input::matrix_b && ACCESS == m_layout::col_major),
                            void>::type
{
    const T* tmp = reinterpret_cast<const T*>(data);
    for(auto it = frag.begin(); it != frag.end(); ++it)
    {
        *it = *tmp;
        tmp += M;
    }
}

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_FRAGMENT_HPP
