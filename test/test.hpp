#include <common/hip_utils.hpp>
#include <common/matrix.hpp>
#include <gtest/gtest.h>

/**
 * @brief Initialize matrix with random values
 * @tparam T Matrix element type
 * @tparam L Matrix layout
 * @param input Matrix to initialize
 */
template<class T, m_layout L>
void init_matrix(matrix<T, L>& input)
{
    std::random_device                    rd;
    std::mt19937                          gen(1);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for(size_t m = 0; m < input.m(); ++m)
    {
        for(size_t n = 0; n < input.n(); ++n)
        {
            input(m, n) = static_cast<T>(dis(gen));
        }
    }
}

/**
 * @brief CPU reference implementation for float GEMM
 */
template<m_layout L1, m_layout L2, m_layout L3>
void sgemm_cpu(matrix<float, L1>& C, const matrix<float, L2>& A, const matrix<float, L3>& B)
{
    for(size_t i = 0; i < C.m(); ++i)
    {
        for(size_t j = 0; j < C.n(); ++j)
        {
            float acc = 0.0f;
            for(size_t k = 0; k < A.n(); ++k)
            {
                acc += A(i, k) * B(k, j);
            }
            C(i, j) = acc;
        }
    }
}
