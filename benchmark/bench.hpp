
#include <benchmark/benchmark.h>
#include <common/hip_utils.hpp>
#include <common/matrix.hpp>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief Initialize matrix with random values
 * @tparam T Matrix element type
 * @tparam L Matrix layout
 * @param input Matrix to initialize
 */
template<class T, m_layout L>
void init_matrix(matrix<T, L>& input)
{
    float tmp[5] = {1.025f, 1.05f, 1.075f, 1.1f, 1.0f};

    int l = 0;
    for(size_t m = 0; m < input.m(); ++m)
    {
        for(size_t n = 0; n < input.n(); ++n)
        {
            input(m, n) = static_cast<T>(tmp[l % 5]);
            l++;
        }
    }
}

class CustomReporter : public benchmark::ConsoleReporter
{
public:
    explicit CustomReporter() : benchmark::ConsoleReporter(OO_ColorTabular) {}
};
