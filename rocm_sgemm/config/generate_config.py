#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from collections import defaultdict

def generate_config_header(config_file, output_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract unique configurations
    unique_configs = set()
    for conf in config['configurations']:
        cfg = conf['config']
        unique_configs.add((cfg['block_size'], cfg['block_m'], cfg['block_n'], cfg['block_k'],
                           cfg['warp_tile_m_count'], cfg['warp_tile_n_count'],
                           cfg['thread_tile_m'], cfg['thread_tile_n'], cfg['threads_n']))

    # Always add default config if not present
    default_config = (128, 128, 128, 8, 4, 4, 2, 4, 8)
    if default_config not in unique_configs:
        unique_configs.add(default_config)

    # Sort configs for stable output and indexing
    unique_configs = sorted(list(unique_configs))
    num_configs = len(unique_configs)

    # Group configurations by matrix dimensions and (A,B,C) layout
    size_abc_configs = {}
    for conf in config['configurations']:
        range_info = conf['range']
        M, N, K = range_info['M'], range_info['N'], range_info['K']

        # Create key for this matrix size
        size_key = (M, N, K)

        # Create layout key from (A,B,C) layout
        layout_dict = conf['layout']
        abc_layout_key = (
            layout_dict.get('A', 'any'),
            layout_dict.get('B', 'any'),
            layout_dict.get('C', 'any')
        )

        if size_key not in size_abc_configs:
            size_abc_configs[size_key] = {}

        cfg = conf['config']
        config_tuple = (cfg['block_size'], cfg['block_m'], cfg['block_n'], cfg['block_k'],
                       cfg['warp_tile_m_count'], cfg['warp_tile_n_count'],
                       cfg['thread_tile_m'], cfg['thread_tile_n'], cfg['threads_n'])
        config_idx = unique_configs.index(config_tuple)

        size_abc_configs[size_key][abc_layout_key] = config_idx

    # Create sorted configuration list for binary search
    sorted_configs = []
    for size_key, abc_layouts in size_abc_configs.items():
        M, N, K = size_key
        for abc_layout_key, config_idx in abc_layouts.items():
            a_layout, b_layout, c_layout = abc_layout_key
            sorted_configs.append((M, N, K, a_layout, b_layout, c_layout, config_idx))

    # Sort by (M, N, K, A, B, C) for binary search
    def sort_key(x):
        M, N, K, a_layout, b_layout, c_layout, config_idx = x
        # Convert layouts to comparable values: row_major=0, col_major=1
        a_val = 0 if a_layout == "row_major" else 1
        b_val = 0 if b_layout == "row_major" else 1
        c_val = 0 if c_layout == "row_major" else 1
        return (M, N, K, a_val, b_val, c_val)

    sorted_configs.sort(key=sort_key)

    # Generate code
    code = f"""// Auto-generated file - DO NOT EDIT
#ifndef ROCM_SGEMM_CONFIG_GENERATED_HPP
#define ROCM_SGEMM_CONFIG_GENERATED_HPP

#include <rocm_sgemm/kernel/common.hpp>
#include <array>
#include <tuple>
#include <cstddef>
#include <cmath>
#include <limits>
#include <algorithm>

namespace rocm_sgemm
{{

// Number of unique kernel variants
static constexpr size_t KERNEL_VARIANTS = {num_configs};

// Configuration parameters for a specific problem size
struct gemm_params
{{
    int block_size;
    int block_m;
    int block_n;
    int block_k;
    int warp_tile_m_count;
    int warp_tile_n_count;
    int thread_tile_m;
    int thread_tile_n;
    int threads_n;
}};

namespace detail
{{
    // Kernel configuration tuple
    using kernel_config = std::tuple<int, int, int, int, int, int, int, int, int>; // block_size, block_m, block_n, block_k, warp_tile_m_count, warp_tile_n_count, thread_tile_m, thread_tile_n, threads_n

    // All unique kernel configurations
    static constexpr std::array<kernel_config, KERNEL_VARIANTS> kernel_configs = {{
"""

    # Generate config array
    for i, (bs, bm, bn, bk, wmc, wnc, tm, tn, threads_n) in enumerate(unique_configs):
        code += f"        std::tuple<int, int, int, int, int, int, int, int, int>{{{bs}, {bm}, {bn}, {bk}, {wmc}, {wnc}, {tm}, {tn}, {threads_n}}}"
        code += "," if i < len(unique_configs) - 1 else ""
        code += f" // Config {i}\n"

    code += f"""    }};

    // Default config (last in the array)
    static constexpr size_t DEFAULT_CONFIG_IDX = KERNEL_VARIANTS - 1;

    // Configuration lookup key
    struct config_key
    {{
        size_t m, n, k;
        m_layout layout_a, layout_b, layout_c;

        constexpr bool operator<(const config_key& other) const
        {{
            if(m != other.m) return m < other.m;
            if(n != other.n) return n < other.n;
            if(k != other.k) return k < other.k;
            if(layout_a != other.layout_a) return layout_a < other.layout_a;
            if(layout_b != other.layout_b) return layout_b < other.layout_b;
            return layout_c < other.layout_c;
        }}

        constexpr bool operator==(const config_key& other) const
        {{
            return m == other.m && n == other.n && k == other.k &&
                   layout_a == other.layout_a && layout_b == other.layout_b && layout_c == other.layout_c;
        }}
    }};

    // Sorted configuration map for binary search
    static constexpr std::array<std::pair<config_key, size_t>, {len(sorted_configs)}> sorted_config_map = {{{{
"""

    # Generate sorted config array
    for i, (M, N, K, a_layout, b_layout, c_layout, config_idx) in enumerate(sorted_configs):
        # Convert layout strings to enum values
        a_enum = f"m_layout::{a_layout}" if a_layout != "any" else "m_layout::row_major"
        b_enum = f"m_layout::{b_layout}" if b_layout != "any" else "m_layout::row_major"
        c_enum = f"m_layout::{c_layout}" if c_layout != "any" else "m_layout::row_major"

        code += f"        {{{{{M}, {N}, {K}, {a_enum}, {b_enum}, {c_enum}}}, {config_idx}}}"
        code += "," if i < len(sorted_configs) - 1 else ""
        code += f" // {M}x{N}x{K}, A={a_layout}, B={b_layout}, C={c_layout}\n"

    code += """    }};

    // Find closest configuration when exact match not found
    constexpr size_t find_closest_config(size_t m, size_t n, size_t k,
                                         m_layout layout_a,
                                         m_layout layout_b,
                                         m_layout layout_c)
    {
        // If empty, return default config
        if(sorted_config_map.empty())
        {
            return DEFAULT_CONFIG_IDX;
        }

        // Logarithmic distance metric (better for matrix operations)
        auto size_distance = [](size_t m1, size_t n1, size_t k1,
                               size_t m2, size_t n2, size_t k2) -> double
        {
            double log_diff_m = std::log2(static_cast<double>(m1)) - std::log2(static_cast<double>(m2));
            double log_diff_n = std::log2(static_cast<double>(n1)) - std::log2(static_cast<double>(n2));
            double log_diff_k = std::log2(static_cast<double>(k1)) - std::log2(static_cast<double>(k2));
            return log_diff_m * log_diff_m + log_diff_n * log_diff_n + log_diff_k * log_diff_k;
        };

        // Find config with exact matching (A,B,C) layout and closest size
        double min_distance = std::numeric_limits<double>::max();
        size_t best_idx = DEFAULT_CONFIG_IDX;

        for(size_t i = 0; i < sorted_config_map.size(); ++i)
        {
            const auto& entry = sorted_config_map[i];
            const auto& key = entry.first;

            // Check if (A,B,C) layout matches exactly
            if(key.layout_a == layout_a && key.layout_b == layout_b && key.layout_c == layout_c)
            {
                double dist = size_distance(m, n, k, key.m, key.n, key.k);
                if(dist < min_distance)
                {
                    min_distance = dist;
                    best_idx = i;
                }
            }
        }

        // If we found a match with right (A,B,C) layout, return it
        if(min_distance < std::numeric_limits<double>::max())
        {
            return sorted_config_map[best_idx].second;
        }

        // Fallback: find closest size regardless of layout
        min_distance = std::numeric_limits<double>::max();

        for(size_t i = 0; i < sorted_config_map.size(); ++i)
        {
            const auto& entry = sorted_config_map[i];
            const auto& key = entry.first;
            double dist = size_distance(m, n, k, key.m, key.n, key.k);
            if(dist < min_distance)
            {
                min_distance = dist;
                best_idx = i;
            }
        }

        return sorted_config_map[best_idx].second;
    }

    // Find the best configuration using binary search
    constexpr size_t find_best_config(size_t m, size_t n, size_t k,
                                      m_layout layout_a,
                                      m_layout layout_b,
                                      m_layout layout_c)
    {
        config_key target{m, n, k, layout_a, layout_b, layout_c};

        // Binary search using std::lower_bound
        auto it = std::lower_bound(
            sorted_config_map.begin(),
            sorted_config_map.end(),
            std::make_pair(target, size_t(0)),
            [](const auto& a, const auto& b)
            {
                return a.first < b.first;
            }
        );

        // Check if we found an exact match
        if(it != sorted_config_map.end() && it->first == target)
        {
            return it->second;
        }

        // Fall back to closest match
        return find_closest_config(m, n, k, layout_a, layout_b, layout_c);
    }

} // namespace detail

/**
 * @brief Get the optimal configuration parameters for a specific problem size and layout
 *
 * @param m Number of rows in matrices C and A
 * @param n Number of columns in matrices C and B
 * @param k Number of columns in matrix A / rows in matrix B
 * @param layout_c Layout of matrix C
 * @param layout_a Layout of matrix A
 * @param layout_b Layout of matrix B
 * @return Tuned parameters for the given problem
 */
constexpr gemm_params get_gemm_params(size_t m, size_t n, size_t k,
                                       m_layout layout_c,
                                       m_layout layout_a,
                                       m_layout layout_b)
{
    // Find the best configuration for this problem (now includes C layout)
    const size_t config_idx = detail::find_best_config(m, n, k, layout_a, layout_b, layout_c);

    // Get the configuration parameters
    const auto& config = detail::kernel_configs[config_idx];
    return gemm_params{
        std::get<0>(config),
        std::get<1>(config),
        std::get<2>(config),
        std::get<3>(config),
        std::get<4>(config),
        std::get<5>(config),
        std::get<6>(config),
        std::get<7>(config),
        std::get<8>(config)
    };
}

/**
 * @brief Get the index of a kernel configuration in the configuration table
 *
 * @param params GEMM kernel parameters
 * @return Index of the kernel configuration
 */
constexpr size_t get_kernel_config_index(const gemm_params& params)
{
    for(size_t i = 0; i < detail::kernel_configs.size(); ++i)
    {
        const auto& config = detail::kernel_configs[i];
        if(std::get<0>(config) == params.block_size &&
           std::get<1>(config) == params.block_m &&
           std::get<2>(config) == params.block_n &&
           std::get<3>(config) == params.block_k &&
           std::get<4>(config) == params.warp_tile_m_count &&
           std::get<5>(config) == params.warp_tile_n_count &&
           std::get<6>(config) == params.thread_tile_m &&
           std::get<7>(config) == params.thread_tile_n &&
           std::get<8>(config) == params.threads_n)
        {
            return i;
        }
    }

    // Return the default config index if not found
    return detail::kernel_configs.size() - 1;
}

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_CONFIG_GENERATED_HPP"""

    with open(output_file, 'w') as f:
        f.write(code)

def main():
    parser = argparse.ArgumentParser(description='Generate GEMM configuration header with binary search')
    parser.add_argument('config_file', type=str, help='Input JSON configuration file')
    parser.add_argument('output_file', type=str, help='Output header file')

    args = parser.parse_args()
    generate_config_header(args.config_file, args.output_file)

if __name__ == '__main__':
    main()
