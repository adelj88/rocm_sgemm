#!/usr/bin/env python3

import json
import argparse
from pathlib import Path

def generate_config_header(config_file, output_file):
    """Generate the config_generated.hpp header file"""
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Extract unique configurations (10 generator parameters)
    unique_configs = set()
    for conf in config['configurations']:
        cfg = conf['config']
        unique_configs.add(_config_tuple_of(cfg))

    # Always add default config if not present. Generator form of the original 128x128x8 default
    # (warps(4,1) x warp_tile(4,4) x thread_tile(2,4) x threads_n=8 -> block_m=block_n=128, bs=128).
    default_config = (4, 1, 4, 4, 2, 4, 8, 8, 0, 8)
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

        cfg          = conf['config']
        config_tuple = _config_tuple_of(cfg)
        config_idx   = unique_configs.index(config_tuple)

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

    # Emit the config lookup header consumed by the runtime dispatch.
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

// Configuration parameters for a specific problem size. Block dimensions are derived from these
// generators in the kernel (device) and at launch (host); they are never stored here.
struct gemm_params
{{
    int warps_m;
    int warps_n;
    int warp_tile_m_count;
    int warp_tile_n_count;
    int thread_tile_m;
    int thread_tile_n;
    int threads_n;
    int block_k;
    int single_buffer;
    int swizzle;
}};

namespace detail
{{
    // Kernel configuration tuple (10 generator parameters)
    using kernel_config = std::tuple<int, int, int, int, int, int, int, int, int, int>;

    // All unique kernel configurations
    static constexpr std::array<kernel_config, KERNEL_VARIANTS> kernel_configs = {{
"""

    # Generate config array
    for i, (wm, wn, wmc, wnc, tm, tn, threads_n, bk, single_buffer, swizzle) in enumerate(unique_configs):
        code += f"        std::tuple<int, int, int, int, int, int, int, int, int, int>{{{wm}, {wn}, {wmc}, {wnc}, {tm}, {tn}, {threads_n}, {bk}, {single_buffer}, {swizzle}}}"
        code += "," if i < len(unique_configs) - 1 else ""
        code += f" // Config {i}: warps({wm},{wn}) warp_tile({wmc},{wnc}) thread_tile({tm},{tn}) threads_n={threads_n} bk={bk} single_buffer={single_buffer} swizzle={swizzle}\n"

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
    // Find the best configuration for this problem
    const size_t config_idx = detail::find_best_config(m, n, k, layout_a, layout_b, layout_c);

    // Get the configuration parameters
    const auto& config = detail::kernel_configs[config_idx];
    return gemm_params{
        std::get<0>(config),  // warps_m
        std::get<1>(config),  // warps_n
        std::get<2>(config),  // warp_tile_m_count
        std::get<3>(config),  // warp_tile_n_count
        std::get<4>(config),  // thread_tile_m
        std::get<5>(config),  // thread_tile_n
        std::get<6>(config),  // threads_n
        std::get<7>(config),  // block_k
        std::get<8>(config),  // single_buffer
        std::get<9>(config)   // swizzle
    };
}

} // namespace rocm_sgemm

#endif // ROCM_SGEMM_CONFIG_GENERATED_HPP"""

    with open(output_file, 'w') as f:
        f.write(code)

# =====================================================================================
# Content-hash identity: filenames and getter symbols are derived from the config+layout
# values (not list position), so inserting/reordering configs leaves unrelated files
# untouched. Combined with write_if_changed + stale pruning, only genuinely changed
# kernels recompile — the build stays incremental and never links a stale object.
# =====================================================================================
def config_signature(config_tuple, layout_str):
    """Stable identity string for a (config, layout) from its parameter values."""
    wm, wn, wmc, wnc, tm, tn, threads_n, bk, single_buffer, swizzle = config_tuple
    return (f"wm{wm}_wn{wn}_wtm{wmc}_wtn{wnc}_tm{tm}_tn{tn}"
            f"_thn{threads_n}_bk{bk}_sb{single_buffer}_sw{swizzle}_{layout_str}")


def write_if_changed(filepath, content):
    """Write only when content differs, so unchanged files keep their mtime and the build
    system does not recompile them. Returns True if written."""
    if filepath.exists() and filepath.read_text() == content:
        return False
    filepath.write_text(content)
    return True


def layout_str_of(layout):
    """Three-letter layout tag, e.g. row/col/col -> 'rcc'."""
    return f"{layout['A'][0]}{layout['B'][0]}{layout['C'][0]}"


def _config_tuple_of(cfg):
    return (
        cfg['warps_m'], cfg['warps_n'],
        cfg['warp_tile_m_count'], cfg['warp_tile_n_count'],
        cfg['thread_tile_m'], cfg['thread_tile_n'], cfg['threads_n'],
        cfg['block_k'], cfg.get('single_buffer', 0), cfg.get('swizzle', 8)
    )


def generate_kernel_sources(config_file, output_dir):
    """One source file per (config, layout) with both aligned and unaligned ::run getters."""
    with open(config_file, 'r') as f:
        config = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seen_combinations = set()
    file_list = []
    written = 0

    for conf in config['configurations']:
        cfg    = conf['config']
        layout = conf['layout']

        config_tuple = _config_tuple_of(cfg)
        layout_key   = (layout['A'], layout['B'], layout['C'])
        if (config_tuple, layout_key) in seen_combinations:
            continue
        seen_combinations.add((config_tuple, layout_key))

        wm, wn, wmc, wnc, tm, tn, threads_n, bk, single_buffer, swizzle = config_tuple
        layout_a, layout_b, layout_c = layout_key
        layout_str = layout_str_of(layout)
        sig        = config_signature(config_tuple, layout_str)

        filename = f"kernel_inst_{sig}.cpp"
        filepath = output_path / filename
        file_list.append(filename)

        code = f"""// Auto-generated kernel instantiation file - DO NOT EDIT
// Config: warps({wm},{wn}) warp_tile({wmc},{wnc}) thread_tile({tm},{tn}) threads_n={threads_n} bk={bk} single_buffer={single_buffer} swizzle={swizzle}
// Layout: A={layout_a}, B={layout_b}, C={layout_c}

#include <rocm_sgemm/kernel/kernel.hpp>

namespace rocm_sgemm
{{

extern "C" void* get_kernel_inst_{sig}_aligned() {{
    return (void*)&kernel_gemm_impl<float,
        m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b},
        {wm}, {wn}, {wmc}, {wnc}, {tm}, {tn}, {threads_n}, {bk}, {single_buffer}, {swizzle},
        1>::run;
}}

extern "C" void* get_kernel_inst_{sig}_unaligned() {{
    return (void*)&kernel_gemm_impl<float,
        m_layout::{layout_c}, m_layout::{layout_a}, m_layout::{layout_b},
        {wm}, {wn}, {wmc}, {wnc}, {tm}, {tn}, {threads_n}, {bk}, {single_buffer}, {swizzle},
        0>::run;
}}

}} // namespace rocm_sgemm
"""
        if write_if_changed(filepath, code):
            written += 1

    # Prune stale kernel files from previous generations that no longer match any config.
    keep = set(file_list)
    removed = 0
    for stale in output_path.glob("kernel_inst_*.cpp"):
        if stale.name not in keep:
            stale.unlink()
            removed += 1

    # File lists for CMake (write_if_changed so CMake only reconfigures when the set changes).
    filelist_path = output_path / "kernel_sources.txt"
    write_if_changed(filelist_path, "".join(f"{name}\n" for name in file_list))

    cmake_lines = ["# Auto-generated list of kernel source files", "set(KERNEL_INST_SOURCES"]
    cmake_lines += [f"    {(output_path / name).as_posix()}" for name in file_list]
    cmake_lines.append(")")
    cmake_lines.append("")
    write_if_changed(output_path / "kernel_sources.cmake", "\n".join(cmake_lines))

    print(f"Generated {len(file_list)} kernel source files in {output_dir} "
          f"({written} written/changed, {removed} stale removed)")

    generate_kernel_lookup(config_file, output_path, file_list)
    return file_list


def generate_kernel_lookup(config_file, output_dir, file_list):
    """Static [config_idx][layout_idx][alignment_idx] table of signature-named getters."""
    with open(config_file, 'r') as f:
        config = json.load(f)

    unique_configs = set()
    for conf in config['configurations']:
        unique_configs.add(_config_tuple_of(conf['config']))
    default_config = (4, 1, 4, 4, 2, 4, 8, 8, 0, 8)
    unique_configs.add(default_config)
    unique_configs = sorted(list(unique_configs))
    num_configs = len(unique_configs)

    layouts = [
        ('row_major', 'row_major', 'row_major', 'rrr'),
        ('row_major', 'row_major', 'col_major', 'rrc'),
        ('row_major', 'col_major', 'row_major', 'rcr'),
        ('row_major', 'col_major', 'col_major', 'rcc'),
        ('col_major', 'row_major', 'row_major', 'crr'),
        ('col_major', 'row_major', 'col_major', 'crc'),
        ('col_major', 'col_major', 'row_major', 'ccr'),
        ('col_major', 'col_major', 'col_major', 'ccc'),
    ]

    # (config_tuple, layout_tuple) -> signature (stable identity)
    config_layout_to_sig = {}
    seen = set()
    for conf in config['configurations']:
        cfg    = conf['config']
        layout = conf['layout']
        config_tuple = _config_tuple_of(cfg)
        layout_tuple = (layout['A'], layout['B'], layout['C'])
        if (config_tuple, layout_tuple) in seen:
            continue
        seen.add((config_tuple, layout_tuple))
        config_layout_to_sig[(config_tuple, layout_tuple)] = config_signature(
            config_tuple, layout_str_of(layout))

    code = """// Auto-generated kernel lookup - DO NOT EDIT
#include <rocm_sgemm/kernel/common.hpp>
#include <cstddef>

namespace rocm_sgemm
{

"""
    for (config_tuple, layout_tuple), sig in config_layout_to_sig.items():
        code += f'extern "C" void* get_kernel_inst_{sig}_aligned();\n'
        code += f'extern "C" void* get_kernel_inst_{sig}_unaligned();\n'

    code += f"""
// Static kernel lookup table: [config_idx][layout_idx][alignment_idx]
// config_idx: 0 to {num_configs-1}
// layout_idx: 0=rrr, 1=rrc, 2=rcr, 3=rcc, 4=crr, 5=crc, 6=ccr, 7=ccc
// alignment_idx: 0=unaligned, 1=aligned
static void* kernel_table[{num_configs}][8][2] = {{
"""
    for config_idx, config_tuple in enumerate(unique_configs):
        wm, wn, wmc, wnc, tm, tn, threads_n, bk, single_buffer, swizzle = config_tuple
        code += (f"    // Config {config_idx}: warps({wm},{wn}) warp_tile({wmc},{wnc}) "
                 f"thread_tile({tm},{tn}) threads_n={threads_n} bk={bk} "
                 f"single_buffer={single_buffer} swizzle={swizzle}\n")
        code += "    {\n"
        for layout_idx, (la, lb, lc, layout_str) in enumerate(layouts):
            key = (config_tuple, (la, lb, lc))
            code += "        {"
            if key in config_layout_to_sig:
                sig = config_layout_to_sig[key]
                code += f"get_kernel_inst_{sig}_unaligned(), get_kernel_inst_{sig}_aligned()"
            else:
                code += "nullptr, nullptr"
            code += "}"
            if layout_idx < 7:
                code += ","
            code += "\n"
        code += "    }"
        if config_idx < num_configs - 1:
            code += ","
        code += "\n"

    code += """};

void* lookup_kernel(size_t config_idx, size_t layout_idx, size_t alignment_idx)
{
    return kernel_table[config_idx][layout_idx][alignment_idx];
}

} // namespace rocm_sgemm
"""
    write_if_changed(output_dir / "kernel_lookup.cpp", code)
    print(f"Generated kernel lookup table in {output_dir / 'kernel_lookup.cpp'}")


def main():
    parser = argparse.ArgumentParser(description='Generate SGEMM configuration header and kernel sources')
    parser.add_argument('config_file', type=str, help='Input JSON configuration file')
    parser.add_argument('output_file', type=str, help='Output header file')
    parser.add_argument('--kernel-dir', type=str, help='Output directory for kernel source files')
    args = parser.parse_args()

    generate_config_header(args.config_file, args.output_file)
    if args.kernel_dir:
        generate_kernel_sources(args.config_file, args.kernel_dir)


if __name__ == '__main__':
    main()
