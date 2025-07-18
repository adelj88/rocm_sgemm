def tune_abc_layout(self, M, N, K, layout_a, layout_b, layout_c, max_evaluations=150):
        """Tune for (A,B,C) layout combination using Optuna TPE."""
        layout_names = {0: "r", 1: "c"}
        print(f"\nTuning {M}×{N}×{K} with layout A={layout_names[layout_a]}, B={layout_names[layout_b]}, C={layout_names[layout_c]}")
        print("Using Optuna TPE (Tree-structured Parzen Estimators)")

        # Reset state for this problem
        self.current_problem = (M, N, K, layout_a, layout_b, layout_c)
        self.total_evaluations = 0
        self.best_config = None
        self.best_time = float('inf')
        self.improvement_history = []

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(
                seed=self.random_seed,
                n_startup_trials=min(10, max_evaluations // 4),
                n_ei_candidates=24,
                multivariate=True
            )
        )

        print("Starting Optuna optimization...")

        # Enqueue baseline configurations first
        baseline#!/usr/bin/env python3

import subprocess
import json
import numpy as np
import optuna
import warnings
import re
import argparse
import sys
import random

warnings.filterwarnings('ignore')

class OptunaTuner:
    """Optuna-based TPE tuner for GEMM kernel optimization."""

    def __init__(self, max_shared_memory=65336, gpu_arch="gfx1100", baselines=None, random_seed=42):
        self.max_shared_memory = max_shared_memory
        self.gpu_arch = gpu_arch
        self.random_seed = random_seed

        # Set random seed for reproducible results
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Enhanced baseline configurations
        if baselines is None:
            self.baselines = [
                (128, 128, 128, 8, 4, 4, 2, 4, 8),  # Your baseline
                (128, 128, 128, 8, 4, 4, 2, 4, 4),  # threads_n=4 variant
                (256, 128, 128, 8, 2, 2, 4, 4, 4),  # Different block size
                (128, 128, 128, 8, 4, 1, 8, 4, 16), # Random blocks
                (128, 128, 128, 8, 1, 4, 8, 4, 8),  # Random blocks
                (128, 128, 128, 8, 2, 2, 8, 4, 8),  # Random blocks
                (128, 128, 128, 16, 1, 4, 8, 4, 8), # Random blocks
                (256, 128, 256, 8, 4, 1, 8, 4, 16), # Random blocks
                (256, 128, 64, 16, 1, 1, 4, 8, 2),  # Random blocks
                (512, 256, 256, 8, 2, 8, 4, 2, 8),  # Random blocks
                (64, 128, 64, 8, 4, 4, 2, 4, 4),    # Random blocks
                (128, 64, 256, 4, 2, 8, 4, 2, 4),   # Random blocks
                (128, 64, 256, 4, 2, 8, 4, 2, 8),   # Random blocks
            ]
        else:
            self.baselines = baselines

        # Parameter space definitions
        self.param_space = {
            'block_size': [64, 128, 256, 512],
            'block_m': [32, 64, 128, 256],
            'block_n': [32, 64, 128, 256],
            'block_k': [4, 8, 16, 32],
            'warp_tile_m_count': [1, 2, 4, 8],
            'warp_tile_n_count': [1, 2, 4, 8],
            'thread_tile_m': [1, 2, 4, 8],
            'thread_tile_n': [1, 2, 4, 8],
            'threads_n': [2, 4, 8, 16]
        }

        # Generate all valid configurations for sampling
        self.valid_configs = self._generate_valid_configs()
        self.filtered_param_space = self._create_filtered_parameter_space()
        print(f"Generated {len(self.valid_configs)} valid configurations")

        # State tracking
        self.current_problem = None
        self.total_evaluations = 0
        self.best_config = None
        self.best_time = float('inf')
        self.improvement_history = []

    def _create_filtered_parameter_space(self):
        """Create parameter space containing only values that appear in valid configurations."""
        filtered_space = {}

        # Extract unique values for each parameter from valid configs
        filtered_space['block_size'] = sorted(list(set(config[0] for config in self.valid_configs)))
        filtered_space['block_m'] = sorted(list(set(config[1] for config in self.valid_configs)))
        filtered_space['block_n'] = sorted(list(set(config[2] for config in self.valid_configs)))
        filtered_space['block_k'] = sorted(list(set(config[3] for config in self.valid_configs)))
        filtered_space['warp_tile_m_count'] = sorted(list(set(config[4] for config in self.valid_configs)))
        filtered_space['warp_tile_n_count'] = sorted(list(set(config[5] for config in self.valid_configs)))
        filtered_space['thread_tile_m'] = sorted(list(set(config[6] for config in self.valid_configs)))
        filtered_space['thread_tile_n'] = sorted(list(set(config[7] for config in self.valid_configs)))
        filtered_space['threads_n'] = sorted(list(set(config[8] for config in self.valid_configs)))

        print("Filtered parameter space:")
        for param, values in filtered_space.items():
            print(f"  {param}: {values}")

        return filtered_space

    def _generate_valid_configs(self):
        """Generate all valid discrete configurations."""
        configs = []
        count = 0
        for block_size in self.param_space['block_size']:
            for block_m in self.param_space['block_m']:
                for block_n in self.param_space['block_n']:
                    for block_k in self.param_space['block_k']:
                        for warp_tile_m_count in self.param_space['warp_tile_m_count']:
                            for warp_tile_n_count in self.param_space['warp_tile_n_count']:
                                for thread_tile_m in self.param_space['thread_tile_m']:
                                    for thread_tile_n in self.param_space['thread_tile_n']:
                                        for threads_n in self.param_space['threads_n']:
                                            count += 1
                                            config = (block_size, block_m, block_n, block_k,
                                                    warp_tile_m_count, warp_tile_n_count,
                                                    thread_tile_m, thread_tile_n, threads_n)
                                            if self._check_constraints(config):
                                                configs.append(config)
        print(f"Checked {count} total combinations, {len(configs)} valid after constraints")
        return configs

    def _check_constraints(self, config):
        """Check memory and resource constraints."""
        (block_size, block_m, block_n, block_k,
         warp_tile_m_count, warp_tile_n_count,
         thread_tile_m, thread_tile_n, threads_n) = config

        if any(x <= 0 for x in config):
            return False

        # threads_n must divide 32
        if 32 % threads_n != 0:
            return False

        # Calculate derived values
        threads_m = 32 // threads_n
        sub_tile_m = thread_tile_m * threads_m
        sub_tile_n = thread_tile_n * threads_n
        warp_m = warp_tile_m_count * sub_tile_m
        warp_n = warp_tile_n_count * sub_tile_n

        # Check kernel constraints
        if block_m % warp_m != 0:
            return False
        if block_n % warp_n != 0:
            return False

        warps_m = block_m // warp_m
        warps_n = block_n // warp_n
        num_warps = warps_m * warps_n

        if num_warps != block_size // 32:
            return False

        # Memory constraint
        lds_size = (block_m * block_k) + (block_k * block_n)
        memory_bytes = 2 * lds_size * 4  # Double buffering * sizeof(float)

        if memory_bytes > self.max_shared_memory:
            return False

        # Resource constraint
        if num_warps > 32:
            return False

        return True

    def _parse_benchmark_output(self, output):
        """Parse benchmark output to extract timing."""
        lines = output.strip().split('\n')
        for line in lines:
            if 'dynamic_kernel/manual_time' in line:
                match = re.search(r'(\d+\.?\d*)\s*ms', line)
                if match:
                    return float(match.group(1))
        return None

    def _evaluate_config(self, M, N, K, layout_a, layout_b, layout_c, config):
        """Evaluate a configuration and return timing."""
        (block_size, block_m, block_n, block_k,
         warp_tile_m_count, warp_tile_n_count,
         thread_tile_m, thread_tile_n, threads_n) = config

        try:
            result = subprocess.run([
                "rocm_sgemm/tuner",
                str(M), str(N), str(K),
                str(block_size), str(block_m), str(block_n), str(block_k),
                str(warp_tile_m_count), str(warp_tile_n_count),
                str(thread_tile_m), str(thread_tile_n), str(threads_n),
                str(layout_a), str(layout_b), str(layout_c),
                self.gpu_arch
            ], capture_output=True, text=True, timeout=60, check=False)

            if result.returncode != 0:
                return float('inf')

            time_ms = self._parse_benchmark_output(result.stdout)
            if time_ms is None:
                return float('inf')

            return time_ms

        except Exception:
            return float('inf')

    def _objective_function_with_params(self, trial):
        """Optuna objective function that works with actual parameters and prunes invalid configs."""
        # Get parameters suggested by Optuna
        block_size = trial.suggest_categorical('block_size', self.param_space['block_size'])
        block_m = trial.suggest_categorical('block_m', self.param_space['block_m'])
        block_n = trial.suggest_categorical('block_n', self.param_space['block_n'])
        block_k = trial.suggest_categorical('block_k', self.param_space['block_k'])
        warp_tile_m_count = trial.suggest_categorical('warp_tile_m_count', self.param_space['warp_tile_m_count'])
        warp_tile_n_count = trial.suggest_categorical('warp_tile_n_count', self.param_space['warp_tile_n_count'])
        thread_tile_m = trial.suggest_categorical('thread_tile_m', self.param_space['thread_tile_m'])
        thread_tile_n = trial.suggest_categorical('thread_tile_n', self.param_space['thread_tile_n'])
        threads_n = trial.suggest_categorical('threads_n', self.param_space['threads_n'])

        config = (block_size, block_m, block_n, block_k,
                  warp_tile_m_count, warp_tile_n_count,
                  thread_tile_m, thread_tile_n, threads_n)

        # Check constraints - if invalid, PRUNE the trial (don't count it)
        if not self._check_constraints(config):
            trial.set_user_attr("invalid_combination", True)
            raise optuna.TrialPruned("Invalid parameter combination")

        # Evaluate the valid configuration
        M, N, K, layout_a, layout_b, layout_c = self.current_problem
        time_ms = self._evaluate_config(M, N, K, layout_a, layout_b, layout_c, config)

        self.total_evaluations += 1

        # Track improvements
        if time_ms < self.best_time:
            self.best_time = time_ms
            self.best_config = config
            self.improvement_history.append(self.total_evaluations)
            print(f"  New best: {config} -> {time_ms:.3f}ms (trial {self.total_evaluations})")
        elif time_ms == self.best_time and self.best_time != float('inf'):
            self.best_config = config
            print(f"  New best: {config} -> {time_ms:.3f}ms (equal, trial {self.total_evaluations})")
        else:
            print(f"  Trial {self.total_evaluations}: {config} -> {time_ms:.3f}ms")

        return time_ms


    def tune_abc_layout(self, M, N, K, layout_a, layout_b, layout_c, max_evaluations=150):
        """Tune for (A,B,C) layout combination using Optuna TPE."""
        layout_names = {0: "r", 1: "c"}
        print(f"\nTuning {M}×{N}×{K} with layout A={layout_names[layout_a]}, B={layout_names[layout_b]}, C={layout_names[layout_c]}")
        print("Using Optuna TPE (Tree-structured Parzen Estimators)")

        # Reset state for this problem
        self.current_problem = (M, N, K, layout_a, layout_b, layout_c)
        self.total_evaluations = 0
        self.best_config = None
        self.best_time = float('inf')
        self.improvement_history = []

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(
                seed=self.random_seed,
                n_startup_trials=min(10, max_evaluations // 4),
                n_ei_candidates=24,
                multivariate=True
            )
        )

        print("Starting Optuna optimization...")

        # Add baseline configurations
        for i, baseline in enumerate(self.baselines):
            if self._check_constraints(baseline):
                (block_size, block_m, block_n, block_k,
                 warp_tile_m_count, warp_tile_n_count,
                 thread_tile_m, thread_tile_n, threads_n) = baseline

                study.enqueue_trial({
                    'block_size': block_size,
                    'block_m': block_m,
                    'block_n': block_n,
                    'block_k': block_k,
                    'warp_tile_m_count': warp_tile_m_count,
                    'warp_tile_n_count': warp_tile_n_count,
                    'thread_tile_m': thread_tile_m,
                    'thread_tile_n': thread_tile_n,
                    'threads_n': threads_n
                })

        # Run optimization - invalid configs will be pruned automatically
        study.optimize(self._objective_function_with_params, n_trials=max_evaluations, show_progress_bar=False)

        if self.best_config is None:
            print("  No valid configuration found!")
            return None

        # Calculate memory usage
        (block_size, block_m, block_n, block_k,
         warp_tile_m_count, warp_tile_n_count,
         thread_tile_m, thread_tile_n, threads_n) = self.best_config
        lds_size = (block_m * block_k) + (block_k * block_n)
        memory_used = 2 * lds_size * 4

        # Count only completed (non-pruned) trials
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

        coverage = (self.total_evaluations / len(self.valid_configs)) * 100
        print(f"  Best config: {self.best_config} -> {self.best_time:.3f}ms")
        print(f"  Memory usage: {memory_used}/{self.max_shared_memory} bytes")
        print(f"  Improvements found: {len(self.improvement_history)}")
        print(f"  Valid evaluations: {self.total_evaluations}")
        print(f"  Completed trials: {completed_trials}, Pruned trials: {pruned_trials}")
        print(f"  Space coverage: {coverage:.2f}%")

        return {
            'config': {
                'block_size': int(self.best_config[0]),
                'block_m': int(self.best_config[1]),
                'block_n': int(self.best_config[2]),
                'block_k': int(self.best_config[3]),
                'warp_tile_m_count': int(self.best_config[4]),
                'warp_tile_n_count': int(self.best_config[5]),
                'thread_tile_m': int(self.best_config[6]),
                'thread_tile_n': int(self.best_config[7]),
                'threads_n': int(self.best_config[8])
            },
            'time_ms': float(self.best_time),
            'evaluations': self.total_evaluations,
            'memory_used_bytes': memory_used,
            'space_coverage_percent': coverage
        }

    def tune_all(self, sizes=None, abc_layouts=None, max_evaluations=150):
        """Tune all size and (A,B,C) layout combinations."""
        if sizes is None:
            sizes = [
                (1024, 1024, 1024),
                (2048, 2048, 2048),
                (4096, 4096, 4096),
                (8192, 8192, 8192)
            ]

        if abc_layouts is None:
            # All 8 combinations of (A,B,C) layouts
            abc_layouts = [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]

        results = {}

        for M, N, K in sizes:
            size_key = f"{M}x{N}x{K}"
            results[size_key] = {}

            for layout_a, layout_b, layout_c in abc_layouts:
                layout_key = f"{layout_a}_{layout_b}_{layout_c}"

                result = self.tune_abc_layout(M, N, K, layout_a, layout_b, layout_c, max_evaluations)

                if result:
                    results[size_key][layout_key] = {
                        "M": M, "N": N, "K": K,
                        "layout": {
                            "A": "row_major" if layout_a == 0 else "col_major",
                            "B": "row_major" if layout_b == 0 else "col_major",
                            "C": "row_major" if layout_c == 0 else "col_major"
                        },
                        "config": result['config'],
                        "avg_time_ms": result['time_ms'],
                        "evaluations": result['evaluations'],
                        "memory_used_bytes": result['memory_used_bytes'],
                        "space_coverage_percent": result['space_coverage_percent']
                    }

        return results

def parse_matrix_sizes(size_strings):
    """Parse matrix size strings like '1024,1024,1024' into tuples."""
    sizes = []
    for size_str in size_strings:
        try:
            parts = size_str.split(',')
            if len(parts) != 3:
                raise ValueError(f"Invalid size format: {size_str}. Expected M,N,K")
            M, N, K = map(int, parts)
            sizes.append((M, N, K))
        except ValueError as e:
            print(f"Error parsing size '{size_str}': {e}")
            sys.exit(1)
    return sizes

def parse_layouts(layout_strings):
    """Parse layout strings like 'r,r,r' or 'c,c,c' into tuples (A,B,C)."""
    layouts = []
    layout_map = {'r': 0, 'c': 1, 'row_major': 0, 'col_major': 1}

    for layout_str in layout_strings:
        try:
            parts = layout_str.split(',')
            if len(parts) != 3:
                raise ValueError(f"Invalid layout format: {layout_str}. Expected A,B,C")

            layout_tuple = []
            for part in parts:
                part = part.strip().lower()
                if part not in layout_map:
                    raise ValueError(f"Invalid layout '{part}'. Use 'r' or 'c' (or 'row_major'/'col_major')")
                layout_tuple.append(layout_map[part])

            layouts.append(tuple(layout_tuple))
        except ValueError as e:
            print(f"Error parsing layout '{layout_str}': {e}")
            sys.exit(1)

    return layouts

def parse_baselines(baseline_strings):
    """Parse baseline strings like '128,128,128,8,4,4,2,4,8' into tuples."""
    baselines = []
    for baseline_str in baseline_strings:
        try:
            parts = baseline_str.split(',')
            if len(parts) != 9:
                raise ValueError(f"Invalid baseline format: {baseline_str}. Expected block_size,block_m,block_n,block_k,warp_tile_m_count,warp_tile_n_count,thread_tile_m,thread_tile_n,threads_n")
            config = tuple(map(int, parts))
            baselines.append(config)
        except ValueError as e:
            print(f"Error parsing baseline '{baseline_str}': {e}")
            sys.exit(1)
    return baselines

def main():
    parser = argparse.ArgumentParser(
        description='Optuna TPE GEMM Tuner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run with Optuna TPE
  python tune.py

  # Specific seed for reproducibility
  python tune.py --seed 123

  # Larger budget for better results
  python tune.py --budget 100

  # Test specific sizes
  python tune.py --sizes 1024,1024,1024 2048,2048,2048

  # Test specific (A,B,C) layout combinations
  python tune.py --layouts r,r,r c,c,c r,c,r

  # Custom baselines
  python tune.py --baselines 128,128,128,8,4,4,2,4,8 256,128,128,8,2,2,4,4,4

  # Different GPU architecture
  python tune.py --gpu-arch gfx1103
        """)

    parser.add_argument('--sizes', nargs='*',
                       help='Matrix sizes as M,N,K (e.g., 1024,1024,1024 2048,2048,2048)')
    parser.add_argument('--layouts', nargs='*',
                       help='Matrix (A,B,C) layouts as A,B,C (e.g., r,r,r c,c,c or row_major,col_major,row_major)')
    parser.add_argument('--baselines', nargs='*',
                       help='Baseline configs as block_size,block_m,block_n,block_k,warp_tile_m_count,warp_tile_n_count,thread_tile_m,thread_tile_n,threads_n')
    parser.add_argument('--budget', type=int, default=150,
                       help='Evaluation budget per (A,B,C) layout combination (default: 150)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--gpu-arch', default='gfx1100', help='GPU architecture (default: gfx1100)')
    parser.add_argument('--max-memory', type=int, default=65336,
                       help='Maximum shared memory in bytes (default: 65336)')
    parser.add_argument('--output', default='gemm_config_tuned.json',
                       help='Output JSON file (default: gemm_config_tuned.json)')

    args = parser.parse_args()

    # Parse inputs
    if args.sizes:
        sizes = parse_matrix_sizes(args.sizes)
    else:
        sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192),
        ]

    if args.layouts:
        abc_layouts = parse_layouts(args.layouts)
    else:
        # All 8 combinations of (A,B,C) layouts
        abc_layouts = [(a, b, c) for a in [0, 1] for b in [0, 1] for c in [0, 1]]

    if args.baselines:
        baselines = parse_baselines(args.baselines)
    else:
        baselines = None

    print("Optuna TPE GEMM Tuner")
    print(f"Random seed: {args.seed}")
    print(f"GPU Architecture: {args.gpu_arch}")
    print(f"Evaluation budget per (A,B,C) layout: {args.budget}")
    print(f"Shared memory limit: {args.max_memory} bytes")
    print(f"Matrix sizes to test: {len(sizes)}")
    for size in sizes:
        print(f"  {size[0]}×{size[1]}×{size[2]}")
    print(f"(A,B,C) layout combinations to test: {len(abc_layouts)}")
    for i, layout in enumerate(abc_layouts):
        layout_names = ["r" if x == 0 else "c" for x in layout]
        print(f"  {i+1}: A={layout_names[0]}, B={layout_names[1]}, C={layout_names[2]}")

    if baselines:
        print(f"Custom baselines: {len(baselines)}")
        for baseline in baselines:
            print(f"  {baseline}")

    tuner = OptunaTuner(
        max_shared_memory=args.max_memory,
        gpu_arch=args.gpu_arch,
        baselines=baselines,
        random_seed=args.seed
    )

    results = tuner.tune_all(sizes=sizes, abc_layouts=abc_layouts, max_evaluations=args.budget)

    # Generate configuration JSON
    configs = []
    for size_results in results.values():
        for result in size_results.values():
            config = {
                "range": {"M": result["M"], "N": result["N"], "K": result["K"]},
                "layout": {
                    "A": result["layout"]["A"],
                    "B": result["layout"]["B"],
                    "C": result["layout"]["C"]
                },
                "config": result["config"]
            }
            configs.append(config)

    config_data = {"configurations": configs}

    # Save results
    with open(args.output, "w") as f:
        json.dump(config_data, f, indent=4)

    # Print summary
    print("\n" + "="*80)
    print("OPTUNA TPE OPTIMIZATION RESULTS:")
    print("="*80)

    total_evaluations = 0
    total_coverage = 0
    count = 0

    for size_key, size_results in results.items():
        print(f"\n{size_key}:")
        for layout_key, result in size_results.items():
            config = result['config']
            coverage = result.get('space_coverage_percent', 0)
            print(f"  {layout_key}: bs={config['block_size']}, bm={config['block_m']}, bn={config['block_n']}, bk={config['block_k']}, "
                  f"wm={config['warp_tile_m_count']}, wn={config['warp_tile_n_count']}, "
                  f"tm={config['thread_tile_m']}, tn={config['thread_tile_n']}, threads_n={config['threads_n']} -> "
                  f"{result['avg_time_ms']:.3f}ms ({result['evaluations']} evals, {coverage:.2f}% coverage)")
            total_evaluations += result['evaluations']
            total_coverage += coverage
            count += 1

    if count > 0:
        avg_evals = total_evaluations / count
        avg_coverage = total_coverage / count
        print(f"\nTotal evaluations: {total_evaluations}")
        print(f"Average evaluations per (A,B,C) problem: {avg_evals:.1f}")
        print(f"Average space coverage: {avg_coverage:.2f}%")
    print(f"Configuration saved to: {args.output}")

if __name__ == "__main__":
    main()
