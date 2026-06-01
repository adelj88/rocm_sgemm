#!/usr/bin/env python3
"""SGEMM kernel configuration tuner: population-free estimation-of-distribution search.

Finds a fast kernel configuration (warps, tiles, threads, block-k, buffering, swizzle) for each
matrix size and memory layout within a fixed budget of real GPU benchmarks, learning purely
from measured times with no hand-coded performance model.

The only persistent state is the archive of benchmarked configs and their times. There is no
population, island, or working set -- nothing whose size must be chosen or which carries state
forward, and the valid configuration space is never enumerated. Each step everything is rebuilt
from the archive:

  1. Soft weighting. Every benchmarked config gets a scale-invariant weight w = (t_min / t)^beta,
     so faster configs have more statistical leverage while slow ones still inform the model.
     Failures get zero weight.

  2. Linkage estimation (which parameters are coupled), from data only:
       - weighted, Dirichlet-smoothed contingency tables over the archive;
       - mutual information between every parameter pair;
       - a finite-sample bias correction, MI -= (|Vi|-1)(|Vj|-1) / (2 * N_eff), removing the MI a
         pair shows under independence purely from limited samples, so spurious couplings are
         not learned (N_eff is the effective sample size of the weighting);
       - symmetric-uncertainty normalization SU = 2*MI / (H_i + H_j) in [0,1], so a many-valued
         parameter (e.g. block_k) does not outweigh a binary one;
       - greedy agglomeration of parameters whose SU exceeds a floor into a partition of blocks.

  3. Factored generative model. P(x) = product over blocks of P(x_block), each a Dirichlet-
     smoothed weighted joint over that block's values. Sampling it produces configs that respect
     the learned couplings while smoothing still allows novel-but-plausible combinations.

  4. Acquisition. Candidates are SAMPLED from the model (plus random immigrants) and validated
     against the constraints on the fly -- the full space is never enumerated, so cost is
     independent of how many valid configs exist. A random-forest surrogate fit on the whole
     archive scores the candidate pool by Expected Improvement; the argmax is benchmarked.

Nothing scales with the size of the configuration space: the archive is bounded by the budget,
the linkage model by the parameter count, and candidate generation is sampling with rejection.
Determinism is per (size, layout) via one seeded RNG and a fixed-seed surrogate.

Parameter-agnostic: adding or removing a tunable requires editing only `param_space`.

Dependencies: numpy, scikit-learn.
"""

import subprocess
import json
import re
import argparse
import sys
import random
import itertools
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor


# Canonical tunable-parameter order. Single source of truth shared by the tuner's
# param_space and the JSON read/merge helpers.
PARAM_ORDER = [
    'warps_m', 'warps_n',
    'warp_tile_m_count', 'warp_tile_n_count',
    'thread_tile_m', 'thread_tile_n', 'threads_n', 'block_k',
    'single_buffer', 'swizzle'
]

# LDS inner-dimension padding (floats) applied to the transposed staging tiles, matching
# lds_pad in the kernel's common.hpp. Row-major A and col-major B get padded.
LDS_PAD = 4


class SGEMMTuner:
    """Population-free EDA tuner for SGEMM kernel configurations."""

    # ---- Search hyperparameters ----
    WEIGHT_BETA        = 4.0   # soft-weight sharpness: w = (t_min / t) ** beta
    DIRICHLET_ALPHA    = 0.5   # Jeffreys smoothing on contingency cells
    SU_FLOOR           = 0.10  # min symmetric uncertainty to couple parameters
    GROUP_CAP          = 4     # max parameters per linkage block
    MIN_LINK_SAMPLES   = 12    # below this many finite evals -> univariate model (no linkage)
    POOL_MODEL         = 48    # candidates sampled from the generative model per step
    POOL_IMMIGRANT     = 16    # random-immigrant candidates per step
    RESAMPLE_TRIES     = 200   # attempts to draw one valid, unseen candidate (space is sparse)
    RF_TREES           = 48

    def __init__(self, max_shared_memory=65536, gpu_arch="gfx1100",
                 random_seed=42):
        self.max_shared_memory = max_shared_memory
        self.gpu_arch = gpu_arch
        self.random_seed = random_seed

        # All tunable parameters (generators). block_m/block_n are derived from these, never
        # tuned directly, so every point in the Cartesian product is a real kernel geometry.
        self.param_space = {
            'warps_m':           [1, 2, 4, 8],
            'warps_n':           [1, 2, 4, 8],
            'warp_tile_m_count': [1, 2, 4, 8],
            'warp_tile_n_count': [1, 2, 4, 8],
            'thread_tile_m':     [1, 2, 4, 8],
            'thread_tile_n':     [1, 2, 4, 8],
            'threads_n':         [2, 4, 8, 16],
            'block_k':           [4, 8, 16, 32],
            'single_buffer':     [0, 1],
            'swizzle':           [4, 8],
        }

        self.param_names = list(self.param_space.keys())
        self.n_params = len(self.param_names)
        self.IDX = {name: i for i, name in enumerate(self.param_names)}
        self._param_values = [sorted(self.param_space[n]) for n in self.param_names]

        assert self.param_names == PARAM_ORDER, (
            f"param_space order {self.param_names} != PARAM_ORDER {PARAM_ORDER}")

        # The valid space is NOT enumerated -- it is sampled with rejection during search.
        # Print an inexpensive sampled estimate of its size per layout for orientation only.
        self._print_layout_estimate()

    def _print_layout_estimate(self):
        """Prints a sampled estimate of the valid-config count per layout (no enumeration)."""
        total_cartesian = 1
        for vals in self._param_values:
            total_cartesian *= len(vals)
        rng = random.Random(self.random_seed)
        n_samples = 20000
        picks = [tuple(rng.choice(self._param_values[i]) for i in range(self.n_params))
                 for _ in range(n_samples)]
        print(f"Estimated valid configurations per layout (sampled, cartesian={total_cartesian}):")
        for la in (0, 1):
            for lb in (0, 1):
                for lc in (0, 1):
                    hits = sum(1 for c in picks if self._check_constraints(c, la, lb, lc))
                    est = int(round(total_cartesian * hits / n_samples))
                    la_s = "row" if la == 0 else "col"
                    lb_s = "row" if lb == 0 else "col"
                    lc_s = "row" if lc == 0 else "col"
                    print(f"  A={la_s}, B={lb_s}, C={lc_s}: ~{est}")

    # ------------------------------------------------------------------
    # Configuration & constraint helpers
    # ------------------------------------------------------------------

    def _check_constraints(self, config, layout_a=None, layout_b=None, layout_c=None):
        """Hardware constraint checker. Rejects configs that exceed shared memory or warp limits.

        Block dimensions are derived from the generators here (the same formula the kernel uses),
        so geometry is valid by construction -- the only rejections are hardware limits and the
        row-row mapper's swizzle rule.
        """
        warps_m = config[self.IDX['warps_m']]
        warps_n = config[self.IDX['warps_n']]
        warp_tile_m_count = config[self.IDX['warp_tile_m_count']]
        warp_tile_n_count = config[self.IDX['warp_tile_n_count']]
        thread_tile_m = config[self.IDX['thread_tile_m']]
        thread_tile_n = config[self.IDX['thread_tile_n']]
        threads_n = config[self.IDX['threads_n']]
        block_k = config[self.IDX['block_k']]
        single_buffer = config[self.IDX['single_buffer']]
        swizzle = config[self.IDX['swizzle']]

        if any(x < 0 for x in config):
            return False
        # single_buffer is the only parameter allowed to be 0.
        if any(config[i] == 0 for i in range(self.n_params)
               if self.param_names[i] != 'single_buffer'):
            return False

        # threads_n must divide the wave.
        if 32 % threads_n != 0:
            return False

        # Row-major A + row-major B falls back to the plain tile mapper, which requires swizzle==8.
        if layout_a is not None and layout_b is not None:
            if layout_a == 0 and layout_b == 0:
                if swizzle != 8:
                    return False

        # Derive the block tile from the generators (mirrors the kernel).
        threads_m = 32 // threads_n
        warp_m = warp_tile_m_count * thread_tile_m * threads_m
        warp_n = warp_tile_n_count * thread_tile_n * threads_n
        block_m = warps_m * warp_m
        block_n = warps_n * warp_n
        num_warps = warps_m * warps_n

        # Threads per block must not exceed the hardware maximum (1024 => num_warps <= 32).
        if num_warps > 32:
            return False
        if block_m > 1024 or block_n > 1024:
            return False

        lds = self._calc_lds(block_m, block_n, block_k, layout_a, layout_b)
        buffers = 1 if single_buffer else 2
        if buffers * lds * 4 > self.max_shared_memory:  # sizeof(float)
            return False
        return True

    @staticmethod
    def _calc_lds(block_m, block_n, block_k, layout_a=None, layout_b=None):
        """LDS (shared-memory) element footprint of a config (before the float/byte + buffer
        factors). Single source of truth for both the memory constraint and the report."""
        if layout_a is not None and layout_b is not None:
            pad_a = LDS_PAD if layout_a == 0 else 0
            pad_b = LDS_PAD if layout_b == 1 else 0
        else:
            pad_a, pad_b = 0, 0
        return block_k * (block_m + pad_a) + block_k * (block_n + pad_b)

    def _block_dims(self, d):
        """Derive (block_m, block_n) from a config dict's generators (mirrors the kernel)."""
        threads_m = 32 // d['threads_n']
        block_m = d['warps_m'] * d['warp_tile_m_count'] * d['thread_tile_m'] * threads_m
        block_n = d['warps_n'] * d['warp_tile_n_count'] * d['thread_tile_n'] * d['threads_n']
        return block_m, block_n

    def _config_to_dict(self, config):
        return {name: config[i] for i, name in enumerate(self.param_names)}

    @staticmethod
    def _hamming(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    def _one_hot(self, config):
        vec = []
        for i in range(self.n_params):
            for value in self._param_values[i]:
                vec.append(1.0 if config[i] == value else 0.0)
        return vec

    # ------------------------------------------------------------------
    # Benchmarking execution
    # ------------------------------------------------------------------

    def _parse_benchmark_output(self, output):
        for line in output.strip().split('\n'):
            if 'manual_time_mean' in line and 'repeats:' in line:
                m = re.search(r'(\d+\.?\d*)\s+ms', line)
                if m:
                    return float(m.group(1))
            elif ('dynamic_kernel' in line and 'manual_time' in line and '_mean' not in line):
                m = re.search(r'(\d+\.?\d*)\s+ms', line)
                if m:
                    return float(m.group(1))
        return None

    def _warmup(self, M, N, K, la, lb, lc, config, rounds=5):
        print(f"  [warmup] {rounds} rounds at {M}x{N}x{K} to ramp GPU clocks...", flush=True)
        for _ in range(rounds):
            if self._evaluate_config(M, N, K, la, lb, lc, config) == float('inf'):
                print("  [warmup] warning: warmup config failed to run", flush=True)
                break

    def _evaluate_config(self, M, N, K, la, lb, lc, config):
        d = self._config_to_dict(config)
        try:
            result = subprocess.run([
                "./rocm_sgemm/tuner",
                str(M), str(N), str(K),
                str(d['warps_m']), str(d['warps_n']),
                str(d['warp_tile_m_count']), str(d['warp_tile_n_count']),
                str(d['thread_tile_m']), str(d['thread_tile_n']), str(d['threads_n']),
                str(d['block_k']), str(d['single_buffer']), str(d['swizzle']),
                str(la), str(lb), str(lc), self.gpu_arch
            ], capture_output=True, text=True, timeout=60, check=False)
            if result.returncode != 0:
                return float('inf')
            t = self._parse_benchmark_output(result.stdout)
            return t if t is not None else float('inf')
        except Exception:
            return float('inf')

    # ------------------------------------------------------------------
    # Linkage estimation (weighted, bias-corrected mutual information)
    # ------------------------------------------------------------------

    def _learn_linkage(self, configs, weights):
        """Returns a partition of parameter indices into linkage blocks (list of tuples).

        Blocks are formed by greedily merging parameters whose bias-corrected symmetric
        uncertainty exceeds SU_FLOOR, capped at GROUP_CAP. Below MIN_LINK_SAMPLES effective
        samples, returns singletons (a univariate model -- nothing reliable to couple yet).
        """
        n = self.n_params
        singletons = [(i,) for i in range(n)]
        if len(configs) < self.MIN_LINK_SAMPLES:
            return singletons

        w_sum = sum(weights)
        w_sq = sum(w * w for w in weights)
        if w_sum <= 0 or w_sq <= 0:
            return singletons
        n_eff = (w_sum * w_sum) / w_sq
        alpha = self.DIRICHLET_ALPHA

        marg = []
        entropy = [0.0] * n
        for i in range(n):
            vals = self._param_values[i]
            counts = {v: 0.0 for v in vals}
            for cfg, w in zip(configs, weights):
                counts[cfg[i]] += w
            denom = w_sum + alpha * len(vals)
            p = {v: (counts[v] + alpha) / denom for v in vals}
            marg.append(p)
            entropy[i] = -sum(pv * math.log(pv) for pv in p.values() if pv > 0)

        su = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = self._param_values[i], self._param_values[j]
                joint = defaultdict(float)
                for cfg, w in zip(configs, weights):
                    joint[(cfg[i], cfg[j])] += w
                n_joint = len(vi) * len(vj)
                denom = w_sum + alpha * n_joint
                mi = 0.0
                for a in vi:
                    for b in vj:
                        p_ij = (joint.get((a, b), 0.0) + alpha) / denom
                        mi += p_ij * math.log(p_ij / (marg[i][a] * marg[j][b]))
                mi_corr = mi - (len(vi) - 1) * (len(vj) - 1) / (2.0 * n_eff)
                mi_corr = max(0.0, mi_corr)
                denom_h = entropy[i] + entropy[j]
                su[i][j] = su[j][i] = (2.0 * mi_corr / denom_h) if denom_h > 1e-12 else 0.0

        blocks = [[i] for i in range(n)]
        while True:
            best, best_su = None, self.SU_FLOOR
            for a in range(len(blocks)):
                for b in range(a + 1, len(blocks)):
                    if len(blocks[a]) + len(blocks[b]) > self.GROUP_CAP:
                        continue
                    pairs = [(x, y) for x in blocks[a] for y in blocks[b]]
                    avg = sum(su[x][y] for x, y in pairs) / len(pairs)
                    if avg > best_su:
                        best_su, best = avg, (a, b)
            if best is None:
                break
            a, b = best
            blocks[a] = blocks[a] + blocks[b]
            blocks.pop(b)
        return [tuple(sorted(g)) for g in blocks]

    def _build_block_models(self, blocks, configs, weights):
        w_sum = sum(weights)
        alpha = self.DIRICHLET_ALPHA
        models = []
        for g in blocks:
            value_lists = [self._param_values[i] for i in g]
            cells = list(itertools.product(*value_lists))  # bounded by GROUP_CAP
            counts = defaultdict(float)
            for cfg, w in zip(configs, weights):
                counts[tuple(cfg[i] for i in g)] += w
            denom = w_sum + alpha * len(cells)
            probs = [(counts.get(cell, 0.0) + alpha) / denom for cell in cells]
            total = sum(probs)
            cum, acc = [], 0.0
            for p in probs:
                acc += p / total
                cum.append(acc)
            models.append((g, cells, cum))
        return models

    def _sample_from_model(self, block_models, rng):
        cfg = [None] * self.n_params
        for g, cells, cum in block_models:
            r = rng.random()
            idx = 0
            while idx < len(cum) - 1 and r > cum[idx]:
                idx += 1
            for pos, i in enumerate(g):
                cfg[i] = cells[idx][pos]
        return tuple(cfg)

    def _sample_random_valid(self, rng, la, lb, lc, seen, produced):
        for _ in range(self.RESAMPLE_TRIES):
            cand = tuple(rng.choice(self._param_values[i]) for i in range(self.n_params))
            if cand in seen or cand in produced:
                continue
            if self._check_constraints(cand, la, lb, lc):
                return cand
        return None

    # ------------------------------------------------------------------
    # Expected Improvement
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_improvement(mu, sigma, best_val):
        sigma = np.maximum(sigma, 1e-8)
        diff = best_val - mu
        z = diff / sigma
        cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        pdf = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * z * z)
        return np.maximum(diff * cdf + sigma * pdf, 0.0)

    # ------------------------------------------------------------------
    # Core: population-free EDA search
    # ------------------------------------------------------------------

    def tune_layout(self, M, N, K, layout_a, layout_b, layout_c,
                    max_evaluations=300, existing_baseline=None):
        rng = random.Random(self.random_seed)

        print(f"\nTuning {M}x{N}x{K} layout A={layout_a} B={layout_b} C={layout_c}")
        print(f"  budget: {max_evaluations} evals")

        total_evals = 0
        best_config = None
        best_time = float('inf')
        seen = {}
        cache_hits = 0

        def evaluate(config):
            nonlocal total_evals, cache_hits, best_config, best_time
            if config in seen:
                cache_hits += 1
                return seen[config]
            t = self._evaluate_config(M, N, K, layout_a, layout_b, layout_c, config)
            seen[config] = t
            total_evals += 1
            if t < best_time:
                best_time = t
                best_config = config
                print(f"  New best: {config} -> {t:.3f}ms (eval {total_evals})")
            return t

        # 1. Space-filling initial design: farthest-point over random valid candidate batches
        #    (batch sampling keeps this independent of the size of the valid space).
        init_seeds = min(max(16, max_evaluations // 12), max_evaluations)
        chosen = []
        if existing_baseline is not None and self._check_constraints(
                existing_baseline, layout_a, layout_b, layout_c):
            evaluate(existing_baseline)
            chosen.append(existing_baseline)
        while total_evals < init_seeds and total_evals < max_evaluations:
            produced = set()
            batch = []
            for _ in range(64):
                c = self._sample_random_valid(rng, layout_a, layout_b, layout_c, seen, produced)
                if c is not None:
                    produced.add(c)
                    batch.append(c)
            if not batch:
                break
            if not chosen:
                pick = batch[0]
            else:
                pick = max(batch, key=lambda c: min(self._hamming(c, s) for s in chosen))
            evaluate(pick)
            chosen.append(pick)

        # 2. Model-based loop: rebuild linkage + generative model from the archive each step,
        #    sample candidates, screen by EI, benchmark the argmax.
        while total_evals < max_evaluations:
            finite = [(c, t) for c, t in seen.items() if math.isfinite(t)]
            if not finite:
                cand = self._sample_random_valid(rng, layout_a, layout_b, layout_c, seen, set())
                if cand is None:
                    break
                evaluate(cand)
                continue

            t_min = min(t for _, t in finite)
            good_cfgs = [c for c, _ in finite]
            weights = [(t_min / t) ** self.WEIGHT_BETA for _, t in finite]

            blocks = self._learn_linkage(good_cfgs, weights)
            block_models = self._build_block_models(blocks, good_cfgs, weights)

            produced = set()
            candidates = []
            tries = 0
            while len(candidates) < self.POOL_MODEL and tries < self.POOL_MODEL * 8:
                tries += 1
                cand = self._sample_from_model(block_models, rng)
                if (cand in seen or cand in produced
                        or not self._check_constraints(cand, layout_a, layout_b, layout_c)):
                    continue
                produced.add(cand)
                candidates.append(cand)
            for _ in range(self.POOL_IMMIGRANT):
                cand = self._sample_random_valid(rng, layout_a, layout_b, layout_c, seen, produced)
                if cand is not None:
                    produced.add(cand)
                    candidates.append(cand)
            if not candidates:
                break

            eval_cfgs = list(seen.keys())
            y_raw = np.array([seen[c] for c in eval_cfgs], dtype=np.float64)
            finite_mask = np.isfinite(y_raw)
            penalty = float(np.max(y_raw[finite_mask])) * 5.0 if np.any(finite_mask) else 1.0e6
            y_log = np.log(np.clip(np.where(finite_mask, y_raw, penalty), 1e-6, None))
            x_train = np.array([self._one_hot(c) for c in eval_cfgs], dtype=np.float32)
            x_pool = np.array([self._one_hot(c) for c in candidates], dtype=np.float32)

            rf = RandomForestRegressor(n_estimators=self.RF_TREES, max_features='sqrt',
                                       min_samples_leaf=1, random_state=self.random_seed,
                                       n_jobs=-1)
            rf.fit(x_train, y_log)
            tree_preds = np.array([tree.predict(x_pool) for tree in rf.estimators_])
            mu = np.mean(tree_preds, axis=0)
            sigma = np.std(tree_preds, axis=0)
            best_y_log = float(np.min(y_log[finite_mask]))
            ei = self._expected_improvement(mu, sigma, best_y_log)

            max_ei = np.max(ei)
            cand_idx = np.where(np.abs(ei - max_ei) < 1e-12)[0]
            chosen_idx = cand_idx[0] if len(cand_idx) == 1 else cand_idx[int(np.argmin(mu[cand_idx]))]
            evaluate(candidates[int(chosen_idx)])

        print(f"  Population-free EDA done (Evaluated {total_evals} configs)")

        if best_config is None:
            print("  No valid configurations found!")
            return None

        d = self._config_to_dict(best_config)
        block_m, block_n = self._block_dims(d)
        lds = self._calc_lds(block_m, block_n, d['block_k'], layout_a, layout_b)
        mem = (1 if d['single_buffer'] else 2) * lds * 4
        unique = len(seen)
        print(f"\n  Best: {best_config} -> {best_time:.3f}ms")
        print(f"  Memory: {mem}/{self.max_shared_memory} bytes")
        print(f"  Evals: {unique} unique, {cache_hits} cache hits")

        return {
            'config': self._config_to_dict(best_config),
            'raw_config': best_config,
            'time_ms': float(best_time),
            'evaluations': total_evals,
            'memory_used_bytes': mem,
            'space_coverage_percent': 0.0,
        }

    def tune_all(self, sizes=None, layouts=None, max_evaluations=300,
                 existing_configs=None, overwrite=False):
        if sizes is None:
            sizes = [(1024, 1024, 1024), (2048, 2048, 2048),
                     (4096, 4096, 4096), (8192, 8192, 8192)]
        if layouts is None:
            layouts = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                       (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

        results = {}
        for M, N, K in sizes:
            sk = f"{M}x{N}x{K}"
            results[sk] = {}
            for la, lb, lc in layouts:
                lk = f"{la}_{lb}_{lc}"
                baseline = None if overwrite else (existing_configs or {}).get(sk, {}).get(lk)

                warm_cfg = baseline
                if warm_cfg is None:
                    warm_cfg = self._sample_random_valid(
                        random.Random(self.random_seed), la, lb, lc, {}, set())
                if warm_cfg is not None:
                    self._warmup(M, N, K, la, lb, lc, warm_cfg)

                result = self.tune_layout(M, N, K, la, lb, lc,
                                          max_evaluations=max_evaluations,
                                          existing_baseline=baseline)
                if result:
                    results[sk][lk] = {
                        "M": M, "N": N, "K": K,
                        "layout": {
                            "A": "row_major" if la == 0 else "col_major",
                            "B": "row_major" if lb == 0 else "col_major",
                            "C": "row_major" if lc == 0 else "col_major"},
                        "config": result['config'],
                        "avg_time_ms": result['time_ms'],
                        "evaluations": result['evaluations'],
                        "memory_used_bytes": result['memory_used_bytes'],
                        "space_coverage_percent": result['space_coverage_percent']
                    }
        return results


# ======================================================================
# I/O helpers
# ======================================================================

def load_existing_json(path):
    if not Path(path).exists():
        print(f"Warning: '{path}' not found.")
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        out = {}
        for e in data.get('configurations', []):
            sk = f"{e['range']['M']}x{e['range']['N']}x{e['range']['K']}"
            l = e['layout']
            lk = (f"{'0' if l['A']=='row_major' else '1'}_"
                  f"{'0' if l['B']=='row_major' else '1'}_"
                  f"{'0' if l['C']=='row_major' else '1'}")
            c = e['config']
            out.setdefault(sk, {})[lk] = tuple(
                c.get(p, 8 if p == 'swizzle' else 0) for p in PARAM_ORDER)
        print(f"Loaded {len(data.get('configurations',[]))} configs from '{path}'")
        return out
    except Exception as e:
        print(f"Error loading '{path}': {e}")
        return None


def merge_results(existing, new):
    if existing is None:
        return new
    merged = {}
    for sk, ld in existing.items():
        merged[sk] = {}
        for lk, bl in ld.items():
            M, N, K = map(int, sk.split('x'))
            la, lb, lc = map(int, lk.split('_'))
            merged[sk][lk] = {
                "M": M, "N": N, "K": K,
                "layout": {
                    "A": "row_major" if la == 0 else "col_major",
                    "B": "row_major" if lb == 0 else "col_major",
                    "C": "row_major" if lc == 0 else "col_major"},
                "config": {p: int(bl[i]) for i, p in enumerate(PARAM_ORDER)},
                "avg_time_ms": None, "evaluations": 0,
                "memory_used_bytes": None, "space_coverage_percent": 0}
    for sk, ld in new.items():
        merged.setdefault(sk, {}).update(ld)
    return merged


def parse_matrix_sizes(ss):
    sizes = []
    for s in ss:
        parts = s.split(',')
        if len(parts) != 3:
            print(f"Error: expected M,N,K, got '{s}'")
            sys.exit(1)
        sizes.append(tuple(map(int, parts)))
    return sizes


def parse_layouts(ss):
    lm = {'row_major': 0, 'col_major': 1, 'r': 0, 'c': 1}
    layouts = []
    for s in ss:
        parts = s.split(',')
        if len(parts) != 3:
            print(f"Error: expected A,B,C, got '{s}'")
            sys.exit(1)
        layouts.append(tuple(lm[p.strip().lower()] for p in parts))
    return layouts


def main():
    p = argparse.ArgumentParser(
        description='SGEMM kernel tuner (population-free EDA)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tune.py
  python tune.py --input gemm_config_gfx1100.json --layouts r,r,r c,c,c
  python tune.py --sizes 4096,4096,4096
  python tune.py --budget 300 --seed 123
  python tune.py --gpu-arch gfx1103
        """)
    p.add_argument('--input', '-i', help='Input JSON with existing configs')
    p.add_argument('--sizes', nargs='*', help='Matrix sizes as M,N,K')
    p.add_argument('--layouts', nargs='*', help='Layouts as A,B,C')
    p.add_argument('--budget', type=int, default=300,
                   help='Eval budget per layout (default: 300)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu-arch', default='gfx1100')
    p.add_argument('--max-memory', type=int, default=65536)
    p.add_argument('--output', default=None,
                   help='Output JSON path (default: gemm_config_<arch>.json)')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing configs without using them as baselines')
    args = p.parse_args()

    if args.output is None:
        args.output = f'gemm_config_{args.gpu_arch}.json'

    existing = load_existing_json(args.input) if args.input else None
    sizes = parse_matrix_sizes(args.sizes) if args.sizes else [
        (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
        (8192, 8192, 8192)]
    layouts = parse_layouts(args.layouts) if args.layouts else [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]

    print("SGEMM Tuner (population-free EDA)")
    print(f"  Seed: {args.seed}  GPU: {args.gpu_arch}  Budget: {args.budget}  Memory: {args.max_memory}")
    print(f"  Sizes: {len(sizes)}  Layouts: {len(layouts)}")

    tuner = SGEMMTuner(args.max_memory, args.gpu_arch, args.seed)
    results = tuner.tune_all(sizes, layouts, args.budget, existing, args.overwrite)
    merged = merge_results(existing, results)

    configs = []
    for sr in merged.values():
        for r in sr.values():
            configs.append({
                "range": {"M": r["M"], "N": r["N"], "K": r["K"]},
                "layout": r["layout"], "config": r["config"]})

    with open(args.output, "w") as f:
        json.dump({"configurations": configs}, f, indent=4)

    print("\n" + "=" * 80)
    print("TUNING RESULTS")
    print("=" * 80)
    total_evals = 0
    for sk, sr in results.items():
        print(f"\n{sk}:")
        for lk, r in sr.items():
            c = r['config']
            print(f"  {lk}: warps_m={c['warps_m']},warps_n={c['warps_n']},"
                  f"wtm={c['warp_tile_m_count']},wtn={c['warp_tile_n_count']},"
                  f"tm={c['thread_tile_m']},tn={c['thread_tile_n']},threads_n={c['threads_n']},"
                  f"bk={c['block_k']},single_buffer={c['single_buffer']},"
                  f"swizzle={c['swizzle']}"
                  f" -> {r['avg_time_ms']:.3f}ms ({r['evaluations']} evals)")
            total_evals += r['evaluations']
    n = sum(len(d) for d in results.values()) if results else 0
    if n:
        print(f"\nTotal evals: {total_evals}  Avg: {total_evals // n}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
