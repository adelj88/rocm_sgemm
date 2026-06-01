#!/usr/bin/env python3
"""Generate docs/sgemm_line_comparison.png from fp32_sgemm_amd benchmark output.

Usage:
    ./sgemm 300 > sgemm_results.txt   # in the fp32_sgemm_amd repo, on gfx1100
    python3 generate_line_comparison.py sgemm_results.txt [-o sgemm_line_comparison.png]
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

KERNEL_LABELS = {
    "Kernel 0": "rocBLAS",
    "Kernel 5": "Sebastien K5",
    "Kernel 6": "Sebastien K6",
    "Kernel 7": "Sebastien K7",
    "Kernel 8": "Sebastien K8",
    "Kernel 9": "rocm_sgemm",
}

RESULT_RE = re.compile(
    r"min:\s*([\d.]+)\s*ms\s*->\s*([\d.]+)\s*GFLOPS\s*\|\s*avg:\s*([\d.]+)\s*ms\s*->\s*([\d.]+)\s*GFLOPS"
)


def parse(path: Path):
    entries = []
    current = None
    for line in path.read_text().splitlines():
        key = line.split(":")[0].strip()
        if key in KERNEL_LABELS:
            current = KERNEL_LABELS[key]
            continue
        m = RESULT_RE.search(line)
        if m and current:
            min_ms, min_gflops, avg_ms, avg_gflops = map(float, m.groups())
            entries.append(
                {
                    "label": current,
                    "min_ms": min_ms,
                    "min_tflops": min_gflops / 1000.0,
                    "avg_ms": avg_ms,
                    "avg_tflops": avg_gflops / 1000.0,
                }
            )
            current = None
    return entries


def plot(entries, output: Path):
    labels = [e["label"] for e in entries]
    min_tflops = [e["min_tflops"] for e in entries]
    avg_tflops = [e["avg_tflops"] for e in entries]
    x = range(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    (l_min,) = ax.plot(
        x, min_tflops, marker="o", linewidth=2, label="Minimum time"
    )
    (l_avg,) = ax.plot(
        x, avg_tflops, marker="s", linewidth=2, linestyle="--", label="Average time"
    )
    for line in (l_min, l_avg):
        for xi, yi in zip(line.get_xdata(), line.get_ydata()):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    rocblas = entries[0]
    for i, e in enumerate(entries[1:], start=1):
        ax.annotate(
            f"+{(rocblas['min_ms'] / e['min_ms'] - 1) * 100:.0f}%",
            (i, min_tflops[i]), textcoords="offset points", xytext=(0, -16),
            ha="center", fontsize=8, color="dimgray",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("TFLOPS (4096$^3$, row-major)")
    ax.set_title("SGEMM performance vs rocBLAS (AMD RX 7900 GRE)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Captured ./sgemm output")
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path(__file__).parent / "sgemm_line_comparison.png",
    )
    args = parser.parse_args()

    entries = parse(args.results)
    if not entries:
        sys.exit("No benchmark results found in input")
    plot(entries, args.output)


if __name__ == "__main__":
    main()
