#!/usr/bin/env bash
set -euo pipefail

# Temporary directory for benchmark outputs
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Default values
SHAPES=""
BATCH_COUNT=1
TITLE="FP32 SGEMM Performance Benchmarks"
MARKDOWN_OUTPUT="sgemm_performance.md"
PLOT_OUTPUT="sgemm_performance_plot.png"
NO_MARKDOWN=0
NO_PLOT=0
VERBOSE=0
DEBUG=0

usage() {
    cat << EOF
Usage: $0 --sgemm-bin PATH --rocblas-bin PATH --gpu NAME [OPTIONS]

Required:
  --sgemm-bin PATH         Path to rocm_sgemm benchmark binary
  --rocblas-bin PATH       Path to rocBLAS benchmark binary
  --gpu NAME               GPU name (e.g., "AMD Radeon RX 7900 GRE")

Optional:
  --shapes SHAPES          Colon-separated matrix shapes (e.g., "1024:2048:4096")
  --batch-count N          Batch count for benchmarks (default: 1)
  --rocblas-env VAR=VALUE  Environment variable for rocBLAS binary (e.g., "HSA_OVERRIDE_GFX_VERSION=11.0.0")
  --title TITLE            Report title
  --os NAME                Operating system (e.g., "Ubuntu 24.04.1 LTS")
  --rocm-version VERSION   ROCm version (e.g., "6.4.1")
  --markdown-output FILE   Output markdown file (default: sgemm_performance.md)
  --plot-output FILE       Output plot file (default: sgemm_performance_plot.png)
  --no-markdown            Skip markdown generation
  --no-plot                Skip plot generation
  --verbose                Print benchmark output to console
  --debug                  Print debug information for parsing

Examples:
  $0 --sgemm-bin ./build/bench/bench_float \\
     --rocblas-bin ./build/bench/bench_rocblas \\
     --gpu "AMD Radeon RX 7900 GRE" \\
     --os "Ubuntu 24.04.1 LTS" \\
     --rocm-version "6.4.1" \\
     --title "Square Matrix FP32 Performance Benchmarks" \\
     --markdown-output gfx1100_square.md \\
     --plot-output gfx1100_square.png

  $0 --sgemm-bin ./build/bench/bench_float \\
     --rocblas-bin ./build/bench/bench_rocblas \\
     --gpu "AMD Radeon RX 7900 GRE" \\
     --os "Ubuntu 24.04.1 LTS" \\
     --rocm-version "6.4.1" \\
     --title "Rectangle Matrix FP32 Performance Benchmarks" \\
     --markdown-output gfx1100_rectangle.md \\
     --plot-output gfx1100_rectangle.png \\
     --shapes "4096,4096,1024:8192,8192,1024:4096,2048,64:8192,4096,128"

  # With environment variable override for rocBLAS
  $0 --sgemm-bin ./build/bench/bench_float \\
     --rocblas-bin ./build/bench/bench_rocblas \\
     --rocblas-env "HSA_OVERRIDE_GFX_VERSION=11.0.0" \\
     --gpu "AMD Radeon RX 7900 GRE" \\
     --os "Ubuntu 24.04.1 LTS" \\
     --rocm-version "6.4.1"
EOF
    exit 1
}

# Parse command line arguments
SGEMM_BIN=""
ROCBLAS_BIN=""
GPU=""
OS=""
ROCM_VERSION=""
ROCBLAS_ENV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --sgemm-bin)
            SGEMM_BIN="$2"
            shift 2
            ;;
        --rocblas-bin)
            ROCBLAS_BIN="$2"
            shift 2
            ;;
        --rocblas-env)
            ROCBLAS_ENV="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --os)
            OS="$2"
            shift 2
            ;;
        --rocm-version)
            ROCM_VERSION="$2"
            shift 2
            ;;
        --shapes)
            SHAPES="$2"
            shift 2
            ;;
        --batch-count)
            BATCH_COUNT="$2"
            shift 2
            ;;
        --title)
            TITLE="$2"
            shift 2
            ;;
        --markdown-output)
            MARKDOWN_OUTPUT="$2"
            shift 2
            ;;
        --plot-output)
            PLOT_OUTPUT="$2"
            shift 2
            ;;
        --no-markdown)
            NO_MARKDOWN=1
            shift
            ;;
        --no-plot)
            NO_PLOT=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --debug)
            DEBUG=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SGEMM_BIN" ]] || [[ -z "$ROCBLAS_BIN" ]] || [[ -z "$GPU" ]]; then
    echo "Error: --sgemm-bin, --rocblas-bin, and --gpu are required"
    usage
fi

if [[ ! -f "$SGEMM_BIN" ]]; then
    echo "Error: SGEMM binary not found: $SGEMM_BIN"
    exit 1
fi

if [[ ! -f "$ROCBLAS_BIN" ]]; then
    echo "Error: rocBLAS binary not found: $ROCBLAS_BIN"
    exit 1
fi

# Temporary output files
SGEMM_OUTPUT="$TEMP_DIR/sgemm_output.txt"
ROCBLAS_OUTPUT="$TEMP_DIR/rocblas_output.txt"

# Build command for benchmark
build_cmd() {
    local binary=$1
    local -a cmd=("$binary")
    
    if [[ -n "$SHAPES" ]]; then
        cmd+=(--shapes "$SHAPES")
    fi
    
    if [[ "$BATCH_COUNT" != "1" ]]; then
        cmd+=(--batch_count "$BATCH_COUNT")
    fi
    
    printf '%q ' "${cmd[@]}"
}

# Run benchmark and save output
run_benchmark() {
    local name=$1
    local output_file=$2
    local cmd=$3
    local env_var=$4
    
    echo "================================================================================"
    echo "Running $name benchmark..."
    if [[ -n "$env_var" ]]; then
        echo "With environment: $env_var"
    fi
    echo "================================================================================"
    
    # Prepend environment variable if provided
    local full_cmd="$cmd"
    if [[ -n "$env_var" ]]; then
        full_cmd="$env_var $cmd"
    fi
    
    if [[ $VERBOSE -eq 1 ]]; then
        # Show output to console and save to file
        if eval "$full_cmd" 2>&1 | tee "$output_file"; then
            echo ""
        else
            echo "✗ $name benchmark failed"
            exit 1
        fi
    else
        # Just save to file
        echo "Command: $full_cmd"
        if eval "$full_cmd" > "$output_file" 2>&1; then
            echo "✓ $name benchmark completed"
        else
            echo "✗ $name benchmark failed"
            cat "$output_file"
            exit 1
        fi
    fi
    echo ""
}

# Run benchmarks
SGEMM_CMD=$(build_cmd "$SGEMM_BIN")
run_benchmark "SGEMM" "$SGEMM_OUTPUT" "$SGEMM_CMD" ""

ROCBLAS_CMD=$(build_cmd "$ROCBLAS_BIN")
run_benchmark "rocBLAS" "$ROCBLAS_OUTPUT" "$ROCBLAS_CMD" "$ROCBLAS_ENV"

# Build Python command
PYTHON_CMD=(python3 generate_report.py)
PYTHON_CMD+=(--sgemm-output "$SGEMM_OUTPUT")
PYTHON_CMD+=(--rocblas-output "$ROCBLAS_OUTPUT")
PYTHON_CMD+=(--gpu "$GPU")

if [[ -n "$OS" ]]; then
    PYTHON_CMD+=(--os "$OS")
fi

if [[ -n "$ROCM_VERSION" ]]; then
    PYTHON_CMD+=(--rocm-version "$ROCM_VERSION")
fi

if [[ "$TITLE" != "FP32 SGEMM Performance Benchmarks" ]]; then
    PYTHON_CMD+=(--title "$TITLE")
fi

if [[ "$MARKDOWN_OUTPUT" != "sgemm_performance.md" ]]; then
    PYTHON_CMD+=(--markdown-output "$MARKDOWN_OUTPUT")
fi

if [[ "$PLOT_OUTPUT" != "sgemm_performance_plot.png" ]]; then
    PYTHON_CMD+=(--plot-output "$PLOT_OUTPUT")
fi

if [[ $NO_MARKDOWN -eq 1 ]]; then
    PYTHON_CMD+=(--no-markdown)
fi

if [[ $NO_PLOT -eq 1 ]]; then
    PYTHON_CMD+=(--no-plot)
fi

if [[ $DEBUG -eq 1 ]]; then
    PYTHON_CMD+=(--debug)
fi

# Generate report
echo "================================================================================"
echo "Generating report..."
echo "================================================================================"
"${PYTHON_CMD[@]}"

echo ""
echo "================================================================================"
echo "Complete!"
echo "================================================================================"
