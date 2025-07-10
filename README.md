# ROCm SGEMM

This repository provides a standalone, high-performance General Matrix Multiplication (GEMM) implementation optimized for AMD GPUs for single-precision floating-point operations (SGEMM).

## Purpose
This repository aims to:
- Provide a focused, high-performance GEMM kernel for single-precision floating-point operations (SGEMM).
- Explore and implement support for various matrix data layouts (e.g., row-major, column-major, potentially tiled formats).
- Tune the GEMM kernel for different M, N, K sizes

## Building the Project

### Prerequisites
- AMD ROCm installed with HIP support
- CMake version 3.10 or higher
- Python3 (required for config generation and tuning)
  - Python packages (can be installed with pip or conda)
    - ``numpy``
    - ``optuna``
- AMD RDNA GPU (code needs to be modified to support CDNA GPUs)

### Build Steps
1. Clone the repository:
   ```bash
   git https://github.com/adelj88/rocm_wmma_gemm.git
   cd rocm_wmma_gemm
   ```
2. Build:
   ```bash
   mkdir build
   cd build
   CXX=/opt/rocm/bin/hipcc cmake ..
   make
   ```

### Usage
Run the executable after building:
```bash
# Assumes you're currently in /build directory
# To run unit tests
./test/gemm_test

# To run unit benchmarks
./benchmark/gemm_bench

# To run rocblas equivalent for verification
./test/rocblas_test
./benchmark/rocblas_bench
```

### Automatic Kernel Tuning
The library includes an Optuna-based Tree-structured Parzen Estimator (TPE) tuner that automatically finds optimal kernel configurations for different matrix sizes and data layouts.

#### **Tuning Approach**
The tuner uses **Optuna TPE (Tree-structured Parzen Estimators)** to efficiently explore the discrete parameter space:

- **TPE optimization**: Models the performance landscape using probabilistic distributions to intelligently sample promising regions
- **Smart initialization**: Tests proven baseline configurations first to seed the optimization with known good solutions
- **Multivariate learning**: Understands relationships between parameters (e.g., block sizes and tile configurations)
- **Adaptive sampling**: Balances exploration of uncertain regions with exploitation of high-performing areas
- **Reproducible results**: Uses configurable random seeds for consistent and repeatable tuning runs

To run the tuner:
```bash
cd build

# Default behavior (all sizes and layouts)
python3 tune.py # Results written to gemm_config_tuned.json

# Test specific sizes
python3 tune.py --sizes 1024,1024,1024 2048,2048,2048

# Adjust evaluation budget
python3 tune.py --budget 100

# Test specific layouts
python3 tune.py --layouts r,c,r c,c,c

# Reproducible results with specific seed
python3 tune.py --seed 123

# Different GPU architecture
python3 tune.py --gpu-arch gfx1103

# Custom output file
python3 tune.py --output my_config.json

# Custom baseline configurations
python3 tune.py --baselines 128,128,128,8,4,4,2,4,8 256,128,128,8,2,2,4,4,4
```

## Performance Results
Below are benchmark results (in TFLOPs) that compares `rocm_wmma_gemm` against `rocblas` for all layouts and different sizes.

- [View detailed square matrix benchmarks](docs/square.md)

## Future Plans
1. Address lower performance for column-major output
   - A separate tuning profile is probably necessary, given that the current tuning script focuses on row-major outputs
2. Further tuning to get better performance
3. Explore any possibility of further optimizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
