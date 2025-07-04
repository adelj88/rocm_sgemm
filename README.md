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
The library includes a Genetic Algorithm-based tuner that automatically finds optimal kernel configurations for different matrix sizes and data layouts.

#### **Tuning Approach**
The tuner uses **Genetic Algorithm (GA)** to efficiently explore the discrete parameter space:

- **Population-based search**: Explores multiple configurations simultaneously through evolutionary generations
- **Smart initialization**: Seeds initial population with proven baseline configurations plus diverse random individuals
- **Tournament selection**: Selects high-performing configurations as parents for the next generation
- **Uniform crossover**: Combines successful parameter combinations from different parent configurations
- **Constraint-aware mutation**: Randomly explores new parameter values while respecting hardware constraints
- **Elitism**: Preserves the best configurations across generations to prevent loss of good solutions
- **Reproducible results**: Uses configurable random seeds for consistent and repeatable tuning runs

To run the tuner:
```bash
cd build
# Default behavior (all sizes and layouts)
python3 tune.py # Results written to gemm_config_tuned.json

# Test specific sizes
python3 tune.py --sizes 1024,1024,1024 2048,2048,2048

# Adjust evaluation budget
python3 tune.py --budget 80

# Test specific layouts
python3 tune.py --layouts r,c c,c

# Reproducible results with specific seed
python3 tune.py --seed 123

# Adjust GA parameters for different exploration
python3 tune.py --pop-size 30 --mutation-rate 0.4

# Different GPU architecture
python3 tune.py --gpu-arch gfx1103

# Custom output file
python3 tune.py --output my_config.json
```

## Performance Results
- [View detailed square matrix benchmarks](docs/square.md)

## Future Plans
1. Address lower performance for column-major output
   - A separate tuning profile is probably necessary, given that the current tuning script focuses on row-major outputs
2. Further tuning to get better performance
3. Explore any possibility of further optimizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
