# Create an executable target
add_executable(bench_float gemm/bench.cpp)

# Include HIP include directories
target_include_directories(bench_float PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(bench_float PRIVATE rocm_sgemm benchmark::benchmark)

# Create an executable target
add_executable(bench_rocblas rocblas/bench.cpp)

# Include HIP include directories
target_include_directories(bench_rocblas PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(bench_rocblas PRIVATE rocblas_wrapper benchmark::benchmark)
