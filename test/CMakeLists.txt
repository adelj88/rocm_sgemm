# Create an executable target
add_executable(test_float gemm/test.cpp)

# Include HIP include directories
target_include_directories(test_float PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(test_float PRIVATE rocm_sgemm GTest::GTest GTest::Main)

# Create an executable target
add_executable(test_rocblas rocblas/test.cpp)

# Include HIP include directories
target_include_directories(test_rocblas PRIVATE ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(test_rocblas PRIVATE rocblas_wrapper GTest::GTest GTest::Main)
