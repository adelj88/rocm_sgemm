# Create the config directory in the binary dir if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/rocm_sgemm/kernel)

# Add configuration generation
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/rocm_sgemm/kernel/config_generated.hpp
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/config/generate_config.py
            ${CMAKE_CURRENT_SOURCE_DIR}/config/gemm_config.json
            ${CMAKE_CURRENT_BINARY_DIR}/include/rocm_sgemm/kernel/config_generated.hpp
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/config/gemm_config.json
            ${CMAKE_CURRENT_SOURCE_DIR}/config/generate_config.py
    COMMENT "Generating GEMM configuration header"
)

add_custom_target(generate_gemm_config
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/include/rocm_sgemm/kernel/config_generated.hpp
)

set_source_files_properties(src/gemm.cpp PROPERTIES COMPILE_OPTIONS -mcumode)

add_library(rocm_sgemm STATIC src/gemm.cpp)
add_dependencies(rocm_sgemm generate_gemm_config)

# Include HIP include directories and generated headers
target_include_directories(rocm_sgemm PUBLIC
    ${HIP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include  # Add this for generated headers
)

# Link HIP libraries
target_link_libraries(rocm_sgemm PUBLIC ${HIP_LIBRARIES})

# Add optimization flags
target_compile_options(rocm_sgemm PRIVATE
    -O3                 # High optimization
    -ffast-math         # Fast math operations (may break strict IEEE compliance)
)

# Add tuner executable
add_executable(tuner config/tuner.cpp)

# Include HIP include directories and generated headers
target_include_directories(tuner PUBLIC
    ${HIP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include  # Add this for generated headers
)

# Link HIP libraries
target_link_libraries(tuner PUBLIC ${HIP_LIBRARIES} hiprtc benchmark::benchmark)

# Add optimization flags
target_compile_options(tuner PRIVATE
    -O3                 # High optimization
    -ffast-math         # Fast math operations (may break strict IEEE compliance)
)

target_compile_definitions(tuner PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

# Copy tune.py to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config/tune.py
    ${CMAKE_BINARY_DIR}/tune.py
    COPYONLY
)
