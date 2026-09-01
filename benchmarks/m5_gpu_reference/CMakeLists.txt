cmake_minimum_required(VERSION 3.24)
project(gfss_m5_gpu_reference LANGUAGES CXX CUDA)

# GPU productionization staging project. Keep the new M5 CUDA source isolated
# from the root target graph until it has compiled and passed the FP64 oracle on
# the local reference GPU.
set(GFSS_ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(GFSS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GFSS_CPU_AVX2 ON CACHE BOOL "" FORCE)
set(GFSS_CPU_FAST_FP ON CACHE BOOL "" FORCE)

# This staging project now owns a standalone CUDA executable in addition to CXX
# executables that consume CUDA objects through gfss_cuda_operator. Declaring
# CUDA as a project language is required so CMake initializes the CUDA compiler
# rules (for example CMAKE_CUDA_COMPILE_OBJECT and _CMAKE_CUDA_WHOLE_FLAG).
# find_package(CUDAToolkit) alone only provides imported CUDA library targets.
find_package(CUDAToolkit REQUIRED)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. gfss_repo)

if(NOT TARGET gfss_cuda_operator)
    message(FATAL_ERROR "M5 GPU reference requires a working CUDA compiler")
endif()

# Stage the persistent M5 GPU productionization sources only in this reference
# build until each slice has compiled and passed its independent FP64 oracle.
target_sources(gfss_cuda_operator PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_fine_level.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_l2_materialized.cu)

add_executable(gfss_gpu_m5_fine_pre_restrict_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_fine_pre_restrict_bench.cpp)
target_link_libraries(gfss_gpu_m5_fine_pre_restrict_bench PRIVATE
    gfss_cuda_operator
    gfss_multigrid_reference
    CUDA::cudart)
target_include_directories(gfss_gpu_m5_fine_pre_restrict_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_gpu_m5_fine_full_shell_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_fine_full_shell_bench.cpp)
target_link_libraries(gfss_gpu_m5_fine_full_shell_bench PRIVATE
    gfss_cuda_operator
    gfss_multigrid_reference
    CUDA::cudart)
target_include_directories(gfss_gpu_m5_fine_full_shell_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

# Standalone CUDA TU for the first persistent L1 production slice. It includes
# gpu_smoothed_aggregation.cu directly so the experimental block smoother can
# reuse the context's private device-resident A1=P0^T A0 P0 pipeline. Do not
# link gfss_cuda_operator here or the included CUDA symbols would be duplicated.
add_executable(gfss_gpu_m5_l1_block_smoother_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l1_block_smoother_bench.cu)
target_link_libraries(gfss_gpu_m5_l1_block_smoother_bench PRIVATE
    gfss_multigrid_reference
    CUDA::cudart
    CUDA::cublas)
target_include_directories(gfss_gpu_m5_l1_block_smoother_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_l1_block_smoother_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

# Final-hierarchy L2 representation decision. CPU setup reconstructs the exact
# localized A2 used by the frozen reference hierarchy and compresses structural
# zeros to FP32 CSR. The GPU side benchmarks that CSR against a strict lower
# bound for the fully factorized m1=2 representation.
add_executable(gfss_gpu_m5_l2_representation_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l2_representation_bench.cpp)
target_link_libraries(gfss_gpu_m5_l2_representation_bench PRIVATE
    gfss_cuda_operator
    gfss_multigrid_reference
    CUDA::cudart
    CUDA::cublas)
target_include_directories(gfss_gpu_m5_l2_representation_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)
