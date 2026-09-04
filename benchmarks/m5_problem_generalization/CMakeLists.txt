cmake_minimum_required(VERSION 3.24)
project(gfss_m5_problem_generalization LANGUAGES CXX CUDA)

set(GFSS_ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(GFSS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GFSS_CPU_AVX2 ON CACHE BOOL "" FORCE)
set(GFSS_CPU_FAST_FP ON CACHE BOOL "" FORCE)

find_package(CUDAToolkit REQUIRED)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. gfss_repo)

if(NOT TARGET gfss_cuda_operator)
    message(FATAL_ERROR "M5 generalization suite requires a working CUDA compiler")
endif()

target_sources(gfss_cuda_operator PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_fine_level.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_l2_materialized.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_l2_full_shell.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_bottom_solve.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_rectangular_csr_only.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_block6_transfer_optimized.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_block6_transpose_soa.cu)

add_executable(gfss_gpu_m5_problem_generalization_suite_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_problem_generalization_suite_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_persistent_vpcg_staging.cu)
target_link_libraries(gfss_gpu_m5_problem_generalization_suite_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_problem_generalization_suite_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_problem_generalization_suite_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

# Setup/reference probe for the scalable coarse architecture.  The validated
# L0->L2 construction feeds an exact support-pruned sparse recursive tail; no
# CUDA runtime changes are required for this decision benchmark.
add_executable(gfss_gpu_m5_recursive_sparse_tail_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_recursive_sparse_tail_bench.cpp)
target_link_libraries(gfss_gpu_m5_recursive_sparse_tail_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_gpu_m5_recursive_sparse_tail_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)
