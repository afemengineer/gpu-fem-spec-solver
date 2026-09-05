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

# Persistent recursive coarse-tail runtime validation.  The host builder exports
# each non-bottom operator as dense FP32 or structural scalar CSR FP32 according
# to the representation policy, and exports both P and P^T as CSR.
add_executable(gfss_gpu_m5_recursive_tail_gpu_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_recursive_tail_gpu_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_recursive_tail.cu)
target_link_libraries(gfss_gpu_m5_recursive_tail_gpu_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_recursive_tail_gpu_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_recursive_tail_gpu_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

# End-to-end production-candidate gate.  This keeps the validated L0/P0/L1/P1
# persistent path but replaces fixed L2/L3 with the arbitrary-depth recursive
# device context, and runs it inside the existing FP64 defect-correction policy.
add_executable(gfss_gpu_m5_recursive_production_solver_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_recursive_production_solver_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_persistent_recursive_vpcg_staging.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_recursive_tail_context.cu)
target_link_libraries(gfss_gpu_m5_recursive_production_solver_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_recursive_production_solver_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_recursive_production_solver_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

# First genuinely non-box topology gate.  Active parent-grid HEX8 elements are
# compacted into orphan-style arbitrary coordinates/connectivity, so aggregation
# sees only a generic NodalGraph3D.  The dense A1 solve is reference-only: this
# target tests topology sensitivity before generalizing the production CUDA fine
# operator/importer.
add_executable(gfss_m5_active_hex_topology_reference_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_active_hex_topology_reference_bench.cpp)
target_link_libraries(gfss_m5_active_hex_topology_reference_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_m5_active_hex_topology_reference_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

# Diagnostic companion to the frozen topology gate.  It decomposes the residual
# after pre-smoothing, exact coarse correction, and post-smoothing; compares the
# one-step smoothed transfer with the raw tentative transfer; and reports the
# geometry of any locally rank-deficient aggregates without changing parameters.
add_executable(gfss_m5_active_hex_topology_diagnostic_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_active_hex_topology_diagnostic_bench.cpp)
target_link_libraries(gfss_m5_active_hex_topology_diagnostic_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_m5_active_hex_topology_diagnostic_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

# PCG relevance test for the topology diagnostic.  The shallow exact two-grid
# cycle is used as a preconditioner and compared with tentative-transfer and
# Jacobi controls.  This distinguishes harmless Euclidean residual amplification
# inside a Galerkin cycle from genuine Krylov/preconditioner degradation.
add_executable(gfss_m5_active_hex_topology_pcg_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_active_hex_topology_pcg_bench.cpp)
target_link_libraries(gfss_m5_active_hex_topology_pcg_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_m5_active_hex_topology_pcg_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)
