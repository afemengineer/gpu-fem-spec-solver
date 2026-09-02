cmake_minimum_required(VERSION 3.24)
project(gfss_m5_gpu_reference LANGUAGES CXX CUDA)

set(GFSS_ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(GFSS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GFSS_CPU_AVX2 ON CACHE BOOL "" FORCE)
set(GFSS_CPU_FAST_FP ON CACHE BOOL "" FORCE)

find_package(CUDAToolkit REQUIRED)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. gfss_repo)

if(NOT TARGET gfss_cuda_operator)
    message(FATAL_ERROR "M5 GPU reference requires a working CUDA compiler")
endif()

target_sources(gfss_cuda_operator PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_fine_level.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_l2_materialized.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_l2_full_shell.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_bottom_solve.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_rectangular_csr_only.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_block6_transfer_optimized.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_block6_transpose_soa.cu)

add_executable(gfss_gpu_m5_fine_pre_restrict_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_fine_pre_restrict_bench.cpp)
target_link_libraries(gfss_gpu_m5_fine_pre_restrict_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart)
target_include_directories(gfss_gpu_m5_fine_pre_restrict_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_gpu_m5_fine_full_shell_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_fine_full_shell_bench.cpp)
target_link_libraries(gfss_gpu_m5_fine_full_shell_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart)
target_include_directories(gfss_gpu_m5_fine_full_shell_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_gpu_m5_l1_block_smoother_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l1_block_smoother_bench.cu)
target_link_libraries(gfss_gpu_m5_l1_block_smoother_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l1_block_smoother_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_l1_block_smoother_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gfss_gpu_m5_l1_full_shell_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l1_full_shell_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l1_full_shell_impl.cu)
target_link_libraries(gfss_gpu_m5_l1_full_shell_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l1_full_shell_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_l1_full_shell_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gfss_gpu_m5_l2_representation_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l2_representation_bench.cpp)
target_link_libraries(gfss_gpu_m5_l2_representation_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l2_representation_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(gfss_gpu_m5_l2_full_shell_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l2_full_shell_bench.cpp)
target_link_libraries(gfss_gpu_m5_l2_full_shell_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l2_full_shell_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(gfss_gpu_m5_l3_bottom_solve_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l3_bottom_solve_bench.cpp)
target_link_libraries(gfss_gpu_m5_l3_bottom_solve_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l3_bottom_solve_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(gfss_gpu_m5_l3_inverse_apply_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l3_inverse_apply_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_bottom_inverse_staging.cu)
target_link_libraries(gfss_gpu_m5_l3_inverse_apply_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l3_inverse_apply_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_l3_inverse_apply_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gfss_gpu_m5_l1_l2_transfer_representation_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l1_l2_transfer_representation_bench.cpp)
target_link_libraries(gfss_gpu_m5_l1_l2_transfer_representation_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l1_l2_transfer_representation_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(gfss_gpu_m5_l1_l2_block_transfer_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l1_l2_block_transfer_bench.cpp)
target_link_libraries(gfss_gpu_m5_l1_l2_block_transfer_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l1_l2_block_transfer_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)
target_compile_definitions(gfss_gpu_m5_l1_l2_block_transfer_bench PRIVATE
    GFSS_M5_BLOCK_TRANSFER_BENCH=1)

add_executable(gfss_gpu_m5_l1_l2_block_transpose_soa_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l1_l2_block_transpose_soa_bench.cpp)
target_link_libraries(gfss_gpu_m5_l1_l2_block_transpose_soa_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l1_l2_block_transpose_soa_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(gfss_gpu_m5_complete_vcycle_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_complete_vcycle_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_complete_vcycle_impl.cu)
target_link_libraries(gfss_gpu_m5_complete_vcycle_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_complete_vcycle_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_complete_vcycle_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gfss_gpu_m5_vcycle_pcg_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_vcycle_pcg_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_vcycle_pcg_impl.cu)
target_link_libraries(gfss_gpu_m5_vcycle_pcg_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_vcycle_pcg_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_vcycle_pcg_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gfss_gpu_m5_vcycle_defect_correction_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_vcycle_defect_correction_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_vcycle_pcg_impl.cu)
target_link_libraries(gfss_gpu_m5_vcycle_defect_correction_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_vcycle_defect_correction_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_vcycle_defect_correction_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gfss_gpu_m5_vcycle_defect_iteration_sweep_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_vcycle_defect_iteration_sweep_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_vcycle_pcg_impl.cu)
target_link_libraries(gfss_gpu_m5_vcycle_defect_iteration_sweep_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_vcycle_defect_iteration_sweep_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_vcycle_defect_iteration_sweep_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gfss_gpu_m5_vcycle_defect_persistent_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_vcycle_defect_persistent_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_persistent_vpcg_staging.cu)
target_link_libraries(gfss_gpu_m5_vcycle_defect_persistent_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_vcycle_defect_persistent_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_vcycle_defect_persistent_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

add_executable(gfss_gpu_m5_hierarchy_setup_profile_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_hierarchy_setup_profile_bench.cpp)
target_link_libraries(gfss_gpu_m5_hierarchy_setup_profile_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_gpu_m5_hierarchy_setup_profile_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(gfss_gpu_m5_hierarchy_fast_setup_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_hierarchy_fast_setup_bench.cpp)
target_link_libraries(gfss_gpu_m5_hierarchy_fast_setup_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_gpu_m5_hierarchy_fast_setup_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

add_executable(gfss_gpu_m5_fastsetup_persistent_solver_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_fastsetup_persistent_solver_bench.cpp
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_persistent_vpcg_staging.cu)
target_link_libraries(gfss_gpu_m5_fastsetup_persistent_solver_bench PRIVATE
    gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_fastsetup_persistent_solver_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/../../src
    ${CMAKE_CURRENT_LIST_DIR}/..)
set_target_properties(gfss_gpu_m5_fastsetup_persistent_solver_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

# Setup-only exact A1 materialization experiment. The temporary block-sparse A1
# is built from the same exact off-diagonal blocks already needed for strength,
# used to construct the deeper hierarchy, and discarded before GPU runtime.
add_executable(gfss_gpu_m5_hierarchy_materialized_a1_setup_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_hierarchy_materialized_a1_setup_bench.cpp)
target_link_libraries(gfss_gpu_m5_hierarchy_materialized_a1_setup_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_gpu_m5_hierarchy_materialized_a1_setup_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

# Exact A1 setup microbenchmark: compare the established serial element
# accumulation against thread-local OpenMP maps plus deterministic reduction.
add_executable(gfss_gpu_m5_a1_offdiagonal_parallel_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_a1_offdiagonal_parallel_bench.cpp)
target_link_libraries(gfss_gpu_m5_a1_offdiagonal_parallel_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_gpu_m5_a1_offdiagonal_parallel_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

# Current production hierarchy profiler after all accepted setup optimizations.
# Validation is disabled so this reports only required setup work.
add_executable(gfss_gpu_m5_hierarchy_current_profile_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_hierarchy_current_profile_bench.cpp)
target_link_libraries(gfss_gpu_m5_hierarchy_current_profile_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_gpu_m5_hierarchy_current_profile_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

# Exact A1 candidate: preindex aggregate pairs so the element hot loop writes
# compact thread-local block arrays instead of thread-local unordered_maps.
add_executable(gfss_gpu_m5_a1_indexed_parallel_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_a1_indexed_parallel_bench.cpp)
target_link_libraries(gfss_gpu_m5_a1_indexed_parallel_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_gpu_m5_a1_indexed_parallel_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

# Cached A1P1 candidate: keep the exact temporary A1 representation but replace
# output LocalColumns hash probes with reusable per-worker dense node scratch.
add_executable(gfss_gpu_m5_cached_a1p1_dense_scratch_bench
    ${CMAKE_CURRENT_LIST_DIR}/../m5_cached_a1p1_dense_scratch_bench.cpp)
target_link_libraries(gfss_gpu_m5_cached_a1p1_dense_scratch_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_gpu_m5_cached_a1p1_dense_scratch_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)
