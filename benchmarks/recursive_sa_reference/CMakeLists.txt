cmake_minimum_required(VERSION 3.24)
project(gfss_recursive_sa_reference LANGUAGES CXX)

# These numerical-reference benchmarks are intentionally CPU-only. Keep them
# in a small standalone subproject so recursive-algorithm experiments do not
# perturb the production/root target graph while the hierarchy is still being
# validated.
set(GFSS_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(GFSS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GFSS_CPU_AVX2 ON CACHE BOOL "" FORCE)
set(GFSS_CPU_FAST_FP ON CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. gfss_repo)

add_executable(gfss_recursive_sa_reference_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_reference_bench.cpp)
target_link_libraries(gfss_recursive_sa_reference_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_reference_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_smoothed_reference_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_smoothed_reference_bench.cpp)
target_link_libraries(gfss_recursive_sa_smoothed_reference_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_smoothed_reference_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_actual_a2_reference_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_actual_a2_reference_bench.cpp)
target_link_libraries(gfss_recursive_sa_actual_a2_reference_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_actual_a2_reference_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_block_jacobi_reference_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_block_jacobi_reference_bench.cpp)
target_link_libraries(gfss_recursive_sa_block_jacobi_reference_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_block_jacobi_reference_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_l1_block_jacobi_reference_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_l1_block_jacobi_reference_bench.cpp)
target_link_libraries(gfss_recursive_sa_l1_block_jacobi_reference_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_l1_block_jacobi_reference_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_local_l2_reference_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_local_l2_reference_bench.cpp)
target_link_libraries(gfss_recursive_sa_local_l2_reference_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_local_l2_reference_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_actual_a1_strength_reference_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_actual_a1_strength_reference_bench.cpp)
target_link_libraries(gfss_recursive_sa_actual_a1_strength_reference_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_actual_a1_strength_reference_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_actual_a1_strength_sweep_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_actual_a1_strength_sweep_bench.cpp)
target_link_libraries(gfss_recursive_sa_actual_a1_strength_sweep_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_actual_a1_strength_sweep_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_strength_local_l2_reference_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_strength_local_l2_consistency_bench.cpp)
target_link_libraries(gfss_recursive_sa_strength_local_l2_reference_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_strength_local_l2_reference_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_compatible_relaxation_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_compatible_relaxation_bench.cpp)
target_link_libraries(gfss_recursive_sa_compatible_relaxation_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_compatible_relaxation_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_l3_cr_oracle_enrichment_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_l3_cr_oracle_enrichment_bench.cpp)
target_link_libraries(gfss_recursive_sa_l3_cr_oracle_enrichment_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_l3_cr_oracle_enrichment_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_block_cr_spectral_enrichment_sweep_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_block_cr_spectral_enrichment_sweep_entry.cpp)
target_link_libraries(gfss_recursive_sa_block_cr_spectral_enrichment_sweep_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_block_cr_spectral_enrichment_sweep_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_full_vcycle_spectral_diagnostic_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_full_vcycle_spectral_diagnostic_bench.cpp)
target_link_libraries(gfss_recursive_sa_full_vcycle_spectral_diagnostic_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_full_vcycle_spectral_diagnostic_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_gamma_cycle_sweep_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_gamma_cycle_sweep_bench.cpp)
target_link_libraries(gfss_recursive_sa_gamma_cycle_sweep_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_gamma_cycle_sweep_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_selective_gamma_sweep_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_selective_gamma_sweep_bench.cpp)
target_link_libraries(gfss_recursive_sa_selective_gamma_sweep_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_selective_gamma_sweep_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_selective_pcg_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_selective_pcg_bench.cpp)
target_link_libraries(gfss_recursive_sa_selective_pcg_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_selective_pcg_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)

add_executable(gfss_recursive_sa_smoothing_degree_pcg_sweep_bench
    ${CMAKE_CURRENT_LIST_DIR}/../recursive_sa_smoothing_degree_pcg_sweep_bench.cpp)
target_compile_definitions(gfss_recursive_sa_smoothing_degree_pcg_sweep_bench PRIVATE
    GFSS_RECURSIVE_SA_SELECTIVE_PCG_NO_MAIN=1)
target_link_libraries(gfss_recursive_sa_smoothing_degree_pcg_sweep_bench PRIVATE
    gfss_multigrid_reference)
target_include_directories(gfss_recursive_sa_smoothing_degree_pcg_sweep_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)
