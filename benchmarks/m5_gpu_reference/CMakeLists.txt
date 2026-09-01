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

# Standalone CUDA TU for the first persistent L1 production slice. It includes
# gpu_smoothed_aggregation.cu directly, so do not link gfss_cuda_operator.
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

# Complete staged L1 shell. The CUDA implementation includes
# gpu_smoothed_aggregation.cu directly, so keep it separate from the library.
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

# Final localized bottom level: FP64 setup/factor oracle with a persistent FP32
# one-warp triangular solve on device.
add_executable(gfss_gpu_m5_l3_bottom_solve_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l3_bottom_solve_bench.cpp)
target_link_libraries(gfss_gpu_m5_l3_bottom_solve_bench PRIVATE
    gfss_cuda_operator gfss_multigrid_reference CUDA::cudart CUDA::cublas)
target_include_directories(gfss_gpu_m5_l3_bottom_solve_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

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
