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
# The rectangular transfer experiment is split into scalar CSR and optimized
# block6 implementations so the latter can evolve without duplicating symbols.
target_sources(gfss_cuda_operator PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_fine_level.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_l2_materialized.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_rectangular_csr_only.cu
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_block6_transfer_optimized.cu)

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
# localized A2 used by the frozen reference hierarchy and benchmarks both CSR
# and dense FP32 materializations against the factorized m1=2 lower bound.
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

# L1<->L2 transfer representation decision. CPU setup builds the exact smoothed
# P1 supports from the frozen hierarchy, assembles explicit structural CSR for
# both P1 and P1^T, audits against the factorized FP64 transfer, and benchmarks
# the device-resident rectangular matvecs against the strict m1=2 A1 lower bound.
add_executable(gfss_gpu_m5_l1_l2_transfer_representation_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l1_l2_transfer_representation_bench.cpp)
target_link_libraries(gfss_gpu_m5_l1_l2_transfer_representation_bench PRIVATE
    gfss_cuda_operator
    gfss_multigrid_reference
    CUDA::cudart
    CUDA::cublas)
target_include_directories(gfss_gpu_m5_l1_l2_transfer_representation_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)

# Natural 6x6 block representation of the same exact P1. The FP32 block values
# are stored once; P1 and P1^T use separate lightweight index structures. This
# benchmark compares memory and runtime directly with the passed dual scalar-CSR
# path and with the strict factorized m1=2 lower bound.
add_executable(gfss_gpu_m5_l1_l2_block_transfer_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_l1_l2_block_transfer_bench.cpp)
target_link_libraries(gfss_gpu_m5_l1_l2_block_transfer_bench PRIVATE
    gfss_cuda_operator
    gfss_multigrid_reference
    CUDA::cudart
    CUDA::cublas)
target_include_directories(gfss_gpu_m5_l1_l2_block_transfer_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}/..)
# The block benchmark includes the scalar benchmark only for its reusable CPU
# assembly/oracle helpers. Mark this target so the transfer header restores the
# scalar driver's renamed main after the recursive helper stack undefines it.
target_compile_definitions(gfss_gpu_m5_l1_l2_block_transfer_bench PRIVATE
    GFSS_M5_BLOCK_TRANSFER_BENCH=1)
