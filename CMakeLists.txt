cmake_minimum_required(VERSION 3.24)

project(gpu_fem_spec_solver VERSION 0.1.0 LANGUAGES CXX)

option(GFSS_ENABLE_CUDA "Build CUDA instrumentation and GPU solver components" ON)
option(GFSS_BUILD_TESTS "Build unit tests" ON)
option(GFSS_CPU_AVX2 "Compile optimized CPU stencil/Gold sources with AVX2/FMA" OFF)
option(GFSS_CPU_VEC_REPORT "Emit MSVC automatic-vectorization diagnostics for CPU stencil sources" OFF)
option(GFSS_CPU_FAST_FP "Compile optimized CPU stencil/Gold sources with fast floating-point semantics" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(gfss_core STATIC
    src/memory_model.cpp
    src/hex8.cpp
    src/structured_hex_mesh.cpp
    src/cpu_elasticity.cpp
    src/cpu_solver.cpp
    src/cpu_stencil.cpp
    src/cpu_gold.cpp
    src/cpu_gold_padded.cpp
    src/cpu_gold_window.cpp
)

target_include_directories(gfss_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gfss_core PUBLIC cxx_std_17)

if(GFSS_CPU_AVX2)
    target_compile_definitions(gfss_core PUBLIC GFSS_CPU_STENCIL_AVX2=1)
else()
    target_compile_definitions(gfss_core PUBLIC GFSS_CPU_STENCIL_AVX2=0)
endif()

if(MSVC)
    target_compile_options(gfss_core PRIVATE /W4 /permissive-)

    set(GFSS_CPU_OPT_COMPILE_OPTIONS "")
    if(GFSS_CPU_AVX2)
        list(APPEND GFSS_CPU_OPT_COMPILE_OPTIONS /arch:AVX2)
        message(STATUS "CPU stencil/Gold AVX2+FMA code generation enabled")
    else()
        message(STATUS "CPU stencil/Gold using default MSVC x64 ISA target")
    endif()

    if(GFSS_CPU_FAST_FP)
        list(APPEND GFSS_CPU_OPT_COMPILE_OPTIONS /fp:fast)
        message(STATUS "CPU stencil/Gold fast floating-point semantics enabled")
    endif()

    if(GFSS_CPU_VEC_REPORT)
        list(APPEND GFSS_CPU_OPT_COMPILE_OPTIONS /Qvec-report:2)
        message(STATUS "MSVC CPU vectorization diagnostics enabled")
    endif()

    if(GFSS_CPU_OPT_COMPILE_OPTIONS)
        set_source_files_properties(src/cpu_stencil.cpp src/cpu_gold.cpp src/cpu_gold_padded.cpp src/cpu_gold_window.cpp PROPERTIES
            COMPILE_OPTIONS "${GFSS_CPU_OPT_COMPILE_OPTIONS}")
    endif()
else()
    target_compile_options(gfss_core PRIVATE -Wall -Wextra -Wpedantic)

    set(GFSS_CPU_OPT_COMPILE_OPTIONS "")
    if(GFSS_CPU_AVX2)
        list(APPEND GFSS_CPU_OPT_COMPILE_OPTIONS -mavx2 -mfma)
        message(STATUS "CPU stencil/Gold AVX2+FMA code generation enabled")
    endif()
    if(GFSS_CPU_FAST_FP)
        list(APPEND GFSS_CPU_OPT_COMPILE_OPTIONS -ffast-math)
        message(STATUS "CPU stencil/Gold fast floating-point semantics enabled")
    endif()
    if(GFSS_CPU_OPT_COMPILE_OPTIONS)
        set_source_files_properties(src/cpu_stencil.cpp src/cpu_gold.cpp src/cpu_gold_padded.cpp src/cpu_gold_window.cpp PROPERTIES
            COMPILE_OPTIONS "${GFSS_CPU_OPT_COMPILE_OPTIONS}")
    endif()
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gfss_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(gfss_core PUBLIC GFSS_HAS_OPENMP=1)
    message(STATUS "OpenMP CPU baseline enabled")
else()
    target_compile_definitions(gfss_core PUBLIC GFSS_HAS_OPENMP=0)
    message(WARNING "OpenMP was not found; CPU parallel operator will fall back to serial")
endif()

add_library(gfss_multigrid_reference STATIC
    src/multigrid_reference.cpp
    src/multilevel_reference.cpp
    src/aggregation_coarse_space.cpp
    src/aggregation_coarse_operator.cpp
    src/aggregation_two_grid_reference.cpp
    src/smoothed_aggregation_reference.cpp
)
target_link_libraries(gfss_multigrid_reference PUBLIC gfss_core)
target_include_directories(gfss_multigrid_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(GFSS_APP_SOURCES src/main.cpp)

if(GFSS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)

        if(WIN32)
            get_filename_component(GFSS_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
            get_filename_component(GFSS_CUDA_ROOT "${GFSS_CUDA_BIN_DIR}" DIRECTORY)
            set(GFSS_CUBLAS_LIBRARY "${GFSS_CUDA_ROOT}/lib/x64/cublas.lib")
            if(NOT EXISTS "${GFSS_CUBLAS_LIBRARY}")
                message(FATAL_ERROR
                    "cuBLAS import library not found at ${GFSS_CUBLAS_LIBRARY}; "
                    "nvcc is ${CMAKE_CUDA_COMPILER}")
            endif()
        else()
            find_library(GFSS_CUBLAS_LIBRARY
                NAMES cublas
                HINTS "${CUDAToolkit_LIBRARY_DIR}"
                REQUIRED)
        endif()
        message(STATUS "cuBLAS library: ${GFSS_CUBLAS_LIBRARY}")

        list(APPEND GFSS_APP_SOURCES src/gpu_info.cu)
        set(GFSS_HAS_CUDA TRUE)
    else()
        message(WARNING "GFSS_ENABLE_CUDA=ON but no CUDA compiler was found; building CPU-only instrumentation.")
        list(APPEND GFSS_APP_SOURCES src/gpu_info_stub.cpp)
        set(GFSS_HAS_CUDA FALSE)
    endif()
else()
    list(APPEND GFSS_APP_SOURCES src/gpu_info_stub.cpp)
    set(GFSS_HAS_CUDA FALSE)
endif()

add_executable(gfss ${GFSS_APP_SOURCES})
target_link_libraries(gfss PRIVATE gfss_core)
target_include_directories(gfss PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(GFSS_HAS_CUDA)
    target_compile_definitions(gfss PRIVATE GFSS_HAS_CUDA=1)
    set_target_properties(gfss PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    add_library(gfss_cuda_operator STATIC
        src/gpu_elasticity.cu
        src/gpu_gold_3d.cu
        src/gpu_gold_sparse.cu
        src/gpu_pcg_persistent.cu
        src/gpu_pcg_seeded.cu
    )
    target_link_libraries(gfss_cuda_operator PUBLIC gfss_core "${GFSS_CUBLAS_LIBRARY}")
    target_include_directories(gfss_cuda_operator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(gfss_cuda_operator PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

    add_library(gfss_mixed_refinement STATIC
        src/mixed_refinement.cpp
    )
    target_link_libraries(gfss_mixed_refinement PUBLIC gfss_cuda_operator gfss_core)
    target_include_directories(gfss_mixed_refinement PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(gfss_cuda_operator_bench benchmarks/cuda_operator_bench.cpp)
    target_link_libraries(gfss_cuda_operator_bench PRIVATE gfss_cuda_operator gfss_core)
    target_include_directories(gfss_cuda_operator_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(gfss_gpu_sparse_profile benchmarks/gpu_sparse_profile.cpp)
    target_link_libraries(gfss_gpu_sparse_profile PRIVATE gfss_cuda_operator gfss_core)
    target_include_directories(gfss_gpu_sparse_profile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(gfss_gpu_pcg_bench benchmarks/gpu_pcg_bench.cpp)
    target_link_libraries(gfss_gpu_pcg_bench PRIVATE gfss_cuda_operator gfss_core)
    target_include_directories(gfss_gpu_pcg_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(gfss_gpu_mixed_refinement_bench benchmarks/gpu_mixed_refinement_bench.cpp)
    target_link_libraries(gfss_gpu_mixed_refinement_bench PRIVATE gfss_mixed_refinement)
    target_include_directories(gfss_gpu_mixed_refinement_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(gfss_gpu_adaptive_refinement_bench benchmarks/gpu_adaptive_refinement_bench.cpp)
    target_link_libraries(gfss_gpu_adaptive_refinement_bench PRIVATE gfss_mixed_refinement)
    target_include_directories(gfss_gpu_adaptive_refinement_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(gfss_gpu_adaptive_problem_suite_bench benchmarks/gpu_adaptive_problem_suite_bench.cpp)
    target_link_libraries(gfss_gpu_adaptive_problem_suite_bench PRIVATE gfss_mixed_refinement)
    target_include_directories(gfss_gpu_adaptive_problem_suite_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(gfss_gpu_two_level_seed_bench benchmarks/gpu_two_level_seed_bench.cpp)
    target_link_libraries(gfss_gpu_two_level_seed_bench PRIVATE gfss_cuda_operator gfss_core)
    target_include_directories(gfss_gpu_two_level_seed_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

    add_executable(gfss_gpu_two_level_consistency_bench benchmarks/gpu_two_level_consistency_bench.cpp)
    target_link_libraries(gfss_gpu_two_level_consistency_bench PRIVATE gfss_cuda_operator gfss_core)
    target_include_directories(gfss_gpu_two_level_consistency_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
else()
    target_compile_definitions(gfss PRIVATE GFSS_HAS_CUDA=0)
endif()

add_executable(gfss_cpu_operator_bench benchmarks/cpu_operator_bench.cpp)
target_link_libraries(gfss_cpu_operator_bench PRIVATE gfss_core)
target_include_directories(gfss_cpu_operator_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_cpu_stencil_bench benchmarks/cpu_stencil_bench.cpp)
target_link_libraries(gfss_cpu_stencil_bench PRIVATE gfss_core)
target_include_directories(gfss_cpu_stencil_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_cpu_gold_bench benchmarks/cpu_gold_bench.cpp)
target_link_libraries(gfss_cpu_gold_bench PRIVATE gfss_core)
target_include_directories(gfss_cpu_gold_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_cpu_gold_profile benchmarks/cpu_gold_profile.cpp)
target_link_libraries(gfss_cpu_gold_profile PRIVATE gfss_core)
target_include_directories(gfss_cpu_gold_profile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_geometric_vcycle_reference_bench benchmarks/geometric_vcycle_reference_bench.cpp)
target_link_libraries(gfss_geometric_vcycle_reference_bench PRIVATE gfss_multigrid_reference)
target_include_directories(gfss_geometric_vcycle_reference_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_aggregation_coarse_space_bench benchmarks/aggregation_coarse_space_bench.cpp)
target_link_libraries(gfss_aggregation_coarse_space_bench PRIVATE gfss_multigrid_reference)
target_include_directories(gfss_aggregation_coarse_space_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_aggregation_coarse_operator_bench benchmarks/aggregation_coarse_operator_bench.cpp)
target_link_libraries(gfss_aggregation_coarse_operator_bench PRIVATE gfss_multigrid_reference)
target_include_directories(gfss_aggregation_coarse_operator_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_aggregation_coarse_storage_bench benchmarks/aggregation_coarse_storage_bench.cpp)
target_link_libraries(gfss_aggregation_coarse_storage_bench PRIVATE gfss_multigrid_reference)
target_include_directories(gfss_aggregation_coarse_storage_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_aggregation_two_grid_reference_bench benchmarks/aggregation_two_grid_reference_bench.cpp)
target_link_libraries(gfss_aggregation_two_grid_reference_bench PRIVATE gfss_multigrid_reference)
target_include_directories(gfss_aggregation_two_grid_reference_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_smoothed_aggregation_reference_bench benchmarks/smoothed_aggregation_reference_bench.cpp)
target_link_libraries(gfss_smoothed_aggregation_reference_bench PRIVATE gfss_multigrid_reference)
target_include_directories(gfss_smoothed_aggregation_reference_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(GFSS_BUILD_TESTS)
    enable_testing()

    add_executable(gfss_memory_model_test tests/memory_model_test.cpp)
    target_link_libraries(gfss_memory_model_test PRIVATE gfss_core)
    add_test(NAME memory_model COMMAND gfss_memory_model_test)

    add_executable(gfss_hex8_test tests/hex8_test.cpp)
    target_link_libraries(gfss_hex8_test PRIVATE gfss_core)
    add_test(NAME hex8_reference COMMAND gfss_hex8_test)

    add_executable(gfss_cpu_operator_test tests/cpu_operator_test.cpp)
    target_link_libraries(gfss_cpu_operator_test PRIVATE gfss_core)
    add_test(NAME cpu_operator COMMAND gfss_cpu_operator_test)

    add_executable(gfss_cpu_solver_test tests/cpu_solver_test.cpp)
    target_link_libraries(gfss_cpu_solver_test PRIVATE gfss_core)
    add_test(NAME cpu_solver COMMAND gfss_cpu_solver_test)

    add_executable(gfss_cpu_stencil_test tests/cpu_stencil_test.cpp)
    target_link_libraries(gfss_cpu_stencil_test PRIVATE gfss_core)
    add_test(NAME cpu_stencil COMMAND gfss_cpu_stencil_test)

    add_executable(gfss_cpu_gold_test tests/cpu_gold_test.cpp)
    target_link_libraries(gfss_cpu_gold_test PRIVATE gfss_core)
    add_test(NAME cpu_gold COMMAND gfss_cpu_gold_test)

    add_executable(gfss_multilevel_reference_test tests/multilevel_reference_test.cpp)
    target_link_libraries(gfss_multilevel_reference_test PRIVATE gfss_multigrid_reference)
    add_test(NAME multilevel_reference COMMAND gfss_multilevel_reference_test)

    add_executable(gfss_aggregation_coarse_space_test tests/aggregation_coarse_space_test.cpp)
    target_link_libraries(gfss_aggregation_coarse_space_test PRIVATE gfss_multigrid_reference)
    add_test(NAME aggregation_coarse_space COMMAND gfss_aggregation_coarse_space_test)

    add_executable(gfss_aggregation_coarse_operator_test tests/aggregation_coarse_operator_test.cpp)
    target_link_libraries(gfss_aggregation_coarse_operator_test PRIVATE gfss_multigrid_reference)
    add_test(NAME aggregation_coarse_operator COMMAND gfss_aggregation_coarse_operator_test)

    add_executable(gfss_aggregation_two_grid_reference_test tests/aggregation_two_grid_reference_test.cpp)
    target_link_libraries(gfss_aggregation_two_grid_reference_test PRIVATE gfss_multigrid_reference)
    add_test(NAME aggregation_two_grid_reference COMMAND gfss_aggregation_two_grid_reference_test)

    add_executable(gfss_smoothed_aggregation_reference_test tests/smoothed_aggregation_reference_test.cpp)
    target_link_libraries(gfss_smoothed_aggregation_reference_test PRIVATE gfss_multigrid_reference)
    add_test(NAME smoothed_aggregation_reference COMMAND gfss_smoothed_aggregation_reference_test)

    if(GFSS_HAS_CUDA)
        add_executable(gfss_cuda_operator_test tests/cuda_operator_test.cpp)
        target_link_libraries(gfss_cuda_operator_test PRIVATE gfss_cuda_operator gfss_core)
        add_test(NAME cuda_operator COMMAND gfss_cuda_operator_test)

        add_executable(gfss_gpu_pcg_test tests/gpu_pcg_test.cpp)
        target_link_libraries(gfss_gpu_pcg_test PRIVATE gfss_cuda_operator gfss_core)
        add_test(NAME gpu_pcg COMMAND gfss_gpu_pcg_test)

        add_executable(gfss_gpu_pcg_seeded_test tests/gpu_pcg_seeded_test.cpp)
        target_link_libraries(gfss_gpu_pcg_seeded_test PRIVATE gfss_cuda_operator gfss_core)
        add_test(NAME gpu_pcg_seeded COMMAND gfss_gpu_pcg_seeded_test)

        add_executable(gfss_adaptive_refinement_test tests/adaptive_refinement_test.cpp)
        target_link_libraries(gfss_adaptive_refinement_test PRIVATE gfss_mixed_refinement)
        add_test(NAME adaptive_refinement COMMAND gfss_adaptive_refinement_test)
    endif()
endif()
