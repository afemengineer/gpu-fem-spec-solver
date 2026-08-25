cmake_minimum_required(VERSION 3.24)

project(gpu_fem_spec_solver VERSION 0.1.0 LANGUAGES CXX)

option(GFSS_ENABLE_CUDA "Build CUDA instrumentation and GPU solver components" ON)
option(GFSS_BUILD_TESTS "Build unit tests" ON)
option(GFSS_CPU_AVX2 "Compile the optimized CPU stencil source with AVX2 on MSVC" OFF)
option(GFSS_CPU_VEC_REPORT "Emit MSVC automatic-vectorization diagnostics for the CPU stencil source" OFF)
option(GFSS_CPU_FAST_FP "Compile the optimized CPU stencil source with /fp:fast on MSVC" OFF)

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
)

target_include_directories(gfss_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(gfss_core PUBLIC cxx_std_17)

if(MSVC)
    target_compile_options(gfss_core PRIVATE /W4 /permissive-)

    set(GFSS_CPU_STENCIL_COMPILE_OPTIONS "")
    if(GFSS_CPU_AVX2)
        list(APPEND GFSS_CPU_STENCIL_COMPILE_OPTIONS /arch:AVX2)
        target_compile_definitions(gfss_core PUBLIC GFSS_CPU_STENCIL_AVX2=1)
        message(STATUS "CPU stencil AVX2 code generation enabled")
    else()
        target_compile_definitions(gfss_core PUBLIC GFSS_CPU_STENCIL_AVX2=0)
        message(STATUS "CPU stencil using default MSVC x64 ISA target")
    endif()

    if(GFSS_CPU_FAST_FP)
        list(APPEND GFSS_CPU_STENCIL_COMPILE_OPTIONS /fp:fast)
        message(STATUS "CPU stencil fast floating-point semantics enabled")
    endif()

    if(GFSS_CPU_VEC_REPORT)
        list(APPEND GFSS_CPU_STENCIL_COMPILE_OPTIONS /Qvec-report:2)
        message(STATUS "MSVC CPU stencil vectorization diagnostics enabled")
    endif()

    if(GFSS_CPU_STENCIL_COMPILE_OPTIONS)
        set_source_files_properties(src/cpu_stencil.cpp PROPERTIES
            COMPILE_OPTIONS "${GFSS_CPU_STENCIL_COMPILE_OPTIONS}")
    endif()
else()
    target_compile_options(gfss_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_definitions(gfss_core PUBLIC GFSS_CPU_STENCIL_AVX2=0)
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

set(GFSS_APP_SOURCES src/main.cpp)

if(GFSS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
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

    add_library(gfss_cuda_operator STATIC src/gpu_elasticity.cu)
    target_link_libraries(gfss_cuda_operator PUBLIC gfss_core)
    target_include_directories(gfss_cuda_operator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(gfss_cuda_operator PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

    add_executable(gfss_cuda_operator_bench benchmarks/cuda_operator_bench.cpp)
    target_link_libraries(gfss_cuda_operator_bench PRIVATE gfss_cuda_operator gfss_core)
    target_include_directories(gfss_cuda_operator_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
else()
    target_compile_definitions(gfss PRIVATE GFSS_HAS_CUDA=0)
endif()

add_executable(gfss_cpu_operator_bench benchmarks/cpu_operator_bench.cpp)
target_link_libraries(gfss_cpu_operator_bench PRIVATE gfss_core)
target_include_directories(gfss_cpu_operator_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(gfss_cpu_stencil_bench benchmarks/cpu_stencil_bench.cpp)
target_link_libraries(gfss_cpu_stencil_bench PRIVATE gfss_core)
target_include_directories(gfss_cpu_stencil_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

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

    if(GFSS_HAS_CUDA)
        add_executable(gfss_cuda_operator_test tests/cuda_operator_test.cpp)
        target_link_libraries(gfss_cuda_operator_test PRIVATE gfss_cuda_operator gfss_core)
        add_test(NAME cuda_operator COMMAND gfss_cuda_operator_test)
    endif()
endif()
