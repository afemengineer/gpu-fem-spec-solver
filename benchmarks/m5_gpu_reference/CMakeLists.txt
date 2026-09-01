cmake_minimum_required(VERSION 3.24)
project(gfss_m5_gpu_reference LANGUAGES CXX)

# GPU productionization staging project. Keep the new M5 CUDA source isolated
# from the root target graph until it has compiled and passed the FP64 oracle on
# the local reference GPU.
set(GFSS_ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(GFSS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GFSS_CPU_AVX2 ON CACHE BOOL "" FORCE)
set(GFSS_CPU_FAST_FP ON CACHE BOOL "" FORCE)

# The benchmark executable is a CXX target that consumes CUDA objects through
# the static gfss_cuda_operator archive. Link the CUDA runtime explicitly here;
# otherwise MSVC uses link.exe without CMake's CUDA link-driver injection and
# leaves cuda* / __cuda* runtime symbols unresolved.
find_package(CUDAToolkit REQUIRED)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../.. gfss_repo)

if(NOT TARGET gfss_cuda_operator)
    message(FATAL_ERROR "M5 GPU reference requires a working CUDA compiler")
endif()

# Stage the new persistent fine-level context only in this reference build.
target_sources(gfss_cuda_operator PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../src/gpu_m5_fine_level.cu)

add_executable(gfss_gpu_m5_fine_pre_restrict_bench
    ${CMAKE_CURRENT_LIST_DIR}/../gpu_m5_fine_pre_restrict_bench.cpp)
target_link_libraries(gfss_gpu_m5_fine_pre_restrict_bench PRIVATE
    gfss_cuda_operator
    gfss_multigrid_reference
    CUDA::cudart)
target_include_directories(gfss_gpu_m5_fine_pre_restrict_bench PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../include)
