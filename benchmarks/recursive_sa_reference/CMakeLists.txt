cmake_minimum_required(VERSION 3.24)
project(gfss_recursive_sa_reference LANGUAGES CXX)

# This numerical-reference benchmark is intentionally CPU-only. Keep it as a
# small standalone subproject so the experiment can be built without changing
# the production/root target graph while the recursive architecture is still
# being validated.
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
