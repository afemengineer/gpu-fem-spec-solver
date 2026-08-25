#include "gfss/gpu_info.hpp"

#include <cuda_runtime.h>

namespace gfss {

GpuInfo query_gpu_info() {
    GpuInfo info;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return info;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return info;
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        free_bytes = 0;
        total_bytes = prop.totalGlobalMem;
    }

    info.available = true;
    info.name = prop.name;
    info.compute_major = prop.major;
    info.compute_minor = prop.minor;
    info.total_global_memory = static_cast<std::uint64_t>(total_bytes);
    info.free_global_memory = static_cast<std::uint64_t>(free_bytes);
    info.multiprocessor_count = prop.multiProcessorCount;
    return info;
}

}  // namespace gfss
