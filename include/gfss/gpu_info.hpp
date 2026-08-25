#pragma once

#include <cstdint>
#include <string>

namespace gfss {

struct GpuInfo {
    bool available = false;
    std::string name = "CPU-only build";
    int compute_major = 0;
    int compute_minor = 0;
    std::uint64_t total_global_memory = 0;
    std::uint64_t free_global_memory = 0;
    int multiprocessor_count = 0;
};

[[nodiscard]] GpuInfo query_gpu_info();

}  // namespace gfss
