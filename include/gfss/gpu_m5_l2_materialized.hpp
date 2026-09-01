#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfss {

struct GpuM5L2CsrTiming {
    double median_ms{0.0};
    double best_ms{0.0};
};

struct GpuM5L2CsrResult {
    std::vector<float> y;
    GpuM5L2CsrTiming timing;
    std::size_t rows{0U};
    std::size_t nnz{0U};
    std::size_t device_bytes{0U};
};

// Persistent scalar-CSR SpMV microbenchmark for the selectively materialized
// M5 L2 operator. H2D setup and final D2H are excluded from timings.
GpuM5L2CsrResult benchmark_m5_l2_csr(
    const std::vector<std::uint32_t>& row_offsets,
    const std::vector<std::uint32_t>& column_indices,
    const std::vector<float>& values,
    const std::vector<float>& x,
    int repeats = 100);

}  // namespace gfss
