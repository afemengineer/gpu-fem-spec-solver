#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfss {

struct GpuM5L2Block6Timing {
    double median_ms{0.0};
    double best_ms{0.0};
};

struct GpuM5L2Block6Result {
    std::vector<float> y;
    GpuM5L2Block6Timing timing;
    std::size_t block_rows{0U};
    std::size_t block_nnz{0U};
    std::size_t device_bytes{0U};
};

// Persistent 6x6 block-CSR SpMV microbenchmark for M5 A2. Each stored block is
// row-major and the vector layout is six contiguous scalar DOFs per block row.
// H2D setup, warmup and final D2H are excluded from timings.
GpuM5L2Block6Result benchmark_m5_l2_block6_csr(
    const std::vector<std::uint32_t>& block_row_offsets,
    const std::vector<std::uint32_t>& block_column_indices,
    const std::vector<float>& block_values_row_major_6x6,
    const std::vector<float>& x,
    int repeats = 100);

}  // namespace gfss
