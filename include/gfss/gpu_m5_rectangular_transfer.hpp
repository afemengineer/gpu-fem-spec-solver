#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfss {

struct GpuM5RectangularTransferTiming {
    double median_ms{0.0};
    double best_ms{0.0};
};

struct GpuM5RectangularTransferResult {
    std::vector<float> y;
    GpuM5RectangularTransferTiming timing;
    std::size_t rows{0U};
    std::size_t cols{0U};
    std::size_t nnz{0U};
    std::size_t device_bytes{0U};
};

// Persistent rectangular scalar-CSR matvec microbenchmark used to evaluate
// explicit M5 inter-level transfers. H2D setup and final D2H are excluded from
// timing. For restriction, pass an independently assembled CSR representation
// of the transpose with rows/cols swapped; this avoids atomics in the timed path.
GpuM5RectangularTransferResult benchmark_m5_rectangular_csr(
    std::size_t rows,
    std::size_t cols,
    const std::vector<std::uint32_t>& row_offsets,
    const std::vector<std::uint32_t>& column_indices,
    const std::vector<float>& values,
    const std::vector<float>& x,
    int repeats = 100);

struct GpuM5Block6TransferResult {
    std::vector<float> forward_y_padded;
    std::vector<float> transpose_y_padded;
    GpuM5RectangularTransferTiming forward_timing;
    GpuM5RectangularTransferTiming transpose_timing;
    std::size_t block_rows{0U};
    std::size_t block_cols{0U};
    std::size_t block_nnz{0U};
    std::size_t device_bytes{0U};
};

// Persistent 6x6 block-sparse transfer benchmark. Values are stored exactly
// once in forward block-row order. A lightweight transpose index stores the
// source block row and the corresponding forward block id, so P and P^T share
// the same FP32 6x6 payload without atomics or duplicated values. Vectors are
// padded to six scalar entries per algebraic node; H2D/D2H are outside timing.
GpuM5Block6TransferResult benchmark_m5_block6_transfer(
    std::size_t block_rows,
    std::size_t block_cols,
    const std::vector<std::uint32_t>& forward_row_offsets,
    const std::vector<std::uint32_t>& forward_column_indices,
    const std::vector<float>& block_values_row_major_6x6,
    const std::vector<std::uint32_t>& transpose_column_offsets,
    const std::vector<std::uint32_t>& transpose_row_indices,
    const std::vector<std::uint32_t>& transpose_block_ids,
    const std::vector<float>& forward_x_padded,
    const std::vector<float>& transpose_x_padded,
    int repeats = 100);

}  // namespace gfss
