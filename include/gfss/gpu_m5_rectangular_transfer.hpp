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

}  // namespace gfss
