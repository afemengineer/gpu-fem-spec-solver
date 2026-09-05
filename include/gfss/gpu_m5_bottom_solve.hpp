#pragma once

#include <cstddef>
#include <vector>

namespace gfss {

struct GpuM5BottomSolveResult {
    std::vector<float> x;
    double median_ms{0.0};
    double best_ms{0.0};
    std::size_t dofs{0U};
    std::size_t factor_bytes{0U};
    std::size_t device_bytes{0U};
};

// Persistent microbenchmark for the tiny SPD bottom system. The lower
// Cholesky factor is supplied row-major in FP32 and remains on device. One warp
// performs forward and backward substitution entirely on device; H2D setup and
// final D2H are excluded from timing.
GpuM5BottomSolveResult benchmark_m5_bottom_cholesky_solve(
    const std::vector<float>& lower_row_major,
    std::size_t n,
    const std::vector<float>& rhs,
    int repeats = 100);

}  // namespace gfss
