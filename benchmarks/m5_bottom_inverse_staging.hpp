#pragma once

#include <cstddef>
#include <vector>

namespace gfss {

struct M5BottomInverseStagingResult {
    std::vector<float> cublas_x;
    std::vector<float> custom_x;
    double cublas_median_ms{0.0};
    double cublas_best_ms{0.0};
    double custom_median_ms{0.0};
    double custom_best_ms{0.0};
    std::size_t inverse_bytes{0U};
    std::size_t device_bytes{0U};
};

// inverse_col_major is an explicitly symmetric FP32 inverse of the tiny SPD
// bottom operator. Both candidates keep it resident on device; setup/H2D/final
// D2H are excluded from timings.
M5BottomInverseStagingResult benchmark_m5_bottom_inverse_apply(
    const std::vector<float>& inverse_col_major,
    std::size_t n,
    const std::vector<float>& rhs,
    int repeats = 200);

}  // namespace gfss
