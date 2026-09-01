#pragma once

#include <cstddef>
#include <vector>

namespace gfss {

struct GpuM5L2FullShellTiming {
    double pre_smooth_ms{0.0};
    double residual_a2_ms{0.0};
    double residual_update_ms{0.0};
    double p2t_ms{0.0};
    double p2_ms{0.0};
    double correction_ms{0.0};
    double post_a2_ms{0.0};
    double post_smooth_ms{0.0};
    double total_ms{0.0};
};

struct GpuM5L2FullShellResult {
    std::vector<float> l3_residual;
    std::vector<float> final_l2_correction;
    GpuM5L2FullShellTiming median_timing;
    GpuM5L2FullShellTiming best_timing;
    std::size_t l2_dofs{0U};
    std::size_t l3_dofs{0U};
    std::size_t l2_blocks{0U};
    std::size_t executed_a2_applies{0U};
    std::size_t device_bytes{0U};
};

// Complete persistent L2 symmetric V-cycle shell around an externally supplied
// L3 correction. A2 is a dense symmetric FP32 matrix. P2 is supplied once as a
// dense row-major n2 x n3 matrix; because cuBLAS is column-major the same payload
// represents P2^T directly, so restriction uses OP_N and prolongation OP_T.
// The actual L2 inverse block metric is one padded row-major 6x6 FP32 matrix per
// algebraic node. The degree-1 pre-smoother starts from zero, so A2*0 is elided
// exactly. H2D setup and final D2H are outside the timed shell.
GpuM5L2FullShellResult benchmark_m5_l2_full_shell(
    const std::vector<float>& a2_dense_row_major,
    std::size_t l2_dofs,
    const std::vector<float>& inverse_blocks_6x6,
    const std::vector<float>& p2_dense_row_major,
    std::size_t l3_dofs,
    const std::vector<float>& rhs_l2,
    const std::vector<float>& external_l3_correction,
    double lambda2,
    int repeats = 50);

}  // namespace gfss
