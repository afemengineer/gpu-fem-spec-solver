#pragma once

#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace gfss {

struct GpuM5FinePreRestrictTiming {
    double zero_ms{0.0};
    double pre_smooth_ms{0.0};
    double residual_ms{0.0};
    double transfer_transpose_ms{0.0};
    double p0t_ms{0.0};
    double total_ms{0.0};
};

struct GpuM5FinePreRestrictResult {
    std::vector<float> coarse_residual;
    GpuM5FinePreRestrictTiming median_timing;
    GpuM5FinePreRestrictTiming best_timing;
    std::size_t smoother_degree{0};
    std::size_t transfer_smoothing_steps{0};
    std::size_t fine_operator_applies{0};
    std::size_t device_bytes_total{0};
    std::size_t fine_vector_bytes{0};
    std::size_t coarse_vector_bytes{0};
    std::size_t aggregation_metadata_bytes{0};
    std::size_t model_coordinate_bytes{0};
};

struct GpuM5FineFullShellTiming {
    double zero_ms{0.0};
    double pre_smooth_ms{0.0};
    double residual_ms{0.0};
    double transfer_transpose_ms{0.0};
    double p0t_ms{0.0};
    double p0_ms{0.0};
    double transfer_forward_ms{0.0};
    double correction_ms{0.0};
    double post_smooth_ms{0.0};
    double total_ms{0.0};
};

struct GpuM5FineFullShellResult {
    std::vector<float> coarse_residual;
    // Host-facing fine correction is AoS, matching the rest of the solver API.
    std::vector<float> fine_correction_aos;
    GpuM5FineFullShellTiming median_timing;
    GpuM5FineFullShellTiming best_timing;
    std::size_t smoother_degree{0};
    std::size_t transfer_smoothing_steps{0};
    std::size_t fine_operator_applies{0};
    std::size_t device_bytes_total{0};
    std::size_t fine_vector_bytes{0};
    std::size_t coarse_vector_bytes{0};
    std::size_t aggregation_metadata_bytes{0};
    std::size_t model_coordinate_bytes{0};
};

// Persistent M5 fine-level execution context.
//
// The validated upper-half operation is
//
//   x0 = 0
//   x0 <- degree-n Chebyshev(D0^-1 A0, b0)
//   r0 = b0 - A0 x0
//   r1 = P0^T r0
//
// and full_shell() extends it with an externally supplied L1 correction e1:
//
//   x0 <- x0 + P0 e1
//   x0 <- degree-n Chebyshev(D0^-1 A0, b0, initial=x0)
//
// where P0 = (I - omega0 D0^-1 A0)^m0 P0_tentative and P0^T uses the exact
// transpose (I - omega0 A0 D0^-1)^m0. Fine vectors, the external L1
// correction, and aggregation metadata remain resident on device throughout
// each timed application. Host transfers are excluded from reported timings.
class GpuM5FineLevelContext {
public:
    GpuM5FineLevelContext(
        const StructuredHexMesh& mesh,
        const Material& material,
        const ElasticityAggregationCoarseSpace& space,
        double transfer_omega,
        double smoother_lambda_max,
        int block_y = 4);
    ~GpuM5FineLevelContext();

    GpuM5FineLevelContext(const GpuM5FineLevelContext&) = delete;
    GpuM5FineLevelContext& operator=(const GpuM5FineLevelContext&) = delete;
    GpuM5FineLevelContext(GpuM5FineLevelContext&&) noexcept;
    GpuM5FineLevelContext& operator=(GpuM5FineLevelContext&&) noexcept;

    GpuM5FinePreRestrictResult pre_smooth_restrict(
        const std::vector<float>& rhs_aos,
        std::size_t smoother_degree,
        std::size_t transfer_smoothing_steps,
        int repeats = 50);

    // Execute the complete fine-level symmetric V-cycle shell around an
    // externally supplied L1 correction. This deliberately does not implement
    // the L1 solve yet; it provides the device-resident contract that the
    // recursive L1/L2 production path will plug into next.
    GpuM5FineFullShellResult full_shell(
        const std::vector<float>& rhs_aos,
        const std::vector<float>& coarse_correction,
        std::size_t smoother_degree,
        std::size_t transfer_smoothing_steps,
        int repeats = 50);

    std::size_t fine_dofs() const noexcept;
    std::size_t coarse_dofs() const noexcept;
    double transfer_omega() const noexcept;
    double smoother_lambda_max() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gfss
