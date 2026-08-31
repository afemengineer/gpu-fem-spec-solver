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

// Persistent M5 fine-level execution context.
//
// The first production slice implements the upper half of the frozen 1x1
// recursive SA V-cycle:
//
//   x0 = 0
//   x0 <- degree-n Chebyshev(D0^-1 A0, b0)
//   r0 = b0 - A0 x0
//   r1 = P0^T r0
//
// where P0 = (I - omega0 D0^-1 A0)^m0 P0_tentative and P0^T uses the exact
// transpose (I - omega0 A0 D0^-1)^m0. Fine vectors and aggregation metadata
// remain resident on device for repeated applications. Host transfers are
// excluded from reported timings.
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

    std::size_t fine_dofs() const noexcept;
    std::size_t coarse_dofs() const noexcept;
    double transfer_omega() const noexcept;
    double smoother_lambda_max() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gfss
