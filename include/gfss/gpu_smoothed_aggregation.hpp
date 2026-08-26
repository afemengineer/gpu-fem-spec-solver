#pragma once

#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace gfss {

struct GpuSmoothedAggregationTiming {
    double p0_ms{0.0};
    double fine_operator_ms{0.0};
    double jacobi_ms{0.0};
    double vector_update_ms{0.0};
    double p0t_ms{0.0};
    double total_ms{0.0};
};

struct GpuSmoothedAggregationApplyResult {
    std::vector<float> coarse_y;
    GpuSmoothedAggregationTiming median_timing;
    GpuSmoothedAggregationTiming best_timing;
    std::size_t transfer_smoothing_steps{0};
    std::size_t fine_operator_applies{0};
    std::size_t device_bytes_total{0};
    std::size_t fine_workspace_bytes{0};
    std::size_t coarse_workspace_bytes{0};
    std::size_t aggregation_metadata_bytes{0};
    std::size_t model_coordinate_bytes{0};
};

// Persistent first GPU implementation of
//
//   y_c = P_m^T A P_m x_c,
//   P_m = (I - omega D^{-1} A)^m P0.
//
// P0/P0^T are reconstructed from aggregation metadata; smoothed interpolation
// is never materialized. The context keeps all work vectors and metadata on the
// device across applies. The current structured-Q1 implementation deliberately
// reuses the existing GoldSparse/Jacobi kernels so stage timings are directly
// comparable with the M2/M3 operator work.
class GpuSmoothedAggregationContext {
public:
    GpuSmoothedAggregationContext(
        const StructuredHexMesh& mesh,
        const Material& material,
        const ElasticityAggregationCoarseSpace& space,
        double omega,
        int block_y = 4);
    ~GpuSmoothedAggregationContext();

    GpuSmoothedAggregationContext(const GpuSmoothedAggregationContext&) = delete;
    GpuSmoothedAggregationContext& operator=(const GpuSmoothedAggregationContext&) = delete;
    GpuSmoothedAggregationContext(GpuSmoothedAggregationContext&&) noexcept;
    GpuSmoothedAggregationContext& operator=(GpuSmoothedAggregationContext&&) noexcept;

    GpuSmoothedAggregationApplyResult apply(
        const std::vector<float>& coarse_x,
        std::size_t transfer_smoothing_steps,
        int repeats = 20);

    std::size_t fine_dofs() const noexcept;
    std::size_t coarse_dofs() const noexcept;
    double omega() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gfss
