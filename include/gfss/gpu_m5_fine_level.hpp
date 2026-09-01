#pragma once

#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <cstdint>
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

// Timing decomposition for one complete frozen 4-level 5x1x1 V-cycle.
// Each interval includes all kernels needed to move from the named level to the
// next boundary; setup/H2D/final D2H are excluded.
struct GpuM5CompleteVcycleTiming {
    double l0_down_ms{0.0};
    double l1_down_ms{0.0};
    double l2_down_ms{0.0};
    double l3_solve_ms{0.0};
    double l2_up_ms{0.0};
    double l1_up_ms{0.0};
    double l0_up_ms{0.0};
    double total_ms{0.0};
};

struct GpuM5CompleteVcycleResult {
    // Final fine correction in AoS layout, matching the solver-facing vector.
    std::vector<float> fine_correction_aos;
    // Bottom RHS before the direct solve, retained as an intermediate oracle.
    std::vector<float> l3_rhs;
    GpuM5CompleteVcycleTiming median_timing;
    GpuM5CompleteVcycleTiming best_timing;
    std::size_t device_bytes_total{0};
    std::size_t l0_operator_applies{0};
    std::size_t l1_operator_applies{0};
    std::size_t l2_operator_applies{0};
    std::size_t bottom_dofs{0};
};

// Fixed-iteration staging result for a fully device-resident left-preconditioned
// CG solve. The complete 5x1x1 V-cycle is applied as M^-1 without H2D/D2H or
// host scalar reductions inside a timed solve. A final host-facing solution is
// returned for the independent FP64 true-residual oracle.
struct GpuM5VcyclePcgResult {
    std::vector<float> solution_aos;
    double median_solve_ms{0.0};
    double best_solve_ms{0.0};
    double recursive_relative_residual{0.0};
    std::size_t iterations{0};
    std::size_t preconditioner_applications{0};
    std::size_t pcg_operator_applications{0};
    std::size_t vcycle_l0_operator_applies{0};
    std::size_t total_l0_operator_applies{0};
    std::size_t device_bytes_total{0};
    bool breakdown{false};
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
    // recursive L1/L2 production path plugs into.
    GpuM5FineFullShellResult full_shell(
        const std::vector<float>& rhs_aos,
        const std::vector<float>& coarse_correction,
        std::size_t smoother_degree,
        std::size_t transfer_smoothing_steps,
        int repeats = 50);

    // Staged complete 4-level GPU V-cycle for the frozen 5x1x1 hierarchy.
    // This entry point is defined only by the standalone complete-cycle
    // benchmark TU until the full cycle passes its independent FP64 oracle.
    // P1 is dual-order block6, A2/P2 are dense FP32, and the L3 Cholesky factor
    // is supplied as a row-major FP32 lower triangle.
    GpuM5CompleteVcycleResult complete_vcycle_5x1x1(
        const std::vector<float>& rhs_aos,
        std::size_t l0_smoother_degree,
        std::size_t m0,
        const std::vector<float>& l1_inverse_blocks_6x6,
        double lambda1,
        std::size_t l2_nodes,
        const std::vector<std::uint32_t>& p1_forward_row_offsets,
        const std::vector<std::uint32_t>& p1_forward_column_indices,
        const std::vector<float>& p1_forward_values_6x6,
        const std::vector<std::uint32_t>& p1_transpose_column_offsets,
        const std::vector<std::uint32_t>& p1_transpose_row_indices,
        const std::vector<float>& p1_transpose_values_q_r_entry,
        const std::vector<float>& a2_dense_row_major,
        const std::vector<float>& l2_inverse_blocks_6x6,
        double lambda2,
        const std::vector<float>& p2_dense_row_major,
        std::size_t l3_dofs,
        const std::vector<float>& l3_cholesky_lower_row_major,
        int repeats = 50);

    // Fixed-iteration fully device-resident PCG staging path. The number of
    // iterations is intentionally fixed so no host convergence check or scalar
    // round trip contaminates the timed inner solve. The caller performs the
    // authoritative FP64 true-residual verification on the returned solution.
    GpuM5VcyclePcgResult solve_pcg_vcycle_5x1x1_fixed(
        const std::vector<float>& rhs_aos,
        std::size_t iterations,
        std::size_t l0_smoother_degree,
        std::size_t m0,
        const std::vector<float>& l1_inverse_blocks_6x6,
        double lambda1,
        std::size_t l2_nodes,
        const std::vector<std::uint32_t>& p1_forward_row_offsets,
        const std::vector<std::uint32_t>& p1_forward_column_indices,
        const std::vector<float>& p1_forward_values_6x6,
        const std::vector<std::uint32_t>& p1_transpose_column_offsets,
        const std::vector<std::uint32_t>& p1_transpose_row_indices,
        const std::vector<float>& p1_transpose_values_q_r_entry,
        const std::vector<float>& a2_dense_row_major,
        const std::vector<float>& l2_inverse_blocks_6x6,
        double lambda2,
        const std::vector<float>& p2_dense_row_major,
        std::size_t l3_dofs,
        const std::vector<float>& l3_cholesky_lower_row_major,
        int repeats = 20);

    std::size_t fine_dofs() const noexcept;
    std::size_t coarse_dofs() const noexcept;
    double transfer_omega() const noexcept;
    double smoother_lambda_max() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gfss
