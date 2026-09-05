#pragma once

#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <cstdint>
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

struct GpuSmoothedAggregationCoarsePcgResult {
    std::vector<float> x;
    std::size_t iterations{0};
    double relative_residual{0.0};
    double solve_ms{0.0};
    std::size_t persistent_coarse_pcg_bytes{0};
};

// Experimental M5 L1 one-step block-Chebyshev result. This method is staged
// in the standalone L1 benchmark TU until the exact-block metric and factorized
// A1 path have passed the FP64 oracle on the reference GPU.
struct GpuSmoothedAggregationL1BlockStepResult {
    std::vector<float> x;
    double median_a1_ms{0.0};
    double median_block_update_ms{0.0};
    double median_total_ms{0.0};
    double best_a1_ms{0.0};
    double best_block_update_ms{0.0};
    double best_total_ms{0.0};
    std::size_t fine_operator_applies{0};
    std::size_t persistent_l1_bytes{0};
};

// Complete staged L1 V-cycle shell around an externally supplied L2 correction.
// The exact frozen P1 is materialized in dual-order 6x6 block form: forward
// row-major block values for P1 and transpose-ordered [q][r][entry] values for
// P1^T. All shell vectors and transfers stay on device during the timed region.
// Since every preconditioner application starts L1 from zero and nu1=1, the
// mathematically redundant A1*0 in the degree-1 pre-smoother is elided exactly.
struct GpuSmoothedAggregationL1ShellResult {
    std::vector<float> final_x;
    std::vector<float> l2_residual_padded;
    double median_pre_smooth_ms{0.0};
    double median_residual_ms{0.0};
    double median_pack_ms{0.0};
    double median_p1t_ms{0.0};
    double median_p1_ms{0.0};
    double median_correction_ms{0.0};
    double median_post_smooth_ms{0.0};
    double median_total_ms{0.0};
    double best_total_ms{0.0};
    std::size_t mathematical_a1_applies{0U};
    std::size_t executed_a1_applies{0U};
    bool zero_start_pre_a1_elided{false};
    std::size_t persistent_l1_shell_bytes{0U};
};

// Persistent GPU implementation of
//
//   y_c = P_m^T A P_m x_c,
//   P_m = (I - omega D^{-1} A)^m P0.
//
// P0/P0^T are reconstructed from aggregation metadata; smoothed interpolation
// is never materialized. Transfer uses aggregate-owned CSR node lists: one warp
// owns one aggregate, P0 writes uniquely owned fine nodes, and P0^T performs a
// warp-local reduction followed by one direct store per coarse DOF. Therefore
// tentative restriction needs neither global atomics nor a coarse-vector
// memset. Fused forward and exact-transpose smoothing ping-pong two persistent
// fine FP32 work vectors. The structured-Q1 path deliberately reuses the
// existing GoldSparse stencil data so timings remain comparable with M2/M3.
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

    // Fixed-budget FP32 PCG on A_c=P_m^T A P_m. The tentative A_c inverse
    // diagonal is supplied by the reference setup and used only as a Jacobi
    // preconditioner. All Krylov vectors, coarse matvecs and updates stay on
    // device during the timed iteration loop; H2D setup and final D2H are
    // excluded. This is intentionally a budget experiment, so it performs
    // exactly max_iterations unless numerical breakdown occurs.
    GpuSmoothedAggregationCoarsePcgResult solve_coarse_pcg_fixed_iterations(
        const std::vector<float>& rhs,
        const std::vector<float>& inverse_preconditioner,
        std::size_t transfer_smoothing_steps,
        std::size_t max_iterations);

    // One degree-1 block-Chebyshev/Jacobi update on L1 using the actual
    // diagonal-block inverse supplied as one padded row-major 6x6 matrix per
    // fine aggregate. The factorized A1=P_m^T A0 P_m action and block update
    // remain device-resident inside the timed loop. This entry point is defined
    // only by the standalone M5 L1 staging benchmark for now.
    GpuSmoothedAggregationL1BlockStepResult l1_block_chebyshev_step(
        const std::vector<float>& rhs,
        const std::vector<float>& initial_x,
        const std::vector<float>& inverse_blocks_6x6,
        double lambda_max,
        std::size_t transfer_smoothing_steps,
        int repeats = 50);

    // Complete L1 shell with nu1=1 and explicit dual-order block6 P1/P1^T.
    // external_l2_correction_padded has 6*l2_nodes entries. Forward P1 uses
    // block-row offsets/columns plus row-major 6x6 values. Restriction P1^T uses
    // column offsets/source rows plus transpose-ordered [q][r][entry] values.
    GpuSmoothedAggregationL1ShellResult l1_full_shell(
        const std::vector<float>& rhs,
        const std::vector<float>& inverse_blocks_6x6,
        double lambda_max,
        std::size_t a1_transfer_smoothing_steps,
        std::size_t l2_nodes,
        const std::vector<std::uint32_t>& p1_forward_row_offsets,
        const std::vector<std::uint32_t>& p1_forward_column_indices,
        const std::vector<float>& p1_forward_values_6x6,
        const std::vector<std::uint32_t>& p1_transpose_column_offsets,
        const std::vector<std::uint32_t>& p1_transpose_row_indices,
        const std::vector<float>& p1_transpose_values_q_r_entry,
        const std::vector<float>& external_l2_correction_padded,
        int repeats = 50);

    std::size_t fine_dofs() const noexcept;
    std::size_t coarse_dofs() const noexcept;
    double omega() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gfss
