#pragma once

#include "gfss/hex8.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <vector>

namespace gfss {

struct AdaptiveForcingStep {
    std::size_t outer_index{0};
    std::size_t inner_iterations{0};
    std::size_t inner_matvecs{0};
    std::size_t inner_audits{0};
    std::size_t predicted_outer_corrections{0};
    bool inner_converged{false};
    bool inner_stagnated{false};
    bool economic_stop{false};
    double outer_residual_before{0.0};
    // Zero in the economic controller: there is deliberately no requested
    // inner tolerance. Kept only so old result readers remain source-compatible.
    double requested_inner_tolerance{0.0};
    double achieved_inner_residual{0.0};
    double best_inner_residual{0.0};
    double outer_contraction{0.0};
    double estimated_contraction_gain{1.0};
    double predicted_total_ms{0.0};
    double inner_solve_ms{0.0};
};

struct MixedRefinementResult {
    std::vector<double> x;
    std::vector<double> outer_relative_residuals;
    std::vector<AdaptiveForcingStep> adaptive_steps;
    std::size_t outer_iterations{0};
    std::size_t inner_solves{0};
    std::size_t total_inner_iterations{0};
    std::size_t total_inner_matvecs{0};
    bool converged{false};
    bool adaptive_controller{false};
    double initial_relative_residual{0.0};
    double final_relative_residual{0.0};
    double accurate_residual_ms{0.0};
    double gpu_context_setup_ms{0.0};
    double gpu_correction_solve_ms{0.0};
    double gpu_correction_wall_ms{0.0};
    double total_ms{0.0};
};

// Fixed-forcing mixed-precision defect correction retained as an experimental
// baseline. The outer iterate/residual are FP64 and the inner correction uses
// a reusable FP32 GPU GoldSparse Jacobi-PCG context.
MixedRefinementResult solve_mixed_refinement_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    double outer_relative_tolerance = 1.0e-6,
    double inner_relative_tolerance = 1.0e-2,
    std::size_t max_outer_iterations = 8,
    std::size_t max_inner_iterations = 1500,
    int block_y = 4);

// Self-tuning mixed-precision refinement. The user supplies only the true
// outer tolerance. Each FP32 correction follows one uninterrupted PCG
// trajectory and decides at audited residual milestones whether continuing
// deeper is predicted to reduce total remaining solve time. The FP64 outer
// residual measures the actual contraction and updates the gain used by the
// next inner economic decision.
MixedRefinementResult solve_mixed_refinement_adaptive_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    double outer_relative_tolerance = 1.0e-6,
    std::size_t max_outer_iterations = 12,
    std::size_t max_inner_iterations = 5000,
    int block_y = 4);

}  // namespace gfss
