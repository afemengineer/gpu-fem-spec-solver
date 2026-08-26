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
    bool inner_converged{false};
    bool inner_stagnated{false};
    double outer_residual_before{0.0};
    double requested_inner_tolerance{0.0};
    double achieved_inner_residual{0.0};
    double best_inner_residual{0.0};
    double outer_contraction{0.0};
    double estimated_contraction_gain{1.0};
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

// Fixed-forcing mixed-precision defect correction retained as a reproducible
// research baseline. The outer iterate and residual are FP64 and use the
// trusted CPU matrix-free operator. Each correction equation is solved with
// the reusable FP32 GPU GoldSparse Jacobi-PCG context.
MixedRefinementResult solve_mixed_refinement_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    double outer_relative_tolerance = 1.0e-6,
    double inner_relative_tolerance = 1.0e-2,
    std::size_t max_outer_iterations = 8,
    std::size_t max_inner_iterations = 1500,
    int block_y = 4);

// Adaptive-forcing mixed refinement. The caller supplies only the true outer
// accuracy target and resource limits. The first forcing term is derived from
// the remaining outer reduction assuming three calibration corrections. After
// each correction the controller learns the actual outer/inner contraction
// gain and a local cost-vs-log-residual model from PCG audit samples. It then
// chooses the next forcing term by minimizing predicted remaining correction
// plus accurate-residual time. An unreachable FP32 target is treated as a
// measured capability floor rather than a fatal solver error.
MixedRefinementResult solve_mixed_refinement_adaptive_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    double outer_relative_tolerance = 1.0e-6,
    std::size_t max_outer_iterations = 12,
    std::size_t max_inner_iterations = 5000,
    int block_y = 4);

}  // namespace gfss
