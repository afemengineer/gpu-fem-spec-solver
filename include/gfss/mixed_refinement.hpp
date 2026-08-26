#pragma once

#include "gfss/hex8.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <vector>

namespace gfss {

struct MixedRefinementResult {
    std::vector<double> x;
    std::vector<double> outer_relative_residuals;
    std::size_t outer_iterations{0};
    std::size_t inner_solves{0};
    std::size_t total_inner_iterations{0};
    std::size_t total_inner_matvecs{0};
    bool converged{false};
    double initial_relative_residual{0.0};
    double final_relative_residual{0.0};
    double accurate_residual_ms{0.0};
    double gpu_context_setup_ms{0.0};
    double gpu_correction_solve_ms{0.0};
    double gpu_correction_wall_ms{0.0};
    double total_ms{0.0};
};

// Mixed-precision defect correction for the clamped structured-Q1 elasticity
// problem. The outer iterate and residual are FP64 and use the trusted CPU
// matrix-free operator. Each correction equation is solved approximately with
// a reusable FP32 GPU GoldSparse Jacobi-PCG context and accumulated into the
// FP64 iterate. The accurate outer residual owns the convergence decision.
MixedRefinementResult solve_mixed_refinement_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    double outer_relative_tolerance = 1.0e-6,
    double inner_relative_tolerance = 1.0e-2,
    std::size_t max_outer_iterations = 8,
    std::size_t max_inner_iterations = 1500,
    int block_y = 4);

}  // namespace gfss
