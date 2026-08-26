#include "gfss/mixed_refinement.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

std::vector<double> make_rhs(const gfss::StructuredHexMesh& mesh) {
    std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const double load = -1.0 /
        static_cast<double>((mesh.ny + 1U) * (mesh.nz + 1U));
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = load;
        }
    }
    return rhs;
}

}  // namespace

int main() {
    const gfss::StructuredHexMesh mesh{6, 4, 3, 1.0, 0.8, 0.6};
    const gfss::Material material{73000.0, 0.29};
    const auto rhs = make_rhs(mesh);

    const auto result = gfss::solve_mixed_refinement_adaptive_x0(
        mesh,
        material,
        rhs,
        1.0e-5,
        8U,
        1500U,
        4);

    require(result.adaptive_controller,
            "adaptive refinement must identify controller mode");
    require(result.converged,
            "adaptive refinement must converge on smoke-test problem");
    require(result.final_relative_residual <= 1.0e-5,
            "adaptive refinement must satisfy the true outer tolerance");
    require(result.initial_relative_residual > result.final_relative_residual,
            "adaptive refinement must reduce the true residual");
    require(!result.adaptive_steps.empty(),
            "adaptive refinement must perform at least one correction");
    require(result.adaptive_steps.size() == result.inner_solves,
            "adaptive step accounting must match inner solve count");
    require(result.outer_iterations == result.inner_solves,
            "each accepted correction must advance one outer iteration");
    require(result.outer_relative_residuals.size() ==
                result.outer_iterations + 1U,
            "outer residual history must include initial and final residuals");

    for (const auto& step : result.adaptive_steps) {
        require(step.requested_inner_tolerance >= 1.0e-4 &&
                    step.requested_inner_tolerance <= 2.0e-1,
                "adaptive forcing must remain inside controller bounds");
        require(step.inner_iterations > 0U,
                "adaptive correction must perform inner iterations");
        require(step.inner_audits > 0U,
                "adaptive correction must produce audited telemetry");
        require(step.achieved_inner_residual > 0.0 &&
                    step.achieved_inner_residual < 0.9,
                "adaptive correction must produce a useful audited residual");
        require(step.inner_solve_ms > 0.0,
                "adaptive correction timing must be positive");
    }

    std::cout << "Adaptive refinement checks passed; outer="
              << result.outer_iterations
              << " inner_iterations=" << result.total_inner_iterations
              << " final_true_rel=" << result.final_relative_residual
              << " first_eta="
              << result.adaptive_steps.front().requested_inner_tolerance
              << " total_ms=" << result.total_ms << '\n';
    return 0;
}
