#include "gfss/mixed_refinement.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t parse_u32(const char* text) {
    const auto value = std::stoul(text);
    if (value == 0 || value > 1000000UL) {
        throw std::invalid_argument("mesh dimensions must be in [1, 1000000]");
    }
    return static_cast<std::uint32_t>(value);
}

std::size_t parse_size(const char* text, const char* name) {
    const auto value = std::stoull(text);
    if (value == 0ULL) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

double parse_outer_tolerance(const char* text) {
    const double value = std::stod(text);
    if (!(value > 0.0) || value >= 1.0) {
        throw std::invalid_argument("outer tolerance must be in (0, 1)");
    }
    return value;
}

std::vector<double> make_cantilever_rhs(const gfss::StructuredHexMesh& mesh) {
    std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const double face_nodes = static_cast<double>(mesh.ny + 1U) *
                              static_cast<double>(mesh.nz + 1U);
    const double nodal_load = -1.0 / face_nodes;

    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = nodal_load;
        }
    }
    return rhs;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t nx = argc > 1 ? parse_u32(argv[1]) : 32U;
        const std::uint32_t ny = argc > 2 ? parse_u32(argv[2]) : nx;
        const std::uint32_t nz = argc > 3 ? parse_u32(argv[3]) : nx;
        const double outer_tolerance = argc > 4
            ? parse_outer_tolerance(argv[4])
            : 1.0e-6;
        const std::size_t max_outer = argc > 5
            ? parse_size(argv[5], "max outer iterations")
            : 12U;
        const std::size_t max_inner = argc > 6
            ? parse_size(argv[6], "max inner iterations")
            : 5000U;

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        const auto rhs = make_cantilever_rhs(mesh);

        const auto result = gfss::solve_mixed_refinement_adaptive_x0(
            mesh,
            material,
            rhs,
            outer_tolerance,
            max_outer,
            max_inner,
            4);

        std::cout << std::fixed << std::setprecision(6)
                  << "GFSS adaptive mixed-precision defect-correction benchmark\n"
                  << "mesh=" << nx << 'x' << ny << 'x' << nz
                  << " elements=" << mesh.element_count()
                  << " dofs=" << mesh.dof_count() << '\n'
                  << "constraint=x0_zero_dirichlet\n"
                  << "rhs=uniform_z_load_on_xmax_face total_load=-1\n"
                  << std::scientific
                  << "outer_tolerance=" << outer_tolerance << '\n'
                  << std::fixed
                  << "max_outer_iterations=" << max_outer
                  << " max_inner_iterations=" << max_inner << '\n'
                  << "controller=in_pcg_economic_stopping\n"
                  << "inner_tolerance=user_unspecified\n"
                  << "converged=" << (result.converged ? "true" : "false")
                  << " outer_iterations=" << result.outer_iterations
                  << " inner_solves=" << result.inner_solves << '\n'
                  << "total_inner_iterations=" << result.total_inner_iterations
                  << " total_inner_matvecs=" << result.total_inner_matvecs << '\n'
                  << std::scientific
                  << "initial_true_relative_residual="
                  << result.initial_relative_residual << '\n'
                  << "final_true_relative_residual="
                  << result.final_relative_residual << '\n';

        for (std::size_t i = 0; i < result.outer_relative_residuals.size(); ++i) {
            std::cout << "outer_residual[" << i << "]="
                      << result.outer_relative_residuals[i] << '\n';
        }

        for (std::size_t i = 0; i < result.adaptive_steps.size(); ++i) {
            const auto& step = result.adaptive_steps[i];
            std::cout << "adaptive_step[" << i << "]"
                      << " selected_eta=" << step.achieved_inner_residual
                      << " best_eta=" << step.best_inner_residual
                      << " outer_q=" << step.outer_contraction
                      << " gain_used=" << step.estimated_contraction_gain
                      << std::fixed
                      << " predicted_outer=" << step.predicted_outer_corrections
                      << " iterations=" << step.inner_iterations
                      << " matvecs=" << step.inner_matvecs
                      << " audits=" << step.inner_audits
                      << " economic_stop=" << (step.economic_stop ? "true" : "false")
                      << " stagnated=" << (step.inner_stagnated ? "true" : "false")
                      << " predicted_total_ms=" << step.predicted_total_ms
                      << " solve_ms=" << step.inner_solve_ms
                      << std::scientific << '\n';
        }

        std::cout << std::fixed
                  << "accurate_residual_ms=" << result.accurate_residual_ms << '\n'
                  << "gpu_context_setup_ms=" << result.gpu_context_setup_ms << '\n'
                  << "gpu_correction_solve_ms="
                  << result.gpu_correction_solve_ms << '\n'
                  << "gpu_correction_wall_ms="
                  << result.gpu_correction_wall_ms << '\n'
                  << "total_ms=" << result.total_ms << '\n'
                  << "outer_residual=cpu_fp64_matrix_free\n"
                  << "inner_correction=gpu_fp32_goldsparse_jacobi_pcg_persistent_economic\n";

        if (!result.converged) {
            std::cerr << "WARNING: adaptive mixed refinement did not reach the requested outer tolerance\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_adaptive_refinement_bench "
                  << "[nx [ny nz [outer_tol [max_outer [max_inner]]]]]\n";
        return 1;
    }
}
