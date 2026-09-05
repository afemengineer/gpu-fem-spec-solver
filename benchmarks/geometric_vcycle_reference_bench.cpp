#include "gfss/multigrid_reference.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class LoadKind {
    UniformZ,
    CheckerboardZ,
};

struct CaseDef {
    const char* name;
    const char* description;
    gfss::StructuredHexMesh mesh;
    gfss::Material material;
    LoadKind load;
};

std::vector<CaseDef> make_cases() {
    constexpr double E = 210.0e9;
    return {
        {"baseline_cube",
         "64^3 cube, uniform transverse face load",
         {64U, 64U, 64U, 1.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::UniformZ},
        {"slender_beam",
         "4:1:1 cantilever with near-isotropic cells",
         {128U, 32U, 32U, 4.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::UniformZ},
        {"thin_plate",
         "thin 1:1:0.125 domain with near-isotropic cells",
         {64U, 64U, 8U, 1.0, 1.0, 0.125},
         {E, 0.30},
         LoadKind::UniformZ},
        {"near_incompressible",
         "48^3 cube with Poisson ratio 0.49",
         {48U, 48U, 48U, 1.0, 1.0, 1.0},
         {E, 0.49},
         LoadKind::UniformZ},
        {"checkerboard_face",
         "48^3 cube with high-frequency alternating face load",
         {48U, 48U, 48U, 1.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::CheckerboardZ},
    };
}

std::vector<double> make_rhs(const CaseDef& test) {
    const auto& mesh = test.mesh;
    std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const double count =
        static_cast<double>(mesh.ny + 1U) *
        static_cast<double>(mesh.nz + 1U);
    const double magnitude = 1.0 / count;
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            double value = -magnitude;
            if (test.load == LoadKind::CheckerboardZ) {
                value = ((j + k) & 1U) == 0U ? magnitude : -magnitude;
            }
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = value;
        }
    }
    return rhs;
}

double parse_tolerance(const char* text) {
    const double value = std::stod(text);
    if (!(value > 0.0) || value >= 1.0) {
        throw std::invalid_argument("relative tolerance must be in (0,1)");
    }
    return value;
}

std::size_t parse_size(const char* text, const char* name) {
    const auto value = std::stoull(text);
    if (value == 0ULL) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

void run_case(const CaseDef& test,
              double tolerance,
              std::size_t max_cycles,
              std::size_t pre_degree,
              std::size_t post_degree,
              std::size_t power_iterations) {
    const auto rhs = make_rhs(test);
    const auto result = gfss::solve_geometric_vcycle_reference_x0(
        test.mesh,
        test.material,
        rhs,
        tolerance,
        max_cycles,
        pre_degree,
        post_degree,
        power_iterations);

    std::cout << std::fixed << std::setprecision(6)
              << "case=" << test.name << '\n'
              << "description=" << test.description << '\n'
              << "mesh=" << test.mesh.nx << 'x' << test.mesh.ny << 'x' << test.mesh.nz
              << " physical=" << test.mesh.lx << 'x' << test.mesh.ly << 'x' << test.mesh.lz
              << " dofs=" << test.mesh.dof_count() << '\n'
              << std::scientific
              << "material_E=" << test.material.young_modulus
              << " poisson=" << test.material.poisson_ratio << '\n'
              << "target_true_relative_residual=" << tolerance << '\n'
              << std::fixed
              << "reference_execution=cpu_fp64_openmp\n"
              << "performance_status=numerical_hypothesis_only\n"
              << "smoother=chebyshev_jacobi"
              << " pre_degree=" << pre_degree
              << " post_degree=" << post_degree
              << " power_iterations=" << power_iterations << '\n'
              << "hierarchy_levels=" << result.levels.size() << '\n';

    for (std::size_t i = 0; i < result.levels.size(); ++i) {
        const auto& level = result.levels[i];
        std::cout << "level[" << i << "]="
                  << level.mesh.nx << 'x' << level.mesh.ny << 'x' << level.mesh.nz
                  << " dofs=" << level.dofs
                  << std::scientific
                  << " lambda_max_est=" << level.estimated_lambda_max
                  << std::fixed << '\n';
    }

    std::cout << "converged=" << (result.converged ? "true" : "false")
              << " cycles=" << result.cycles << '\n';
    for (std::size_t i = 0; i < result.relative_residuals.size(); ++i) {
        const double r = result.relative_residuals[i];
        std::cout << std::scientific
                  << "true_residual[" << i << "]=" << r;
        if (i > 0U && result.relative_residuals[i - 1U] > 0.0) {
            std::cout << " cycle_q=" << r / result.relative_residuals[i - 1U];
        }
        std::cout << '\n';
    }

    std::cout << std::fixed
              << "setup_ms=" << result.setup_ms
              << " solve_ms=" << result.solve_ms << '\n'
              << "estimated_three_vector_coarse_bytes_per_fine_dof="
              << result.estimated_three_vector_coarse_bytes_per_fine_dof << '\n'
              << "estimated_six_vector_coarse_bytes_per_fine_dof="
              << result.estimated_six_vector_coarse_bytes_per_fine_dof << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "thin_plate";
        const double tolerance = argc > 2 ? parse_tolerance(argv[2]) : 1.0e-6;
        const std::size_t max_cycles = argc > 3
            ? parse_size(argv[3], "max cycles") : 12U;
        const std::size_t pre_degree = argc > 4
            ? parse_size(argv[4], "pre smoother degree") : 3U;
        const std::size_t post_degree = argc > 5
            ? parse_size(argv[5], "post smoother degree") : 3U;
        const std::size_t power_iterations = argc > 6
            ? parse_size(argv[6], "power iterations") : 8U;

        const auto cases = make_cases();
        std::size_t selected = 0U;
        std::cout << "GFSS M5 geometric V-cycle numerical reference\n"
                  << "restriction=P^T\n"
                  << "prolongation=trilinear_P\n"
                  << "coarse_operator=rediscretized_Q1_equals_Galerkin_for_tested_structured_case\n"
                  << "bottom_solver=cpu_fp64_cg\n";

        for (const auto& test : cases) {
            if (selector != "all" && selector != test.name) continue;
            ++selected;
            std::cout << "\n========================================\n";
            run_case(test,
                     tolerance,
                     max_cycles,
                     pre_degree,
                     post_degree,
                     power_iterations);
        }
        if (selected == 0U) {
            std::cerr << "error: unknown case '" << selector << "'\n"
                      << "available:";
            for (const auto& test : cases) std::cerr << ' ' << test.name;
            std::cerr << " all\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_geometric_vcycle_reference_bench "
                  << "[case|all [tol [max_cycles [pre_degree [post_degree [power_iterations]]]]]]\n";
        return 1;
    }
}
