#include "gfss/mixed_refinement.hpp"

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
    UniformY,
    PatchZ,
    Torsion,
    CheckerboardZ,
};

struct CaseDef {
    const char* name;
    const char* description;
    gfss::StructuredHexMesh mesh;
    gfss::Material material;
    LoadKind load;
};

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

const char* load_name(LoadKind load) {
    switch (load) {
    case LoadKind::UniformZ: return "uniform_z_xmax";
    case LoadKind::UniformY: return "uniform_y_xmax";
    case LoadKind::PatchZ: return "central_patch_z_xmax";
    case LoadKind::Torsion: return "torsion_xmax";
    case LoadKind::CheckerboardZ: return "checkerboard_z_xmax";
    }
    return "unknown";
}

std::vector<double> make_rhs(const CaseDef& test) {
    const auto& mesh = test.mesh;
    std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);

    if (test.load == LoadKind::UniformZ || test.load == LoadKind::UniformY) {
        const double count =
            static_cast<double>(mesh.ny + 1U) *
            static_cast<double>(mesh.nz + 1U);
        const double value = -1.0 / count;
        const std::size_t component =
            test.load == LoadKind::UniformZ ? 2U : 1U;
        for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
            for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
                const auto node = mesh.node_index(mesh.nx, j, k);
                rhs[static_cast<std::size_t>(3ULL * node + component)] = value;
            }
        }
        return rhs;
    }

    if (test.load == LoadKind::PatchZ) {
        const std::uint32_t j0 = mesh.ny / 3U;
        const std::uint32_t j1 = (2U * mesh.ny) / 3U;
        const std::uint32_t k0 = mesh.nz / 3U;
        const std::uint32_t k1 = (2U * mesh.nz) / 3U;
        const std::uint64_t count =
            static_cast<std::uint64_t>(j1 - j0 + 1U) *
            static_cast<std::uint64_t>(k1 - k0 + 1U);
        const double value = -1.0 / static_cast<double>(count);
        for (std::uint32_t k = k0; k <= k1; ++k) {
            for (std::uint32_t j = j0; j <= j1; ++j) {
                const auto node = mesh.node_index(mesh.nx, j, k);
                rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = value;
            }
        }
        return rhs;
    }

    if (test.load == LoadKind::Torsion) {
        double normalization = 0.0;
        for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
            const double z = static_cast<double>(k) /
                             static_cast<double>(mesh.nz) - 0.5;
            for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
                const double y = static_cast<double>(j) /
                                 static_cast<double>(mesh.ny) - 0.5;
                normalization += std::sqrt(y * y + z * z);
            }
        }
        if (!(normalization > 0.0)) {
            throw std::runtime_error("torsion load normalization is zero");
        }
        for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
            const double z = static_cast<double>(k) /
                             static_cast<double>(mesh.nz) - 0.5;
            for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
                const double y = static_cast<double>(j) /
                                 static_cast<double>(mesh.ny) - 0.5;
                const auto node = mesh.node_index(mesh.nx, j, k);
                const auto base = static_cast<std::size_t>(3ULL * node);
                rhs[base + 1U] = -z / normalization;
                rhs[base + 2U] = y / normalization;
            }
        }
        return rhs;
    }

    if (test.load == LoadKind::CheckerboardZ) {
        const double count =
            static_cast<double>(mesh.ny + 1U) *
            static_cast<double>(mesh.nz + 1U);
        const double magnitude = 1.0 / count;
        for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
            for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
                const auto node = mesh.node_index(mesh.nx, j, k);
                const double sign = ((j + k) & 1U) == 0U ? 1.0 : -1.0;
                rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] =
                    sign * magnitude;
            }
        }
        return rhs;
    }

    throw std::runtime_error("unsupported problem-suite load");
}

std::vector<CaseDef> make_cases() {
    constexpr double E = 210.0e9;
    return {
        {"baseline_cube",
         "isotropic cube, standard material, uniform transverse face load",
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
        {"anisotropic_cells",
         "unit cube with 4:1 element-size anisotropy",
         {96U, 24U, 24U, 1.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::UniformZ},
        {"near_incompressible",
         "cube with Poisson ratio 0.49",
         {48U, 48U, 48U, 1.0, 1.0, 1.0},
         {E, 0.49},
         LoadKind::UniformZ},
        {"localized_patch",
         "cube with concentrated central face patch load",
         {64U, 64U, 64U, 1.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::PatchZ},
        {"torsion_face",
         "cube with zero-net-force torsional face traction",
         {64U, 64U, 64U, 1.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::Torsion},
        {"checkerboard_face",
         "cube with high-spatial-frequency alternating face load",
         {48U, 48U, 48U, 1.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::CheckerboardZ},
    };
}

void print_result(const CaseDef& test,
                  const gfss::MixedRefinementResult& result,
                  double outer_tolerance) {
    const auto& mesh = test.mesh;
    std::cout << std::fixed << std::setprecision(6)
              << "case=" << test.name << '\n'
              << "description=" << test.description << '\n'
              << "mesh=" << mesh.nx << 'x' << mesh.ny << 'x' << mesh.nz
              << " physical=" << mesh.lx << 'x' << mesh.ly << 'x' << mesh.lz
              << " elements=" << mesh.element_count()
              << " dofs=" << mesh.dof_count() << '\n'
              << "material_E=" << std::scientific << test.material.young_modulus
              << " poisson=" << test.material.poisson_ratio << '\n'
              << "load=" << load_name(test.load) << '\n'
              << "outer_tolerance=" << outer_tolerance << '\n'
              << std::fixed
              << "converged=" << (result.converged ? "true" : "false")
              << " outer_iterations=" << result.outer_iterations
              << " inner_solves=" << result.inner_solves
              << " total_inner_iterations=" << result.total_inner_iterations
              << " total_inner_matvecs=" << result.total_inner_matvecs << '\n'
              << std::scientific
              << "initial_true_relative_residual="
              << result.initial_relative_residual
              << " final_true_relative_residual="
              << result.final_relative_residual << '\n';

    for (std::size_t i = 0; i < result.adaptive_steps.size(); ++i) {
        const auto& step = result.adaptive_steps[i];
        std::cout << "step[" << i << "]"
                  << " selected_eta=" << step.achieved_inner_residual
                  << " outer_q=" << step.outer_contraction
                  << " gain=" << step.estimated_contraction_gain
                  << std::fixed
                  << " iterations=" << step.inner_iterations
                  << " audits=" << step.inner_audits
                  << " predicted_outer=" << step.predicted_outer_corrections
                  << " economic_stop=" << (step.economic_stop ? "true" : "false")
                  << " stagnated=" << (step.inner_stagnated ? "true" : "false")
                  << " solve_ms=" << step.inner_solve_ms
                  << std::scientific << '\n';
    }

    const double warm_ms =
        result.accurate_residual_ms + result.gpu_correction_wall_ms;
    std::cout << std::fixed
              << "accurate_residual_ms=" << result.accurate_residual_ms
              << " gpu_correction_wall_ms=" << result.gpu_correction_wall_ms
              << " warm_ms=" << warm_ms
              << " gpu_context_setup_ms=" << result.gpu_context_setup_ms
              << " total_ms=" << result.total_ms << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "all";
        const double outer_tolerance = argc > 2
            ? parse_outer_tolerance(argv[2])
            : 1.0e-6;
        const std::size_t max_outer = argc > 3
            ? parse_size(argv[3], "max outer iterations")
            : 12U;
        const std::size_t max_inner = argc > 4
            ? parse_size(argv[4], "max inner iterations")
            : 5000U;

        const auto cases = make_cases();
        std::size_t selected = 0U;
        std::size_t converged = 0U;
        std::size_t failed = 0U;

        std::cout << "GFSS adaptive problem-generalization suite\n"
                  << "controller=in_pcg_economic_stopping\n"
                  << "constraint=x0_zero_dirichlet\n"
                  << "selector=" << selector << '\n';

        for (const auto& test : cases) {
            if (selector != "all" && selector != test.name) {
                continue;
            }
            ++selected;
            std::cout << "\n========================================\n";
            try {
                const auto rhs = make_rhs(test);
                const auto result = gfss::solve_mixed_refinement_adaptive_x0(
                    test.mesh,
                    test.material,
                    rhs,
                    outer_tolerance,
                    max_outer,
                    max_inner,
                    4);
                print_result(test, result, outer_tolerance);
                if (result.converged &&
                    result.final_relative_residual <= outer_tolerance) {
                    ++converged;
                } else {
                    ++failed;
                }
            } catch (const std::exception& e) {
                ++failed;
                std::cout << "case=" << test.name << '\n'
                          << "error=" << e.what() << '\n';
            }
        }

        if (selected == 0U) {
            std::cerr << "error: unknown suite case '" << selector << "'\n"
                      << "available cases:";
            for (const auto& test : cases) {
                std::cerr << ' ' << test.name;
            }
            std::cerr << " all\n";
            return 1;
        }

        std::cout << "\n========================================\n"
                  << "suite_selected=" << selected
                  << " suite_converged=" << converged
                  << " suite_failed=" << failed << '\n';
        return failed == 0U ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_adaptive_problem_suite_bench "
                  << "[all|case [outer_tol [max_outer [max_inner]]]]\n";
        return 1;
    }
}
