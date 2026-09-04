// M5 structured-elasticity generalization gate.
// Reuses the established M4 problem-suite definitions verbatim while running
// the current M5 fast hierarchy + persistent FP64-defect / FP32 MG-PCG policy.
// No per-case solver or hierarchy tuning is permitted.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_fast_hierarchy_bundle.hpp"
#include "m5_persistent_vpcg_staging.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

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

struct Step {
    double outer_before{0.0};
    double recursive_rel{0.0};
    double true_correction_rel{0.0};
    double gpu_solve_ms{0.0};
    double wall_ms{0.0};
    bool breakdown{false};
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
        const double count = static_cast<double>(mesh.ny + 1U) *
                             static_cast<double>(mesh.nz + 1U);
        const double value = -1.0 / count;
        const std::size_t component = test.load == LoadKind::UniformZ ? 2U : 1U;
        for (std::uint32_t k = 0U; k <= mesh.nz; ++k) {
            for (std::uint32_t j = 0U; j <= mesh.ny; ++j) {
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
        for (std::uint32_t k = 0U; k <= mesh.nz; ++k) {
            const double z = static_cast<double>(k) / static_cast<double>(mesh.nz) - 0.5;
            for (std::uint32_t j = 0U; j <= mesh.ny; ++j) {
                const double y = static_cast<double>(j) / static_cast<double>(mesh.ny) - 0.5;
                normalization += std::sqrt(y * y + z * z);
            }
        }
        if (!(normalization > 0.0)) {
            throw std::runtime_error("torsion load normalization is zero");
        }
        for (std::uint32_t k = 0U; k <= mesh.nz; ++k) {
            const double z = static_cast<double>(k) / static_cast<double>(mesh.nz) - 0.5;
            for (std::uint32_t j = 0U; j <= mesh.ny; ++j) {
                const double y = static_cast<double>(j) / static_cast<double>(mesh.ny) - 0.5;
                const auto node = mesh.node_index(mesh.nx, j, k);
                const auto base = static_cast<std::size_t>(3ULL * node);
                rhs[base + 1U] = -z / normalization;
                rhs[base + 2U] = y / normalization;
            }
        }
        return rhs;
    }

    if (test.load == LoadKind::CheckerboardZ) {
        const double count = static_cast<double>(mesh.ny + 1U) *
                             static_cast<double>(mesh.nz + 1U);
        const double magnitude = 1.0 / count;
        for (std::uint32_t k = 0U; k <= mesh.nz; ++k) {
            for (std::uint32_t j = 0U; j <= mesh.ny; ++j) {
                const auto node = mesh.node_index(mesh.nx, j, k);
                const double sign = ((j + k) & 1U) == 0U ? 1.0 : -1.0;
                rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = sign * magnitude;
            }
        }
        return rhs;
    }

    throw std::runtime_error("unsupported M5 generalization-suite load");
}

std::vector<CaseDef> make_cases() {
    constexpr double E = 210.0e9;
    return {
        {"baseline_cube", "isotropic cube, standard material, uniform transverse face load",
         {64U, 64U, 64U, 1.0, 1.0, 1.0}, {E, 0.30}, LoadKind::UniformZ},
        {"slender_beam", "4:1:1 cantilever with near-isotropic cells",
         {128U, 32U, 32U, 4.0, 1.0, 1.0}, {E, 0.30}, LoadKind::UniformZ},
        {"thin_plate", "thin 1:1:0.125 domain with near-isotropic cells",
         {64U, 64U, 8U, 1.0, 1.0, 0.125}, {E, 0.30}, LoadKind::UniformZ},
        {"anisotropic_cells", "unit cube with 4:1 element-size anisotropy",
         {96U, 24U, 24U, 1.0, 1.0, 1.0}, {E, 0.30}, LoadKind::UniformZ},
        {"near_incompressible", "cube with Poisson ratio 0.49",
         {48U, 48U, 48U, 1.0, 1.0, 1.0}, {E, 0.49}, LoadKind::UniformZ},
        {"localized_patch", "cube with concentrated central face patch load",
         {64U, 64U, 64U, 1.0, 1.0, 1.0}, {E, 0.30}, LoadKind::PatchZ},
        {"torsion_face", "cube with zero-net-force torsional face traction",
         {64U, 64U, 64U, 1.0, 1.0, 1.0}, {E, 0.30}, LoadKind::Torsion},
        {"checkerboard_face", "cube with high-spatial-frequency alternating face load",
         {48U, 48U, 48U, 1.0, 1.0, 1.0}, {E, 0.30}, LoadKind::CheckerboardZ},
    };
}

std::vector<float> to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0U; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

std::vector<double> to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0U; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

bool run_case(const CaseDef& test,
              double outer_tolerance,
              std::size_t inner_iterations,
              std::size_t max_outer,
              int block_y,
              std::size_t target_nodes,
              std::size_t min_nodes) {
    constexpr std::size_t nu0 = 5U;
    constexpr std::size_t m0 = 1U;
    const auto case_start = Clock::now();
    std::string stage = "hierarchy_build";

    try {
        std::cout << "case=" << test.name << '\n'
                  << "description=" << test.description << '\n'
                  << "mesh=" << test.mesh.nx << 'x' << test.mesh.ny << 'x' << test.mesh.nz
                  << " physical=" << test.mesh.lx << 'x' << test.mesh.ly << 'x' << test.mesh.lz
                  << " elements=" << test.mesh.element_count()
                  << " dofs=" << test.mesh.dof_count() << '\n'
                  << std::scientific << std::setprecision(9)
                  << "material_E=" << test.material.young_modulus
                  << " poisson=" << test.material.poisson_ratio << '\n'
                  << "load=" << load_name(test.load) << '\n';

        const auto hierarchy = m5_fast_bundle::build(
            test.mesh, test.material, target_nodes, min_nodes, true);
        const auto rhs = make_rhs(test);
        const double rhs_norm = norm(rhs);
        if (!(rhs_norm > 0.0)) throw std::runtime_error("M5 generalization RHS is zero");

        std::cout << std::fixed << std::setprecision(6)
                  << "hierarchy_L0_dofs=" << test.mesh.dof_count()
                  << " hierarchy_L1_dofs=" << hierarchy.space0.coarse_dofs
                  << " hierarchy_L2_dofs=" << hierarchy.block2.dofs()
                  << " hierarchy_L3_dofs=" << hierarchy.bottom.factor.n << '\n'
                  << std::scientific << std::setprecision(9)
                  << "lambda0=" << hierarchy.lambda0
                  << " lambda1=" << hierarchy.lambda1
                  << " lambda2=" << hierarchy.lambda2 << '\n'
                  << "oracle_L1=" << hierarchy.oracle.l1_block
                  << " oracle_A1=" << hierarchy.oracle.a1_materialized
                  << " oracle_L2=" << hierarchy.oracle.l2_block
                  << " oracle_A2=" << hierarchy.oracle.a2_dense
                  << " oracle_P2=" << hierarchy.oracle.p2_dense
                  << " oracle_bottom=" << hierarchy.oracle.bottom
                  << " oracle_bottom_inverse=" << hierarchy.oracle.bottom_inverse_identity
                  << " hierarchy_oracle_accept=" << (hierarchy.oracle.accept ? "true" : "false") << '\n';

        stage = "fine_gpu_context";
        const auto fine_start = Clock::now();
        gfss::GpuM5FineLevelContext fine_context(
            test.mesh, test.material, hierarchy.space0,
            hierarchy.omega0, hierarchy.lambda0, block_y);
        const double fine_gpu_setup_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - fine_start).count();

        stage = "deep_gpu_persistent";
        const auto deep_start = Clock::now();
        gfss::M5PersistentPcgStaging persistent(
            fine_context, nu0, m0,
            hierarchy.inverse1, hierarchy.lambda1,
            hierarchy.transfer1.aggregates.size(),
            hierarchy.p1.forward_row_offsets,
            hierarchy.p1.forward_column_indices,
            hierarchy.p1.forward_values_row_major,
            hierarchy.p1.transpose_column_offsets,
            hierarchy.p1.transpose_row_indices,
            hierarchy.p1.transpose_values_q_r_entry,
            hierarchy.a2_fp32, hierarchy.inverse2, hierarchy.lambda2,
            hierarchy.p2_fp32, hierarchy.transfer2.coarse_dofs,
            hierarchy.bottom_inverse_fp32);
        const double deep_gpu_setup_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - deep_start).count();

        stage = "warmup";
        const auto rhs_fp32 = to_float(rhs);
        persistent.warmup(rhs_fp32, inner_iterations);

        stage = "solve";
        const Apply apply0 = [&](const Vec& x) {
            return apply_fine_clamped(test.mesh, test.material, x);
        };
        Vec x(rhs.size(), 0.0);
        Vec residual(rhs.size(), 0.0);
        std::vector<double> outer_history;
        std::vector<Step> steps;
        outer_history.reserve(max_outer + 1U);
        steps.reserve(max_outer);
        double accurate_residual_ms = 0.0;
        double gpu_inner_solve_ms = 0.0;
        double correction_wall_ms = 0.0;
        std::size_t total_inner_iterations = 0U;
        std::size_t total_l0_operator_applies = 0U;
        bool converged = false;
        bool breakdown = false;

        const auto solve_start = Clock::now();
        for (std::size_t outer = 0U; outer <= max_outer; ++outer) {
            const auto residual_start = Clock::now();
            const auto ax = apply0(x);
            for (std::size_t i = 0U; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
            const double rel = norm(residual) / rhs_norm;
            accurate_residual_ms += std::chrono::duration<double, std::milli>(
                Clock::now() - residual_start).count();
            outer_history.push_back(rel);

            if (!steps.empty() && steps.back().true_correction_rel == 0.0 &&
                steps.back().outer_before > 0.0) {
                steps.back().true_correction_rel = rel / steps.back().outer_before;
            }
            if (!std::isfinite(rel)) throw std::runtime_error("true residual non-finite");
            if (rel <= outer_tolerance) {
                converged = true;
                break;
            }
            if (outer == max_outer) break;

            const auto correction_start = Clock::now();
            const auto correction = persistent.solve(to_float(residual), inner_iterations);
            const double one_wall_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - correction_start).count();

            Step step;
            step.outer_before = rel;
            step.recursive_rel = correction.recursive_relative_residual;
            step.gpu_solve_ms = correction.solve_ms;
            step.wall_ms = one_wall_ms;
            step.breakdown = correction.breakdown;
            if (correction.breakdown) {
                breakdown = true;
                steps.push_back(step);
                break;
            }

            const auto delta = to_double(correction.solution_aos);
            for (std::size_t i = 0U; i < x.size(); ++i) x[i] += delta[i];
            steps.push_back(step);
            gpu_inner_solve_ms += correction.solve_ms;
            correction_wall_ms += one_wall_ms;
            total_inner_iterations += correction.iterations;
            total_l0_operator_applies += correction.total_l0_operator_applies;
        }
        const double solve_wall_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - solve_start).count();
        const double total_case_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - case_start).count();
        const double final_rel = outer_history.empty() ? 1.0 : outer_history.back();

        for (std::size_t i = 0U; i < outer_history.size(); ++i) {
            std::cout << std::scientific << std::setprecision(9)
                      << "outer_true_residual[" << i << "]=" << outer_history[i];
            if (i > 0U) std::cout << " outer_q=" << outer_history[i] / outer_history[i - 1U];
            std::cout << '\n';
        }
        for (std::size_t i = 0U; i < steps.size(); ++i) {
            const auto& s = steps[i];
            std::cout << std::scientific << std::setprecision(9)
                      << "correction[" << i << "] recursive_rel=" << s.recursive_rel
                      << " true_correction_rel=" << s.true_correction_rel
                      << " breakdown=" << (s.breakdown ? "true" : "false")
                      << std::fixed << std::setprecision(6)
                      << " gpu_solve_ms=" << s.gpu_solve_ms
                      << " wall_ms=" << s.wall_ms << '\n';
        }

        std::cout << std::scientific << std::setprecision(9)
                  << "converged=" << (converged ? "true" : "false")
                  << " breakdown=" << (breakdown ? "true" : "false")
                  << " final_true_relative_residual=" << final_rel << '\n'
                  << std::fixed << std::setprecision(6)
                  << "outer_corrections=" << steps.size()
                  << " total_inner_iterations=" << total_inner_iterations
                  << " total_L0_operator_applies=" << total_l0_operator_applies << '\n'
                  << "cpu_hierarchy_setup_ms=" << hierarchy.production_setup_ms
                  << " validation_oracle_ms=" << hierarchy.validation_oracle_ms
                  << " fine_gpu_context_setup_ms=" << fine_gpu_setup_ms
                  << " deep_gpu_persistent_setup_ms=" << deep_gpu_setup_ms << '\n'
                  << "setup_A1_offdiagonal_ms=" << hierarchy.stages.actual_a1_offdiagonal_ms
                  << " setup_A2_ms=" << hierarchy.stages.a2_ms
                  << " setup_P2_ms=" << hierarchy.stages.p2_ms
                  << " setup_A3_ms=" << hierarchy.stages.bottom_ms << '\n'
                  << "gpu_inner_solve_ms=" << gpu_inner_solve_ms
                  << " accurate_FP64_residual_ms=" << accurate_residual_ms
                  << " correction_wall_ms=" << correction_wall_ms
                  << " solve_wall_ms=" << solve_wall_ms << '\n'
                  << "persistent_device_bytes=" << persistent.device_bytes_total()
                  << " temporary_A1_logical_bytes=" << hierarchy.temporary_a1_logical_bytes
                  << " total_case_wall_ms=" << total_case_ms << '\n';

        return hierarchy.oracle.accept && converged && !breakdown;
    } catch (const std::exception& e) {
        std::cout << "case_failed=true failure_stage=" << stage
                  << " error=" << e.what() << '\n';
        return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "all";
        const double outer_tolerance = argc > 2 ? parse_outer_tolerance(argv[2]) : 1.0e-6;
        const std::size_t inner_iterations = argc > 3
            ? parse_size(argv[3], "inner iterations") : 5U;
        const std::size_t max_outer = argc > 4
            ? parse_size(argv[4], "max outer corrections") : 8U;
        const int block_y = argc > 5 ? std::stoi(argv[5]) : 4;
        const std::size_t target_nodes = argc > 6
            ? parse_size(argv[6], "target nodes") : 12U;
        const std::size_t min_nodes = argc > 7
            ? parse_size(argv[7], "minimum nodes") : 4U;
        if (inner_iterations > 64U || max_outer > 32U || block_y <= 0 ||
            min_nodes > target_nodes) {
            throw std::invalid_argument("invalid M5 generalization-suite options");
        }

        const auto cases = make_cases();
        std::size_t selected = 0U;
        std::size_t passed = 0U;
        std::size_t failed = 0U;

        std::cout << "GFSS M5 structured-elasticity problem-generalization suite\n"
                  << "policy=frozen_M5_FP64_defect_FP32_MGPCG\n"
                  << "constraint=x0_zero_dirichlet\n"
                  << "no_per_case_tuning=true\n"
                  << "inner_iterations=" << inner_iterations
                  << " max_outer=" << max_outer
                  << " block_y=" << block_y
                  << " target_nodes=" << target_nodes
                  << " min_nodes=" << min_nodes
                  << " outer_tolerance=" << std::scientific << outer_tolerance << '\n'
                  << "selector=" << selector << '\n';

        for (const auto& test : cases) {
            if (selector != "all" && selector != test.name) continue;
            ++selected;
            std::cout << "\n========================================\n";
            if (run_case(test, outer_tolerance, inner_iterations, max_outer,
                         block_y, target_nodes, min_nodes)) {
                ++passed;
            } else {
                ++failed;
            }
        }

        if (selected == 0U) {
            std::cerr << "error: unknown suite case '" << selector << "'\n"
                      << "available cases:";
            for (const auto& test : cases) std::cerr << ' ' << test.name;
            std::cerr << " all\n";
            return 1;
        }

        std::cout << "\n========================================\n"
                  << "suite_selected=" << selected
                  << " suite_passed=" << passed
                  << " suite_failed=" << failed << '\n';
        return failed == 0U ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_problem_generalization_suite_bench "
                  << "[all|case [outer_tol=1e-6 [inner_iterations=5 [max_outer=8 "
                  << "[block_y=4 [target_nodes=12 [min_nodes=4]]]]]]]\n";
        return 1;
    }
}
