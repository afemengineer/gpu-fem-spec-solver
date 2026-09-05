// End-to-end M5 production-candidate probe: exact sparse recursive setup plus
// persistent FP32 recursive-tail MG-PCG inside the established FP64 defect loop.
// The fixed-depth staging remains untouched and serves as the A/B baseline.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_p1_block6_setup.hpp"
#include "m5_l2_dense_setup.hpp"
#include "m5_fast_hierarchy_setup.hpp"
#include "m5_materialized_a1_setup.hpp"
#include "m5_parallel_actual_a1_setup.hpp"
#include "m5_sparse_a2_setup.hpp"
#include "m5_recursive_sparse_tail.hpp"
#include "m5_recursive_tail_gpu_bridge.hpp"
#include "m5_persistent_recursive_vpcg_staging.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct CaseDef {
    const char* name;
    gfss::StructuredHexMesh mesh;
    gfss::Material material;
};

struct CorrectionStep {
    double outer_before{0.0};
    double recursive_rel{0.0};
    double true_correction_rel{0.0};
    double gpu_solve_ms{0.0};
    double wall_ms{0.0};
    bool breakdown{false};
};

std::size_t parse_size(const char* text, const char* name) {
    const auto value = std::stoull(text);
    if (value == 0ULL) throw std::invalid_argument(std::string(name) + " must be positive");
    return static_cast<std::size_t>(value);
}

double parse_tolerance(const char* text) {
    const double value = std::stod(text);
    if (!(value > 0.0) || !(value < 1.0) || !std::isfinite(value)) {
        throw std::invalid_argument("outer tolerance must be in (0,1)");
    }
    return value;
}

std::vector<double> uniform_z_xmax_rhs(const gfss::StructuredHexMesh& mesh) {
    std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const double count = static_cast<double>(mesh.ny + 1U) *
                         static_cast<double>(mesh.nz + 1U);
    const double value = -1.0 / count;
    for (std::uint32_t k = 0U; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0U; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = value;
        }
    }
    return rhs;
}

std::vector<float> to_float(const std::vector<double>& x) {
    std::vector<float> out(x.size(), 0.0f);
    for (std::size_t i = 0U; i < x.size(); ++i) out[i] = static_cast<float>(x[i]);
    return out;
}

std::vector<double> to_double(const std::vector<float>& x) {
    std::vector<double> out(x.size(), 0.0);
    for (std::size_t i = 0U; i < x.size(); ++i) out[i] = static_cast<double>(x[i]);
    return out;
}

std::vector<CaseDef> cases() {
    constexpr double E = 210.0e9;
    return {
        {"thin_plate", {64U, 64U, 8U, 1.0, 1.0, 0.125}, {E, 0.30}},
        {"baseline_cube", {64U, 64U, 64U, 1.0, 1.0, 1.0}, {E, 0.30}},
    };
}

bool run_case(const CaseDef& test,
              double outer_tolerance,
              std::size_t inner_iterations,
              std::size_t max_outer,
              int block_y,
              std::size_t target_nodes,
              std::size_t min_nodes,
              std::size_t dense_bottom_threshold) {
    constexpr std::size_t nu0 = 5U;
    constexpr std::size_t m0 = 1U;
    constexpr std::size_t m1 = 2U;
    constexpr double strength_threshold = 0.05;

    const auto case_start = Clock::now();
    std::string stage = "recursive_hierarchy_build";
    try {
        std::cout << "case=" << test.name << '\n'
                  << "mesh=" << test.mesh.nx << 'x' << test.mesh.ny << 'x' << test.mesh.nz
                  << " physical=" << test.mesh.lx << 'x' << test.mesh.ly << 'x' << test.mesh.lz
                  << " elements=" << test.mesh.element_count()
                  << " dofs=" << test.mesh.dof_count() << '\n'
                  << std::scientific << std::setprecision(9)
                  << "material_E=" << test.material.young_modulus
                  << " poisson=" << test.material.poisson_ratio << '\n';

        const auto hierarchy_start = Clock::now();
        auto graph0 = gfss::build_structured_hex_nodal_graph_x0(test.mesh);
        auto space0 = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
        const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
            test.mesh, test.material, space0);
        const auto fine_inverse = build_fine_inverse_diagonal(test.mesh, test.material, space0);
        const Apply apply0 = [&](const Vec& x) {
            return apply_fine_clamped(test.mesh, test.material, x);
        };
        const double lambda0 = estimate_lambda_max(apply0, fine_inverse, 8U);
        const double omega0 = kSaDampingNumerator / lambda0;
        const FineSmoothedTransfer transfer0{
            test.mesh, test.material, space0, fine_inverse, omega0, m0};
        const Apply apply1 = [&](const Vec& x) {
            return transfer0.restrict_transpose(apply0(transfer0.prolong(x)));
        };

        const auto graph1_tentative = graph_from_variable_blocks(tentative_a1);
        const auto candidates1 = make_level1_candidates(space0);
        auto block1 = build_exact_l1_block_metric(
            test.mesh, test.material, space0, graph1_tentative, fine_inverse, omega0);

        double p0_support_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(
            test.mesh, test.material, space0, fine_inverse, omega0, p0_support_ms);
        const auto element_supports = build_element_support_index(test.mesh, fine_supports);
        const auto parallel_a1 = m5_parallel_a1::assemble(
            test.mesh, test.material, fine_supports, element_supports);
        const auto temporary_a1 = m5_materialized_a1::build(block1, parallel_a1.blocks);
        const Apply apply1_materialized = [&](const Vec& x) {
            return m5_materialized_a1::apply_vector(temporary_a1, block1, x);
        };
        const double lambda1 = estimate_lambda_max_l1_block(apply1_materialized, block1, 8U);
        const double omega1 = kSaDampingNumerator / lambda1;
        const auto strength1 = build_combined_strength_graph(
            graph1_tentative, block1, parallel_a1.blocks, strength_threshold);
        auto transfer1 = build_candidate_transfer(
            strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
        const auto local_a1 = [&](const LocalColumns& x) {
            return m5_materialized_a1::apply_columns(temporary_a1, block1, x);
        };
        const auto l2_basis = m5_fast_setup::build_smoothed_supports_parallel(
            transfer1, strength1.graph, block1, omega1, m1, local_a1);
        const auto applied_l2_basis = m5_fast_setup::apply_supports_parallel(
            l2_basis, local_a1);
        auto block2 = m5_fast_setup::metric_from_cached_applied(
            transfer1, block1, l2_basis, applied_l2_basis);

        const auto p1 = m5_p1_setup::assemble_dual_order_block6(transfer1, block1, l2_basis);
        const auto inverse1 = m5_l2_setup::inverse_blocks_6x6_fp32(block1);
        const std::size_t temporary_a1_bytes = temporary_a1.logical_bytes;

        const auto a2_start = Clock::now();
        auto sparse_a2 = m5_sparse_a2::assemble_from_cached_applied(
            transfer1, block1, l2_basis, applied_l2_basis);
        const double direct_sparse_a2_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - a2_start).count();

        // Exact top-level sparse-vs-nested oracle before ownership moves into tail.
        const L1BlockSmoothedTransfer transfer1_nested{
            transfer1, apply1, block1, omega1, m1};
        const Apply apply2_nested = [&](const Vec& x) {
            return transfer1_nested.restrict_transpose(apply1(transfer1_nested.prolong(x)));
        };
        double a2_oracle = 0.0;
        for (const double phase : {0.31, 0.73, 1.11}) {
            Vec probe(sparse_a2.n, 0.0);
            for (std::size_t i = 0U; i < probe.size(); ++i) {
                const double t = static_cast<double>(i + 1U);
                probe[i] = std::sin(0.017 * t + phase) +
                           0.23 * std::cos(0.039 * t - 0.31 * phase);
            }
            const auto sparse_y = m5_sparse_a2::apply_fp64(sparse_a2, transfer1, probe);
            const auto nested_y = apply2_nested(probe);
            Vec diff(probe.size(), 0.0);
            for (std::size_t i = 0U; i < diff.size(); ++i) diff[i] = sparse_y[i] - nested_y[i];
            a2_oracle = std::max(a2_oracle, norm(diff) / std::max(norm(nested_y), 1.0e-300));
        }

        auto tail = m5_recursive_tail::build(
            std::move(transfer1), std::move(block2), std::move(sparse_a2),
            2U, target_nodes, min_nodes, dense_bottom_threshold, 12U);
        const double hierarchy_setup_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - hierarchy_start).count();

        stage = "gpu_payload_export";
        const auto export_start = Clock::now();
        auto payloads = m5_recursive_tail_gpu_bridge::export_payloads(tail);
        auto bottom_inverse = m5_recursive_tail_gpu_bridge::bottom_inverse_fp32(tail);
        const double payload_export_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - export_start).count();

        const bool hierarchy_oracle_accept =
            a2_oracle <= 1.0e-10 &&
            tail.bottom_sparse_apply_relative_error <= 1.0e-12 &&
            tail.bottom_solve_relative_residual <= 1.0e-10;

        std::cout << "hierarchy_L0_dofs=" << test.mesh.dof_count()
                  << " hierarchy_L1_dofs=" << space0.coarse_dofs;
        for (const auto& level : tail.levels) {
            std::cout << " hierarchy_L" << level.level << "_dofs=" << level.op.n;
        }
        std::cout << '\n'
                  << std::scientific << std::setprecision(9)
                  << "lambda0=" << lambda0 << " lambda1=" << lambda1
                  << " A2_vs_nested_relative_error=" << a2_oracle
                  << " bottom_sparse_vs_dense_relative_error="
                  << tail.bottom_sparse_apply_relative_error
                  << " bottom_solve_relative_residual=" << tail.bottom_solve_relative_residual
                  << " hierarchy_oracle_accept="
                  << (hierarchy_oracle_accept ? "true" : "false") << '\n';

        for (std::size_t i = 0U; i < tail.levels.size(); ++i) {
            const auto& level = tail.levels[i];
            std::cout << std::fixed << std::setprecision(6)
                      << "level=" << level.level
                      << " nodes=" << level.op.nodes
                      << " dofs=" << level.op.n
                      << " density=" << tail.telemetry[i].scalar_density
                      << " runtime_representation=";
            if (i + 1U == tail.levels.size()) {
                std::cout << "dense_inverse_fp32";
            } else if (payloads[i].operator_kind ==
                       gfss::M5RecursiveTailOperatorKind::structural_scalar_csr_fp32) {
                std::cout << "structural_scalar_CSR_fp32";
            } else {
                std::cout << "dense_fp32";
            }
            if (i + 1U < tail.levels.size()) {
                std::cout << " next_dofs=" << tail.levels[i + 1U].op.n
                          << " P_nnz=" << payloads[i].p_values.size();
            }
            std::cout << '\n';
        }

        const auto rhs = uniform_z_xmax_rhs(test.mesh);
        const double rhs_norm = norm(rhs);
        if (!(rhs_norm > 0.0)) throw std::runtime_error("recursive production RHS is zero");

        stage = "fine_gpu_context";
        const auto fine_start = Clock::now();
        gfss::GpuM5FineLevelContext fine_context(
            test.mesh, test.material, space0, omega0, lambda0, block_y);
        const double fine_gpu_setup_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - fine_start).count();

        stage = "recursive_gpu_persistent";
        const auto persistent_start = Clock::now();
        gfss::M5PersistentRecursivePcgStaging persistent(
            fine_context, nu0, m0,
            inverse1, lambda1,
            tail.levels.front().op.nodes,
            p1.forward_row_offsets,
            p1.forward_column_indices,
            p1.forward_values_row_major,
            p1.transpose_column_offsets,
            p1.transpose_row_indices,
            p1.transpose_values_q_r_entry,
            payloads, bottom_inverse);
        const double persistent_setup_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - persistent_start).count();

        stage = "warmup";
        persistent.warmup(to_float(rhs), inner_iterations);

        stage = "solve";
        Vec x(rhs.size(), 0.0);
        Vec residual(rhs.size(), 0.0);
        std::vector<double> outer_history;
        std::vector<CorrectionStep> steps;
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
            if (!std::isfinite(rel)) throw std::runtime_error("true residual became non-finite");
            if (rel <= outer_tolerance) {
                converged = true;
                break;
            }
            if (outer == max_outer) break;

            const auto correction_start = Clock::now();
            const auto correction = persistent.solve(to_float(residual), inner_iterations);
            const double one_wall_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - correction_start).count();

            CorrectionStep step;
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

        const double final_rel = outer_history.empty() ? 1.0 : outer_history.back();
        std::cout << std::scientific << std::setprecision(9)
                  << "converged=" << (converged ? "true" : "false")
                  << " breakdown=" << (breakdown ? "true" : "false")
                  << " final_true_relative_residual=" << final_rel << '\n'
                  << std::fixed << std::setprecision(6)
                  << "outer_corrections=" << steps.size()
                  << " total_inner_iterations=" << total_inner_iterations
                  << " total_L0_operator_applies=" << total_l0_operator_applies << '\n'
                  << "recursive_hierarchy_setup_ms=" << hierarchy_setup_ms
                  << " direct_sparse_A2_ms=" << direct_sparse_a2_ms
                  << " recursive_tail_setup_ms=" << tail.total_ms
                  << " gpu_payload_export_ms=" << payload_export_ms
                  << " fine_gpu_context_setup_ms=" << fine_gpu_setup_ms
                  << " recursive_gpu_persistent_setup_ms=" << persistent_setup_ms << '\n'
                  << "gpu_inner_solve_ms=" << gpu_inner_solve_ms
                  << " accurate_FP64_residual_ms=" << accurate_residual_ms
                  << " correction_wall_ms=" << correction_wall_ms
                  << " solve_wall_ms=" << solve_wall_ms << '\n'
                  << "persistent_device_bytes=" << persistent.device_bytes_total()
                  << " recursive_tail_device_bytes=" << persistent.recursive_tail_device_bytes()
                  << " temporary_A1_logical_bytes=" << temporary_a1_bytes
                  << " total_case_wall_ms=" << total_case_ms << '\n';

        return hierarchy_oracle_accept && converged && !breakdown;
    } catch (const std::exception& e) {
        std::cout << "case_failed=true failure_stage=" << stage
                  << " error=" << e.what() << '\n';
        return false;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "thin_plate";
        const double outer_tolerance = argc > 2 ? parse_tolerance(argv[2]) : 1.0e-6;
        const std::size_t inner_iterations = argc > 3
            ? parse_size(argv[3], "inner iterations") : 5U;
        const std::size_t max_outer = argc > 4
            ? parse_size(argv[4], "max outer corrections") : 12U;
        const int block_y = argc > 5 ? std::stoi(argv[5]) : 4;
        const std::size_t target_nodes = argc > 6 ? parse_size(argv[6], "target nodes") : 12U;
        const std::size_t min_nodes = argc > 7 ? parse_size(argv[7], "min nodes") : 4U;
        const std::size_t bottom_threshold = argc > 8
            ? parse_size(argv[8], "bottom threshold") : 512U;
        if (inner_iterations > 64U || max_outer > 32U || block_y <= 0 ||
            min_nodes > target_nodes) {
            throw std::invalid_argument("invalid recursive production-solver options");
        }

        std::cout << "GFSS M5 recursive production solver probe\n"
                  << "policy=FP64_defect_plus_5step_FP32_recursive_MGPCG\n"
                  << "setup=direct_sparse_A2_plus_recursive_sparse_tail\n"
                  << "no_per_case_tuning=true\n"
                  << "outer_tolerance=" << std::scientific << outer_tolerance
                  << " inner_iterations=" << std::fixed << inner_iterations
                  << " max_outer=" << max_outer
                  << " block_y=" << block_y
                  << " target_nodes=" << target_nodes
                  << " min_nodes=" << min_nodes
                  << " dense_bottom_threshold=" << bottom_threshold << '\n'
                  << "selector=" << selector << '\n';

        std::size_t selected = 0U;
        std::size_t passed = 0U;
        for (const auto& test : cases()) {
            if (selector != "all" && selector != test.name) continue;
            ++selected;
            std::cout << "\n========================================\n";
            if (run_case(test, outer_tolerance, inner_iterations, max_outer,
                         block_y, target_nodes, min_nodes, bottom_threshold)) {
                ++passed;
            }
        }
        if (selected == 0U) {
            throw std::invalid_argument("selector must be thin_plate, baseline_cube, or all");
        }
        const std::size_t failed = selected - passed;
        std::cout << "\n========================================\n"
                  << "suite_selected=" << selected
                  << " suite_passed=" << passed
                  << " suite_failed=" << failed << '\n';
        return failed == 0U ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_recursive_production_solver_bench "
                  << "[thin_plate|baseline_cube|all [outer_tol=1e-6 [inner=5 [max_outer=12 "
                  << "[block_y=4 [target_nodes=12 [min_nodes=4 [bottom=512]]]]]]]]\n";
        return 1;
    }
}
