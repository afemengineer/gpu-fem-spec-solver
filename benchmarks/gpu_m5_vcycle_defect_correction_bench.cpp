// M5 GPU productionization stage 9: close the FP32 Krylov residual gap with
// the already-validated M4 defect-correction architecture. The accumulated
// solution and authoritative defect are FP64 on the host. Each correction
// equation is solved by a fixed-iteration FP32 MG-PCG using the complete
// persistent 5x1x1 V-cycle, then accumulated back into the FP64 iterate.
//
// This first target is a numerical staging benchmark: the fine-level GPU
// context is reused, but solve_pcg_vcycle_5x1x1_fixed currently uploads and
// allocates its deep hierarchy payload on every outer correction. Its reported
// GPU solve_ms excludes that re-upload; correction_wall_ms exposes the current
// staging overhead. If the outer convergence closes, the next step is to hoist
// the deep payload into a truly persistent correction context.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_p1_block6_setup.hpp"
#include "m5_l2_dense_setup.hpp"
#include "gfss/gpu_m5_fine_level.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using DefectClock = std::chrono::steady_clock;

std::vector<float> defect_to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

std::vector<double> defect_to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

std::vector<float> defect_bottom_lower_to_float(const DenseCholesky& factor) {
    return defect_to_float(factor.lower);
}

struct DefectStep {
    std::size_t outer_index{0U};
    double outer_residual_before{0.0};
    double inner_recursive_relative_residual{0.0};
    double inner_true_relative_residual{0.0};
    double gpu_solve_ms{0.0};
    double correction_wall_ms{0.0};
    bool breakdown{false};
};

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t inner_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 11U;
        const std::size_t max_outer = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 6U;
        const int block_y = argc > 3 ? std::stoi(argv[3]) : 4;
        const std::size_t target_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 12U;
        const std::size_t min_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 4U;
        const double outer_tolerance = argc > 6 ? std::stod(argv[6]) : 1.0e-6;
        if (inner_iterations == 0U || inner_iterations > 256U ||
            max_outer == 0U || max_outer > 32U || block_y <= 0 ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes ||
            !(outer_tolerance > 0.0) || outer_tolerance >= 1.0) {
            throw std::invalid_argument("invalid M5 defect-correction options");
        }

        constexpr std::size_t nu0 = 5U;
        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto setup_start = DefectClock::now();

        auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
        const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
            mesh, material, space0);
        const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
        const Apply apply0 = [&](const Vec& x) { return apply_fine_clamped(mesh, material, x); };
        const double lambda0 = estimate_lambda_max(apply0, fine_inverse, 8U);
        const double omega0 = kSaDampingNumerator / lambda0;
        const FineSmoothedTransfer transfer0{mesh, material, space0, fine_inverse, omega0, m0};
        const Apply apply1 = [&](const Vec& x) {
            return transfer0.restrict_transpose(apply0(transfer0.prolong(x)));
        };

        const auto graph1_tentative = graph_from_variable_blocks(tentative_a1);
        const auto candidates1 = make_level1_candidates(space0);
        const auto block1 = build_exact_l1_block_metric(
            mesh, material, space0, graph1_tentative, fine_inverse, omega0);
        const double block1_oracle_error = audit_l1_block_metric(block1, apply1);
        const double lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
        const double omega1 = kSaDampingNumerator / lambda1;

        double p0_support_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(
            mesh, material, space0, fine_inverse, omega0, p0_support_ms);
        const auto element_supports = build_element_support_index(mesh, fine_supports);
        const auto actual_a1_offdiagonal = accumulate_combined_actual_a1_offdiagonal_blocks(
            mesh, material, fine_supports, element_supports);
        const auto strength1 = build_combined_strength_graph(
            graph1_tentative, block1, actual_a1_offdiagonal, strength_threshold);
        const auto transfer1_tentative = build_candidate_transfer(
            strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
        const L1BlockSmoothedTransfer transfer1{
            transfer1_tentative, apply1, block1, omega1, m1};
        const Apply apply2 = [&](const Vec& x) {
            return transfer1.restrict_transpose(apply1(transfer1.prolong(x)));
        };

        const auto local_a1_apply_lambda = [&](const LocalColumns& x) {
            return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
        };
        const std::function<LocalColumns(const LocalColumns&)> local_a1_apply =
            local_a1_apply_lambda;
        const auto l2_basis = build_smoothed_candidate_supports(
            transfer1_tentative, strength1.graph, block1,
            omega1, m1, local_a1_apply_lambda);
        const auto block2 = build_metric_from_local_supports(
            transfer1_tentative, block1, l2_basis, local_a1_apply_lambda);
        const double block2_oracle_error = audit_l1_block_metric(block2, apply2);
        const double lambda2 = estimate_lambda_max_l1_block(apply2, block2, 8U);
        const double omega2 = kSaDampingNumerator / lambda2;

        const auto p1 = m5_p1_setup::assemble_dual_order_block6(
            transfer1_tentative, block1, l2_basis);
        const auto inverse1 = m5_l2_setup::inverse_blocks_6x6_fp32(block1);
        const auto inverse2 = m5_l2_setup::inverse_blocks_6x6_fp32(block2);
        const auto a2 = m5_l2_setup::assemble_dense_a2(
            transfer1_tentative, block1, l2_basis, local_a1_apply);

        const auto transfer2_tentative = build_candidate_transfer(
            transfer1_tentative.coarse_graph,
            transfer1_tentative.coarse_candidates,
            target_nodes,
            min_nodes,
            1.0e-10);
        const L1BlockSmoothedTransfer transfer2{
            transfer2_tentative, apply2, block2, omega2, m2};
        const auto local_a2_apply_lambda = [&](const LocalColumns& x) {
            return apply_local_a2_columns(x, l2_basis, block1, local_a1_apply_lambda);
        };
        const std::function<LocalColumns(const LocalColumns&)> local_a2_apply =
            local_a2_apply_lambda;
        const auto bottom_basis = build_smoothed_candidate_supports(
            transfer2_tentative,
            transfer1_tentative.coarse_graph,
            block2,
            omega2,
            m2,
            local_a2_apply_lambda);
        const auto p2 = m5_l2_setup::assemble_dense_p2(
            transfer2_tentative, block2, bottom_basis);
        const auto bottom = build_local_bottom(
            transfer2_tentative, block2, bottom_basis, local_a2_apply);
        const Apply apply3_nested = [&](const Vec& x) {
            return transfer2.restrict_transpose(apply2(transfer2.prolong(x)));
        };
        const double bottom_oracle_error = bottom_local_oracle_error(bottom, apply3_nested);
        const auto setup_stop = DefectClock::now();

        const auto rhs = make_rhs(mesh);
        const double rhs_norm = norm(rhs);
        if (!(rhs_norm > 0.0)) throw std::runtime_error("M5 defect-correction RHS is zero");

        gfss::GpuM5FineLevelContext gpu_context(
            mesh, material, space0, omega0, lambda0, block_y);
        const auto a2_fp32 = m5_l2_setup::to_float(a2.fp64);
        const auto p2_fp32 = m5_l2_setup::to_float(p2.fp64);
        const auto bottom_fp32 = defect_bottom_lower_to_float(bottom.factor);

        Vec x(rhs.size(), 0.0);
        Vec residual(rhs.size(), 0.0);
        std::vector<double> outer_history;
        std::vector<DefectStep> steps;
        outer_history.reserve(max_outer + 1U);
        steps.reserve(max_outer);
        double accurate_residual_ms = 0.0;
        double gpu_solve_ms = 0.0;
        double correction_wall_ms = 0.0;
        std::size_t total_inner_iterations = 0U;
        std::size_t total_l0_operator_applies = 0U;
        bool converged = false;
        bool breakdown = false;

        const auto solve_wall_start = DefectClock::now();
        for (std::size_t outer = 0; outer <= max_outer; ++outer) {
            const auto residual_start = DefectClock::now();
            const auto ax = apply0(x);
            for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
            const double relative_residual = norm(residual) / rhs_norm;
            accurate_residual_ms += std::chrono::duration<double, std::milli>(
                DefectClock::now() - residual_start).count();
            outer_history.push_back(relative_residual);
            if (!std::isfinite(relative_residual)) {
                throw std::runtime_error("M5 defect-correction FP64 residual non-finite");
            }
            if (relative_residual <= outer_tolerance) {
                converged = true;
                break;
            }
            if (outer == max_outer) break;

            const double residual_norm = norm(residual);
            const auto correction_wall_start = DefectClock::now();
            const auto correction = gpu_context.solve_pcg_vcycle_5x1x1_fixed(
                defect_to_float(residual),
                inner_iterations,
                nu0,
                m0,
                inverse1,
                lambda1,
                transfer1_tentative.aggregates.size(),
                p1.forward_row_offsets,
                p1.forward_column_indices,
                p1.forward_values_row_major,
                p1.transpose_column_offsets,
                p1.transpose_row_indices,
                p1.transpose_values_q_r_entry,
                a2_fp32,
                inverse2,
                lambda2,
                p2_fp32,
                transfer2_tentative.coarse_dofs,
                bottom_fp32,
                1);
            const double one_wall_ms = std::chrono::duration<double, std::milli>(
                DefectClock::now() - correction_wall_start).count();

            DefectStep step;
            step.outer_index = outer;
            step.outer_residual_before = relative_residual;
            step.inner_recursive_relative_residual = correction.recursive_relative_residual;
            step.gpu_solve_ms = correction.median_solve_ms;
            step.correction_wall_ms = one_wall_ms;
            step.breakdown = correction.breakdown;

            if (correction.breakdown) {
                breakdown = true;
                steps.push_back(step);
                break;
            }
            const auto delta = defect_to_double(correction.solution_aos);
            const auto adelta = apply0(delta);
            Vec inner_true_residual(residual.size(), 0.0);
            for (std::size_t i = 0; i < residual.size(); ++i) {
                inner_true_residual[i] = residual[i] - adelta[i];
                x[i] += delta[i];
            }
            step.inner_true_relative_residual =
                norm(inner_true_residual) / std::max(residual_norm, 1.0e-300);
            steps.push_back(step);

            gpu_solve_ms += correction.median_solve_ms;
            correction_wall_ms += one_wall_ms;
            total_inner_iterations += correction.iterations;
            total_l0_operator_applies += correction.total_l0_operator_applies;
        }
        const double solve_wall_ms = std::chrono::duration<double, std::milli>(
            DefectClock::now() - solve_wall_start).count();

        const bool hierarchy_oracle_ok =
            block1_oracle_error <= 1.0e-10 &&
            block2_oracle_error <= 1.0e-10 &&
            bottom_oracle_error <= 1.0e-10;
        const double final_true_residual = outer_history.empty()
            ? 1.0 : outer_history.back();

        std::cout << "GFSS M5 FP64-outer / FP32-MGPCG defect correction\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "rhs=physical_uniform_z_xmax\n"
                  << "outer_state=CPU_FP64_solution_and_true_residual\n"
                  << "inner_correction=GPU_FP32_fixed_iteration_MGPCG_5x1x1\n"
                  << "deep_hierarchy_reuploaded_each_outer=true\n"
                  << "performance_status=numerical_staging_not_final_persistent_outer\n"
                  << "inner_iterations_per_outer=" << inner_iterations
                  << " max_outer=" << max_outer << '\n'
                  << std::scientific << std::setprecision(9)
                  << "outer_tolerance=" << outer_tolerance << '\n'
                  << std::fixed << std::setprecision(6)
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L2_dofs=" << block2.dofs()
                  << " L3_dofs=" << bottom.factor.n << '\n'
                  << "lambda0=" << lambda0 << " omega0=" << omega0
                  << " lambda1=" << lambda1 << " omega1=" << omega1
                  << " lambda2=" << lambda2 << " omega2=" << omega2 << '\n'
                  << "P0_support_setup_ms=" << p0_support_ms
                  << " P1_block6_assembly_ms=" << p1.assembly_ms
                  << " A2_dense_assembly_ms=" << a2.assembly_ms
                  << " P2_dense_assembly_ms=" << p2.assembly_ms
                  << " bottom_assembly_ms=" << bottom.assembly_ms
                  << " total_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error
                  << " L2_block_vs_nested_relative_error=" << block2_oracle_error
                  << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error << '\n';

        for (std::size_t i = 0; i < outer_history.size(); ++i) {
            std::cout << "outer_true_residual[" << i << "]=" << outer_history[i];
            if (i > 0U) {
                std::cout << " outer_q=" << outer_history[i] / outer_history[i - 1U];
            }
            std::cout << '\n';
        }
        for (const auto& step : steps) {
            std::cout << "correction[" << step.outer_index << "]"
                      << " recursive_rel=" << step.inner_recursive_relative_residual
                      << " true_correction_rel=" << step.inner_true_relative_residual
                      << " breakdown=" << (step.breakdown ? "true" : "false")
                      << std::fixed << std::setprecision(6)
                      << " gpu_solve_ms=" << step.gpu_solve_ms
                      << " wall_ms=" << step.correction_wall_ms
                      << std::scientific << std::setprecision(9) << '\n';
        }

        std::cout << "hierarchy_oracle_accept=" << (hierarchy_oracle_ok ? "true" : "false")
                  << " breakdown=" << (breakdown ? "true" : "false")
                  << " converged=" << (converged ? "true" : "false")
                  << " final_true_relative_residual=" << final_true_residual << '\n'
                  << std::fixed << std::setprecision(6)
                  << "outer_corrections=" << steps.size()
                  << " total_inner_iterations=" << total_inner_iterations
                  << " total_L0_operator_applies=" << total_l0_operator_applies << '\n'
                  << "gpu_inner_solve_ms=" << gpu_solve_ms
                  << " correction_wall_ms=" << correction_wall_ms
                  << " accurate_FP64_residual_ms=" << accurate_residual_ms
                  << " staged_solve_wall_ms=" << solve_wall_ms << '\n'
                  << "gpu_inner_solve_ms_excludes_hierarchy_reupload=true\n";

        return hierarchy_oracle_ok && !breakdown && converged ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_vcycle_defect_correction_bench "
                  << "[inner_iterations=11 [max_outer=6 [block_y=4 [target_nodes=12 [min_nodes=4 [outer_tol=1e-6]]]]]]\n";
        return 1;
    }
}
