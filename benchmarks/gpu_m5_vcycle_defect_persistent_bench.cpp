// M5 stage 11: production-shaped defect correction timing.
// Build the validated hierarchy once, upload/allocate the complete MG-PCG state
// once, warm it once, then reuse it for every FP64 outer correction.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_p1_block6_setup.hpp"
#include "m5_l2_dense_setup.hpp"
#include "m5_persistent_vpcg_staging.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using PersistClock = std::chrono::steady_clock;

std::vector<float> persist_to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0U; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

std::vector<double> persist_to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0U; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

struct PersistStep {
    double outer_before{0.0};
    double recursive_rel{0.0};
    double true_correction_rel{0.0};
    double gpu_solve_ms{0.0};
    double wall_ms{0.0};
    bool breakdown{false};
};
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t inner_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 5U;
        const std::size_t max_outer = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 8U;
        const int block_y = argc > 3 ? std::stoi(argv[3]) : 4;
        const std::size_t target_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 12U;
        const std::size_t min_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 4U;
        const double outer_tolerance = argc > 6 ? std::stod(argv[6]) : 1.0e-6;
        if (inner_iterations == 0U || inner_iterations > 64U ||
            max_outer == 0U || max_outer > 32U || block_y <= 0 ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes ||
            !(outer_tolerance > 0.0) || outer_tolerance >= 1.0) {
            throw std::invalid_argument("invalid persistent M5 defect options");
        }

        constexpr std::size_t nu0 = 5U;
        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};

        const auto hierarchy_start = PersistClock::now();
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
        const std::function<LocalColumns(const LocalColumns&)> local_a1_apply = local_a1_apply_lambda;
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
            target_nodes, min_nodes, 1.0e-10);
        const L1BlockSmoothedTransfer transfer2{
            transfer2_tentative, apply2, block2, omega2, m2};
        const auto local_a2_apply_lambda = [&](const LocalColumns& x) {
            return apply_local_a2_columns(x, l2_basis, block1, local_a1_apply_lambda);
        };
        const std::function<LocalColumns(const LocalColumns&)> local_a2_apply = local_a2_apply_lambda;
        const auto bottom_basis = build_smoothed_candidate_supports(
            transfer2_tentative, transfer1_tentative.coarse_graph,
            block2, omega2, m2, local_a2_apply_lambda);
        const auto p2 = m5_l2_setup::assemble_dense_p2(
            transfer2_tentative, block2, bottom_basis);
        const auto bottom = build_local_bottom(
            transfer2_tentative, block2, bottom_basis, local_a2_apply);
        const Apply apply3_nested = [&](const Vec& x) {
            return transfer2.restrict_transpose(apply2(transfer2.prolong(x)));
        };
        const double bottom_oracle_error = bottom_local_oracle_error(bottom, apply3_nested);
        const auto hierarchy_stop = PersistClock::now();

        const bool hierarchy_ok = block1_oracle_error <= 1.0e-10 &&
                                  block2_oracle_error <= 1.0e-10 &&
                                  bottom_oracle_error <= 1.0e-10;
        const auto rhs = make_rhs(mesh);
        const double rhs_norm = norm(rhs);
        if (!(rhs_norm > 0.0)) throw std::runtime_error("persistent M5 RHS is zero");

        gfss::GpuM5FineLevelContext fine_context(
            mesh, material, space0, omega0, lambda0, block_y);
        const auto a2_fp32 = m5_l2_setup::to_float(a2.fp64);
        const auto p2_fp32 = m5_l2_setup::to_float(p2.fp64);
        const auto bottom_fp32 = persist_to_float(bottom.factor.lower);

        const auto gpu_setup_start = PersistClock::now();
        gfss::M5PersistentPcgStaging persistent(
            fine_context, nu0, m0,
            inverse1, lambda1, transfer1_tentative.aggregates.size(),
            p1.forward_row_offsets, p1.forward_column_indices,
            p1.forward_values_row_major, p1.transpose_column_offsets,
            p1.transpose_row_indices, p1.transpose_values_q_r_entry,
            a2_fp32, inverse2, lambda2, p2_fp32,
            transfer2_tentative.coarse_dofs, bottom_fp32);
        const double gpu_hierarchy_setup_ms = std::chrono::duration<double, std::milli>(
            PersistClock::now() - gpu_setup_start).count();

        // Warm-up is explicitly outside the production solve timer and happens
        // only once for the persistent context.
        persistent.warmup(persist_to_float(rhs), inner_iterations);

        Vec x(rhs.size(), 0.0);
        Vec residual(rhs.size(), 0.0);
        std::vector<double> outer_history;
        std::vector<PersistStep> steps;
        outer_history.reserve(max_outer + 1U);
        steps.reserve(max_outer);
        double accurate_residual_ms = 0.0;
        double gpu_inner_solve_ms = 0.0;
        double correction_wall_ms = 0.0;
        std::size_t total_inner_iterations = 0U;
        std::size_t total_l0_operator_applies = 0U;
        bool converged = false;
        bool breakdown = false;

        const auto solve_start = PersistClock::now();
        for (std::size_t outer = 0U; outer <= max_outer; ++outer) {
            const auto residual_start = PersistClock::now();
            const auto ax = apply0(x);
            for (std::size_t i = 0U; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
            const double rel = norm(residual) / rhs_norm;
            accurate_residual_ms += std::chrono::duration<double, std::milli>(
                PersistClock::now() - residual_start).count();
            outer_history.push_back(rel);
            if (!std::isfinite(rel)) throw std::runtime_error("persistent M5 FP64 residual non-finite");
            if (rel <= outer_tolerance) {
                converged = true;
                break;
            }
            if (outer == max_outer) break;

            const double residual_norm = norm(residual);
            const auto one_start = PersistClock::now();
            const auto correction = persistent.solve(persist_to_float(residual), inner_iterations);
            const double one_wall = std::chrono::duration<double, std::milli>(
                PersistClock::now() - one_start).count();

            PersistStep step;
            step.outer_before = rel;
            step.recursive_rel = correction.recursive_relative_residual;
            step.gpu_solve_ms = correction.solve_ms;
            step.wall_ms = one_wall;
            step.breakdown = correction.breakdown;
            if (correction.breakdown) {
                breakdown = true;
                steps.push_back(step);
                break;
            }

            const auto delta = persist_to_double(correction.solution_aos);
            const auto adelta = apply0(delta);
            Vec true_inner(residual.size(), 0.0);
            for (std::size_t i = 0U; i < residual.size(); ++i) {
                true_inner[i] = residual[i] - adelta[i];
                x[i] += delta[i];
            }
            step.true_correction_rel = norm(true_inner) / std::max(residual_norm, 1.0e-300);
            steps.push_back(step);
            gpu_inner_solve_ms += correction.solve_ms;
            correction_wall_ms += one_wall;
            total_inner_iterations += correction.iterations;
            total_l0_operator_applies += correction.total_l0_operator_applies;
        }
        const double solve_wall_ms = std::chrono::duration<double, std::milli>(
            PersistClock::now() - solve_start).count();

        const double final_true_residual = outer_history.empty() ? 1.0 : outer_history.back();
        const double core_compute_ms = gpu_inner_solve_ms + accurate_residual_ms;

        std::cout << "GFSS M5 persistent FP64-defect / FP32-MGPCG solve\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "rhs=physical_uniform_z_xmax\n"
                  << "policy=frozen_5x1x1 inner_iterations=" << inner_iterations << '\n'
                  << "deep_hierarchy_uploaded_once=true\n"
                  << "pcg_vectors_allocated_once=true\n"
                  << "cublas_handles_created_once=true\n"
                  << "warmup_once_outside_solve_timer=true\n"
                  << "host_round_trip_per_outer=rhs_H2D_plus_solution_D2H\n"
                  << std::scientific << std::setprecision(9)
                  << "outer_tolerance=" << outer_tolerance << '\n'
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error
                  << " L2_block_vs_nested_relative_error=" << block2_oracle_error
                  << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error
                  << " hierarchy_oracle_accept=" << (hierarchy_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "hierarchy_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(hierarchy_stop - hierarchy_start).count()
                  << " gpu_persistent_setup_ms=" << gpu_hierarchy_setup_ms
                  << " persistent_device_bytes=" << persistent.device_bytes_total() << '\n';

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
                  << " final_true_relative_residual=" << final_true_residual << '\n'
                  << std::fixed << std::setprecision(6)
                  << "outer_corrections=" << steps.size()
                  << " total_inner_iterations=" << total_inner_iterations
                  << " total_L0_operator_applies=" << total_l0_operator_applies << '\n'
                  << "gpu_inner_solve_ms=" << gpu_inner_solve_ms
                  << " accurate_FP64_residual_ms=" << accurate_residual_ms
                  << " core_compute_ms=" << core_compute_ms
                  << " correction_wall_ms=" << correction_wall_ms
                  << " persistent_solve_wall_ms=" << solve_wall_ms << '\n'
                  << "setup_included_in_solve_wall=false\n";

        return hierarchy_ok && converged && !breakdown ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_vcycle_defect_persistent_bench "
                  << "[inner_iterations=5 [max_outer=8 [block_y=4 [target_nodes=12 [min_nodes=4 [outer_tol=1e-6]]]]]]\n";
        return 1;
    }
}
