// M5 stage 10: sweep fixed FP32 MG-PCG work inside the validated FP64 defect-
// correction outer loop. The expensive algebraic hierarchy is constructed once;
// each candidate then solves the same physical thin-plate problem from x=0.
// gpu_inner_solve_ms excludes the current per-correction deep hierarchy upload,
// so this benchmark selects the numerical/compute policy before persistence work.
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
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using SweepClock = std::chrono::steady_clock;

std::vector<float> to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}
std::vector<double> to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

struct PolicyResult {
    std::size_t inner_iterations{0U};
    std::size_t outer_corrections{0U};
    std::size_t total_inner_iterations{0U};
    std::size_t total_l0_operator_applies{0U};
    bool converged{false};
    bool breakdown{false};
    double final_true_residual{1.0};
    double gpu_inner_solve_ms{0.0};
    double correction_wall_ms{0.0};
    double accurate_fp64_residual_ms{0.0};
    double solve_wall_ms{0.0};
    double geometric_outer_q{std::numeric_limits<double>::infinity()};
    std::vector<double> outer_history;
    std::vector<double> inner_recursive;
    std::vector<double> inner_true;
};
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t min_inner = argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 4U;
        const std::size_t max_inner = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 11U;
        const std::size_t max_outer = argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 8U;
        const int block_y = argc > 4 ? std::stoi(argv[4]) : 4;
        const std::size_t target_nodes = argc > 5 ? static_cast<std::size_t>(std::stoull(argv[5])) : 12U;
        const std::size_t min_nodes = argc > 6 ? static_cast<std::size_t>(std::stoull(argv[6])) : 4U;
        const double outer_tolerance = argc > 7 ? std::stod(argv[7]) : 1.0e-6;
        if (min_inner == 0U || max_inner < min_inner || max_inner > 64U ||
            max_outer == 0U || max_outer > 32U || block_y <= 0 || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes || !(outer_tolerance > 0.0) ||
            outer_tolerance >= 1.0) {
            throw std::invalid_argument("invalid M5 defect iteration sweep options");
        }

        constexpr std::size_t nu0 = 5U;
        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto setup_start = SweepClock::now();

        auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
        const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(mesh, material, space0);
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
            transfer1_tentative, strength1.graph, block1, omega1, m1, local_a1_apply_lambda);
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
        const auto setup_stop = SweepClock::now();

        const bool hierarchy_ok = block1_oracle_error <= 1.0e-10 &&
                                  block2_oracle_error <= 1.0e-10 &&
                                  bottom_oracle_error <= 1.0e-10;
        const auto rhs = make_rhs(mesh);
        const double rhs_norm = norm(rhs);
        if (!(rhs_norm > 0.0)) throw std::runtime_error("M5 defect sweep RHS is zero");

        gfss::GpuM5FineLevelContext gpu_context(mesh, material, space0, omega0, lambda0, block_y);
        const auto a2_fp32 = m5_l2_setup::to_float(a2.fp64);
        const auto p2_fp32 = m5_l2_setup::to_float(p2.fp64);
        const auto bottom_fp32 = to_float(bottom.factor.lower);

        std::vector<PolicyResult> policies;
        policies.reserve(max_inner - min_inner + 1U);

        for (std::size_t inner = min_inner; inner <= max_inner; ++inner) {
            PolicyResult pr;
            pr.inner_iterations = inner;
            Vec x(rhs.size(), 0.0);
            Vec residual(rhs.size(), 0.0);
            const auto policy_start = SweepClock::now();

            for (std::size_t outer = 0; outer <= max_outer; ++outer) {
                const auto residual_start = SweepClock::now();
                const auto ax = apply0(x);
                for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
                const double rel = norm(residual) / rhs_norm;
                pr.accurate_fp64_residual_ms += std::chrono::duration<double, std::milli>(
                    SweepClock::now() - residual_start).count();
                pr.outer_history.push_back(rel);
                if (!std::isfinite(rel)) throw std::runtime_error("M5 defect sweep residual non-finite");
                if (rel <= outer_tolerance) {
                    pr.converged = true;
                    break;
                }
                if (outer == max_outer) break;

                const double residual_norm = norm(residual);
                const auto correction_start = SweepClock::now();
                const auto correction = gpu_context.solve_pcg_vcycle_5x1x1_fixed(
                    to_float(residual), inner, nu0, m0,
                    inverse1, lambda1, transfer1_tentative.aggregates.size(),
                    p1.forward_row_offsets, p1.forward_column_indices,
                    p1.forward_values_row_major, p1.transpose_column_offsets,
                    p1.transpose_row_indices, p1.transpose_values_q_r_entry,
                    a2_fp32, inverse2, lambda2, p2_fp32,
                    transfer2_tentative.coarse_dofs, bottom_fp32, 1);
                pr.correction_wall_ms += std::chrono::duration<double, std::milli>(
                    SweepClock::now() - correction_start).count();
                pr.gpu_inner_solve_ms += correction.median_solve_ms;
                pr.total_inner_iterations += correction.iterations;
                pr.total_l0_operator_applies += correction.total_l0_operator_applies;
                pr.inner_recursive.push_back(correction.recursive_relative_residual);
                if (correction.breakdown) {
                    pr.breakdown = true;
                    break;
                }

                const auto delta = to_double(correction.solution_aos);
                const auto adelta = apply0(delta);
                Vec inner_true_residual(residual.size(), 0.0);
                for (std::size_t i = 0; i < residual.size(); ++i) {
                    inner_true_residual[i] = residual[i] - adelta[i];
                    x[i] += delta[i];
                }
                pr.inner_true.push_back(
                    norm(inner_true_residual) / std::max(residual_norm, 1.0e-300));
                ++pr.outer_corrections;
            }
            pr.solve_wall_ms = std::chrono::duration<double, std::milli>(
                SweepClock::now() - policy_start).count();
            pr.final_true_residual = pr.outer_history.empty() ? 1.0 : pr.outer_history.back();
            if (pr.outer_history.size() > 1U) {
                double log_sum = 0.0;
                std::size_t q_count = 0U;
                for (std::size_t i = 1U; i < pr.outer_history.size(); ++i) {
                    const double q = pr.outer_history[i] / pr.outer_history[i - 1U];
                    if (q > 0.0 && std::isfinite(q)) {
                        log_sum += std::log(q);
                        ++q_count;
                    }
                }
                if (q_count > 0U) pr.geometric_outer_q = std::exp(log_sum / static_cast<double>(q_count));
            }
            policies.push_back(std::move(pr));
        }

        const PolicyResult* best = nullptr;
        for (const auto& pr : policies) {
            if (!pr.converged || pr.breakdown) continue;
            if (best == nullptr || pr.gpu_inner_solve_ms < best->gpu_inner_solve_ms) best = &pr;
        }

        std::cout << "GFSS M5 FP64-defect / FP32-MGPCG inner-iteration sweep\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "hierarchy=frozen_5x1x1_dual_block6_dense_A2_P2_L3_direct\n"
                  << "hierarchy_constructed_once=true\n"
                  << "deep_hierarchy_reuploaded_each_correction=true\n"
                  << "policy_objective=min_gpu_inner_compute_ms_subject_to_true_FP64_1e-6\n"
                  << "inner_range=" << min_inner << ".." << max_inner
                  << " max_outer=" << max_outer << '\n'
                  << std::scientific << std::setprecision(9)
                  << "outer_tolerance=" << outer_tolerance << '\n'
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error
                  << " L2_block_vs_nested_relative_error=" << block2_oracle_error
                  << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error
                  << " hierarchy_oracle_accept=" << (hierarchy_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "hierarchy_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << " persistent_base_context_reused=true\n\n";

        for (const auto& pr : policies) {
            std::cout << "POLICY inner_iterations=" << pr.inner_iterations
                      << " converged=" << (pr.converged ? "true" : "false")
                      << " breakdown=" << (pr.breakdown ? "true" : "false")
                      << " outer_corrections=" << pr.outer_corrections
                      << " total_inner_iterations=" << pr.total_inner_iterations
                      << " total_L0_operator_applies=" << pr.total_l0_operator_applies << '\n'
                      << std::scientific << std::setprecision(9)
                      << "final_true_relative_residual=" << pr.final_true_residual
                      << " geometric_outer_q=" << pr.geometric_outer_q << '\n'
                      << std::fixed << std::setprecision(6)
                      << "gpu_inner_solve_ms=" << pr.gpu_inner_solve_ms
                      << " accurate_FP64_residual_ms=" << pr.accurate_fp64_residual_ms
                      << " correction_wall_ms=" << pr.correction_wall_ms
                      << " staged_policy_wall_ms=" << pr.solve_wall_ms << '\n';
            for (std::size_t i = 0; i < pr.outer_history.size(); ++i) {
                std::cout << std::scientific << std::setprecision(9)
                          << "  outer_true_residual[" << i << "]=" << pr.outer_history[i];
                if (i > 0U) std::cout << " q=" << pr.outer_history[i] / pr.outer_history[i - 1U];
                std::cout << '\n';
            }
            for (std::size_t i = 0; i < pr.inner_true.size(); ++i) {
                std::cout << "  correction[" << i << "] recursive_rel=" << pr.inner_recursive[i]
                          << " true_rel=" << pr.inner_true[i] << '\n';
            }
        }

        std::cout << "\nSWEEP_VERDICT hierarchy_oracle_accept=" << (hierarchy_ok ? "true" : "false");
        if (best != nullptr) {
            std::cout << " preferred_inner_iterations=" << best->inner_iterations
                      << std::fixed << std::setprecision(6)
                      << " preferred_gpu_inner_solve_ms=" << best->gpu_inner_solve_ms
                      << " preferred_outer_corrections=" << best->outer_corrections
                      << std::scientific << std::setprecision(9)
                      << " preferred_final_true_residual=" << best->final_true_residual;
        } else {
            std::cout << " preferred_inner_iterations=none";
        }
        std::cout << '\n';
        return hierarchy_ok && best != nullptr ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_vcycle_defect_iteration_sweep_bench "
                  << "[min_inner=4 [max_inner=11 [max_outer=8 [block_y=4 [target_nodes=12 [min_nodes=4 [outer_tol=1e-6]]]]]]]\n";
        return 1;
    }
}
