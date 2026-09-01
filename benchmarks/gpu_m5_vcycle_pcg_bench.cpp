// M5 GPU productionization stage 8: first real solver benchmark. Reconstruct the
// frozen hierarchy, run a fixed number of fully device-resident FP32 PCG
// iterations with the validated 5x1x1 V-cycle as M^-1, then independently
// verify the returned solution with the FP64 fine operator.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_p1_block6_setup.hpp"
#include "m5_l2_dense_setup.hpp"
#include "gfss/gpu_m5_fine_level.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using SolverClock = std::chrono::steady_clock;

std::vector<float> solver_to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

std::vector<double> solver_to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

std::vector<float> bottom_lower_to_float(const DenseCholesky& factor) {
    return solver_to_float(factor.lower);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 11U;
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 20;
        const int block_y = argc > 3 ? std::stoi(argv[3]) : 4;
        const std::size_t target_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 12U;
        const std::size_t min_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 4U;
        if (iterations == 0U || iterations > 256U || repeats <= 0 || block_y <= 0 ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid M5 V-cycle PCG benchmark options");
        }

        constexpr std::size_t nu0 = 5U;
        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto setup_start = SolverClock::now();

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
        const auto setup_stop = SolverClock::now();

        const auto rhs = make_rhs(mesh);
        const double rhs_norm = norm(rhs);
        if (!(rhs_norm > 0.0)) throw std::runtime_error("M5 V-cycle PCG RHS is zero");

        gfss::GpuM5FineLevelContext gpu_context(
            mesh, material, space0, omega0, lambda0, block_y);
        const auto gpu = gpu_context.solve_pcg_vcycle_5x1x1_fixed(
            solver_to_float(rhs),
            iterations,
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
            m5_l2_setup::to_float(a2.fp64),
            inverse2,
            lambda2,
            m5_l2_setup::to_float(p2.fp64),
            transfer2_tentative.coarse_dofs,
            bottom_lower_to_float(bottom.factor),
            repeats);

        const auto x_gpu = solver_to_double(gpu.solution_aos);
        const auto ax_true = apply0(x_gpu);
        Vec true_residual(rhs.size(), 0.0);
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            true_residual[i] = rhs[i] - ax_true[i];
        }
        const double true_relative_residual = norm(true_residual) / rhs_norm;
        const double recurrence_vs_true_disagreement =
            std::abs(gpu.recursive_relative_residual - true_relative_residual) /
            std::max(true_relative_residual, 1.0e-300);
        const bool hierarchy_oracle_ok =
            block1_oracle_error <= 1.0e-10 &&
            block2_oracle_error <= 1.0e-10 &&
            bottom_oracle_error <= 1.0e-10;
        const bool true_tolerance_met = true_relative_residual <= 1.0e-6;

        std::cout << "GFSS M5 fully device-resident V-cycle PCG solve\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "rhs=physical_uniform_z_xmax\n"
                  << "krylov=left_preconditioned_conjugate_gradient_FP32\n"
                  << "preconditioner=complete_persistent_5x1x1_Vcycle\n"
                  << "stopping_policy=fixed_iterations_no_host_check_inside_timed_solve\n"
                  << "authoritative_convergence_oracle=final_CPU_FP64_true_residual\n"
                  << "host_round_trips_inside_timed_solve=false\n"
                  << "iterations=" << iterations << " repeats=" << repeats << '\n'
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L2_dofs=" << block2.dofs()
                  << " L3_dofs=" << bottom.factor.n << '\n'
                  << std::fixed << std::setprecision(6)
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
                  << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error << '\n'
                  << "gpu_recursive_relative_residual=" << gpu.recursive_relative_residual
                  << " final_true_FP64_relative_residual=" << true_relative_residual
                  << " recurrence_vs_true_relative_disagreement="
                  << recurrence_vs_true_disagreement << '\n'
                  << "hierarchy_oracle_accept=" << (hierarchy_oracle_ok ? "true" : "false")
                  << " pcg_breakdown=" << (gpu.breakdown ? "true" : "false")
                  << " true_tolerance_1e-6_met=" << (true_tolerance_met ? "true" : "false")
                  << '\n'
                  << std::fixed << std::setprecision(6)
                  << "gpu_median_solve_ms=" << gpu.median_solve_ms
                  << " gpu_best_solve_ms=" << gpu.best_solve_ms
                  << " median_ms_per_iteration="
                  << gpu.median_solve_ms / static_cast<double>(iterations) << '\n'
                  << "preconditioner_applications=" << gpu.preconditioner_applications
                  << " pcg_operator_applications=" << gpu.pcg_operator_applications
                  << " vcycle_L0_operator_applies=" << gpu.vcycle_l0_operator_applies
                  << " total_L0_operator_applies=" << gpu.total_l0_operator_applies << '\n'
                  << "persistent_device_bytes=" << gpu.device_bytes_total << '\n';

        return hierarchy_oracle_ok && !gpu.breakdown ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_vcycle_pcg_bench "
                  << "[iterations=11 [repeats=20 [block_y=4 [target_nodes=12 [min_nodes=4]]]]]\n";
        return 1;
    }
}
