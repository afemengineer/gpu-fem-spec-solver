// M5 GPU productionization stage 7: validate the first complete persistent
// 5x1x1 V-cycle against the frozen FP64 recursive hierarchy on the physical
// thin-plate RHS. All hierarchy representations have already passed isolated
// oracles; this benchmark verifies their composition and measures one real
// preconditioner application end to end.
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

using VClock = std::chrono::steady_clock;

std::vector<float> vcycle_to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

std::vector<double> vcycle_to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

double vcycle_relative_error(const std::vector<double>& got,
                             const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("M5 complete V-cycle oracle size mismatch");
    }
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = got[i] - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    if (!(r2 > 0.0)) throw std::runtime_error("M5 complete V-cycle oracle norm zero");
    return std::sqrt(d2 / r2);
}

std::vector<double> vcycle_residual(const Apply& apply,
                                    const std::vector<double>& rhs,
                                    const std::vector<double>& x) {
    const auto ax = apply(x);
    if (ax.size() != rhs.size()) throw std::invalid_argument("M5 V-cycle residual size");
    std::vector<double> r(rhs.size(), 0.0);
    for (std::size_t i = 0; i < rhs.size(); ++i) r[i] = rhs[i] - ax[i];
    return r;
}

std::vector<float> bottom_lower_to_float(const DenseCholesky& factor) {
    return vcycle_to_float(factor.lower);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 50;
        const int block_y = argc > 2 ? std::stoi(argv[2]) : 4;
        const std::size_t target_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t min_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 4U;
        if (repeats <= 0 || block_y <= 0 || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid M5 complete V-cycle options");
        }

        constexpr std::size_t nu0 = 5U;
        constexpr std::size_t nu1 = 1U;
        constexpr std::size_t nu2 = 1U;
        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto setup_start = VClock::now();

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
        const double a2_oracle_error = vcycle_relative_error(
            m5_l2_setup::apply_dense_a2(a2, deterministic_actual_a2_probe(a2.n, 0.43)),
            apply2(deterministic_actual_a2_probe(a2.n, 0.43)));

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
        const auto setup_stop = VClock::now();

        // One exact FP64 5x1x1 V-cycle on the physical thin-plate load.
        const auto rhs = make_rhs(mesh);
        const double rhs_norm = norm(rhs);
        const auto cpu_start = VClock::now();
        Vec x0(rhs.size(), 0.0);
        chebyshev_smooth(apply0, fine_inverse, lambda0, rhs, x0, nu0);
        clamp_x0(mesh, x0);
        const auto r0 = vcycle_residual(apply0, rhs, x0);
        const auto b1 = transfer0.restrict_transpose(r0);

        Vec x1(b1.size(), 0.0);
        chebyshev_l1_block_smooth(apply1, block1, lambda1, b1, x1, nu1);
        const auto r1 = vcycle_residual(apply1, b1, x1);
        const auto b2 = transfer1.restrict_transpose(r1);

        Vec x2(b2.size(), 0.0);
        chebyshev_l1_block_smooth(apply2, block2, lambda2, b2, x2, nu2);
        const auto r2 = vcycle_residual(apply2, b2, x2);
        const auto b3 = transfer2.restrict_transpose(r2);
        const auto x3 = bottom.factor.solve(b3);

        x2 = axpy(x2, transfer2.prolong(x3), 1.0);
        chebyshev_l1_block_smooth(apply2, block2, lambda2, b2, x2, nu2);
        x1 = axpy(x1, transfer1.prolong(x2), 1.0);
        chebyshev_l1_block_smooth(apply1, block1, lambda1, b1, x1, nu1);
        x0 = axpy(x0, transfer0.prolong(x1), 1.0);
        chebyshev_smooth(apply0, fine_inverse, lambda0, rhs, x0, nu0);
        clamp_x0(mesh, x0);
        const double cpu_vcycle_ms = std::chrono::duration<double, std::milli>(
            VClock::now() - cpu_start).count();
        const double cpu_true_residual_ratio = norm(vcycle_residual(apply0, rhs, x0)) / rhs_norm;

        gfss::GpuM5FineLevelContext gpu_context(
            mesh, material, space0, omega0, lambda0, block_y);
        const auto gpu = gpu_context.complete_vcycle_5x1x1(
            vcycle_to_float(rhs),
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

        const auto gpu_x = vcycle_to_double(gpu.fine_correction_aos);
        const auto gpu_b3 = vcycle_to_double(gpu.l3_rhs);
        const double final_correction_error = vcycle_relative_error(gpu_x, x0);
        const double l3_rhs_error = vcycle_relative_error(gpu_b3, b3);
        const double gpu_true_residual_ratio = norm(vcycle_residual(apply0, rhs, gpu_x)) / rhs_norm;
        const double contraction_ratio_disagreement =
            std::abs(gpu_true_residual_ratio - cpu_true_residual_ratio) /
            std::max(cpu_true_residual_ratio, 1.0e-300);

        const bool oracle_ok =
            block1_oracle_error <= 1.0e-10 &&
            block2_oracle_error <= 1.0e-10 &&
            a2_oracle_error <= 1.0e-10 &&
            bottom_oracle_error <= 1.0e-10 &&
            l3_rhs_error <= 1.0e-4 &&
            final_correction_error <= 1.0e-4 &&
            contraction_ratio_disagreement <= 1.0e-3;

        const double speedup_vs_cpu_reference = gpu.median_timing.total_ms > 0.0
            ? cpu_vcycle_ms / gpu.median_timing.total_ms : 0.0;

        std::cout << "GFSS M5 complete persistent CUDA 5x1x1 V-cycle\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "rhs=physical_uniform_z_xmax\n"
                  << "hierarchy=L0_matrix_free_L1_factorized_P1_dual_block6_A2_dense_P2_dense_L3_direct\n"
                  << "schedule=nu0xnu1xnu2=" << nu0 << 'x' << nu1 << 'x' << nu2 << '\n'
                  << "recursion=1x1\n"
                  << "host_round_trips_inside_timed_vcycle=false\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L1_nodes=" << block1.nodes()
                  << " L2_dofs=" << block2.dofs()
                  << " L2_nodes=" << block2.nodes()
                  << " L3_dofs=" << bottom.factor.n
                  << " L3_nodes=" << transfer2_tentative.aggregates.size() << '\n'
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
                  << " L2_block_vs_nested_relative_error=" << block2_oracle_error << '\n'
                  << "A2_dense_vs_nested_relative_error=" << a2_oracle_error
                  << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error << '\n'
                  << "gpu_vs_cpu_fp64_L3_rhs_relative_error=" << l3_rhs_error
                  << " gpu_vs_cpu_fp64_final_vcycle_relative_error=" << final_correction_error << '\n'
                  << "cpu_true_residual_ratio_after_one_vcycle=" << cpu_true_residual_ratio
                  << " gpu_true_residual_ratio_after_one_vcycle=" << gpu_true_residual_ratio
                  << " contraction_ratio_relative_disagreement=" << contraction_ratio_disagreement
                  << " oracle_accept_1e-4=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "cpu_fp64_reference_vcycle_ms=" << cpu_vcycle_ms
                  << " gpu_median_vcycle_ms=" << gpu.median_timing.total_ms
                  << " gpu_best_vcycle_ms=" << gpu.best_timing.total_ms
                  << " speedup_vs_cpu_fp64_reference=" << speedup_vs_cpu_reference << '\n'
                  << "median_L0_down_ms=" << gpu.median_timing.l0_down_ms
                  << " median_L1_down_ms=" << gpu.median_timing.l1_down_ms
                  << " median_L2_down_ms=" << gpu.median_timing.l2_down_ms
                  << " median_L3_solve_ms=" << gpu.median_timing.l3_solve_ms
                  << " median_L2_up_ms=" << gpu.median_timing.l2_up_ms
                  << " median_L1_up_ms=" << gpu.median_timing.l1_up_ms
                  << " median_L0_up_ms=" << gpu.median_timing.l0_up_ms << '\n'
                  << "L0_operator_applies=" << gpu.l0_operator_applies
                  << " L1_operator_applies=" << gpu.l1_operator_applies
                  << " L2_operator_applies=" << gpu.l2_operator_applies
                  << " persistent_device_bytes=" << gpu.device_bytes_total << '\n';

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_complete_vcycle_bench "
                  << "[repeats=50 [block_y=4 [target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
