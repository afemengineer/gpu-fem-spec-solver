// M5 GPU productionization stage 5: validate a complete L1 V-cycle shell using
// the frozen actual-block degree-1 smoother and explicit dual-order block6 P1.
// L2 remains an externally supplied deterministic correction in this stage so
// the L1 algebra can be audited independently against the FP64 recursive-SA
// reference before the dense A2 / bottom hierarchy is connected.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_p1_block6_setup.hpp"
#include "gfss/gpu_smoothed_aggregation.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using BenchClock = std::chrono::steady_clock;

std::vector<double> deterministic_probe(std::size_t n, double scale, double phase) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = scale * (std::sin(0.013 * t + phase) +
                        0.33 * std::cos(0.029 * t - 0.47 * phase));
    }
    return v;
}

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

double relative_error(const std::vector<double>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("M5 L1 shell oracle size mismatch");
    }
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = got[i] - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    if (!(r2 > 0.0)) throw std::runtime_error("M5 L1 shell oracle norm zero");
    return std::sqrt(d2 / r2);
}

std::vector<float> inverse_blocks_fp32(const L1BlockMetric& metric) {
    std::vector<float> inverse(metric.nodes() * 36U, 0.0f);
    for (std::size_t a = 0; a < metric.nodes(); ++a) {
        const std::size_t begin = metric.dof_offsets[a];
        const std::size_t rank = metric.dof_offsets[a + 1U] - begin;
        const double* l = metric.lower.data() + metric.value_offsets[a];
        float* out = inverse.data() + a * 36U;
        for (std::size_t col = 0; col < rank; ++col) {
            std::array<double, 6U> y{};
            std::array<double, 6U> x{};
            for (std::size_t i = 0; i < rank; ++i) {
                double value = i == col ? 1.0 : 0.0;
                for (std::size_t j = 0; j < i; ++j) {
                    value -= l[i * rank + j] * y[j];
                }
                y[i] = value / l[i * rank + i];
            }
            for (std::size_t ii = rank; ii-- > 0U;) {
                double value = y[ii];
                for (std::size_t j = ii + 1U; j < rank; ++j) {
                    value -= l[j * rank + ii] * x[j];
                }
                x[ii] = value / l[ii * rank + ii];
            }
            for (std::size_t row = 0; row < rank; ++row) {
                out[row * 6U + col] = static_cast<float>(x[row]);
            }
        }
    }
    return inverse;
}

std::vector<double> axpy(const std::vector<double>& x,
                         const std::vector<double>& y,
                         double alpha) {
    if (x.size() != y.size()) throw std::invalid_argument("M5 L1 shell axpy size mismatch");
    std::vector<double> out(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] + alpha * y[i];
    return out;
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
            throw std::invalid_argument("invalid M5 L1 full-shell options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t nu1 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto setup_start = BenchClock::now();

        auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
        const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
            mesh, material, space0);
        const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
        const Apply apply0 = [&](const Vec& x) { return apply_fine_clamped(mesh, material, x); };
        const double lambda0 = estimate_lambda_max(apply0, fine_inverse, 8U);
        const double omega0 = kSaDampingNumerator / lambda0;
        const FineSmoothedTransfer transfer0{
            mesh, material, space0, fine_inverse, omega0, m0};
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

        const auto local_a1_apply = [&](const LocalColumns& x) {
            return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
        };
        const auto p1_basis_start = BenchClock::now();
        const auto l2_basis = build_smoothed_candidate_supports(
            transfer1_tentative, strength1.graph, block1, omega1, m1, local_a1_apply);
        const double p1_basis_setup_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - p1_basis_start).count();
        const auto p1 = m5_p1_setup::assemble_dual_order_block6(
            transfer1_tentative, block1, l2_basis);
        const auto inverse_blocks = inverse_blocks_fp32(block1);
        const auto setup_stop = BenchClock::now();

        // Build a deterministic but operator-scaled L1 RHS so the shell remains
        // well-conditioned in both FP64 and FP32 without depending on a physical
        // load projection. A1*x_seed provides the stiffness scale directly.
        const auto x_seed = deterministic_probe(space0.coarse_dofs, 1.0e-9, 0.19);
        const auto rhs = apply1(x_seed);
        const auto external_l2 = deterministic_probe(
            transfer1_tentative.coarse_dofs, 1.5e-10, 0.73);
        const auto external_l2_padded = m5_p1_setup::pad_l2(
            transfer1_tentative, external_l2);

        // Independent FP64 oracle for the exact same mathematical L1 shell.
        const double weight1 = 1.0 / (0.55 * lambda1);
        auto x_cpu = block1.solve(rhs);
        for (double& value : x_cpu) value *= weight1;

        const auto ax_pre = apply1(x_cpu);
        Vec residual1(rhs.size(), 0.0);
        for (std::size_t i = 0; i < rhs.size(); ++i) residual1[i] = rhs[i] - ax_pre[i];
        const auto residual2_cpu = transfer1.restrict_transpose(residual1);

        const auto correction1_cpu = transfer1.prolong(external_l2);
        x_cpu = axpy(x_cpu, correction1_cpu, 1.0);
        const auto ax_post = apply1(x_cpu);
        Vec post_residual(rhs.size(), 0.0);
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            post_residual[i] = rhs[i] - ax_post[i];
        }
        const auto post_delta = block1.solve(post_residual);
        x_cpu = axpy(x_cpu, post_delta, weight1);

        gfss::GpuSmoothedAggregationContext context(
            mesh, material, space0, omega0, block_y);
        const auto gpu = context.l1_full_shell(
            to_float(rhs),
            inverse_blocks,
            lambda1,
            m0,
            transfer1_tentative.aggregates.size(),
            p1.forward_row_offsets,
            p1.forward_column_indices,
            p1.forward_values_row_major,
            p1.transpose_column_offsets,
            p1.transpose_row_indices,
            p1.transpose_values_q_r_entry,
            to_float(external_l2_padded),
            repeats);

        const auto residual2_gpu = m5_p1_setup::unpad_l2(
            transfer1_tentative, to_double(gpu.l2_residual_padded));
        const auto x_gpu = to_double(gpu.final_x);
        const double residual2_error = relative_error(residual2_gpu, residual2_cpu);
        const double final_x_error = relative_error(x_gpu, x_cpu);
        const bool oracle_ok = block1_oracle_error <= 1.0e-10 &&
                               residual2_error <= 1.0e-4 &&
                               final_x_error <= 1.0e-4;

        const std::size_t a0_per_a1 = 2U * m0 + 1U;
        const std::size_t executed_a0_equiv = gpu.executed_a1_applies * a0_per_a1;
        const std::size_t mathematical_a0_equiv = gpu.mathematical_a1_applies * a0_per_a1;
        const std::size_t legacy_factorized_l1_a0_equiv =
            (2U * nu1 + 1U + 2U * m1) * a0_per_a1;

        std::cout << "GFSS M5 persistent CUDA complete L1 V-cycle shell\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "shell=degree1_actual_block_pre_then_P1T_then_external_L2_correction_then_P1_then_degree1_post\n"
                  << "P1_representation=dual_order_block6_forward_row_major_plus_transpose_q_r_entry\n"
                  << "recursive_schedule_target=1x1\n"
                  << "production_policy_target=5x1x1\n"
                  << "zero_start_pre_A1_elided=true\n"
                  << "host_round_trips_inside_timed_shell=false\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L1_nodes=" << block1.nodes()
                  << " L2_dofs=" << transfer1_tentative.coarse_dofs
                  << " L2_nodes=" << transfer1_tentative.aggregates.size() << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0 << " omega0=" << omega0
                  << " lambda1=" << lambda1 << " omega1=" << omega1 << '\n'
                  << "P0_support_setup_ms=" << p0_support_ms
                  << " P1_smoothed_support_setup_ms=" << p1_basis_setup_ms
                  << " P1_dual_block6_assembly_ms=" << p1.assembly_ms
                  << " total_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error << '\n'
                  << "gpu_vs_cpu_fp64_L2_residual_relative_error=" << residual2_error
                  << " gpu_vs_cpu_fp64_final_L1_correction_relative_error=" << final_x_error
                  << " oracle_accept_1e-4=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "P1_block_nnz=" << p1.block_nnz()
                  << " P1_matrix_bytes_fp32=" << p1.matrix_bytes_fp32()
                  << " P1_bytes_per_fine_dof="
                  << static_cast<double>(p1.matrix_bytes_fp32()) /
                     static_cast<double>(mesh.dof_count()) << '\n'
                  << "mathematical_A1_applies=" << gpu.mathematical_a1_applies
                  << " executed_A1_applies=" << gpu.executed_a1_applies
                  << " mathematical_A0_equiv=" << mathematical_a0_equiv
                  << " executed_A0_equiv=" << executed_a0_equiv
                  << " legacy_fully_factorized_L1_A0_equiv=" << legacy_factorized_l1_a0_equiv
                  << '\n'
                  << "median_total_ms=" << gpu.median_total_ms
                  << " best_total_ms=" << gpu.best_total_ms
                  << " median_pre_smooth_ms=" << gpu.median_pre_smooth_ms
                  << " median_residual_ms=" << gpu.median_residual_ms
                  << " median_pack_ms=" << gpu.median_pack_ms
                  << " median_P1T_ms=" << gpu.median_p1t_ms
                  << " median_P1_ms=" << gpu.median_p1_ms
                  << " median_correction_ms=" << gpu.median_correction_ms
                  << " median_post_smooth_ms=" << gpu.median_post_smooth_ms << '\n'
                  << "persistent_L1_shell_device_bytes=" << gpu.persistent_l1_shell_bytes
                  << " zero_start_pre_A1_elided="
                  << (gpu.zero_start_pre_a1_elided ? "true" : "false") << '\n';

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_l1_full_shell_bench "
                  << "[repeats=50 [block_y=4 [target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
