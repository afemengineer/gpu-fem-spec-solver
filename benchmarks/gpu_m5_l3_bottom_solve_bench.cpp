// M5 GPU productionization stage 6: validate the final 132-DOF bottom solve.
// The frozen hierarchy is reconstructed exactly as in the L2 shell, then the
// localized FP64 bottom Galerkin operator is factored on the CPU oracle path.
// Its lower Cholesky factor is cast once to FP32 and solved persistently on the
// GPU by a single warp. Timed execution contains no host round trips.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "gfss/gpu_m5_bottom_solve.hpp"

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

using BottomClock = std::chrono::steady_clock;

std::vector<double> bottom_probe(std::size_t n, double scale, double phase) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = scale * (std::sin(0.019 * t + phase) +
                        0.33 * std::cos(0.047 * t - 0.61 * phase));
    }
    return v;
}

std::vector<float> bottom_to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

std::vector<double> bottom_to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

double bottom_relative_error(const std::vector<double>& got,
                             const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("M5 bottom oracle vector size mismatch");
    }
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = got[i] - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    if (!(r2 > 0.0)) throw std::runtime_error("M5 bottom oracle norm zero");
    return std::sqrt(d2 / r2);
}

double fp32_cholesky_reconstruction_error(
    const LocalBottomReference& bottom,
    const std::vector<float>& lower) {
    const std::size_t n = bottom.factor.n;
    if (lower.size() != n * n) throw std::invalid_argument("M5 bottom factor size");
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double reconstructed = 0.0;
            const std::size_t stop = std::min(i, j);
            for (std::size_t k = 0; k <= stop; ++k) {
                reconstructed += static_cast<double>(lower[i * n + k]) *
                                 static_cast<double>(lower[j * n + k]);
            }
            const double ref = bottom.values[i * n + j];
            const double d = reconstructed - ref;
            d2 += d * d;
            r2 += ref * ref;
        }
    }
    return std::sqrt(d2 / std::max(r2, 1.0e-300));
}

double true_bottom_residual(
    const LocalBottomReference& bottom,
    const std::vector<double>& x,
    const std::vector<double>& rhs) {
    const auto ax = apply_dense_bottom(bottom, x);
    if (ax.size() != rhs.size()) throw std::invalid_argument("M5 bottom residual size");
    double d2 = 0.0;
    double b2 = 0.0;
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        const double d = rhs[i] - ax[i];
        d2 += d * d;
        b2 += rhs[i] * rhs[i];
    }
    return std::sqrt(d2 / std::max(b2, 1.0e-300));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 200;
        const int block_y = argc > 2 ? std::stoi(argv[2]) : 4;
        const std::size_t target_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t min_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 4U;
        if (repeats <= 0 || block_y <= 0 || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid M5 L3 bottom options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto setup_start = BottomClock::now();

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

        const auto local_a1_apply = [&](const LocalColumns& x) {
            return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
        };
        const auto l2_basis = build_smoothed_candidate_supports(
            transfer1_tentative, strength1.graph, block1, omega1, m1, local_a1_apply);
        const auto block2 = build_metric_from_local_supports(
            transfer1_tentative, block1, l2_basis, local_a1_apply);
        const double block2_oracle_error = audit_l1_block_metric(block2, apply2);
        const double lambda2 = estimate_lambda_max_l1_block(apply2, block2, 8U);
        const double omega2 = kSaDampingNumerator / lambda2;

        const auto transfer2_tentative = build_candidate_transfer(
            transfer1_tentative.coarse_graph,
            transfer1_tentative.coarse_candidates,
            target_nodes,
            min_nodes,
            1.0e-10);
        const L1BlockSmoothedTransfer transfer2{
            transfer2_tentative, apply2, block2, omega2, m2};
        const auto local_a2_apply_lambda = [&](const LocalColumns& x) {
            return apply_local_a2_columns(x, l2_basis, block1, local_a1_apply);
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
        const auto bottom = build_local_bottom(
            transfer2_tentative, block2, bottom_basis, local_a2_apply);
        const Apply apply3_nested = [&](const Vec& x) {
            return transfer2.restrict_transpose(apply2(transfer2.prolong(x)));
        };
        const double bottom_oracle_error = bottom_local_oracle_error(bottom, apply3_nested);
        const auto setup_stop = BottomClock::now();

        const auto x_seed = bottom_probe(bottom.factor.n, 1.0e-9, 0.37);
        const auto rhs = apply_dense_bottom(bottom, x_seed);
        const auto cpu_x = bottom.factor.solve(rhs);

        std::vector<float> lower_fp32(bottom.factor.lower.size(), 0.0f);
        for (std::size_t i = 0; i < lower_fp32.size(); ++i) {
            lower_fp32[i] = static_cast<float>(bottom.factor.lower[i]);
        }
        const double reconstruction_error =
            fp32_cholesky_reconstruction_error(bottom, lower_fp32);
        const auto gpu = gfss::benchmark_m5_bottom_cholesky_solve(
            lower_fp32, bottom.factor.n, bottom_to_float(rhs), repeats);
        const auto gpu_x = bottom_to_double(gpu.x);
        const double solution_error = bottom_relative_error(gpu_x, cpu_x);
        const double residual_error = true_bottom_residual(bottom, gpu_x, rhs);

        const bool oracle_ok =
            block1_oracle_error <= 1.0e-10 &&
            block2_oracle_error <= 1.0e-10 &&
            bottom_oracle_error <= 1.0e-10 &&
            reconstruction_error <= 1.0e-4 &&
            solution_error <= 1.0e-4 &&
            residual_error <= 1.0e-4;

        std::cout << "GFSS M5 persistent CUDA exact L3 bottom solve\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "hierarchy=frozen_theta_0p05_actual_L1_L2_block_metrics\n"
                  << "bottom_operator=localized_exact_recursive_energy\n"
                  << "bottom_factorization=CPU_FP64_Cholesky_cast_once_to_FP32\n"
                  << "bottom_runtime=single_warp_forward_plus_backward_substitution\n"
                  << "host_round_trips_inside_timed_solve=false\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L2_dofs=" << block2.dofs()
                  << " L3_dofs=" << bottom.factor.n
                  << " L3_nodes=" << transfer2_tentative.aggregates.size() << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0 << " omega0=" << omega0
                  << " lambda1=" << lambda1 << " omega1=" << omega1
                  << " lambda2=" << lambda2 << " omega2=" << omega2 << '\n'
                  << "P0_support_setup_ms=" << p0_support_ms
                  << " bottom_local_assembly_ms=" << bottom.assembly_ms
                  << " total_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error
                  << " L2_block_vs_nested_relative_error=" << block2_oracle_error << '\n'
                  << "bottom_local_vs_nested_relative_error=" << bottom_oracle_error
                  << " bottom_symmetry_relative_defect="
                  << bottom.factor.symmetry_relative_defect << '\n'
                  << "fp32_cholesky_reconstruction_relative_error=" << reconstruction_error
                  << " gpu_vs_cpu_fp64_solution_relative_error=" << solution_error
                  << " gpu_true_bottom_residual_relative_error=" << residual_error
                  << " oracle_accept_1e-4=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "bottom_min_cholesky_pivot=" << bottom.factor.min_pivot
                  << " bottom_factor_bytes_fp32=" << gpu.factor_bytes
                  << " bottom_device_bytes=" << gpu.device_bytes << '\n'
                  << "median_bottom_solve_ms=" << gpu.median_ms
                  << " best_bottom_solve_ms=" << gpu.best_ms << '\n';

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_l3_bottom_solve_bench "
                  << "[repeats=200 [block_y=4 [target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
