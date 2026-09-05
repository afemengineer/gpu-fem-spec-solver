// M5 stage 12: remove the serial 132-DOF triangular-solve bottleneck.
// Compare the validated FP32 Cholesky solve against an explicitly symmetric
// FP32 A3^-1 applied by cuBLAS SGEMV and by one custom coalesced CTA.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "gfss/gpu_m5_bottom_solve.hpp"
#include "m5_bottom_inverse_staging.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using InverseClock = std::chrono::steady_clock;

std::vector<double> inverse_probe(std::size_t n) {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0U; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        x[i] = 1.0e-9 * (std::sin(0.019 * t + 0.37) +
                         0.33 * std::cos(0.047 * t - 0.2257));
    }
    return x;
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

double relative_error(const std::vector<double>& got,
                      const std::vector<double>& ref) {
    if (got.size() != ref.size()) throw std::invalid_argument("M5 inverse oracle size mismatch");
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0U; i < got.size(); ++i) {
        const double d = got[i] - ref[i];
        d2 += d * d;
        r2 += ref[i] * ref[i];
    }
    return std::sqrt(d2 / std::max(r2, 1.0e-300));
}

double true_residual(const LocalBottomReference& bottom,
                     const std::vector<double>& x,
                     const std::vector<double>& rhs) {
    const auto ax = apply_dense_bottom(bottom, x);
    double d2 = 0.0;
    double b2 = 0.0;
    for (std::size_t i = 0U; i < rhs.size(); ++i) {
        const double d = rhs[i] - ax[i];
        d2 += d * d;
        b2 += rhs[i] * rhs[i];
    }
    return std::sqrt(d2 / std::max(b2, 1.0e-300));
}

std::vector<float> symmetric_inverse_col_major(const DenseCholesky& factor) {
    const std::size_t n = factor.n;
    std::vector<double> inverse(n * n, 0.0);
    for (std::size_t j = 0U; j < n; ++j) {
        Vec e(n, 0.0);
        e[j] = 1.0;
        const auto column = factor.solve(e);
        for (std::size_t i = 0U; i < n; ++i) inverse[i * n + j] = column[i];
    }
    std::vector<float> out(n * n, 0.0f);
    for (std::size_t i = 0U; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            const float v = static_cast<float>(
                0.5 * (inverse[i * n + j] + inverse[j * n + i]));
            out[j * n + i] = v;
            out[i * n + j] = v;
        }
    }
    return out;
}

double inverse_identity_relative_error(const LocalBottomReference& bottom,
                                       const std::vector<float>& inverse_col_major) {
    const std::size_t n = bottom.factor.n;
    double d2 = 0.0;
    const double i2 = static_cast<double>(n);
    for (std::size_t row = 0U; row < n; ++row) {
        for (std::size_t col = 0U; col < n; ++col) {
            double value = 0.0;
            for (std::size_t k = 0U; k < n; ++k) {
                value += bottom.values[row * n + k] *
                         static_cast<double>(inverse_col_major[col * n + k]);
            }
            const double target = row == col ? 1.0 : 0.0;
            const double d = value - target;
            d2 += d * d;
        }
    }
    return std::sqrt(d2 / i2);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 500;
        const int block_y = argc > 2 ? std::stoi(argv[2]) : 4;
        const std::size_t target_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t min_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 4U;
        if (repeats <= 0 || block_y <= 0 || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid M5 bottom inverse options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto setup_start = InverseClock::now();

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
        const double block1_error = audit_l1_block_metric(block1, apply1);
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
        const double block2_error = audit_l1_block_metric(block2, apply2);
        const double lambda2 = estimate_lambda_max_l1_block(apply2, block2, 8U);
        const double omega2 = kSaDampingNumerator / lambda2;

        const auto transfer2_tentative = build_candidate_transfer(
            transfer1_tentative.coarse_graph,
            transfer1_tentative.coarse_candidates,
            target_nodes, min_nodes, 1.0e-10);
        const L1BlockSmoothedTransfer transfer2{
            transfer2_tentative, apply2, block2, omega2, m2};
        const auto local_a2_apply_lambda = [&](const LocalColumns& x) {
            return apply_local_a2_columns(x, l2_basis, block1, local_a1_apply);
        };
        const std::function<LocalColumns(const LocalColumns&)> local_a2_apply =
            local_a2_apply_lambda;
        const auto bottom_basis = build_smoothed_candidate_supports(
            transfer2_tentative, transfer1_tentative.coarse_graph,
            block2, omega2, m2, local_a2_apply_lambda);
        const auto bottom = build_local_bottom(
            transfer2_tentative, block2, bottom_basis, local_a2_apply);
        const Apply apply3_nested = [&](const Vec& x) {
            return transfer2.restrict_transpose(apply2(transfer2.prolong(x)));
        };
        const double bottom_error = bottom_local_oracle_error(bottom, apply3_nested);
        const auto setup_stop = InverseClock::now();

        const auto x_seed = inverse_probe(bottom.factor.n);
        const auto rhs = apply_dense_bottom(bottom, x_seed);
        const auto cpu_x = bottom.factor.solve(rhs);
        const auto rhs_fp32 = to_float(rhs);
        const auto lower_fp32 = to_float(bottom.factor.lower);
        const auto inverse_fp32 = symmetric_inverse_col_major(bottom.factor);
        const double inverse_identity_error =
            inverse_identity_relative_error(bottom, inverse_fp32);

        const auto triangular = gfss::benchmark_m5_bottom_cholesky_solve(
            lower_fp32, bottom.factor.n, rhs_fp32, repeats);
        const auto inverse = gfss::benchmark_m5_bottom_inverse_apply(
            inverse_fp32, bottom.factor.n, rhs_fp32, repeats);

        const auto triangular_x = to_double(triangular.x);
        const auto cublas_x = to_double(inverse.cublas_x);
        const auto custom_x = to_double(inverse.custom_x);
        const double triangular_solution_error = relative_error(triangular_x, cpu_x);
        const double cublas_solution_error = relative_error(cublas_x, cpu_x);
        const double custom_solution_error = relative_error(custom_x, cpu_x);
        const double triangular_residual = true_residual(bottom, triangular_x, rhs);
        const double cublas_residual = true_residual(bottom, cublas_x, rhs);
        const double custom_residual = true_residual(bottom, custom_x, rhs);

        const bool hierarchy_ok = block1_error <= 1.0e-10 &&
                                  block2_error <= 1.0e-10 &&
                                  bottom_error <= 1.0e-10;
        const bool inverse_ok = inverse_identity_error <= 1.0e-4 &&
                                cublas_solution_error <= 1.0e-4 &&
                                custom_solution_error <= 1.0e-4 &&
                                cublas_residual <= 1.0e-4 &&
                                custom_residual <= 1.0e-4;

        const char* preferred = "triangular_cholesky";
        double preferred_ms = triangular.median_ms;
        if (inverse_ok && inverse.cublas_median_ms < preferred_ms) {
            preferred = "symmetric_inverse_cublas_sgemv";
            preferred_ms = inverse.cublas_median_ms;
        }
        if (inverse_ok && inverse.custom_median_ms < preferred_ms) {
            preferred = "symmetric_inverse_custom_coalesced_CTA";
            preferred_ms = inverse.custom_median_ms;
        }

        std::cout << "GFSS M5 L3 direct representation decision\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "L3_dofs=" << bottom.factor.n << '\n'
                  << "baseline=FP32_Cholesky_single_warp_triangular\n"
                  << "candidate_A=explicit_symmetric_FP32_inverse_cuBLAS_SGEMV\n"
                  << "candidate_B=explicit_symmetric_FP32_inverse_custom_coalesced_CTA\n"
                  << "host_round_trips_inside_timing=false\n"
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_error
                  << " L2_block_vs_nested_relative_error=" << block2_error
                  << " bottom_local_vs_nested_relative_error=" << bottom_error << '\n'
                  << "inverse_identity_relative_error=" << inverse_identity_error << '\n'
                  << "triangular_solution_relative_error=" << triangular_solution_error
                  << " triangular_true_residual=" << triangular_residual << '\n'
                  << "cublas_inverse_solution_relative_error=" << cublas_solution_error
                  << " cublas_inverse_true_residual=" << cublas_residual << '\n'
                  << "custom_inverse_solution_relative_error=" << custom_solution_error
                  << " custom_inverse_true_residual=" << custom_residual << '\n'
                  << "hierarchy_oracle_accept=" << (hierarchy_ok ? "true" : "false")
                  << " inverse_oracle_accept=" << (inverse_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "triangular_median_ms=" << triangular.median_ms
                  << " triangular_best_ms=" << triangular.best_ms << '\n'
                  << "cublas_inverse_median_ms=" << inverse.cublas_median_ms
                  << " cublas_inverse_best_ms=" << inverse.cublas_best_ms << '\n'
                  << "custom_inverse_median_ms=" << inverse.custom_median_ms
                  << " custom_inverse_best_ms=" << inverse.custom_best_ms << '\n'
                  << "inverse_speedup_over_triangular_cublas="
                  << triangular.median_ms / inverse.cublas_median_ms
                  << " inverse_speedup_over_triangular_custom="
                  << triangular.median_ms / inverse.custom_median_ms << '\n'
                  << "triangular_factor_bytes=" << triangular.factor_bytes
                  << " inverse_bytes=" << inverse.inverse_bytes
                  << " inverse_device_bytes=" << inverse.device_bytes << '\n'
                  << "total_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << '\n'
                  << "preferred_L3_representation=" << preferred
                  << " preferred_median_ms=" << preferred_ms << '\n';
        return hierarchy_ok && inverse_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_l3_inverse_apply_bench "
                  << "[repeats=500 [block_y=4 [target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
