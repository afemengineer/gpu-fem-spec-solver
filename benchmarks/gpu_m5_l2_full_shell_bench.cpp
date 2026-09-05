// M5 GPU productionization stage 5: validate the complete L2 symmetric V-cycle
// shell around an externally supplied L3 correction. The final frozen hierarchy
// uses the actual theta=0.05 L1 strength graph, actual L1/L2 block metrics,
// explicit dense FP32 A2, and an exact smoothed P2 materialized as a tiny dense
// rectangular matrix. The independent oracle remains the original FP64 nested
// factorized hierarchy.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "gfss/gpu_m5_l2_full_shell.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using BenchClock = std::chrono::steady_clock;

std::vector<double> m5_l2_shell_probe(std::size_t n, double scale, double phase) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = scale * (std::sin(0.013 * t + phase) +
                        0.29 * std::cos(0.037 * t - 0.41 * phase));
    }
    return v;
}

std::vector<float> m5_l2_shell_to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

std::vector<double> m5_l2_shell_to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

double m5_l2_shell_relative_error(const std::vector<double>& got,
                                   const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("M5 L2 shell oracle vector size mismatch");
    }
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = got[i] - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    if (!(r2 > 0.0)) throw std::runtime_error("M5 L2 shell oracle norm zero");
    return std::sqrt(d2 / r2);
}

std::vector<double> m5_l2_shell_axpy(const std::vector<double>& x,
                                      const std::vector<double>& y,
                                      double alpha) {
    if (x.size() != y.size()) throw std::invalid_argument("M5 L2 shell axpy size");
    std::vector<double> out(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = x[i] + alpha * y[i];
    return out;
}

struct M5DenseA2 {
    std::size_t n{0U};
    std::vector<double> fp64;
    double symmetry_relative_defect{0.0};
    double assembly_ms{0.0};
};

M5DenseA2 m5_assemble_dense_a2(
    const CandidateTransfer& transfer1,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis,
    const std::function<LocalColumns(const LocalColumns&)>& local_a1_apply) {
    const auto start = BenchClock::now();
    M5DenseA2 out;
    out.n = transfer1.coarse_dofs;
    if (out.n == 0U || l2_basis.size() != transfer1.aggregates.size()) {
        throw std::invalid_argument("M5 L2 shell A2 support/layout mismatch");
    }
    out.fp64.assign(out.n * out.n, 0.0);
    for (std::size_t jnode = 0; jnode < l2_basis.size(); ++jnode) {
        const auto applied = local_a1_apply(l2_basis[jnode]);
        const std::size_t joff = transfer1.aggregates[jnode].coarse_offset;
        const std::size_t jrank = transfer1.aggregates[jnode].rank;
        for (std::size_t inode = 0; inode < l2_basis.size(); ++inode) {
            const auto block = local_cross_gram(l2_basis[inode], applied, block1);
            const std::size_t ioff = transfer1.aggregates[inode].coarse_offset;
            const std::size_t irank = transfer1.aggregates[inode].rank;
            for (std::size_t i = 0; i < irank; ++i) {
                for (std::size_t j = 0; j < jrank; ++j) {
                    out.fp64[(ioff + i) * out.n + (joff + j)] =
                        block[i * kCandidates + j];
                }
            }
        }
    }
    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0; i < out.n; ++i) {
        const double d = out.fp64[i * out.n + i];
        if (!(d > 0.0) || !std::isfinite(d)) {
            throw std::runtime_error("M5 L2 shell A2 diagonal invalid");
        }
        norm2 += d * d;
        for (std::size_t j = i + 1U; j < out.n; ++j) {
            const double aij = out.fp64[i * out.n + j];
            const double aji = out.fp64[j * out.n + i];
            const double diff = aij - aji;
            asym2 += 2.0 * diff * diff;
            norm2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            out.fp64[i * out.n + j] = sym;
            out.fp64[j * out.n + i] = sym;
        }
    }
    out.symmetry_relative_defect = norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;
    out.assembly_ms = std::chrono::duration<double, std::milli>(
        BenchClock::now() - start).count();
    return out;
}

std::vector<double> m5_dense_square_apply(const M5DenseA2& a,
                                           const std::vector<double>& x) {
    if (x.size() != a.n) throw std::invalid_argument("M5 L2 shell dense A2 apply size");
    std::vector<double> y(a.n, 0.0);
    for (std::size_t i = 0; i < a.n; ++i) {
        const double* row = a.fp64.data() + i * a.n;
        for (std::size_t j = 0; j < a.n; ++j) y[i] += row[j] * x[j];
    }
    return y;
}

std::vector<float> m5_inverse_blocks_6x6_fp32(const L1BlockMetric& metric) {
    if (metric.nodes() == 0U || metric.dofs() != metric.nodes() * 6U) {
        throw std::runtime_error("M5 L2 shell requires rank-6 L2 blocks");
    }
    std::vector<float> inverse(metric.nodes() * 36U, 0.0f);
    for (std::size_t node = 0; node < metric.nodes(); ++node) {
        const std::size_t begin = metric.dof_offsets[node];
        const std::size_t rank = metric.dof_offsets[node + 1U] - begin;
        if (rank != 6U) throw std::runtime_error("M5 L2 shell found non-rank6 block");
        const double* l = metric.lower.data() + metric.value_offsets[node];
        float* dst = inverse.data() + node * 36U;
        for (std::size_t col = 0; col < 6U; ++col) {
            std::array<double, 6U> y{};
            std::array<double, 6U> x{};
            for (std::size_t i = 0; i < 6U; ++i) {
                double value = i == col ? 1.0 : 0.0;
                for (std::size_t j = 0; j < i; ++j) value -= l[i * 6U + j] * y[j];
                y[i] = value / l[i * 6U + i];
            }
            for (std::size_t ii = 6U; ii-- > 0U;) {
                double value = y[ii];
                for (std::size_t j = ii + 1U; j < 6U; ++j) {
                    value -= l[j * 6U + ii] * x[j];
                }
                x[ii] = value / l[ii * 6U + ii];
            }
            for (std::size_t row = 0; row < 6U; ++row) {
                dst[row * 6U + col] = static_cast<float>(x[row]);
            }
        }
    }
    return inverse;
}

struct M5DenseP2 {
    std::size_t rows{0U};
    std::size_t cols{0U};
    std::vector<double> fp64;
    double assembly_ms{0.0};
};

M5DenseP2 m5_assemble_dense_p2(
    const CandidateTransfer& transfer2,
    const L1BlockMetric& block2,
    const std::vector<LocalColumns>& bottom_basis) {
    const auto start = BenchClock::now();
    M5DenseP2 out;
    out.rows = block2.dofs();
    out.cols = transfer2.coarse_dofs;
    if (out.rows == 0U || out.cols == 0U ||
        bottom_basis.size() != transfer2.aggregates.size()) {
        throw std::invalid_argument("M5 L2 shell P2 support/layout mismatch");
    }
    out.fp64.assign(out.rows * out.cols, 0.0);
    for (std::size_t cnode = 0; cnode < bottom_basis.size(); ++cnode) {
        const auto& basis = bottom_basis[cnode];
        const std::size_t coff = transfer2.aggregates[cnode].coarse_offset;
        const std::size_t crank = transfer2.aggregates[cnode].rank;
        if (basis.cols != crank || crank == 0U || crank > 6U) {
            throw std::runtime_error("M5 L2 shell P2 coarse rank mismatch");
        }
        for (const auto& entry : basis.values) {
            const std::size_t rnode = entry.first;
            if (rnode >= block2.nodes()) throw std::out_of_range("M5 L2 shell P2 row node");
            const std::size_t roff = block2.dof_offsets[rnode];
            const std::size_t rrank = block2.dof_offsets[rnode + 1U] - roff;
            for (std::size_t r = 0; r < rrank; ++r) {
                for (std::size_t q = 0; q < crank; ++q) {
                    out.fp64[(roff + r) * out.cols + (coff + q)] =
                        entry.second[r * 6U + q];
                }
            }
        }
    }
    out.assembly_ms = std::chrono::duration<double, std::milli>(
        BenchClock::now() - start).count();
    return out;
}

std::vector<double> m5_dense_p2_apply(const M5DenseP2& p,
                                       const std::vector<double>& x) {
    if (x.size() != p.cols) throw std::invalid_argument("M5 L2 shell P2 apply size");
    std::vector<double> y(p.rows, 0.0);
    for (std::size_t i = 0; i < p.rows; ++i) {
        for (std::size_t j = 0; j < p.cols; ++j) {
            y[i] += p.fp64[i * p.cols + j] * x[j];
        }
    }
    return y;
}

std::vector<double> m5_dense_p2t_apply(const M5DenseP2& p,
                                        const std::vector<double>& x) {
    if (x.size() != p.rows) throw std::invalid_argument("M5 L2 shell P2T apply size");
    std::vector<double> y(p.cols, 0.0);
    for (std::size_t i = 0; i < p.rows; ++i) {
        for (std::size_t j = 0; j < p.cols; ++j) {
            y[j] += p.fp64[i * p.cols + j] * x[i];
        }
    }
    return y;
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
            throw std::invalid_argument("invalid M5 L2 full-shell options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr std::size_t nu2 = 1U;
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
        const auto l2_basis_start = BenchClock::now();
        const auto l2_basis = build_smoothed_candidate_supports(
            transfer1_tentative, strength1.graph, block1, omega1, m1, local_a1_apply_lambda);
        const auto block2 = build_metric_from_local_supports(
            transfer1_tentative, block1, l2_basis, local_a1_apply_lambda);
        const double l2_basis_metric_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - l2_basis_start).count();
        const double block2_oracle_error = audit_l1_block_metric(block2, apply2);
        const double lambda2 = estimate_lambda_max_l1_block(apply2, block2, 8U);
        const double omega2 = kSaDampingNumerator / lambda2;

        const auto a2 = m5_assemble_dense_a2(
            transfer1_tentative, block1, l2_basis, local_a1_apply);
        const auto inverse2 = m5_inverse_blocks_6x6_fp32(block2);

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
        const auto p2_basis_start = BenchClock::now();
        const auto bottom_basis = build_smoothed_candidate_supports(
            transfer2_tentative,
            transfer1_tentative.coarse_graph,
            block2,
            omega2,
            m2,
            local_a2_apply_lambda);
        const auto p2 = m5_assemble_dense_p2(transfer2_tentative, block2, bottom_basis);
        const double p2_basis_total_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - p2_basis_start).count();
        const auto setup_stop = BenchClock::now();

        const auto a2_probe = m5_l2_shell_probe(block2.dofs(), 1.0e-9, 0.41);
        const double a2_materialized_error = m5_l2_shell_relative_error(
            m5_dense_square_apply(a2, a2_probe), apply2(a2_probe));
        const auto p2_probe = m5_l2_shell_probe(transfer2_tentative.coarse_dofs, 2.0e-10, 0.57);
        const double p2_materialized_error = m5_l2_shell_relative_error(
            m5_dense_p2_apply(p2, p2_probe), transfer2.prolong(p2_probe));
        const auto p2t_probe = m5_l2_shell_probe(block2.dofs(), 1.0e-9, 0.79);
        const double p2t_materialized_error = m5_l2_shell_relative_error(
            m5_dense_p2t_apply(p2, p2t_probe), transfer2.restrict_transpose(p2t_probe));

        // Use A2*x_seed as the shell RHS so its numerical scale comes from the
        // actual hierarchy rather than an arbitrary physical-unit magnitude.
        const auto x_seed = m5_l2_shell_probe(block2.dofs(), 1.0e-9, 0.23);
        const auto rhs = apply2(x_seed);
        const auto external_l3 = m5_l2_shell_probe(
            transfer2_tentative.coarse_dofs, 1.0e-10, 0.91);

        // Independent FP64 oracle uses the original nested A2 and factorized P2.
        const double weight2 = 1.0 / (0.55 * lambda2);
        auto x_cpu = block2.solve(rhs);
        for (double& v : x_cpu) v *= weight2;
        const auto ax_pre = apply2(x_cpu);
        Vec residual2(rhs.size(), 0.0);
        for (std::size_t i = 0; i < rhs.size(); ++i) residual2[i] = rhs[i] - ax_pre[i];
        const auto residual3_cpu = transfer2.restrict_transpose(residual2);
        x_cpu = m5_l2_shell_axpy(x_cpu, transfer2.prolong(external_l3), 1.0);
        const auto ax_post = apply2(x_cpu);
        Vec post_residual(rhs.size(), 0.0);
        for (std::size_t i = 0; i < rhs.size(); ++i) post_residual[i] = rhs[i] - ax_post[i];
        x_cpu = m5_l2_shell_axpy(x_cpu, block2.solve(post_residual), weight2);

        const auto gpu = gfss::benchmark_m5_l2_full_shell(
            m5_l2_shell_to_float(a2.fp64),
            block2.dofs(),
            inverse2,
            m5_l2_shell_to_float(p2.fp64),
            transfer2_tentative.coarse_dofs,
            m5_l2_shell_to_float(rhs),
            m5_l2_shell_to_float(external_l3),
            lambda2,
            repeats);

        const double residual3_error = m5_l2_shell_relative_error(
            m5_l2_shell_to_double(gpu.l3_residual), residual3_cpu);
        const double final_x_error = m5_l2_shell_relative_error(
            m5_l2_shell_to_double(gpu.final_l2_correction), x_cpu);
        const bool oracle_ok =
            block1_oracle_error <= 1.0e-10 &&
            block2_oracle_error <= 1.0e-10 &&
            a2_materialized_error <= 1.0e-10 &&
            p2_materialized_error <= 1.0e-10 &&
            p2t_materialized_error <= 1.0e-10 &&
            residual3_error <= 1.0e-4 &&
            final_x_error <= 1.0e-4;

        const std::size_t legacy_factorized_l2_a2_applies = 2U * nu2 + 1U + 2U * m2;
        const std::size_t dense_a2_bytes = a2.fp64.size() * sizeof(float);
        const std::size_t dense_p2_bytes = p2.fp64.size() * sizeof(float);

        std::cout << "GFSS M5 persistent CUDA complete L2 V-cycle shell\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "shell=degree1_actual_block_pre_then_dense_P2T_then_external_L3_correction_then_dense_P2_then_degree1_post\n"
                  << "A2_representation=dense_FP32_cuBLAS_SGEMV\n"
                  << "P2_representation=single_dense_FP32_payload_cuBLAS_OP_N_OP_T\n"
                  << "hierarchy=frozen_theta_0p05_actual_L1_L2_block_metrics\n"
                  << "recursive_schedule_target=1x1\n"
                  << "production_policy_target=5x1x1\n"
                  << "zero_start_pre_A2_elided=true\n"
                  << "host_round_trips_inside_timed_shell=false\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L2_dofs=" << block2.dofs()
                  << " L2_nodes=" << block2.nodes()
                  << " L3_dofs=" << transfer2_tentative.coarse_dofs
                  << " L3_nodes=" << transfer2_tentative.aggregates.size() << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0 << " omega0=" << omega0
                  << " lambda1=" << lambda1 << " omega1=" << omega1
                  << " lambda2=" << lambda2 << " omega2=" << omega2 << '\n'
                  << "P0_support_setup_ms=" << p0_support_ms
                  << " L2_basis_metric_setup_ms=" << l2_basis_metric_ms
                  << " A2_dense_assembly_ms=" << a2.assembly_ms
                  << " P2_basis_dense_setup_ms=" << p2_basis_total_ms
                  << " P2_dense_assembly_ms=" << p2.assembly_ms
                  << " total_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error
                  << " L2_block_vs_nested_relative_error=" << block2_oracle_error << '\n'
                  << "A2_dense_vs_nested_relative_error=" << a2_materialized_error
                  << " A2_symmetry_relative_defect=" << a2.symmetry_relative_defect << '\n'
                  << "P2_dense_vs_factorized_relative_error=" << p2_materialized_error
                  << " P2T_dense_vs_factorized_relative_error=" << p2t_materialized_error << '\n'
                  << "gpu_vs_cpu_fp64_L3_residual_relative_error=" << residual3_error
                  << " gpu_vs_cpu_fp64_final_L2_correction_relative_error=" << final_x_error
                  << " oracle_accept_1e-4=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "A2_dense_matrix_bytes_fp32=" << dense_a2_bytes
                  << " P2_dense_matrix_bytes_fp32=" << dense_p2_bytes
                  << " L2_inverse_block_bytes_fp32=" << inverse2.size() * sizeof(float)
                  << " hierarchy_payload_bytes="
                  << dense_a2_bytes + dense_p2_bytes + inverse2.size() * sizeof(float) << '\n'
                  << "mathematical_A2_applies=3"
                  << " executed_A2_applies=" << gpu.executed_a2_applies
                  << " legacy_factorized_L2_A2_applies=" << legacy_factorized_l2_a2_applies
                  << '\n'
                  << "median_total_ms=" << gpu.median_timing.total_ms
                  << " best_total_ms=" << gpu.best_timing.total_ms
                  << " median_pre_smooth_ms=" << gpu.median_timing.pre_smooth_ms
                  << " median_residual_A2_ms=" << gpu.median_timing.residual_a2_ms
                  << " median_residual_update_ms=" << gpu.median_timing.residual_update_ms
                  << " median_P2T_ms=" << gpu.median_timing.p2t_ms
                  << " median_P2_ms=" << gpu.median_timing.p2_ms
                  << " median_correction_ms=" << gpu.median_timing.correction_ms
                  << " median_post_A2_ms=" << gpu.median_timing.post_a2_ms
                  << " median_post_smooth_ms=" << gpu.median_timing.post_smooth_ms << '\n'
                  << "persistent_L2_shell_device_bytes=" << gpu.device_bytes
                  << " zero_start_pre_A2_elided=true\n";

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_l2_full_shell_bench "
                  << "[repeats=50 [block_y=4 [target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
