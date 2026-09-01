// M5 GPU productionization stage 3: decide whether L2 should remain fully
// factorized or be selectively materialized. Build the exact frozen-hierarchy
// A2=P1^T A1 P1 from localized supports, compress structural zeros into scalar
// CSR, audit it against the nested FP64 operator, then benchmark persistent FP32
// CSR SpMV on GPU. The factorized comparator is a strict lower bound: with
// m1=2, any A2 action requires at least 2*m1+1 = 5 complete A1 actions before
// counting P1/P1^T and block-metric vector work.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "gfss/gpu_m5_l2_materialized.hpp"
#include "gfss/gpu_smoothed_aggregation.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using BenchClock = std::chrono::steady_clock;

struct ExactA2Csr {
    std::size_t n{0U};
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
    std::vector<float> values_fp32;
    std::vector<double> dense_fp64;
    double assembly_ms{0.0};
    double symmetry_relative_defect{0.0};

    std::size_t matrix_bytes_fp32() const noexcept {
        return row_offsets.size() * sizeof(std::uint32_t) +
               column_indices.size() * sizeof(std::uint32_t) +
               values_fp32.size() * sizeof(float);
    }
};

std::vector<double> deterministic_probe(std::size_t n, double scale, double phase) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = scale * (std::sin(0.013 * t + phase) +
                        0.29 * std::cos(0.037 * t - 0.41 * phase));
    }
    return v;
}

std::vector<double> dense_apply(const ExactA2Csr& a, const std::vector<double>& x) {
    if (x.size() != a.n) throw std::invalid_argument("M5 L2 dense apply size mismatch");
    std::vector<double> y(a.n, 0.0);
    for (std::size_t i = 0; i < a.n; ++i) {
        const double* row = a.dense_fp64.data() + i * a.n;
        double sum = 0.0;
        for (std::size_t j = 0; j < a.n; ++j) sum += row[j] * x[j];
        y[i] = sum;
    }
    return y;
}

double relative_error(const std::vector<double>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) throw std::invalid_argument("M5 L2 oracle size mismatch");
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = got[i] - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    if (!(r2 > 0.0)) throw std::runtime_error("M5 L2 oracle norm zero");
    return std::sqrt(d2 / r2);
}

double relative_error(const std::vector<float>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) throw std::invalid_argument("M5 L2 GPU oracle size mismatch");
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    if (!(r2 > 0.0)) throw std::runtime_error("M5 L2 GPU oracle norm zero");
    return std::sqrt(d2 / r2);
}

std::vector<float> to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

ExactA2Csr assemble_exact_a2_csr(
    const CandidateTransfer& transfer1_tentative,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis,
    const std::function<LocalColumns(const LocalColumns&)>& local_a1_apply) {
    const auto start = BenchClock::now();
    if (l2_basis.size() != transfer1_tentative.aggregates.size()) {
        throw std::invalid_argument("M5 L2 basis/transfer size mismatch");
    }
    ExactA2Csr out;
    out.n = transfer1_tentative.coarse_dofs;
    if (out.n == 0U || out.n > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("M5 L2 CSR unsupported dimension");
    }
    out.dense_fp64.assign(out.n * out.n, 0.0);

    // Exact block Galerkin assembly from localized supports. This is the same
    // identity used by the validated local L2 diagonal-block oracle, extended
    // from diagonal blocks to all structurally coupled block pairs.
    for (std::size_t jnode = 0; jnode < l2_basis.size(); ++jnode) {
        const auto applied = local_a1_apply(l2_basis[jnode]);
        const std::size_t joff = transfer1_tentative.aggregates[jnode].coarse_offset;
        const std::size_t jrank = transfer1_tentative.aggregates[jnode].rank;
        for (std::size_t inode = 0; inode < l2_basis.size(); ++inode) {
            const auto block = local_cross_gram(l2_basis[inode], applied, block1);
            const std::size_t ioff = transfer1_tentative.aggregates[inode].coarse_offset;
            const std::size_t irank = transfer1_tentative.aggregates[inode].rank;
            for (std::size_t i = 0; i < irank; ++i) {
                for (std::size_t j = 0; j < jrank; ++j) {
                    out.dense_fp64[(ioff + i) * out.n + (joff + j)] =
                        block[i * kCandidates + j];
                }
            }
        }
    }

    // Symmetrize only roundoff disagreement. Track the defect before averaging.
    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0; i < out.n; ++i) {
        const double di = out.dense_fp64[i * out.n + i];
        if (!(di > 0.0) || !std::isfinite(di)) {
            throw std::runtime_error("M5 L2 materialized diagonal is not positive finite");
        }
        norm2 += di * di;
        for (std::size_t j = i + 1U; j < out.n; ++j) {
            const double aij = out.dense_fp64[i * out.n + j];
            const double aji = out.dense_fp64[j * out.n + i];
            const double d = aij - aji;
            asym2 += 2.0 * d * d;
            norm2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            out.dense_fp64[i * out.n + j] = sym;
            out.dense_fp64[j * out.n + i] = sym;
        }
    }
    out.symmetry_relative_defect = norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;

    // Structural compression: disjoint local supports produce exact zero block
    // entries in the local-cross-Gram construction, so no numerical drop
    // tolerance is needed here. Keep every nonzero scalar generated by a
    // structurally coupled block.
    out.row_offsets.resize(out.n + 1U, 0U);
    for (std::size_t i = 0; i < out.n; ++i) {
        out.row_offsets[i] = static_cast<std::uint32_t>(out.column_indices.size());
        for (std::size_t j = 0; j < out.n; ++j) {
            const double v = out.dense_fp64[i * out.n + j];
            if (v == 0.0) continue;
            if (out.column_indices.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("M5 L2 CSR nnz exceeds 32-bit indexing");
            }
            out.column_indices.push_back(static_cast<std::uint32_t>(j));
            out.values_fp32.push_back(static_cast<float>(v));
        }
    }
    out.row_offsets[out.n] = static_cast<std::uint32_t>(out.column_indices.size());
    out.assembly_ms = std::chrono::duration<double, std::milli>(
        BenchClock::now() - start).count();
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 100;
        const int block_y = argc > 2 ? std::stoi(argv[2]) : 4;
        const std::size_t target_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t min_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 4U;
        if (repeats <= 0 || block_y <= 0 || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid M5 L2 representation options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
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
        const double block_lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
        const double block_omega1 = kSaDampingNumerator / block_lambda1;

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
            transfer1_tentative, apply1, block1, block_omega1, m1};
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
            transfer1_tentative, strength1.graph, block1,
            block_omega1, m1, local_a1_apply_lambda);
        const double l2_basis_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - l2_basis_start).count();

        const auto a2 = assemble_exact_a2_csr(
            transfer1_tentative, block1, l2_basis, local_a1_apply);
        const auto setup_stop = BenchClock::now();

        const auto probe = deterministic_probe(a2.n, 1.0e-9, 0.51);
        const auto nested = apply2(probe);
        const auto materialized = dense_apply(a2, probe);
        const double materialized_vs_nested = relative_error(materialized, nested);

        const auto probe_f = to_float(probe);
        const auto gpu_csr = gfss::benchmark_m5_l2_csr(
            a2.row_offsets, a2.column_indices, a2.values_fp32, probe_f, repeats);
        const double gpu_vs_nested = relative_error(gpu_csr.y, nested);

        // Measure A1 on the same run/GPU. This context is the validated
        // factorized P0^T A0 P0 path used by the L1 stage.
        const auto l1_probe = deterministic_probe(space0.coarse_dofs, 1.0e-9, 0.23);
        gfss::GpuSmoothedAggregationContext a1_gpu(
            mesh, material, space0, omega0, block_y);
        const auto a1_timing = a1_gpu.apply(to_float(l1_probe), m0, repeats);
        const double factorized_a2_lower_bound_ms =
            static_cast<double>(2U * m1 + 1U) * a1_timing.median_timing.total_ms;
        const double lower_bound_speedup = gpu_csr.timing.median_ms > 0.0
            ? factorized_a2_lower_bound_ms / gpu_csr.timing.median_ms : 0.0;

        const std::size_t dense_fp32_bytes = a2.n * a2.n * sizeof(float);
        const double density = a2.n > 0U
            ? static_cast<double>(a2.values_fp32.size()) /
              static_cast<double>(a2.n * a2.n) : 0.0;
        const double nnz_per_row = a2.n > 0U
            ? static_cast<double>(a2.values_fp32.size()) / static_cast<double>(a2.n) : 0.0;
        const std::size_t matrix_csr_bytes = a2.matrix_bytes_fp32();
        const bool oracle_ok = materialized_vs_nested <= 1.0e-10 && gpu_vs_nested <= 1.0e-4;
        const bool speed_ok = gpu_csr.timing.median_ms < factorized_a2_lower_bound_ms;

        std::cout << "GFSS M5 L2 factorized-vs-materialized representation decision\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "hierarchy=frozen_theta_0p05_actual_L1_block_metric\n"
                  << "m0=1 m1=2\n"
                  << "materialized_operator=exact_local_support_A2_then_structural_CSR_FP32\n"
                  << "factorized_comparator=strict_lower_bound_5x_measured_A1_no_P1_overhead\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L2_dofs=" << a2.n
                  << " L2_nodes=" << transfer1_tentative.coarse_graph.nodes() << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0 << " omega0=" << omega0
                  << " block_lambda1=" << block_lambda1
                  << " block_omega1=" << block_omega1 << '\n'
                  << "P0_support_setup_ms=" << p0_support_ms
                  << " L2_basis_setup_ms=" << l2_basis_ms
                  << " A2_exact_block_assembly_ms=" << a2.assembly_ms
                  << " total_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error << '\n'
                  << "A2_symmetry_relative_defect=" << a2.symmetry_relative_defect
                  << " materialized_A2_vs_nested_relative_error=" << materialized_vs_nested
                  << " gpu_CSR_vs_nested_relative_error=" << gpu_vs_nested << '\n'
                  << std::fixed << std::setprecision(6)
                  << "A2_nnz=" << a2.values_fp32.size()
                  << " A2_nnz_per_row=" << nnz_per_row
                  << " A2_scalar_density=" << density << '\n'
                  << "A2_CSR_matrix_bytes_fp32=" << matrix_csr_bytes
                  << " A2_dense_matrix_bytes_fp32=" << dense_fp32_bytes
                  << " CSR_vs_dense_memory_ratio="
                  << static_cast<double>(matrix_csr_bytes) /
                     static_cast<double>(std::max<std::size_t>(dense_fp32_bytes, 1U)) << '\n'
                  << "CSR_bytes_per_L2_dof="
                  << static_cast<double>(matrix_csr_bytes) / static_cast<double>(a2.n)
                  << " CSR_bytes_per_fine_dof="
                  << static_cast<double>(matrix_csr_bytes) /
                     static_cast<double>(mesh.dof_count()) << '\n'
                  << "A1_median_ms=" << a1_timing.median_timing.total_ms
                  << " A1_best_ms=" << a1_timing.best_timing.total_ms << '\n'
                  << "factorized_A2_strict_lower_bound_ms=" << factorized_a2_lower_bound_ms
                  << " materialized_A2_CSR_median_ms=" << gpu_csr.timing.median_ms
                  << " materialized_A2_CSR_best_ms=" << gpu_csr.timing.best_ms
                  << " lower_bound_speedup_materialized_over_factorized="
                  << lower_bound_speedup << '\n'
                  << "gpu_CSR_device_bytes_including_vectors=" << gpu_csr.device_bytes << '\n'
                  << "oracle_accept=" << (oracle_ok ? "true" : "false")
                  << " materialized_beats_factorized_lower_bound="
                  << (speed_ok ? "true" : "false") << '\n'
                  << "selective_L2_materialization_candidate="
                  << (oracle_ok && speed_ok ? "true" : "false") << '\n';

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_l2_representation_bench "
                  << "[repeats=100 [block_y=4 [target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
