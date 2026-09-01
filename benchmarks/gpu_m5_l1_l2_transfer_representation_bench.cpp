// M5 GPU productionization stage 4: decide whether the frozen m1=2 L1<->L2
// smoothed transfer should remain factorized at runtime or be materialized once
// from the already-validated localized P1 supports. The explicit matrix is
// assembled exactly from those supports, audited against the FP64 factorized
// transfer in both directions, then benchmarked as persistent FP32 rectangular
// CSR on GPU. The factorized comparator is a strict lower bound: each direction
// requires at least m1=2 complete A1 actions before tentative transfer and block
// metric work are counted.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "gfss/gpu_m5_rectangular_transfer.hpp"
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
#include <utility>
#include <vector>

namespace {

using BenchClock = std::chrono::steady_clock;

struct RectCsr {
    std::size_t rows{0U};
    std::size_t cols{0U};
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
    std::vector<double> values_fp64;
    std::vector<float> values_fp32;

    std::size_t matrix_bytes_fp32() const noexcept {
        return row_offsets.size() * sizeof(std::uint32_t) +
               column_indices.size() * sizeof(std::uint32_t) +
               values_fp32.size() * sizeof(float);
    }
};

struct ExplicitP1 {
    RectCsr forward;
    RectCsr transpose;
    double assembly_ms{0.0};
};

std::vector<double> deterministic_probe(std::size_t n, double scale, double phase) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = scale * (std::sin(0.011 * t + phase) +
                        0.37 * std::cos(0.031 * t - 0.53 * phase));
    }
    return v;
}

std::vector<float> to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

double relative_error(const std::vector<double>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("M5 P1 CPU oracle size mismatch");
    }
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = got[i] - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    if (!(r2 > 0.0)) throw std::runtime_error("M5 P1 CPU oracle norm zero");
    return std::sqrt(d2 / r2);
}

double relative_error(const std::vector<float>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("M5 P1 GPU oracle size mismatch");
    }
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    if (!(r2 > 0.0)) throw std::runtime_error("M5 P1 GPU oracle norm zero");
    return std::sqrt(d2 / r2);
}

RectCsr finalize_rows(
    std::size_t rows,
    std::size_t cols,
    std::vector<std::vector<std::pair<std::uint32_t, double>>>& entries) {
    if (entries.size() != rows) throw std::invalid_argument("M5 P1 row-entry size mismatch");
    RectCsr out;
    out.rows = rows;
    out.cols = cols;
    out.row_offsets.resize(rows + 1U, 0U);

    for (std::size_t r = 0; r < rows; ++r) {
        auto& row = entries[r];
        std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        out.row_offsets[r] = static_cast<std::uint32_t>(out.column_indices.size());
        std::size_t p = 0U;
        while (p < row.size()) {
            const std::uint32_t col = row[p].first;
            double value = 0.0;
            do {
                value += row[p].second;
                ++p;
            } while (p < row.size() && row[p].first == col);
            if (value == 0.0) continue;
            if (col >= cols ||
                out.column_indices.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("M5 P1 CSR indexing overflow");
            }
            out.column_indices.push_back(col);
            out.values_fp64.push_back(value);
            out.values_fp32.push_back(static_cast<float>(value));
        }
    }
    out.row_offsets[rows] = static_cast<std::uint32_t>(out.column_indices.size());
    return out;
}

ExplicitP1 assemble_explicit_p1(
    const CandidateTransfer& transfer1_tentative,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis) {
    const auto start = BenchClock::now();
    const std::size_t l1_dofs = block1.dofs();
    const std::size_t l2_dofs = transfer1_tentative.coarse_dofs;
    if (l2_basis.size() != transfer1_tentative.aggregates.size() ||
        l1_dofs == 0U || l2_dofs == 0U) {
        throw std::invalid_argument("M5 P1 support/layout mismatch");
    }

    std::vector<std::vector<std::pair<std::uint32_t, double>>> forward_rows(l1_dofs);
    std::vector<std::vector<std::pair<std::uint32_t, double>>> transpose_rows(l2_dofs);

    for (std::size_t l2_node = 0; l2_node < l2_basis.size(); ++l2_node) {
        const auto& basis = l2_basis[l2_node];
        const std::size_t coarse_offset = transfer1_tentative.aggregates[l2_node].coarse_offset;
        const std::size_t rank2 = transfer1_tentative.aggregates[l2_node].rank;
        if (basis.cols != rank2 || rank2 == 0U || rank2 > kCandidates) {
            throw std::runtime_error("M5 P1 L2 basis rank mismatch");
        }
        for (const auto& entry : basis.values) {
            const std::size_t l1_node = entry.first;
            if (l1_node >= block1.nodes()) throw std::out_of_range("M5 P1 L1 node out of range");
            const std::size_t row_offset = block1.dof_offsets[l1_node];
            const std::size_t rank1 = block1.dof_offsets[l1_node + 1U] - row_offset;
            for (std::size_t r = 0; r < rank1; ++r) {
                const std::uint32_t global_row = static_cast<std::uint32_t>(row_offset + r);
                for (std::size_t q = 0; q < rank2; ++q) {
                    const double value = entry.second[r * kCandidates + q];
                    if (value == 0.0) continue;
                    const std::uint32_t global_col =
                        static_cast<std::uint32_t>(coarse_offset + q);
                    forward_rows[global_row].emplace_back(global_col, value);
                    transpose_rows[global_col].emplace_back(global_row, value);
                }
            }
        }
    }

    ExplicitP1 out;
    out.forward = finalize_rows(l1_dofs, l2_dofs, forward_rows);
    out.transpose = finalize_rows(l2_dofs, l1_dofs, transpose_rows);
    if (out.forward.values_fp64.size() != out.transpose.values_fp64.size()) {
        throw std::runtime_error("M5 P1 forward/transpose nnz mismatch");
    }
    out.assembly_ms = std::chrono::duration<double, std::milli>(
        BenchClock::now() - start).count();
    return out;
}

std::vector<double> csr_apply(const RectCsr& a, const std::vector<double>& x) {
    if (x.size() != a.cols) throw std::invalid_argument("M5 P1 CSR apply size mismatch");
    std::vector<double> y(a.rows, 0.0);
    for (std::size_t r = 0; r < a.rows; ++r) {
        double sum = 0.0;
        for (std::size_t p = a.row_offsets[r]; p < a.row_offsets[r + 1U]; ++p) {
            sum += a.values_fp64[p] * x[a.column_indices[p]];
        }
        y[r] = sum;
    }
    return y;
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
            throw std::invalid_argument("invalid M5 P1 representation options");
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

        const auto local_a1_apply = [&](const LocalColumns& x) {
            return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
        };
        const auto basis_start = BenchClock::now();
        const auto l2_basis = build_smoothed_candidate_supports(
            transfer1_tentative, strength1.graph, block1,
            block_omega1, m1, local_a1_apply);
        const double basis_setup_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - basis_start).count();
        const auto explicit_p1 = assemble_explicit_p1(
            transfer1_tentative, block1, l2_basis);
        const auto setup_stop = BenchClock::now();

        const auto coarse_probe = deterministic_probe(
            transfer1_tentative.coarse_dofs, 1.0e-9, 0.27);
        const auto factorized_forward = transfer1.prolong(coarse_probe);
        const auto explicit_forward = csr_apply(explicit_p1.forward, coarse_probe);
        const double cpu_forward_error = relative_error(explicit_forward, factorized_forward);

        const auto fine_probe = deterministic_probe(space0.coarse_dofs, 1.0e-9, 0.71);
        const auto factorized_transpose = transfer1.restrict_transpose(fine_probe);
        const auto explicit_transpose = csr_apply(explicit_p1.transpose, fine_probe);
        const double cpu_transpose_error = relative_error(explicit_transpose, factorized_transpose);

        const auto gpu_forward = gfss::benchmark_m5_rectangular_csr(
            explicit_p1.forward.rows,
            explicit_p1.forward.cols,
            explicit_p1.forward.row_offsets,
            explicit_p1.forward.column_indices,
            explicit_p1.forward.values_fp32,
            to_float(coarse_probe),
            repeats);
        const auto gpu_transpose = gfss::benchmark_m5_rectangular_csr(
            explicit_p1.transpose.rows,
            explicit_p1.transpose.cols,
            explicit_p1.transpose.row_offsets,
            explicit_p1.transpose.column_indices,
            explicit_p1.transpose.values_fp32,
            to_float(fine_probe),
            repeats);
        const double gpu_forward_error = relative_error(gpu_forward.y, factorized_forward);
        const double gpu_transpose_error = relative_error(gpu_transpose.y, factorized_transpose);

        const auto a1_probe = deterministic_probe(space0.coarse_dofs, 1.0e-9, 0.43);
        gfss::GpuSmoothedAggregationContext a1_gpu(
            mesh, material, space0, omega0, block_y);
        const auto a1_timing = a1_gpu.apply(to_float(a1_probe), m0, repeats);
        const double per_direction_factorized_lower_bound_ms =
            static_cast<double>(m1) * a1_timing.median_timing.total_ms;
        const double roundtrip_factorized_lower_bound_ms =
            2.0 * per_direction_factorized_lower_bound_ms;
        const double explicit_roundtrip_ms =
            gpu_forward.timing.median_ms + gpu_transpose.timing.median_ms;

        const std::size_t single_matrix_bytes = explicit_p1.forward.matrix_bytes_fp32();
        const std::size_t dual_direction_matrix_bytes =
            explicit_p1.forward.matrix_bytes_fp32() +
            explicit_p1.transpose.matrix_bytes_fp32();
        const double density =
            static_cast<double>(explicit_p1.forward.values_fp32.size()) /
            static_cast<double>(explicit_p1.forward.rows * explicit_p1.forward.cols);
        const double nnz_per_l1_row =
            static_cast<double>(explicit_p1.forward.values_fp32.size()) /
            static_cast<double>(explicit_p1.forward.rows);
        const double nnz_per_l2_row_transpose =
            static_cast<double>(explicit_p1.transpose.values_fp32.size()) /
            static_cast<double>(explicit_p1.transpose.rows);

        const bool oracle_ok =
            block1_oracle_error <= 1.0e-10 &&
            cpu_forward_error <= 1.0e-10 &&
            cpu_transpose_error <= 1.0e-10 &&
            gpu_forward_error <= 1.0e-4 &&
            gpu_transpose_error <= 1.0e-4;
        const bool speed_ok =
            gpu_forward.timing.median_ms < per_direction_factorized_lower_bound_ms &&
            gpu_transpose.timing.median_ms < per_direction_factorized_lower_bound_ms;

        std::cout << "GFSS M5 L1<->L2 transfer representation decision\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "hierarchy=frozen_theta_0p05_actual_L1_block_metric\n"
                  << "m0=1 m1=2\n"
                  << "factorized_transfer=P1=(I-omega1*B1^-1*A1)^2*P1_tentative\n"
                  << "explicit_transfer=exact_local_support_P1_structural_CSR_FP32\n"
                  << "transpose_benchmark=separate_CSR_of_P1T_no_atomics\n"
                  << "factorized_comparator=strict_lower_bound_2x_measured_A1_per_direction\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L2_dofs=" << transfer1_tentative.coarse_dofs
                  << " L2_nodes=" << transfer1_tentative.coarse_graph.nodes() << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0 << " omega0=" << omega0
                  << " block_lambda1=" << block_lambda1
                  << " block_omega1=" << block_omega1 << '\n'
                  << "P0_support_setup_ms=" << p0_support_ms
                  << " P1_smoothed_support_setup_ms=" << basis_setup_ms
                  << " P1_explicit_CSR_assembly_ms=" << explicit_p1.assembly_ms
                  << " total_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error << '\n'
                  << "cpu_explicit_P1_vs_factorized_relative_error=" << cpu_forward_error
                  << " cpu_explicit_P1T_vs_factorized_relative_error=" << cpu_transpose_error << '\n'
                  << "gpu_explicit_P1_vs_factorized_relative_error=" << gpu_forward_error
                  << " gpu_explicit_P1T_vs_factorized_relative_error=" << gpu_transpose_error << '\n'
                  << std::fixed << std::setprecision(6)
                  << "P1_nnz=" << explicit_p1.forward.values_fp32.size()
                  << " P1_nnz_per_L1_row=" << nnz_per_l1_row
                  << " P1T_nnz_per_L2_row=" << nnz_per_l2_row_transpose
                  << " P1_scalar_density=" << density << '\n'
                  << "P1_single_CSR_matrix_bytes_fp32=" << single_matrix_bytes
                  << " P1_dual_direction_CSR_matrix_bytes_fp32=" << dual_direction_matrix_bytes
                  << " single_CSR_bytes_per_fine_dof="
                  << static_cast<double>(single_matrix_bytes) /
                     static_cast<double>(mesh.dof_count())
                  << " dual_CSR_bytes_per_fine_dof="
                  << static_cast<double>(dual_direction_matrix_bytes) /
                     static_cast<double>(mesh.dof_count()) << '\n'
                  << "A1_median_ms=" << a1_timing.median_timing.total_ms
                  << " A1_best_ms=" << a1_timing.best_timing.total_ms << '\n'
                  << "factorized_P1_per_direction_strict_lower_bound_ms="
                  << per_direction_factorized_lower_bound_ms
                  << " explicit_P1_median_ms=" << gpu_forward.timing.median_ms
                  << " explicit_P1_best_ms=" << gpu_forward.timing.best_ms
                  << " P1_speedup_over_factorized_lower_bound="
                  << per_direction_factorized_lower_bound_ms /
                     std::max(gpu_forward.timing.median_ms, 1.0e-30) << '\n'
                  << "factorized_P1T_per_direction_strict_lower_bound_ms="
                  << per_direction_factorized_lower_bound_ms
                  << " explicit_P1T_median_ms=" << gpu_transpose.timing.median_ms
                  << " explicit_P1T_best_ms=" << gpu_transpose.timing.best_ms
                  << " P1T_speedup_over_factorized_lower_bound="
                  << per_direction_factorized_lower_bound_ms /
                     std::max(gpu_transpose.timing.median_ms, 1.0e-30) << '\n'
                  << "factorized_roundtrip_strict_lower_bound_ms="
                  << roundtrip_factorized_lower_bound_ms
                  << " explicit_roundtrip_median_ms=" << explicit_roundtrip_ms
                  << " roundtrip_speedup_over_factorized_lower_bound="
                  << roundtrip_factorized_lower_bound_ms /
                     std::max(explicit_roundtrip_ms, 1.0e-30) << '\n'
                  << "gpu_forward_device_bytes_including_vectors=" << gpu_forward.device_bytes
                  << " gpu_transpose_device_bytes_including_vectors="
                  << gpu_transpose.device_bytes << '\n'
                  << "oracle_accept=" << (oracle_ok ? "true" : "false")
                  << " explicit_beats_factorized_lower_bound="
                  << (speed_ok ? "true" : "false") << '\n'
                  << "selective_P1_materialization_candidate="
                  << (oracle_ok && speed_ok ? "true" : "false") << '\n';

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_l1_l2_transfer_representation_bench "
                  << "[repeats=100 [block_y=4 [target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
