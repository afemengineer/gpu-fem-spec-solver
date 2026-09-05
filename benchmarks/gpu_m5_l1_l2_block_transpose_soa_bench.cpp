// M5 GPU productionization stage 4c: close the remaining P1^T gap by keeping
// the natural 6x6 block representation but duplicating only the block-value
// payload in transpose/component order. The transpose payload is packed per L2
// node as [q][r][incident_block], allowing one warp per output component to read
// coefficients coalesced while retaining block-level indices and no atomics.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "gfss/gpu_m5_rectangular_transfer.hpp"
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

struct DualScalarTransfer {
    RectCsr forward;
    RectCsr transpose;
};

struct Block6TransposeSoA {
    std::size_t block_rows{0U};
    std::size_t block_cols{0U};
    std::size_t block_nnz{0U};
    std::vector<std::uint32_t> column_offsets;
    std::vector<std::uint32_t> row_indices;
    std::vector<float> values_q_r_entry;
    std::size_t forward_index_bytes{0U};

    std::size_t one_value_payload_bytes() const noexcept {
        return block_nnz * 36U * sizeof(float);
    }
    std::size_t transpose_index_bytes() const noexcept {
        return (column_offsets.size() + row_indices.size()) * sizeof(std::uint32_t);
    }
    std::size_t dual_value_matrix_bytes() const noexcept {
        // One 6x6 payload in forward block-row order plus one transpose-ordered
        // 6x6 payload. P1^T needs only column offsets + source block rows.
        return 2U * one_value_payload_bytes() +
               forward_index_bytes + transpose_index_bytes();
    }
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

std::vector<double> to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

double relative_error(const std::vector<double>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("M5 transpose SoA oracle size mismatch");
    }
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = got[i] - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    if (!(r2 > 0.0)) throw std::runtime_error("M5 transpose SoA oracle norm zero");
    return std::sqrt(d2 / r2);
}

RectCsr finalize_rows(
    std::size_t rows,
    std::size_t cols,
    std::vector<std::vector<std::pair<std::uint32_t, double>>>& entries) {
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
            const auto col = row[p].first;
            double value = 0.0;
            do {
                value += row[p].second;
                ++p;
            } while (p < row.size() && row[p].first == col);
            if (value == 0.0) continue;
            if (col >= cols ||
                out.column_indices.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("M5 transpose SoA scalar indexing overflow");
            }
            out.column_indices.push_back(col);
            out.values_fp64.push_back(value);
            out.values_fp32.push_back(static_cast<float>(value));
        }
    }
    out.row_offsets[rows] = static_cast<std::uint32_t>(out.column_indices.size());
    return out;
}

DualScalarTransfer assemble_scalar_transfer(
    const CandidateTransfer& transfer,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis) {
    const std::size_t l1_dofs = block1.dofs();
    const std::size_t l2_dofs = transfer.coarse_dofs;
    std::vector<std::vector<std::pair<std::uint32_t, double>>> forward_rows(l1_dofs);
    std::vector<std::vector<std::pair<std::uint32_t, double>>> transpose_rows(l2_dofs);

    for (std::size_t l2_node = 0; l2_node < l2_basis.size(); ++l2_node) {
        const auto& basis = l2_basis[l2_node];
        const std::size_t coarse_offset = transfer.aggregates[l2_node].coarse_offset;
        const std::size_t rank2 = transfer.aggregates[l2_node].rank;
        for (const auto& entry : basis.values) {
            const std::size_t l1_node = entry.first;
            const std::size_t row_offset = block1.dof_offsets[l1_node];
            const std::size_t rank1 = block1.dof_offsets[l1_node + 1U] - row_offset;
            for (std::size_t r = 0; r < rank1; ++r) {
                for (std::size_t q = 0; q < rank2; ++q) {
                    const double value = entry.second[r * 6U + q];
                    if (value == 0.0) continue;
                    const std::uint32_t grow = static_cast<std::uint32_t>(row_offset + r);
                    const std::uint32_t gcol = static_cast<std::uint32_t>(coarse_offset + q);
                    forward_rows[grow].emplace_back(gcol, value);
                    transpose_rows[gcol].emplace_back(grow, value);
                }
            }
        }
    }
    DualScalarTransfer out;
    out.forward = finalize_rows(l1_dofs, l2_dofs, forward_rows);
    out.transpose = finalize_rows(l2_dofs, l1_dofs, transpose_rows);
    return out;
}

Block6TransposeSoA assemble_block6_transpose_soa(
    const CandidateTransfer& transfer,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis) {
    Block6TransposeSoA out;
    out.block_rows = block1.nodes();
    out.block_cols = l2_basis.size();
    out.column_offsets.resize(out.block_cols + 1U, 0U);
    std::vector<std::size_t> forward_counts(out.block_rows, 0U);

    using Entry = std::pair<std::uint32_t, std::array<double, 36U>>;
    std::vector<std::vector<Entry>> columns(out.block_cols);
    for (std::size_t col = 0; col < out.block_cols; ++col) {
        const auto& basis = l2_basis[col];
        const std::size_t rank2 = transfer.aggregates[col].rank;
        if (rank2 == 0U || rank2 > 6U || basis.cols != rank2) {
            throw std::runtime_error("M5 transpose SoA L2 rank mismatch");
        }
        auto& entries = columns[col];
        entries.reserve(basis.values.size());
        for (const auto& item : basis.values) {
            const std::size_t row = item.first;
            if (row >= out.block_rows) throw std::out_of_range("M5 transpose SoA L1 node");
            const std::size_t rank1 = block1.dof_offsets[row + 1U] - block1.dof_offsets[row];
            std::array<double, 36U> block{};
            bool nonzero = false;
            for (std::size_t r = 0; r < rank1; ++r) {
                for (std::size_t q = 0; q < rank2; ++q) {
                    const double value = item.second[r * 6U + q];
                    block[r * 6U + q] = value;
                    nonzero = nonzero || (value != 0.0);
                }
            }
            if (nonzero) entries.emplace_back(static_cast<std::uint32_t>(row), block);
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return a.first < b.first;
        });
        out.column_offsets[col] = static_cast<std::uint32_t>(out.row_indices.size());
        for (const auto& e : entries) {
            out.row_indices.push_back(e.first);
            ++forward_counts[e.first];
        }
        const std::size_t count = entries.size();
        // The CUDA kernel expects each column payload as [q][r][entry].
        for (std::size_t q = 0; q < 6U; ++q) {
            for (std::size_t r = 0; r < 6U; ++r) {
                for (std::size_t k = 0; k < count; ++k) {
                    out.values_q_r_entry.push_back(
                        static_cast<float>(entries[k].second[r * 6U + q]));
                }
            }
        }
    }
    out.column_offsets[out.block_cols] = static_cast<std::uint32_t>(out.row_indices.size());
    out.block_nnz = out.row_indices.size();
    if (out.values_q_r_entry.size() != out.block_nnz * 36U) {
        throw std::runtime_error("M5 transpose SoA value payload mismatch");
    }
    out.forward_index_bytes =
        (out.block_rows + 1U + out.block_nnz) * sizeof(std::uint32_t);
    return out;
}

std::vector<double> pad_l1(const L1BlockMetric& block1, const std::vector<double>& packed) {
    if (packed.size() != block1.dofs()) throw std::invalid_argument("M5 transpose SoA pad size");
    std::vector<double> padded(block1.nodes() * 6U, 0.0);
    for (std::size_t node = 0; node < block1.nodes(); ++node) {
        const std::size_t begin = block1.dof_offsets[node];
        const std::size_t rank = block1.dof_offsets[node + 1U] - begin;
        for (std::size_t r = 0; r < rank; ++r) padded[node * 6U + r] = packed[begin + r];
    }
    return padded;
}

std::vector<double> unpad_l2(const CandidateTransfer& transfer, const std::vector<double>& padded) {
    if (padded.size() != transfer.aggregates.size() * 6U) {
        throw std::invalid_argument("M5 transpose SoA unpad size");
    }
    std::vector<double> packed(transfer.coarse_dofs, 0.0);
    for (std::size_t node = 0; node < transfer.aggregates.size(); ++node) {
        const std::size_t begin = transfer.aggregates[node].coarse_offset;
        const std::size_t rank = transfer.aggregates[node].rank;
        for (std::size_t q = 0; q < rank; ++q) packed[begin + q] = padded[node * 6U + q];
    }
    return packed;
}

std::vector<double> block6_transpose_soa_cpu(
    const Block6TransposeSoA& p,
    const std::vector<double>& x_padded) {
    if (x_padded.size() != p.block_rows * 6U) {
        throw std::invalid_argument("M5 transpose SoA CPU x size");
    }
    std::vector<double> y(p.block_cols * 6U, 0.0);
    for (std::size_t col = 0; col < p.block_cols; ++col) {
        const std::size_t first = p.column_offsets[col];
        const std::size_t last = p.column_offsets[col + 1U];
        const std::size_t count = last - first;
        const std::size_t base = first * 36U;
        for (std::size_t q = 0; q < 6U; ++q) {
            double sum = 0.0;
            for (std::size_t local = 0; local < count; ++local) {
                const std::size_t row = p.row_indices[first + local];
                for (std::size_t r = 0; r < 6U; ++r) {
                    const std::size_t coeff = base + (q * 6U + r) * count + local;
                    sum += static_cast<double>(p.values_q_r_entry[coeff]) *
                           x_padded[row * 6U + r];
                }
            }
            y[col * 6U + q] = sum;
        }
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
            throw std::invalid_argument("invalid M5 transpose SoA options");
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

        const auto scalar_start = BenchClock::now();
        const auto scalar = assemble_scalar_transfer(transfer1_tentative, block1, l2_basis);
        const double scalar_assembly_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - scalar_start).count();
        const auto block_start = BenchClock::now();
        const auto block_soa = assemble_block6_transpose_soa(
            transfer1_tentative, block1, l2_basis);
        const double block_assembly_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - block_start).count();
        const auto setup_stop = BenchClock::now();

        const auto fine_probe = deterministic_probe(space0.coarse_dofs, 1.0e-9, 0.71);
        const auto factorized_transpose = transfer1.restrict_transpose(fine_probe);
        const auto fine_padded = pad_l1(block1, fine_probe);
        const auto cpu_soa_packed = unpad_l2(
            transfer1_tentative, block6_transpose_soa_cpu(block_soa, fine_padded));
        const double cpu_soa_error = relative_error(cpu_soa_packed, factorized_transpose);

        const auto gpu_scalar = gfss::benchmark_m5_rectangular_csr(
            scalar.transpose.rows, scalar.transpose.cols,
            scalar.transpose.row_offsets, scalar.transpose.column_indices,
            scalar.transpose.values_fp32, to_float(fine_probe), repeats);
        const auto gpu_soa = gfss::benchmark_m5_block6_transpose_soa(
            block_soa.block_rows, block_soa.block_cols,
            block_soa.column_offsets, block_soa.row_indices,
            block_soa.values_q_r_entry, to_float(fine_padded), repeats);
        const auto gpu_soa_packed = unpad_l2(
            transfer1_tentative, to_double(gpu_soa.y_padded));
        const double gpu_soa_error = relative_error(gpu_soa_packed, factorized_transpose);

        const auto a1_probe = deterministic_probe(space0.coarse_dofs, 1.0e-9, 0.43);
        gfss::GpuSmoothedAggregationContext a1_gpu(
            mesh, material, space0, omega0, block_y);
        const auto a1_timing = a1_gpu.apply(to_float(a1_probe), m0, repeats);
        const double factorized_lower_bound_ms =
            static_cast<double>(m1) * a1_timing.median_timing.total_ms;

        const std::size_t scalar_dual_bytes =
            scalar.forward.matrix_bytes_fp32() + scalar.transpose.matrix_bytes_fp32();
        const std::size_t dual_block_bytes = block_soa.dual_value_matrix_bytes();
        const bool oracle_ok =
            block1_oracle_error <= 1.0e-10 &&
            cpu_soa_error <= 1.0e-6 &&
            gpu_soa_error <= 1.0e-4;
        const bool soa_beats_scalar = gpu_soa.timing.median_ms < gpu_scalar.timing.median_ms;
        const bool soa_beats_factorized = gpu_soa.timing.median_ms < factorized_lower_bound_ms;
        const bool dual_block_less_memory = dual_block_bytes < scalar_dual_bytes;

        std::cout << "GFSS M5 L1->L2 transpose-ordered block6 decision\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "hierarchy=frozen_theta_0p05_actual_L1_block_metric\n"
                  << "m0=1 m1=2\n"
                  << "candidate=dual_block6_values_forward_row_order_plus_transpose_q_r_entry_order\n"
                  << "transpose_kernel=6_warps_per_L2_node_one_warp_per_output_component\n"
                  << "scalar_baseline=explicit_scalar_CSR_P1T\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L1_nodes=" << block1.nodes()
                  << " L2_dofs=" << transfer1_tentative.coarse_dofs
                  << " L2_nodes=" << transfer1_tentative.aggregates.size() << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0 << " omega0=" << omega0
                  << " block_lambda1=" << block_lambda1
                  << " block_omega1=" << block_omega1 << '\n'
                  << "P0_support_setup_ms=" << p0_support_ms
                  << " P1_smoothed_support_setup_ms=" << basis_setup_ms
                  << " scalar_CSR_assembly_ms=" << scalar_assembly_ms
                  << " block6_transpose_SoA_assembly_ms=" << block_assembly_ms
                  << " total_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error
                  << " cpu_block6_transpose_SoA_vs_factorized_relative_error=" << cpu_soa_error
                  << " gpu_block6_transpose_SoA_vs_factorized_relative_error=" << gpu_soa_error << '\n'
                  << std::fixed << std::setprecision(6)
                  << "P1_block_nnz=" << block_soa.block_nnz
                  << " P1_blocks_per_L2_node="
                  << static_cast<double>(block_soa.block_nnz) /
                     static_cast<double>(block_soa.block_cols) << '\n'
                  << "block6_one_value_payload_bytes_fp32="
                  << block_soa.one_value_payload_bytes()
                  << " block6_forward_index_bytes=" << block_soa.forward_index_bytes
                  << " block6_transpose_index_bytes=" << block_soa.transpose_index_bytes()
                  << " dual_block6_matrix_bytes_fp32=" << dual_block_bytes << '\n'
                  << "scalar_dual_CSR_matrix_bytes_fp32=" << scalar_dual_bytes
                  << " dual_block6_vs_scalar_dual_memory_ratio="
                  << static_cast<double>(dual_block_bytes) /
                     static_cast<double>(std::max<std::size_t>(scalar_dual_bytes, 1U))
                  << " dual_block6_bytes_per_fine_dof="
                  << static_cast<double>(dual_block_bytes) /
                     static_cast<double>(mesh.dof_count()) << '\n'
                  << "A1_median_ms=" << a1_timing.median_timing.total_ms
                  << " factorized_P1T_strict_lower_bound_ms=" << factorized_lower_bound_ms << '\n'
                  << "scalar_P1T_median_ms=" << gpu_scalar.timing.median_ms
                  << " scalar_P1T_best_ms=" << gpu_scalar.timing.best_ms
                  << " block6_transpose_SoA_median_ms=" << gpu_soa.timing.median_ms
                  << " block6_transpose_SoA_best_ms=" << gpu_soa.timing.best_ms
                  << " block6_SoA_vs_scalar_P1T_speedup="
                  << gpu_scalar.timing.median_ms /
                     std::max(gpu_soa.timing.median_ms, 1.0e-30)
                  << " block6_SoA_vs_factorized_lower_bound_speedup="
                  << factorized_lower_bound_ms /
                     std::max(gpu_soa.timing.median_ms, 1.0e-30) << '\n'
                  << "gpu_block6_transpose_SoA_device_bytes=" << gpu_soa.device_bytes << '\n'
                  << "oracle_accept=" << (oracle_ok ? "true" : "false")
                  << " transpose_SoA_beats_scalar=" << (soa_beats_scalar ? "true" : "false")
                  << " transpose_SoA_beats_factorized_lower_bound="
                  << (soa_beats_factorized ? "true" : "false")
                  << " dual_block6_uses_less_memory_than_scalar_dual="
                  << (dual_block_less_memory ? "true" : "false") << '\n'
                  << "preferred_transpose_representation="
                  << (oracle_ok && soa_beats_scalar && dual_block_less_memory
                          ? "transpose_ordered_block6_SoA"
                          : (oracle_ok && dual_block_less_memory
                                ? "scalar_speed_vs_dual_block6_memory_tradeoff"
                                : "scalar_CSR"))
                  << '\n';

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_l1_l2_block_transpose_soa_bench "
                  << "[repeats=100 [block_y=4 [target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
