// M5 stage 16: use the exact A1 block data already assembled for strength as a
// temporary setup-only block-sparse operator. This avoids repeatedly evaluating
// A1 = P0^T A0 P0 through the fine element path while constructing P1/A2.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_p1_block6_setup.hpp"
#include "m5_l2_dense_setup.hpp"
#include "m5_fast_hierarchy_setup.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using FastClock = m5_fast_setup::Clock;

inline double fast_ms(FastClock::time_point a, FastClock::time_point b) {
    return m5_fast_setup::elapsed_ms(a, b);
}

inline double rel_error(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("materialized A1 oracle size mismatch");
    }
    Vec d(a.size(), 0.0);
    for (std::size_t i = 0U; i < a.size(); ++i) d[i] = a[i] - b[i];
    return norm(d) / std::max(norm(b), 1.0e-300);
}

using m5_fast_setup::apply_supports_parallel;
using m5_fast_setup::build_smoothed_supports_parallel;
using m5_fast_setup::dense_a2_from_cached_applied;
using m5_fast_setup::dense_bottom_from_a2_p2;
using m5_fast_setup::dense_smoothed_p2_from_a2;
using m5_fast_setup::metric_from_cached_applied;

struct MaterializedA1ColumnEntry {
    std::uint32_t row{0U};
    CombinedBlock block{};  // A(row,col), stride 6
};

struct MaterializedA1 {
    std::vector<std::vector<MaterializedA1ColumnEntry>> columns;
    std::size_t directed_blocks{0U};
    std::size_t logical_bytes{0U};
    double setup_ms{0.0};
};

CombinedBlock transpose_block6(const CombinedBlock& a) {
    CombinedBlock t{};
    for (std::size_t i = 0U; i < kCandidates; ++i) {
        for (std::size_t j = 0U; j < kCandidates; ++j) {
            t[i * kCandidates + j] = a[j * kCandidates + i];
        }
    }
    return t;
}

MaterializedA1 materialize_actual_a1(
    const L1BlockMetric& block1,
    const std::unordered_map<std::uint64_t, CombinedBlock>& offdiagonal) {
    const auto start = FastClock::now();
    MaterializedA1 out;
    out.columns.resize(block1.nodes());

    for (std::size_t node = 0U; node < block1.nodes(); ++node) {
        const std::size_t rank = block1.dof_offsets[node + 1U] - block1.dof_offsets[node];
        CombinedBlock diag{};
        for (std::size_t i = 0U; i < rank; ++i) {
            for (std::size_t j = 0U; j < rank; ++j) {
                diag[i * kCandidates + j] = block1.block_entry(node, i, j);
            }
        }
        out.columns[node].push_back({static_cast<std::uint32_t>(node), diag});
    }

    for (const auto& item : offdiagonal) {
        const auto [a, b] = combined_decode_pair(item.first);
        // item.second is A_ab with a < b. Column b contributes A_ab*x_b to row a.
        out.columns[b].push_back({a, item.second});
        // Symmetry gives A_ba = A_ab^T.
        out.columns[a].push_back({b, transpose_block6(item.second)});
    }

    for (auto& column : out.columns) {
        std::sort(column.begin(), column.end(), [](const auto& x, const auto& y) {
            return x.row < y.row;
        });
        out.directed_blocks += column.size();
    }
    out.logical_bytes = out.directed_blocks *
        (sizeof(std::uint32_t) + sizeof(CombinedBlock));
    out.setup_ms = fast_ms(start, FastClock::now());
    return out;
}

LocalColumns apply_materialized_a1(
    const MaterializedA1& a1,
    const L1BlockMetric& layout,
    const LocalColumns& input) {
    if (input.cols == 0U || input.cols > kCandidates) {
        throw std::invalid_argument("materialized A1 local column count invalid");
    }
    LocalColumns out;
    out.cols = input.cols;
    for (const auto& xentry : input.values) {
        const std::size_t col_node = xentry.first;
        if (col_node >= a1.columns.size()) {
            throw std::out_of_range("materialized A1 input node out of range");
        }
        const std::size_t col_rank = layout.dof_offsets[col_node + 1U] -
                                     layout.dof_offsets[col_node];
        for (const auto& entry : a1.columns[col_node]) {
            const std::size_t row_node = entry.row;
            const std::size_t row_rank = layout.dof_offsets[row_node + 1U] -
                                         layout.dof_offsets[row_node];
            auto& y = zero_block(out.values, entry.row);
            for (std::size_t i = 0U; i < row_rank; ++i) {
                for (std::size_t q = 0U; q < col_rank; ++q) {
                    const double a = entry.block[i * kCandidates + q];
                    if (a == 0.0) continue;
                    for (std::size_t c = 0U; c < input.cols; ++c) {
                        y[i * kCandidates + c] +=
                            a * xentry.second[q * kCandidates + c];
                    }
                }
            }
        }
    }
    return out;
}

Vec apply_materialized_a1_vector(
    const MaterializedA1& a1,
    const L1BlockMetric& layout,
    const Vec& x) {
    if (x.size() != layout.dofs()) {
        throw std::invalid_argument("materialized A1 vector size mismatch");
    }
    Vec y(x.size(), 0.0);
    for (std::size_t col_node = 0U; col_node < a1.columns.size(); ++col_node) {
        const std::size_t cb = layout.dof_offsets[col_node];
        const std::size_t cr = layout.dof_offsets[col_node + 1U] - cb;
        for (const auto& entry : a1.columns[col_node]) {
            const std::size_t rb = layout.dof_offsets[entry.row];
            const std::size_t rr = layout.dof_offsets[entry.row + 1U] - rb;
            for (std::size_t i = 0U; i < rr; ++i) {
                double sum = 0.0;
                for (std::size_t q = 0U; q < cr; ++q) {
                    sum += entry.block[i * kCandidates + q] * x[cb + q];
                }
                y[rb + i] += sum;
            }
        }
    }
    return y;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t target_nodes = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 12U;
        const std::size_t min_nodes = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 4U;
        if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid materialized A1 setup options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto production_start = FastClock::now();

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
        const double lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
        const double omega1 = kSaDampingNumerator / lambda1;

        double p0_support_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(
            mesh, material, space0, fine_inverse, omega0, p0_support_ms);
        const auto element_supports = build_element_support_index(mesh, fine_supports);

        const auto offdiag_start = FastClock::now();
        const auto actual_a1_offdiagonal = accumulate_combined_actual_a1_offdiagonal_blocks(
            mesh, material, fine_supports, element_supports);
        const double offdiag_ms = fast_ms(offdiag_start, FastClock::now());

        const auto a1mat = materialize_actual_a1(block1, actual_a1_offdiagonal);
        const auto strength1 = build_combined_strength_graph(
            graph1_tentative, block1, actual_a1_offdiagonal, strength_threshold);
        const auto transfer1 = build_candidate_transfer(
            strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
        const L1BlockSmoothedTransfer transfer1_nested{transfer1, apply1, block1, omega1, m1};
        const Apply apply2_nested = [&](const Vec& x) {
            return transfer1_nested.restrict_transpose(apply1(transfer1_nested.prolong(x)));
        };
        const auto local_a1 = [&](const LocalColumns& x) {
            return apply_materialized_a1(a1mat, block1, x);
        };

        const auto l2_basis_start = FastClock::now();
        const auto l2_basis = build_smoothed_supports_parallel(
            transfer1, strength1.graph, block1, omega1, m1, local_a1);
        const double l2_basis_ms = fast_ms(l2_basis_start, FastClock::now());

        const auto cached_start = FastClock::now();
        const auto applied_l2_basis = apply_supports_parallel(l2_basis, local_a1);
        const double cached_ms = fast_ms(cached_start, FastClock::now());

        const auto metric_start = FastClock::now();
        const auto block2 = metric_from_cached_applied(
            transfer1, block1, l2_basis, applied_l2_basis);
        const double block2_ms = fast_ms(metric_start, FastClock::now());

        const auto payload1_start = FastClock::now();
        const auto p1 = m5_p1_setup::assemble_dual_order_block6(transfer1, block1, l2_basis);
        const auto inverse1 = m5_l2_setup::inverse_blocks_6x6_fp32(block1);
        const auto inverse2 = m5_l2_setup::inverse_blocks_6x6_fp32(block2);
        const double p1_payload_ms = fast_ms(payload1_start, FastClock::now());

        const auto a2_start = FastClock::now();
        const auto a2 = dense_a2_from_cached_applied(
            transfer1, block1, l2_basis, applied_l2_basis);
        const double a2_ms = fast_ms(a2_start, FastClock::now());
        const Apply apply2_dense = [&](const Vec& x) {
            return m5_l2_setup::apply_dense_a2(a2, x);
        };
        const auto lambda2_start = FastClock::now();
        const double lambda2 = estimate_lambda_max_l1_block(apply2_dense, block2, 8U);
        const double omega2 = kSaDampingNumerator / lambda2;
        const double lambda2_ms = fast_ms(lambda2_start, FastClock::now());

        const auto transfer2 = build_candidate_transfer(
            transfer1.coarse_graph, transfer1.coarse_candidates,
            target_nodes, min_nodes, 1.0e-10);
        const auto p2_start = FastClock::now();
        const auto p2 = dense_smoothed_p2_from_a2(transfer2, block2, a2, omega2);
        const double p2_ms = fast_ms(p2_start, FastClock::now());
        const auto bottom_start = FastClock::now();
        const auto bottom = dense_bottom_from_a2_p2(a2, p2);
        const double bottom_ms = fast_ms(bottom_start, FastClock::now());

        const auto final_payload_start = FastClock::now();
        const auto bottom_inverse = m5_fast_setup::symmetric_inverse_col_major(bottom.factor);
        const auto a2_fp32 = m5_l2_setup::to_float(a2.fp64);
        const auto p2_fp32 = m5_l2_setup::to_float(p2.fp64);
        const double final_payload_ms = fast_ms(final_payload_start, FastClock::now());
        const double production_ms = fast_ms(production_start, FastClock::now());

        const auto oracle_start = FastClock::now();
        const auto a1_probe = deterministic_actual_a2_probe(block1.dofs(), 0.47);
        const double a1_error = rel_error(
            apply_materialized_a1_vector(a1mat, block1, a1_probe), apply1(a1_probe));
        const double block2_error = audit_l1_block_metric(block2, apply2_nested);
        const auto probe2 = deterministic_actual_a2_probe(a2.n, 0.63);
        const double a2_error = rel_error(apply2_dense(probe2), apply2_nested(probe2));
        const L1BlockSmoothedTransfer transfer2_nested{
            transfer2, apply2_nested, block2, omega2, m2};
        const auto probe3 = deterministic_actual_a2_probe(transfer2.coarse_dofs, 0.91);
        const double p2_error = rel_error(
            m5_l2_setup::apply_p2(p2, probe3), transfer2_nested.prolong(probe3));
        const Apply apply3_nested = [&](const Vec& x) {
            return transfer2_nested.restrict_transpose(
                apply2_nested(transfer2_nested.prolong(x)));
        };
        const double bottom_error = bottom_local_oracle_error(bottom, apply3_nested);
        const double oracle_ms = fast_ms(oracle_start, FastClock::now());
        const bool oracle_ok = a1_error <= 1.0e-10 && block2_error <= 1.0e-10 &&
                               a2_error <= 1.0e-10 && p2_error <= 1.0e-10 &&
                               bottom_error <= 1.0e-10;

        std::cout << "GFSS M5 temporary materialized-A1 exact setup\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "method=actual_A1_blocks_setup_only_then_cached_A1P1_dense_A2_driven_L3\n"
                  << "runtime_policy=materialized_A1_discarded_before_GPU_runtime\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << block1.dofs()
                  << " L2_dofs=" << block2.dofs()
                  << " L3_dofs=" << bottom.factor.n << '\n'
                  << std::scientific << std::setprecision(9)
                  << "A1_materialized_vs_nested_relative_error=" << a1_error
                  << " L2_block_vs_nested_relative_error=" << block2_error
                  << " A2_dense_vs_nested_relative_error=" << a2_error
                  << " P2_dense_vs_factorized_relative_error=" << p2_error
                  << " bottom_dense_vs_nested_relative_error=" << bottom_error
                  << " oracle_accept_1e-10=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "production_required_setup_ms=" << production_ms
                  << " validation_oracle_ms=" << oracle_ms << '\n'
                  << "actual_A1_offdiagonal_ms=" << offdiag_ms
                  << " temporary_A1_materialization_ms=" << a1mat.setup_ms
                  << " materialized_L2_basis_ms=" << l2_basis_ms
                  << " materialized_cached_A1P1_ms=" << cached_ms
                  << " L2_metric_ms=" << block2_ms
                  << " P1_plus_block_inverses_ms=" << p1_payload_ms
                  << " A2_dense_ms=" << a2_ms
                  << " lambda2_ms=" << lambda2_ms
                  << " P2_dense_ms=" << p2_ms
                  << " A3_dense_ms=" << bottom_ms
                  << " final_FP32_payload_ms=" << final_payload_ms << '\n'
                  << "temporary_A1_offdiag_pairs=" << actual_a1_offdiagonal.size()
                  << " temporary_A1_directed_blocks=" << a1mat.directed_blocks
                  << " temporary_A1_logical_bytes=" << a1mat.logical_bytes << '\n'
                  << "payload_P1_nnz=" << p1.forward_column_indices.size()
                  << " inverse1_values=" << inverse1.size()
                  << " inverse2_values=" << inverse2.size()
                  << " A2_fp32_values=" << a2_fp32.size()
                  << " P2_fp32_values=" << p2_fp32.size()
                  << " bottom_inverse_values=" << bottom_inverse.size() << '\n';
        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_hierarchy_materialized_a1_setup_bench "
                  << "[target_nodes=12 [min_nodes=4]]\n";
        return 1;
    }
}
