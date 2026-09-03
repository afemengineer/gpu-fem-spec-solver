// M5 scaling decision probe: compare three exact A2 setup paths from the same
// cached L2 supports: all-pairs dense, support-pruned dense, and support-pruned
// direct sparse assembly with no dense candidate payload.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_l2_dense_setup.hpp"
#include "m5_fast_hierarchy_setup.hpp"
#include "m5_materialized_a1_setup.hpp"
#include "m5_parallel_actual_a1_setup.hpp"
#include "m5_symmetric_a2_setup.hpp"
#include "m5_sparse_a2_setup.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using BenchClock = std::chrono::steady_clock;

std::vector<double> deterministic_probe(std::size_t n, double phase) {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0U; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        x[i] = 1.0e-9 * (std::sin(0.013 * t + phase) +
                         0.29 * std::cos(0.037 * t - 0.41 * phase));
    }
    return x;
}

double relative_error(const std::vector<double>& a,
                      const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("A2 setup oracle size mismatch");
    double d2 = 0.0;
    double b2 = 0.0;
    for (std::size_t i = 0U; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        d2 += d * d;
        b2 += b[i] * b[i];
    }
    return std::sqrt(d2 / std::max(b2, 1.0e-300));
}

double dense_relative_error(const std::vector<double>& a,
                            const std::vector<double>& b) {
    return relative_error(a, b);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t nx = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 64U;
        const std::size_t ny = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 64U;
        const std::size_t nz = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 8U;
        const std::size_t target_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 12U;
        const std::size_t min_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 4U;
        if (nx < 2U || ny < 2U || nz < 1U || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid support-pruned A2 benchmark options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};

        const auto prerequisites_start = BenchClock::now();
        auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
        auto space0 = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
        const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
            mesh, material, space0);
        const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
        const Apply apply0 = [&](const Vec& x) {
            return apply_fine_clamped(mesh, material, x);
        };
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

        double p0_support_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(
            mesh, material, space0, fine_inverse, omega0, p0_support_ms);
        const auto element_supports = build_element_support_index(mesh, fine_supports);
        const auto parallel_a1 = m5_parallel_a1::assemble(
            mesh, material, fine_supports, element_supports);
        const auto temporary_a1 = m5_materialized_a1::build(
            block1, parallel_a1.blocks);
        const Apply apply1_materialized = [&](const Vec& x) {
            return m5_materialized_a1::apply_vector(temporary_a1, block1, x);
        };
        const double lambda1 = estimate_lambda_max_l1_block(
            apply1_materialized, block1, 8U);
        const double omega1 = kSaDampingNumerator / lambda1;
        const auto strength1 = build_combined_strength_graph(
            graph1_tentative, block1, parallel_a1.blocks, strength_threshold);
        const auto transfer1 = build_candidate_transfer(
            strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
        const auto local_a1 = [&](const LocalColumns& x) {
            return m5_materialized_a1::apply_columns(temporary_a1, block1, x);
        };
        const auto l2_basis = m5_fast_setup::build_smoothed_supports_parallel(
            transfer1, strength1.graph, block1, omega1, m1, local_a1);
        const auto applied_l2_basis = m5_fast_setup::apply_supports_parallel(
            l2_basis, local_a1);
        const double prerequisites_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - prerequisites_start).count();

        const auto dense = m5_symmetric_a2::assemble_from_cached_applied(
            transfer1, block1, l2_basis, applied_l2_basis);
        const auto pruned_dense = m5_symmetric_a2::assemble_from_cached_applied_support_pruned(
            transfer1, block1, l2_basis, applied_l2_basis);
        const auto sparse = m5_sparse_a2::assemble_from_cached_applied(
            transfer1, block1, l2_basis, applied_l2_basis);

        const double pruned_dense_error = dense_relative_error(
            pruned_dense.dense.fp64, dense.fp64);
        double sparse_error = 0.0;
        for (const double phase : {0.31, 0.67, 1.13}) {
            const auto x = deterministic_probe(dense.n, phase);
            sparse_error = std::max(
                sparse_error,
                relative_error(
                    m5_sparse_a2::apply_fp64(sparse, transfer1, x),
                    m5_l2_setup::apply_dense_a2(dense, x)));
        }
        const bool accept = pruned_dense_error <= 1.0e-12 && sparse_error <= 1.0e-12;

        const std::size_t dense_fp64_bytes = dense.fp64.size() * sizeof(double);
        const std::size_t dense_fp32_bytes = dense.n * dense.n * sizeof(float);
        const std::size_t sparse_fp64_block_bytes = sparse.fp64_block_logical_bytes();
        const std::size_t sparse_fp32_csr_bytes = sparse.fp32_csr_logical_bytes();
        const double candidate_fraction = sparse.all_block_pairs > 0U
            ? static_cast<double>(sparse.candidate_upper_block_pairs) /
              static_cast<double>(sparse.all_block_pairs)
            : 0.0;
        const double pruned_speedup = pruned_dense.total_ms > 0.0
            ? dense.assembly_ms / pruned_dense.total_ms : 0.0;
        const double sparse_speedup = sparse.total_ms > 0.0
            ? dense.assembly_ms / sparse.total_ms : 0.0;
        const double sparse_vs_pruned = sparse.total_ms > 0.0
            ? pruned_dense.total_ms / sparse.total_ms : 0.0;

        std::cout << "GFSS M5 exact A2 setup representation decision probe\n"
                  << "problem=thin_plate mesh=" << nx << 'x' << ny << 'x' << nz
                  << " physical=1x1x0.125\n"
                  << "paths=all_pairs_dense,support_pruned_dense,support_pruned_direct_sparse\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L2_dofs=" << dense.n
                  << " L2_nodes=" << transfer1.aggregates.size() << '\n'
                  << std::scientific << std::setprecision(12)
                  << "support_pruned_dense_relative_error=" << pruned_dense_error
                  << " direct_sparse_apply_relative_error=" << sparse_error
                  << " oracle_accept_1e-12=" << (accept ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "all_upper_block_pairs=" << sparse.all_block_pairs
                  << " candidate_upper_block_pairs=" << sparse.candidate_upper_block_pairs
                  << " candidate_fraction=" << candidate_fraction
                  << " directed_sparse_blocks=" << sparse.blocks.size()
                  << " scalar_CSR_nnz=" << sparse.values_fp32.size() << '\n'
                  << "prerequisites_to_cached_A1P1_ms=" << prerequisites_ms << '\n'
                  << "all_pairs_dense_A2_ms=" << dense.assembly_ms << '\n'
                  << "support_pruned_dense_index_ms=" << pruned_dense.support_index_ms
                  << " support_pruned_dense_assembly_ms=" << pruned_dense.assembly_ms
                  << " support_pruned_dense_total_ms=" << pruned_dense.total_ms
                  << " speedup_dense_vs_pruned_dense=" << pruned_speedup << '\n'
                  << "direct_sparse_support_index_ms=" << sparse.support_index_ms
                  << " direct_sparse_block_assembly_ms=" << sparse.block_assembly_ms
                  << " direct_sparse_CSR_export_ms=" << sparse.csr_export_ms
                  << " direct_sparse_total_ms=" << sparse.total_ms
                  << " speedup_dense_vs_direct_sparse=" << sparse_speedup
                  << " speedup_pruned_dense_vs_direct_sparse=" << sparse_vs_pruned << '\n'
                  << "dense_FP64_host_bytes=" << dense_fp64_bytes
                  << " direct_sparse_FP64_block_bytes=" << sparse_fp64_block_bytes
                  << " sparse_over_dense_FP64_host_ratio="
                  << static_cast<double>(sparse_fp64_block_bytes) /
                     static_cast<double>(dense_fp64_bytes) << '\n'
                  << "dense_FP32_runtime_bytes=" << dense_fp32_bytes
                  << " direct_sparse_FP32_CSR_bytes=" << sparse_fp32_csr_bytes
                  << " sparse_over_dense_FP32_runtime_ratio="
                  << static_cast<double>(sparse_fp32_csr_bytes) /
                     static_cast<double>(dense_fp32_bytes) << '\n';
        return accept ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_a2_support_pruned_bench "
                  << "[nx=64 [ny=64 [nz=8 [target_nodes=12 [min_nodes=4]]]]]\n";
        return 1;
    }
}
