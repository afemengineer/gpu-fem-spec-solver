// M5 setup microbenchmark: compare the validated thread-local unordered_map
// exact-A1 assembly against a preindexed compact block-array implementation.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_parallel_actual_a1_setup.hpp"
#include "m5_indexed_actual_a1_setup.hpp"
#include "m5_materialized_a1_setup.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

double rel_vec(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("indexed A1 oracle size mismatch");
    Vec d(a.size(), 0.0);
    for (std::size_t i = 0U; i < a.size(); ++i) d[i] = a[i] - b[i];
    return norm(d) / std::max(norm(b), 1.0e-300);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t target_nodes = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 12U;
        const std::size_t min_nodes = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 4U;
        if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid indexed A1 benchmark aggregation options");
        }

        constexpr std::size_t m0 = 1U;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};

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
        const Apply apply1_nested = [&](const Vec& x) {
            return transfer0.restrict_transpose(apply0(transfer0.prolong(x)));
        };

        const auto graph1_tentative = graph_from_variable_blocks(tentative_a1);
        const auto block1 = build_exact_l1_block_metric(
            mesh, material, space0, graph1_tentative, fine_inverse, omega0);

        double p0_support_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(
            mesh, material, space0, fine_inverse, omega0, p0_support_ms);
        const auto element_supports = build_element_support_index(mesh, fine_supports);

        const auto reference = m5_parallel_a1::assemble(
            mesh, material, fine_supports, element_supports);
        const auto candidate = m5_indexed_a1::assemble(
            mesh, material, fine_supports, element_supports);

        const double block_map_error = m5_parallel_a1::block_map_relative_error(
            candidate.blocks, reference.blocks);
        const auto reference_a1 = m5_materialized_a1::build(block1, reference.blocks);
        const auto candidate_a1 = m5_materialized_a1::build(block1, candidate.blocks);
        const auto probe = deterministic_actual_a2_probe(block1.dofs(), 0.47);
        const auto nested = apply1_nested(probe);
        const auto ref_apply = m5_materialized_a1::apply_vector(reference_a1, block1, probe);
        const auto candidate_apply = m5_materialized_a1::apply_vector(candidate_a1, block1, probe);
        const double reference_vs_nested = rel_vec(ref_apply, nested);
        const double candidate_vs_nested = rel_vec(candidate_apply, nested);
        const double candidate_vs_reference = rel_vec(candidate_apply, ref_apply);
        const bool oracle_ok = block_map_error <= 1.0e-10 &&
                               reference_vs_nested <= 1.0e-10 &&
                               candidate_vs_nested <= 1.0e-10 &&
                               candidate_vs_reference <= 1.0e-10 &&
                               candidate.unique_pairs == reference.blocks.size();

        std::cout << "GFSS M5 indexed exact actual-A1 assembly\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "reference=OpenMP_thread_local_unordered_maps_deterministic_reduction\n"
                  << "candidate=preindexed_pair_plan_compact_thread_blocks_indexed_reduction\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << block1.dofs()
                  << " threads=" << candidate.threads << '\n'
                  << std::scientific << std::setprecision(12)
                  << "indexed_block_map_relative_error=" << block_map_error
                  << " reference_A1_vs_nested_relative_error=" << reference_vs_nested
                  << " indexed_A1_vs_nested_relative_error=" << candidate_vs_nested
                  << " indexed_vs_reference_apply_relative_error=" << candidate_vs_reference
                  << " oracle_accept_1e-10=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "reference_local_accumulation_ms=" << reference.local_accumulation_ms
                  << " reference_reduction_ms=" << reference.deterministic_reduction_ms
                  << " reference_total_ms=" << reference.total_ms << '\n'
                  << "indexed_plan_ms=" << candidate.plan_ms
                  << " indexed_local_accumulation_ms=" << candidate.local_accumulation_ms
                  << " indexed_reduction_ms=" << candidate.indexed_reduction_ms
                  << " indexed_export_map_ms=" << candidate.export_map_ms
                  << " indexed_total_ms=" << candidate.total_ms
                  << " speedup_vs_reference="
                  << (candidate.total_ms > 0.0 ? reference.total_ms / candidate.total_ms : 0.0)
                  << '\n'
                  << "reference_unique_pairs=" << reference.blocks.size()
                  << " indexed_unique_pairs=" << candidate.unique_pairs
                  << " reference_summed_thread_entries=" << reference.summed_thread_entries
                  << " indexed_summed_thread_pairs=" << candidate.summed_thread_pairs
                  << " indexed_pair_contributions=" << candidate.pair_contributions << '\n'
                  << "indexed_local_block_bytes=" << candidate.local_block_bytes
                  << " indexed_index_bytes=" << candidate.index_bytes
                  << " reference_materialization_ms=" << reference_a1.setup_ms
                  << " indexed_materialization_ms=" << candidate_a1.setup_ms << '\n';
        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_a1_indexed_parallel_bench "
                  << "[target_nodes=12 [min_nodes=4]]\n";
        return 1;
    }
}
