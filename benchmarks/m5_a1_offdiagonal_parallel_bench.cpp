// M5 stage 17: benchmark thread-local OpenMP accumulation for the exact
// actual-A1 off-diagonal blocks. The established serial routine remains the
// numerical oracle; the resulting materialized A1 is also checked against the
// independent nested Galerkin operator.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_materialized_a1_setup.hpp"
#include "m5_parallel_actual_a1_setup.hpp"
#include "m5_sorted_reduce_actual_a1_setup.hpp"

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

inline double rel_error(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("parallel A1 oracle size mismatch");
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
            throw std::invalid_argument("invalid parallel A1 benchmark aggregation options");
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
        const Apply apply1 = [&](const Vec& x) {
            return transfer0.restrict_transpose(apply0(transfer0.prolong(x)));
        };
        const auto graph1_tentative = graph_from_variable_blocks(tentative_a1);
        const auto block1 = build_exact_l1_block_metric(
            mesh, material, space0, graph1_tentative, fine_inverse, omega0);

        double p0_support_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(
            mesh, material, space0, fine_inverse, omega0, p0_support_ms);
        const auto element_supports = build_element_support_index(mesh, fine_supports);

        const auto serial_start = BenchClock::now();
        const auto serial_blocks = accumulate_combined_actual_a1_offdiagonal_blocks(
            mesh, material, fine_supports, element_supports);
        const double serial_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - serial_start).count();

        const auto parallel = m5_parallel_a1::assemble(
            mesh, material, fine_supports, element_supports);
        const auto sorted_reduce = m5_sorted_reduce_a1::assemble(
            mesh, material, fine_supports, element_supports);

        const double parallel_block_error = m5_parallel_a1::block_map_relative_error(
            parallel.blocks, serial_blocks);
        const double sorted_block_error = m5_parallel_a1::block_map_relative_error(
            sorted_reduce.blocks, serial_blocks);
        const double sorted_vs_parallel_block_error = m5_parallel_a1::block_map_relative_error(
            sorted_reduce.blocks, parallel.blocks);

        const auto serial_a1 = m5_materialized_a1::build(block1, serial_blocks);
        const auto parallel_a1 = m5_materialized_a1::build(block1, parallel.blocks);
        const auto sorted_a1 = m5_materialized_a1::build(block1, sorted_reduce.blocks);
        const auto probe = deterministic_actual_a2_probe(block1.dofs(), 0.47);
        const auto nested_y = apply1(probe);
        const double serial_nested_error = rel_error(
            m5_materialized_a1::apply_vector(serial_a1, block1, probe), nested_y);
        const double parallel_nested_error = rel_error(
            m5_materialized_a1::apply_vector(parallel_a1, block1, probe), nested_y);
        const double sorted_nested_error = rel_error(
            m5_materialized_a1::apply_vector(sorted_a1, block1, probe), nested_y);
        const double sorted_vs_parallel_apply_error = rel_error(
            m5_materialized_a1::apply_vector(sorted_a1, block1, probe),
            m5_materialized_a1::apply_vector(parallel_a1, block1, probe));

        const bool keys_match = parallel.blocks.size() == serial_blocks.size() &&
                                sorted_reduce.blocks.size() == serial_blocks.size();
        const bool oracle_ok = keys_match &&
                               parallel_block_error <= 1.0e-12 &&
                               sorted_block_error <= 1.0e-12 &&
                               sorted_vs_parallel_block_error <= 1.0e-12 &&
                               parallel_nested_error <= 1.0e-10 &&
                               sorted_nested_error <= 1.0e-10 &&
                               sorted_vs_parallel_apply_error <= 1.0e-12;
        const double speedup = parallel.total_ms > 0.0
            ? parallel.total_ms / sorted_reduce.total_ms : 0.0;

        std::cout << "GFSS M5 parallel exact actual-A1 reduction comparison\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "reference=thread_local_maps_sorted_keys_global_hash_reduction\n"
                  << "candidate=thread_local_maps_sorted_entry_stream_kway_merge_single_export\n"
#ifdef _OPENMP
                  << "openmp_enabled=true\n"
#else
                  << "openmp_enabled=false\n"
#endif
                  << "threads=" << parallel.threads
                  << " fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << block1.dofs() << '\n'
                  << std::scientific << std::setprecision(12)
                  << "parallel_block_map_relative_error=" << parallel_block_error
                  << " sorted_block_map_relative_error=" << sorted_block_error
                  << " sorted_vs_parallel_block_relative_error=" << sorted_vs_parallel_block_error << '\n'
                  << "serial_A1_vs_nested_relative_error=" << serial_nested_error
                  << " parallel_A1_vs_nested_relative_error=" << parallel_nested_error
                  << " sorted_A1_vs_nested_relative_error=" << sorted_nested_error
                  << " sorted_vs_parallel_apply_relative_error=" << sorted_vs_parallel_apply_error
                  << " oracle_accept=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "serial_offdiagonal_ms=" << serial_ms << '\n'
                  << "reference_local_accumulation_ms=" << parallel.local_accumulation_ms
                  << " reference_reduction_ms=" << parallel.deterministic_reduction_ms
                  << " reference_total_ms=" << parallel.total_ms << '\n'
                  << "sorted_local_accumulation_ms=" << sorted_reduce.local_accumulation_ms
                  << " sorted_stream_sort_ms=" << sorted_reduce.stream_sort_ms
                  << " sorted_merge_ms=" << sorted_reduce.merge_ms
                  << " sorted_export_map_ms=" << sorted_reduce.export_map_ms
                  << " sorted_total_ms=" << sorted_reduce.total_ms
                  << " speedup_vs_reference=" << speedup << '\n'
                  << "serial_unique_pairs=" << serial_blocks.size()
                  << " reference_unique_pairs=" << parallel.blocks.size()
                  << " sorted_unique_pairs=" << sorted_reduce.blocks.size()
                  << " reference_summed_thread_entries=" << parallel.summed_thread_entries
                  << " sorted_summed_thread_entries=" << sorted_reduce.summed_thread_entries << '\n'
                  << "serial_materialization_ms=" << serial_a1.setup_ms
                  << " reference_materialization_ms=" << parallel_a1.setup_ms
                  << " sorted_materialization_ms=" << sorted_a1.setup_ms << '\n';

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_a1_offdiagonal_parallel_bench "
                  << "[target_nodes=12 [min_nodes=4]]\n";
        return 1;
    }
}
