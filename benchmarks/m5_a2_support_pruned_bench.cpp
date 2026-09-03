// M5 scaling decision probe: retain the exact dense A2 representation but
// avoid evaluating aggregate block pairs whose localized supports cannot
// intersect. The established all-pairs symmetric assembly remains production.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_fast_hierarchy_bundle.hpp"

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

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

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto hierarchy = m5_fast_bundle::build(
            mesh, material, target_nodes, min_nodes,
            false,  // independent hierarchy validation not needed for this A/B probe
            true);  // run exact support-pruned A2 probe outside production timing

        const auto& probe = hierarchy.a2_support_probe;
        if (!probe.ran) throw std::runtime_error("support-pruned A2 probe did not run");
        const double speedup = probe.total_ms > 0.0
            ? hierarchy.stages.a2_ms / probe.total_ms : 0.0;
        const bool accept = probe.relative_error <= 1.0e-12;

        std::cout << "GFSS M5 exact support-pruned A2 decision probe\n"
                  << "problem=thin_plate mesh=" << nx << 'x' << ny << 'x' << nz
                  << " physical=1x1x0.125\n"
                  << "production=all_symmetric_L2_aggregate_pairs_dense_A2\n"
                  << "candidate=support_intersection_candidate_pairs_same_dense_A2\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << hierarchy.space0.coarse_dofs
                  << " L2_dofs=" << hierarchy.a2.n
                  << " L2_nodes=" << hierarchy.transfer1.aggregates.size()
                  << " L3_dofs=" << hierarchy.bottom.factor.n << '\n'
                  << std::scientific << std::setprecision(12)
                  << "candidate_A2_relative_error=" << probe.relative_error
                  << " oracle_accept_1e-12=" << (accept ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "all_block_pairs=" << probe.all_block_pairs
                  << " candidate_block_pairs=" << probe.candidate_block_pairs
                  << " candidate_fraction=" << probe.candidate_fraction << '\n'
                  << "production_A2_ms=" << hierarchy.stages.a2_ms
                  << " candidate_support_index_ms=" << probe.support_index_ms
                  << " candidate_assembly_ms=" << probe.assembly_ms
                  << " candidate_total_ms=" << probe.total_ms
                  << " speedup_vs_production_A2=" << speedup << '\n'
                  << "production_hierarchy_setup_ms=" << hierarchy.production_setup_ms
                  << " probe_excluded_from_production_timing=true\n";
        return accept ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_a2_support_pruned_bench "
                  << "[nx=64 [ny=64 [nz=8 [target_nodes=12 [min_nodes=4]]]]]\n";
        return 1;
    }
}
