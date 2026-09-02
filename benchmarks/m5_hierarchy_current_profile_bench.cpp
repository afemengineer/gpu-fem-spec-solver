// M5 current production hierarchy profile after exact temporary-A1 and
// parallel exact-A1 setup optimization. Validation oracles are disabled so this
// measures only production-required hierarchy construction.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_fast_hierarchy_bundle.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void print_stage(const char* name, double ms, double total_ms) {
    const double pct = total_ms > 0.0 ? 100.0 * ms / total_ms : 0.0;
    std::cout << "setup_stage=" << name
              << " ms=" << std::fixed << std::setprecision(6) << ms
              << " pct_of_required=" << std::setprecision(3) << pct << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t target_nodes = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 12U;
        const std::size_t min_nodes = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 4U;
        if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid current hierarchy profile options");
        }

        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto hierarchy = m5_fast_bundle::build(
            mesh, material, target_nodes, min_nodes, false);
        const auto& s = hierarchy.stages;
        const double stage_sum = s.sum_ms();
        const double unattributed = hierarchy.production_setup_ms - stage_sum;

        std::cout << "GFSS M5 current production hierarchy profile\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "hierarchy=parallel_exact_A1_temporary_materialized_A1_dense_A2_driven_L3\n"
                  << "lambda1_operator=temporary_exact_materialized_A1\n"
                  << "validation_oracles=false\n"
#ifdef _OPENMP
                  << "openmp_enabled=true\n"
#else
                  << "openmp_enabled=false\n"
#endif
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << hierarchy.space0.coarse_dofs
                  << " L2_dofs=" << hierarchy.block2.dofs()
                  << " L3_dofs=" << hierarchy.bottom.factor.n << '\n'
                  << std::fixed << std::setprecision(6)
                  << "production_required_setup_ms=" << hierarchy.production_setup_ms
                  << " profiled_stage_sum_ms=" << stage_sum
                  << " unattributed_ms=" << unattributed << '\n'
                  << "actual_A1_threads=" << s.actual_a1_threads
                  << " actual_A1_local_accumulation_ms=" << s.actual_a1_local_accumulation_ms
                  << " actual_A1_reduction_ms=" << s.actual_a1_reduction_ms
                  << " actual_A1_summed_thread_entries=" << s.actual_a1_summed_thread_entries << '\n'
                  << "temporary_A1_logical_bytes=" << hierarchy.temporary_a1_logical_bytes << '\n';

        print_stage("graph0_aggregation", s.graph0_aggregation_ms, hierarchy.production_setup_ms);
        print_stage("tentative_A1", s.tentative_a1_ms, hierarchy.production_setup_ms);
        print_stage("fine_inverse", s.fine_inverse_ms, hierarchy.production_setup_ms);
        print_stage("lambda0_estimate", s.lambda0_ms, hierarchy.production_setup_ms);
        print_stage("L1_graph_candidates", s.l1_graph_candidates_ms, hierarchy.production_setup_ms);
        print_stage("L1_block_metric", s.l1_block_metric_ms, hierarchy.production_setup_ms);
        print_stage("P0_support_cache", s.p0_support_cache_ms, hierarchy.production_setup_ms);
        print_stage("element_support_index", s.element_support_index_ms, hierarchy.production_setup_ms);
        print_stage("actual_A1_offdiagonal", s.actual_a1_offdiagonal_ms, hierarchy.production_setup_ms);
        print_stage("temporary_A1_materialization", s.materialized_a1_ms, hierarchy.production_setup_ms);
        print_stage("lambda1_estimate_materialized_A1", s.lambda1_ms, hierarchy.production_setup_ms);
        print_stage("strength1_graph", s.strength1_ms, hierarchy.production_setup_ms);
        print_stage("transfer1_aggregation", s.transfer1_ms, hierarchy.production_setup_ms);
        print_stage("L2_basis", s.l2_basis_ms, hierarchy.production_setup_ms);
        print_stage("cached_A1P1", s.cached_a1p1_ms, hierarchy.production_setup_ms);
        print_stage("L2_block_metric", s.l2_metric_ms, hierarchy.production_setup_ms);
        print_stage("P1_plus_block_inverses", s.p1_payload_ms, hierarchy.production_setup_ms);
        print_stage("A2_dense", s.a2_ms, hierarchy.production_setup_ms);
        print_stage("lambda2_estimate", s.lambda2_ms, hierarchy.production_setup_ms);
        print_stage("transfer2_aggregation", s.transfer2_ms, hierarchy.production_setup_ms);
        print_stage("P2_dense", s.p2_ms, hierarchy.production_setup_ms);
        print_stage("A3_dense", s.bottom_ms, hierarchy.production_setup_ms);
        print_stage("final_FP32_payload", s.final_payload_ms, hierarchy.production_setup_ms);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_hierarchy_current_profile_bench "
                  << "[target_nodes=12 [min_nodes=4]]\n";
        return 1;
    }
}
