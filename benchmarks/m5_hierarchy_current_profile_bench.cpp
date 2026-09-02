// M5 current production hierarchy profile after exact temporary-A1 and
// parallel exact-A1 setup optimization. Validation oracles are disabled so this
// measures only production-required hierarchy construction.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_fast_hierarchy_bundle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t power_iterations = 8U;
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
        print_stage("lambda1_estimate", s.lambda1_ms, hierarchy.production_setup_ms);
        print_stage("P0_support_cache", s.p0_support_cache_ms, hierarchy.production_setup_ms);
        print_stage("element_support_index", s.element_support_index_ms, hierarchy.production_setup_ms);
        print_stage("actual_A1_offdiagonal", s.actual_a1_offdiagonal_ms, hierarchy.production_setup_ms);
        print_stage("temporary_A1_materialization", s.materialized_a1_ms, hierarchy.production_setup_ms);
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

        // Decision probe only: reconstruct the already-validated temporary A1 after
        // the production timing has finished, then run the identical block-Jacobi
        // power iteration on it. This extra work is deliberately excluded from
        // production_required_setup_ms.
        const auto fine_inverse = build_fine_inverse_diagonal(
            mesh, material, hierarchy.space0);
        double p0_support_probe_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(
            mesh, material, hierarchy.space0, fine_inverse,
            hierarchy.omega0, p0_support_probe_ms);
        const auto element_supports = build_element_support_index(mesh, fine_supports);
        const auto parallel_a1 = m5_parallel_a1::assemble(
            mesh, material, fine_supports, element_supports);
        const auto temporary_a1 = m5_materialized_a1::build(
            hierarchy.block1, parallel_a1.blocks);
        const Apply apply1_materialized = [&](const Vec& x) {
            return m5_materialized_a1::apply_vector(
                temporary_a1, hierarchy.block1, x);
        };

        const auto candidate_start = std::chrono::steady_clock::now();
        const double lambda1_materialized = estimate_lambda_max_l1_block(
            apply1_materialized, hierarchy.block1, power_iterations);
        const double materialized_lambda1_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - candidate_start).count();
        const double lambda1_rel = std::abs(lambda1_materialized - hierarchy.lambda1) /
            std::max(std::abs(hierarchy.lambda1), 1.0e-300);
        const double omega1_materialized = kSaDampingNumerator / lambda1_materialized;
        const double omega1_rel = std::abs(omega1_materialized - hierarchy.omega1) /
            std::max(std::abs(hierarchy.omega1), 1.0e-300);
        const bool lambda1_candidate_accept =
            lambda1_rel <= 1.0e-10 && omega1_rel <= 1.0e-10;

        std::cout << std::scientific << std::setprecision(12)
                  << "lambda1_current_nested=" << hierarchy.lambda1
                  << " lambda1_materialized_candidate=" << lambda1_materialized
                  << " lambda1_relative_difference=" << lambda1_rel << '\n'
                  << "omega1_current_nested=" << hierarchy.omega1
                  << " omega1_materialized_candidate=" << omega1_materialized
                  << " omega1_relative_difference=" << omega1_rel << '\n'
                  << "lambda1_materialized_candidate_accept_1e-10="
                  << (lambda1_candidate_accept ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda1_current_nested_ms=" << s.lambda1_ms
                  << " lambda1_materialized_candidate_ms=" << materialized_lambda1_ms
                  << " lambda1_candidate_speedup="
                  << (materialized_lambda1_ms > 0.0
                          ? s.lambda1_ms / materialized_lambda1_ms : 0.0) << '\n'
                  << "decision_probe_excluded_from_production_profile=true\n";

        return lambda1_candidate_accept ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_hierarchy_current_profile_bench "
                  << "[target_nodes=12 [min_nodes=4]]\n";
        return 1;
    }
}
