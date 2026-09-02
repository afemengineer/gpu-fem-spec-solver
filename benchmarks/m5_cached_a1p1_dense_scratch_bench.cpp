// M5 setup microbenchmark: remove repeated output hash-table probes from the
// cached A1*P1 construction by using one reusable dense 6x6 scratch block per
// L1 node inside each OpenMP worker. Production remains on the established
// LocalColumns hash accumulator until this candidate is validated.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_fast_hierarchy_setup.hpp"
#include "m5_materialized_a1_setup.hpp"
#include "m5_parallel_actual_a1_setup.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
using BenchClock = std::chrono::steady_clock;

double elapsed_ms(BenchClock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        BenchClock::now() - start).count();
}

double local_columns_set_relative_error(
    const std::vector<LocalColumns>& candidate,
    const std::vector<LocalColumns>& reference) {
    if (candidate.size() != reference.size()) {
        throw std::invalid_argument("A1P1 support count mismatch");
    }
    double diff2 = 0.0;
    double ref2 = 0.0;
    for (std::size_t s = 0U; s < reference.size(); ++s) {
        if (candidate[s].cols != reference[s].cols) {
            throw std::invalid_argument("A1P1 column count mismatch");
        }
        std::unordered_set<std::uint32_t> rows;
        rows.reserve(candidate[s].values.size() + reference[s].values.size());
        for (const auto& item : candidate[s].values) rows.insert(item.first);
        for (const auto& item : reference[s].values) rows.insert(item.first);
        for (const auto row : rows) {
            const auto cit = candidate[s].values.find(row);
            const auto rit = reference[s].values.find(row);
            for (std::size_t k = 0U; k < kCandidates * kCandidates; ++k) {
                const double got = cit == candidate[s].values.end() ? 0.0 : cit->second[k];
                const double ref = rit == reference[s].values.end() ? 0.0 : rit->second[k];
                const double d = got - ref;
                diff2 += d * d;
                ref2 += ref * ref;
            }
        }
    }
    return ref2 > 0.0 ? std::sqrt(diff2 / ref2) : std::sqrt(diff2);
}

double dense_relative_error(const std::vector<double>& a,
                            const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("dense A2 size mismatch");
    double diff2 = 0.0;
    double ref2 = 0.0;
    for (std::size_t i = 0U; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        diff2 += d * d;
        ref2 += b[i] * b[i];
    }
    return ref2 > 0.0 ? std::sqrt(diff2 / ref2) : std::sqrt(diff2);
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t target_nodes = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 12U;
        const std::size_t min_nodes = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 4U;
        if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid cached A1P1 benchmark options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr double strength_threshold = 0.05;
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

        auto start = BenchClock::now();
        const auto reference_applied = m5_fast_setup::apply_supports_parallel(
            l2_basis, local_a1);
        const double reference_ms = elapsed_ms(start);

        std::size_t scratch_bytes_per_worker = 0U;
        start = BenchClock::now();
        const auto scratch_applied =
            m5_materialized_a1::apply_supports_parallel_dense_scratch(
                temporary_a1, block1, l2_basis, &scratch_bytes_per_worker);
        const double scratch_ms = elapsed_ms(start);

        const double applied_error = local_columns_set_relative_error(
            scratch_applied, reference_applied);

        const auto reference_metric = m5_fast_setup::metric_from_cached_applied(
            transfer1, block1, l2_basis, reference_applied);
        const auto scratch_metric = m5_fast_setup::metric_from_cached_applied(
            transfer1, block1, l2_basis, scratch_applied);
        const double pivot_rel = std::abs(
            scratch_metric.min_cholesky_pivot - reference_metric.min_cholesky_pivot) /
            std::max(std::abs(reference_metric.min_cholesky_pivot), 1.0e-300);

        const auto reference_a2 = m5_fast_setup::dense_a2_from_cached_applied_symmetric(
            transfer1, block1, l2_basis, reference_applied);
        const auto scratch_a2 = m5_fast_setup::dense_a2_from_cached_applied_symmetric(
            transfer1, block1, l2_basis, scratch_applied);
        const double a2_error = dense_relative_error(scratch_a2.fp64, reference_a2.fp64);

        const bool oracle_ok = applied_error <= 1.0e-12 &&
                               pivot_rel <= 1.0e-12 &&
                               a2_error <= 1.0e-12;
        const double speedup = scratch_ms > 0.0 ? reference_ms / scratch_ms : 0.0;

        std::cout << "GFSS M5 cached A1P1 dense-scratch apply\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "reference=parallel_LocalColumns_output_unordered_map\n"
                  << "candidate=parallel_dense_node_scratch_touched_row_export\n"
#ifdef _OPENMP
                  << "openmp_enabled=true\n"
#else
                  << "openmp_enabled=false\n"
#endif
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << block1.dofs()
                  << " L1_nodes=" << block1.nodes()
                  << " L2_dofs=" << transfer1.coarse_dofs
                  << " L2_nodes=" << transfer1.aggregates.size() << '\n'
                  << std::scientific << std::setprecision(12)
                  << "cached_A1P1_relative_error=" << applied_error
                  << " L2_min_pivot_relative_difference=" << pivot_rel
                  << " A2_relative_error=" << a2_error
                  << " oracle_accept_1e-12=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "reference_cached_A1P1_ms=" << reference_ms
                  << " dense_scratch_cached_A1P1_ms=" << scratch_ms
                  << " speedup_vs_reference=" << speedup << '\n'
                  << "scratch_bytes_per_worker=" << scratch_bytes_per_worker
                  << " scratch_bytes_all_workers_estimate="
                  << scratch_bytes_per_worker * static_cast<std::size_t>(
#ifdef _OPENMP
                         omp_get_max_threads()
#else
                         1
#endif
                     ) << '\n'
                  << "reference_A2_ms=" << reference_a2.assembly_ms
                  << " scratch_A2_ms=" << scratch_a2.assembly_ms << '\n';

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_cached_a1p1_dense_scratch_bench "
                  << "[target_nodes=12 [min_nodes=4]]\n";
        return 1;
    }
}
