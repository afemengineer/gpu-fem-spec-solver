// M5 recursive coarse-depth probe.  Build the validated L0->L2 hierarchy, then
// continue with exact support-pruned sparse Galerkin levels until the current
// coarse operator is small enough for a dense direct bottom solve.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_l2_dense_setup.hpp"
#include "m5_fast_hierarchy_setup.hpp"
#include "m5_materialized_a1_setup.hpp"
#include "m5_parallel_actual_a1_setup.hpp"
#include "m5_sparse_a2_setup.hpp"
#include "m5_recursive_sparse_tail.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using BenchClock = std::chrono::steady_clock;

std::size_t parse_size(const char* text, const char* name) {
    const auto value = std::stoull(text);
    if (value == 0ULL) throw std::invalid_argument(std::string(name) + " must be positive");
    return static_cast<std::size_t>(value);
}

double parse_positive(const char* text, const char* name) {
    const double value = std::stod(text);
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + " must be positive finite");
    }
    return value;
}

double parse_poisson(const char* text) {
    const double value = std::stod(text);
    if (!(value > -1.0) || !(value < 0.5) || !std::isfinite(value)) {
        throw std::invalid_argument("poisson must be in (-1,0.5)");
    }
    return value;
}

double relative_error(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("recursive-tail oracle size mismatch");
    Vec d(a.size(), 0.0);
    for (std::size_t i = 0U; i < a.size(); ++i) d[i] = a[i] - b[i];
    return norm(d) / std::max(norm(b), 1.0e-300);
}

Vec deterministic_probe(std::size_t n, double phase) {
    Vec v(n, 0.0);
    for (std::size_t i = 0U; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = std::sin(0.017 * t + phase) + 0.23 * std::cos(0.039 * t - 0.31 * phase);
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t nx = argc > 1 ? parse_size(argv[1], "nx") : 64U;
        const std::size_t ny = argc > 2 ? parse_size(argv[2], "ny") : 64U;
        const std::size_t nz = argc > 3 ? parse_size(argv[3], "nz") : 8U;
        const double lx = argc > 4 ? parse_positive(argv[4], "lx") : 1.0;
        const double ly = argc > 5 ? parse_positive(argv[5], "ly") : 1.0;
        const double lz = argc > 6 ? parse_positive(argv[6], "lz") : 0.125;
        const double poisson = argc > 7 ? parse_poisson(argv[7]) : 0.30;
        const std::size_t dense_bottom_threshold = argc > 8
            ? parse_size(argv[8], "dense bottom threshold") : 512U;
        const std::size_t target_nodes = argc > 9 ? parse_size(argv[9], "target nodes") : 12U;
        const std::size_t min_nodes = argc > 10 ? parse_size(argv[10], "min nodes") : 4U;
        if (min_nodes > target_nodes) throw std::invalid_argument("min nodes exceeds target nodes");

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{
            static_cast<std::uint32_t>(nx), static_cast<std::uint32_t>(ny),
            static_cast<std::uint32_t>(nz), lx, ly, lz};
        const gfss::Material material{210.0e9, poisson};

        const auto total_start = BenchClock::now();
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
        auto block1 = build_exact_l1_block_metric(
            mesh, material, space0, graph1_tentative, fine_inverse, omega0);

        double p0_support_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(
            mesh, material, space0, fine_inverse, omega0, p0_support_ms);
        const auto element_supports = build_element_support_index(mesh, fine_supports);
        const auto parallel_a1 = m5_parallel_a1::assemble(
            mesh, material, fine_supports, element_supports);
        const auto temporary_a1 = m5_materialized_a1::build(block1, parallel_a1.blocks);
        const Apply apply1_materialized = [&](const Vec& x) {
            return m5_materialized_a1::apply_vector(temporary_a1, block1, x);
        };
        const double lambda1 = estimate_lambda_max_l1_block(
            apply1_materialized, block1, 8U);
        const double omega1 = kSaDampingNumerator / lambda1;
        const auto strength1 = build_combined_strength_graph(
            graph1_tentative, block1, parallel_a1.blocks, strength_threshold);
        auto transfer1 = build_candidate_transfer(
            strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
        const auto local_a1 = [&](const LocalColumns& x) {
            return m5_materialized_a1::apply_columns(temporary_a1, block1, x);
        };
        const auto l2_basis = m5_fast_setup::build_smoothed_supports_parallel(
            transfer1, strength1.graph, block1, omega1, m1, local_a1);
        const auto applied_l2_basis = m5_fast_setup::apply_supports_parallel(
            l2_basis, local_a1);
        auto block2 = m5_fast_setup::metric_from_cached_applied(
            transfer1, block1, l2_basis, applied_l2_basis);
        const double prerequisites_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - prerequisites_start).count();

        auto sparse_a2 = m5_sparse_a2::assemble_from_cached_applied(
            transfer1, block1, l2_basis, applied_l2_basis);
        const double a2_sparse_ms = sparse_a2.total_ms;

        const L1BlockSmoothedTransfer transfer1_nested{
            transfer1, apply1, block1, omega1, m1};
        const Apply apply2_nested = [&](const Vec& x) {
            return transfer1_nested.restrict_transpose(
                apply1(transfer1_nested.prolong(x)));
        };
        double a2_oracle = 0.0;
        for (const double phase : {0.31, 0.73, 1.11}) {
            const auto x = deterministic_probe(sparse_a2.n, phase);
            a2_oracle = std::max(
                a2_oracle,
                relative_error(
                    m5_sparse_a2::apply_fp64(sparse_a2, transfer1, x),
                    apply2_nested(x)));
        }

        auto tail = m5_recursive_tail::build(
            std::move(transfer1),
            std::move(block2),
            std::move(sparse_a2),
            2U,
            target_nodes,
            min_nodes,
            dense_bottom_threshold,
            12U);

        const double total_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - total_start).count();
        const bool oracle_accept = a2_oracle <= 1.0e-10 &&
            tail.bottom_sparse_apply_relative_error <= 1.0e-12 &&
            tail.bottom_solve_relative_residual <= 1.0e-10;

        std::cout << "GFSS M5 recursive sparse coarse-tail probe\n"
                  << "problem=structured_elasticity mesh=" << nx << 'x' << ny << 'x' << nz
                  << " physical=" << lx << 'x' << ly << 'x' << lz
                  << " poisson=" << poisson << '\n'
                  << "policy=exact_support_pruned_sparse_until_dense_bottom"
                  << " dense_bottom_threshold=" << dense_bottom_threshold
                  << " target_nodes=" << target_nodes
                  << " min_nodes=" << min_nodes << '\n'
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs << '\n'
                  << std::scientific << std::setprecision(12)
                  << "A2_vs_nested_relative_error=" << a2_oracle
                  << " bottom_sparse_vs_dense_relative_error="
                  << tail.bottom_sparse_apply_relative_error
                  << " bottom_solve_relative_residual="
                  << tail.bottom_solve_relative_residual
                  << " oracle_accept=" << (oracle_accept ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0
                  << " lambda1=" << lambda1 << '\n'
                  << "prerequisites_to_L2_metric_ms=" << prerequisites_ms
                  << " direct_sparse_A2_ms=" << a2_sparse_ms
                  << " recursive_tail_ms=" << tail.total_ms
                  << " total_probe_ms=" << total_ms << '\n';

        for (const auto& level : tail.telemetry) {
            std::cout << "level=" << level.level
                      << " nodes=" << level.nodes
                      << " dofs=" << level.dofs
                      << " scalar_nnz=" << level.scalar_nnz
                      << " directed_blocks=" << level.directed_blocks
                      << " scalar_density=" << level.scalar_density
                      << " dense_fp32_bytes=" << level.dense_fp32_bytes
                      << " sparse_fp32_bytes=" << level.sparse_fp32_bytes
                      << " memory_preferred_runtime_representation="
                      << level.memory_preferred_runtime_representation << '\n'
                      << "level=" << level.level
                      << " lambda=" << level.lambda
                      << " omega=" << level.omega
                      << " lambda_ms=" << level.lambda_ms
                      << " transfer_ms=" << level.transfer_ms
                      << " smoothed_support_ms=" << level.smoothed_support_ms
                      << " applied_support_ms=" << level.applied_support_ms
                      << " metric_ms=" << level.metric_ms
                      << " galerkin_ms=" << level.galerkin_ms
                      << " transfer_support_blocks=" << level.transfer_support_blocks
                      << " next_nodes=" << level.next_nodes
                      << " next_dofs=" << level.next_dofs << '\n';
        }

        std::cout << "recursive_levels=" << tail.levels.size()
                  << " bottom_level=" << tail.bottom_level
                  << " bottom_dofs=" << tail.bottom_dofs
                  << " bottom_materialize_factor_ms=" << tail.bottom_materialize_factor_ms
                  << " fixed_depth_would_bottom_at_level=3";
        if (!tail.telemetry.empty() && tail.telemetry.front().next_dofs > 0U) {
            std::cout << " fixed_depth_L3_dofs=" << tail.telemetry.front().next_dofs;
        }
        std::cout << '\n';

        return oracle_accept ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_recursive_sparse_tail_bench "
                  << "[nx=64 [ny=64 [nz=8 [lx=1 [ly=1 [lz=.125 [poisson=.3 "
                  << "[dense_bottom=512 [target_nodes=12 [min_nodes=4]]]]]]]]]]\n";
        return 1;
    }
}
