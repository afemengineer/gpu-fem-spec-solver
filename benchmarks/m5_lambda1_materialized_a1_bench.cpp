// M5 setup microbenchmark: estimate the L1 block-Jacobi spectral radius using
// the established nested A1=P0^T A0 P0 action versus the exact temporary
// materialized block-sparse A1 already constructed for hierarchy setup.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_parallel_actual_a1_setup.hpp"
#include "m5_materialized_a1_setup.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using BenchClock = std::chrono::steady_clock;

double ms_since(BenchClock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        BenchClock::now() - start).count();
}

double rel_scalar(double a, double b) {
    return std::abs(a - b) / std::max(std::abs(b), 1.0e-300);
}

double rel_vec(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("lambda1 oracle size mismatch");
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
            throw std::invalid_argument("invalid lambda1 benchmark aggregation options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t power_iterations = 8U;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};

        auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
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
        const auto parallel_a1 = m5_parallel_a1::assemble(
            mesh, material, fine_supports, element_supports);
        const auto temporary_a1 = m5_materialized_a1::build(
            block1, parallel_a1.blocks);
        const Apply apply1_materialized = [&](const Vec& x) {
            return m5_materialized_a1::apply_vector(temporary_a1, block1, x);
        };

        const auto probe = deterministic_actual_a2_probe(block1.dofs(), 0.47);
        const double a1_apply_error = rel_vec(
            apply1_materialized(probe), apply1_nested(probe));

        auto start = BenchClock::now();
        const double lambda1_nested = estimate_lambda_max_l1_block(
            apply1_nested, block1, power_iterations);
        const double nested_ms = ms_since(start);

        start = BenchClock::now();
        const double lambda1_materialized = estimate_lambda_max_l1_block(
            apply1_materialized, block1, power_iterations);
        const double materialized_ms = ms_since(start);

        const double lambda_rel = rel_scalar(lambda1_materialized, lambda1_nested);
        const double omega_nested = kSaDampingNumerator / lambda1_nested;
        const double omega_materialized = kSaDampingNumerator / lambda1_materialized;
        const double omega_rel = rel_scalar(omega_materialized, omega_nested);
        const bool oracle_ok = a1_apply_error <= 1.0e-10 &&
                               lambda_rel <= 1.0e-10 && omega_rel <= 1.0e-10;

        std::cout << "GFSS M5 lambda1 materialized-A1 setup decision\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "nested_operator=P0T_A0_P0\n"
                  << "candidate_operator=temporary_exact_materialized_A1\n"
                  << "power_iterations=" << power_iterations << '\n'
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << block1.dofs() << '\n'
                  << std::scientific << std::setprecision(12)
                  << "A1_materialized_vs_nested_relative_error=" << a1_apply_error << '\n'
                  << "lambda1_nested=" << lambda1_nested
                  << " lambda1_materialized=" << lambda1_materialized
                  << " lambda1_relative_difference=" << lambda_rel << '\n'
                  << "omega1_nested=" << omega_nested
                  << " omega1_materialized=" << omega_materialized
                  << " omega1_relative_difference=" << omega_rel << '\n'
                  << "oracle_accept_1e-10=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "nested_lambda1_ms=" << nested_ms
                  << " materialized_lambda1_ms=" << materialized_ms
                  << " speedup=" << (materialized_ms > 0.0 ? nested_ms / materialized_ms : 0.0) << '\n'
                  << "parallel_A1_setup_ms=" << parallel_a1.total_ms
                  << " temporary_A1_materialization_ms=" << temporary_a1.setup_ms
                  << " temporary_A1_logical_bytes=" << temporary_a1.logical_bytes << '\n';
        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_lambda1_materialized_a1_bench "
                  << "[target_nodes=12 [min_nodes=4]]\n";
        return 1;
    }
}
