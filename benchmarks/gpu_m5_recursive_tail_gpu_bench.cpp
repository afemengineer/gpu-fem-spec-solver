// M5 recursive coarse-tail GPU validation.  Builds the same exact sparse tail
// as m5_recursive_sparse_tail_bench, exports each non-bottom level to the
// memory-preferred FP32 operator representation, and compares one persistent
// GPU V-cycle against an FP64 CPU evaluation of the identical recursion.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_l2_dense_setup.hpp"
#include "m5_fast_hierarchy_setup.hpp"
#include "m5_materialized_a1_setup.hpp"
#include "m5_parallel_actual_a1_setup.hpp"
#include "m5_sparse_a2_setup.hpp"
#include "m5_recursive_sparse_tail.hpp"
#include "m5_recursive_tail_gpu_bridge.hpp"

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
        const int repeats = argc > 11 ? static_cast<int>(parse_size(argv[11], "repeats")) : 50;
        if (min_nodes > target_nodes) throw std::invalid_argument("min nodes exceeds target nodes");

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{
            static_cast<std::uint32_t>(nx), static_cast<std::uint32_t>(ny),
            static_cast<std::uint32_t>(nz), lx, ly, lz};
        const gfss::Material material{210.0e9, poisson};

        const auto setup_start = BenchClock::now();
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
        auto sparse_a2 = m5_sparse_a2::assemble_from_cached_applied(
            transfer1, block1, l2_basis, applied_l2_basis);

        auto tail = m5_recursive_tail::build(
            std::move(transfer1),
            std::move(block2),
            std::move(sparse_a2),
            2U,
            target_nodes,
            min_nodes,
            dense_bottom_threshold,
            12U);
        const double exact_setup_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - setup_start).count();

        const auto export_start = BenchClock::now();
        auto payloads = m5_recursive_tail_gpu_bridge::export_payloads(tail);
        auto bottom_inverse = m5_recursive_tail_gpu_bridge::bottom_inverse_fp32(tail);
        const double payload_export_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - export_start).count();

        std::vector<float> rhs_fp32(tail.levels.front().op.n, 0.0f);
        Vec rhs_fp64(rhs_fp32.size(), 0.0);
        for (std::size_t i = 0U; i < rhs_fp32.size(); ++i) {
            const double t = static_cast<double>(i + 1U);
            rhs_fp32[i] = static_cast<float>(
                std::sin(0.013 * t + 0.41) + 0.31 * std::cos(0.037 * t - 0.17));
            rhs_fp64[i] = static_cast<double>(rhs_fp32[i]);
        }

        const auto cpu_start = BenchClock::now();
        const auto cpu_x = m5_recursive_tail_gpu_bridge::cpu_vcycle(tail, 0U, rhs_fp64);
        const double cpu_vcycle_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - cpu_start).count();

        const auto gpu = gfss::benchmark_m5_recursive_tail_vcycle(
            payloads, bottom_inverse, rhs_fp32, repeats);
        const double gpu_vs_cpu =
            m5_recursive_tail_gpu_bridge::gpu_vs_cpu_relative_error(gpu.x, cpu_x);
        const bool runtime_oracle_accept = gpu_vs_cpu <= 1.0e-4;
        const bool exact_oracle_accept = tail.bottom_sparse_apply_relative_error <= 1.0e-12 &&
                                         tail.bottom_solve_relative_residual <= 1.0e-10;

        std::cout << "GFSS M5 recursive coarse-tail GPU validation\n"
                  << "problem=structured_elasticity mesh=" << nx << 'x' << ny << 'x' << nz
                  << " physical=" << lx << 'x' << ly << 'x' << lz
                  << " poisson=" << poisson << '\n'
                  << "policy=recursive_depth_plus_per_level_adaptive_runtime_representation"
                  << " dense_bottom_threshold=" << dense_bottom_threshold
                  << " target_nodes=" << target_nodes
                  << " min_nodes=" << min_nodes
                  << " repeats=" << repeats << '\n'
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " tail_top_dofs=" << tail.levels.front().op.n
                  << " bottom_level=" << tail.bottom_level
                  << " bottom_dofs=" << tail.bottom_dofs << '\n'
                  << std::scientific << std::setprecision(12)
                  << "bottom_sparse_vs_dense_relative_error="
                  << tail.bottom_sparse_apply_relative_error
                  << " bottom_solve_relative_residual="
                  << tail.bottom_solve_relative_residual
                  << " gpu_tail_vs_FP64_relative_error=" << gpu_vs_cpu
                  << " exact_oracle_accept=" << (exact_oracle_accept ? "true" : "false")
                  << " runtime_oracle_accept_1e-4="
                  << (runtime_oracle_accept ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0
                  << " lambda1=" << lambda1
                  << " exact_recursive_setup_ms=" << exact_setup_ms
                  << " gpu_payload_export_ms=" << payload_export_ms
                  << " cpu_FP64_tail_vcycle_ms=" << cpu_vcycle_ms
                  << " gpu_FP32_tail_median_ms=" << gpu.median_ms
                  << " gpu_FP32_tail_best_ms=" << gpu.best_ms
                  << " gpu_tail_device_bytes=" << gpu.device_bytes << '\n';

        for (std::size_t i = 0U; i < tail.levels.size(); ++i) {
            const auto& level = tail.levels[i];
            std::cout << "level=" << level.level
                      << " nodes=" << level.op.nodes
                      << " dofs=" << level.op.n
                      << " density=" << tail.telemetry[i].scalar_density
                      << " runtime_representation=" << gpu.runtime_representations[i];
            if (i + 1U < tail.levels.size()) {
                std::cout << " next_dofs=" << tail.levels[i + 1U].op.n
                          << " P_nnz=" << payloads[i].p_values.size()
                          << " PT_nnz=" << payloads[i].pt_values.size();
            }
            std::cout << '\n';
        }

        return exact_oracle_accept && runtime_oracle_accept ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_recursive_tail_gpu_bench "
                  << "[nx=64 [ny=64 [nz=8 [lx=1 [ly=1 [lz=.125 [poisson=.3 "
                  << "[dense_bottom=512 [target_nodes=12 [min_nodes=4 [repeats=50]]]]]]]]]]]\n";
        return 1;
    }
}
