// M5 stage 16: integrate the exact cached/parallel hierarchy setup with the
// validated persistent FP64-defect / FP32-MGPCG runtime. Validation oracles and
// benchmark warmup are measured separately from production-required end-to-end.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_fast_hierarchy_bundle.hpp"
#include "m5_persistent_vpcg_staging.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using EndClock = std::chrono::steady_clock;

std::vector<float> end_to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0U; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

std::vector<double> end_to_double(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0U; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

struct Step {
    double outer_before{0.0};
    double recursive_rel{0.0};
    double true_correction_rel{0.0};
    double gpu_solve_ms{0.0};
    double wall_ms{0.0};
    bool breakdown{false};
};
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t inner_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 5U;
        const std::size_t max_outer = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 8U;
        const int block_y = argc > 3 ? std::stoi(argv[3]) : 4;
        const std::size_t target_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 12U;
        const std::size_t min_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 4U;
        const double outer_tolerance = argc > 6 ? std::stod(argv[6]) : 1.0e-6;
        if (inner_iterations == 0U || inner_iterations > 64U ||
            max_outer == 0U || max_outer > 32U || block_y <= 0 ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes ||
            !(outer_tolerance > 0.0) || outer_tolerance >= 1.0) {
            throw std::invalid_argument("invalid M5 integrated end-to-end options");
        }

        constexpr std::size_t nu0 = 5U;
        constexpr std::size_t m0 = 1U;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto benchmark_start = EndClock::now();

        auto hierarchy = m5_fast_bundle::build(
            mesh, material, target_nodes, min_nodes, true);
        const auto rhs = make_rhs(mesh);
        const double rhs_norm = norm(rhs);
        if (!(rhs_norm > 0.0)) throw std::runtime_error("M5 integrated RHS is zero");

        const auto fine_setup_start = EndClock::now();
        gfss::GpuM5FineLevelContext fine_context(
            mesh, material, hierarchy.space0,
            hierarchy.omega0, hierarchy.lambda0, block_y);
        const double fine_gpu_context_ms = std::chrono::duration<double, std::milli>(
            EndClock::now() - fine_setup_start).count();

        const auto deep_setup_start = EndClock::now();
        gfss::M5PersistentPcgStaging persistent(
            fine_context, nu0, m0,
            hierarchy.inverse1, hierarchy.lambda1,
            hierarchy.transfer1.aggregates.size(),
            hierarchy.p1.forward_row_offsets,
            hierarchy.p1.forward_column_indices,
            hierarchy.p1.forward_values_row_major,
            hierarchy.p1.transpose_column_offsets,
            hierarchy.p1.transpose_row_indices,
            hierarchy.p1.transpose_values_q_r_entry,
            hierarchy.a2_fp32, hierarchy.inverse2, hierarchy.lambda2,
            hierarchy.p2_fp32, hierarchy.transfer2.coarse_dofs,
            hierarchy.bottom_inverse_fp32);
        const double deep_gpu_setup_ms = std::chrono::duration<double, std::milli>(
            EndClock::now() - deep_setup_start).count();

        const auto warmup_start = EndClock::now();
        persistent.warmup(end_to_float(rhs), inner_iterations);
        const double benchmark_warmup_ms = std::chrono::duration<double, std::milli>(
            EndClock::now() - warmup_start).count();

        const Apply apply0 = [&](const Vec& x) {
            return apply_fine_clamped(mesh, material, x);
        };
        Vec x(rhs.size(), 0.0);
        Vec residual(rhs.size(), 0.0);
        std::vector<double> outer_history;
        std::vector<Step> steps;
        outer_history.reserve(max_outer + 1U);
        steps.reserve(max_outer);
        double accurate_residual_ms = 0.0;
        double gpu_inner_solve_ms = 0.0;
        double correction_wall_ms = 0.0;
        std::size_t total_inner_iterations = 0U;
        std::size_t total_l0_operator_applies = 0U;
        bool converged = false;
        bool breakdown = false;

        const auto solve_start = EndClock::now();
        for (std::size_t outer = 0U; outer <= max_outer; ++outer) {
            const auto residual_start = EndClock::now();
            const auto ax = apply0(x);
            for (std::size_t i = 0U; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
            const double rel = norm(residual) / rhs_norm;
            accurate_residual_ms += std::chrono::duration<double, std::milli>(
                EndClock::now() - residual_start).count();
            outer_history.push_back(rel);

            if (!steps.empty() && steps.back().true_correction_rel == 0.0 &&
                steps.back().outer_before > 0.0) {
                steps.back().true_correction_rel = rel / steps.back().outer_before;
            }
            if (!std::isfinite(rel)) throw std::runtime_error("M5 integrated true residual non-finite");
            if (rel <= outer_tolerance) {
                converged = true;
                break;
            }
            if (outer == max_outer) break;

            const auto correction_start = EndClock::now();
            const auto correction = persistent.solve(end_to_float(residual), inner_iterations);
            const double one_wall_ms = std::chrono::duration<double, std::milli>(
                EndClock::now() - correction_start).count();

            Step step;
            step.outer_before = rel;
            step.recursive_rel = correction.recursive_relative_residual;
            step.gpu_solve_ms = correction.solve_ms;
            step.wall_ms = one_wall_ms;
            step.breakdown = correction.breakdown;
            if (correction.breakdown) {
                breakdown = true;
                steps.push_back(step);
                break;
            }

            const auto delta = end_to_double(correction.solution_aos);
            for (std::size_t i = 0U; i < x.size(); ++i) x[i] += delta[i];
            steps.push_back(step);
            gpu_inner_solve_ms += correction.solve_ms;
            correction_wall_ms += one_wall_ms;
            total_inner_iterations += correction.iterations;
            total_l0_operator_applies += correction.total_l0_operator_applies;
        }
        const double solve_wall_ms = std::chrono::duration<double, std::milli>(
            EndClock::now() - solve_start).count();
        const double observed_benchmark_wall_ms = std::chrono::duration<double, std::milli>(
            EndClock::now() - benchmark_start).count();

        const double final_true_residual = outer_history.empty() ? 1.0 : outer_history.back();
        const double runtime_gpu_setup_ms = fine_gpu_context_ms + deep_gpu_setup_ms;
        const double production_end_to_end_required_ms =
            hierarchy.production_setup_ms + runtime_gpu_setup_ms + solve_wall_ms;
        const double excluded_validation_and_warmup_ms =
            hierarchy.validation_oracle_ms + benchmark_warmup_ms;

        std::cout << "GFSS M5 exact fast-setup + persistent solver integration\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "rhs=physical_uniform_z_xmax\n"
                  << "hierarchy_setup=parallel_L2_cached_A1P1_dense_A2_driven_L3\n"
                  << "runtime=FP64_defect_FP32_MGPCG_5x1x1_inverse_L3\n"
                  << "production_timing_excludes=validation_oracles_and_benchmark_warmup\n"
                  << "inner_iterations=" << inner_iterations
                  << " max_outer=" << max_outer << '\n'
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << hierarchy.space0.coarse_dofs
                  << " L2_dofs=" << hierarchy.block2.dofs()
                  << " L3_dofs=" << hierarchy.bottom.factor.n << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << hierarchy.oracle.l1_block
                  << " L2_block_vs_nested_relative_error=" << hierarchy.oracle.l2_block
                  << " A2_dense_vs_nested_relative_error=" << hierarchy.oracle.a2_dense
                  << " P2_dense_vs_factorized_relative_error=" << hierarchy.oracle.p2_dense
                  << " bottom_dense_vs_nested_relative_error=" << hierarchy.oracle.bottom
                  << " bottom_inverse_identity_relative_error=" << hierarchy.oracle.bottom_inverse_identity
                  << " hierarchy_oracle_accept=" << (hierarchy.oracle.accept ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "cpu_production_hierarchy_setup_ms=" << hierarchy.production_setup_ms
                  << " validation_oracle_ms=" << hierarchy.validation_oracle_ms
                  << " fine_gpu_context_setup_ms=" << fine_gpu_context_ms
                  << " deep_gpu_persistent_setup_ms=" << deep_gpu_setup_ms
                  << " benchmark_warmup_ms=" << benchmark_warmup_ms
                  << " persistent_device_bytes=" << persistent.device_bytes_total() << '\n'
                  << "setup_fast_L2_basis_ms=" << hierarchy.stages.l2_basis_ms
                  << " setup_cached_A1P1_ms=" << hierarchy.stages.cached_a1p1_ms
                  << " setup_L2_metric_ms=" << hierarchy.stages.l2_metric_ms
                  << " setup_P1_payload_ms=" << hierarchy.stages.p1_payload_ms
                  << " setup_A2_ms=" << hierarchy.stages.a2_ms
                  << " setup_lambda2_ms=" << hierarchy.stages.lambda2_ms
                  << " setup_P2_ms=" << hierarchy.stages.p2_ms
                  << " setup_A3_ms=" << hierarchy.stages.bottom_ms
                  << " setup_final_payload_ms=" << hierarchy.stages.final_payload_ms << '\n';

        for (std::size_t i = 0U; i < outer_history.size(); ++i) {
            std::cout << std::scientific << std::setprecision(9)
                      << "outer_true_residual[" << i << "]=" << outer_history[i];
            if (i > 0U) std::cout << " outer_q=" << outer_history[i] / outer_history[i - 1U];
            std::cout << '\n';
        }
        for (std::size_t i = 0U; i < steps.size(); ++i) {
            const auto& s = steps[i];
            std::cout << std::scientific << std::setprecision(9)
                      << "correction[" << i << "] recursive_rel=" << s.recursive_rel
                      << " true_correction_rel=" << s.true_correction_rel
                      << " breakdown=" << (s.breakdown ? "true" : "false")
                      << std::fixed << std::setprecision(6)
                      << " gpu_solve_ms=" << s.gpu_solve_ms
                      << " wall_ms=" << s.wall_ms << '\n';
        }

        std::cout << std::scientific << std::setprecision(9)
                  << "converged=" << (converged ? "true" : "false")
                  << " breakdown=" << (breakdown ? "true" : "false")
                  << " final_true_relative_residual=" << final_true_residual << '\n'
                  << std::fixed << std::setprecision(6)
                  << "outer_corrections=" << steps.size()
                  << " total_inner_iterations=" << total_inner_iterations
                  << " total_L0_operator_applies=" << total_l0_operator_applies << '\n'
                  << "gpu_inner_solve_ms=" << gpu_inner_solve_ms
                  << " accurate_FP64_residual_ms=" << accurate_residual_ms
                  << " correction_wall_ms=" << correction_wall_ms
                  << " persistent_solve_wall_ms=" << solve_wall_ms << '\n'
                  << "runtime_gpu_setup_ms=" << runtime_gpu_setup_ms
                  << " production_end_to_end_required_ms=" << production_end_to_end_required_ms
                  << " excluded_validation_and_warmup_ms=" << excluded_validation_and_warmup_ms
                  << " observed_benchmark_wall_ms=" << observed_benchmark_wall_ms << '\n';

        return hierarchy.oracle.accept && converged && !breakdown ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_fastsetup_persistent_solver_bench "
                  << "[inner_iterations=5 [max_outer=8 [block_y=4 [target_nodes=12 [min_nodes=4 [outer_tol=1e-6]]]]]]\n";
        return 1;
    }
}
