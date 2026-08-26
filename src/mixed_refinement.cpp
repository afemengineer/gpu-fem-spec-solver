#include "gfss/mixed_refinement.hpp"

#include "gfss/cpu_elasticity.hpp"
#include "gfss/gpu_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gfss {
namespace {

using Clock = std::chrono::steady_clock;

double squared_norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double value : v) {
        sum += value * value;
    }
    return sum;
}

double clamp_value(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

void convert_residual_to_fp32(const std::vector<double>& residual,
                              std::vector<float>& residual_fp32) {
    for (std::size_t i = 0; i < residual.size(); ++i) {
        const double value = residual[i];
        if (value > static_cast<double>(std::numeric_limits<float>::max()) ||
            value < -static_cast<double>(std::numeric_limits<float>::max())) {
            throw std::runtime_error(
                "mixed refinement residual cannot be represented in FP32");
        }
        residual_fp32[i] = static_cast<float>(value);
    }
}

}  // namespace

MixedRefinementResult solve_mixed_refinement_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    double outer_relative_tolerance,
    double inner_relative_tolerance,
    std::size_t max_outer_iterations,
    std::size_t max_inner_iterations,
    int block_y) {
    if (rhs.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument(
            "mixed refinement RHS size does not match mesh DOF count");
    }
    if (!(outer_relative_tolerance > 0.0) ||
        !(inner_relative_tolerance > 0.0)) {
        throw std::invalid_argument(
            "mixed refinement tolerances must be positive");
    }
    if (max_outer_iterations == 0U || max_inner_iterations == 0U) {
        throw std::invalid_argument(
            "mixed refinement iteration limits must be positive");
    }

    const double bnorm2 = squared_norm(rhs);
    if (!std::isfinite(bnorm2)) {
        throw std::runtime_error("mixed refinement RHS norm is non-finite");
    }

    MixedRefinementResult result;
    result.x.assign(rhs.size(), 0.0);

    const auto total_start = Clock::now();

    if (bnorm2 == 0.0) {
        result.converged = true;
        result.initial_relative_residual = 0.0;
        result.final_relative_residual = 0.0;
        result.outer_relative_residuals.push_back(0.0);
        result.total_ms = 0.0;
        return result;
    }

    const auto context_start = Clock::now();
    GpuPcgContext correction_context(mesh, material, block_y);
    const auto context_stop = Clock::now();
    result.gpu_context_setup_ms =
        std::chrono::duration<double, std::milli>(
            context_stop - context_start).count();

    const double bnorm = std::sqrt(bnorm2);
    std::vector<double> residual(rhs.size(), 0.0);
    std::vector<float> residual_fp32(rhs.size(), 0.0f);

    for (std::size_t outer = 0; outer <= max_outer_iterations; ++outer) {
        const auto residual_start = Clock::now();
        const auto ax = apply_clamped_x0_matrix_free(
            mesh, material, result.x);

        double rr = 0.0;
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            const double value = rhs[i] - ax[i];
            residual[i] = value;
            rr += value * value;
        }
        const double relative_residual = std::sqrt(rr) / bnorm;
        const auto residual_stop = Clock::now();
        result.accurate_residual_ms +=
            std::chrono::duration<double, std::milli>(
                residual_stop - residual_start).count();

        result.outer_relative_residuals.push_back(relative_residual);
        if (outer == 0U) {
            result.initial_relative_residual = relative_residual;
        }
        result.final_relative_residual = relative_residual;

        if (!std::isfinite(relative_residual)) {
            throw std::runtime_error(
                "mixed refinement accurate residual became non-finite");
        }
        if (relative_residual <= outer_relative_tolerance) {
            result.converged = true;
            break;
        }
        if (outer == max_outer_iterations) {
            break;
        }

        convert_residual_to_fp32(residual, residual_fp32);

        const auto correction_wall_start = Clock::now();
        const auto correction = correction_context.solve(
            residual_fp32,
            inner_relative_tolerance,
            max_inner_iterations);
        const auto correction_wall_stop = Clock::now();

        result.gpu_correction_wall_ms +=
            std::chrono::duration<double, std::milli>(
                correction_wall_stop - correction_wall_start).count();
        result.gpu_correction_solve_ms += correction.solve_ms;
        ++result.inner_solves;
        result.total_inner_iterations += correction.iterations;
        result.total_inner_matvecs += correction.matvecs;

        if (!correction.converged) {
            throw std::runtime_error(
                "mixed refinement inner FP32 correction solve did not reach its audited tolerance");
        }
        if (correction.x.size() != result.x.size()) {
            throw std::runtime_error(
                "mixed refinement correction vector size mismatch");
        }

        for (std::size_t i = 0; i < result.x.size(); ++i) {
            result.x[i] += static_cast<double>(correction.x[i]);
        }
        ++result.outer_iterations;
    }

    const auto total_stop = Clock::now();
    result.total_ms =
        std::chrono::duration<double, std::milli>(
            total_stop - total_start).count();
    return result;
}

MixedRefinementResult solve_mixed_refinement_adaptive_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    double outer_relative_tolerance,
    std::size_t max_outer_iterations,
    std::size_t max_inner_iterations,
    int block_y) {
    if (rhs.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument(
            "adaptive refinement RHS size does not match mesh DOF count");
    }
    if (!(outer_relative_tolerance > 0.0) ||
        outer_relative_tolerance >= 1.0) {
        throw std::invalid_argument(
            "adaptive refinement outer tolerance must be in (0, 1)");
    }
    if (max_outer_iterations == 0U || max_inner_iterations == 0U) {
        throw std::invalid_argument(
            "adaptive refinement iteration limits must be positive");
    }

    const double bnorm2 = squared_norm(rhs);
    if (!std::isfinite(bnorm2)) {
        throw std::runtime_error("adaptive refinement RHS norm is non-finite");
    }

    MixedRefinementResult result;
    result.adaptive_controller = true;
    result.x.assign(rhs.size(), 0.0);

    const auto total_start = Clock::now();

    if (bnorm2 == 0.0) {
        result.converged = true;
        result.initial_relative_residual = 0.0;
        result.final_relative_residual = 0.0;
        result.outer_relative_residuals.push_back(0.0);
        result.total_ms = 0.0;
        return result;
    }

    const auto context_start = Clock::now();
    GpuPcgContext correction_context(mesh, material, block_y);
    const auto context_stop = Clock::now();
    result.gpu_context_setup_ms =
        std::chrono::duration<double, std::milli>(
            context_stop - context_start).count();

    const double bnorm = std::sqrt(bnorm2);
    std::vector<double> residual(rhs.size(), 0.0);
    std::vector<float> residual_fp32(rhs.size(), 0.0f);

    double contraction_gain = 1.0;
    bool have_gain = false;

    for (std::size_t outer = 0; outer <= max_outer_iterations; ++outer) {
        const auto residual_start = Clock::now();
        const auto ax = apply_clamped_x0_matrix_free(
            mesh, material, result.x);

        double rr = 0.0;
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            const double value = rhs[i] - ax[i];
            residual[i] = value;
            rr += value * value;
        }
        const double relative_residual = std::sqrt(rr) / bnorm;
        const auto residual_stop = Clock::now();
        const double residual_elapsed_ms =
            std::chrono::duration<double, std::milli>(
                residual_stop - residual_start).count();
        result.accurate_residual_ms += residual_elapsed_ms;

        result.outer_relative_residuals.push_back(relative_residual);
        if (outer == 0U) {
            result.initial_relative_residual = relative_residual;
        }
        result.final_relative_residual = relative_residual;

        if (!std::isfinite(relative_residual)) {
            throw std::runtime_error(
                "adaptive refinement accurate residual became non-finite");
        }

        // Close the previous correction with the newly measured true residual.
        // This is the only learned outer model: how achieved inner residual
        // maps to actual outer contraction for this problem.
        if (!result.adaptive_steps.empty()) {
            auto& previous = result.adaptive_steps.back();
            if (previous.outer_contraction == 0.0 &&
                previous.outer_residual_before > 0.0) {
                const double q =
                    relative_residual / previous.outer_residual_before;
                previous.outer_contraction = q;
                if (previous.achieved_inner_residual > 0.0 &&
                    previous.achieved_inner_residual < 1.0 &&
                    q > 0.0 && std::isfinite(q)) {
                    const double observed_gain = clamp_value(
                        q / previous.achieved_inner_residual,
                        0.1,
                        10.0);
                    if (!have_gain) {
                        contraction_gain = observed_gain;
                        have_gain = true;
                    } else {
                        contraction_gain =
                            0.5 * contraction_gain +
                            0.5 * observed_gain;
                    }
                }
            }
        }

        if (relative_residual <= outer_relative_tolerance) {
            result.converged = true;
            break;
        }
        if (outer == max_outer_iterations) {
            break;
        }

        convert_residual_to_fp32(residual, residual_fp32);

        const double residual_cost_average =
            result.accurate_residual_ms /
            static_cast<double>(result.outer_relative_residuals.size());
        const double outer_remaining_ratio =
            outer_relative_tolerance / relative_residual;
        const double gain_used = contraction_gain;

        const auto correction_wall_start = Clock::now();
        const auto correction = correction_context.solve_economic(
            residual_fp32,
            outer_remaining_ratio,
            residual_cost_average,
            gain_used,
            max_inner_iterations);
        const auto correction_wall_stop = Clock::now();

        result.gpu_correction_wall_ms +=
            std::chrono::duration<double, std::milli>(
                correction_wall_stop - correction_wall_start).count();
        result.gpu_correction_solve_ms += correction.solve_ms;
        ++result.inner_solves;
        result.total_inner_iterations += correction.iterations;
        result.total_inner_matvecs += correction.matvecs;

        if (correction.x.size() != result.x.size()) {
            throw std::runtime_error(
                "adaptive refinement correction vector size mismatch");
        }

        AdaptiveForcingStep step;
        step.outer_index = outer;
        step.inner_iterations = correction.iterations;
        step.inner_matvecs = correction.matvecs;
        step.inner_audits = correction.residual_audits;
        step.predicted_outer_corrections =
            correction.predicted_outer_corrections;
        step.inner_converged = correction.converged;
        step.inner_stagnated = correction.stagnated;
        step.economic_stop = correction.economic_stop;
        step.outer_residual_before = relative_residual;
        step.requested_inner_tolerance = 0.0;
        step.achieved_inner_residual =
            correction.audited_relative_residual;
        step.best_inner_residual =
            correction.best_audited_relative_residual;
        step.estimated_contraction_gain = gain_used;
        step.predicted_total_ms = correction.predicted_total_ms;
        step.inner_solve_ms = correction.solve_ms;
        result.adaptive_steps.push_back(step);

        // Even a stagnated inner solve can be a useful defect correction. Only
        // reject corrections whose independently audited residual says they
        // remove less than ten percent of the correction equation.
        if (!(correction.audited_relative_residual > 0.0) ||
            correction.audited_relative_residual >= 0.9) {
            break;
        }

        for (std::size_t i = 0; i < result.x.size(); ++i) {
            result.x[i] += static_cast<double>(correction.x[i]);
        }
        ++result.outer_iterations;
    }

    const auto total_stop = Clock::now();
    result.total_ms =
        std::chrono::duration<double, std::milli>(
            total_stop - total_start).count();
    return result;
}

}  // namespace gfss
