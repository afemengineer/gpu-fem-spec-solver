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

struct InnerCostModel {
    bool valid{false};
    double intercept_ms{0.0};
    double slope_ms_per_log{0.0};
};

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

void update_cost_model(const GpuPcgResult& correction,
                       InnerCostModel& model) {
    double intercept = 0.0;
    double slope = 0.0;
    bool estimate_valid = false;

    if (correction.audit_samples.size() >= 2U) {
        const auto& first = correction.audit_samples.front();
        const auto& last = correction.audit_samples.back();
        if (first.audited_relative_residual > 0.0 &&
            first.audited_relative_residual < 1.0 &&
            last.audited_relative_residual > 0.0 &&
            last.audited_relative_residual <
                first.audited_relative_residual) {
            const double x0 =
                std::log(1.0 / first.audited_relative_residual);
            const double x1 =
                std::log(1.0 / last.audited_relative_residual);
            const double dx = x1 - x0;
            if (dx > 1.0e-9) {
                slope = std::max(
                    0.0,
                    (last.elapsed_ms - first.elapsed_ms) / dx);
                intercept = std::max(0.0, first.elapsed_ms - slope * x0);
                estimate_valid = std::isfinite(slope) &&
                                 std::isfinite(intercept);
            }
        }
    }

    if (!estimate_valid &&
        correction.audited_relative_residual > 0.0 &&
        correction.audited_relative_residual < 1.0 &&
        correction.solve_ms > 0.0) {
        const double reduction_log =
            std::log(1.0 / correction.audited_relative_residual);
        if (reduction_log > 1.0e-9) {
            slope = correction.solve_ms / reduction_log;
            intercept = 0.0;
            estimate_valid = std::isfinite(slope);
        }
    }

    if (!estimate_valid) {
        return;
    }

    if (!model.valid) {
        model.valid = true;
        model.intercept_ms = intercept;
        model.slope_ms_per_log = slope;
    } else {
        constexpr double weight = 0.5;
        model.intercept_ms =
            (1.0 - weight) * model.intercept_ms + weight * intercept;
        model.slope_ms_per_log =
            (1.0 - weight) * model.slope_ms_per_log + weight * slope;
    }
}

double predicted_inner_ms(const InnerCostModel& model,
                          double forcing,
                          double fallback_ms) {
    if (!model.valid || !(forcing > 0.0) || forcing >= 1.0) {
        return std::max(fallback_ms, 0.0);
    }
    const double predicted =
        model.intercept_ms +
        model.slope_ms_per_log * std::log(1.0 / forcing);
    return std::max(predicted, 0.0);
}

double initial_forcing(double current_residual,
                       double outer_tolerance) {
    constexpr double eta_min = 1.0e-4;
    constexpr double eta_max = 2.0e-1;
    constexpr double calibration_corrections = 3.0;
    const double remaining =
        clamp_value(outer_tolerance / current_residual, 1.0e-16, 0.999999);
    const double eta =
        std::pow(remaining, 1.0 / calibration_corrections);
    return clamp_value(eta, eta_min, eta_max);
}

double choose_forcing(double current_residual,
                      double outer_tolerance,
                      double contraction_gain,
                      double accurate_residual_cost_ms,
                      const InnerCostModel& cost_model,
                      double fallback_inner_ms,
                      double capability_floor,
                      double demonstrated_inner_residual) {
    constexpr double eta_min = 1.0e-4;
    constexpr double eta_max = 2.0e-1;
    constexpr int max_planned_corrections = 5;

    const double remaining =
        clamp_value(outer_tolerance / current_residual, 1.0e-16, 0.999999);
    const double gain = clamp_value(contraction_gain, 0.1, 10.0);

    // Do not extrapolate more than one decade tighter than the last forcing
    // actually demonstrated. A measured stagnation floor can tighten this
    // trust region further in the loose direction.
    double trust_floor = eta_min;
    if (demonstrated_inner_residual > 0.0 &&
        demonstrated_inner_residual < 1.0) {
        trust_floor = std::max(
            trust_floor,
            demonstrated_inner_residual * 0.1);
    }
    if (capability_floor > 0.0) {
        trust_floor = std::max(trust_floor, capability_floor);
    }
    trust_floor = std::min(trust_floor, eta_max);

    double best_eta = clamp_value(
        std::pow(remaining, 1.0 / 3.0) / gain,
        trust_floor,
        eta_max);
    double best_total_ms = std::numeric_limits<double>::infinity();

    for (int planned = 1; planned <= max_planned_corrections; ++planned) {
        double eta =
            std::pow(remaining, 1.0 / static_cast<double>(planned)) / gain;
        eta = clamp_value(eta, trust_floor, eta_max);

        const double predicted_q =
            clamp_value(gain * eta, 1.0e-12, 0.95);
        const double denominator = std::log(predicted_q);
        if (!(denominator < 0.0)) {
            continue;
        }

        const double raw_count = std::log(remaining) / denominator;
        const std::size_t corrections = static_cast<std::size_t>(
            std::max(1.0, std::ceil(raw_count)));
        const double inner_ms = predicted_inner_ms(
            cost_model, eta, fallback_inner_ms);
        const double predicted_total =
            static_cast<double>(corrections) *
            (inner_ms + std::max(accurate_residual_cost_ms, 0.0));

        if (predicted_total < best_total_ms) {
            best_total_ms = predicted_total;
            best_eta = eta;
        }
    }

    return best_eta;
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

    InnerCostModel cost_model;
    double contraction_gain = 1.0;
    bool have_gain = false;
    double capability_floor = 0.0;
    double demonstrated_inner_residual = 0.0;
    double fallback_inner_ms = 0.0;

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
                "adaptive refinement accurate residual became non-finite");
        }

        // The newly measured accurate residual closes the previous correction
        // step. Use it to learn how inner audited accuracy maps to true outer
        // contraction on this particular problem.
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
        const double forcing = result.adaptive_steps.empty()
            ? initial_forcing(relative_residual, outer_relative_tolerance)
            : choose_forcing(
                relative_residual,
                outer_relative_tolerance,
                contraction_gain,
                residual_cost_average,
                cost_model,
                fallback_inner_ms,
                capability_floor,
                demonstrated_inner_residual);

        const auto correction_wall_start = Clock::now();
        const auto correction = correction_context.solve(
            residual_fp32,
            forcing,
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

        update_cost_model(correction, cost_model);
        fallback_inner_ms = correction.solve_ms;
        demonstrated_inner_residual =
            correction.audited_relative_residual;

        if ((!correction.converged || correction.stagnated) &&
            correction.best_audited_relative_residual > 0.0 &&
            correction.best_audited_relative_residual < 1.0) {
            capability_floor = std::max(
                capability_floor,
                std::min(
                    2.0e-1,
                    correction.best_audited_relative_residual * 1.25));
        }

        AdaptiveForcingStep step;
        step.outer_index = outer;
        step.inner_iterations = correction.iterations;
        step.inner_matvecs = correction.matvecs;
        step.inner_audits = correction.residual_audits;
        step.inner_converged = correction.converged;
        step.inner_stagnated = correction.stagnated;
        step.outer_residual_before = relative_residual;
        step.requested_inner_tolerance = forcing;
        step.achieved_inner_residual =
            correction.audited_relative_residual;
        step.best_inner_residual =
            correction.best_audited_relative_residual;
        step.estimated_contraction_gain = contraction_gain;
        step.inner_solve_ms = correction.solve_ms;
        result.adaptive_steps.push_back(step);

        // A correction that cannot reduce its own audited residual below 0.9
        // is not useful enough to apply blindly. Return a non-converged outer
        // result rather than turning a preconditioner failure into divergence.
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
