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
        result.total_ms = 0.0;
        return result;
    }

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

        for (std::size_t i = 0; i < residual.size(); ++i) {
            const double value = residual[i];
            if (value > static_cast<double>(std::numeric_limits<float>::max()) ||
                value < -static_cast<double>(std::numeric_limits<float>::max())) {
                throw std::runtime_error(
                    "mixed refinement residual cannot be represented in FP32");
            }
            residual_fp32[i] = static_cast<float>(value);
        }

        const auto correction_wall_start = Clock::now();
        const auto correction = solve_pcg_cuda_gold_sparse_x0(
            mesh,
            material,
            residual_fp32,
            inner_relative_tolerance,
            max_inner_iterations,
            block_y);
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

}  // namespace gfss
