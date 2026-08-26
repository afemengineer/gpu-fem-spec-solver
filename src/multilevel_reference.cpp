#include "gfss/multilevel_reference.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gfss {
namespace {

using Clock = std::chrono::steady_clock;

void validate_level(const ReferenceMultilevelLevel& level,
                    std::size_t index,
                    std::size_t level_count) {
    if (level.dofs == 0U) {
        throw std::invalid_argument("multilevel level has zero DOFs");
    }
    if (!level.apply) {
        throw std::invalid_argument("multilevel level is missing apply operator");
    }
    if (index + 1U == level_count) {
        if (!level.bottom_solve) {
            throw std::invalid_argument("multilevel bottom level is missing bottom solve");
        }
    } else {
        if (!level.smooth || !level.restrict_to_coarse || !level.prolong_from_coarse) {
            throw std::invalid_argument(
                "multilevel non-bottom level is missing smoother or transfer");
        }
    }
}

double relative_residual(const ReferenceMultilevelLevel& level,
                         const ReferenceVector& b,
                         const ReferenceVector& x) {
    const auto ax = level.apply(x);
    if (ax.size() != b.size()) {
        throw std::runtime_error("multilevel apply returned wrong vector size");
    }
    double rr = 0.0;
    double bb = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double r = b[i] - ax[i];
        rr += r * r;
        bb += b[i] * b[i];
    }
    return bb > 0.0 ? std::sqrt(rr / bb) : 0.0;
}

void vcycle(std::size_t level_index,
            const std::vector<ReferenceMultilevelLevel>& levels,
            const ReferenceVector& b,
            ReferenceVector& x,
            std::size_t pre_smooth_steps,
            std::size_t post_smooth_steps) {
    const auto& level = levels[level_index];
    if (b.size() != level.dofs || x.size() != level.dofs) {
        throw std::runtime_error("multilevel level vector size mismatch");
    }

    if (level_index + 1U == levels.size()) {
        x = level.bottom_solve(b);
        if (x.size() != level.dofs) {
            throw std::runtime_error("multilevel bottom solve returned wrong vector size");
        }
        return;
    }

    level.smooth(b, x, pre_smooth_steps);
    const auto ax = level.apply(x);
    if (ax.size() != level.dofs) {
        throw std::runtime_error("multilevel apply returned wrong vector size inside V-cycle");
    }

    ReferenceVector residual(level.dofs, 0.0);
    for (std::size_t i = 0; i < level.dofs; ++i) {
        residual[i] = b[i] - ax[i];
    }

    auto coarse_b = level.restrict_to_coarse(residual);
    const auto& coarse = levels[level_index + 1U];
    if (coarse_b.size() != coarse.dofs) {
        throw std::runtime_error("multilevel restriction returned wrong coarse size");
    }

    ReferenceVector coarse_e(coarse.dofs, 0.0);
    vcycle(level_index + 1U,
           levels,
           coarse_b,
           coarse_e,
           pre_smooth_steps,
           post_smooth_steps);

    const auto correction = level.prolong_from_coarse(coarse_e);
    if (correction.size() != level.dofs) {
        throw std::runtime_error("multilevel prolongation returned wrong fine size");
    }
    for (std::size_t i = 0; i < level.dofs; ++i) {
        x[i] += correction[i];
    }
    level.smooth(b, x, post_smooth_steps);
}

}  // namespace

ReferenceMultilevelResult solve_reference_multilevel_vcycle(
    const std::vector<ReferenceMultilevelLevel>& levels,
    const ReferenceVector& rhs,
    double relative_tolerance,
    std::size_t max_cycles,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps) {
    if (levels.empty()) {
        throw std::invalid_argument("multilevel hierarchy is empty");
    }
    if (levels.size() < 2U) {
        throw std::invalid_argument("multilevel reference requires at least two levels");
    }
    if (!(relative_tolerance > 0.0) || relative_tolerance >= 1.0) {
        throw std::invalid_argument("multilevel relative tolerance must be in (0,1)");
    }
    if (max_cycles == 0U) {
        throw std::invalid_argument("multilevel max_cycles must be positive");
    }
    if (rhs.size() != levels.front().dofs) {
        throw std::invalid_argument("multilevel RHS size does not match top level");
    }
    for (std::size_t i = 0; i < levels.size(); ++i) {
        validate_level(levels[i], i, levels.size());
    }

    ReferenceMultilevelResult result;
    result.x.assign(rhs.size(), 0.0);
    result.relative_residuals.push_back(
        relative_residual(levels.front(), rhs, result.x));
    if (result.relative_residuals.back() <= relative_tolerance) {
        result.converged = true;
        return result;
    }

    const auto solve_start = Clock::now();
    for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
        vcycle(0U,
               levels,
               rhs,
               result.x,
               pre_smooth_steps,
               post_smooth_steps);
        result.cycles = cycle + 1U;
        const double rel = relative_residual(levels.front(), rhs, result.x);
        if (!std::isfinite(rel)) {
            throw std::runtime_error("multilevel true residual became non-finite");
        }
        result.relative_residuals.push_back(rel);
        if (rel <= relative_tolerance) {
            result.converged = true;
            break;
        }
    }
    const auto solve_stop = Clock::now();
    result.solve_ms = std::chrono::duration<double, std::milli>(
        solve_stop - solve_start).count();
    return result;
}

}  // namespace gfss
