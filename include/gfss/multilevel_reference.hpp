#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace gfss {

using ReferenceVector = std::vector<double>;
using ReferenceApply = std::function<ReferenceVector(const ReferenceVector&)>;
using ReferenceSmooth = std::function<void(const ReferenceVector&,
                                           ReferenceVector&,
                                           std::size_t)>;
using ReferenceTransfer = std::function<ReferenceVector(const ReferenceVector&)>;
using ReferenceBottomSolve = std::function<ReferenceVector(const ReferenceVector&)>;

// Backend-neutral operations for one level of a reference multilevel solve.
// The V-cycle engine deliberately knows nothing about meshes, element types,
// geometry, or how the coarse space was constructed. A level can therefore be
// geometric, p-coarsened, or algebraically aggregated as long as it supplies
// these operations.
struct ReferenceMultilevelLevel {
    std::size_t dofs{0};
    std::string label;
    double diagnostic_lambda_max{0.0};

    ReferenceApply apply;
    ReferenceSmooth smooth;

    // Required on every non-bottom level. restrict_to_coarse maps a residual
    // in this level to the next level; prolong_from_coarse maps a correction
    // from the next level back to this level.
    ReferenceTransfer restrict_to_coarse;
    ReferenceTransfer prolong_from_coarse;

    // Required only on the bottom level.
    ReferenceBottomSolve bottom_solve;
};

struct ReferenceMultilevelResult {
    ReferenceVector x;
    std::vector<double> relative_residuals;
    std::size_t cycles{0};
    bool converged{false};
    double solve_ms{0.0};
};

// Generic recursive V-cycle reference. Numerical correctness is measured by
// the top-level operator's true Euclidean residual. The engine does not impose
// a particular smoother or coarse-space construction.
ReferenceMultilevelResult solve_reference_multilevel_vcycle(
    const std::vector<ReferenceMultilevelLevel>& levels,
    const ReferenceVector& rhs,
    double relative_tolerance = 1.0e-6,
    std::size_t max_cycles = 12,
    std::size_t pre_smooth_steps = 3,
    std::size_t post_smooth_steps = 3);

}  // namespace gfss
