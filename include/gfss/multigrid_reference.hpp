#pragma once

#include "gfss/hex8.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <vector>

namespace gfss {

struct GeometricVcycleLevelInfo {
    StructuredHexMesh mesh{};
    std::size_t dofs{0};
    double estimated_lambda_max{0.0};
};

struct GeometricVcycleReferenceResult {
    std::vector<double> x;
    std::vector<double> relative_residuals;
    std::vector<GeometricVcycleLevelInfo> levels;
    std::size_t cycles{0};
    bool converged{false};
    double setup_ms{0.0};
    double solve_ms{0.0};
    double estimated_three_vector_coarse_bytes_per_fine_dof{0.0};
    double estimated_six_vector_coarse_bytes_per_fine_dof{0.0};
};

// FP64/OpenMP numerical reference for the M5 geometric multigrid hypothesis.
// This is intentionally not a performance implementation. It builds a nested
// 2x hierarchy while every element count remains even and greater than one,
// uses a matrix-free clamped HEX8 operator on every level, Chebyshev-Jacobi
// pre/post smoothing, variational restriction R=P^T, trilinear prolongation,
// and an accurate CPU CG solve only on the small bottom level.
//
// The smoother estimates lambda_max of D^{-1/2} A D^{-1/2} independently on
// every non-bottom level by power iteration. A conservative fixed fraction of
// that measured spectral radius defines the high-frequency Chebyshev window;
// there is no mesh- or problem-specific relaxation parameter.
GeometricVcycleReferenceResult solve_geometric_vcycle_reference_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    double relative_tolerance = 1.0e-6,
    std::size_t max_cycles = 12,
    std::size_t pre_smooth_degree = 3,
    std::size_t post_smooth_degree = 3,
    std::size_t power_iterations = 8);

}  // namespace gfss
