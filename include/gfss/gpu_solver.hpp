#pragma once

#include "gfss/hex8.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <vector>

namespace gfss {

struct GpuPcgResult {
    std::vector<float> x;
    std::size_t iterations{0};
    std::size_t matvecs{0};
    // Kept for the archived restart baseline compiled inside gpu_pcg_audit.cu.
    std::size_t residual_replacements{0};
    std::size_t residual_audits{0};
    bool converged{false};
    double reported_relative_residual{0.0};
    double audited_relative_residual{0.0};
    double solve_ms{0.0};
    std::size_t explicit_device_bytes{0};
};

// Persistent-device PCG for the structured-Q1 elasticity problem.
// The x=0 face is zero-Dirichlet, matching the trusted CPU reference. The
// matrix-free matvec uses the GoldSparse structured stencil and Jacobi uses
// the exact diagonal of the same clamped operator. FP32 recursive residuals
// drive PCG, but convergence is accepted only after a non-mutating r=b-Ax
// audit of the current solution. Failed audits do not overwrite the recursive
// residual or restart the Krylov search direction. Setup/allocation/H2D/D2H
// are excluded from solve_ms; PCG kernels, cuBLAS reductions, residual audits,
// and their host synchronization are included.
GpuPcgResult solve_pcg_cuda_gold_sparse_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<float>& rhs,
    double relative_tolerance = 1.0e-5,
    std::size_t max_iterations = 2000,
    int block_y = 4);

}  // namespace gfss
