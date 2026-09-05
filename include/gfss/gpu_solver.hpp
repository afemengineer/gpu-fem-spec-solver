#pragma once

#include "gfss/hex8.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace gfss {

struct GpuPcgAuditSample {
    std::size_t iteration{0};
    double recursive_relative_residual{0.0};
    double audited_relative_residual{0.0};
    double elapsed_ms{0.0};
};

struct GpuPcgResult {
    std::vector<float> x;
    std::vector<GpuPcgAuditSample> audit_samples;
    std::size_t iterations{0};
    std::size_t matvecs{0};
    // Kept for the archived restart baseline compiled inside gpu_pcg_audit.cu.
    std::size_t residual_replacements{0};
    std::size_t residual_audits{0};
    std::size_t best_audited_iteration{0};
    std::size_t predicted_outer_corrections{0};
    bool converged{false};
    bool stagnated{false};
    bool economic_stop{false};
    double requested_relative_residual{0.0};
    double reported_relative_residual{0.0};
    double audited_relative_residual{0.0};
    double best_audited_relative_residual{0.0};
    double predicted_total_ms{0.0};
    double solve_ms{0.0};
    std::size_t explicit_device_bytes{0};
};

// Standalone audited Jacobi-PCG from a zero initial guess.
GpuPcgResult solve_pcg_cuda_gold_sparse_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<float>& rhs,
    double relative_tolerance = 1.0e-5,
    std::size_t max_iterations = 2000,
    int block_y = 4);

// M5 experimental path: identical fine-grid GoldSparse Jacobi-PCG, but starts
// from an explicit interleaved FP32 initial guess. This lets a geometric coarse
// correction seed the fine Krylov solve without changing its audited stopping
// criterion. Allocation, stencil upload, H2D and D2H remain outside solve_ms.
GpuPcgResult solve_pcg_cuda_gold_sparse_seeded_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<float>& rhs,
    const std::vector<float>& initial_guess,
    double relative_tolerance = 1.0e-5,
    std::size_t max_iterations = 2000,
    int block_y = 4);

// Reusable PCG device state for repeated correction solves on one mesh and
// material. The constructor uploads stencil/Jacobi data, allocates the six
// solver vectors, creates cuBLAS state, and warms the matvec path once.
class GpuPcgContext {
public:
    GpuPcgContext(const StructuredHexMesh& mesh,
                  const Material& material,
                  int block_y = 4);
    ~GpuPcgContext();

    GpuPcgContext(GpuPcgContext&&) noexcept;
    GpuPcgContext& operator=(GpuPcgContext&&) noexcept;

    GpuPcgContext(const GpuPcgContext&) = delete;
    GpuPcgContext& operator=(const GpuPcgContext&) = delete;

    // Fixed-forcing path retained for reproducible experiments.
    GpuPcgResult solve(const std::vector<float>& rhs,
                       double relative_tolerance = 1.0e-5,
                       std::size_t max_iterations = 2000);

    // Adaptive path: no inner tolerance is supplied. PCG audits its current
    // correction at geometric residual milestones and estimates the total
    // remaining outer-refinement cost if it stops at that point. It continues
    // only while a one-decade-trusted extrapolation predicts that a deeper
    // correction can reduce total time. outer_remaining_ratio is
    // outer_tolerance/current_outer_residual. contraction_gain maps achieved
    // inner audited residual to measured outer contraction and is learned by
    // the accurate outer loop (1.0 before the first correction).
    GpuPcgResult solve_economic(
        const std::vector<float>& rhs,
        double outer_remaining_ratio,
        double accurate_residual_cost_ms,
        double contraction_gain,
        std::size_t max_iterations = 5000);

    std::size_t explicit_device_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gfss
