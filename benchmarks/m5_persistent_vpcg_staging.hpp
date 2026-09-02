#pragma once

#include "gfss/gpu_m5_fine_level.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gfss {

struct M5PersistentPcgSolveResult {
    std::vector<float> solution_aos;
    double solve_ms{0.0};
    double recursive_relative_residual{0.0};
    std::size_t iterations{0U};
    std::size_t total_l0_operator_applies{0U};
    std::size_t device_bytes_total{0U};
    bool breakdown{false};
};

// Staging-only persistent deep-hierarchy MG-PCG context. The fine-level context
// owns A0/P0 metadata and work vectors. This object uploads the L1/L2/L3 payload,
// PCG vectors, cuBLAS handles and timing events exactly once, then solves many
// correction equations by replacing only the fine RHS. L3 is the validated
// explicit symmetric FP32 inverse and is applied with cuBLAS SGEMV.
class M5PersistentPcgStaging {
public:
    M5PersistentPcgStaging(
        GpuM5FineLevelContext& fine,
        std::size_t l0_smoother_degree,
        std::size_t m0,
        const std::vector<float>& l1_inverse_blocks_6x6,
        double lambda1,
        std::size_t l2_nodes,
        const std::vector<std::uint32_t>& p1_forward_row_offsets,
        const std::vector<std::uint32_t>& p1_forward_column_indices,
        const std::vector<float>& p1_forward_values_6x6,
        const std::vector<std::uint32_t>& p1_transpose_column_offsets,
        const std::vector<std::uint32_t>& p1_transpose_row_indices,
        const std::vector<float>& p1_transpose_values_q_r_entry,
        const std::vector<float>& a2_dense_row_major,
        const std::vector<float>& l2_inverse_blocks_6x6,
        double lambda2,
        const std::vector<float>& p2_dense_row_major,
        std::size_t l3_dofs,
        const std::vector<float>& l3_inverse_col_major);
    ~M5PersistentPcgStaging();

    M5PersistentPcgStaging(const M5PersistentPcgStaging&) = delete;
    M5PersistentPcgStaging& operator=(const M5PersistentPcgStaging&) = delete;
    M5PersistentPcgStaging(M5PersistentPcgStaging&&) noexcept;
    M5PersistentPcgStaging& operator=(M5PersistentPcgStaging&&) noexcept;

    // Exercise all kernels once before the solver-level wall timer starts.
    void warmup(const std::vector<float>& rhs_aos, std::size_t iterations);

    // H2D RHS, one fixed-iteration device-resident MG-PCG solve, final scalar
    // audits and D2H solution. solve_ms contains GPU execution only; caller wall
    // timing additionally captures the small RHS/solution transfers.
    M5PersistentPcgSolveResult solve(
        const std::vector<float>& rhs_aos,
        std::size_t iterations);

    std::size_t device_bytes_total() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gfss
