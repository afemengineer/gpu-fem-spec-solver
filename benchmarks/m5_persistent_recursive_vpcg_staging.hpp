#pragma once

#include "gfss/gpu_m5_fine_level.hpp"
#include "gfss/gpu_m5_recursive_tail.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gfss {

struct M5PersistentRecursivePcgSolveResult {
    std::vector<float> solution_aos;
    double solve_ms{0.0};
    double recursive_relative_residual{0.0};
    std::size_t iterations{0U};
    std::size_t total_l0_operator_applies{0U};
    std::size_t device_bytes_total{0U};
    std::size_t recursive_tail_device_bytes{0U};
    bool breakdown{false};
};

// Production-integration sibling to M5PersistentPcgStaging.  L0, P0, L1 and
// P1/P1T are the same validated persistent path.  The former hard-coded
// L2->L3 section is replaced by a device-resident arbitrary-depth recursive
// tail whose top level is exactly the P1-restricted L2 space.
class M5PersistentRecursivePcgStaging {
public:
    M5PersistentRecursivePcgStaging(
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
        const std::vector<M5RecursiveTailLevelPayload>& tail_levels,
        const std::vector<float>& bottom_inverse_col_major);
    ~M5PersistentRecursivePcgStaging();

    M5PersistentRecursivePcgStaging(const M5PersistentRecursivePcgStaging&) = delete;
    M5PersistentRecursivePcgStaging& operator=(const M5PersistentRecursivePcgStaging&) = delete;
    M5PersistentRecursivePcgStaging(M5PersistentRecursivePcgStaging&&) noexcept;
    M5PersistentRecursivePcgStaging& operator=(M5PersistentRecursivePcgStaging&&) noexcept;

    void warmup(const std::vector<float>& rhs_aos, std::size_t iterations);
    M5PersistentRecursivePcgSolveResult solve(
        const std::vector<float>& rhs_aos,
        std::size_t iterations);

    std::size_t device_bytes_total() const noexcept;
    std::size_t recursive_tail_device_bytes() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace gfss
