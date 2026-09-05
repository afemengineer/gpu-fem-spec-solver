#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gfss {

enum class M5RecursiveTailOperatorKind {
    dense_fp32,
    structural_scalar_csr_fp32,
};

struct M5RecursiveTailLevelPayload {
    std::size_t level{0U};
    std::size_t dofs{0U};
    std::size_t nodes{0U};
    std::vector<std::uint32_t> block_dof_offsets;
    std::vector<float> inverse_blocks_padded_6x6;
    double lambda{0.0};
    M5RecursiveTailOperatorKind operator_kind{M5RecursiveTailOperatorKind::dense_fp32};

    // Exactly one operator payload is active on non-bottom levels.
    std::vector<float> dense_row_major;
    std::vector<std::uint32_t> csr_row_offsets;
    std::vector<std::uint32_t> csr_column_indices;
    std::vector<float> csr_values;

    // Explicit scalar CSR transfer from this level to level+1.  P is stored
    // row-wise as current_dofs x next_dofs; PT is independently row-wise as
    // next_dofs x current_dofs so restriction needs no atomics.
    std::size_t next_dofs{0U};
    std::vector<std::uint32_t> p_row_offsets;
    std::vector<std::uint32_t> p_column_indices;
    std::vector<float> p_values;
    std::vector<std::uint32_t> pt_row_offsets;
    std::vector<std::uint32_t> pt_column_indices;
    std::vector<float> pt_values;
};

struct M5RecursiveTailGpuResult {
    std::vector<float> x;
    double median_ms{0.0};
    double best_ms{0.0};
    std::size_t device_bytes{0U};
    std::vector<std::string> runtime_representations;
};

// Persistent FP32 recursive V-cycle for algebraic coarse levels.  All payloads
// are uploaded once.  Each non-bottom level performs one weighted block-Jacobi
// pre step from zero, one recursive coarse correction, and one weighted
// block-Jacobi post step.  The final level is solved by an explicit symmetric
// FP32 inverse through cuBLAS SGEMV.
M5RecursiveTailGpuResult benchmark_m5_recursive_tail_vcycle(
    const std::vector<M5RecursiveTailLevelPayload>& levels,
    const std::vector<float>& bottom_inverse_col_major,
    const std::vector<float>& rhs,
    int repeats = 50);

}  // namespace gfss
