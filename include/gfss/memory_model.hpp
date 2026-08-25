#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gfss/problem.hpp"

namespace gfss {

struct MemoryComponent {
    std::string name;
    std::uint64_t bytes = 0;
};

struct MemoryEstimate {
    std::uint64_t dofs = 0;
    std::uint64_t variable_bytes = 0;
    std::uint64_t fixed_reserve_bytes = 0;
    std::vector<MemoryComponent> components;

    [[nodiscard]] std::uint64_t total_bytes() const {
        return variable_bytes + fixed_reserve_bytes;
    }

    [[nodiscard]] double bytes_per_dof() const {
        return dofs == 0 ? 0.0 : static_cast<double>(variable_bytes) / static_cast<double>(dofs);
    }
};

struct BaselineMemoryPolicy {
    // x, b, r, p, Ap, z. Kept explicit so later solvers cannot hide work vectors.
    std::uint32_t fp32_vector_count = 6;
    bool store_fp32_jacobi_diagonal = true;
    bool store_dirichlet_mask = true;
    std::uint64_t fixed_reserve_bytes = 512ULL * 1024ULL * 1024ULL;
};

[[nodiscard]] MemoryEstimate estimate_structured_matrix_free_memory(
    const StructuredHexProblem& problem,
    const BaselineMemoryPolicy& policy = {});

[[nodiscard]] std::uint64_t max_cubic_mesh_for_budget(
    std::uint64_t budget_bytes,
    const BaselineMemoryPolicy& policy = {});

[[nodiscard]] std::string format_bytes(std::uint64_t bytes);

}  // namespace gfss
