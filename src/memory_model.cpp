#include "gfss/memory_model.hpp"

#include <iomanip>
#include <iterator>
#include <sstream>

namespace gfss {
namespace {
constexpr std::uint64_t kFp32Bytes = 4;
constexpr std::uint64_t kMaskBytes = 1;
}

MemoryEstimate estimate_structured_matrix_free_memory(
    const StructuredHexProblem& problem,
    const BaselineMemoryPolicy& policy) {
    problem.validate();

    MemoryEstimate result;
    result.dofs = problem.dof_count();
    result.fixed_reserve_bytes = policy.fixed_reserve_bytes;

    const auto vector_bytes = result.dofs * kFp32Bytes * policy.fp32_vector_count;
    result.components.push_back({"fp32_work_vectors", vector_bytes});
    result.variable_bytes += vector_bytes;

    if (policy.store_fp32_jacobi_diagonal) {
        const auto diagonal_bytes = result.dofs * kFp32Bytes;
        result.components.push_back({"jacobi_diagonal", diagonal_bytes});
        result.variable_bytes += diagonal_bytes;
    }

    if (policy.store_dirichlet_mask) {
        const auto mask_bytes = result.dofs * kMaskBytes;
        result.components.push_back({"dirichlet_mask", mask_bytes});
        result.variable_bytes += mask_bytes;
    }

    return result;
}

std::uint64_t max_cubic_mesh_for_budget(
    const std::uint64_t budget_bytes,
    const BaselineMemoryPolicy& policy) {
    if (budget_bytes <= policy.fixed_reserve_bytes) {
        return 0;
    }

    std::uint64_t lo = 0;
    std::uint64_t hi = 1;

    auto fits = [&](std::uint64_t n) {
        StructuredHexProblem problem;
        problem.nx = n;
        problem.ny = n;
        problem.nz = n;
        return estimate_structured_matrix_free_memory(problem, policy).total_bytes() <= budget_bytes;
    };

    while (fits(hi) && hi < (1ULL << 20U)) {
        hi *= 2;
    }

    while (lo + 1 < hi) {
        const auto mid = lo + (hi - lo) / 2;
        if (fits(mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

std::string format_bytes(const std::uint64_t bytes) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < std::size(units)) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2) << value << ' ' << units[unit];
    return out.str();
}

}  // namespace gfss
