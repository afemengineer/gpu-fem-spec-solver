#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace gfss {

struct SolveCriteria {
    double relative_residual_tolerance = 1.0e-6;
    std::uint32_t max_iterations = 2000;
};

struct SolveReport {
    bool converged = false;
    std::uint32_t iterations = 0;
    double initial_residual_norm = 0.0;
    double final_residual_norm = 0.0;
    double solve_seconds = 0.0;
    std::uint64_t peak_device_bytes = 0;
    std::string solver_name;
    std::string preconditioner_name;
};

// Deliberately backend-agnostic. Concrete CPU/CUDA vector/operator interfaces
// arrive in M1/M2 after the trusted reference representation is fixed.
class OperatorMetadata {
public:
    virtual ~OperatorMetadata() = default;
    [[nodiscard]] virtual std::uint64_t rows() const = 0;
    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual bool is_matrix_free() const = 0;
};

}  // namespace gfss
