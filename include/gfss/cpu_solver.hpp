#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace gfss {

using CpuLinearOperator = std::function<std::vector<double>(const std::vector<double>&)>;

struct CgResult {
    std::vector<double> x;
    std::size_t iterations{0};
    double relative_residual{0.0};
    bool converged{false};
};

CgResult conjugate_gradient(const CpuLinearOperator& apply,
                            const std::vector<double>& rhs,
                            double relative_tolerance = 1.0e-10,
                            std::size_t max_iterations = 10000);

std::vector<double> solve_dense_gaussian(std::vector<double> matrix,
                                         std::vector<double> rhs);

}  // namespace gfss
