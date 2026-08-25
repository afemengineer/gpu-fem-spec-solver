#include "gfss/cpu_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gfss {
namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot product size mismatch");
    }
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        value += a[i] * b[i];
    }
    return value;
}

double l2_norm(const std::vector<double>& x) {
    return std::sqrt(dot(x, x));
}

}  // namespace

CgResult conjugate_gradient(const CpuLinearOperator& apply,
                            const std::vector<double>& rhs,
                            double relative_tolerance,
                            std::size_t max_iterations) {
    if (rhs.empty()) {
        throw std::invalid_argument("CG RHS must not be empty");
    }
    if (!(relative_tolerance > 0.0)) {
        throw std::invalid_argument("CG tolerance must be positive");
    }

    CgResult result;
    result.x.assign(rhs.size(), 0.0);
    std::vector<double> r = rhs;
    std::vector<double> p = r;

    const double rhs_norm = l2_norm(rhs);
    if (rhs_norm == 0.0) {
        result.relative_residual = 0.0;
        result.converged = true;
        return result;
    }

    double rr = dot(r, r);
    result.relative_residual = std::sqrt(rr) / rhs_norm;

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        const std::vector<double> ap = apply(p);
        if (ap.size() != rhs.size()) {
            throw std::runtime_error("linear operator returned wrong vector size");
        }

        const double pap = dot(p, ap);
        if (!(pap > 0.0) || !std::isfinite(pap)) {
            throw std::runtime_error("CG breakdown: operator is not positive definite along search direction");
        }

        const double alpha = rr / pap;
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            result.x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        const double rr_new = dot(r, r);
        result.iterations = iteration + 1;
        result.relative_residual = std::sqrt(rr_new) / rhs_norm;
        if (result.relative_residual <= relative_tolerance) {
            result.converged = true;
            return result;
        }

        if (!std::isfinite(rr_new)) {
            throw std::runtime_error("CG breakdown: residual became non-finite");
        }

        const double beta = rr_new / rr;
        for (std::size_t i = 0; i < rhs.size(); ++i) {
            p[i] = r[i] + beta * p[i];
        }
        rr = rr_new;
    }

    return result;
}

std::vector<double> solve_dense_gaussian(std::vector<double> matrix,
                                         std::vector<double> rhs) {
    const std::size_t n = rhs.size();
    if (n == 0 || matrix.size() != n * n) {
        throw std::invalid_argument("dense system dimensions are inconsistent");
    }

    for (std::size_t k = 0; k < n; ++k) {
        std::size_t pivot = k;
        double pivot_abs = std::abs(matrix[k * n + k]);
        for (std::size_t row = k + 1; row < n; ++row) {
            const double candidate = std::abs(matrix[row * n + k]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("dense validation solve encountered a singular matrix");
        }

        if (pivot != k) {
            for (std::size_t col = k; col < n; ++col) {
                std::swap(matrix[k * n + col], matrix[pivot * n + col]);
            }
            std::swap(rhs[k], rhs[pivot]);
        }

        const double diagonal = matrix[k * n + k];
        for (std::size_t row = k + 1; row < n; ++row) {
            const double factor = matrix[row * n + k] / diagonal;
            matrix[row * n + k] = 0.0;
            for (std::size_t col = k + 1; col < n; ++col) {
                matrix[row * n + col] -= factor * matrix[k * n + col];
            }
            rhs[row] -= factor * rhs[k];
        }
    }

    std::vector<double> x(n, 0.0);
    for (std::size_t row_rev = 0; row_rev < n; ++row_rev) {
        const std::size_t row = n - 1 - row_rev;
        double sum = rhs[row];
        for (std::size_t col = row + 1; col < n; ++col) {
            sum -= matrix[row * n + col] * x[col];
        }
        x[row] = sum / matrix[row * n + row];
    }
    return x;
}

}  // namespace gfss
