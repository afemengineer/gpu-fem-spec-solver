#include "gfss/multilevel_reference.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

int main() {
    try {
        gfss::ReferenceMultilevelLevel fine;
        fine.dofs = 2U;
        fine.label = "toy_fine";
        fine.apply = [](const std::vector<double>& x) {
            return x;  // Identity operator.
        };
        fine.smooth = [](const std::vector<double>& b,
                         std::vector<double>& x,
                         std::size_t steps) {
            if (steps > 0U) {
                // Exact smoothing of the component intentionally omitted from
                // the coarse space; leave component 0 for coarse correction.
                x[1] = b[1];
            }
        };
        fine.restrict_to_coarse = [](const std::vector<double>& r) {
            return std::vector<double>{r[0]};
        };
        fine.prolong_from_coarse = [](const std::vector<double>& e) {
            return std::vector<double>{e[0], 0.0};
        };

        gfss::ReferenceMultilevelLevel coarse;
        coarse.dofs = 1U;
        coarse.label = "toy_bottom";
        coarse.apply = [](const std::vector<double>& x) { return x; };
        coarse.bottom_solve = [](const std::vector<double>& b) { return b; };

        const std::vector<gfss::ReferenceMultilevelLevel> levels{
            std::move(fine), std::move(coarse)};
        const std::vector<double> rhs{1.25, -2.5};
        const auto result = gfss::solve_reference_multilevel_vcycle(
            levels, rhs, 1.0e-12, 2U, 1U, 1U);

        if (!result.converged || result.cycles != 1U) {
            throw std::runtime_error("toy multilevel solve did not converge in one cycle");
        }
        if (result.x.size() != rhs.size() ||
            std::abs(result.x[0] - rhs[0]) > 1.0e-12 ||
            std::abs(result.x[1] - rhs[1]) > 1.0e-12) {
            throw std::runtime_error("toy multilevel solution mismatch");
        }
        if (result.relative_residuals.empty() ||
            result.relative_residuals.back() > 1.0e-12) {
            throw std::runtime_error("toy multilevel true residual mismatch");
        }

        std::cout << "backend-neutral multilevel recursion test passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "multilevel_reference_test failed: " << e.what() << '\n';
        return 1;
    }
}
