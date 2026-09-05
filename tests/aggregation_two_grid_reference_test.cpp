#include "gfss/aggregation_coarse_operator.hpp"
#include "gfss/aggregation_two_grid_reference.hpp"
#include "gfss/cpu_elasticity.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

double norm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}
}

int main() {
    try {
        const gfss::StructuredHexMesh mesh{8U, 6U, 4U, 2.0, 1.5, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        auto graph = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph), {10U, 4U, 1.0e-10});
        const auto matrix = gfss::assemble_structured_hex_aggregation_galerkin(
            mesh, material, space);

        std::vector<double> coarse(space.coarse_dofs, 0.0);
        for (std::size_t i = 0; i < coarse.size(); ++i) {
            const double t = static_cast<double>(i + 1U);
            coarse[i] = std::sin(0.11 * t) + 0.23 * std::cos(0.07 * t);
        }
        const auto explicit_y = gfss::apply_aggregation_variable_block_matrix(matrix, coarse);
        const auto apply_fine = [&](const std::vector<double>& x) {
            return gfss::apply_clamped_x0_matrix_free(mesh, material, x);
        };
        const auto oracle_y = gfss::apply_elasticity_aggregation_coarse_operator(
            space, apply_fine, coarse);
        std::vector<double> diff(oracle_y.size(), 0.0);
        for (std::size_t i = 0; i < diff.size(); ++i) diff[i] = explicit_y[i] - oracle_y[i];
        const double rel = norm(diff) / norm(oracle_y);
        if (!(rel < 1.0e-10)) {
            throw std::runtime_error("element-local Galerkin assembly disagrees with P^T A P oracle");
        }
        std::cout << "aggregation two-grid Galerkin assembly regression passed rel=" << rel << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "aggregation_two_grid_reference_test failed: " << e.what() << '\n';
        return 1;
    }
}
