#include "gfss/smoothed_aggregation_reference.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        const gfss::StructuredHexMesh mesh{8U, 6U, 4U, 2.0, 1.5, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
        for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
            for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
                const auto node = mesh.node_index(mesh.nx, j, k);
                rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = -1.0;
            }
        }

        gfss::SmoothedAggregationOptions options;
        options.target_nodes_per_aggregate = 10U;
        options.min_nodes_per_aggregate = 4U;
        options.transfer_smoothing_steps = 1U;
        options.power_iterations = 6U;
        options.pre_smooth_degree = 1U;
        options.post_smooth_degree = 1U;
        options.coarse_relative_tolerance = 1.0e-4;
        options.coarse_max_iterations = 500U;
        options.true_relative_tolerance = 1.0e-12;
        options.max_cycles = 1U;

        const auto result = gfss::solve_smoothed_aggregation_two_grid_reference(
            mesh, material, rhs, options);
        if (!(result.transfer_adjoint_relative_error < 1.0e-10)) {
            throw std::runtime_error("factorized smoothed transfer failed adjoint audit");
        }
        if (!(result.coarse_symmetry_relative_defect < 1.0e-10)) {
            throw std::runtime_error("smoothed Galerkin coarse action lost symmetry");
        }
        if (!result.coarse_spd_probe) {
            throw std::runtime_error("smoothed Galerkin coarse action failed SPD probe");
        }
        if (result.true_relative_residuals.size() != 2U) {
            throw std::runtime_error("smoothed aggregation one-cycle regression missing residual");
        }
        std::cout << "smoothed aggregation regression passed adjoint="
                  << result.transfer_adjoint_relative_error
                  << " symmetry=" << result.coarse_symmetry_relative_defect
                  << " q=" << result.cycle_contractions.front() << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "smoothed_aggregation_reference_test failed: " << e.what() << '\n';
        return 1;
    }
}
