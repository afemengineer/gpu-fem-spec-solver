#include "gfss/aggregation_coarse_space.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

int main() {
    try {
        const gfss::StructuredHexMesh mesh{8U, 6U, 4U, 2.0, 1.5, 1.0};
        auto graph = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph),
            gfss::ElasticityAggregationOptions{10U, 4U, 1.0e-10});

        if (space.aggregates.empty() || space.coarse_dofs == 0U) {
            throw std::runtime_error("aggregation produced empty coarse space");
        }
        if (space.coarse_dofs >= space.fine_free_dofs) {
            throw std::runtime_error("aggregation did not reduce free DOF count");
        }
        for (const auto& aggregate : space.aggregates) {
            if (aggregate.rank < 3U || aggregate.rank > 6U) {
                throw std::runtime_error("aggregate rigid-body rank outside [3,6]");
            }
        }

        const double rigid_error =
            gfss::audit_elasticity_rigid_body_reproduction(space);
        if (!(rigid_error < 1.0e-10)) {
            throw std::runtime_error("tentative P failed rigid-body reproduction audit");
        }

        // Because each aggregate basis is locally orthonormal, P^T P should
        // act as identity on the coarse vector to roundoff.
        std::vector<double> coarse(space.coarse_dofs, 0.0);
        for (std::size_t i = 0; i < coarse.size(); ++i) {
            coarse[i] = std::sin(0.17 * static_cast<double>(i + 1U));
        }
        const auto fine =
            gfss::apply_elasticity_tentative_prolongation(space, coarse);
        const auto recovered =
            gfss::apply_elasticity_tentative_restriction(space, fine);
        double dd = 0.0;
        double cc = 0.0;
        for (std::size_t i = 0; i < coarse.size(); ++i) {
            const double d = recovered[i] - coarse[i];
            dd += d * d;
            cc += coarse[i] * coarse[i];
        }
        const double rel = std::sqrt(dd / cc);
        if (!(rel < 1.0e-10)) {
            throw std::runtime_error("tentative transfer failed P^T P identity audit");
        }

        std::cout << "aggregation coarse-space regression passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "aggregation_coarse_space_test failed: " << e.what() << '\n';
        return 1;
    }
}
