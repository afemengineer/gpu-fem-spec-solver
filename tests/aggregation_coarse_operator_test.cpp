#include "gfss/aggregation_coarse_operator.hpp"
#include "gfss/cpu_elasticity.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("dot size mismatch");
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

double relative_difference(const std::vector<double>& a,
                           const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("difference size mismatch");
    double dd = 0.0;
    double bb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        dd += d * d;
        bb += b[i] * b[i];
    }
    return std::sqrt(dd / std::max(bb, 1.0e-300));
}

}  // namespace

int main() {
    try {
        const gfss::StructuredHexMesh mesh{8U, 6U, 4U, 2.0, 1.5, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        auto graph = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph),
            gfss::ElasticityAggregationOptions{10U, 4U, 1.0e-10});

        const auto fine_apply = [&](const std::vector<double>& x) {
            return gfss::apply_clamped_x0_matrix_free(mesh, material, x);
        };

        std::vector<double> u(space.coarse_dofs, 0.0);
        std::vector<double> v(space.coarse_dofs, 0.0);
        for (std::size_t i = 0; i < space.coarse_dofs; ++i) {
            const double t = static_cast<double>(i + 1U);
            u[i] = std::sin(0.173 * t) + 0.2 * std::cos(0.071 * t);
            v[i] = std::cos(0.113 * t) - 0.3 * std::sin(0.047 * t);
        }

        const auto ac_u = gfss::apply_elasticity_aggregation_coarse_operator(
            space, fine_apply, u);
        const auto ac_v = gfss::apply_elasticity_aggregation_coarse_operator(
            space, fine_apply, v);

        const auto pu = gfss::apply_elasticity_tentative_prolongation(space, u);
        const auto af_pu = fine_apply(pu);
        const auto manual_ac_u =
            gfss::apply_elasticity_tentative_restriction(space, af_pu);
        if (!(relative_difference(ac_u, manual_ac_u) < 1.0e-13)) {
            throw std::runtime_error("aggregation coarse action != P^T A P audit");
        }

        const double coarse_energy = dot(u, ac_u);
        const double fine_energy = dot(pu, af_pu);
        const double energy_rel = std::abs(coarse_energy - fine_energy) /
            std::max(std::abs(fine_energy), 1.0e-300);
        if (!(coarse_energy > 0.0) || !(energy_rel < 1.0e-12)) {
            throw std::runtime_error("aggregation coarse energy identity failed");
        }

        const double uv = dot(u, ac_v);
        const double vu = dot(v, ac_u);
        const double symmetry_rel = std::abs(uv - vu) /
            std::max({std::abs(uv), std::abs(vu), 1.0});
        if (!(symmetry_rel < 1.0e-11)) {
            throw std::runtime_error("aggregation coarse operator symmetry audit failed");
        }

        std::cout << "aggregation matrix-light coarse-operator regression passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "aggregation_coarse_operator_test failed: " << e.what() << '\n';
        return 1;
    }
}
