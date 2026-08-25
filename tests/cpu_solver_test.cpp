#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

double norm2(const std::vector<double>& x) {
    double sum = 0.0;
    for (double v : x) {
        sum += v * v;
    }
    return std::sqrt(sum);
}

}  // namespace

int main() {
    const gfss::StructuredHexMesh mesh{2, 1, 1, 2.0, 1.0, 1.0};
    const gfss::Material material{210000.0, 0.30};
    const std::size_t ndof = static_cast<std::size_t>(mesh.dof_count());

    std::vector<double> load(ndof, 0.0);
    constexpr double total_fz = -100.0;
    const double nodal_fz = total_fz / 4.0;
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            load[static_cast<std::size_t>(3ULL * node + 2ULL)] += nodal_fz;
        }
    }

    auto dense = gfss::assemble_dense_stiffness(mesh, material);
    auto dense_rhs = load;
    const auto constrained = gfss::clamped_x0_dofs(mesh);
    gfss::apply_symmetric_dirichlet(dense, dense_rhs, constrained);
    const auto x_direct = gfss::solve_dense_gaussian(dense, dense_rhs);

    const auto op = [&](const std::vector<double>& x) {
        return gfss::apply_clamped_x0_matrix_free(mesh, material, x);
    };
    const auto cg = gfss::conjugate_gradient(op, load, 1.0e-11, 1000);
    require(cg.converged, "serial matrix-free CG must converge");

    std::vector<double> delta(ndof, 0.0);
    for (std::size_t i = 0; i < ndof; ++i) {
        delta[i] = cg.x[i] - x_direct[i];
    }
    const double solution_relative_error = norm2(delta) / std::max(1.0e-30, norm2(x_direct));
    require(solution_relative_error < 1.0e-9,
            "matrix-free CG solution must match the dense direct reference");

    const auto true_ax = op(cg.x);
    std::vector<double> true_residual(ndof, 0.0);
    for (std::size_t i = 0; i < ndof; ++i) {
        true_residual[i] = load[i] - true_ax[i];
    }
    require(norm2(true_residual) / norm2(load) < 5.0e-10,
            "recomputed true residual must satisfy the accuracy contract");

    // Recover reactions from the unconstrained physical operator K*u - f.
    const auto internal = gfss::apply_matrix_free(mesh, material, cg.x);
    double reaction_x = 0.0;
    double reaction_y = 0.0;
    double reaction_z = 0.0;
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(0, j, k);
            reaction_x += internal[static_cast<std::size_t>(3ULL * node + 0ULL)] -
                          load[static_cast<std::size_t>(3ULL * node + 0ULL)];
            reaction_y += internal[static_cast<std::size_t>(3ULL * node + 1ULL)] -
                          load[static_cast<std::size_t>(3ULL * node + 1ULL)];
            reaction_z += internal[static_cast<std::size_t>(3ULL * node + 2ULL)] -
                          load[static_cast<std::size_t>(3ULL * node + 2ULL)];
        }
    }

    const double force_scale = std::abs(total_fz);
    require(std::abs(reaction_x) < 1.0e-8 * force_scale,
            "net x reaction must be approximately zero");
    require(std::abs(reaction_y) < 1.0e-8 * force_scale,
            "net y reaction must be approximately zero");
    require(std::abs(reaction_z + total_fz) < 1.0e-8 * force_scale,
            "support reaction must balance the applied z load");

    std::cout << "CPU solver checks passed: iterations=" << cg.iterations
              << " rel_residual=" << cg.relative_residual
              << " solution_rel_error=" << solution_relative_error << '\n';
    return 0;
}
