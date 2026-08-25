#include "gfss/cpu_elasticity.hpp"

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

bool close(double a, double b, double atol = 1.0e-9, double rtol = 1.0e-10) {
    return std::abs(a - b) <= atol + rtol * std::max(std::abs(a), std::abs(b));
}

std::vector<double> dense_apply(const gfss::DenseMatrix& a,
                                const std::vector<double>& x) {
    const std::size_t n = x.size();
    std::vector<double> y(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            y[i] += a[i * n + j] * x[j];
        }
    }
    return y;
}

}  // namespace

int main() {
    const gfss::StructuredHexMesh mesh{2, 2, 1, 2.0, 1.5, 0.75};
    const gfss::Material material{73000.0, 0.29};

    require(mesh.node_count() == 18, "2x2x1 mesh must have 18 nodes");
    require(mesh.element_count() == 4, "2x2x1 mesh must have 4 elements");
    require(mesh.dof_count() == 54, "2x2x1 mesh must have 54 displacement DOFs");

    const auto first = mesh.element_nodes(0, 0, 0);
    require(first[0] == mesh.node_index(0, 0, 0), "element node 0 indexing mismatch");
    require(first[6] == mesh.node_index(1, 1, 1), "element node 6 indexing mismatch");

    const auto assembled = gfss::assemble_dense_stiffness(mesh, material);
    std::vector<double> x(static_cast<std::size_t>(mesh.dof_count()));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.31 * std::sin(0.17 * static_cast<double>(i + 1)) +
               0.07 * std::cos(0.11 * static_cast<double>(i + 3));
    }

    const auto ya = dense_apply(assembled, x);
    const auto ymf = gfss::apply_matrix_free(mesh, material, x);

    double max_abs = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(ya[i] - ymf[i]));
        scale = std::max(scale, std::abs(ya[i]));
    }
    require(max_abs <= 5.0e-11 * std::max(1.0, scale),
            "matrix-free A*x must match assembled K*x");

    auto constrained_matrix = assembled;
    std::vector<double> rhs(x.size(), 1.0);
    const auto constrained = gfss::clamped_x0_dofs(mesh);
    gfss::apply_symmetric_dirichlet(constrained_matrix, rhs, constrained);

    const auto ybc_dense = dense_apply(constrained_matrix, x);
    const auto ybc_mf = gfss::apply_clamped_x0_matrix_free(mesh, material, x);
    for (std::size_t i = 0; i < x.size(); ++i) {
        require(close(ybc_dense[i], ybc_mf[i], 1.0e-8, 1.0e-10),
                "matrix-free clamped operator must match symmetric dense elimination");
    }

    for (const auto dof : constrained) {
        require(rhs[static_cast<std::size_t>(dof)] == 0.0,
                "Dirichlet RHS entry must be zero");
        require(ybc_mf[static_cast<std::size_t>(dof)] == x[static_cast<std::size_t>(dof)],
                "constrained operator row must act as identity");
    }

    std::cout << "CPU assembled/matrix-free operator checks passed\n";
    return 0;
}
