#include "gfss/cpu_elasticity.hpp"

#include "gfss/hex8.hpp"

#include <algorithm>
#include <stdexcept>

namespace gfss {
namespace {

std::uint64_t global_dof(std::uint64_t node, int component) {
    return 3ULL * node + static_cast<std::uint64_t>(component);
}

void validate_vector_size(const StructuredHexMesh& mesh, const std::vector<double>& x) {
    if (x.size() != mesh.dof_count()) {
        throw std::invalid_argument("vector size does not match mesh DOF count");
    }
}

}  // namespace

DenseMatrix assemble_dense_stiffness(const StructuredHexMesh& mesh,
                                     const Material& material) {
    const std::uint64_t ndof = mesh.dof_count();
    if (ndof > 10000) {
        throw std::invalid_argument("dense reference assembly is restricted to small validation meshes");
    }

    DenseMatrix matrix(static_cast<std::size_t>(ndof * ndof), 0.0);
    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                const auto coords = mesh.element_coordinates(ex, ey, ez);
                const auto ke = hex8_stiffness(coords, material);

                for (int a = 0; a < 8; ++a) {
                    for (int ca = 0; ca < 3; ++ca) {
                        const auto row = global_dof(nodes[a], ca);
                        const int local_row = 3 * a + ca;
                        for (int b = 0; b < 8; ++b) {
                            for (int cb = 0; cb < 3; ++cb) {
                                const auto col = global_dof(nodes[b], cb);
                                const int local_col = 3 * b + cb;
                                matrix[static_cast<std::size_t>(row * ndof + col)] +=
                                    ke[local_row][local_col];
                            }
                        }
                    }
                }
            }
        }
    }
    return matrix;
}

std::vector<double> apply_matrix_free(const StructuredHexMesh& mesh,
                                      const Material& material,
                                      const std::vector<double>& x) {
    validate_vector_size(mesh, x);
    std::vector<double> y(x.size(), 0.0);

    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                const auto coords = mesh.element_coordinates(ex, ey, ez);
                Hex8Vector xe{};
                for (int a = 0; a < 8; ++a) {
                    for (int c = 0; c < 3; ++c) {
                        xe[3 * a + c] = x[global_dof(nodes[a], c)];
                    }
                }

                const auto ye = hex8_apply(coords, material, xe);
                for (int a = 0; a < 8; ++a) {
                    for (int c = 0; c < 3; ++c) {
                        y[global_dof(nodes[a], c)] += ye[3 * a + c];
                    }
                }
            }
        }
    }
    return y;
}

std::vector<std::uint64_t> clamped_x0_dofs(const StructuredHexMesh& mesh) {
    std::vector<std::uint64_t> dofs;
    dofs.reserve(static_cast<std::size_t>(3ULL * (mesh.ny + 1ULL) * (mesh.nz + 1ULL)));
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(0, j, k);
            dofs.push_back(global_dof(node, 0));
            dofs.push_back(global_dof(node, 1));
            dofs.push_back(global_dof(node, 2));
        }
    }
    return dofs;
}

void apply_symmetric_dirichlet(DenseMatrix& matrix,
                               std::vector<double>& rhs,
                               const std::vector<std::uint64_t>& constrained_dofs) {
    const std::uint64_t ndof = static_cast<std::uint64_t>(rhs.size());
    if (matrix.size() != static_cast<std::size_t>(ndof * ndof)) {
        throw std::invalid_argument("dense matrix dimensions do not match RHS");
    }

    for (const auto dof : constrained_dofs) {
        if (dof >= ndof) {
            throw std::out_of_range("constrained DOF out of range");
        }
        for (std::uint64_t j = 0; j < ndof; ++j) {
            matrix[static_cast<std::size_t>(dof * ndof + j)] = 0.0;
            matrix[static_cast<std::size_t>(j * ndof + dof)] = 0.0;
        }
        matrix[static_cast<std::size_t>(dof * ndof + dof)] = 1.0;
        rhs[static_cast<std::size_t>(dof)] = 0.0;
    }
}

std::vector<double> apply_clamped_x0_matrix_free(const StructuredHexMesh& mesh,
                                                  const Material& material,
                                                  const std::vector<double>& x) {
    validate_vector_size(mesh, x);
    auto x_free = x;
    const auto constrained = clamped_x0_dofs(mesh);
    for (const auto dof : constrained) {
        x_free[static_cast<std::size_t>(dof)] = 0.0;
    }

    auto y = apply_matrix_free(mesh, material, x_free);
    for (const auto dof : constrained) {
        y[static_cast<std::size_t>(dof)] = x[static_cast<std::size_t>(dof)];
    }
    return y;
}

}  // namespace gfss
