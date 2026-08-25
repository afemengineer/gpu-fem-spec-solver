#include "gfss/cpu_elasticity.hpp"

#include "gfss/hex8.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

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

Hex8Matrix uniform_element_stiffness(const StructuredHexMesh& mesh,
                                     const Material& material) {
    if (mesh.nx == 0 || mesh.ny == 0 || mesh.nz == 0) {
        throw std::invalid_argument("structured mesh dimensions must be non-zero");
    }
    return hex8_stiffness(mesh.element_coordinates(0, 0, 0), material);
}

Hex8Vector apply_element_matrix(const Hex8Matrix& ke, const Hex8Vector& x) {
    Hex8Vector y{};
    for (int row = 0; row < 24; ++row) {
        double sum = 0.0;
        for (int col = 0; col < 24; ++col) {
            sum += ke[row][col] * x[col];
        }
        y[row] = sum;
    }
    return y;
}

void apply_one_element(const StructuredHexMesh& mesh,
                       const Hex8Matrix& ke,
                       const std::vector<double>& x,
                       std::vector<double>& y,
                       std::uint32_t ex,
                       std::uint32_t ey,
                       std::uint32_t ez) {
    const auto nodes = mesh.element_nodes(ex, ey, ez);
    Hex8Vector xe{};
    for (int a = 0; a < 8; ++a) {
        for (int c = 0; c < 3; ++c) {
            xe[3 * a + c] = x[global_dof(nodes[a], c)];
        }
    }

    const auto ye = apply_element_matrix(ke, xe);
    for (int a = 0; a < 8; ++a) {
        for (int c = 0; c < 3; ++c) {
            y[global_dof(nodes[a], c)] += ye[3 * a + c];
        }
    }
}

std::uint32_t color_count(std::uint32_t n, std::uint32_t parity) {
    if (n <= parity) {
        return 0;
    }
    return 1U + (n - 1U - parity) / 2U;
}

}  // namespace

DenseMatrix assemble_dense_stiffness(const StructuredHexMesh& mesh,
                                     const Material& material) {
    const std::uint64_t ndof = mesh.dof_count();
    if (ndof > 10000) {
        throw std::invalid_argument("dense reference assembly is restricted to small validation meshes");
    }

    DenseMatrix matrix(static_cast<std::size_t>(ndof * ndof), 0.0);
    const auto ke = uniform_element_stiffness(mesh, material);
    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);

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
    const auto ke = uniform_element_stiffness(mesh, material);

    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                apply_one_element(mesh, ke, x, y, ex, ey, ez);
            }
        }
    }
    return y;
}

std::vector<double> apply_matrix_free_openmp(const StructuredHexMesh& mesh,
                                             const Material& material,
                                             const std::vector<double>& x) {
    validate_vector_size(mesh, x);
#ifndef _OPENMP
    return apply_matrix_free(mesh, material, x);
#else
    std::vector<double> y(x.size(), 0.0);
    const auto ke = uniform_element_stiffness(mesh, material);

    // HEX8 elements with the same (ex,ey,ez) parity never share a node.
    // Eight-color traversal therefore permits lock-free parallel scatter.
    for (std::uint32_t color = 0; color < 8; ++color) {
        const std::uint32_t px = color & 1U;
        const std::uint32_t py = (color >> 1U) & 1U;
        const std::uint32_t pz = (color >> 2U) & 1U;
        const std::uint32_t cx = color_count(mesh.nx, px);
        const std::uint32_t cy = color_count(mesh.ny, py);
        const std::uint32_t cz = color_count(mesh.nz, pz);
        const std::int64_t count = static_cast<std::int64_t>(cx) * cy * cz;

#pragma omp parallel for schedule(static)
        for (std::int64_t local = 0; local < count; ++local) {
            const auto ulocal = static_cast<std::uint64_t>(local);
            const std::uint32_t ix = static_cast<std::uint32_t>(ulocal % cx);
            const std::uint64_t yz = ulocal / cx;
            const std::uint32_t iy = static_cast<std::uint32_t>(yz % cy);
            const std::uint32_t iz = static_cast<std::uint32_t>(yz / cy);
            const std::uint32_t ex = px + 2U * ix;
            const std::uint32_t ey = py + 2U * iy;
            const std::uint32_t ez = pz + 2U * iz;
            apply_one_element(mesh, ke, x, y, ex, ey, ez);
        }
    }
    return y;
#endif
}

int cpu_openmp_max_threads() noexcept {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
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
