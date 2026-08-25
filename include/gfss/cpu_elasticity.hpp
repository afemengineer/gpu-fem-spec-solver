#pragma once

#include "gfss/structured_hex_mesh.hpp"

#include <cstdint>
#include <vector>

namespace gfss {

using DenseMatrix = std::vector<double>;

DenseMatrix assemble_dense_stiffness(const StructuredHexMesh& mesh,
                                     const Material& material);

std::vector<double> apply_matrix_free(const StructuredHexMesh& mesh,
                                      const Material& material,
                                      const std::vector<double>& x);

std::vector<std::uint64_t> clamped_x0_dofs(const StructuredHexMesh& mesh);

void apply_symmetric_dirichlet(DenseMatrix& matrix,
                               std::vector<double>& rhs,
                               const std::vector<std::uint64_t>& constrained_dofs);

std::vector<double> apply_clamped_x0_matrix_free(const StructuredHexMesh& mesh,
                                                  const Material& material,
                                                  const std::vector<double>& x);

}  // namespace gfss
