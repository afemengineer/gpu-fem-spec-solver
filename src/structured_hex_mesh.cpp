#include "gfss/structured_hex_mesh.hpp"

#include <stdexcept>

namespace gfss {

std::uint64_t StructuredHexMesh::node_count() const noexcept {
    return static_cast<std::uint64_t>(nx + 1) *
           static_cast<std::uint64_t>(ny + 1) *
           static_cast<std::uint64_t>(nz + 1);
}

std::uint64_t StructuredHexMesh::element_count() const noexcept {
    return static_cast<std::uint64_t>(nx) *
           static_cast<std::uint64_t>(ny) *
           static_cast<std::uint64_t>(nz);
}

std::uint64_t StructuredHexMesh::dof_count() const noexcept {
    return 3ULL * node_count();
}

std::uint64_t StructuredHexMesh::node_index(std::uint32_t i,
                                             std::uint32_t j,
                                             std::uint32_t k) const {
    if (i > nx || j > ny || k > nz) {
        throw std::out_of_range("structured mesh node index out of range");
    }
    const std::uint64_t sx = static_cast<std::uint64_t>(nx + 1);
    const std::uint64_t sy = static_cast<std::uint64_t>(ny + 1);
    return static_cast<std::uint64_t>(i) +
           sx * (static_cast<std::uint64_t>(j) + sy * static_cast<std::uint64_t>(k));
}

Vec3 StructuredHexMesh::node_coordinate(std::uint32_t i,
                                         std::uint32_t j,
                                         std::uint32_t k) const {
    if (nx == 0 || ny == 0 || nz == 0) {
        throw std::invalid_argument("structured mesh dimensions must be nonzero");
    }
    (void)node_index(i, j, k);
    return {{
        lx * static_cast<double>(i) / static_cast<double>(nx),
        ly * static_cast<double>(j) / static_cast<double>(ny),
        lz * static_cast<double>(k) / static_cast<double>(nz),
    }};
}

std::array<std::uint64_t, 8> StructuredHexMesh::element_nodes(std::uint32_t ex,
                                                               std::uint32_t ey,
                                                               std::uint32_t ez) const {
    if (ex >= nx || ey >= ny || ez >= nz) {
        throw std::out_of_range("structured mesh element index out of range");
    }
    return {{
        node_index(ex,     ey,     ez),
        node_index(ex + 1, ey,     ez),
        node_index(ex + 1, ey + 1, ez),
        node_index(ex,     ey + 1, ez),
        node_index(ex,     ey,     ez + 1),
        node_index(ex + 1, ey,     ez + 1),
        node_index(ex + 1, ey + 1, ez + 1),
        node_index(ex,     ey + 1, ez + 1),
    }};
}

Hex8Coordinates StructuredHexMesh::element_coordinates(std::uint32_t ex,
                                                        std::uint32_t ey,
                                                        std::uint32_t ez) const {
    if (ex >= nx || ey >= ny || ez >= nz) {
        throw std::out_of_range("structured mesh element index out of range");
    }
    return {{
        node_coordinate(ex,     ey,     ez),
        node_coordinate(ex + 1, ey,     ez),
        node_coordinate(ex + 1, ey + 1, ez),
        node_coordinate(ex,     ey + 1, ez),
        node_coordinate(ex,     ey,     ez + 1),
        node_coordinate(ex + 1, ey,     ez + 1),
        node_coordinate(ex + 1, ey + 1, ez + 1),
        node_coordinate(ex,     ey + 1, ez + 1),
    }};
}

}  // namespace gfss
