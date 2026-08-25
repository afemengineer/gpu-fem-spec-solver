#pragma once

#include "gfss/hex8.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace gfss {

struct StructuredHexMesh {
    std::uint32_t nx{1};
    std::uint32_t ny{1};
    std::uint32_t nz{1};
    double lx{1.0};
    double ly{1.0};
    double lz{1.0};

    std::uint64_t node_count() const noexcept;
    std::uint64_t element_count() const noexcept;
    std::uint64_t dof_count() const noexcept;

    std::uint64_t node_index(std::uint32_t i,
                             std::uint32_t j,
                             std::uint32_t k) const;
    Vec3 node_coordinate(std::uint32_t i,
                         std::uint32_t j,
                         std::uint32_t k) const;
    std::array<std::uint64_t, 8> element_nodes(std::uint32_t ex,
                                                std::uint32_t ey,
                                                std::uint32_t ez) const;
    Hex8Coordinates element_coordinates(std::uint32_t ex,
                                        std::uint32_t ey,
                                        std::uint32_t ez) const;
};

}  // namespace gfss
