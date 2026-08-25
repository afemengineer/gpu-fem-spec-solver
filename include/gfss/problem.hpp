#pragma once

#include <cstdint>
#include <stdexcept>

namespace gfss {

struct StructuredHexProblem {
    std::uint64_t nx = 32;
    std::uint64_t ny = 16;
    std::uint64_t nz = 16;

    double lx = 1.0;
    double ly = 0.2;
    double lz = 0.2;

    double youngs_modulus = 210.0e9;
    double poisson_ratio = 0.30;

    [[nodiscard]] std::uint64_t element_count() const {
        return nx * ny * nz;
    }

    [[nodiscard]] std::uint64_t node_count() const {
        return (nx + 1) * (ny + 1) * (nz + 1);
    }

    [[nodiscard]] std::uint64_t dof_count() const {
        return 3ULL * node_count();
    }

    void validate() const {
        if (nx == 0 || ny == 0 || nz == 0) {
            throw std::invalid_argument("mesh dimensions must be positive");
        }
        if (lx <= 0.0 || ly <= 0.0 || lz <= 0.0) {
            throw std::invalid_argument("physical dimensions must be positive");
        }
        if (youngs_modulus <= 0.0) {
            throw std::invalid_argument("Young's modulus must be positive");
        }
        if (poisson_ratio <= -1.0 || poisson_ratio >= 0.5) {
            throw std::invalid_argument("Poisson ratio must lie in (-1, 0.5)");
        }
    }
};

}  // namespace gfss
