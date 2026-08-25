#include "gfss/cpu_stencil.hpp"

#include "gfss/hex8.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace gfss {
namespace {

constexpr std::array<std::array<int, 3>, 8> kLocalNodePos{{
    {{0, 0, 0}},
    {{1, 0, 0}},
    {{1, 1, 0}},
    {{0, 1, 0}},
    {{0, 0, 1}},
    {{1, 0, 1}},
    {{1, 1, 1}},
    {{0, 1, 1}},
}};

int class_index(int cx, int cy, int cz) {
    return cx + 3 * (cy + 3 * cz);
}

int offset_index(int dx, int dy, int dz) {
    return (dx + 1) + 3 * ((dy + 1) + 3 * (dz + 1));
}

bool local_node_allowed_for_class(int local_node, int cx, int cy, int cz) {
    const auto& p = kLocalNodePos[static_cast<std::size_t>(local_node)];
    const int cls[3] = {cx, cy, cz};
    for (int axis = 0; axis < 3; ++axis) {
        if (cls[axis] == 0 && p[axis] != 0) {
            return false;
        }
        if (cls[axis] == 2 && p[axis] != 1) {
            return false;
        }
    }
    return true;
}

RegularNodeStencil<double> build_stencil_fp64(const StructuredHexMesh& mesh,
                                              const Material& material) {
    if (mesh.nx == 0 || mesh.ny == 0 || mesh.nz == 0) {
        throw std::invalid_argument("regular node stencil requires non-empty mesh dimensions");
    }

    const auto ke = hex8_stiffness(mesh.element_coordinates(0, 0, 0), material);
    RegularNodeStencil<double> result{};

    for (int cz = 0; cz < 3; ++cz) {
        for (int cy = 0; cy < 3; ++cy) {
            for (int cx = 0; cx < 3; ++cx) {
                std::array<std::array<double, 9>, 27> blocks{};

                for (int a = 0; a < 8; ++a) {
                    if (!local_node_allowed_for_class(a, cx, cy, cz)) {
                        continue;
                    }
                    const auto& pa = kLocalNodePos[static_cast<std::size_t>(a)];
                    for (int b = 0; b < 8; ++b) {
                        const auto& pb = kLocalNodePos[static_cast<std::size_t>(b)];
                        const int dx = pb[0] - pa[0];
                        const int dy = pb[1] - pa[1];
                        const int dz = pb[2] - pa[2];
                        auto& block = blocks[static_cast<std::size_t>(offset_index(dx, dy, dz))];

                        for (int ca = 0; ca < 3; ++ca) {
                            for (int cb = 0; cb < 3; ++cb) {
                                block[static_cast<std::size_t>(3 * ca + cb)] +=
                                    ke[3 * a + ca][3 * b + cb];
                            }
                        }
                    }
                }

                const int cls = class_index(cx, cy, cz);
                std::uint8_t count = 0;
                for (int oi = 0; oi < 27; ++oi) {
                    const auto& block = blocks[static_cast<std::size_t>(oi)];
                    bool nonzero = false;
                    for (double value : block) {
                        if (value != 0.0) {
                            nonzero = true;
                            break;
                        }
                    }
                    if (!nonzero) {
                        continue;
                    }

                    const int ox = (oi % 3) - 1;
                    const int oy = ((oi / 3) % 3) - 1;
                    const int oz = (oi / 9) - 1;
                    auto& entry = result.entries[static_cast<std::size_t>(cls)][count];
                    entry.dx = static_cast<std::int8_t>(ox);
                    entry.dy = static_cast<std::int8_t>(oy);
                    entry.dz = static_cast<std::int8_t>(oz);
                    entry.block = block;
                    ++count;
                }
                result.counts[static_cast<std::size_t>(cls)] = count;
            }
        }
    }

    return result;
}

int axis_class(std::int64_t coordinate, std::int64_t max_coordinate) {
    if (coordinate == 0) {
        return 0;
    }
    if (coordinate == max_coordinate) {
        return 2;
    }
    return 1;
}

template <typename T>
void apply_impl(const StructuredHexMesh& mesh,
                const RegularNodeStencil<T>& stencil,
                const T* x,
                T* y) {
    if (x == nullptr || y == nullptr) {
        throw std::invalid_argument("node stencil input/output pointer must not be null");
    }

    const std::int64_t nx = static_cast<std::int64_t>(mesh.nx);
    const std::int64_t ny = static_cast<std::int64_t>(mesh.ny);
    const std::int64_t nz = static_cast<std::int64_t>(mesh.nz);
    const std::int64_t sx = nx + 1;
    const std::int64_t sy = ny + 1;

#if GFSS_HAS_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t k = 0; k <= nz; ++k) {
        for (std::int64_t j = 0; j <= ny; ++j) {
            for (std::int64_t i = 0; i <= nx; ++i) {
                const int cls = class_index(axis_class(i, nx),
                                            axis_class(j, ny),
                                            axis_class(k, nz));
                const auto count = stencil.counts[static_cast<std::size_t>(cls)];

                T y0 = T{};
                T y1 = T{};
                T y2 = T{};
                for (std::uint8_t e = 0; e < count; ++e) {
                    const auto& entry = stencil.entries[static_cast<std::size_t>(cls)][e];
                    const std::int64_t ni = i + entry.dx;
                    const std::int64_t nj = j + entry.dy;
                    const std::int64_t nk = k + entry.dz;
                    const std::int64_t node = ni + sx * (nj + sy * nk);
                    const std::size_t base = static_cast<std::size_t>(3 * node);
                    const T x0 = x[base + 0];
                    const T x1 = x[base + 1];
                    const T x2 = x[base + 2];
                    const auto& b = entry.block;
                    y0 += b[0] * x0 + b[1] * x1 + b[2] * x2;
                    y1 += b[3] * x0 + b[4] * x1 + b[5] * x2;
                    y2 += b[6] * x0 + b[7] * x1 + b[8] * x2;
                }

                const std::int64_t node = i + sx * (j + sy * k);
                const std::size_t base = static_cast<std::size_t>(3 * node);
                y[base + 0] = y0;
                y[base + 1] = y1;
                y[base + 2] = y2;
            }
        }
    }
}

}  // namespace

RegularNodeStencil<double> build_regular_node_stencil_fp64(
    const StructuredHexMesh& mesh,
    const Material& material) {
    return build_stencil_fp64(mesh, material);
}

RegularNodeStencil<float> build_regular_node_stencil_fp32(
    const StructuredHexMesh& mesh,
    const Material& material) {
    const auto source = build_stencil_fp64(mesh, material);
    RegularNodeStencil<float> result{};
    result.counts = source.counts;
    for (std::size_t cls = 0; cls < source.entries.size(); ++cls) {
        for (std::size_t e = 0; e < source.counts[cls]; ++e) {
            result.entries[cls][e].dx = source.entries[cls][e].dx;
            result.entries[cls][e].dy = source.entries[cls][e].dy;
            result.entries[cls][e].dz = source.entries[cls][e].dz;
            for (std::size_t q = 0; q < 9; ++q) {
                result.entries[cls][e].block[q] =
                    static_cast<float>(source.entries[cls][e].block[q]);
            }
        }
    }
    return result;
}

void apply_node_stencil_openmp(const StructuredHexMesh& mesh,
                               const RegularNodeStencil<double>& stencil,
                               const double* x,
                               double* y) {
    apply_impl(mesh, stencil, x, y);
}

void apply_node_stencil_openmp(const StructuredHexMesh& mesh,
                               const RegularNodeStencil<float>& stencil,
                               const float* x,
                               float* y) {
    apply_impl(mesh, stencil, x, y);
}

}  // namespace gfss
