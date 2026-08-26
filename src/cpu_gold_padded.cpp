#include "gfss/cpu_gold_padded.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if GFSS_CPU_STENCIL_AVX2
#include <immintrin.h>
#endif

namespace gfss {
namespace {

constexpr int class_index(int cx, int cy, int cz) noexcept {
    return cx + 3 * (cy + 3 * cz);
}

constexpr int kInteriorClass = class_index(1, 1, 1);

int axis_class(std::int64_t coordinate, std::int64_t max_coordinate) noexcept {
    if (coordinate == 0) {
        return 0;
    }
    if (coordinate == max_coordinate) {
        return 2;
    }
    return 1;
}

bool is_aligned_32(const void* ptr) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) & 31U) == 0U;
}

void apply_scalar_node_padded(
    const StructuredHexMesh& mesh,
    const CpuGoldPaddedStencilFp32& stencil,
    std::int64_t i,
    std::int64_t j,
    std::int64_t k,
    const float* ux,
    const float* uy,
    const float* uz,
    float* yx,
    float* yy,
    float* yz) {
    const std::int64_t nx = static_cast<std::int64_t>(mesh.nx);
    const std::int64_t ny = static_cast<std::int64_t>(mesh.ny);
    const std::int64_t nz = static_cast<std::int64_t>(mesh.nz);
    const auto row_stride = stencil.layout.row_stride;
    const auto plane_stride = stencil.layout.plane_stride;

    const int cls = class_index(axis_class(i, nx), axis_class(j, ny), axis_class(k, nz));
    const auto count = stencil.regular.counts[static_cast<std::size_t>(cls)];

    float out_x = 0.0f;
    float out_y = 0.0f;
    float out_z = 0.0f;
    for (std::uint8_t e = 0; e < count; ++e) {
        const auto& entry = stencil.regular.entries[static_cast<std::size_t>(cls)][e];
        const std::int64_t ni = i + entry.dx;
        const std::int64_t nj = j + entry.dy;
        const std::int64_t nk = k + entry.dz;
        const std::size_t n = static_cast<std::size_t>(
            ni + row_stride * nj + plane_stride * nk);
        const float x0 = ux[n];
        const float x1 = uy[n];
        const float x2 = uz[n];
        const auto& b = entry.block;
        out_x += b[0] * x0 + b[1] * x1 + b[2] * x2;
        out_y += b[3] * x0 + b[4] * x1 + b[5] * x2;
        out_z += b[6] * x0 + b[7] * x1 + b[8] * x2;
    }

    const std::size_t n = static_cast<std::size_t>(
        i + row_stride * j + plane_stride * k);
    yx[n] = out_x;
    yy[n] = out_y;
    yz[n] = out_z;
}

#if GFSS_CPU_STENCIL_AVX2
void apply_interior_vector8_padded(
    const CpuGoldPaddedStencilFp32& stencil,
    std::int64_t center_node,
    const float* ux,
    const float* uy,
    const float* uz,
    float* yx,
    float* yy,
    float* yz) {
    __m256 out_x = _mm256_setzero_ps();
    __m256 out_y = _mm256_setzero_ps();
    __m256 out_z = _mm256_setzero_ps();

    for (std::uint8_t e = 0; e < stencil.diag_count; ++e) {
        const auto& entry = stencil.diag[e];
        const std::size_t n = static_cast<std::size_t>(center_node + entry.offset);
        const __m256 x0 = _mm256_loadu_ps(ux + n);
        const __m256 x1 = _mm256_loadu_ps(uy + n);
        const __m256 x2 = _mm256_loadu_ps(uz + n);
        out_x = _mm256_fmadd_ps(_mm256_set1_ps(entry.b00), x0, out_x);
        out_y = _mm256_fmadd_ps(_mm256_set1_ps(entry.b11), x1, out_y);
        out_z = _mm256_fmadd_ps(_mm256_set1_ps(entry.b22), x2, out_z);
    }

    for (std::uint8_t e = 0; e < stencil.edge_xy_count; ++e) {
        const auto& entry = stencil.edge_xy[e];
        const std::size_t n = static_cast<std::size_t>(center_node + entry.offset);
        const __m256 x0 = _mm256_loadu_ps(ux + n);
        const __m256 x1 = _mm256_loadu_ps(uy + n);
        const __m256 x2 = _mm256_loadu_ps(uz + n);
        out_x = _mm256_fmadd_ps(_mm256_set1_ps(entry.b00), x0, out_x);
        out_x = _mm256_fmadd_ps(_mm256_set1_ps(entry.b01), x1, out_x);
        out_y = _mm256_fmadd_ps(_mm256_set1_ps(entry.b10), x0, out_y);
        out_y = _mm256_fmadd_ps(_mm256_set1_ps(entry.b11), x1, out_y);
        out_z = _mm256_fmadd_ps(_mm256_set1_ps(entry.b22), x2, out_z);
    }

    for (std::uint8_t e = 0; e < stencil.edge_xz_count; ++e) {
        const auto& entry = stencil.edge_xz[e];
        const std::size_t n = static_cast<std::size_t>(center_node + entry.offset);
        const __m256 x0 = _mm256_loadu_ps(ux + n);
        const __m256 x1 = _mm256_loadu_ps(uy + n);
        const __m256 x2 = _mm256_loadu_ps(uz + n);
        out_x = _mm256_fmadd_ps(_mm256_set1_ps(entry.b00), x0, out_x);
        out_x = _mm256_fmadd_ps(_mm256_set1_ps(entry.b02), x2, out_x);
        out_y = _mm256_fmadd_ps(_mm256_set1_ps(entry.b11), x1, out_y);
        out_z = _mm256_fmadd_ps(_mm256_set1_ps(entry.b20), x0, out_z);
        out_z = _mm256_fmadd_ps(_mm256_set1_ps(entry.b22), x2, out_z);
    }

    for (std::uint8_t e = 0; e < stencil.edge_yz_count; ++e) {
        const auto& entry = stencil.edge_yz[e];
        const std::size_t n = static_cast<std::size_t>(center_node + entry.offset);
        const __m256 x0 = _mm256_loadu_ps(ux + n);
        const __m256 x1 = _mm256_loadu_ps(uy + n);
        const __m256 x2 = _mm256_loadu_ps(uz + n);
        out_x = _mm256_fmadd_ps(_mm256_set1_ps(entry.b00), x0, out_x);
        out_y = _mm256_fmadd_ps(_mm256_set1_ps(entry.b11), x1, out_y);
        out_y = _mm256_fmadd_ps(_mm256_set1_ps(entry.b12), x2, out_y);
        out_z = _mm256_fmadd_ps(_mm256_set1_ps(entry.b21), x1, out_z);
        out_z = _mm256_fmadd_ps(_mm256_set1_ps(entry.b22), x2, out_z);
    }

    for (std::uint8_t e = 0; e < stencil.corner_count; ++e) {
        const auto& entry = stencil.corner[e];
        const std::size_t n = static_cast<std::size_t>(center_node + entry.offset);
        const __m256 x0 = _mm256_loadu_ps(ux + n);
        const __m256 x1 = _mm256_loadu_ps(uy + n);
        const __m256 x2 = _mm256_loadu_ps(uz + n);
        const auto& b = entry.block;
        out_x = _mm256_fmadd_ps(_mm256_set1_ps(b[0]), x0, out_x);
        out_x = _mm256_fmadd_ps(_mm256_set1_ps(b[1]), x1, out_x);
        out_x = _mm256_fmadd_ps(_mm256_set1_ps(b[2]), x2, out_x);
        out_y = _mm256_fmadd_ps(_mm256_set1_ps(b[3]), x0, out_y);
        out_y = _mm256_fmadd_ps(_mm256_set1_ps(b[4]), x1, out_y);
        out_y = _mm256_fmadd_ps(_mm256_set1_ps(b[5]), x2, out_y);
        out_z = _mm256_fmadd_ps(_mm256_set1_ps(b[6]), x0, out_z);
        out_z = _mm256_fmadd_ps(_mm256_set1_ps(b[7]), x1, out_z);
        out_z = _mm256_fmadd_ps(_mm256_set1_ps(b[8]), x2, out_z);
    }

    const std::size_t c = static_cast<std::size_t>(center_node);
    _mm256_store_ps(yx + c, out_x);
    _mm256_store_ps(yy + c, out_y);
    _mm256_store_ps(yz + c, out_z);
}
#endif

}  // namespace

CpuGoldPaddedLayoutFp32 make_cpu_gold_padded_layout_fp32(
    const StructuredHexMesh& mesh) {
    const std::int64_t physical_row = static_cast<std::int64_t>(mesh.nx) + 1;
    const std::int64_t rows_per_plane = static_cast<std::int64_t>(mesh.ny) + 1;
    const std::int64_t planes = static_cast<std::int64_t>(mesh.nz) + 1;

    const std::int64_t row_stride = (physical_row + 15) & ~std::int64_t{15};
    const std::int64_t plane_stride = row_stride * rows_per_plane;
    if (row_stride <= 0 || plane_stride <= 0 || planes <= 0) {
        throw std::overflow_error("invalid padded CPU Gold layout");
    }

    return CpuGoldPaddedLayoutFp32{
        row_stride,
        plane_stride,
        static_cast<std::size_t>(plane_stride * planes)};
}

CpuGoldPaddedStencilFp32 build_cpu_gold_padded_stencil_fp32(
    const StructuredHexMesh& mesh,
    const Material& material,
    const CpuGoldPaddedLayoutFp32& layout) {
    CpuGoldPaddedStencilFp32 result{};
    result.layout = layout;
    result.regular = build_regular_node_stencil_fp32(mesh, material);

    const auto interior_count = result.regular.counts[static_cast<std::size_t>(kInteriorClass)];
    for (std::uint8_t e = 0; e < interior_count; ++e) {
        const auto& src = result.regular.entries[static_cast<std::size_t>(kInteriorClass)][e];
        const std::int64_t offset =
            static_cast<std::int64_t>(src.dx) +
            layout.row_stride * static_cast<std::int64_t>(src.dy) +
            layout.plane_stride * static_cast<std::int64_t>(src.dz);
        const auto& b = src.block;
        const int active_axes =
            (src.dx != 0 ? 1 : 0) +
            (src.dy != 0 ? 1 : 0) +
            (src.dz != 0 ? 1 : 0);

        if (active_axes <= 1) {
            result.diag[result.diag_count++] =
                CpuGoldDiagEntryFp32{offset, b[0], b[4], b[8]};
        } else if (src.dz == 0) {
            result.edge_xy[result.edge_xy_count++] =
                CpuGoldEdgeXYEntryFp32{offset, b[0], b[1], b[3], b[4], b[8]};
        } else if (src.dy == 0) {
            result.edge_xz[result.edge_xz_count++] =
                CpuGoldEdgeXZEntryFp32{offset, b[0], b[2], b[4], b[6], b[8]};
        } else if (src.dx == 0) {
            result.edge_yz[result.edge_yz_count++] =
                CpuGoldEdgeYZEntryFp32{offset, b[0], b[4], b[5], b[7], b[8]};
        } else {
            auto& dst = result.corner[result.corner_count++];
            dst.offset = offset;
            dst.block = b;
        }
    }

    if (result.diag_count != result.diag.size() ||
        result.edge_xy_count != result.edge_xy.size() ||
        result.edge_xz_count != result.edge_xz.size() ||
        result.edge_yz_count != result.edge_yz.size() ||
        result.corner_count != result.corner.size()) {
        throw std::runtime_error("unexpected padded regular HEX8 interior stencil topology");
    }
    return result;
}

void aos_to_padded_soa_fp32(const StructuredHexMesh& mesh,
                            const CpuGoldPaddedLayoutFp32& layout,
                            const float* aos,
                            float* ux,
                            float* uy,
                            float* uz) {
    if (aos == nullptr || ux == nullptr || uy == nullptr || uz == nullptr) {
        throw std::invalid_argument("padded AoS/SoA pointers must not be null");
    }
    std::fill(ux, ux + layout.storage_nodes, 0.0f);
    std::fill(uy, uy + layout.storage_nodes, 0.0f);
    std::fill(uz, uz + layout.storage_nodes, 0.0f);

    const std::int64_t sx = static_cast<std::int64_t>(mesh.nx) + 1;
    const std::int64_t sy = static_cast<std::int64_t>(mesh.ny) + 1;
    for (std::int64_t k = 0; k <= static_cast<std::int64_t>(mesh.nz); ++k) {
        for (std::int64_t j = 0; j <= static_cast<std::int64_t>(mesh.ny); ++j) {
            for (std::int64_t i = 0; i <= static_cast<std::int64_t>(mesh.nx); ++i) {
                const std::size_t physical = static_cast<std::size_t>(i + sx * (j + sy * k));
                const std::size_t padded = static_cast<std::size_t>(
                    i + layout.row_stride * j + layout.plane_stride * k);
                ux[padded] = aos[3 * physical + 0];
                uy[padded] = aos[3 * physical + 1];
                uz[padded] = aos[3 * physical + 2];
            }
        }
    }
}

void padded_soa_to_aos_fp32(const StructuredHexMesh& mesh,
                            const CpuGoldPaddedLayoutFp32& layout,
                            const float* ux,
                            const float* uy,
                            const float* uz,
                            float* aos) {
    if (aos == nullptr || ux == nullptr || uy == nullptr || uz == nullptr) {
        throw std::invalid_argument("padded AoS/SoA pointers must not be null");
    }
    const std::int64_t sx = static_cast<std::int64_t>(mesh.nx) + 1;
    const std::int64_t sy = static_cast<std::int64_t>(mesh.ny) + 1;
    for (std::int64_t k = 0; k <= static_cast<std::int64_t>(mesh.nz); ++k) {
        for (std::int64_t j = 0; j <= static_cast<std::int64_t>(mesh.ny); ++j) {
            for (std::int64_t i = 0; i <= static_cast<std::int64_t>(mesh.nx); ++i) {
                const std::size_t physical = static_cast<std::size_t>(i + sx * (j + sy * k));
                const std::size_t padded = static_cast<std::size_t>(
                    i + layout.row_stride * j + layout.plane_stride * k);
                aos[3 * physical + 0] = ux[padded];
                aos[3 * physical + 1] = uy[padded];
                aos[3 * physical + 2] = uz[padded];
            }
        }
    }
}

void apply_cpu_gold_padded_soa_fp32(
    const StructuredHexMesh& mesh,
    const CpuGoldPaddedStencilFp32& stencil,
    const float* ux,
    const float* uy,
    const float* uz,
    float* yx,
    float* yy,
    float* yz) {
    if (ux == nullptr || uy == nullptr || uz == nullptr ||
        yx == nullptr || yy == nullptr || yz == nullptr) {
        throw std::invalid_argument("padded CPU Gold pointers must not be null");
    }
#if GFSS_CPU_STENCIL_AVX2
    if (!is_aligned_32(ux) || !is_aligned_32(uy) || !is_aligned_32(uz) ||
        !is_aligned_32(yx) || !is_aligned_32(yy) || !is_aligned_32(yz) ||
        (stencil.layout.row_stride % 8) != 0) {
        throw std::invalid_argument("padded CPU Gold AVX2 buffers/rows must be 32-byte aligned");
    }
#endif

    const std::int64_t nx = static_cast<std::int64_t>(mesh.nx);
    const std::int64_t ny = static_cast<std::int64_t>(mesh.ny);
    const std::int64_t nz = static_cast<std::int64_t>(mesh.nz);

#if GFSS_HAS_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (std::int64_t k = 0; k <= nz; ++k) {
        for (std::int64_t j = 0; j <= ny; ++j) {
            if (k == 0 || k == nz || j == 0 || j == ny || nx < 16) {
                for (std::int64_t i = 0; i <= nx; ++i) {
                    apply_scalar_node_padded(mesh, stencil, i, j, k,
                                             ux, uy, uz, yx, yy, yz);
                }
                continue;
            }

            std::int64_t i = 0;
#if GFSS_CPU_STENCIL_AVX2
            for (; i < 8 && i <= nx; ++i) {
                apply_scalar_node_padded(mesh, stencil, i, j, k,
                                         ux, uy, uz, yx, yy, yz);
            }
            const std::int64_t last_vector_start = nx - 8;
            for (; i <= last_vector_start; i += 8) {
                const std::int64_t center =
                    i + stencil.layout.row_stride * j + stencil.layout.plane_stride * k;
                apply_interior_vector8_padded(
                    stencil, center, ux, uy, uz, yx, yy, yz);
            }
#endif
            for (; i <= nx; ++i) {
                apply_scalar_node_padded(mesh, stencil, i, j, k,
                                         ux, uy, uz, yx, yy, yz);
            }
        }
    }
}

}  // namespace gfss
