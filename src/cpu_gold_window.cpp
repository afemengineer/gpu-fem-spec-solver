#include "gfss/cpu_gold_window.hpp"

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

void apply_scalar_node_soa(const StructuredHexMesh& mesh,
                           const RegularNodeStencil<float>& stencil,
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
    const std::int64_t sx = nx + 1;
    const std::int64_t sy = ny + 1;

    const int cls = class_index(axis_class(i, nx), axis_class(j, ny), axis_class(k, nz));
    const auto count = stencil.counts[static_cast<std::size_t>(cls)];

    float out_x = 0.0f;
    float out_y = 0.0f;
    float out_z = 0.0f;
    for (std::uint8_t e = 0; e < count; ++e) {
        const auto& entry = stencil.entries[static_cast<std::size_t>(cls)][e];
        const std::int64_t ni = i + entry.dx;
        const std::int64_t nj = j + entry.dy;
        const std::int64_t nk = k + entry.dz;
        const std::size_t n = static_cast<std::size_t>(ni + sx * (nj + sy * nk));
        const float x0 = ux[n];
        const float x1 = uy[n];
        const float x2 = uz[n];
        const auto& b = entry.block;
        out_x += b[0] * x0 + b[1] * x1 + b[2] * x2;
        out_y += b[3] * x0 + b[4] * x1 + b[5] * x2;
        out_z += b[6] * x0 + b[7] * x1 + b[8] * x2;
    }

    const std::size_t n = static_cast<std::size_t>(i + sx * (j + sy * k));
    yx[n] = out_x;
    yy[n] = out_y;
    yz[n] = out_z;
}

#if GFSS_CPU_STENCIL_AVX2

struct Window3 {
    __m256 minus;
    __m256 center;
    __m256 plus;
};

inline Window3 load_window3(const float* p) noexcept {
    // Two overlapping 256-bit loads cover ten consecutive values needed by
    // the dx=-1,0,+1 vectors for eight adjacent output nodes. The two shifted
    // vectors are assembled in registers rather than issued as extra loads.
    const __m256 minus = _mm256_loadu_ps(p - 1);
    const __m256 upper = _mm256_loadu_ps(p + 7);
    const __m256 bridge = _mm256_permute2f128_ps(minus, upper, 0x21);
    const __m256i minus_i = _mm256_castps_si256(minus);
    const __m256i bridge_i = _mm256_castps_si256(bridge);
    const __m256 center = _mm256_castsi256_ps(_mm256_alignr_epi8(bridge_i, minus_i, 4));
    const __m256 plus = _mm256_castsi256_ps(_mm256_alignr_epi8(bridge_i, minus_i, 8));
    return Window3{minus, center, plus};
}

inline void accumulate_diag(const std::array<float, 9>& b,
                            __m256 x0,
                            __m256 x1,
                            __m256 x2,
                            __m256& out_x,
                            __m256& out_y,
                            __m256& out_z) noexcept {
    out_x = _mm256_fmadd_ps(_mm256_set1_ps(b[0]), x0, out_x);
    out_y = _mm256_fmadd_ps(_mm256_set1_ps(b[4]), x1, out_y);
    out_z = _mm256_fmadd_ps(_mm256_set1_ps(b[8]), x2, out_z);
}

inline void accumulate_edge_xy(const std::array<float, 9>& b,
                               __m256 x0,
                               __m256 x1,
                               __m256 x2,
                               __m256& out_x,
                               __m256& out_y,
                               __m256& out_z) noexcept {
    out_x = _mm256_fmadd_ps(_mm256_set1_ps(b[0]), x0, out_x);
    out_x = _mm256_fmadd_ps(_mm256_set1_ps(b[1]), x1, out_x);
    out_y = _mm256_fmadd_ps(_mm256_set1_ps(b[3]), x0, out_y);
    out_y = _mm256_fmadd_ps(_mm256_set1_ps(b[4]), x1, out_y);
    out_z = _mm256_fmadd_ps(_mm256_set1_ps(b[8]), x2, out_z);
}

inline void accumulate_edge_xz(const std::array<float, 9>& b,
                               __m256 x0,
                               __m256 x1,
                               __m256 x2,
                               __m256& out_x,
                               __m256& out_y,
                               __m256& out_z) noexcept {
    out_x = _mm256_fmadd_ps(_mm256_set1_ps(b[0]), x0, out_x);
    out_x = _mm256_fmadd_ps(_mm256_set1_ps(b[2]), x2, out_x);
    out_y = _mm256_fmadd_ps(_mm256_set1_ps(b[4]), x1, out_y);
    out_z = _mm256_fmadd_ps(_mm256_set1_ps(b[6]), x0, out_z);
    out_z = _mm256_fmadd_ps(_mm256_set1_ps(b[8]), x2, out_z);
}

inline void accumulate_edge_yz(const std::array<float, 9>& b,
                               __m256 x0,
                               __m256 x1,
                               __m256 x2,
                               __m256& out_x,
                               __m256& out_y,
                               __m256& out_z) noexcept {
    out_x = _mm256_fmadd_ps(_mm256_set1_ps(b[0]), x0, out_x);
    out_y = _mm256_fmadd_ps(_mm256_set1_ps(b[4]), x1, out_y);
    out_y = _mm256_fmadd_ps(_mm256_set1_ps(b[5]), x2, out_y);
    out_z = _mm256_fmadd_ps(_mm256_set1_ps(b[7]), x1, out_z);
    out_z = _mm256_fmadd_ps(_mm256_set1_ps(b[8]), x2, out_z);
}

inline void accumulate_full(const std::array<float, 9>& b,
                            __m256 x0,
                            __m256 x1,
                            __m256 x2,
                            __m256& out_x,
                            __m256& out_y,
                            __m256& out_z) noexcept {
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

void apply_interior_window8(const CpuGoldWindowStencilFp32& stencil,
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

    for (const auto& plane : stencil.planes) {
        const std::size_t base = static_cast<std::size_t>(center_node + plane.center_offset);
        const Window3 wx = load_window3(ux + base);
        const Window3 wy = load_window3(uy + base);
        const Window3 wz = load_window3(uz + base);

        const bool y0 = plane.dy == 0;
        const bool z0 = plane.dz == 0;
        if (y0 && z0) {
            accumulate_diag(plane.block[0], wx.minus, wy.minus, wz.minus, out_x, out_y, out_z);
            accumulate_diag(plane.block[1], wx.center, wy.center, wz.center, out_x, out_y, out_z);
            accumulate_diag(plane.block[2], wx.plus, wy.plus, wz.plus, out_x, out_y, out_z);
        } else if (z0) {
            accumulate_edge_xy(plane.block[0], wx.minus, wy.minus, wz.minus, out_x, out_y, out_z);
            accumulate_diag(plane.block[1], wx.center, wy.center, wz.center, out_x, out_y, out_z);
            accumulate_edge_xy(plane.block[2], wx.plus, wy.plus, wz.plus, out_x, out_y, out_z);
        } else if (y0) {
            accumulate_edge_xz(plane.block[0], wx.minus, wy.minus, wz.minus, out_x, out_y, out_z);
            accumulate_diag(plane.block[1], wx.center, wy.center, wz.center, out_x, out_y, out_z);
            accumulate_edge_xz(plane.block[2], wx.plus, wy.plus, wz.plus, out_x, out_y, out_z);
        } else {
            accumulate_full(plane.block[0], wx.minus, wy.minus, wz.minus, out_x, out_y, out_z);
            accumulate_edge_yz(plane.block[1], wx.center, wy.center, wz.center, out_x, out_y, out_z);
            accumulate_full(plane.block[2], wx.plus, wy.plus, wz.plus, out_x, out_y, out_z);
        }
    }

    const std::size_t c = static_cast<std::size_t>(center_node);
    _mm256_storeu_ps(yx + c, out_x);
    _mm256_storeu_ps(yy + c, out_y);
    _mm256_storeu_ps(yz + c, out_z);
}

#endif

}  // namespace

CpuGoldWindowStencilFp32 build_cpu_gold_window_stencil_fp32(
    const StructuredHexMesh& mesh,
    const Material& material) {
    CpuGoldWindowStencilFp32 result{};
    result.regular = build_regular_node_stencil_fp32(mesh, material);

    const std::int64_t sx = static_cast<std::int64_t>(mesh.nx) + 1;
    const std::int64_t sy = static_cast<std::int64_t>(mesh.ny) + 1;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            const std::size_t p = static_cast<std::size_t>((dz + 1) * 3 + (dy + 1));
            result.planes[p].dy = static_cast<std::int8_t>(dy);
            result.planes[p].dz = static_cast<std::int8_t>(dz);
            result.planes[p].center_offset = sx * (dy + sy * dz);
        }
    }

    const auto count = result.regular.counts[static_cast<std::size_t>(kInteriorClass)];
    for (std::uint8_t e = 0; e < count; ++e) {
        const auto& src = result.regular.entries[static_cast<std::size_t>(kInteriorClass)][e];
        const std::size_t p = static_cast<std::size_t>((src.dz + 1) * 3 + (src.dy + 1));
        const std::size_t xslot = static_cast<std::size_t>(src.dx + 1);
        result.planes[p].block[xslot] = src.block;
    }

    return result;
}

void apply_cpu_gold_window_soa_fp32(const StructuredHexMesh& mesh,
                                    const CpuGoldWindowStencilFp32& stencil,
                                    const float* ux,
                                    const float* uy,
                                    const float* uz,
                                    float* yx,
                                    float* yy,
                                    float* yz) {
    if (ux == nullptr || uy == nullptr || uz == nullptr ||
        yx == nullptr || yy == nullptr || yz == nullptr) {
        throw std::invalid_argument("CPU Gold window input/output pointers must not be null");
    }

    const std::int64_t nx = static_cast<std::int64_t>(mesh.nx);
    const std::int64_t ny = static_cast<std::int64_t>(mesh.ny);
    const std::int64_t nz = static_cast<std::int64_t>(mesh.nz);
    const std::int64_t sx = nx + 1;
    const std::int64_t sy = ny + 1;

#if GFSS_HAS_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (std::int64_t k = 0; k <= nz; ++k) {
        for (std::int64_t j = 0; j <= ny; ++j) {
            if (k == 0 || k == nz || j == 0 || j == ny || nx < 2) {
                for (std::int64_t i = 0; i <= nx; ++i) {
                    apply_scalar_node_soa(mesh, stencil.regular, i, j, k,
                                          ux, uy, uz, yx, yy, yz);
                }
                continue;
            }

            apply_scalar_node_soa(mesh, stencil.regular, 0, j, k,
                                  ux, uy, uz, yx, yy, yz);

            std::int64_t i = 1;
#if GFSS_CPU_STENCIL_AVX2
            const std::int64_t last_vector_start = nx - 8;
            for (; i <= last_vector_start; i += 8) {
                const std::int64_t center = i + sx * (j + sy * k);
                apply_interior_window8(stencil, center, ux, uy, uz, yx, yy, yz);
            }
#endif
            for (; i < nx; ++i) {
                apply_scalar_node_soa(mesh, stencil.regular, i, j, k,
                                      ux, uy, uz, yx, yy, yz);
            }

            apply_scalar_node_soa(mesh, stencil.regular, nx, j, k,
                                  ux, uy, uz, yx, yy, yz);
        }
    }
}

}  // namespace gfss
