#include "gfss/cpu_gold.hpp"

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
        const std::int64_t node = ni + sx * (nj + sy * nk);
        const std::size_t n = static_cast<std::size_t>(node);
        const float x0 = ux[n];
        const float x1 = uy[n];
        const float x2 = uz[n];
        const auto& b = entry.block;
        out_x += b[0] * x0 + b[1] * x1 + b[2] * x2;
        out_y += b[3] * x0 + b[4] * x1 + b[5] * x2;
        out_z += b[6] * x0 + b[7] * x1 + b[8] * x2;
    }

    const std::int64_t node = i + sx * (j + sy * k);
    const std::size_t n = static_cast<std::size_t>(node);
    yx[n] = out_x;
    yy[n] = out_y;
    yz[n] = out_z;
}

#if GFSS_CPU_STENCIL_AVX2
void apply_interior_vector8(const CpuGoldStencilFp32& stencil,
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

    for (std::uint8_t e = 0; e < stencil.interior_count; ++e) {
        const std::int64_t neighbor = center_node + stencil.interior_node_offsets[e];
        const std::size_t n = static_cast<std::size_t>(neighbor);
        const __m256 x0 = _mm256_loadu_ps(ux + n);
        const __m256 x1 = _mm256_loadu_ps(uy + n);
        const __m256 x2 = _mm256_loadu_ps(uz + n);

        const __m256 b00 = _mm256_set1_ps(stencil.interior_coeff[0][e]);
        const __m256 b01 = _mm256_set1_ps(stencil.interior_coeff[1][e]);
        const __m256 b02 = _mm256_set1_ps(stencil.interior_coeff[2][e]);
        const __m256 b10 = _mm256_set1_ps(stencil.interior_coeff[3][e]);
        const __m256 b11 = _mm256_set1_ps(stencil.interior_coeff[4][e]);
        const __m256 b12 = _mm256_set1_ps(stencil.interior_coeff[5][e]);
        const __m256 b20 = _mm256_set1_ps(stencil.interior_coeff[6][e]);
        const __m256 b21 = _mm256_set1_ps(stencil.interior_coeff[7][e]);
        const __m256 b22 = _mm256_set1_ps(stencil.interior_coeff[8][e]);

        out_x = _mm256_fmadd_ps(b00, x0, out_x);
        out_x = _mm256_fmadd_ps(b01, x1, out_x);
        out_x = _mm256_fmadd_ps(b02, x2, out_x);
        out_y = _mm256_fmadd_ps(b10, x0, out_y);
        out_y = _mm256_fmadd_ps(b11, x1, out_y);
        out_y = _mm256_fmadd_ps(b12, x2, out_y);
        out_z = _mm256_fmadd_ps(b20, x0, out_z);
        out_z = _mm256_fmadd_ps(b21, x1, out_z);
        out_z = _mm256_fmadd_ps(b22, x2, out_z);
    }

    const std::size_t c = static_cast<std::size_t>(center_node);
    _mm256_storeu_ps(yx + c, out_x);
    _mm256_storeu_ps(yy + c, out_y);
    _mm256_storeu_ps(yz + c, out_z);
}
#endif

}  // namespace

CpuGoldStencilFp32 build_cpu_gold_stencil_fp32(
    const StructuredHexMesh& mesh,
    const Material& material) {
    CpuGoldStencilFp32 result{};
    result.regular = build_regular_node_stencil_fp32(mesh, material);

    const std::int64_t sx = static_cast<std::int64_t>(mesh.nx) + 1;
    const std::int64_t sy = static_cast<std::int64_t>(mesh.ny) + 1;
    result.interior_count = result.regular.counts[static_cast<std::size_t>(kInteriorClass)];

    for (std::uint8_t e = 0; e < result.interior_count; ++e) {
        const auto& entry = result.regular.entries[static_cast<std::size_t>(kInteriorClass)][e];
        result.interior_node_offsets[e] =
            static_cast<std::int64_t>(entry.dx) +
            sx * (static_cast<std::int64_t>(entry.dy) +
                  sy * static_cast<std::int64_t>(entry.dz));
        for (std::size_t q = 0; q < 9; ++q) {
            result.interior_coeff[q][e] = entry.block[q];
        }
    }

    return result;
}

void aos_to_soa_fp32(const float* aos,
                     std::size_t node_count,
                     float* ux,
                     float* uy,
                     float* uz) {
    if (aos == nullptr || ux == nullptr || uy == nullptr || uz == nullptr) {
        throw std::invalid_argument("AoS/SoA pointers must not be null");
    }
    for (std::size_t n = 0; n < node_count; ++n) {
        ux[n] = aos[3 * n + 0];
        uy[n] = aos[3 * n + 1];
        uz[n] = aos[3 * n + 2];
    }
}

void soa_to_aos_fp32(const float* ux,
                     const float* uy,
                     const float* uz,
                     std::size_t node_count,
                     float* aos) {
    if (aos == nullptr || ux == nullptr || uy == nullptr || uz == nullptr) {
        throw std::invalid_argument("AoS/SoA pointers must not be null");
    }
    for (std::size_t n = 0; n < node_count; ++n) {
        aos[3 * n + 0] = ux[n];
        aos[3 * n + 1] = uy[n];
        aos[3 * n + 2] = uz[n];
    }
}

bool cpu_gold_avx2_enabled() noexcept {
#if GFSS_CPU_STENCIL_AVX2
    return true;
#else
    return false;
#endif
}

void apply_cpu_gold_soa_fp32(const StructuredHexMesh& mesh,
                             const CpuGoldStencilFp32& stencil,
                             const float* ux,
                             const float* uy,
                             const float* uz,
                             float* yx,
                             float* yy,
                             float* yz) {
    if (ux == nullptr || uy == nullptr || uz == nullptr ||
        yx == nullptr || yy == nullptr || yz == nullptr) {
        throw std::invalid_argument("CPU Gold input/output pointers must not be null");
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
                apply_interior_vector8(stencil, center, ux, uy, uz, yx, yy, yz);
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
