#pragma once

#include "gfss/cpu_stencil.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace gfss {

struct CpuGoldStencilFp32 {
    RegularNodeStencil<float> regular{};
    std::array<std::int64_t, 27> interior_node_offsets{};
    std::array<std::array<float, 27>, 9> interior_coeff{};
    std::uint8_t interior_count{0};
};

CpuGoldStencilFp32 build_cpu_gold_stencil_fp32(
    const StructuredHexMesh& mesh,
    const Material& material);

void aos_to_soa_fp32(const float* aos,
                     std::size_t node_count,
                     float* ux,
                     float* uy,
                     float* uz);

void soa_to_aos_fp32(const float* ux,
                     const float* uy,
                     const float* uz,
                     std::size_t node_count,
                     float* aos);

bool cpu_gold_avx2_enabled() noexcept;

void apply_cpu_gold_soa_fp32(const StructuredHexMesh& mesh,
                             const CpuGoldStencilFp32& stencil,
                             const float* ux,
                             const float* uy,
                             const float* uz,
                             float* yx,
                             float* yy,
                             float* yz);

}  // namespace gfss
