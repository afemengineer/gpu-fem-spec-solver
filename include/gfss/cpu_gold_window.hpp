#pragma once

#include "gfss/cpu_stencil.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <array>
#include <cstdint>

namespace gfss {

struct CpuGoldWindowPlaneFp32 {
    // Offset of the (dx=0, dy, dz) node relative to the output center node.
    std::int64_t center_offset{0};
    std::int8_t dy{0};
    std::int8_t dz{0};

    // Blocks for dx=-1,0,+1 respectively.
    std::array<std::array<float, 9>, 3> block{};
};

struct CpuGoldWindowStencilFp32 {
    RegularNodeStencil<float> regular{};
    std::array<CpuGoldWindowPlaneFp32, 9> planes{};
};

CpuGoldWindowStencilFp32 build_cpu_gold_window_stencil_fp32(
    const StructuredHexMesh& mesh,
    const Material& material);

void apply_cpu_gold_window_soa_fp32(const StructuredHexMesh& mesh,
                                    const CpuGoldWindowStencilFp32& stencil,
                                    const float* ux,
                                    const float* uy,
                                    const float* uz,
                                    float* yx,
                                    float* yy,
                                    float* yz);

}  // namespace gfss
