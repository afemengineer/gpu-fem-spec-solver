#pragma once

#include "gfss/cpu_gold.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace gfss {

struct CpuGoldPaddedLayoutFp32 {
    std::int64_t row_stride{0};
    std::int64_t plane_stride{0};
    std::size_t storage_nodes{0};
};

struct CpuGoldPaddedStencilFp32 {
    RegularNodeStencil<float> regular{};
    CpuGoldPaddedLayoutFp32 layout{};

    std::array<CpuGoldDiagEntryFp32, 7> diag{};
    std::array<CpuGoldEdgeXYEntryFp32, 4> edge_xy{};
    std::array<CpuGoldEdgeXZEntryFp32, 4> edge_xz{};
    std::array<CpuGoldEdgeYZEntryFp32, 4> edge_yz{};
    std::array<CpuGoldCornerEntryFp32, 8> corner{};
    std::uint8_t diag_count{0};
    std::uint8_t edge_xy_count{0};
    std::uint8_t edge_xz_count{0};
    std::uint8_t edge_yz_count{0};
    std::uint8_t corner_count{0};
};

CpuGoldPaddedLayoutFp32 make_cpu_gold_padded_layout_fp32(
    const StructuredHexMesh& mesh);

CpuGoldPaddedStencilFp32 build_cpu_gold_padded_stencil_fp32(
    const StructuredHexMesh& mesh,
    const Material& material,
    const CpuGoldPaddedLayoutFp32& layout);

void aos_to_padded_soa_fp32(const StructuredHexMesh& mesh,
                            const CpuGoldPaddedLayoutFp32& layout,
                            const float* aos,
                            float* ux,
                            float* uy,
                            float* uz);

void padded_soa_to_aos_fp32(const StructuredHexMesh& mesh,
                            const CpuGoldPaddedLayoutFp32& layout,
                            const float* ux,
                            const float* uy,
                            const float* uz,
                            float* aos);

void apply_cpu_gold_padded_soa_fp32(
    const StructuredHexMesh& mesh,
    const CpuGoldPaddedStencilFp32& stencil,
    const float* ux,
    const float* uy,
    const float* uz,
    float* yx,
    float* yy,
    float* yz);

}  // namespace gfss
