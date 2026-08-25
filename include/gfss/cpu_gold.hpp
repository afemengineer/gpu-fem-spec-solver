#pragma once

#include "gfss/cpu_stencil.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace gfss {

struct CpuGoldDiagEntryFp32 {
    std::int64_t offset{0};
    float b00{0.0f};
    float b11{0.0f};
    float b22{0.0f};
};

struct CpuGoldEdgeXYEntryFp32 {
    std::int64_t offset{0};
    float b00{0.0f};
    float b01{0.0f};
    float b10{0.0f};
    float b11{0.0f};
    float b22{0.0f};
};

struct CpuGoldEdgeXZEntryFp32 {
    std::int64_t offset{0};
    float b00{0.0f};
    float b02{0.0f};
    float b11{0.0f};
    float b20{0.0f};
    float b22{0.0f};
};

struct CpuGoldEdgeYZEntryFp32 {
    std::int64_t offset{0};
    float b00{0.0f};
    float b11{0.0f};
    float b12{0.0f};
    float b21{0.0f};
    float b22{0.0f};
};

struct CpuGoldCornerEntryFp32 {
    std::int64_t offset{0};
    std::array<float, 9> block{};
};

struct CpuGoldStencilFp32 {
    RegularNodeStencil<float> regular{};

    // Exact structural sparsity of the regular orthogonal HEX8 elasticity
    // stencil. Center + faces are diagonal-only, edges couple only their two
    // active axes, and corners retain the full 3x3 block.
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
