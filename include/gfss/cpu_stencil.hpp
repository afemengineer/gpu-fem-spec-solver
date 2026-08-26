#pragma once

#include "gfss/structured_hex_mesh.hpp"

#include <array>
#include <cstdint>

namespace gfss {

template <typename T>
struct NodeStencilEntry {
    std::int8_t dx{0};
    std::int8_t dy{0};
    std::int8_t dz{0};
    std::array<T, 9> block{};
};

template <typename T>
struct RegularNodeStencil {
    // 27 boundary classes (low/interior/high on x,y,z), each with at most
    // 27 neighboring nodes. Every entry stores one 3x3 displacement block.
    std::array<std::array<NodeStencilEntry<T>, 27>, 27> entries{};
    std::array<std::uint8_t, 27> counts{};
};

RegularNodeStencil<double> build_regular_node_stencil_fp64(
    const StructuredHexMesh& mesh,
    const Material& material);

RegularNodeStencil<float> build_regular_node_stencil_fp32(
    const StructuredHexMesh& mesh,
    const Material& material);

void apply_node_stencil_openmp(const StructuredHexMesh& mesh,
                               const RegularNodeStencil<double>& stencil,
                               const double* x,
                               double* y);

void apply_node_stencil_openmp(const StructuredHexMesh& mesh,
                               const RegularNodeStencil<float>& stencil,
                               const float* x,
                               float* y);

}  // namespace gfss
