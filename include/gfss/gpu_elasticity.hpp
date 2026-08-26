#pragma once

#include "gfss/hex8.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <vector>

namespace gfss {

struct CudaOperatorTiming {
    double best_ms{0.0};
    double median_ms{0.0};
    double mean_ms{0.0};
    double p95_ms{0.0};

    double best_zero_ms{0.0};
    double median_zero_ms{0.0};
    double mean_zero_ms{0.0};

    double best_kernel_ms{0.0};
    double median_kernel_ms{0.0};
    double mean_kernel_ms{0.0};
};

struct CudaOperatorResult {
    std::vector<float> y;
    CudaOperatorTiming timing;
    std::size_t device_bytes{0};
};

CudaOperatorResult apply_matrix_free_cuda_atomic(const StructuredHexMesh& mesh,
                                                 const Material& material,
                                                 const std::vector<float>& x,
                                                 int repeats = 5);

// Structured-Q1 performance path: one CUDA thread owns one output node, reads
// the exact 27-class regular node stencil, and writes three SoA components.
// There are no atomics and no output memset in the timed region. AoS<->SoA
// host conversion, allocation, H2D, and D2H are setup/audit work and excluded.
CudaOperatorResult apply_node_stencil_cuda_soa(const StructuredHexMesh& mesh,
                                               const Material& material,
                                               const std::vector<float>& x,
                                               int repeats = 5,
                                               int threads_per_block = 256);

// Same mathematical node stencil with a topology-aware 3D CUDA launch.
// x maps directly to a warp lane, so the hot path avoids integer div/mod used
// to decode a linear node id. Interior nodes use a fixed 27-entry stencil with
// precomputed linear neighbor offsets; exact 27-class handling remains for
// boundary nodes. block_y controls 32 x block_y threads per block.
CudaOperatorResult apply_node_stencil_cuda_gold3d(const StructuredHexMesh& mesh,
                                                  const Material& material,
                                                  const std::vector<float>& x,
                                                  int repeats = 5,
                                                  int block_y = 8);

// Shared-memory structured stencil: a 32 x block_y x block_z output tile
// cooperatively stages one-node halos and all three SoA components. Interior
// threads reuse the staged 3x3x3 neighborhood; boundary nodes retain the exact
// generic stencil. Allocation, transfers, layout conversion, and setup remain
// outside the timed kernel region.
CudaOperatorResult apply_node_stencil_cuda_shared_tile(const StructuredHexMesh& mesh,
                                                       const Material& material,
                                                       const std::vector<float>& x,
                                                       int repeats = 5,
                                                       int block_y = 8,
                                                       int block_z = 2);

}  // namespace gfss
