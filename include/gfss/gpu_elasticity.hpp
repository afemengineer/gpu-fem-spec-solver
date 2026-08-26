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

// Gold3D launch geometry with the exact structural zeros removed from the
// interior orthogonal HEX8 stencil: 153 FMAs/node instead of 243. Boundary
// nodes retain the generic exact 27-class path. This is an arithmetic/
// instruction-count experiment; vector layout and device-vector memory match
// Gold3D.
CudaOperatorResult apply_node_stencil_cuda_gold_sparse(const StructuredHexMesh& mesh,
                                                       const Material& material,
                                                       const std::vector<float>& x,
                                                       int repeats = 5,
                                                       int block_y = 16);

// GoldSparse arithmetic with warp-register reuse along x. Full interior
// 32-lane x-warps load only the dx=0 value for each of the nine yz planes and
// obtain dx=+/-1 through warp shuffles; lane 0/31 fetch one halo value. Partial
// x-warps and physical boundaries fall back to the proven sparse/generic paths.
CudaOperatorResult apply_node_stencil_cuda_gold_shuffle(const StructuredHexMesh& mesh,
                                                        const Material& material,
                                                        const std::vector<float>& x,
                                                        int repeats = 5,
                                                        int block_y = 4);

}  // namespace gfss
