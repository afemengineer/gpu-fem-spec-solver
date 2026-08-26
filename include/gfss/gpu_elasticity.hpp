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

CudaOperatorResult apply_node_stencil_cuda_soa(const StructuredHexMesh& mesh,
                                               const Material& material,
                                               const std::vector<float>& x,
                                               int repeats = 5,
                                               int threads_per_block = 256);

CudaOperatorResult apply_node_stencil_cuda_gold3d(const StructuredHexMesh& mesh,
                                                  const Material& material,
                                                  const std::vector<float>& x,
                                                  int repeats = 5,
                                                  int block_y = 8);

// Nsight-guided variant: launch the nx-1 by ny-1 by nz-1 interior separately
// with a nearly perfectly packed 32x16 block geometry, then evaluate the small
// boundary set with a compact exact kernel. This removes most padded lanes from
// the 161x161 Gold3D launch while preserving the same operator mathematics.
CudaOperatorResult apply_node_stencil_cuda_interior_split(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<float>& x,
    int repeats = 5);

}  // namespace gfss
