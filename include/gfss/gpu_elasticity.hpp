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

}  // namespace gfss
