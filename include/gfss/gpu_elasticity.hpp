#pragma once

#include "gfss/hex8.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <vector>

namespace gfss {

struct CudaOperatorTiming {
    double best_ms{0.0};
    double mean_ms{0.0};
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

}  // namespace gfss
