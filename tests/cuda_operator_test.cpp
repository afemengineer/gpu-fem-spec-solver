#include "gfss/cpu_elasticity.hpp"
#include "gfss/gpu_elasticity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

double relative_max_difference(const std::vector<double>& reference,
                               const std::vector<float>& candidate) {
    double max_abs = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        max_abs = std::max(max_abs,
                           std::abs(reference[i] - static_cast<double>(candidate[i])));
        scale = std::max(scale, std::abs(reference[i]));
    }
    return max_abs / std::max(1.0, scale);
}

}  // namespace

int main() {
    const gfss::StructuredHexMesh mesh{8, 7, 6, 2.0, 1.25, 0.75};
    const gfss::Material material{73000.0, 0.29};

    std::vector<float> x_float(static_cast<std::size_t>(mesh.dof_count()));
    std::vector<double> x_double(x_float.size());
    for (std::size_t i = 0; i < x_float.size(); ++i) {
        const double value = 0.17 * std::sin(0.071 * static_cast<double>(i + 1)) +
                             0.03 * std::cos(0.113 * static_cast<double>(i + 5));
        x_float[i] = static_cast<float>(value);
        x_double[i] = static_cast<double>(x_float[i]);
    }

    const auto cpu = gfss::apply_matrix_free(mesh, material, x_double);
    const auto atomic = gfss::apply_matrix_free_cuda_atomic(mesh, material, x_float, 2);
    const auto node = gfss::apply_node_stencil_cuda_soa(mesh, material, x_float, 2, 256);
    const auto gold3d = gfss::apply_node_stencil_cuda_gold3d(mesh, material, x_float, 2, 8);
    const auto sparse = gfss::apply_node_stencil_cuda_gold_sparse(mesh, material, x_float, 2, 16);

    const double atomic_rel = relative_max_difference(cpu, atomic.y);
    const double node_rel = relative_max_difference(cpu, node.y);
    const double gold3d_rel = relative_max_difference(cpu, gold3d.y);
    const double sparse_rel = relative_max_difference(cpu, sparse.y);

    require(atomic_rel < 2.0e-5,
            "FP32 CUDA atomic operator must match double CPU reference within tolerance");
    require(node_rel < 2.0e-5,
            "FP32 CUDA node stencil must match double CPU reference within tolerance");
    require(gold3d_rel < 2.0e-5,
            "FP32 CUDA Gold3D stencil must match double CPU reference within tolerance");
    require(sparse_rel < 2.0e-5,
            "FP32 CUDA GoldSparse stencil must match double CPU reference within tolerance");
    require(atomic.timing.best_ms > 0.0, "CUDA atomic timing must be positive");
    require(node.timing.best_ms > 0.0, "CUDA node timing must be positive");
    require(gold3d.timing.best_ms > 0.0, "CUDA Gold3D timing must be positive");
    require(sparse.timing.best_ms > 0.0, "CUDA GoldSparse timing must be positive");
    require(atomic.device_bytes == 2 * x_float.size() * sizeof(float),
            "CUDA atomic device-vector accounting mismatch");
    require(node.device_bytes == 2 * x_float.size() * sizeof(float),
            "CUDA node device-vector accounting mismatch");
    require(gold3d.device_bytes == 2 * x_float.size() * sizeof(float),
            "CUDA Gold3D device-vector accounting mismatch");
    require(sparse.device_bytes == 2 * x_float.size() * sizeof(float),
            "CUDA GoldSparse device-vector accounting mismatch");
    require(node.timing.median_zero_ms == 0.0,
            "CUDA node stencil must not require output zeroing");
    require(gold3d.timing.median_zero_ms == 0.0,
            "CUDA Gold3D stencil must not require output zeroing");
    require(sparse.timing.median_zero_ms == 0.0,
            "CUDA GoldSparse stencil must not require output zeroing");

    std::cout << "CUDA operator checks passed; atomic_rel_max=" << atomic_rel
              << " node_rel_max=" << node_rel
              << " gold3d_rel_max=" << gold3d_rel
              << " sparse_rel_max=" << sparse_rel
              << " atomic_best_ms=" << atomic.timing.best_ms
              << " node_best_ms=" << node.timing.best_ms
              << " gold3d_best_ms=" << gold3d.timing.best_ms
              << " sparse_best_ms=" << sparse.timing.best_ms << '\n';
    return 0;
}
