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
    const auto gpu = gfss::apply_matrix_free_cuda_atomic(mesh, material, x_float, 2);

    double max_abs = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(cpu[i] - static_cast<double>(gpu.y[i])));
        scale = std::max(scale, std::abs(cpu[i]));
    }
    const double rel_max = max_abs / std::max(1.0, scale);

    require(rel_max < 2.0e-5,
            "FP32 CUDA atomic operator must match double CPU reference within tolerance");
    require(gpu.timing.best_ms > 0.0, "CUDA timing must be positive");
    require(gpu.device_bytes == 2 * x_float.size() * sizeof(float),
            "CUDA baseline device-vector accounting mismatch");

    std::cout << "CUDA atomic operator check passed; rel_max=" << rel_max
              << " best_ms=" << gpu.timing.best_ms << '\n';
    return 0;
}
