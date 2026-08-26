#include "gfss/gpu_elasticity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t parse_u32(const char* text) {
    const auto value = std::stoul(text);
    if (value == 0 || value > 1000000UL) {
        throw std::invalid_argument("mesh dimensions must be in [1, 1000000]");
    }
    return static_cast<std::uint32_t>(value);
}

int parse_positive_int(const char* text) {
    const int value = std::stoi(text);
    if (value <= 0) {
        throw std::invalid_argument("repeat count must be positive");
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t nx = argc > 1 ? parse_u32(argv[1]) : 160U;
        const std::uint32_t ny = argc > 2 ? parse_u32(argv[2]) : nx;
        const std::uint32_t nz = argc > 3 ? parse_u32(argv[3]) : nx;
        const int repeats = argc > 4 ? parse_positive_int(argv[4]) : 2;

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        const std::size_t ndof = static_cast<std::size_t>(mesh.dof_count());

        std::vector<float> x(ndof);
        for (std::size_t i = 0; i < ndof; ++i) {
            const double value = 0.1 * std::sin(0.001 * static_cast<double>(i + 1)) +
                                 0.03 * std::cos(0.0007 * static_cast<double>(i + 11));
            x[i] = static_cast<float>(value);
        }

        const auto result =
            gfss::apply_node_stencil_cuda_gold3d(mesh, material, x, repeats, 16);

        double checksum = 0.0;
        const std::size_t stride = std::max<std::size_t>(1U, result.y.size() / 1024U);
        for (std::size_t i = 0; i < result.y.size(); i += stride) {
            checksum += static_cast<double>(result.y[i]);
        }

        const double mdof_s =
            static_cast<double>(mesh.dof_count()) /
            (result.timing.median_ms * 1.0e3);

        std::cout << std::fixed << std::setprecision(3)
                  << "GFSS CUDA Gold3D profiler workload\n"
                  << "mesh=" << nx << 'x' << ny << 'x' << nz
                  << " dofs=" << mesh.dof_count() << '\n'
                  << "block=32x16x1\n"
                  << "repeats=" << repeats << '\n'
                  << "median_ms=" << result.timing.median_ms << '\n'
                  << "median_MDOF/s=" << mdof_s << '\n'
                  << std::scientific
                  << "checksum=" << checksum << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_gold_profile [nx [ny nz [repeats]]]\n";
        return 1;
    }
}
