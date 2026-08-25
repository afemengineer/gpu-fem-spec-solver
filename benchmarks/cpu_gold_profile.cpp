#include "gfss/cpu_gold.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

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
        throw std::invalid_argument("iteration count must be positive");
    }
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t nx = argc > 1 ? parse_u32(argv[1]) : 160U;
        const std::uint32_t ny = argc > 2 ? parse_u32(argv[2]) : nx;
        const std::uint32_t nz = argc > 3 ? parse_u32(argv[3]) : nx;
        const int iterations = argc > 4 ? parse_positive_int(argv[4]) : 500;

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        const std::size_t nodes = static_cast<std::size_t>(mesh.node_count());
        const std::size_t ndof = static_cast<std::size_t>(mesh.dof_count());

        std::vector<float> aos(ndof);
        for (std::size_t i = 0; i < ndof; ++i) {
            const double value = 0.1 * std::sin(0.001 * static_cast<double>(i + 1)) +
                                 0.03 * std::cos(0.0007 * static_cast<double>(i + 11));
            aos[i] = static_cast<float>(value);
        }

        std::vector<float> ux(nodes), uy(nodes), uz(nodes);
        std::vector<float> yx(nodes), yy(nodes), yz(nodes);
        gfss::aos_to_soa_fp32(aos.data(), nodes, ux.data(), uy.data(), uz.data());
        const auto stencil = gfss::build_cpu_gold_stencil_fp32(mesh, material);

        // Warm up code/data paths before the measured/profiled repetition block.
        for (int i = 0; i < 5; ++i) {
            gfss::apply_cpu_gold_soa_fp32(mesh, stencil,
                                          ux.data(), uy.data(), uz.data(),
                                          yx.data(), yy.data(), yz.data());
        }

        const auto start = Clock::now();
        for (int i = 0; i < iterations; ++i) {
            gfss::apply_cpu_gold_soa_fp32(mesh, stencil,
                                          ux.data(), uy.data(), uz.data(),
                                          yx.data(), yy.data(), yz.data());
        }
        const auto stop = Clock::now();

        double checksum = 0.0;
        const std::size_t stride = nodes / 1024 + 1;
        for (std::size_t i = 0; i < nodes; i += stride) {
            checksum += static_cast<double>(yx[i]) +
                        0.5 * static_cast<double>(yy[i]) +
                        0.25 * static_cast<double>(yz[i]);
        }

        const double total_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        const double avg_ms = total_ms / static_cast<double>(iterations);
        const double mdof_s = static_cast<double>(mesh.dof_count()) / (avg_ms * 1.0e3);

        std::cout << std::fixed << std::setprecision(3)
                  << "GFSS CPU Gold profiler workload\n"
                  << "mesh=" << nx << 'x' << ny << 'x' << nz
                  << " dofs=" << mesh.dof_count() << '\n'
                  << "iterations=" << iterations << '\n'
                  << "avx2_explicit=" << (gfss::cpu_gold_avx2_enabled() ? "yes" : "no") << '\n'
                  << "profile_region_total_ms=" << total_ms << '\n'
                  << "profile_region_avg_ms=" << avg_ms << '\n'
                  << "profile_region_MDOF/s=" << mdof_s << '\n'
                  << std::scientific
                  << "checksum=" << checksum << '\n';

        return gfss::cpu_gold_avx2_enabled() ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_cpu_gold_profile [nx [ny nz [iterations]]]\n";
        return 1;
    }
}
