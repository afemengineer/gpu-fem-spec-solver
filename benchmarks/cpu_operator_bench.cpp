#include "gfss/cpu_elasticity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double checksum(const std::vector<double>& values) {
    double sum = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        sum += values[i] * (1.0 + static_cast<double>(i % 17) * 1.0e-3);
    }
    return sum;
}

template <typename Apply>
double benchmark_ms(Apply&& apply, int warmup, int repeats, double& sink) {
    for (int i = 0; i < warmup; ++i) {
        sink += checksum(apply());
    }

    double best_ms = 1.0e300;
    double total_ms = 0.0;
    for (int i = 0; i < repeats; ++i) {
        const auto start = Clock::now();
        const auto y = apply();
        const auto stop = Clock::now();
        sink += checksum(y);
        const double ms = std::chrono::duration<double, std::milli>(stop - start).count();
        total_ms += ms;
        best_ms = std::min(best_ms, ms);
    }

    std::cout << "  best_ms=" << best_ms
              << " mean_ms=" << (total_ms / repeats);
    return best_ms;
}

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
        const std::uint32_t nx = argc > 1 ? parse_u32(argv[1]) : 64U;
        const std::uint32_t ny = argc > 2 ? parse_u32(argv[2]) : nx;
        const std::uint32_t nz = argc > 3 ? parse_u32(argv[3]) : nx;
        const int repeats = argc > 4 ? parse_positive_int(argv[4]) : 5;

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        std::vector<double> x(static_cast<std::size_t>(mesh.dof_count()));
        for (std::size_t i = 0; i < x.size(); ++i) {
            x[i] = 0.1 * std::sin(0.001 * static_cast<double>(i + 1)) +
                   0.03 * std::cos(0.0007 * static_cast<double>(i + 11));
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GFSS CPU matrix-free HEX8 operator benchmark\n"
                  << "mesh=" << nx << 'x' << ny << 'x' << nz
                  << " elements=" << mesh.element_count()
                  << " dofs=" << mesh.dof_count() << '\n'
                  << "openmp_max_threads=" << gfss::cpu_openmp_max_threads() << '\n'
                  << "repeats=" << repeats << " (best and mean reported)\n";

        double sink = 0.0;
        std::cout << "serial:";
        const double serial_ms = benchmark_ms(
            [&] { return gfss::apply_matrix_free(mesh, material, x); }, 1, repeats, sink);
        const double serial_mdof_s =
            static_cast<double>(mesh.dof_count()) / (serial_ms * 1.0e3);
        std::cout << " MDOF/s=" << serial_mdof_s << '\n';

        std::cout << "openmp:";
        const double omp_ms = benchmark_ms(
            [&] { return gfss::apply_matrix_free_openmp(mesh, material, x); }, 1, repeats, sink);
        const double omp_mdof_s = static_cast<double>(mesh.dof_count()) / (omp_ms * 1.0e3);
        std::cout << " MDOF/s=" << omp_mdof_s
                  << " speedup=" << (serial_ms / omp_ms) << "x\n";

        const auto serial = gfss::apply_matrix_free(mesh, material, x);
        const auto parallel = gfss::apply_matrix_free_openmp(mesh, material, x);
        double max_abs = 0.0;
        double scale = 0.0;
        for (std::size_t i = 0; i < serial.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(serial[i] - parallel[i]));
            scale = std::max(scale, std::abs(serial[i]));
        }
        const double rel_max = max_abs / std::max(1.0, scale);
        std::cout << "serial_vs_openmp_rel_max=" << std::scientific << rel_max << '\n'
                  << "checksum_sink=" << sink << '\n';

        if (rel_max > 1.0e-11) {
            std::cerr << "ERROR: OpenMP operator differs from serial reference beyond tolerance\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::cerr << "usage: gfss_cpu_operator_bench [nx [ny nz [repeats]]]\n";
        return 1;
    }
}
