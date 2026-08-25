#include "gfss/cpu_elasticity.hpp"
#include "gfss/gpu_elasticity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

struct CpuTiming {
    double best_ms{0.0};
    double median_ms{0.0};
    double mean_ms{0.0};
    double p95_ms{0.0};
};

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

double percentile(std::vector<double> samples, double p) {
    std::sort(samples.begin(), samples.end());
    const double position = p * static_cast<double>(samples.size() - 1);
    const auto lower = static_cast<std::size_t>(position);
    const auto upper = std::min(lower + 1, samples.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return samples[lower] * (1.0 - fraction) + samples[upper] * fraction;
}

CpuTiming benchmark_cpu_openmp(const gfss::StructuredHexMesh& mesh,
                               const gfss::Material& material,
                               const std::vector<double>& x,
                               int repeats,
                               std::vector<double>& last_result) {
    last_result = gfss::apply_matrix_free_openmp(mesh, material, x);

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        last_result = gfss::apply_matrix_free_openmp(mesh, material, x);
        const auto stop = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    CpuTiming timing;
    timing.best_ms = *std::min_element(samples.begin(), samples.end());
    timing.median_ms = percentile(samples, 0.50);
    timing.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) /
                     static_cast<double>(samples.size());
    timing.p95_ms = percentile(samples, 0.95);
    return timing;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t nx = argc > 1 ? parse_u32(argv[1]) : 64U;
        const std::uint32_t ny = argc > 2 ? parse_u32(argv[2]) : nx;
        const std::uint32_t nz = argc > 3 ? parse_u32(argv[3]) : nx;
        const int repeats = argc > 4 ? parse_positive_int(argv[4]) : 10;

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 1.0};
        const gfss::Material material{210.0e9, 0.30};

        std::vector<float> xf(static_cast<std::size_t>(mesh.dof_count()));
        std::vector<double> xd(xf.size());
        for (std::size_t i = 0; i < xf.size(); ++i) {
            const double value = 0.1 * std::sin(0.001 * static_cast<double>(i + 1)) +
                                 0.03 * std::cos(0.0007 * static_cast<double>(i + 11));
            xf[i] = static_cast<float>(value);
            xd[i] = static_cast<double>(xf[i]);
        }

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GFSS CUDA FP32 atomic HEX8 operator benchmark\n"
                  << "mesh=" << nx << 'x' << ny << 'x' << nz
                  << " elements=" << mesh.element_count()
                  << " dofs=" << mesh.dof_count() << '\n'
                  << "openmp_max_threads=" << gfss::cpu_openmp_max_threads() << '\n'
                  << "repeats=" << repeats << '\n'
                  << "note: CPU path is currently FP64; GPU path is FP32\n"
                  << "note: CPU steady-state timing still includes output-vector allocation; GPU allocation/H2D/D2H are excluded\n";

        std::vector<double> cpu;
        const auto cpu_timing = benchmark_cpu_openmp(mesh, material, xd, repeats, cpu);
        const double cpu_mdof_s =
            static_cast<double>(mesh.dof_count()) / (cpu_timing.median_ms * 1.0e3);

        const auto gpu = gfss::apply_matrix_free_cuda_atomic(mesh, material, xf, repeats);
        const double gpu_mdof_s =
            static_cast<double>(mesh.dof_count()) / (gpu.timing.median_ms * 1.0e3);

        double max_abs = 0.0;
        double scale = 0.0;
        for (std::size_t i = 0; i < cpu.size(); ++i) {
            max_abs = std::max(max_abs, std::abs(cpu[i] - static_cast<double>(gpu.y[i])));
            scale = std::max(scale, std::abs(cpu[i]));
        }
        const double rel_max = max_abs / std::max(1.0, scale);

        std::cout << "cpu_openmp_fp64: best_ms=" << cpu_timing.best_ms
                  << " median_ms=" << cpu_timing.median_ms
                  << " mean_ms=" << cpu_timing.mean_ms
                  << " p95_ms=" << cpu_timing.p95_ms
                  << " median_MDOF/s=" << cpu_mdof_s << '\n'
                  << "gpu_atomic_fp32_total: best_ms=" << gpu.timing.best_ms
                  << " median_ms=" << gpu.timing.median_ms
                  << " mean_ms=" << gpu.timing.mean_ms
                  << " p95_ms=" << gpu.timing.p95_ms
                  << " median_MDOF/s=" << gpu_mdof_s
                  << " provisional_speedup_vs_fp64_cpu="
                  << (cpu_timing.median_ms / gpu.timing.median_ms) << "x\n"
                  << "gpu_zero: best_ms=" << gpu.timing.best_zero_ms
                  << " median_ms=" << gpu.timing.median_zero_ms
                  << " mean_ms=" << gpu.timing.mean_zero_ms << '\n'
                  << "gpu_kernel: best_ms=" << gpu.timing.best_kernel_ms
                  << " median_ms=" << gpu.timing.median_kernel_ms
                  << " mean_ms=" << gpu.timing.mean_kernel_ms << '\n'
                  << "gpu_device_vectors=" << (static_cast<double>(gpu.device_bytes) / (1024.0 * 1024.0))
                  << " MiB\n"
                  << std::scientific
                  << "cpu_vs_gpu_rel_max=" << rel_max << '\n';

        if (rel_max >= 2.0e-5) {
            std::cerr << "ERROR: CUDA FP32 operator differs from CPU reference beyond tolerance\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_cuda_operator_bench [nx [ny nz [repeats]]]\n";
        return 1;
    }
}
