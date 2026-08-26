#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_gold.hpp"
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

CpuTiming summarize_cpu(const std::vector<double>& samples) {
    CpuTiming timing;
    timing.best_ms = *std::min_element(samples.begin(), samples.end());
    timing.median_ms = percentile(samples, 0.50);
    timing.mean_ms = std::accumulate(samples.begin(), samples.end(), 0.0) /
                     static_cast<double>(samples.size());
    timing.p95_ms = percentile(samples, 0.95);
    return timing;
}

CpuTiming benchmark_cpu_gold(const gfss::StructuredHexMesh& mesh,
                             const gfss::CpuGoldStencilFp32& stencil,
                             const std::vector<float>& ux,
                             const std::vector<float>& uy,
                             const std::vector<float>& uz,
                             int repeats,
                             std::vector<float>& yx,
                             std::vector<float>& yy,
                             std::vector<float>& yz) {
    gfss::apply_cpu_gold_soa_fp32(
        mesh, stencil,
        ux.data(), uy.data(), uz.data(),
        yx.data(), yy.data(), yz.data());

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto start = Clock::now();
        gfss::apply_cpu_gold_soa_fp32(
            mesh, stencil,
            ux.data(), uy.data(), uz.data(),
            yx.data(), yy.data(), yz.data());
        const auto stop = Clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return summarize_cpu(samples);
}

double relative_max_difference(const std::vector<double>& reference,
                               const std::vector<float>& candidate) {
    double max_abs = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        max_abs = std::max(
            max_abs,
            std::abs(reference[i] - static_cast<double>(candidate[i])));
        scale = std::max(scale, std::abs(reference[i]));
    }
    return max_abs / std::max(1.0, scale);
}

void print_gpu(const char* label,
               const gfss::CudaOperatorResult& result,
               std::uint64_t dofs) {
    const double mdof_s =
        static_cast<double>(dofs) / (result.timing.median_ms * 1.0e3);
    std::cout << label
              << ": best_ms=" << result.timing.best_ms
              << " median_ms=" << result.timing.median_ms
              << " mean_ms=" << result.timing.mean_ms
              << " p95_ms=" << result.timing.p95_ms
              << " median_MDOF/s=" << mdof_s << '\n';
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
        const std::size_t nodes = static_cast<std::size_t>(mesh.node_count());
        const std::size_t ndof = static_cast<std::size_t>(mesh.dof_count());

        std::vector<float> xf(ndof);
        std::vector<double> xd(ndof);
        for (std::size_t i = 0; i < xf.size(); ++i) {
            const double value = 0.1 * std::sin(0.001 * static_cast<double>(i + 1)) +
                                 0.03 * std::cos(0.0007 * static_cast<double>(i + 11));
            xf[i] = static_cast<float>(value);
            xd[i] = static_cast<double>(xf[i]);
        }

        // Trusted numerical oracle: untimed FP64 reference.
        const auto oracle = gfss::apply_matrix_free_openmp(mesh, material, xd);

        // CPU Gold baseline under the same input/physics. Conversion/setup is untimed.
        const auto cpu_stencil = gfss::build_cpu_gold_stencil_fp32(mesh, material);
        std::vector<float> cux(nodes), cuy(nodes), cuz(nodes);
        std::vector<float> cyx(nodes), cyy(nodes), cyz(nodes);
        std::vector<float> cpu_gold_aos(ndof);
        gfss::aos_to_soa_fp32(
            xf.data(), nodes, cux.data(), cuy.data(), cuz.data());
        const auto cpu_timing = benchmark_cpu_gold(
            mesh, cpu_stencil, cux, cuy, cuz, repeats, cyx, cyy, cyz);
        gfss::soa_to_aos_fp32(
            cyx.data(), cyy.data(), cyz.data(), nodes, cpu_gold_aos.data());

        // Existing M2 element-scatter baseline.
        const auto atomic =
            gfss::apply_matrix_free_cuda_atomic(mesh, material, xf, repeats);

        // New atomics-free node-centric kernel: sweep launch width.
        const auto node128 =
            gfss::apply_node_stencil_cuda_soa(mesh, material, xf, repeats, 128);
        const auto node256 =
            gfss::apply_node_stencil_cuda_soa(mesh, material, xf, repeats, 256);
        const auto node512 =
            gfss::apply_node_stencil_cuda_soa(mesh, material, xf, repeats, 512);

        const double cpu_mdof_s =
            static_cast<double>(mesh.dof_count()) / (cpu_timing.median_ms * 1.0e3);
        const double atomic_mdof_s =
            static_cast<double>(mesh.dof_count()) / (atomic.timing.median_ms * 1.0e3);
        const double node128_mdof_s =
            static_cast<double>(mesh.dof_count()) / (node128.timing.median_ms * 1.0e3);
        const double node256_mdof_s =
            static_cast<double>(mesh.dof_count()) / (node256.timing.median_ms * 1.0e3);
        const double node512_mdof_s =
            static_cast<double>(mesh.dof_count()) / (node512.timing.median_ms * 1.0e3);
        const double best_node_mdof_s =
            std::max({node128_mdof_s, node256_mdof_s, node512_mdof_s});

        const double cpu_rel = relative_max_difference(oracle, cpu_gold_aos);
        const double atomic_rel = relative_max_difference(oracle, atomic.y);
        const double node128_rel = relative_max_difference(oracle, node128.y);
        const double node256_rel = relative_max_difference(oracle, node256.y);
        const double node512_rel = relative_max_difference(oracle, node512.y);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GFSS CUDA structured-Q1 operator benchmark\n"
                  << "mesh=" << nx << 'x' << ny << 'x' << nz
                  << " elements=" << mesh.element_count()
                  << " dofs=" << mesh.dof_count() << '\n'
                  << "openmp_max_threads=" << gfss::cpu_openmp_max_threads() << '\n'
                  << "repeats=" << repeats << '\n'
                  << "timing excludes allocation, host layout conversion, H2D/D2H, stencil setup, and oracle audit\n";

        std::cout << "cpu_gold_fp32: best_ms=" << cpu_timing.best_ms
                  << " median_ms=" << cpu_timing.median_ms
                  << " mean_ms=" << cpu_timing.mean_ms
                  << " p95_ms=" << cpu_timing.p95_ms
                  << " median_MDOF/s=" << cpu_mdof_s << '\n';
        print_gpu("gpu_atomic_fp32_total", atomic, mesh.dof_count());
        std::cout << "gpu_atomic_zero: median_ms=" << atomic.timing.median_zero_ms << '\n'
                  << "gpu_atomic_kernel: median_ms=" << atomic.timing.median_kernel_ms << '\n';
        print_gpu("gpu_node_soa_tpb128", node128, mesh.dof_count());
        print_gpu("gpu_node_soa_tpb256", node256, mesh.dof_count());
        print_gpu("gpu_node_soa_tpb512", node512, mesh.dof_count());

        std::cout << "best_node_speedup_vs_atomic="
                  << (best_node_mdof_s / atomic_mdof_s) << "x\n"
                  << "best_node_speedup_vs_cpu_gold="
                  << (best_node_mdof_s / cpu_mdof_s) << "x\n"
                  << "atomic_device_vectors="
                  << (static_cast<double>(atomic.device_bytes) / (1024.0 * 1024.0))
                  << " MiB\n"
                  << "node_device_vectors="
                  << (static_cast<double>(node256.device_bytes) / (1024.0 * 1024.0))
                  << " MiB\n"
                  << std::scientific
                  << "cpu_gold_vs_oracle_rel_max=" << cpu_rel << '\n'
                  << "atomic_vs_oracle_rel_max=" << atomic_rel << '\n'
                  << "node128_vs_oracle_rel_max=" << node128_rel << '\n'
                  << "node256_vs_oracle_rel_max=" << node256_rel << '\n'
                  << "node512_vs_oracle_rel_max=" << node512_rel << '\n';

        if (cpu_rel >= 2.0e-5 || atomic_rel >= 2.0e-5 ||
            node128_rel >= 2.0e-5 || node256_rel >= 2.0e-5 ||
            node512_rel >= 2.0e-5) {
            std::cerr << "ERROR: one or more optimized operators differ from FP64 oracle beyond tolerance\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_cuda_operator_bench [nx [ny nz [repeats]]]\n";
        return 1;
    }
}
