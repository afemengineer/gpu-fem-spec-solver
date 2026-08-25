#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_stencil.hpp"

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

struct TimingStats {
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

template <typename Fn>
TimingStats time_repeated(Fn&& fn, int repeats) {
    fn();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const auto start = Clock::now();
        fn();
        const auto stop = Clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                        static_cast<double>(samples.size());
    const std::size_t median_index = sorted.size() / 2;
    const double median = sorted.size() % 2 == 0
                              ? 0.5 * (sorted[median_index - 1] + sorted[median_index])
                              : sorted[median_index];
    const std::size_t p95_index = std::min(sorted.size() - 1,
                                           static_cast<std::size_t>(std::ceil(0.95 * sorted.size())) - 1);
    return TimingStats{sorted.front(), median, mean, sorted[p95_index]};
}

template <typename A, typename B>
double relative_max_difference(const A& a, const B& b) {
    double max_abs = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double av = static_cast<double>(a[i]);
        const double bv = static_cast<double>(b[i]);
        max_abs = std::max(max_abs, std::abs(av - bv));
        scale = std::max(scale, std::abs(av));
    }
    return max_abs / std::max(1.0, scale);
}

void print_stats(const char* label,
                 const TimingStats& timing,
                 std::uint64_t dofs) {
    const double mdof_s = static_cast<double>(dofs) / (timing.median_ms * 1.0e3);
    std::cout << label
              << ": best_ms=" << timing.best_ms
              << " median_ms=" << timing.median_ms
              << " mean_ms=" << timing.mean_ms
              << " p95_ms=" << timing.p95_ms
              << " median_MDOF/s=" << mdof_s << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t nx = argc > 1 ? parse_u32(argv[1]) : 64U;
        const std::uint32_t ny = argc > 2 ? parse_u32(argv[2]) : nx;
        const std::uint32_t nz = argc > 3 ? parse_u32(argv[3]) : nx;
        const int repeats = argc > 4 ? parse_positive_int(argv[4]) : 20;

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        const std::size_t ndof = static_cast<std::size_t>(mesh.dof_count());

        std::vector<double> xd(ndof);
        std::vector<float> xf(ndof);
        for (std::size_t i = 0; i < ndof; ++i) {
            const double value = 0.1 * std::sin(0.001 * static_cast<double>(i + 1)) +
                                 0.03 * std::cos(0.0007 * static_cast<double>(i + 11));
            xf[i] = static_cast<float>(value);
            xd[i] = static_cast<double>(xf[i]);
        }

        const auto stencil64 = gfss::build_regular_node_stencil_fp64(mesh, material);
        const auto stencil32 = gfss::build_regular_node_stencil_fp32(mesh, material);
        std::vector<double> y64(ndof, 0.0);
        std::vector<float> y32(ndof, 0.0f);

        const auto t64 = time_repeated(
            [&] { gfss::apply_node_stencil_openmp(mesh, stencil64, xd.data(), y64.data()); },
            repeats);
        const auto t32 = time_repeated(
            [&] { gfss::apply_node_stencil_openmp(mesh, stencil32, xf.data(), y32.data()); },
            repeats);

        // Accuracy audit is outside the timed region.
        const auto oracle = gfss::apply_matrix_free_openmp(mesh, material, xd);
        const double fp64_rel = relative_max_difference(oracle, y64);
        const double fp32_rel = relative_max_difference(oracle, y32);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GFSS optimized CPU node-stencil benchmark\n"
                  << "mesh=" << nx << 'x' << ny << 'x' << nz
                  << " elements=" << mesh.element_count()
                  << " dofs=" << mesh.dof_count() << '\n'
                  << "openmp_max_threads=" << gfss::cpu_openmp_max_threads() << '\n'
                  << "repeats=" << repeats << '\n'
                  << "timing excludes stencil setup and all output allocation\n";
        print_stats("cpu_stencil_fp64", t64, mesh.dof_count());
        print_stats("cpu_stencil_fp32", t32, mesh.dof_count());
        std::cout << std::scientific
                  << "fp64_vs_oracle_rel_max=" << fp64_rel << '\n'
                  << "fp32_vs_oracle_rel_max=" << fp32_rel << '\n';

        if (fp64_rel >= 2.0e-11 || fp32_rel >= 2.0e-5) {
            std::cerr << "ERROR: optimized stencil failed numerical audit\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_cpu_stencil_bench [nx [ny nz [repeats]]]\n";
        return 1;
    }
}
