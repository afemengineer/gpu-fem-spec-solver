#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_gold.hpp"
#include "gfss/cpu_gold_padded.hpp"
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

class AlignedFloatBuffer {
public:
    explicit AlignedFloatBuffer(std::size_t count)
        : storage_(count + 16U, 0.0f) {
        const auto raw = reinterpret_cast<std::uintptr_t>(storage_.data());
        const auto aligned =
            (raw + static_cast<std::uintptr_t>(63U)) &
            ~static_cast<std::uintptr_t>(63U);
        data_ = reinterpret_cast<float*>(aligned);
    }

    float* data() noexcept { return data_; }
    const float* data() const noexcept { return data_; }

private:
    std::vector<float> storage_;
    float* data_{nullptr};
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
    const std::size_t p95_index = std::min(
        sorted.size() - 1,
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(sorted.size()))) - 1);
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

void print_stats(const char* label, const TimingStats& timing, std::uint64_t dofs) {
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
        const std::uint32_t nx = argc > 1 ? parse_u32(argv[1]) : 160U;
        const std::uint32_t ny = argc > 2 ? parse_u32(argv[2]) : nx;
        const std::uint32_t nz = argc > 3 ? parse_u32(argv[3]) : nx;
        const int repeats = argc > 4 ? parse_positive_int(argv[4]) : 50;

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        const std::size_t nodes = static_cast<std::size_t>(mesh.node_count());
        const std::size_t ndof = static_cast<std::size_t>(mesh.dof_count());

        std::vector<float> xf(ndof);
        std::vector<double> xd(ndof);
        for (std::size_t i = 0; i < ndof; ++i) {
            const double value = 0.1 * std::sin(0.001 * static_cast<double>(i + 1)) +
                                 0.03 * std::cos(0.0007 * static_cast<double>(i + 11));
            xf[i] = static_cast<float>(value);
            xd[i] = static_cast<double>(xf[i]);
        }

        const auto gold_stencil = gfss::build_cpu_gold_stencil_fp32(mesh, material);

        std::vector<float> y_aos(ndof, 0.0f);
        std::vector<float> ux(nodes), uy(nodes), uz(nodes);
        std::vector<float> yx(nodes), yy(nodes), yz(nodes);
        std::vector<float> y_gold_aos(ndof);
        gfss::aos_to_soa_fp32(xf.data(), nodes, ux.data(), uy.data(), uz.data());

        const auto aos_timing = time_repeated(
            [&] {
                gfss::apply_node_stencil_openmp(
                    mesh, gold_stencil.regular, xf.data(), y_aos.data());
            },
            repeats);

        const auto gold_timing = time_repeated(
            [&] {
                gfss::apply_cpu_gold_soa_fp32(
                    mesh, gold_stencil,
                    ux.data(), uy.data(), uz.data(),
                    yx.data(), yy.data(), yz.data());
            },
            repeats);

        // Padded Gold is deliberately allocated after the original Gold timing
        // so the established baseline remains measured under the same state as
        // before this experiment.
        const auto padded_layout = gfss::make_cpu_gold_padded_layout_fp32(mesh);
        const auto padded_stencil =
            gfss::build_cpu_gold_padded_stencil_fp32(mesh, material, padded_layout);
        AlignedFloatBuffer pux(padded_layout.storage_nodes);
        AlignedFloatBuffer puy(padded_layout.storage_nodes);
        AlignedFloatBuffer puz(padded_layout.storage_nodes);
        AlignedFloatBuffer pyx(padded_layout.storage_nodes);
        AlignedFloatBuffer pyy(padded_layout.storage_nodes);
        AlignedFloatBuffer pyz(padded_layout.storage_nodes);
        std::vector<float> y_padded_aos(ndof);
        gfss::aos_to_padded_soa_fp32(
            mesh, padded_layout, xf.data(), pux.data(), puy.data(), puz.data());

        const auto padded_timing = time_repeated(
            [&] {
                gfss::apply_cpu_gold_padded_soa_fp32(
                    mesh, padded_stencil,
                    pux.data(), puy.data(), puz.data(),
                    pyx.data(), pyy.data(), pyz.data());
            },
            repeats);

        // All conversion and correctness work stays outside the timed region.
        gfss::soa_to_aos_fp32(yx.data(), yy.data(), yz.data(), nodes, y_gold_aos.data());
        gfss::padded_soa_to_aos_fp32(
            mesh, padded_layout,
            pyx.data(), pyy.data(), pyz.data(), y_padded_aos.data());
        const auto oracle = gfss::apply_matrix_free_openmp(mesh, material, xd);
        const double aos_rel = relative_max_difference(oracle, y_aos);
        const double gold_rel = relative_max_difference(oracle, y_gold_aos);
        const double padded_rel = relative_max_difference(oracle, y_padded_aos);

        const double aos_mdof_s =
            static_cast<double>(mesh.dof_count()) / (aos_timing.median_ms * 1.0e3);
        const double gold_mdof_s =
            static_cast<double>(mesh.dof_count()) / (gold_timing.median_ms * 1.0e3);
        const double padded_mdof_s =
            static_cast<double>(mesh.dof_count()) / (padded_timing.median_ms * 1.0e3);
        const double state_mib =
            static_cast<double>(6ULL * nodes * sizeof(float)) / (1024.0 * 1024.0);
        const double padded_state_mib =
            static_cast<double>(6ULL * padded_layout.storage_nodes * sizeof(float)) /
            (1024.0 * 1024.0);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "GFSS CPU Gold FP32 SoA AVX2/FMA benchmark\n"
                  << "mesh=" << nx << 'x' << ny << 'x' << nz
                  << " elements=" << mesh.element_count()
                  << " dofs=" << mesh.dof_count() << '\n'
                  << "openmp_max_threads=" << gfss::cpu_openmp_max_threads() << '\n'
                  << "avx2_explicit=" << (gfss::cpu_gold_avx2_enabled() ? "yes" : "no") << '\n'
                  << "vector_width_fp32=8\n"
                  << "repeats=" << repeats << '\n'
                  << "timing excludes stencil setup, AoS<->SoA conversion, allocation, and oracle audit\n";
        print_stats("cpu_aos_fp32", aos_timing, mesh.dof_count());
        print_stats("cpu_gold_soa_fp32", gold_timing, mesh.dof_count());
        print_stats("cpu_gold_padded_soa_fp32", padded_timing, mesh.dof_count());
        std::cout << "gold_speedup_vs_aos=" << (gold_mdof_s / aos_mdof_s) << "x\n"
                  << "padded_speedup_vs_gold=" << (padded_mdof_s / gold_mdof_s) << "x\n"
                  << "gold_soa_input_output_state=" << state_mib << " MiB\n"
                  << "padded_row_stride_floats=" << padded_layout.row_stride << '\n'
                  << "padded_soa_input_output_state=" << padded_state_mib << " MiB\n"
                  << std::scientific
                  << "aos_vs_oracle_rel_max=" << aos_rel << '\n'
                  << "gold_vs_oracle_rel_max=" << gold_rel << '\n'
                  << "padded_gold_vs_oracle_rel_max=" << padded_rel << '\n';

        if (!gfss::cpu_gold_avx2_enabled()) {
            std::cerr << "WARNING: CPU Gold was built without explicit AVX2; use GFSS_CPU_AVX2=ON for the Gold result\n";
        }
        if (aos_rel >= 2.0e-5 || gold_rel >= 2.0e-5 || padded_rel >= 2.0e-5) {
            std::cerr << "ERROR: CPU Gold benchmark failed numerical audit\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_cpu_gold_bench [nx [ny nz [repeats]]]\n";
        return 1;
    }
}
