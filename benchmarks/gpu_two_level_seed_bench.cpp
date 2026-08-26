#include "gfss/cpu_elasticity.hpp"
#include "gfss/gpu_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

enum class LoadKind {
    UniformZ,
    CheckerboardZ,
};

struct CaseDef {
    const char* name;
    const char* description;
    gfss::StructuredHexMesh mesh;
    gfss::Material material;
    LoadKind load;
};

double parse_tolerance(const char* text, const char* name) {
    const double value = std::stod(text);
    if (!(value > 0.0) || value >= 1.0) {
        throw std::invalid_argument(std::string(name) + " must be in (0, 1)");
    }
    return value;
}

std::size_t parse_size(const char* text, const char* name) {
    const auto value = std::stoull(text);
    if (value == 0ULL) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

std::vector<CaseDef> make_cases() {
    constexpr double E = 210.0e9;
    return {
        {"baseline_cube",
         "64^3 cube, uniform transverse face load",
         {64U, 64U, 64U, 1.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::UniformZ},
        {"slender_beam",
         "4:1:1 cantilever with near-isotropic cells",
         {128U, 32U, 32U, 4.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::UniformZ},
        {"thin_plate",
         "thin 1:1:0.125 domain with near-isotropic cells",
         {64U, 64U, 8U, 1.0, 1.0, 0.125},
         {E, 0.30},
         LoadKind::UniformZ},
        {"near_incompressible",
         "48^3 cube with Poisson ratio 0.49",
         {48U, 48U, 48U, 1.0, 1.0, 1.0},
         {E, 0.49},
         LoadKind::UniformZ},
        {"checkerboard_face",
         "48^3 cube with high-frequency alternating face load",
         {48U, 48U, 48U, 1.0, 1.0, 1.0},
         {E, 0.30},
         LoadKind::CheckerboardZ},
    };
}

std::vector<float> make_rhs(const CaseDef& test) {
    const auto& mesh = test.mesh;
    std::vector<float> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0f);
    const double count =
        static_cast<double>(mesh.ny + 1U) *
        static_cast<double>(mesh.nz + 1U);
    const double magnitude = 1.0 / count;

    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            double value = -magnitude;
            if (test.load == LoadKind::CheckerboardZ) {
                value = ((j + k) & 1U) == 0U ? magnitude : -magnitude;
            }
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] =
                static_cast<float>(value);
        }
    }
    return rhs;
}

gfss::StructuredHexMesh make_coarse_mesh(const gfss::StructuredHexMesh& fine) {
    if ((fine.nx & 1U) != 0U || (fine.ny & 1U) != 0U ||
        (fine.nz & 1U) != 0U || fine.nx < 2U || fine.ny < 2U || fine.nz < 2U) {
        throw std::invalid_argument(
            "two-level benchmark requires even fine element counts >= 2");
    }
    return {fine.nx / 2U,
            fine.ny / 2U,
            fine.nz / 2U,
            fine.lx,
            fine.ly,
            fine.lz};
}

struct AxisInterpolation {
    std::uint32_t a{0};
    std::uint32_t b{0};
    float wa{1.0f};
    float wb{0.0f};
};

AxisInterpolation axis_interpolation(std::uint32_t fine_index) {
    if ((fine_index & 1U) == 0U) {
        const std::uint32_t c = fine_index / 2U;
        return {c, c, 1.0f, 0.0f};
    }
    const std::uint32_t c = fine_index / 2U;
    return {c, c + 1U, 0.5f, 0.5f};
}

std::vector<float> restrict_pt(const gfss::StructuredHexMesh& fine,
                               const gfss::StructuredHexMesh& coarse,
                               const std::vector<float>& fine_rhs) {
    if (fine_rhs.size() != static_cast<std::size_t>(fine.dof_count())) {
        throw std::invalid_argument("restriction fine vector size mismatch");
    }
    std::vector<float> coarse_rhs(
        static_cast<std::size_t>(coarse.dof_count()), 0.0f);

    for (std::uint32_t k = 0; k <= fine.nz; ++k) {
        const auto wz = axis_interpolation(k);
        for (std::uint32_t j = 0; j <= fine.ny; ++j) {
            const auto wy = axis_interpolation(j);
            for (std::uint32_t i = 0; i <= fine.nx; ++i) {
                const auto wx = axis_interpolation(i);
                const auto fine_node = fine.node_index(i, j, k);
                const std::size_t fine_base =
                    static_cast<std::size_t>(3ULL * fine_node);

                const std::uint32_t ci[2] = {wx.a, wx.b};
                const std::uint32_t cj[2] = {wy.a, wy.b};
                const std::uint32_t ck[2] = {wz.a, wz.b};
                const float wi[2] = {wx.wa, wx.wb};
                const float wj[2] = {wy.wa, wy.wb};
                const float wk[2] = {wz.wa, wz.wb};

                for (int dz = 0; dz < 2; ++dz) {
                    if (wk[dz] == 0.0f) continue;
                    for (int dy = 0; dy < 2; ++dy) {
                        if (wj[dy] == 0.0f) continue;
                        for (int dx = 0; dx < 2; ++dx) {
                            if (wi[dx] == 0.0f) continue;
                            const float w = wi[dx] * wj[dy] * wk[dz];
                            const auto coarse_node =
                                coarse.node_index(ci[dx], cj[dy], ck[dz]);
                            const std::size_t coarse_base =
                                static_cast<std::size_t>(3ULL * coarse_node);
                            for (std::size_t c = 0; c < 3U; ++c) {
                                coarse_rhs[coarse_base + c] +=
                                    w * fine_rhs[fine_base + c];
                            }
                        }
                    }
                }
            }
        }
    }

    // Preserve the clamped x=0 face exactly on the coarse problem.
    for (std::uint32_t k = 0; k <= coarse.nz; ++k) {
        for (std::uint32_t j = 0; j <= coarse.ny; ++j) {
            const auto node = coarse.node_index(0U, j, k);
            const std::size_t base = static_cast<std::size_t>(3ULL * node);
            coarse_rhs[base + 0U] = 0.0f;
            coarse_rhs[base + 1U] = 0.0f;
            coarse_rhs[base + 2U] = 0.0f;
        }
    }
    return coarse_rhs;
}

std::vector<float> prolongate(const gfss::StructuredHexMesh& fine,
                              const gfss::StructuredHexMesh& coarse,
                              const std::vector<float>& coarse_x) {
    if (coarse_x.size() != static_cast<std::size_t>(coarse.dof_count())) {
        throw std::invalid_argument("prolongation coarse vector size mismatch");
    }
    std::vector<float> fine_x(
        static_cast<std::size_t>(fine.dof_count()), 0.0f);

    for (std::uint32_t k = 0; k <= fine.nz; ++k) {
        const auto wz = axis_interpolation(k);
        for (std::uint32_t j = 0; j <= fine.ny; ++j) {
            const auto wy = axis_interpolation(j);
            for (std::uint32_t i = 0; i <= fine.nx; ++i) {
                if (i == 0U) continue;
                const auto wx = axis_interpolation(i);
                const auto fine_node = fine.node_index(i, j, k);
                const std::size_t fine_base =
                    static_cast<std::size_t>(3ULL * fine_node);

                const std::uint32_t ci[2] = {wx.a, wx.b};
                const std::uint32_t cj[2] = {wy.a, wy.b};
                const std::uint32_t ck[2] = {wz.a, wz.b};
                const float wi[2] = {wx.wa, wx.wb};
                const float wj[2] = {wy.wa, wy.wb};
                const float wk[2] = {wz.wa, wz.wb};

                for (int dz = 0; dz < 2; ++dz) {
                    if (wk[dz] == 0.0f) continue;
                    for (int dy = 0; dy < 2; ++dy) {
                        if (wj[dy] == 0.0f) continue;
                        for (int dx = 0; dx < 2; ++dx) {
                            if (wi[dx] == 0.0f) continue;
                            const float w = wi[dx] * wj[dy] * wk[dz];
                            const auto coarse_node =
                                coarse.node_index(ci[dx], cj[dy], ck[dz]);
                            const std::size_t coarse_base =
                                static_cast<std::size_t>(3ULL * coarse_node);
                            for (std::size_t c = 0; c < 3U; ++c) {
                                fine_x[fine_base + c] +=
                                    w * coarse_x[coarse_base + c];
                            }
                        }
                    }
                }
            }
        }
    }
    return fine_x;
}

double true_relative_residual(const gfss::StructuredHexMesh& mesh,
                              const gfss::Material& material,
                              const std::vector<float>& rhs,
                              const std::vector<float>& x) {
    std::vector<double> xd(x.begin(), x.end());
    const auto ax = gfss::apply_clamped_x0_matrix_free(mesh, material, xd);
    double rr = 0.0;
    double bb = 0.0;
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        const double b = static_cast<double>(rhs[i]);
        const double r = b - ax[i];
        rr += r * r;
        bb += b * b;
    }
    return bb > 0.0 ? std::sqrt(rr / bb) : 0.0;
}

double duration_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void run_case(const CaseDef& test,
              double fine_tolerance,
              double coarse_tolerance,
              std::size_t max_iterations) {
    const auto& fine = test.mesh;
    const auto coarse = make_coarse_mesh(fine);
    const auto rhs = make_rhs(test);

    const auto baseline_wall_start = Clock::now();
    const auto baseline = gfss::solve_pcg_cuda_gold_sparse_x0(
        fine, test.material, rhs, fine_tolerance, max_iterations, 4);
    const auto baseline_wall_stop = Clock::now();
    const double baseline_true =
        true_relative_residual(fine, test.material, rhs, baseline.x);

    const auto restrict_start = Clock::now();
    const auto coarse_rhs = restrict_pt(fine, coarse, rhs);
    const auto restrict_stop = Clock::now();

    const auto coarse_wall_start = Clock::now();
    const auto coarse_result = gfss::solve_pcg_cuda_gold_sparse_x0(
        coarse,
        test.material,
        coarse_rhs,
        coarse_tolerance,
        max_iterations,
        4);
    const auto coarse_wall_stop = Clock::now();

    const auto prolong_start = Clock::now();
    const auto fine_seed = prolongate(fine, coarse, coarse_result.x);
    const auto prolong_stop = Clock::now();
    const double seed_true =
        true_relative_residual(fine, test.material, rhs, fine_seed);

    const auto seeded_wall_start = Clock::now();
    const auto seeded = gfss::solve_pcg_cuda_gold_sparse_seeded_x0(
        fine,
        test.material,
        rhs,
        fine_seed,
        fine_tolerance,
        max_iterations,
        4);
    const auto seeded_wall_stop = Clock::now();
    const double seeded_true =
        true_relative_residual(fine, test.material, rhs, seeded.x);

    const double baseline_wall_ms =
        duration_ms(baseline_wall_start, baseline_wall_stop);
    const double coarse_wall_ms =
        duration_ms(coarse_wall_start, coarse_wall_stop);
    const double seeded_wall_ms =
        duration_ms(seeded_wall_start, seeded_wall_stop);
    const double restriction_ms = duration_ms(restrict_start, restrict_stop);
    const double prolongation_ms = duration_ms(prolong_start, prolong_stop);
    const double two_level_solve_ms =
        coarse_result.solve_ms + seeded.solve_ms;
    const double two_level_wall_ms =
        restriction_ms + coarse_wall_ms + prolongation_ms + seeded_wall_ms;

    const double iteration_reduction = baseline.iterations > 0U
        ? 100.0 *
            (1.0 - static_cast<double>(seeded.iterations) /
                       static_cast<double>(baseline.iterations))
        : 0.0;
    const double solve_speedup = two_level_solve_ms > 0.0
        ? baseline.solve_ms / two_level_solve_ms
        : 0.0;
    const double wall_speedup = two_level_wall_ms > 0.0
        ? baseline_wall_ms / two_level_wall_ms
        : 0.0;
    const double coarse_bytes_per_fine_dof =
        static_cast<double>(coarse_result.explicit_device_bytes) /
        static_cast<double>(fine.dof_count());

    std::cout << std::fixed << std::setprecision(6)
              << "case=" << test.name << '\n'
              << "description=" << test.description << '\n'
              << "fine_mesh=" << fine.nx << 'x' << fine.ny << 'x' << fine.nz
              << " fine_dofs=" << fine.dof_count()
              << " coarse_mesh=" << coarse.nx << 'x' << coarse.ny << 'x'
              << coarse.nz
              << " coarse_dofs=" << coarse.dof_count() << '\n'
              << std::scientific
              << "fine_tolerance=" << fine_tolerance
              << " coarse_tolerance=" << coarse_tolerance << '\n'
              << std::fixed
              << "baseline_converged=" << (baseline.converged ? "true" : "false")
              << " baseline_iterations=" << baseline.iterations
              << " baseline_matvecs=" << baseline.matvecs
              << " baseline_solve_ms=" << baseline.solve_ms
              << " baseline_wall_ms=" << baseline_wall_ms << '\n'
              << std::scientific
              << "baseline_audited_rel=" << baseline.audited_relative_residual
              << " baseline_true_rel=" << baseline_true << '\n'
              << std::fixed
              << "coarse_converged=" << (coarse_result.converged ? "true" : "false")
              << " coarse_iterations=" << coarse_result.iterations
              << " coarse_solve_ms=" << coarse_result.solve_ms
              << " restriction_ms=" << restriction_ms
              << " prolongation_ms=" << prolongation_ms << '\n'
              << std::scientific
              << "coarse_audited_rel=" << coarse_result.audited_relative_residual
              << " prolonged_seed_true_rel=" << seed_true << '\n'
              << std::fixed
              << "seeded_converged=" << (seeded.converged ? "true" : "false")
              << " seeded_fine_iterations=" << seeded.iterations
              << " seeded_fine_matvecs=" << seeded.matvecs
              << " seeded_fine_solve_ms=" << seeded.solve_ms
              << " seeded_fine_wall_ms=" << seeded_wall_ms << '\n'
              << std::scientific
              << "seeded_audited_rel=" << seeded.audited_relative_residual
              << " seeded_true_rel=" << seeded_true << '\n'
              << std::fixed
              << "fine_iteration_reduction_pct=" << iteration_reduction << '\n'
              << "two_level_gpu_solve_ms=" << two_level_solve_ms
              << " baseline_vs_two_level_solve_speedup=" << solve_speedup << '\n'
              << "two_level_wall_ms=" << two_level_wall_ms
              << " baseline_vs_two_level_wall_speedup=" << wall_speedup << '\n'
              << "fine_solver_bytes_per_dof=24.000000\n"
              << "coarse_extra_bytes_per_fine_dof="
              << coarse_bytes_per_fine_dof << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "all";
        const double fine_tolerance = argc > 2
            ? parse_tolerance(argv[2], "fine tolerance")
            : 1.0e-2;
        const double coarse_tolerance = argc > 3
            ? parse_tolerance(argv[3], "coarse tolerance")
            : fine_tolerance;
        const std::size_t max_iterations = argc > 4
            ? parse_size(argv[4], "max iterations")
            : 5000U;

        const auto cases = make_cases();
        std::size_t selected = 0U;

        std::cout << "GFSS M5 two-level coarse-seed experiment\n"
                  << "restriction=P^T variational nodal-force restriction\n"
                  << "prolongation=trilinear_P\n"
                  << "coarse_operator=rediscretized_structured_Q1\n"
                  << "fine_solver=audited_FP32_GoldSparse_Jacobi_PCG\n"
                  << "comparison=same_fine_audited_tolerance_zero_vs_coarse_seed\n";

        for (const auto& test : cases) {
            if (selector != "all" && selector != test.name) continue;
            ++selected;
            std::cout << "\n========================================\n";
            run_case(test, fine_tolerance, coarse_tolerance, max_iterations);
        }

        if (selected == 0U) {
            std::cerr << "error: unknown case '" << selector << "'\n"
                      << "available:";
            for (const auto& test : cases) std::cerr << ' ' << test.name;
            std::cerr << " all\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_two_level_seed_bench "
                  << "[all|case [fine_tol [coarse_tol [max_iterations]]]]\n";
        return 1;
    }
}
