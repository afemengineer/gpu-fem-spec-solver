#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_solver.hpp"
#include "gfss/gpu_solver.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct AxisInterpolation {
    std::uint32_t a{0};
    std::uint32_t b{0};
    double wa{1.0};
    double wb{0.0};
};

AxisInterpolation axis_interpolation(std::uint32_t fine_index) {
    if ((fine_index & 1U) == 0U) {
        const std::uint32_t c = fine_index / 2U;
        return {c, c, 1.0, 0.0};
    }
    const std::uint32_t c = fine_index / 2U;
    return {c, c + 1U, 0.5, 0.5};
}

gfss::StructuredHexMesh make_coarse_mesh(const gfss::StructuredHexMesh& fine) {
    if ((fine.nx & 1U) != 0U || (fine.ny & 1U) != 0U ||
        (fine.nz & 1U) != 0U || fine.nx < 2U || fine.ny < 2U || fine.nz < 2U) {
        throw std::invalid_argument("diagnostic requires even fine element counts >= 2");
    }
    return {fine.nx / 2U, fine.ny / 2U, fine.nz / 2U,
            fine.lx, fine.ly, fine.lz};
}

std::vector<double> prolongate(const gfss::StructuredHexMesh& fine,
                               const gfss::StructuredHexMesh& coarse,
                               const std::vector<double>& coarse_x) {
    if (coarse_x.size() != static_cast<std::size_t>(coarse.dof_count())) {
        throw std::invalid_argument("prolongate coarse vector size mismatch");
    }
    std::vector<double> fine_x(static_cast<std::size_t>(fine.dof_count()), 0.0);
    for (std::uint32_t k = 0; k <= fine.nz; ++k) {
        const auto wz = axis_interpolation(k);
        for (std::uint32_t j = 0; j <= fine.ny; ++j) {
            const auto wy = axis_interpolation(j);
            for (std::uint32_t i = 0; i <= fine.nx; ++i) {
                if (i == 0U) continue;
                const auto wx = axis_interpolation(i);
                const auto fine_node = fine.node_index(i, j, k);
                const std::size_t fb = static_cast<std::size_t>(3ULL * fine_node);
                const std::uint32_t ci[2] = {wx.a, wx.b};
                const std::uint32_t cj[2] = {wy.a, wy.b};
                const std::uint32_t ck[2] = {wz.a, wz.b};
                const double wi[2] = {wx.wa, wx.wb};
                const double wj[2] = {wy.wa, wy.wb};
                const double wk[2] = {wz.wa, wz.wb};
                for (int dz = 0; dz < 2; ++dz) {
                    if (wk[dz] == 0.0) continue;
                    for (int dy = 0; dy < 2; ++dy) {
                        if (wj[dy] == 0.0) continue;
                        for (int dx = 0; dx < 2; ++dx) {
                            if (wi[dx] == 0.0) continue;
                            const double w = wi[dx] * wj[dy] * wk[dz];
                            const auto cn = coarse.node_index(ci[dx], cj[dy], ck[dz]);
                            const std::size_t cb = static_cast<std::size_t>(3ULL * cn);
                            for (std::size_t c = 0; c < 3U; ++c) {
                                fine_x[fb + c] += w * coarse_x[cb + c];
                            }
                        }
                    }
                }
            }
        }
    }
    return fine_x;
}

std::vector<double> restrict_pt(const gfss::StructuredHexMesh& fine,
                                const gfss::StructuredHexMesh& coarse,
                                const std::vector<double>& fine_v) {
    if (fine_v.size() != static_cast<std::size_t>(fine.dof_count())) {
        throw std::invalid_argument("restrict_pt fine vector size mismatch");
    }
    std::vector<double> coarse_v(static_cast<std::size_t>(coarse.dof_count()), 0.0);
    for (std::uint32_t k = 0; k <= fine.nz; ++k) {
        const auto wz = axis_interpolation(k);
        for (std::uint32_t j = 0; j <= fine.ny; ++j) {
            const auto wy = axis_interpolation(j);
            for (std::uint32_t i = 0; i <= fine.nx; ++i) {
                const auto wx = axis_interpolation(i);
                const auto fn = fine.node_index(i, j, k);
                const std::size_t fb = static_cast<std::size_t>(3ULL * fn);
                const std::uint32_t ci[2] = {wx.a, wx.b};
                const std::uint32_t cj[2] = {wy.a, wy.b};
                const std::uint32_t ck[2] = {wz.a, wz.b};
                const double wi[2] = {wx.wa, wx.wb};
                const double wj[2] = {wy.wa, wy.wb};
                const double wk[2] = {wz.wa, wz.wb};
                for (int dz = 0; dz < 2; ++dz) {
                    if (wk[dz] == 0.0) continue;
                    for (int dy = 0; dy < 2; ++dy) {
                        if (wj[dy] == 0.0) continue;
                        for (int dx = 0; dx < 2; ++dx) {
                            if (wi[dx] == 0.0) continue;
                            const double w = wi[dx] * wj[dy] * wk[dz];
                            const auto cn = coarse.node_index(ci[dx], cj[dy], ck[dz]);
                            const std::size_t cb = static_cast<std::size_t>(3ULL * cn);
                            for (std::size_t c = 0; c < 3U; ++c) {
                                coarse_v[cb + c] += w * fine_v[fb + c];
                            }
                        }
                    }
                }
            }
        }
    }
    for (std::uint32_t k = 0; k <= coarse.nz; ++k) {
        for (std::uint32_t j = 0; j <= coarse.ny; ++j) {
            const auto n = coarse.node_index(0U, j, k);
            const std::size_t b = static_cast<std::size_t>(3ULL * n);
            coarse_v[b + 0U] = 0.0;
            coarse_v[b + 1U] = 0.0;
            coarse_v[b + 2U] = 0.0;
        }
    }
    return coarse_v;
}

std::vector<double> make_rhs(const gfss::StructuredHexMesh& mesh) {
    std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const double value = -1.0 /
        (static_cast<double>(mesh.ny + 1U) * static_cast<double>(mesh.nz + 1U));
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto n = mesh.node_index(mesh.nx, j, k);
            rhs[static_cast<std::size_t>(3ULL * n + 2ULL)] = value;
        }
    }
    return rhs;
}

double norm2(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

double relative_difference(const std::vector<double>& a,
                           const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("relative_difference size mismatch");
    }
    double dd = 0.0;
    double bb = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        dd += d * d;
        bb += b[i] * b[i];
    }
    return bb > 0.0 ? std::sqrt(dd / bb) : std::sqrt(dd);
}

double relative_residual(const std::vector<double>& b,
                         const std::vector<double>& ax) {
    if (b.size() != ax.size()) {
        throw std::invalid_argument("relative_residual size mismatch");
    }
    double rr = 0.0;
    double bb = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double r = b[i] - ax[i];
        rr += r * r;
        bb += b[i] * b[i];
    }
    return bb > 0.0 ? std::sqrt(rr / bb) : 0.0;
}

struct SeedDiagnostics {
    double coarse_true_rel{0.0};
    double galerkin_mismatch{0.0};
    double seed_fine_true_rel{0.0};
    double projected_seed_residual_rel{0.0};
    double alpha_l2{0.0};
    double damped_seed_fine_true_rel{0.0};
};

SeedDiagnostics diagnose_coarse_solution(
    const gfss::StructuredHexMesh& fine,
    const gfss::StructuredHexMesh& coarse,
    const gfss::Material& material,
    const std::vector<double>& fine_rhs,
    const std::vector<double>& coarse_rhs,
    const std::vector<double>& coarse_x) {
    SeedDiagnostics d;
    const auto coarse_ax =
        gfss::apply_clamped_x0_matrix_free(coarse, material, coarse_x);
    d.coarse_true_rel = relative_residual(coarse_rhs, coarse_ax);

    const auto fine_seed = prolongate(fine, coarse, coarse_x);
    const auto fine_ap =
        gfss::apply_clamped_x0_matrix_free(fine, material, fine_seed);
    const auto restricted_fine_ap = restrict_pt(fine, coarse, fine_ap);
    d.galerkin_mismatch = relative_difference(restricted_fine_ap, coarse_ax);
    d.seed_fine_true_rel = relative_residual(fine_rhs, fine_ap);

    std::vector<double> fine_seed_r(fine_rhs.size(), 0.0);
    for (std::size_t i = 0; i < fine_rhs.size(); ++i) {
        fine_seed_r[i] = fine_rhs[i] - fine_ap[i];
    }
    const auto projected_seed_r = restrict_pt(fine, coarse, fine_seed_r);
    const double coarse_rhs_norm = norm2(coarse_rhs);
    d.projected_seed_residual_rel = coarse_rhs_norm > 0.0
        ? norm2(projected_seed_r) / coarse_rhs_norm
        : 0.0;

    double b_ap = 0.0;
    double ap_ap = 0.0;
    for (std::size_t i = 0; i < fine_rhs.size(); ++i) {
        b_ap += fine_rhs[i] * fine_ap[i];
        ap_ap += fine_ap[i] * fine_ap[i];
    }
    d.alpha_l2 = ap_ap > 0.0 ? b_ap / ap_ap : 0.0;
    std::vector<double> damped_ax(fine_ap.size(), 0.0);
    for (std::size_t i = 0; i < fine_ap.size(); ++i) {
        damped_ax[i] = d.alpha_l2 * fine_ap[i];
    }
    d.damped_seed_fine_true_rel = relative_residual(fine_rhs, damped_ax);
    return d;
}

}  // namespace

int main() {
    try {
        const gfss::StructuredHexMesh fine{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const auto coarse = make_coarse_mesh(fine);
        const gfss::Material material{210.0e9, 0.30};
        const auto fine_rhs = make_rhs(fine);
        const auto coarse_rhs = restrict_pt(fine, coarse, fine_rhs);

        // First: a solve-independent Galerkin identity check. This must run
        // even if the pathological thin-plate FP32 PCG breaks down.
        std::vector<double> probe(static_cast<std::size_t>(coarse.dof_count()), 0.0);
        for (std::uint32_t k = 0; k <= coarse.nz; ++k) {
            for (std::uint32_t j = 0; j <= coarse.ny; ++j) {
                for (std::uint32_t i = 1U; i <= coarse.nx; ++i) {
                    const auto n = coarse.node_index(i, j, k);
                    const std::size_t b = static_cast<std::size_t>(3ULL * n);
                    const double x = static_cast<double>(i) / coarse.nx;
                    const double y = static_cast<double>(j) / coarse.ny;
                    const double z = static_cast<double>(k) / coarse.nz;
                    probe[b + 0U] = x * (1.0 + 0.1 * y);
                    probe[b + 1U] = x * y * (1.0 - 0.2 * z);
                    probe[b + 2U] = x * (1.0 + y * z);
                }
            }
        }
        const auto coarse_probe_ax =
            gfss::apply_clamped_x0_matrix_free(coarse, material, probe);
        const auto fine_probe = prolongate(fine, coarse, probe);
        const auto fine_probe_ax =
            gfss::apply_clamped_x0_matrix_free(fine, material, fine_probe);
        const auto restricted_probe_ax = restrict_pt(fine, coarse, fine_probe_ax);
        const double probe_galerkin_mismatch =
            relative_difference(restricted_probe_ax, coarse_probe_ax);

        // Trusted FP64 coarse control solve. Its true residual and prolonged
        // correction tell us what the transfer does independently of FP32 PCG.
        const auto coarse_cpu = gfss::conjugate_gradient(
            [&](const std::vector<double>& x) {
                return gfss::apply_clamped_x0_matrix_free(coarse, material, x);
            },
            coarse_rhs,
            1.0e-10,
            20000U);
        const auto cpu_diag = diagnose_coarse_solution(
            fine, coarse, material, fine_rhs, coarse_rhs, coarse_cpu.x);

        // Optional FP32 GPU telemetry. A breakdown is itself a result and must
        // not suppress the solve-independent/FP64 consistency diagnostics.
        bool gpu_succeeded = false;
        std::string gpu_error;
        gfss::GpuPcgResult coarse_gpu;
        SeedDiagnostics gpu_diag;
        try {
            std::vector<float> coarse_rhs_f(coarse_rhs.size(), 0.0f);
            for (std::size_t i = 0; i < coarse_rhs.size(); ++i) {
                coarse_rhs_f[i] = static_cast<float>(coarse_rhs[i]);
            }
            coarse_gpu = gfss::solve_pcg_cuda_gold_sparse_x0(
                coarse, material, coarse_rhs_f, 1.0e-2, 5000U, 4);
            const std::vector<double> coarse_gpu_x(
                coarse_gpu.x.begin(), coarse_gpu.x.end());
            gpu_diag = diagnose_coarse_solution(
                fine, coarse, material, fine_rhs, coarse_rhs, coarse_gpu_x);
            gpu_succeeded = true;
        } catch (const std::exception& e) {
            gpu_error = e.what();
        }

        std::cout << std::scientific << std::setprecision(9)
                  << "GFSS M5 two-level consistency diagnostic\n"
                  << "fine_mesh=64x64x8 coarse_mesh=32x32x4\n"
                  << "galerkin_mismatch_on_smooth_probe="
                  << probe_galerkin_mismatch << '\n'
                  << "coarse_cpu_converged="
                  << (coarse_cpu.converged ? "true" : "false") << '\n'
                  << "coarse_cpu_iterations=" << coarse_cpu.iterations << '\n'
                  << "coarse_cpu_reported_rel=" << coarse_cpu.relative_residual << '\n'
                  << "coarse_cpu_fp64_true_rel=" << cpu_diag.coarse_true_rel << '\n'
                  << "galerkin_mismatch_on_cpu_coarse_solution="
                  << cpu_diag.galerkin_mismatch << '\n'
                  << "cpu_seed_fine_true_rel=" << cpu_diag.seed_fine_true_rel << '\n'
                  << "cpu_projected_seed_residual_rel="
                  << cpu_diag.projected_seed_residual_rel << '\n'
                  << "cpu_residual_l2_optimal_alpha=" << cpu_diag.alpha_l2 << '\n'
                  << "cpu_damped_seed_fine_true_rel="
                  << cpu_diag.damped_seed_fine_true_rel << '\n'
                  << "coarse_gpu_succeeded="
                  << (gpu_succeeded ? "true" : "false") << '\n';

        if (gpu_succeeded) {
            std::cout << "coarse_gpu_converged="
                      << (coarse_gpu.converged ? "true" : "false") << '\n'
                      << "coarse_gpu_iterations=" << coarse_gpu.iterations << '\n'
                      << "coarse_gpu_audited_rel="
                      << coarse_gpu.audited_relative_residual << '\n'
                      << "coarse_gpu_fp64_true_rel=" << gpu_diag.coarse_true_rel << '\n'
                      << "galerkin_mismatch_on_gpu_coarse_solution="
                      << gpu_diag.galerkin_mismatch << '\n'
                      << "gpu_seed_fine_true_rel=" << gpu_diag.seed_fine_true_rel << '\n'
                      << "gpu_projected_seed_residual_rel="
                      << gpu_diag.projected_seed_residual_rel << '\n'
                      << "gpu_residual_l2_optimal_alpha=" << gpu_diag.alpha_l2 << '\n'
                      << "gpu_damped_seed_fine_true_rel="
                      << gpu_diag.damped_seed_fine_true_rel << '\n';
        } else {
            std::cout << "coarse_gpu_error=" << gpu_error << '\n';
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
