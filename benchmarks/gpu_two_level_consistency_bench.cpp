#include "gfss/cpu_elasticity.hpp"
#include "gfss/gpu_solver.hpp"

#include <algorithm>
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
    if (a.size() != b.size()) throw std::invalid_argument("relative_difference size mismatch");
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
    double rr = 0.0;
    double bb = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double r = b[i] - ax[i];
        rr += r * r;
        bb += b[i] * b[i];
    }
    return bb > 0.0 ? std::sqrt(rr / bb) : 0.0;
}

}  // namespace

int main() {
    try {
        const gfss::StructuredHexMesh fine{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const auto coarse = make_coarse_mesh(fine);
        const gfss::Material material{210.0e9, 0.30};

        const auto fine_rhs_d = make_rhs(fine);
        std::vector<float> fine_rhs_f(fine_rhs_d.begin(), fine_rhs_d.end());
        const auto coarse_rhs_d = restrict_pt(fine, coarse, fine_rhs_d);
        std::vector<float> coarse_rhs_f(coarse_rhs_d.begin(), coarse_rhs_d.end());

        const auto coarse_result = gfss::solve_pcg_cuda_gold_sparse_x0(
            coarse, material, coarse_rhs_f, 1.0e-2, 5000U, 4);
        const std::vector<double> coarse_x(coarse_result.x.begin(), coarse_result.x.end());

        const auto coarse_ax = gfss::apply_clamped_x0_matrix_free(coarse, material, coarse_x);
        const double coarse_true = relative_residual(coarse_rhs_d, coarse_ax);

        const auto fine_seed = prolongate(fine, coarse, coarse_x);
        const auto fine_ap = gfss::apply_clamped_x0_matrix_free(fine, material, fine_seed);
        const auto restricted_fine_ap = restrict_pt(fine, coarse, fine_ap);
        const double galerkin_mismatch = relative_difference(restricted_fine_ap, coarse_ax);

        std::vector<double> fine_seed_r(fine_rhs_d.size(), 0.0);
        for (std::size_t i = 0; i < fine_rhs_d.size(); ++i) {
            fine_seed_r[i] = fine_rhs_d[i] - fine_ap[i];
        }
        const auto projected_seed_r = restrict_pt(fine, coarse, fine_seed_r);
        const double projected_seed_rel =
            norm2(coarse_rhs_d) > 0.0 ? norm2(projected_seed_r) / norm2(coarse_rhs_d) : 0.0;
        const double seed_true = relative_residual(fine_rhs_d, fine_ap);

        double b_ap = 0.0;
        double ap_ap = 0.0;
        for (std::size_t i = 0; i < fine_rhs_d.size(); ++i) {
            b_ap += fine_rhs_d[i] * fine_ap[i];
            ap_ap += fine_ap[i] * fine_ap[i];
        }
        const double alpha_l2 = ap_ap > 0.0 ? b_ap / ap_ap : 0.0;
        std::vector<double> damped_ax(fine_ap.size(), 0.0);
        for (std::size_t i = 0; i < fine_ap.size(); ++i) {
            damped_ax[i] = alpha_l2 * fine_ap[i];
        }
        const double damped_true = relative_residual(fine_rhs_d, damped_ax);

        // Independent smooth probe for A_c ~= P^T A_f P, not tied to the
        // approximate coarse solution returned above.
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

        std::cout << std::scientific << std::setprecision(9)
                  << "GFSS M5 two-level consistency diagnostic\n"
                  << "fine_mesh=64x64x8 coarse_mesh=32x32x4\n"
                  << "coarse_gpu_audited_rel=" << coarse_result.audited_relative_residual << '\n'
                  << "coarse_cpu_fp64_true_rel=" << coarse_true << '\n'
                  << "galerkin_mismatch_on_coarse_solution=" << galerkin_mismatch << '\n'
                  << "galerkin_mismatch_on_smooth_probe=" << probe_galerkin_mismatch << '\n'
                  << "seed_fine_true_rel=" << seed_true << '\n'
                  << "projected_seed_residual_rel=" << projected_seed_rel << '\n'
                  << "residual_l2_optimal_alpha=" << alpha_l2 << '\n'
                  << "damped_seed_fine_true_rel=" << damped_true << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
