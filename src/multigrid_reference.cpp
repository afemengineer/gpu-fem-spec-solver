#include "gfss/multigrid_reference.hpp"

#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_gold.hpp"
#include "gfss/cpu_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gfss {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kChebyshevLowerFraction = 0.10;
constexpr double kLambdaSafety = 1.25;

struct AxisInterpolation {
    std::uint32_t a{0};
    std::uint32_t b{0};
    double wa{1.0};
    double wb{0.0};
};

struct LevelData {
    StructuredHexMesh mesh{};
    std::vector<double> inv_diag;
    double lambda_max{0.0};
};

int axis_class(std::uint32_t coordinate, std::uint32_t max_coordinate) {
    return coordinate == 0U ? 0 : (coordinate == max_coordinate ? 2 : 1);
}

AxisInterpolation axis_interpolation(std::uint32_t fine_index) {
    if ((fine_index & 1U) == 0U) {
        const std::uint32_t c = fine_index / 2U;
        return {c, c, 1.0, 0.0};
    }
    const std::uint32_t c = fine_index / 2U;
    return {c, c + 1U, 0.5, 0.5};
}

bool can_coarsen(const StructuredHexMesh& mesh) {
    return mesh.nx > 1U && mesh.ny > 1U && mesh.nz > 1U &&
           (mesh.nx & 1U) == 0U &&
           (mesh.ny & 1U) == 0U &&
           (mesh.nz & 1U) == 0U;
}

StructuredHexMesh coarsen(const StructuredHexMesh& fine) {
    if (!can_coarsen(fine)) {
        throw std::invalid_argument("geometric V-cycle attempted invalid 2x coarsening");
    }
    return {fine.nx / 2U,
            fine.ny / 2U,
            fine.nz / 2U,
            fine.lx,
            fine.ly,
            fine.lz};
}

void clamp_x0(const StructuredHexMesh& mesh, std::vector<double>& v) {
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(0U, j, k);
            const auto base = static_cast<std::size_t>(3ULL * node);
            v[base + 0U] = 0.0;
            v[base + 1U] = 0.0;
            v[base + 2U] = 0.0;
        }
    }
}

std::vector<double> apply_clamped_openmp(const StructuredHexMesh& mesh,
                                          const Material& material,
                                          const std::vector<double>& x) {
    if (x.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("multigrid operator vector size mismatch");
    }
    auto x_free = x;
    clamp_x0(mesh, x_free);
    auto y = apply_matrix_free_openmp(mesh, material, x_free);
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(0U, j, k);
            const auto base = static_cast<std::size_t>(3ULL * node);
            y[base + 0U] = x[base + 0U];
            y[base + 1U] = x[base + 1U];
            y[base + 2U] = x[base + 2U];
        }
    }
    return y;
}

std::vector<double> build_inverse_diagonal(const StructuredHexMesh& mesh,
                                           const Material& material) {
    const auto stencil = build_cpu_gold_stencil_fp32(mesh, material);
    std::vector<double> inv_diag(static_cast<std::size_t>(mesh.dof_count()), 0.0);

    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        const int cz = axis_class(k, mesh.nz);
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const int cy = axis_class(j, mesh.ny);
            for (std::uint32_t i = 0; i <= mesh.nx; ++i) {
                const auto node = mesh.node_index(i, j, k);
                const auto base = static_cast<std::size_t>(3ULL * node);
                if (i == 0U) {
                    continue;
                }
                const int cx = axis_class(i, mesh.nx);
                const std::size_t cls = static_cast<std::size_t>(cx + 3 * (cy + 3 * cz));
                const std::size_t count = stencil.regular.counts[cls];
                bool found = false;
                for (std::size_t e = 0; e < count; ++e) {
                    const auto& entry = stencil.regular.entries[cls][e];
                    if (entry.dx == 0 && entry.dy == 0 && entry.dz == 0) {
                        const double d0 = static_cast<double>(entry.block[0]);
                        const double d1 = static_cast<double>(entry.block[4]);
                        const double d2 = static_cast<double>(entry.block[8]);
                        if (!(d0 > 0.0) || !(d1 > 0.0) || !(d2 > 0.0)) {
                            throw std::runtime_error("multigrid Jacobi diagonal is not positive");
                        }
                        inv_diag[base + 0U] = 1.0 / d0;
                        inv_diag[base + 1U] = 1.0 / d1;
                        inv_diag[base + 2U] = 1.0 / d2;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw std::runtime_error("multigrid could not locate stencil diagonal");
                }
            }
        }
    }
    return inv_diag;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("multigrid dot size mismatch");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double norm(const std::vector<double>& v) {
    return std::sqrt(std::max(0.0, dot(v, v)));
}

double estimate_lambda_max(const StructuredHexMesh& mesh,
                           const Material& material,
                           const std::vector<double>& inv_diag,
                           std::size_t power_iterations) {
    if (power_iterations == 0U) {
        throw std::invalid_argument("multigrid power_iterations must be positive");
    }
    std::vector<double> q(inv_diag.size(), 0.0);
    std::vector<double> scaled(inv_diag.size(), 0.0);

    for (std::size_t i = 0; i < q.size(); ++i) {
        if (inv_diag[i] > 0.0) {
            const double t = static_cast<double>((i % 251U) + 1U);
            q[i] = std::sin(0.173 * t) + 0.37 * std::cos(0.071 * t);
        }
    }
    double qnorm = norm(q);
    if (!(qnorm > 0.0)) {
        throw std::runtime_error("multigrid lambda estimate initial vector is zero");
    }
    for (double& value : q) value /= qnorm;

    double lambda = 0.0;
    for (std::size_t it = 0; it < power_iterations; ++it) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            scaled[i] = inv_diag[i] > 0.0
                ? std::sqrt(inv_diag[i]) * q[i]
                : 0.0;
        }
        auto y = apply_clamped_openmp(mesh, material, scaled);
        for (std::size_t i = 0; i < y.size(); ++i) {
            y[i] = inv_diag[i] > 0.0
                ? std::sqrt(inv_diag[i]) * y[i]
                : 0.0;
        }
        const double rayleigh = dot(q, y);
        if (!(rayleigh > 0.0) || !std::isfinite(rayleigh)) {
            throw std::runtime_error("multigrid lambda estimate became invalid");
        }
        lambda = std::max(lambda, rayleigh);
        const double ynorm = norm(y);
        if (!(ynorm > 0.0) || !std::isfinite(ynorm)) {
            throw std::runtime_error("multigrid lambda estimate vector became invalid");
        }
        q = std::move(y);
        for (double& value : q) value /= ynorm;
    }
    return kLambdaSafety * lambda;
}

std::vector<double> restrict_pt(const StructuredHexMesh& fine,
                                const StructuredHexMesh& coarse,
                                const std::vector<double>& fine_v) {
    if (fine_v.size() != static_cast<std::size_t>(fine.dof_count())) {
        throw std::invalid_argument("multigrid restriction size mismatch");
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
                            coarse_v[cb + 0U] += w * fine_v[fb + 0U];
                            coarse_v[cb + 1U] += w * fine_v[fb + 1U];
                            coarse_v[cb + 2U] += w * fine_v[fb + 2U];
                        }
                    }
                }
            }
        }
    }
    clamp_x0(coarse, coarse_v);
    return coarse_v;
}

std::vector<double> prolongate(const StructuredHexMesh& fine,
                               const StructuredHexMesh& coarse,
                               const std::vector<double>& coarse_v) {
    if (coarse_v.size() != static_cast<std::size_t>(coarse.dof_count())) {
        throw std::invalid_argument("multigrid prolongation size mismatch");
    }
    std::vector<double> fine_v(static_cast<std::size_t>(fine.dof_count()), 0.0);
    for (std::uint32_t k = 0; k <= fine.nz; ++k) {
        const auto wz = axis_interpolation(k);
        for (std::uint32_t j = 0; j <= fine.ny; ++j) {
            const auto wy = axis_interpolation(j);
            for (std::uint32_t i = 1U; i <= fine.nx; ++i) {
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
                            fine_v[fb + 0U] += w * coarse_v[cb + 0U];
                            fine_v[fb + 1U] += w * coarse_v[cb + 1U];
                            fine_v[fb + 2U] += w * coarse_v[cb + 2U];
                        }
                    }
                }
            }
        }
    }
    return fine_v;
}

void chebyshev_jacobi_smooth(const LevelData& level,
                             const Material& material,
                             const std::vector<double>& b,
                             std::vector<double>& x,
                             std::size_t degree) {
    if (degree == 0U) return;
    if (!(level.lambda_max > 0.0)) {
        throw std::runtime_error("multigrid smoother missing lambda_max");
    }
    const double lambda_low = kChebyshevLowerFraction * level.lambda_max;
    const double theta = 0.5 * (level.lambda_max + lambda_low);
    const double delta = 0.5 * (level.lambda_max - lambda_low);

    for (std::size_t k = 0; k < degree; ++k) {
        const double angle = kPi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        // Traverse roots from the high end of the target spectrum toward the
        // low end. The polynomial is unchanged, but this avoids applying the
        // largest Richardson coefficient to an unsmoothed state first.
        const double root = theta + delta * std::cos(angle);
        if (!(root > 0.0)) {
            throw std::runtime_error("multigrid Chebyshev root is not positive");
        }
        const double omega = 1.0 / root;
        const auto ax = apply_clamped_openmp(level.mesh, material, x);
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (level.inv_diag[i] > 0.0) {
                x[i] += omega * level.inv_diag[i] * (b[i] - ax[i]);
            } else {
                x[i] = 0.0;
            }
        }
    }
}

void vcycle(std::size_t level_index,
            const std::vector<LevelData>& levels,
            const Material& material,
            const std::vector<double>& b,
            std::vector<double>& x,
            std::size_t pre_degree,
            std::size_t post_degree) {
    const auto& level = levels[level_index];
    if (level_index + 1U == levels.size()) {
        const auto apply = [&](const std::vector<double>& v) {
            return apply_clamped_openmp(level.mesh, material, v);
        };
        const auto bottom = conjugate_gradient(apply, b, 1.0e-11, 5000U);
        if (!bottom.converged) {
            throw std::runtime_error("multigrid bottom FP64 CG did not converge");
        }
        x = bottom.x;
        clamp_x0(level.mesh, x);
        return;
    }

    chebyshev_jacobi_smooth(level, material, b, x, pre_degree);
    const auto ax = apply_clamped_openmp(level.mesh, material, x);
    std::vector<double> r(b.size(), 0.0);
    for (std::size_t i = 0; i < b.size(); ++i) {
        r[i] = b[i] - ax[i];
    }
    clamp_x0(level.mesh, r);

    const auto& coarse = levels[level_index + 1U];
    const auto coarse_b = restrict_pt(level.mesh, coarse.mesh, r);
    std::vector<double> coarse_e(coarse_b.size(), 0.0);
    vcycle(level_index + 1U,
           levels,
           material,
           coarse_b,
           coarse_e,
           pre_degree,
           post_degree);
    const auto correction = prolongate(level.mesh, coarse.mesh, coarse_e);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] += correction[i];
    }
    clamp_x0(level.mesh, x);
    chebyshev_jacobi_smooth(level, material, b, x, post_degree);
}

double relative_residual(const StructuredHexMesh& mesh,
                         const Material& material,
                         const std::vector<double>& b,
                         const std::vector<double>& x) {
    const auto ax = apply_clamped_openmp(mesh, material, x);
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

GeometricVcycleReferenceResult solve_geometric_vcycle_reference_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    double relative_tolerance,
    std::size_t max_cycles,
    std::size_t pre_smooth_degree,
    std::size_t post_smooth_degree,
    std::size_t power_iterations) {
    if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
        throw std::invalid_argument("multigrid requires non-empty mesh dimensions");
    }
    if (rhs.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("multigrid RHS size mismatch");
    }
    if (!(relative_tolerance > 0.0) || relative_tolerance >= 1.0) {
        throw std::invalid_argument("multigrid relative tolerance must be in (0,1)");
    }
    if (max_cycles == 0U || power_iterations == 0U) {
        throw std::invalid_argument("multigrid cycle/power counts must be positive");
    }

    auto b = rhs;
    clamp_x0(mesh, b);
    const auto setup_start = Clock::now();

    std::vector<StructuredHexMesh> meshes;
    meshes.push_back(mesh);
    while (can_coarsen(meshes.back())) {
        meshes.push_back(coarsen(meshes.back()));
    }
    if (meshes.size() < 2U) {
        throw std::invalid_argument("multigrid reference requires at least two geometric levels");
    }

    std::vector<LevelData> levels;
    levels.reserve(meshes.size());
    for (std::size_t i = 0; i < meshes.size(); ++i) {
        LevelData level;
        level.mesh = meshes[i];
        level.inv_diag = build_inverse_diagonal(level.mesh, material);
        if (i + 1U < meshes.size()) {
            level.lambda_max = estimate_lambda_max(
                level.mesh, material, level.inv_diag, power_iterations);
        }
        levels.push_back(std::move(level));
    }

    GeometricVcycleReferenceResult result;
    result.x.assign(rhs.size(), 0.0);
    result.levels.reserve(levels.size());
    std::uint64_t coarse_dofs = 0ULL;
    for (std::size_t i = 0; i < levels.size(); ++i) {
        result.levels.push_back(GeometricVcycleLevelInfo{
            levels[i].mesh,
            static_cast<std::size_t>(levels[i].mesh.dof_count()),
            levels[i].lambda_max});
        if (i > 0U) coarse_dofs += levels[i].mesh.dof_count();
    }
    const double fine_dofs = static_cast<double>(mesh.dof_count());
    result.estimated_three_vector_coarse_bytes_per_fine_dof =
        12.0 * static_cast<double>(coarse_dofs) / fine_dofs;
    result.estimated_six_vector_coarse_bytes_per_fine_dof =
        24.0 * static_cast<double>(coarse_dofs) / fine_dofs;

    const auto setup_stop = Clock::now();
    result.setup_ms = std::chrono::duration<double, std::milli>(
        setup_stop - setup_start).count();

    result.relative_residuals.push_back(relative_residual(mesh, material, b, result.x));
    if (result.relative_residuals.back() <= relative_tolerance) {
        result.converged = true;
        return result;
    }

    const auto solve_start = Clock::now();
    for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
        vcycle(0U,
               levels,
               material,
               b,
               result.x,
               pre_smooth_degree,
               post_smooth_degree);
        result.cycles = cycle + 1U;
        const double rel = relative_residual(mesh, material, b, result.x);
        if (!std::isfinite(rel)) {
            throw std::runtime_error("multigrid true residual became non-finite");
        }
        result.relative_residuals.push_back(rel);
        if (rel <= relative_tolerance) {
            result.converged = true;
            break;
        }
    }
    const auto solve_stop = Clock::now();
    result.solve_ms = std::chrono::duration<double, std::milli>(
        solve_stop - solve_start).count();
    return result;
}

}  // namespace gfss
