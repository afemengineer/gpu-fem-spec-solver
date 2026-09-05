#include "gfss/smoothed_aggregation_reference.hpp"

#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/aggregation_two_grid_reference.hpp"
#include "gfss/cpu_elasticity.hpp"
#include "gfss/hex8.hpp"

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
constexpr double kSaDampingNumerator = 4.0 / 3.0;

struct CoarseCgResult {
    bool converged{false};
    std::size_t iterations{0};
    double relative_residual{0.0};
    std::vector<double> x;
};

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("smoothed aggregation dot size mismatch");
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}

double norm(const std::vector<double>& v) {
    return std::sqrt(std::max(0.0, dot(v, v)));
}

void clamp_x0(const StructuredHexMesh& mesh, std::vector<double>& v) {
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(0U, j, k);
            const std::size_t base = static_cast<std::size_t>(3ULL * node);
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
        throw std::invalid_argument("smoothed aggregation fine operator size mismatch");
    }
    auto free_x = x;
    clamp_x0(mesh, free_x);
    auto y = apply_matrix_free_openmp(mesh, material, free_x);
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(0U, j, k);
            const std::size_t base = static_cast<std::size_t>(3ULL * node);
            y[base + 0U] = x[base + 0U];
            y[base + 1U] = x[base + 1U];
            y[base + 2U] = x[base + 2U];
        }
    }
    return y;
}

std::vector<double> build_inverse_diagonal(const StructuredHexMesh& mesh,
                                           const Material& material,
                                           const ElasticityAggregationCoarseSpace& space) {
    std::vector<double> diagonal(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const auto ke = hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                for (std::size_t a = 0; a < 8U; ++a) {
                    const auto node = static_cast<std::size_t>(nodes[a]);
                    if (space.graph.constrained[node] != 0U) continue;
                    for (std::size_t c = 0; c < 3U; ++c) {
                        const std::size_t local = 3U * a + c;
                        diagonal[3U * node + c] += ke[local][local];
                    }
                }
            }
        }
    }

    std::vector<double> inverse(diagonal.size(), 0.0);
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        if (space.graph.constrained[node] != 0U) continue;
        for (std::size_t c = 0; c < 3U; ++c) {
            const std::size_t dof = 3U * node + c;
            if (!(diagonal[dof] > 0.0) || !std::isfinite(diagonal[dof])) {
                throw std::runtime_error("smoothed aggregation Jacobi diagonal is invalid");
            }
            inverse[dof] = 1.0 / diagonal[dof];
        }
    }
    return inverse;
}

double estimate_lambda_max(const StructuredHexMesh& mesh,
                           const Material& material,
                           const std::vector<double>& inverse_diagonal,
                           std::size_t power_iterations) {
    if (power_iterations == 0U) throw std::invalid_argument("power_iterations must be positive");
    std::vector<double> q(inverse_diagonal.size(), 0.0);
    std::vector<double> scaled(q.size(), 0.0);
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (inverse_diagonal[i] > 0.0) {
            const double t = static_cast<double>((i % 251U) + 1U);
            q[i] = std::sin(0.173 * t) + 0.37 * std::cos(0.071 * t);
        }
    }
    double qnorm = norm(q);
    if (!(qnorm > 0.0)) throw std::runtime_error("smoothed aggregation lambda probe is zero");
    for (double& value : q) value /= qnorm;

    double lambda = 0.0;
    for (std::size_t it = 0; it < power_iterations; ++it) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            scaled[i] = inverse_diagonal[i] > 0.0 ? std::sqrt(inverse_diagonal[i]) * q[i] : 0.0;
        }
        auto y = apply_clamped_openmp(mesh, material, scaled);
        for (std::size_t i = 0; i < y.size(); ++i) {
            y[i] = inverse_diagonal[i] > 0.0 ? std::sqrt(inverse_diagonal[i]) * y[i] : 0.0;
        }
        const double rq = dot(q, y);
        const double ynorm = norm(y);
        if (!(rq > 0.0) || !(ynorm > 0.0) || !std::isfinite(rq) || !std::isfinite(ynorm)) {
            throw std::runtime_error("smoothed aggregation lambda estimate became invalid");
        }
        lambda = std::max(lambda, rq);
        q = std::move(y);
        for (double& value : q) value /= ynorm;
    }
    return kLambdaSafety * lambda;
}

void chebyshev_jacobi_smooth(const StructuredHexMesh& mesh,
                             const Material& material,
                             const std::vector<double>& inverse_diagonal,
                             double lambda_max,
                             const std::vector<double>& b,
                             std::vector<double>& x,
                             std::size_t degree) {
    if (degree == 0U) return;
    const double lambda_low = kChebyshevLowerFraction * lambda_max;
    const double theta = 0.5 * (lambda_max + lambda_low);
    const double delta = 0.5 * (lambda_max - lambda_low);
    for (std::size_t k = 0; k < degree; ++k) {
        const double angle = kPi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        const double root = theta + delta * std::cos(angle);
        if (!(root > 0.0)) throw std::runtime_error("smoothed aggregation Chebyshev root invalid");
        const auto ax = apply_clamped_openmp(mesh, material, x);
        const double weight = 1.0 / root;
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (inverse_diagonal[i] > 0.0) x[i] += weight * inverse_diagonal[i] * (b[i] - ax[i]);
        }
        clamp_x0(mesh, x);
    }
}

struct FactorizedTransfer {
    const StructuredHexMesh& mesh;
    const Material& material;
    const ElasticityAggregationCoarseSpace& space;
    const std::vector<double>& inverse_diagonal;
    double omega{0.0};
    std::size_t steps{0};

    std::vector<double> prolong(const std::vector<double>& coarse) const {
        auto fine = apply_elasticity_tentative_prolongation(space, coarse);
        for (std::size_t step = 0; step < steps; ++step) {
            const auto af = apply_clamped_openmp(mesh, material, fine);
            for (std::size_t i = 0; i < fine.size(); ++i) {
                if (inverse_diagonal[i] > 0.0) fine[i] -= omega * inverse_diagonal[i] * af[i];
            }
            clamp_x0(mesh, fine);
        }
        return fine;
    }

    std::vector<double> restrict_transpose(const std::vector<double>& fine) const {
        if (fine.size() != inverse_diagonal.size()) {
            throw std::invalid_argument("smoothed aggregation transpose input size mismatch");
        }
        auto work = fine;
        clamp_x0(mesh, work);
        std::vector<double> scaled(work.size(), 0.0);
        for (std::size_t step = 0; step < steps; ++step) {
            for (std::size_t i = 0; i < work.size(); ++i) {
                scaled[i] = inverse_diagonal[i] * work[i];
            }
            const auto a_scaled = apply_clamped_openmp(mesh, material, scaled);
            for (std::size_t i = 0; i < work.size(); ++i) work[i] -= omega * a_scaled[i];
            clamp_x0(mesh, work);
        }
        return apply_elasticity_tentative_restriction(space, work);
    }

    std::vector<double> apply_coarse(const std::vector<double>& coarse) const {
        const auto fine = prolong(coarse);
        const auto af = apply_clamped_openmp(mesh, material, fine);
        return restrict_transpose(af);
    }
};

std::vector<double> deterministic_probe(std::size_t n, double phase) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = std::sin(0.017 * t + phase) + 0.29 * std::cos(0.043 * t - phase);
    }
    const double nrm = norm(v);
    if (!(nrm > 0.0)) throw std::runtime_error("smoothed aggregation probe is zero");
    for (double& value : v) value /= nrm;
    return v;
}

CoarseCgResult solve_coarse_cg(const FactorizedTransfer& transfer,
                               const std::vector<double>& inverse_preconditioner,
                               const std::vector<double>& b,
                               double relative_tolerance,
                               std::size_t max_iterations) {
    if (b.size() != transfer.space.coarse_dofs || b.size() != inverse_preconditioner.size()) {
        throw std::invalid_argument("smoothed aggregation coarse CG size mismatch");
    }
    CoarseCgResult result;
    result.x.assign(b.size(), 0.0);
    const double bnorm = norm(b);
    if (bnorm == 0.0) {
        result.converged = true;
        return result;
    }

    std::vector<double> r = b;
    std::vector<double> z(b.size(), 0.0);
    std::vector<double> p(b.size(), 0.0);
    for (std::size_t i = 0; i < b.size(); ++i) {
        z[i] = inverse_preconditioner[i] * r[i];
        p[i] = z[i];
    }
    double rho = dot(r, z);
    if (!(rho > 0.0) || !std::isfinite(rho)) {
        throw std::runtime_error("smoothed aggregation coarse CG initial rho invalid");
    }

    for (std::size_t it = 0; it < max_iterations; ++it) {
        const auto q = transfer.apply_coarse(p);
        const double pq = dot(p, q);
        if (!(pq > 0.0) || !std::isfinite(pq)) {
            throw std::runtime_error("smoothed aggregation coarse CG lost positive curvature");
        }
        const double alpha = rho / pq;
        for (std::size_t i = 0; i < b.size(); ++i) {
            result.x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        result.iterations = it + 1U;
        result.relative_residual = norm(r) / bnorm;
        if (!std::isfinite(result.relative_residual)) {
            throw std::runtime_error("smoothed aggregation coarse CG residual invalid");
        }
        if (result.relative_residual <= relative_tolerance) {
            result.converged = true;
            return result;
        }
        for (std::size_t i = 0; i < b.size(); ++i) z[i] = inverse_preconditioner[i] * r[i];
        const double rho_new = dot(r, z);
        if (!(rho_new > 0.0) || !std::isfinite(rho_new)) {
            throw std::runtime_error("smoothed aggregation coarse CG rho invalid");
        }
        const double beta = rho_new / rho;
        for (std::size_t i = 0; i < b.size(); ++i) p[i] = z[i] + beta * p[i];
        rho = rho_new;
    }
    return result;
}

}  // namespace

SmoothedAggregationResult solve_smoothed_aggregation_two_grid_reference(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    const SmoothedAggregationOptions& options) {
    if (rhs.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("smoothed aggregation RHS size mismatch");
    }
    if (!(options.true_relative_tolerance > 0.0) ||
        !(options.coarse_relative_tolerance > 0.0) ||
        options.max_cycles == 0U || options.coarse_max_iterations == 0U ||
        options.power_iterations == 0U) {
        throw std::invalid_argument("smoothed aggregation options invalid");
    }

    SmoothedAggregationResult result;
    result.fine_dofs = static_cast<std::size_t>(mesh.dof_count());
    result.transfer_smoothing_steps = options.transfer_smoothing_steps;
    result.fine_operator_applies_per_coarse_apply = 2U * options.transfer_smoothing_steps + 1U;
    const auto total_start = Clock::now();

    const auto aggregation_start = Clock::now();
    auto graph = build_structured_hex_nodal_graph_x0(mesh);
    const auto space = build_elasticity_aggregation_coarse_space(
        std::move(graph),
        {options.target_nodes_per_aggregate,
         options.min_nodes_per_aggregate,
         options.rank_tolerance});
    const auto aggregation_stop = Clock::now();
    result.aggregation_setup_ms = elapsed_ms(aggregation_start, aggregation_stop);
    result.fine_free_dofs = space.fine_free_dofs;
    result.coarse_dofs = space.coarse_dofs;
    result.aggregates = space.aggregates.size();
    result.matrix_free_transfer_bytes_per_fine_free_dof =
        static_cast<double>(space.estimated_matrix_free_transfer_payload_bytes) /
        static_cast<double>(space.fine_free_dofs);
    // Production FP32 factorization needs one fine Jacobi inverse diagonal in
    // addition to the existing tentative-transfer metadata.
    result.fp32_factorized_transfer_extra_bytes_per_fine_free_dof = 4.0;

    const auto precond_start = Clock::now();
    const auto tentative_matrix = assemble_structured_hex_aggregation_galerkin(mesh, material, space);
    const auto precond_stop = Clock::now();
    result.tentative_coarse_preconditioner_setup_ms = elapsed_ms(precond_start, precond_stop);

    const auto spectral_start = Clock::now();
    const auto inverse_diagonal = build_inverse_diagonal(mesh, material, space);
    result.fine_lambda_max = estimate_lambda_max(mesh, material, inverse_diagonal, options.power_iterations);
    result.transfer_omega = kSaDampingNumerator / result.fine_lambda_max;
    const auto spectral_stop = Clock::now();
    result.spectral_setup_ms = elapsed_ms(spectral_start, spectral_stop);

    const FactorizedTransfer transfer{
        mesh, material, space, inverse_diagonal,
        result.transfer_omega, options.transfer_smoothing_steps};

    const auto coarse_probe = deterministic_probe(space.coarse_dofs, 0.31);
    auto fine_probe = deterministic_probe(static_cast<std::size_t>(mesh.dof_count()), 0.67);
    clamp_x0(mesh, fine_probe);
    const auto pc = transfer.prolong(coarse_probe);
    const auto ptf = transfer.restrict_transpose(fine_probe);
    const double lhs = dot(pc, fine_probe);
    const double rhs_adjoint = dot(coarse_probe, ptf);
    result.transfer_adjoint_relative_error =
        std::abs(lhs - rhs_adjoint) /
        std::max({std::abs(lhs), std::abs(rhs_adjoint), 1.0});

    const auto u = deterministic_probe(space.coarse_dofs, 0.17);
    const auto v = deterministic_probe(space.coarse_dofs, 0.73);
    const auto au = transfer.apply_coarse(u);
    const auto av = transfer.apply_coarse(v);
    const double uv = dot(u, av);
    const double vu = dot(v, au);
    result.coarse_symmetry_relative_defect =
        std::abs(uv - vu) / std::max({std::abs(uv), std::abs(vu), 1.0});
    result.coarse_spd_probe = dot(u, au) > 0.0;

    const double bnorm = norm(rhs);
    if (!(bnorm > 0.0) || !std::isfinite(bnorm)) {
        throw std::invalid_argument("smoothed aggregation requires finite non-zero RHS");
    }

    std::vector<double> x(rhs.size(), 0.0);
    std::vector<double> residual(rhs.size(), 0.0);
    const auto solve_start = Clock::now();
    auto ax = apply_clamped_openmp(mesh, material, x);
    for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
    result.true_relative_residuals.push_back(norm(residual) / bnorm);

    for (std::size_t cycle = 0; cycle < options.max_cycles; ++cycle) {
        chebyshev_jacobi_smooth(
            mesh, material, inverse_diagonal, result.fine_lambda_max,
            rhs, x, options.pre_smooth_degree);

        ax = apply_clamped_openmp(mesh, material, x);
        for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
        const auto coarse_rhs = transfer.restrict_transpose(residual);
        const auto coarse = solve_coarse_cg(
            transfer,
            tentative_matrix.inverse_diagonal,
            coarse_rhs,
            options.coarse_relative_tolerance,
            options.coarse_max_iterations);
        result.coarse_iterations.push_back(coarse.iterations);
        result.coarse_final_relative_residuals.push_back(coarse.relative_residual);
        if (!coarse.converged) {
            throw std::runtime_error("smoothed aggregation matrix-free coarse CG did not converge");
        }

        const auto correction = transfer.prolong(coarse.x);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += correction[i];
        clamp_x0(mesh, x);

        chebyshev_jacobi_smooth(
            mesh, material, inverse_diagonal, result.fine_lambda_max,
            rhs, x, options.post_smooth_degree);

        ax = apply_clamped_openmp(mesh, material, x);
        for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
        const double relative = norm(residual) / bnorm;
        const double previous = result.true_relative_residuals.back();
        result.true_relative_residuals.push_back(relative);
        result.cycle_contractions.push_back(relative / previous);
        result.cycles = cycle + 1U;
        if (!std::isfinite(relative)) {
            throw std::runtime_error("smoothed aggregation true residual became invalid");
        }
        if (relative <= options.true_relative_tolerance) {
            result.converged = true;
            break;
        }
    }

    const auto solve_stop = Clock::now();
    result.solve_ms = elapsed_ms(solve_start, solve_stop);
    result.total_ms = elapsed_ms(total_start, solve_stop);
    return result;
}

}  // namespace gfss
