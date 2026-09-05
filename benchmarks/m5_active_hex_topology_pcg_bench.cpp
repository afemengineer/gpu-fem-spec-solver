// PCG relevance test for the active-HEX topology diagnostic.
//
// Standalone two-grid residual contraction is not the production acceptance
// metric: an exact Galerkin coarse correction is an A-orthogonal projection and
// can increase ||r||_2 while reducing error energy.  This probe therefore uses
// the exact same shallow two-grid cycle as a PCG preconditioner and measures the
// quantity that matters to the M5 solver architecture: Krylov convergence and
// positivity of r^T M^{-1} r.  No aggregation/smoother parameters are tuned.
#define main gfss_m5_active_hex_topology_diagnostic_main
#include "m5_active_hex_topology_diagnostic_bench.cpp"
#undef main

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Precondition = std::function<Vec(const Vec&)>;

Vec apply_two_grid_preconditioner(const ActiveHexDomain& domain,
                                  const Apply& apply,
                                  const Vec& inverse,
                                  double lambda0,
                                  const GenericSmoothedTransfer& transfer,
                                  const DenseFactor& a1,
                                  const Vec& b) {
    if (b.size() != domain.dofs()) {
        throw std::invalid_argument("topology PCG two-grid RHS size mismatch");
    }
    Vec x(domain.dofs(), 0.0);
    const auto weights = chebyshev_weights(lambda0, 5U);
    smooth(domain, apply, inverse, weights, b, x);

    const auto ax = apply(x);
    Vec r(b.size(), 0.0);
    for (std::size_t i = 0U; i < b.size(); ++i) r[i] = b[i] - ax[i];
    const auto b1 = transfer.restrict_transpose(r);
    const auto x1 = a1.solve(b1);
    const auto corr = transfer.prolong(x1);
    for (std::size_t i = 0U; i < x.size(); ++i) x[i] += corr[i];

    smooth(domain, apply, inverse, weights, b, x);
    return x;
}

struct PcgResult {
    bool converged{false};
    bool breakdown{false};
    std::size_t iterations{0U};
    double final_true_relative_residual{1.0};
    double min_rz{std::numeric_limits<double>::infinity()};
    double min_pap{std::numeric_limits<double>::infinity()};
    std::vector<double> history;
};

PcgResult pcg_domain(const ActiveHexDomain& domain,
                     const Apply& apply,
                     const Vec& b,
                     const Precondition& precondition,
                     double tolerance,
                     std::size_t max_iterations) {
    if (!(tolerance > 0.0) || !(tolerance < 1.0) || max_iterations == 0U) {
        throw std::invalid_argument("topology PCG options invalid");
    }
    const double bnorm = norm(b);
    if (!(bnorm > 0.0)) throw std::runtime_error("topology PCG RHS is zero");

    PcgResult out;
    Vec x(b.size(), 0.0);
    Vec r = b;
    clamp(domain, r);
    out.history.push_back(norm(r) / bnorm);
    if (out.history.back() <= tolerance) {
        out.converged = true;
        out.final_true_relative_residual = out.history.back();
        return out;
    }

    Vec z = precondition(r);
    clamp(domain, z);
    double rz = dot(r, z);
    out.min_rz = rz;
    if (!(rz > 0.0) || !std::isfinite(rz)) {
        out.breakdown = true;
        out.final_true_relative_residual = out.history.back();
        return out;
    }
    Vec p = z;

    for (std::size_t it = 0U; it < max_iterations; ++it) {
        const auto ap = apply(p);
        const double pap = dot(p, ap);
        out.min_pap = std::min(out.min_pap, pap);
        if (!(pap > 0.0) || !std::isfinite(pap)) {
            out.breakdown = true;
            break;
        }

        const double alpha = rz / pap;
        if (!std::isfinite(alpha)) {
            out.breakdown = true;
            break;
        }
        for (std::size_t i = 0U; i < x.size(); ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        clamp(domain, x);
        clamp(domain, r);
        out.iterations = it + 1U;

        // Recompute the true residual every iteration. These systems are only
        // a few thousand DOFs, so the diagnostic should not depend on recursive
        // residual drift.
        const auto ax_true = apply(x);
        Vec r_true(b.size(), 0.0);
        for (std::size_t i = 0U; i < b.size(); ++i) r_true[i] = b[i] - ax_true[i];
        clamp(domain, r_true);
        const double rel = norm(r_true) / bnorm;
        out.history.push_back(rel);
        if (!std::isfinite(rel)) {
            out.breakdown = true;
            break;
        }
        if (rel <= tolerance) {
            out.converged = true;
            r = std::move(r_true);
            break;
        }

        // Continue from the recursively updated residual, mirroring production
        // PCG while retaining an independent true-residual audit above.
        z = precondition(r);
        clamp(domain, z);
        const double rz_new = dot(r, z);
        out.min_rz = std::min(out.min_rz, rz_new);
        if (!(rz_new > 0.0) || !std::isfinite(rz_new)) {
            out.breakdown = true;
            break;
        }
        const double beta = rz_new / rz;
        if (!std::isfinite(beta)) {
            out.breakdown = true;
            break;
        }
        for (std::size_t i = 0U; i < p.size(); ++i) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }

    out.final_true_relative_residual = out.history.empty() ? 1.0 : out.history.back();
    return out;
}

void print_pcg_result(const char* name, const PcgResult& result) {
    std::cout << name
              << "_converged=" << (result.converged ? "true" : "false")
              << " " << name << "_breakdown=" << (result.breakdown ? "true" : "false")
              << " " << name << "_iterations=" << result.iterations
              << std::scientific << std::setprecision(9)
              << " " << name << "_final_true_relative_residual="
              << result.final_true_relative_residual
              << " " << name << "_min_rz=" << result.min_rz
              << " " << name << "_min_pAp=" << result.min_pap << '\n';

    std::cout << name << "_true_residual_history=";
    for (std::size_t i = 0U; i < result.history.size(); ++i) {
        if (i != 0U) std::cout << ',';
        std::cout << std::scientific << std::setprecision(4) << result.history[i];
    }
    std::cout << '\n';
}

void run_pcg_case(const CaseDef& test,
                  std::size_t target_nodes,
                  std::size_t min_nodes,
                  double tolerance,
                  std::size_t max_iterations) {
    const auto domain = make_domain(test);
    auto graph = build_graph(domain);
    const std::size_t components = graph_components(graph);
    const auto space = gfss::build_elasticity_aggregation_coarse_space(
        std::move(graph), {target_nodes, min_nodes, 1.0e-10});
    const Apply apply = [&](const Vec& x) { return apply_fine(domain, x); };
    const auto inverse = inverse_diagonal(domain);
    const double lambda0 = estimate_lambda(apply, inverse, 8U);
    const double omega0 = kSaDampingNumerator / lambda0;

    const GenericSmoothedTransfer smoothed{domain, space, apply, inverse, omega0};
    const auto smoothed_a1 = materialize_and_factor_a1(smoothed, apply);
    const GenericSmoothedTransfer tentative{domain, space, apply, inverse, 0.0};
    const auto tentative_a1 = materialize_and_factor_a1(tentative, apply);
    const auto b = make_rhs(domain);

    const Precondition smoothed_m = [&](const Vec& r) {
        return apply_two_grid_preconditioner(
            domain, apply, inverse, lambda0, smoothed, smoothed_a1, r);
    };
    const Precondition tentative_m = [&](const Vec& r) {
        return apply_two_grid_preconditioner(
            domain, apply, inverse, lambda0, tentative, tentative_a1, r);
    };
    const Precondition jacobi_m = [&](const Vec& r) {
        Vec z(r.size(), 0.0);
        for (std::size_t i = 0U; i < r.size(); ++i) z[i] = inverse[i] * r[i];
        clamp(domain, z);
        return z;
    };

    const auto smoothed_result = pcg_domain(
        domain, apply, b, smoothed_m, tolerance, max_iterations);
    const auto tentative_result = pcg_domain(
        domain, apply, b, tentative_m, tolerance, max_iterations);
    const auto jacobi_result = pcg_domain(
        domain, apply, b, jacobi_m, tolerance, max_iterations);

    std::size_t rank_deficient = 0U;
    for (const auto& aggregate : space.aggregates) {
        rank_deficient += aggregate.rank < 6U ? 1U : 0U;
    }
    const double rb_error = gfss::audit_elasticity_rigid_body_reproduction(space);
    const double pt_error = adjoint_error(smoothed);

    std::cout << "case=" << test.name << '\n'
              << "description=" << test.description << '\n'
              << "active_elements=" << domain.elements.size()
              << " active_nodes=" << domain.nodes()
              << " dofs=" << domain.dofs()
              << " graph_components=" << components
              << " L1_dofs=" << space.coarse_dofs
              << " aggregates=" << space.aggregates.size()
              << " rank_deficient_aggregates=" << rank_deficient << '\n'
              << std::scientific << std::setprecision(12)
              << "rigid_body_reproduction_error=" << rb_error
              << " smoothed_transfer_adjoint_error=" << pt_error
              << " smoothed_A1_symmetry_relative_defect="
              << smoothed_a1.symmetry_relative_defect
              << " smoothed_A1_min_cholesky_pivot=" << smoothed_a1.min_pivot << '\n'
              << std::fixed << std::setprecision(6)
              << "lambda0=" << lambda0 << " omega0=" << omega0
              << " tolerance=" << std::scientific << tolerance
              << " max_iterations=" << std::fixed << max_iterations << '\n';

    print_pcg_result("smoothed_two_grid_pcg", smoothed_result);
    print_pcg_result("tentative_two_grid_pcg", tentative_result);
    print_pcg_result("jacobi_pcg", jacobi_result);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "all";
        const std::size_t target_nodes = argc > 2 ? std::stoull(argv[2]) : 12U;
        const std::size_t min_nodes = argc > 3 ? std::stoull(argv[3]) : 4U;
        const double tolerance = argc > 4 ? std::stod(argv[4]) : 1.0e-8;
        const std::size_t max_iterations = argc > 5 ? std::stoull(argv[5]) : 500U;
        if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes ||
            !(tolerance > 0.0) || !(tolerance < 1.0) || max_iterations == 0U) {
            throw std::invalid_argument("invalid topology PCG options");
        }

        std::cout << "GFSS M5 active-HEX topology PCG robustness probe\n"
                  << "purpose=test_whether_residual_blowup_is_actual_preconditioner_degradation\n"
                  << "preconditioners=smoothed_two_grid,tentative_two_grid,jacobi\n"
                  << "no_parameter_tuning=true\n"
                  << "target_nodes=" << target_nodes
                  << " min_nodes=" << min_nodes
                  << " tolerance=" << std::scientific << tolerance
                  << " max_iterations=" << std::fixed << max_iterations
                  << " selector=" << selector << '\n';

        std::size_t selected = 0U;
        for (const auto& test : cases()) {
            if (selector != "all" && selector != test.name) continue;
            ++selected;
            std::cout << "\n========================================\n";
            run_pcg_case(test, target_nodes, min_nodes, tolerance, max_iterations);
        }
        if (selected == 0U) throw std::invalid_argument("unknown topology PCG case");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_m5_active_hex_topology_pcg_bench "
                  << "[all|box_control|l_solid|through_hole|notched_beam "
                  << "[target_nodes=12 [min_nodes=4 [tol=1e-8 [max_iterations=500]]]]]\n";
        return 1;
    }
}
