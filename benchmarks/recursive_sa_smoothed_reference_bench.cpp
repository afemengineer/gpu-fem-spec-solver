// Extend the existing recursive-SA numerical reference without duplicating its
// setup/algebra helpers. Rename the legacy main while including it into this
// standalone translation unit; all anonymous-namespace helpers remain internal
// to this TU and are reused below.
#define main gfss_recursive_sa_raw_reference_main
#include "recursive_sa_reference_bench.cpp"
#undef main

namespace {

struct AlgebraicSmoothedTransfer {
    const CandidateTransfer& tentative;
    const Apply& apply_fine;
    const std::vector<double>& inverse_diagonal;
    double omega{0.0};
    std::size_t steps{0};

    Vec prolong(const Vec& coarse) const {
        auto fine = tentative.prolong(coarse);
        if (inverse_diagonal.size() != fine.size()) {
            throw std::invalid_argument("recursive smoothed SA prolongation diagonal size mismatch");
        }
        for (std::size_t step = 0; step < steps; ++step) {
            const auto af = apply_fine(fine);
            if (af.size() != fine.size()) {
                throw std::runtime_error("recursive smoothed SA forward operator size mismatch");
            }
            for (std::size_t i = 0; i < fine.size(); ++i) {
                fine[i] -= omega * inverse_diagonal[i] * af[i];
            }
        }
        return fine;
    }

    Vec restrict_transpose(const Vec& fine) const {
        if (fine.size() != tentative.fine_dofs ||
            inverse_diagonal.size() != fine.size()) {
            throw std::invalid_argument("recursive smoothed SA restriction size mismatch");
        }
        auto work = fine;
        Vec scaled(fine.size(), 0.0);
        for (std::size_t step = 0; step < steps; ++step) {
            for (std::size_t i = 0; i < work.size(); ++i) {
                scaled[i] = inverse_diagonal[i] * work[i];
            }
            const auto a_scaled = apply_fine(scaled);
            if (a_scaled.size() != work.size()) {
                throw std::runtime_error("recursive smoothed SA transpose operator size mismatch");
            }
            for (std::size_t i = 0; i < work.size(); ++i) {
                work[i] -= omega * a_scaled[i];
            }
        }
        return tentative.restrict_transpose(work);
    }
};

Vec deterministic_vector(std::size_t n, double phase) {
    Vec v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = std::sin(0.013 * t + phase) + 0.31 * std::cos(0.037 * t - 0.5 * phase);
    }
    return v;
}

double smoothed_transfer_adjoint_error(const AlgebraicSmoothedTransfer& transfer) {
    const auto coarse = deterministic_vector(transfer.tentative.coarse_dofs, 0.19);
    const auto fine = deterministic_vector(transfer.tentative.fine_dofs, 0.71);
    const auto pc = transfer.prolong(coarse);
    const auto ptf = transfer.restrict_transpose(fine);
    const double lhs = dot(pc, fine);
    const double rhs = dot(coarse, ptf);
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) / scale;
}

double smoothed_candidate_reproduction_error(
    const AlgebraicSmoothedTransfer& transfer,
    const std::vector<double>& fine_candidates) {
    if (fine_candidates.size() != transfer.tentative.fine_dofs * kCandidates) {
        throw std::invalid_argument("recursive smoothed SA candidate diagnostic size mismatch");
    }
    double worst = 0.0;
    for (std::size_t c = 0; c < kCandidates; ++c) {
        Vec fine(transfer.tentative.fine_dofs, 0.0);
        Vec coarse(transfer.tentative.coarse_dofs, 0.0);
        for (std::size_t i = 0; i < fine.size(); ++i) {
            fine[i] = fine_candidates[i * kCandidates + c];
        }
        for (std::size_t i = 0; i < coarse.size(); ++i) {
            coarse[i] = transfer.tentative.coarse_candidates[i * kCandidates + c];
        }
        const auto reproduced = transfer.prolong(coarse);
        Vec diff(fine.size(), 0.0);
        for (std::size_t i = 0; i < fine.size(); ++i) diff[i] = reproduced[i] - fine[i];
        const double fn = norm(fine);
        if (fn > 0.0) worst = std::max(worst, norm(diff) / fn);
    }
    return worst;
}

void run_smoothed_reference(std::size_t fine_transfer_steps,
                            std::size_t level1_transfer_steps,
                            std::size_t level2_transfer_steps,
                            std::size_t max_cycles,
                            std::size_t target_nodes,
                            std::size_t min_nodes) {
    const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
    const gfss::Material material{210.0e9, 0.30};
    const auto setup_start = Clock::now();

    auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
    const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
        std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
    const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
        mesh, material, space0);
    const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
    const Apply apply0 = [&](const Vec& x) { return apply_fine_clamped(mesh, material, x); };
    const double lambda0 = estimate_lambda_max(apply0, fine_inverse, 8U);
    const double omega0 = kSaDampingNumerator / lambda0;
    const FineSmoothedTransfer transfer0{
        mesh, material, space0, fine_inverse, omega0, fine_transfer_steps};

    const Apply apply1 = [&](const Vec& x) {
        const auto fine = transfer0.prolong(x);
        return transfer0.restrict_transpose(apply0(fine));
    };

    const auto graph1 = graph_from_variable_blocks(tentative_a1);
    const auto candidates1 = make_level1_candidates(space0);
    const auto tentative_transfer1 = build_candidate_transfer(
        graph1, candidates1, target_nodes, min_nodes, 1.0e-10);
    const double tentative_candidate1_error =
        candidate_reproduction_error(tentative_transfer1, candidates1);

    std::vector<double> inverse1 = tentative_a1.inverse_diagonal;
    if (inverse1.size() != graph1.dofs()) {
        throw std::runtime_error("recursive smoothed SA L1 diagonal size mismatch");
    }
    const double lambda1 = estimate_lambda_max(apply1, inverse1, 8U);
    const double omega1 = kSaDampingNumerator / lambda1;
    const AlgebraicSmoothedTransfer transfer1{
        tentative_transfer1, apply1, inverse1, omega1, level1_transfer_steps};
    const double transfer1_adjoint = smoothed_transfer_adjoint_error(transfer1);
    const double smoothed_candidate1_error =
        smoothed_candidate_reproduction_error(transfer1, candidates1);

    const Apply apply2 = [&](const Vec& x) {
        const auto l1 = transfer1.prolong(x);
        return transfer1.restrict_transpose(apply1(l1));
    };

    // Deliberately retain the same tentative/propagated Jacobi surrogate as the
    // raw-recursion experiment. This isolates transfer smoothing as the only
    // numerical variable in the comparison.
    const auto inverse2 = tentative_transfer1.approximate_inverse_coarse_diagonal(inverse1);
    const double lambda2 = estimate_lambda_max(apply2, inverse2, 8U);
    const double omega2 = kSaDampingNumerator / lambda2;

    const auto tentative_transfer2 = build_candidate_transfer(
        tentative_transfer1.coarse_graph,
        tentative_transfer1.coarse_candidates,
        target_nodes,
        min_nodes,
        1.0e-10);
    const double tentative_candidate2_error = candidate_reproduction_error(
        tentative_transfer2, tentative_transfer1.coarse_candidates);
    const AlgebraicSmoothedTransfer transfer2{
        tentative_transfer2, apply2, inverse2, omega2, level2_transfer_steps};
    const double transfer2_adjoint = smoothed_transfer_adjoint_error(transfer2);
    const double smoothed_candidate2_error = smoothed_candidate_reproduction_error(
        transfer2, tentative_transfer1.coarse_candidates);

    const Apply apply3 = [&](const Vec& x) {
        const auto l2 = transfer2.prolong(x);
        return transfer2.restrict_transpose(apply2(l2));
    };

    const auto bottom_start = Clock::now();
    const auto bottom = materialize_and_factor_bottom(apply3, tentative_transfer2.coarse_dofs);
    const auto bottom_stop = Clock::now();
    const auto setup_stop = Clock::now();

    std::vector<gfss::ReferenceMultilevelLevel> levels(4U);
    levels[0].dofs = static_cast<std::size_t>(mesh.dof_count());
    levels[0].label = "L0_fine_matrix_free";
    levels[0].diagnostic_lambda_max = lambda0;
    levels[0].apply = apply0;
    levels[0].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_smooth(apply0, fine_inverse, lambda0, b, x, degree);
        clamp_x0(mesh, x);
    };
    levels[0].restrict_to_coarse = [&](const Vec& r) { return transfer0.restrict_transpose(r); };
    levels[0].prolong_from_coarse = [&](const Vec& c) { return transfer0.prolong(c); };

    levels[1].dofs = space0.coarse_dofs;
    levels[1].label = "L1_smoothed_Galerkin_matrix_free";
    levels[1].diagnostic_lambda_max = lambda1;
    levels[1].apply = apply1;
    levels[1].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_smooth(apply1, inverse1, lambda1, b, x, degree);
    };
    levels[1].restrict_to_coarse = [&](const Vec& r) { return transfer1.restrict_transpose(r); };
    levels[1].prolong_from_coarse = [&](const Vec& c) { return transfer1.prolong(c); };

    levels[2].dofs = tentative_transfer1.coarse_dofs;
    levels[2].label = "L2_recursively_smoothed_candidate_Galerkin";
    levels[2].diagnostic_lambda_max = lambda2;
    levels[2].apply = apply2;
    levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_smooth(apply2, inverse2, lambda2, b, x, degree);
    };
    levels[2].restrict_to_coarse = [&](const Vec& r) { return transfer2.restrict_transpose(r); };
    levels[2].prolong_from_coarse = [&](const Vec& c) { return transfer2.prolong(c); };

    levels[3].dofs = tentative_transfer2.coarse_dofs;
    levels[3].label = "L3_dense_Cholesky_bottom";
    levels[3].apply = apply3;
    levels[3].bottom_solve = [&](const Vec& b) { return bottom.solve(b); };

    const auto rhs = make_rhs(mesh);
    const auto result = gfss::solve_reference_multilevel_vcycle(
        levels, rhs, 1.0e-6, max_cycles, 3U, 3U);

    std::cout << "\n========================================\n"
              << "L0_transfer_smoothing_steps=" << fine_transfer_steps
              << " L1_transfer_smoothing_steps=" << level1_transfer_steps
              << " L2_transfer_smoothing_steps=" << level2_transfer_steps << '\n'
              << "hierarchy_levels=4\n"
              << "higher_level_transfer=smoothed_candidate_QR_exact_transpose\n"
              << "recursive_jacobi=tentative_or_propagated_reference_surrogate\n"
              << "coarse_mesh_required_after_L0=false\n"
              << "L0_dofs=" << levels[0].dofs
              << " L1_dofs=" << levels[1].dofs
              << " L1_nodes=" << graph1.nodes()
              << " L2_dofs=" << levels[2].dofs
              << " L2_nodes=" << tentative_transfer1.coarse_graph.nodes()
              << " L3_dofs=" << levels[3].dofs
              << " L3_nodes=" << tentative_transfer2.coarse_graph.nodes() << '\n'
              << std::scientific << std::setprecision(9)
              << "L1_tentative_candidate_reproduction_error=" << tentative_candidate1_error
              << " L1_smoothed_candidate_reproduction_error=" << smoothed_candidate1_error
              << " L1_transfer_adjoint_relative_error=" << transfer1_adjoint << '\n'
              << "L2_tentative_candidate_reproduction_error=" << tentative_candidate2_error
              << " L2_smoothed_candidate_reproduction_error=" << smoothed_candidate2_error
              << " L2_transfer_adjoint_relative_error=" << transfer2_adjoint << '\n'
              << std::fixed << std::setprecision(6)
              << "lambda0=" << lambda0 << " omega0=" << omega0
              << " lambda1=" << lambda1 << " omega1=" << omega1
              << " lambda2=" << lambda2 << " omega2=" << omega2 << '\n'
              << std::scientific << std::setprecision(9)
              << "bottom_symmetry_relative_defect=" << bottom.symmetry_relative_defect
              << " bottom_min_cholesky_pivot=" << bottom.min_pivot << '\n'
              << std::fixed << std::setprecision(6)
              << "bottom_materialize_factor_ms=" << elapsed_ms(bottom_start, bottom_stop)
              << " hierarchy_setup_ms=" << elapsed_ms(setup_start, setup_stop)
              << " solve_ms=" << result.solve_ms << '\n'
              << "converged=" << (result.converged ? "true" : "false")
              << " cycles=" << result.cycles << '\n';

    for (std::size_t i = 0; i < result.relative_residuals.size(); ++i) {
        std::cout << std::scientific << std::setprecision(9)
                  << "true_residual[" << i << "]=" << result.relative_residuals[i];
        if (i > 0U) {
            std::cout << " cycle_q="
                      << result.relative_residuals[i] / result.relative_residuals[i - 1U];
        }
        std::cout << '\n';
    }
    if (result.relative_residuals.size() > 2U) {
        double log_sum = 0.0;
        std::size_t count = 0U;
        for (std::size_t i = 2U; i < result.relative_residuals.size(); ++i) {
            const double q = result.relative_residuals[i] / result.relative_residuals[i - 1U];
            if (q > 0.0 && std::isfinite(q)) {
                log_sum += std::log(q);
                ++count;
            }
        }
        if (count > 0U) {
            std::cout << "post_transient_geomean_q=" << std::scientific << std::setprecision(9)
                      << std::exp(log_sum / static_cast<double>(count)) << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t m0 = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 1U;
        const std::size_t m1 = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 1U;
        const std::size_t m2 = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 1U;
        const std::size_t max_cycles = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 6U;
        const std::size_t target_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 12U;
        const std::size_t min_nodes = argc > 6
            ? static_cast<std::size_t>(std::stoull(argv[6])) : 4U;
        if (m0 > 4U || m1 > 4U || m2 > 4U || max_cycles == 0U ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid recursively smoothed SA reference options");
        }

        std::cout << "GFSS M5 recursively smoothed-aggregation numerical reference\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "purpose=isolate_recursive_transfer_smoothing_effect\n"
                  << "reference_execution=cpu_fp64\n"
                  << "L0_operator=matrix_free_exact_FEM\n"
                  << "all_transfer_restrictions=exact_transpose\n"
                  << "L1_L2_graph=algebraic_block_adjacency\n"
                  << "L3_bottom=dense_materialized_Cholesky\n"
                  << "pre_smooth_degree=3 post_smooth_degree=3\n"
                  << "acceptance_target_post_transient_q<=0.4\n"
                  << "performance_status=numerical_reference_only\n";

        run_smoothed_reference(m0, m1, m2, max_cycles, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_smoothed_reference_bench "
                  << "[m0 [m1 [m2 [max_cycles [target_nodes [min_nodes]]]]]]\n";
        return 1;
    }
}
