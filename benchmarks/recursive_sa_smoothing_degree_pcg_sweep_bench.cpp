// M5 symmetric Chebyshev smoothing-degree sweep for the frozen 1x1 recursive
// SA multigrid preconditioner inside strict FP64 PCG.
//
// Reuse the validated selective-PCG implementation unit without its entry point.
#define main gfss_recursive_sa_selective_pcg_impl_main
#include "recursive_sa_selective_pcg_bench.cpp"
#undef main

namespace {

struct DegreeSweepPoint {
    std::size_t degree{0U};
    MgPreconditionerAudit audit;
    StrictPcgResult pcg;
    FineApplyCounter direct_fine_counter;
    bool theory_valid{false};
};

std::size_t fine_operator_equivalent_per_preconditioner(std::size_t degree) {
    // Frozen factorized hierarchy:
    //   A1 = P0^T A0 P0, m0=1  -> 3 A0-equivalent applies.
    //   A2 = P1^T A1 P1, m1=2  -> 5 A1 = 15 A0-equivalent applies.
    // One 1x1 cycle at a non-bottom level uses degree pre-smooth applies,
    // one residual apply, transfer restriction/prolongation work, and degree
    // post-smooth applies. L2->L3 uses m2=1 and an exact preassembled bottom.
    const std::size_t l0_local = 2U * degree + 3U;
    const std::size_t l1_local = (2U * degree + 5U) * 3U;
    const std::size_t l2_local = (2U * degree + 3U) * 15U;
    return l0_local + l1_local + l2_local;
}

void print_degree_history(const DegreeSweepPoint& p) {
    std::cout << "\nSMOOTHING_PCG_HISTORY degree=" << p.degree << '\n'
              << std::fixed << std::setprecision(6)
              << "solve_ms=" << p.pcg.solve_ms
              << " converged=" << (p.pcg.converged ? "true" : "false")
              << " iterations=" << p.pcg.iterations
              << " breakdown=" << (p.pcg.breakdown ? "true" : "false");
    if (p.pcg.breakdown) std::cout << " breakdown_reason=" << p.pcg.breakdown_reason;
    std::cout << '\n';
    for (std::size_t i = 0; i < p.pcg.true_relative_residuals.size(); ++i) {
        std::cout << std::scientific << std::setprecision(9)
                  << "true_residual[" << i << "]="
                  << p.pcg.true_relative_residuals[i];
        if (i > 0U) {
            std::cout << " iteration_q="
                      << p.pcg.true_relative_residuals[i] /
                         p.pcg.true_relative_residuals[i - 1U];
        }
        std::cout << '\n';
    }
}

void print_degree_profile(const DegreeSweepPoint& p) {
    const auto& prof = p.pcg.preconditioner_profile;
    const double denom = std::max(p.pcg.preconditioner_ms, 1.0e-300);
    const double accounted = prof.accounted_ms();
    const double calls = static_cast<double>(std::max<std::size_t>(
        1U, p.pcg.preconditioner_applications));
    std::cout << "\nSMOOTHING_PCG_PROFILE degree=" << p.degree << '\n'
              << std::fixed << std::setprecision(6)
              << "preconditioner_ms=" << p.pcg.preconditioner_ms
              << " accounted_ms=" << accounted
              << " unaccounted_ms=" << (p.pcg.preconditioner_ms - accounted) << '\n'
              << "pre_smooth_ms=" << prof.pre_smooth_ms
              << " pre_smooth_fraction=" << prof.pre_smooth_ms / denom
              << " residual_apply_ms=" << prof.residual_apply_ms
              << " residual_apply_fraction=" << prof.residual_apply_ms / denom << '\n'
              << "restrict_ms=" << prof.restrict_ms
              << " restrict_fraction=" << prof.restrict_ms / denom
              << " prolong_ms=" << prof.prolong_ms
              << " prolong_fraction=" << prof.prolong_ms / denom << '\n'
              << "post_smooth_ms=" << prof.post_smooth_ms
              << " post_smooth_fraction=" << prof.post_smooth_ms / denom
              << " bottom_solve_ms=" << prof.bottom_solve_ms
              << " bottom_solve_fraction=" << prof.bottom_solve_ms / denom << '\n'
              << "L0_visits_per_preconditioner=" << prof.level_visits[0] / calls
              << " L1_visits_per_preconditioner=" << prof.level_visits[1] / calls
              << " L2_visits_per_preconditioner=" << prof.level_visits[2] / calls
              << " L3_visits_per_preconditioner=" << prof.level_visits[3] / calls << '\n';
}

void run_smoothing_degree_pcg_sweep(std::size_t max_iterations,
                                    double tolerance,
                                    std::size_t target_nodes,
                                    std::size_t min_nodes) {
    constexpr std::size_t m0 = 1U;
    constexpr std::size_t m1 = 2U;
    constexpr std::size_t m2 = 1U;
    constexpr double strength_threshold = 0.05;
    constexpr double symmetry_accept = 1.0e-10;
    constexpr double linearity_accept = 1.0e-11;
    constexpr MgSelectiveSchedule schedule{1U, 1U, 1U};
    const std::array<std::size_t, 4> degrees{1U, 2U, 3U, 4U};

    if (max_iterations == 0U || !(tolerance > 0.0) || tolerance >= 1.0 ||
        target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
        throw std::invalid_argument("invalid smoothing-degree PCG sweep options");
    }

    const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
    const gfss::Material material{210.0e9, 0.30};
    const auto total_start = Clock::now();

    FineApplyCounter* active_fine_counter = nullptr;
    auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
    const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
        std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
    const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
        mesh, material, space0);
    const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
    const Apply apply0 = [&](const Vec& x) {
        const auto start = Clock::now();
        auto y = apply_fine_clamped(mesh, material, x);
        if (active_fine_counter != nullptr) {
            ++active_fine_counter->calls;
            active_fine_counter->ms += elapsed_ms(start, Clock::now());
        }
        return y;
    };
    const double lambda0 = estimate_lambda_max(apply0, fine_inverse, 8U);
    const double omega0 = kSaDampingNumerator / lambda0;
    const FineSmoothedTransfer transfer0{
        mesh, material, space0, fine_inverse, omega0, m0};
    const Apply apply1 = [&](const Vec& x) {
        return transfer0.restrict_transpose(apply0(transfer0.prolong(x)));
    };

    const auto graph1_tentative = graph_from_variable_blocks(tentative_a1);
    const auto candidates1 = make_level1_candidates(space0);
    const auto block1 = build_exact_l1_block_metric(
        mesh, material, space0, graph1_tentative, fine_inverse, omega0);
    const double block1_oracle_error = audit_l1_block_metric(block1, apply1);
    const double block_lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
    const double block_omega1 = kSaDampingNumerator / block_lambda1;

    double p0_support_cache_ms = 0.0;
    const auto fine_supports = build_fine_basis_support_cache(
        mesh, material, space0, fine_inverse, omega0, p0_support_cache_ms);
    const auto element_supports = build_element_support_index(mesh, fine_supports);
    const auto offdiag_start = Clock::now();
    const auto actual_a1_offdiagonal = accumulate_combined_actual_a1_offdiagonal_blocks(
        mesh, material, fine_supports, element_supports);
    const double actual_a1_offdiagonal_ms = elapsed_ms(offdiag_start, Clock::now());
    const auto strength1 = build_combined_strength_graph(
        graph1_tentative, block1, actual_a1_offdiagonal, strength_threshold);

    const auto transfer1_tentative = build_candidate_transfer(
        strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
    const L1BlockSmoothedTransfer transfer1{
        transfer1_tentative, apply1, block1, block_omega1, m1};
    const double transfer1_adjoint = l1_block_transfer_adjoint_error(transfer1);
    const Apply apply2 = [&](const Vec& x) {
        return transfer1.restrict_transpose(apply1(transfer1.prolong(x)));
    };

    const auto local_a1_apply = [&](const LocalColumns& x) {
        return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
    };
    const auto l2_start = Clock::now();
    const auto l2_basis = build_smoothed_candidate_supports(
        transfer1_tentative, strength1.graph, block1,
        block_omega1, m1, local_a1_apply);
    const auto block2 = build_metric_from_local_supports(
        transfer1_tentative, block1, l2_basis, local_a1_apply);
    const double l2_setup_ms = elapsed_ms(l2_start, Clock::now());
    const double block2_oracle_error = audit_l1_block_metric(block2, apply2);
    const double block_lambda2 = estimate_lambda_max_l1_block(apply2, block2, 8U);
    const double block_omega2 = kSaDampingNumerator / block_lambda2;

    const auto transfer2_tentative = build_candidate_transfer(
        transfer1_tentative.coarse_graph,
        transfer1_tentative.coarse_candidates,
        target_nodes,
        min_nodes,
        1.0e-10);
    const L1BlockSmoothedTransfer transfer2{
        transfer2_tentative, apply2, block2, block_omega2, m2};
    const double transfer2_adjoint = l1_block_transfer_adjoint_error(transfer2);

    const auto local_a2_apply_lambda = [&](const LocalColumns& x) {
        return apply_local_a2_columns(x, l2_basis, block1, local_a1_apply);
    };
    const std::function<LocalColumns(const LocalColumns&)> local_a2_apply =
        local_a2_apply_lambda;
    const auto bottom_basis = build_smoothed_candidate_supports(
        transfer2_tentative,
        transfer1_tentative.coarse_graph,
        block2,
        block_omega2,
        m2,
        local_a2_apply_lambda);
    const auto bottom = build_local_bottom(
        transfer2_tentative, block2, bottom_basis, local_a2_apply);
    const Apply apply3_nested = [&](const Vec& x) {
        return transfer2.restrict_transpose(apply2(transfer2.prolong(x)));
    };
    const double bottom_oracle_error = bottom_local_oracle_error(bottom, apply3_nested);

    std::vector<gfss::ReferenceMultilevelLevel> levels(4U);
    levels[0].dofs = static_cast<std::size_t>(mesh.dof_count());
    levels[0].label = "L0_fine_matrix_free";
    levels[0].diagnostic_lambda_max = lambda0;
    levels[0].apply = apply0;
    levels[0].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_smooth(apply0, fine_inverse, lambda0, b, x, degree);
        clamp_x0(mesh, x);
    };
    levels[0].restrict_to_coarse = [&](const Vec& r) {
        return transfer0.restrict_transpose(r);
    };
    levels[0].prolong_from_coarse = [&](const Vec& c) {
        return transfer0.prolong(c);
    };

    levels[1].dofs = space0.coarse_dofs;
    levels[1].label = "L1_actual_block_metric_actual_strength_graph";
    levels[1].diagnostic_lambda_max = block_lambda1;
    levels[1].apply = apply1;
    levels[1].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply1, block1, block_lambda1, b, x, degree);
    };
    levels[1].restrict_to_coarse = [&](const Vec& r) {
        return transfer1.restrict_transpose(r);
    };
    levels[1].prolong_from_coarse = [&](const Vec& c) {
        return transfer1.prolong(c);
    };

    levels[2].dofs = transfer1_tentative.coarse_dofs;
    levels[2].label = "L2_local_exact_block_metric";
    levels[2].diagnostic_lambda_max = block_lambda2;
    levels[2].apply = apply2;
    levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply2, block2, block_lambda2, b, x, degree);
    };
    levels[2].restrict_to_coarse = [&](const Vec& r) {
        return transfer2.restrict_transpose(r);
    };
    levels[2].prolong_from_coarse = [&](const Vec& c) {
        return transfer2.prolong(c);
    };

    levels[3].dofs = transfer2_tentative.coarse_dofs;
    levels[3].label = "L3_local_dense_bottom";
    levels[3].apply = [&](const Vec& x) { return apply_dense_bottom(bottom, x); };
    levels[3].bottom_solve = [&](const Vec& b) { return bottom.factor.solve(b); };

    const auto setup_stop = Clock::now();
    const auto rhs = make_rhs(mesh);
    const auto sanitize_probe = [&](Vec& v) { clamp_x0(mesh, v); };

    std::cout << "GFSS M5 1x1 MG-PCG symmetric smoothing-degree sweep\n"
              << "problem=thin_plate mesh=64x64x8\n"
              << "reference_execution=cpu_fp64\n"
              << "hierarchy=frozen_theta_0p05_actual_metrics\n"
              << "recursive_schedule=1x1\n"
              << "symmetric_smoothing_degrees=1,2,3,4\n"
              << "fixed_transfer_smoothing_steps=m0:1,m1:2,m2:1\n"
              << "preconditioner=B_1x1_from_zero_initial_guess\n"
              << "krylov=left_preconditioned_conjugate_gradient\n"
              << "pcg_recurrence_residual=standard\n"
              << "correctness_oracle=independent_FP64_true_residual_every_iteration\n"
              << "true_relative_tolerance=" << std::scientific << std::setprecision(9)
              << tolerance << " max_iterations=" << max_iterations << '\n'
              << "preconditioner_symmetry_accept=" << symmetry_accept
              << " preconditioner_linearity_accept=" << linearity_accept << '\n'
              << "production_method_claim=false\n\n"
              << "L0_dofs=" << levels[0].dofs
              << " L1_dofs=" << levels[1].dofs
              << " L2_dofs=" << levels[2].dofs
              << " L3_dofs=" << levels[3].dofs << '\n'
              << std::fixed << std::setprecision(6)
              << "strength_directed_edges=" << strength1.stats.directed_edges
              << " strength_degree_avg=" << strength1.stats.average_degree
              << " strength_forced_pairs=" << strength1.stats.forced_pairs << '\n'
              << "lambda0=" << lambda0 << " omega0=" << omega0
              << " block_lambda1=" << block_lambda1
              << " block_omega1=" << block_omega1
              << " block_lambda2=" << block_lambda2
              << " block_omega2=" << block_omega2 << '\n'
              << "P0_smoothed_support_cache_ms=" << p0_support_cache_ms
              << " actual_A1_offdiagonal_setup_ms=" << actual_a1_offdiagonal_ms
              << " L2_local_support_block_setup_ms=" << l2_setup_ms
              << " bottom_local_assembly_ms=" << bottom.assembly_ms
              << " hierarchy_setup_ms=" << elapsed_ms(total_start, setup_stop) << '\n'
              << std::scientific << std::setprecision(9)
              << "L1_local_block_vs_nested_oracle_relative_error=" << block1_oracle_error
              << " L2_local_block_vs_nested_oracle_relative_error=" << block2_oracle_error
              << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error << '\n'
              << "L1_transfer_adjoint_relative_error=" << transfer1_adjoint
              << " L2_transfer_adjoint_relative_error=" << transfer2_adjoint << '\n';

    std::vector<DegreeSweepPoint> points;
    points.reserve(degrees.size());
    for (const auto degree : degrees) {
        DegreeSweepPoint p;
        p.degree = degree;
        p.audit = mg_audit_selective_preconditioner(
            levels, degree, degree, schedule, sanitize_probe);
        p.theory_valid = p.audit.positive &&
            p.audit.max_symmetry_relative_defect <= symmetry_accept &&
            p.audit.max_linearity_relative_error <= linearity_accept;

        std::cout << "\nSMOOTHING_PCG_PRECONDITIONER_AUDIT degree=" << degree << '\n'
                  << std::scientific << std::setprecision(9)
                  << "max_symmetry_relative_defect="
                  << p.audit.max_symmetry_relative_defect
                  << " max_linearity_relative_error="
                  << p.audit.max_linearity_relative_error
                  << " min_normalized_quadratic="
                  << p.audit.min_normalized_quadratic
                  << " positive=" << (p.audit.positive ? "true" : "false")
                  << " pcg_theory_valid=" << (p.theory_valid ? "true" : "false")
                  << '\n';

        active_fine_counter = &p.direct_fine_counter;
        p.pcg = solve_strict_reference_pcg(
            levels, rhs, tolerance, max_iterations,
            degree, degree, schedule);
        active_fine_counter = nullptr;

        print_degree_history(p);
        print_degree_profile(p);
        points.push_back(std::move(p));
    }

    std::cout << "\nSMOOTHING_PCG_SUMMARY\n";
    const DegreeSweepPoint* fastest_verified = nullptr;
    const DegreeSweepPoint* fewest_iterations = nullptr;
    const DegreeSweepPoint* degree3_anchor = nullptr;
    for (const auto& p : points) {
        const double core_ms = std::max(0.0, p.pcg.solve_ms - p.pcg.verification_ms);
        const double precond_per_apply = p.pcg.preconditioner_applications > 0U
            ? p.pcg.preconditioner_ms /
              static_cast<double>(p.pcg.preconditioner_applications)
            : std::numeric_limits<double>::quiet_NaN();
        const std::size_t fine_equiv_per_precond =
            fine_operator_equivalent_per_preconditioner(p.degree);
        const std::size_t total_fine_equiv =
            fine_equiv_per_precond * p.pcg.preconditioner_applications +
            p.pcg.pcg_operator_applications + p.pcg.verification_operator_applications;

        if (p.degree == 3U) degree3_anchor = &p;
        if (p.pcg.converged && !p.pcg.breakdown && p.theory_valid) {
            if (fastest_verified == nullptr || p.pcg.solve_ms < fastest_verified->pcg.solve_ms) {
                fastest_verified = &p;
            }
            if (fewest_iterations == nullptr || p.pcg.iterations < fewest_iterations->pcg.iterations) {
                fewest_iterations = &p;
            }
        }

        std::cout << "degree=" << p.degree
                  << " pre_smooth=" << p.degree
                  << " post_smooth=" << p.degree
                  << " pcg_theory_valid=" << (p.theory_valid ? "true" : "false")
                  << " converged=" << (p.pcg.converged ? "true" : "false")
                  << " breakdown=" << (p.pcg.breakdown ? "true" : "false")
                  << " iterations=" << p.pcg.iterations
                  << std::scientific << std::setprecision(9)
                  << " final_true_relative_residual="
                  << (p.pcg.true_relative_residuals.empty()
                        ? std::numeric_limits<double>::quiet_NaN()
                        : p.pcg.true_relative_residuals.back())
                  << std::fixed << std::setprecision(6)
                  << " solve_ms=" << p.pcg.solve_ms
                  << " core_ms_excluding_verification=" << core_ms
                  << " verification_ms=" << p.pcg.verification_ms
                  << " preconditioner_ms=" << p.pcg.preconditioner_ms
                  << " preconditioner_ms_per_apply=" << precond_per_apply
                  << " preconditioner_applications=" << p.pcg.preconditioner_applications
                  << " pcg_operator_applications=" << p.pcg.pcg_operator_applications
                  << " verification_operator_applications="
                  << p.pcg.verification_operator_applications
                  << " fine_A0_equivalent_per_preconditioner=" << fine_equiv_per_precond
                  << " fine_A0_equivalent_total=" << total_fine_equiv
                  << " direct_wrapper_fine_A0_calls_total=" << p.direct_fine_counter.calls
                  << " direct_wrapper_fine_A0_ms_total=" << p.direct_fine_counter.ms
                  << std::scientific << std::setprecision(9)
                  << " max_recurrence_vs_true_residual_drift="
                  << p.pcg.max_recurrence_vs_true_residual_drift
                  << " preconditioner_symmetry_defect="
                  << p.audit.max_symmetry_relative_defect
                  << " preconditioner_linearity_error="
                  << p.audit.max_linearity_relative_error
                  << '\n';
    }

    std::cout << "\nSMOOTHING_PCG_VERDICT\n"
              << "fastest_verified_degree="
              << (fastest_verified != nullptr ? std::to_string(fastest_verified->degree) : "none")
              << '\n'
              << "fewest_iterations_verified_degree="
              << (fewest_iterations != nullptr ? std::to_string(fewest_iterations->degree) : "none")
              << '\n';
    if (fastest_verified != nullptr) {
        std::cout << std::fixed << std::setprecision(6)
                  << "fastest_verified_solve_ms=" << fastest_verified->pcg.solve_ms
                  << " fastest_verified_iterations=" << fastest_verified->pcg.iterations
                  << '\n';
    }
    if (degree3_anchor != nullptr) {
        std::cout << std::fixed << std::setprecision(6)
                  << "degree3_anchor_solve_ms=" << degree3_anchor->pcg.solve_ms
                  << " degree3_anchor_iterations=" << degree3_anchor->pcg.iterations
                  << " degree3_anchor_expected_previous_iterations=11\n";
    }
    std::cout << "total_reference_ms=" << std::fixed << std::setprecision(6)
              << elapsed_ms(total_start, Clock::now()) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t max_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 80U;
        const double tolerance = argc > 2 ? std::stod(argv[2]) : 1.0e-6;
        const std::size_t target_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t min_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 4U;
        run_smoothing_degree_pcg_sweep(
            max_iterations, tolerance, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_smoothing_degree_pcg_sweep_bench "
                  << "[max_iterations=80 [true_tolerance=1e-6 "
                  << "[target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
