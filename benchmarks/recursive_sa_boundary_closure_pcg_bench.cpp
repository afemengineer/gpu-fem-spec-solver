// M5 boundary-closure sweep for the frozen 1x1 recursive SA MG-PCG reference.
// Incumbent is 5x1x1. Test L0 degrees 6..10 with L1=L2=1.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "recursive_sa_cr_oracle_helpers.inc"
#include "recursive_sa_block_cr_spectral_helpers.inc"
#include "recursive_sa_full_vcycle_spectral_helpers.inc"
#include "recursive_sa_gamma_cycle_helpers.inc"
#include "recursive_sa_selective_gamma_helpers.inc"
#include "recursive_sa_selective_pcg_helpers.inc"
#include "recursive_sa_level_smoothing_pcg_helpers.inc"

namespace {

struct BoundaryPolicy {
    std::string label;
    MgLevelSmoothingSchedule degrees{};
};

struct BoundaryPoint {
    BoundaryPolicy policy;
    MgPreconditionerAudit audit;
    MgLevelSmoothingPcgResult pcg;
    std::size_t fine_equivalent_per_preconditioner{0U};
    std::size_t fine_equivalent_total{0U};
    bool theory_valid{false};
};

void run_boundary_closure(std::size_t max_iterations,
                          double tolerance,
                          std::size_t target_nodes,
                          std::size_t min_nodes) {
    constexpr std::size_t m0 = 1U;
    constexpr std::size_t m1 = 2U;
    constexpr std::size_t m2 = 1U;
    constexpr double strength_threshold = 0.05;
    constexpr double symmetry_accept = 1.0e-10;
    constexpr double linearity_accept = 1.0e-11;
    constexpr std::size_t incumbent_work = 1221U;
    constexpr std::size_t incumbent_iterations = 11U;

    if (max_iterations == 0U || !(tolerance > 0.0) || tolerance >= 1.0 ||
        target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
        throw std::invalid_argument("invalid boundary-closure options");
    }

    std::vector<BoundaryPolicy> policies;
    policies.reserve(6U);
    for (std::size_t l0 = 5U; l0 <= 10U; ++l0) {
        BoundaryPolicy p;
        p.label = std::to_string(l0) + "x1x1";
        p.degrees = {l0, 1U, 1U};
        policies.push_back(std::move(p));
    }

    const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
    const gfss::Material material{210.0e9, 0.30};
    const auto total_start = Clock::now();

    auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
    const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
        std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
    const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
        mesh, material, space0);
    const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
    const Apply apply0 = [&](const Vec& x) {
        return apply_fine_clamped(mesh, material, x);
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

    std::cout << "GFSS M5 MG-PCG L0 smoothing boundary-closure sweep\n"
              << "problem=thin_plate mesh=64x64x8\n"
              << "reference_execution=cpu_fp64\n"
              << "hierarchy=frozen_theta_0p05_actual_metrics\n"
              << "recursive_schedule=1x1\n"
              << "policies=5x1x1,6x1x1,7x1x1,8x1x1,9x1x1,10x1x1\n"
              << "closure_rule=higher_degree_must_beat_incumbent_A0_equivalent_total\n"
              << "incumbent_policy=5x1x1 incumbent_iterations=" << incumbent_iterations
              << " incumbent_A0_equivalent_total=" << incumbent_work << '\n'
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

    std::vector<BoundaryPoint> points;
    points.reserve(policies.size());
    for (const auto& policy : policies) {
        BoundaryPoint p;
        p.policy = policy;
        p.audit = mg_audit_level_smoothing_preconditioner(
            levels, policy.degrees, sanitize_probe);
        p.theory_valid = p.audit.positive &&
            p.audit.max_symmetry_relative_defect <= symmetry_accept &&
            p.audit.max_linearity_relative_error <= linearity_accept;
        p.pcg = mg_solve_level_smoothing_pcg(
            levels, rhs, tolerance, max_iterations, policy.degrees);
        p.fine_equivalent_per_preconditioner =
            mg_level_smoothing_fine_operator_equivalent_per_preconditioner(
                policy.degrees);
        p.fine_equivalent_total =
            p.pcg.preconditioner_applications * p.fine_equivalent_per_preconditioner +
            p.pcg.pcg_operator_applications +
            p.pcg.verification_operator_applications;

        const double preconditioner_per_apply = p.pcg.preconditioner_applications > 0U
            ? p.pcg.preconditioner_ms /
              static_cast<double>(p.pcg.preconditioner_applications)
            : std::numeric_limits<double>::quiet_NaN();

        std::cout << "\nBOUNDARY_PCG_AUDIT policy=" << policy.label
                  << " symmetry_defect=" << std::scientific << std::setprecision(9)
                  << p.audit.max_symmetry_relative_defect
                  << " linearity_error=" << p.audit.max_linearity_relative_error
                  << " min_normalized_quadratic=" << p.audit.min_normalized_quadratic
                  << " positive=" << (p.audit.positive ? "true" : "false")
                  << " pcg_theory_valid=" << (p.theory_valid ? "true" : "false") << '\n';

        std::cout << "BOUNDARY_PCG_RESULT policy=" << policy.label
                  << " iterations=" << p.pcg.iterations
                  << " converged=" << (p.pcg.converged ? "true" : "false")
                  << " breakdown=" << (p.pcg.breakdown ? "true" : "false")
                  << " final_true_relative_residual=" << std::scientific
                  << std::setprecision(9)
                  << (p.pcg.true_relative_residuals.empty()
                        ? std::numeric_limits<double>::quiet_NaN()
                        : p.pcg.true_relative_residuals.back())
                  << std::fixed << std::setprecision(6)
                  << " solve_ms=" << p.pcg.solve_ms
                  << " preconditioner_ms=" << p.pcg.preconditioner_ms
                  << " preconditioner_ms_per_apply=" << preconditioner_per_apply
                  << " preconditioner_applications=" << p.pcg.preconditioner_applications
                  << " fine_A0_equivalent_per_preconditioner="
                  << p.fine_equivalent_per_preconditioner
                  << " fine_A0_equivalent_total=" << p.fine_equivalent_total
                  << std::scientific << std::setprecision(9)
                  << " max_recurrence_vs_true_residual_drift="
                  << p.pcg.max_recurrence_vs_true_residual_drift << '\n';
        points.push_back(std::move(p));
    }

    const BoundaryPoint* fastest = nullptr;
    const BoundaryPoint* minimum_work = nullptr;
    const BoundaryPoint* incumbent = nullptr;
    bool higher_degree_beats_incumbent_work = false;
    bool higher_degree_reaches_ten = false;

    for (const auto& p : points) {
        if (p.policy.label == "5x1x1") incumbent = &p;
        if (!p.theory_valid || !p.pcg.converged || p.pcg.breakdown) continue;
        if (fastest == nullptr || p.pcg.solve_ms < fastest->pcg.solve_ms) fastest = &p;
        if (minimum_work == nullptr ||
            p.fine_equivalent_total < minimum_work->fine_equivalent_total) {
            minimum_work = &p;
        }
        if (p.policy.degrees[0] > 5U) {
            higher_degree_beats_incumbent_work =
                higher_degree_beats_incumbent_work ||
                p.fine_equivalent_total < incumbent_work;
            higher_degree_reaches_ten =
                higher_degree_reaches_ten || p.pcg.iterations <= 10U;
        }
    }

    const bool incumbent_reproduced = incumbent != nullptr && incumbent->theory_valid &&
        incumbent->pcg.converged && !incumbent->pcg.breakdown &&
        incumbent->pcg.iterations == incumbent_iterations &&
        incumbent->fine_equivalent_total == incumbent_work;
    const bool search_closed = incumbent_reproduced &&
        !higher_degree_beats_incumbent_work;

    std::cout << "\nBOUNDARY_PCG_VERDICT\n"
              << "incumbent_reproduced=" << (incumbent_reproduced ? "true" : "false") << '\n'
              << "higher_degree_reaches_10_or_fewer_iterations="
              << (higher_degree_reaches_ten ? "true" : "false") << '\n'
              << "higher_degree_beats_incumbent_equivalent_work="
              << (higher_degree_beats_incumbent_work ? "true" : "false") << '\n'
              << "smoother_search_closed=" << (search_closed ? "true" : "false") << '\n'
              << "fastest_verified_policy="
              << (fastest != nullptr ? fastest->policy.label : "none") << '\n'
              << "minimum_equivalent_work_policy="
              << (minimum_work != nullptr ? minimum_work->policy.label : "none") << '\n';
    if (fastest != nullptr) {
        std::cout << std::fixed << std::setprecision(6)
                  << "fastest_verified_solve_ms=" << fastest->pcg.solve_ms
                  << " fastest_verified_iterations=" << fastest->pcg.iterations
                  << " fastest_verified_A0_equivalent_total="
                  << fastest->fine_equivalent_total << '\n';
    }
    if (minimum_work != nullptr) {
        std::cout << "minimum_equivalent_work_total="
                  << minimum_work->fine_equivalent_total
                  << " minimum_equivalent_work_iterations="
                  << minimum_work->pcg.iterations << '\n';
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
        run_boundary_closure(max_iterations, tolerance, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_boundary_closure_pcg_bench "
                  << "[max_iterations=80 [true_tolerance=1e-6 "
                  << "[target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
