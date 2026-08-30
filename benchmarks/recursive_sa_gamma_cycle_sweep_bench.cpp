// M5 recursive gamma-cycle sweep on the validated thin-plate SA hierarchy.
// CPU/FP64 numerical reference only. The hierarchy is frozen; only gamma changes.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "recursive_sa_cr_oracle_helpers.inc"
#include "recursive_sa_block_cr_spectral_helpers.inc"
#include "recursive_sa_full_vcycle_spectral_helpers.inc"
#include "recursive_sa_gamma_cycle_helpers.inc"

namespace {

struct GammaSweepPoint {
    std::size_t gamma{0U};
    FullMgSpectrum spectrum;
    MgPhysicalSummary dominant_physical;
    MgGammaTraceResult trace;
    MgGammaSolveResult baseline;
    double spectral_ms{0.0};
    double tail5_q{std::numeric_limits<double>::quiet_NaN()};
    double tail10_q{std::numeric_limits<double>::quiet_NaN()};
    double final_residual_alignment{0.0};
    double spectral_tail_gap{std::numeric_limits<double>::quiet_NaN()};
};

void print_gamma_spectrum(const GammaSweepPoint& p) {
    std::cout << "\nGAMMA_SPECTRUM gamma=" << p.gamma << '\n';
    for (std::size_t i = 0; i < p.spectrum.modes.size(); ++i) {
        std::cout << "mode=" << i
                  << " projected_eigenvalue=" << std::scientific << std::setprecision(9)
                  << p.spectrum.eigenvalue[i]
                  << " abs_eigenvalue=" << std::abs(p.spectrum.eigenvalue[i])
                  << " one_cycle_energy_q=" << p.spectrum.one_cycle_energy_q[i]
                  << " one_cycle_residual_q=" << p.spectrum.one_cycle_residual_q[i]
                  << '\n';
    }
}

void print_gamma_subspace_history(const GammaSweepPoint& p) {
    std::cout << "\nGAMMA_SUBSPACE_HISTORY gamma=" << p.gamma << '\n';
    for (std::size_t i = 0; i < p.spectrum.subspace_leakage.size(); ++i) {
        std::cout << "block_iteration=" << (i + 1U)
                  << " subspace_leakage=" << std::scientific << std::setprecision(9)
                  << p.spectrum.subspace_leakage[i] << '\n';
    }
}

void print_gamma_trace(const GammaSweepPoint& p,
                       const std::vector<gfss::ReferenceMultilevelLevel>& levels) {
    const auto& t = p.trace.top;
    const double e0 = std::max(t.initial_energy, 1.0e-300);
    const double r0 = std::max(t.initial_residual_norm, 1.0e-300);
    std::cout << "\nGAMMA_DOMINANT_TOP_TRACE gamma=" << p.gamma << '\n'
              << std::scientific << std::setprecision(9)
              << "initial_energy=" << t.initial_energy
              << " after_pre_energy_ratio=" << t.after_pre_energy / e0
              << " after_coarse_energy_ratio=" << t.after_coarse_energy / e0
              << " after_post_energy_ratio=" << t.after_post_energy / e0 << '\n'
              << "initial_residual_norm=" << t.initial_residual_norm
              << " after_pre_residual_ratio=" << t.after_pre_residual_norm / r0
              << " after_coarse_residual_ratio=" << t.after_coarse_residual_norm / r0
              << " after_post_residual_ratio=" << t.after_post_residual_norm / r0 << '\n';

    std::cout << "GAMMA_RECURSIVE_VISIT_SUMMARY gamma=" << p.gamma << '\n';
    for (std::size_t level = 0; level < levels.size(); ++level) {
        const auto s = mg_summarize_gamma_visits(p.trace.visits, level);
        const std::size_t work_visits = level < p.trace.work.level_visits.size()
            ? p.trace.work.level_visits[level] : 0U;
        std::cout << "level=" << level
                  << " label=" << levels[level].label
                  << " visits=" << work_visits
                  << " visit_local_pre_geomean=" << s.visit_local_pre_geomean
                  << " visit_local_post_geomean=" << s.visit_local_post_geomean
                  << " max_after_coarse_over_initial=" << s.max_after_coarse_over_initial
                  << " last_after_post_over_initial=" << s.last_after_post_over_initial
                  << " last_coarse_correction_energy=" << s.last_correction_energy
                  << '\n';
    }
}

void print_gamma_baseline(const GammaSweepPoint& p) {
    std::cout << "\nGAMMA_LONG_BASELINE gamma=" << p.gamma << '\n'
              << std::fixed << std::setprecision(6)
              << "solve_ms=" << p.baseline.solve.solve_ms
              << " converged=" << (p.baseline.solve.converged ? "true" : "false")
              << " cycles=" << p.baseline.solve.cycles << '\n';
    for (std::size_t i = 0; i < p.baseline.solve.relative_residuals.size(); ++i) {
        std::cout << std::scientific << std::setprecision(9)
                  << "true_residual[" << i << "]="
                  << p.baseline.solve.relative_residuals[i];
        if (i > 0U) {
            std::cout << " cycle_q="
                      << p.baseline.solve.relative_residuals[i] /
                         p.baseline.solve.relative_residuals[i - 1U];
        }
        std::cout << '\n';
    }
    std::cout << "tail5_geomean_q=" << p.tail5_q
              << " tail10_geomean_q=" << p.tail10_q << '\n';
}

void run_gamma_cycle_sweep(std::size_t block_iterations,
                           std::size_t block_width,
                           std::size_t baseline_cycles,
                           std::size_t target_nodes,
                           std::size_t min_nodes,
                           std::size_t pre_smooth,
                           std::size_t post_smooth) {
    constexpr std::size_t m0 = 1U;
    constexpr std::size_t m1 = 2U;
    constexpr std::size_t m2 = 1U;
    constexpr double strength_threshold = 0.05;
    const std::array<std::size_t, 3> gammas{1U, 2U, 3U};

    if (block_iterations < 4U || block_width < 2U || block_width > 8U ||
        baseline_cycles < 8U || target_nodes < 2U || min_nodes == 0U ||
        min_nodes > target_nodes || pre_smooth == 0U || post_smooth == 0U) {
        throw std::invalid_argument("invalid gamma-cycle sweep options");
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
    const Apply apply0 = [&](const Vec& x) { return apply_fine_clamped(mesh, material, x); };
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
    const auto bending_shape = oracle_make_plate_bending_shape(space0);

    std::vector<Vec> seeds;
    seeds.reserve(block_width);
    seeds.push_back(bending_shape);
    for (std::size_t j = 1U; j < block_width; ++j) {
        const double phase = 0.13 + 0.487 * static_cast<double>(j);
        auto seed = oracle_deterministic_cr_seed(levels[0].dofs, phase);
        clamp_x0(mesh, seed);
        seeds.push_back(std::move(seed));
    }

    std::cout << "GFSS M5 recursive gamma-cycle spectral/timing sweep\n"
              << "problem=thin_plate mesh=64x64x8\n"
              << "reference_execution=cpu_fp64\n"
              << "hierarchy=frozen_theta_0p05_actual_metrics\n"
              << "gammas=1,2,3\n"
              << "gamma_semantics=recursive_coarse_cycles_on_same_coarse_equation\n"
              << "spectral_operator=E_gamma=I-B_gamma*A\n"
              << "spectral_inner_product=A0_energy\n"
              << "fixed_transfer_smoothing_steps=m0:1,m1:2,m2:1\n"
              << "block_iterations=" << block_iterations
              << " block_width=" << block_width
              << " baseline_cycles=" << baseline_cycles << '\n'
              << "pre_smooth_degree=" << pre_smooth
              << " post_smooth_degree=" << post_smooth << '\n'
              << "acceptance_target_asymptotic_q<=0.4\n"
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

    std::vector<GammaSweepPoint> points;
    points.reserve(gammas.size());
    for (const auto gamma : gammas) {
        GammaSweepPoint p;
        p.gamma = gamma;
        const auto spectral_start = Clock::now();
        p.spectrum = run_gamma_mg_spectrum(
            seeds, apply0, levels,
            block_iterations, pre_smooth, post_smooth, gamma);
        p.spectral_ms = elapsed_ms(spectral_start, Clock::now());
        if (p.spectrum.modes.empty()) {
            throw std::runtime_error("gamma-cycle spectrum returned no modes");
        }
        p.dominant_physical = mg_summarize_fine_mode(
            p.spectrum.modes.front(), space0, bending_shape);
        p.trace = mg_trace_gamma_error_cycle(
            levels, p.spectrum.modes.front(),
            pre_smooth, post_smooth, gamma);
        p.baseline = mg_solve_gamma_cycles(
            levels, rhs, 1.0e-30, baseline_cycles,
            pre_smooth, post_smooth, gamma);
        p.tail5_q = mg_tail_geomean_q(p.baseline.solve.relative_residuals, 5U);
        p.tail10_q = mg_tail_geomean_q(p.baseline.solve.relative_residuals, 10U);
        const auto final_residual = mg_true_residual(
            levels.front(), rhs, p.baseline.solve.x);
        const auto dominant_a_mode = apply0(p.spectrum.modes.front());
        p.final_residual_alignment = mg_abs_cosine(final_residual, dominant_a_mode);
        p.spectral_tail_gap = std::abs(
            std::abs(p.spectrum.eigenvalue.front()) - p.tail5_q);
        points.push_back(std::move(p));
    }

    for (const auto& p : points) {
        print_gamma_subspace_history(p);
        print_gamma_spectrum(p);
        std::cout << "dominant_physical gamma=" << p.gamma
                  << " x_fraction=" << std::scientific << std::setprecision(9)
                  << p.dominant_physical.component_fraction[0]
                  << " y_fraction=" << p.dominant_physical.component_fraction[1]
                  << " z_fraction=" << p.dominant_physical.component_fraction[2]
                  << " x_centroid=" << p.dominant_physical.weighted_centroid[0]
                  << " bending_abs_cosine=" << p.dominant_physical.bending_abs_cosine
                  << '\n';
        print_gamma_trace(p, levels);
        print_gamma_baseline(p);
    }

    const double gamma1_cycle_ms = points.front().baseline.solve.cycles > 0U
        ? points.front().baseline.solve.solve_ms /
          static_cast<double>(points.front().baseline.solve.cycles)
        : std::numeric_limits<double>::quiet_NaN();

    std::cout << "\nGAMMA_SWEEP_SUMMARY\n";
    std::size_t first_accept = 0U;
    for (const auto& p : points) {
        const double cycle_ms = p.baseline.solve.cycles > 0U
            ? p.baseline.solve.solve_ms /
              static_cast<double>(p.baseline.solve.cycles)
            : std::numeric_limits<double>::quiet_NaN();
        const double relative_cycle_cost =
            std::isfinite(gamma1_cycle_ms) && gamma1_cycle_ms > 0.0
                ? cycle_ms / gamma1_cycle_ms
                : std::numeric_limits<double>::quiet_NaN();
        const double rho = std::abs(p.spectrum.eigenvalue.front());
        const bool accepted = rho <= 0.4 && p.tail5_q <= 0.4;
        if (accepted && first_accept == 0U) first_accept = p.gamma;
        std::cout << "gamma=" << p.gamma
                  << " dominant_abs_eigenvalue=" << std::scientific << std::setprecision(9)
                  << rho
                  << " dominant_energy_q=" << p.spectrum.one_cycle_energy_q.front()
                  << " dominant_residual_q=" << p.spectrum.one_cycle_residual_q.front()
                  << " tail5_q=" << p.tail5_q
                  << " tail10_q=" << p.tail10_q
                  << " spectral_tail_gap=" << p.spectral_tail_gap
                  << " final_residual_alignment=" << p.final_residual_alignment
                  << " final_subspace_leakage="
                  << (p.spectrum.subspace_leakage.empty()
                        ? std::numeric_limits<double>::quiet_NaN()
                        : p.spectrum.subspace_leakage.back())
                  << " operator_symmetry_defect="
                  << p.spectrum.projected_operator_symmetry_defect
                  << " cycle_ms=" << std::fixed << std::setprecision(6) << cycle_ms
                  << " relative_cycle_cost=" << relative_cycle_cost
                  << " spectral_ms=" << p.spectral_ms
                  << " acceptance_q_le_0p4=" << (accepted ? "true" : "false")
                  << '\n';
    }

    std::cout << "\nGAMMA_CYCLE_VERDICT\n"
              << "first_gamma_meeting_spectral_and_tail_q_le_0p4=";
    if (first_accept == 0U) std::cout << "none\n";
    else std::cout << first_accept << '\n';
    if (points.size() == 3U) {
        const double rho1 = std::abs(points[0].spectrum.eigenvalue.front());
        const double rho2 = std::abs(points[1].spectrum.eigenvalue.front());
        const double rho3 = std::abs(points[2].spectrum.eigenvalue.front());
        std::cout << std::scientific << std::setprecision(9)
                  << "rho_gamma1=" << rho1
                  << " rho_gamma2=" << rho2
                  << " rho_gamma3=" << rho3 << '\n'
                  << "rho2_over_rho1=" << rho2 / std::max(rho1, 1.0e-300)
                  << " rho3_over_rho1=" << rho3 / std::max(rho1, 1.0e-300)
                  << " rho2_vs_rho1_squared="
                  << rho2 / std::max(rho1 * rho1, 1.0e-300)
                  << " rho3_vs_rho1_cubed="
                  << rho3 / std::max(rho1 * rho1 * rho1, 1.0e-300) << '\n';
    }
    std::cout << "total_reference_ms=" << std::fixed << std::setprecision(6)
              << elapsed_ms(total_start, Clock::now()) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t block_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 16U;
        const std::size_t block_width = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 4U;
        const std::size_t baseline_cycles = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 20U;
        const std::size_t target_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 12U;
        const std::size_t min_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 4U;
        const std::size_t pre_smooth = argc > 6
            ? static_cast<std::size_t>(std::stoull(argv[6])) : 3U;
        const std::size_t post_smooth = argc > 7
            ? static_cast<std::size_t>(std::stoull(argv[7])) : 3U;
        run_gamma_cycle_sweep(
            block_iterations, block_width, baseline_cycles,
            target_nodes, min_nodes, pre_smooth, post_smooth);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_gamma_cycle_sweep_bench "
                  << "[block_iterations=16 [block_width=4 [baseline_cycles=20 "
                  << "[target_nodes=12 [min_nodes=4 [pre_smooth=3 [post_smooth=3]]]]]]]\n";
        return 1;
    }
}
