// M5 level-selective recursive-cycle sweep on the validated thin-plate SA hierarchy.
// CPU/FP64 numerical reference only. The hierarchy is frozen; only recursion
// multiplicity at L0->L1 and L1->L2 changes. L2->L3 remains one exact solve.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "recursive_sa_cr_oracle_helpers.inc"
#include "recursive_sa_block_cr_spectral_helpers.inc"
#include "recursive_sa_full_vcycle_spectral_helpers.inc"
#include "recursive_sa_gamma_cycle_helpers.inc"
#include "recursive_sa_selective_gamma_helpers.inc"

namespace {

struct SelectiveConfig {
    const char* label;
    MgSelectiveSchedule repetitions;
};

struct SelectiveSweepPoint {
    SelectiveConfig config{};
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

void print_selective_subspace_history(const SelectiveSweepPoint& p) {
    std::cout << "\nSELECTIVE_SUBSPACE_HISTORY config=" << p.config.label << '\n';
    for (std::size_t i = 0; i < p.spectrum.subspace_leakage.size(); ++i) {
        std::cout << "block_iteration=" << (i + 1U)
                  << " subspace_leakage=" << std::scientific << std::setprecision(9)
                  << p.spectrum.subspace_leakage[i] << '\n';
    }
}

void print_selective_spectrum(const SelectiveSweepPoint& p) {
    std::cout << "\nSELECTIVE_SPECTRUM config=" << p.config.label << '\n';
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

void print_selective_trace(
    const SelectiveSweepPoint& p,
    const std::vector<gfss::ReferenceMultilevelLevel>& levels) {
    const auto& t = p.trace.top;
    const double e0 = std::max(t.initial_energy, 1.0e-300);
    const double r0 = std::max(t.initial_residual_norm, 1.0e-300);
    std::cout << "\nSELECTIVE_DOMINANT_TOP_TRACE config=" << p.config.label << '\n'
              << std::scientific << std::setprecision(9)
              << "initial_energy=" << t.initial_energy
              << " after_pre_energy_ratio=" << t.after_pre_energy / e0
              << " after_coarse_energy_ratio=" << t.after_coarse_energy / e0
              << " after_post_energy_ratio=" << t.after_post_energy / e0 << '\n'
              << "initial_residual_norm=" << t.initial_residual_norm
              << " after_pre_residual_ratio=" << t.after_pre_residual_norm / r0
              << " after_coarse_residual_ratio=" << t.after_coarse_residual_norm / r0
              << " after_post_residual_ratio=" << t.after_post_residual_norm / r0 << '\n';

    std::cout << "SELECTIVE_RECURSIVE_VISIT_SUMMARY config=" << p.config.label << '\n';
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

void print_selective_baseline(const SelectiveSweepPoint& p) {
    std::cout << "\nSELECTIVE_LONG_BASELINE config=" << p.config.label << '\n'
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

void run_selective_gamma_sweep(std::size_t block_iterations,
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
    const std::array<SelectiveConfig, 5> configs{{
        {"1x1", {1U, 1U, 1U}},
        {"1x2", {1U, 2U, 1U}},
        {"2x1", {2U, 1U, 1U}},
        {"2x2", {2U, 2U, 1U}},
        {"3x1", {3U, 1U, 1U}},
    }};

    if (block_iterations < 4U || block_width < 2U || block_width > 8U ||
        baseline_cycles < 8U || target_nodes < 2U || min_nodes == 0U ||
        min_nodes > target_nodes || pre_smooth == 0U || post_smooth == 0U) {
        throw std::invalid_argument("invalid selective gamma sweep options");
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

    std::cout << "GFSS M5 level-selective recursive-cycle spectral/timing sweep\n"
              << "problem=thin_plate mesh=64x64x8\n"
              << "reference_execution=cpu_fp64\n"
              << "hierarchy=frozen_theta_0p05_actual_metrics\n"
              << "configs=1x1,1x2,2x1,2x2,3x1\n"
              << "config_semantics=first_digit_L0_to_L1_repeats_second_digit_L1_to_L2_repeats\n"
              << "L2_to_L3_repeats=1 exact_bottom=true\n"
              << "recursive_repeats_continue_existing_coarse_iterate=true\n"
              << "spectral_operator=E_selective=I-B_selective*A\n"
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

    std::vector<SelectiveSweepPoint> points;
    points.reserve(configs.size());
    for (const auto& config : configs) {
        SelectiveSweepPoint p;
        p.config = config;
        const auto spectral_start = Clock::now();
        p.spectrum = run_selective_mg_spectrum(
            seeds, apply0, levels,
            block_iterations, pre_smooth, post_smooth, config.repetitions);
        p.spectral_ms = elapsed_ms(spectral_start, Clock::now());
        if (p.spectrum.modes.empty()) {
            throw std::runtime_error("selective gamma spectrum returned no modes");
        }
        p.dominant_physical = mg_summarize_fine_mode(
            p.spectrum.modes.front(), space0, bending_shape);
        p.trace = mg_trace_selective_error_cycle(
            levels, p.spectrum.modes.front(),
            pre_smooth, post_smooth, config.repetitions);
        p.baseline = mg_solve_selective_cycles(
            levels, rhs, 1.0e-30, baseline_cycles,
            pre_smooth, post_smooth, config.repetitions);
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
        print_selective_subspace_history(p);
        print_selective_spectrum(p);
        std::cout << "dominant_physical config=" << p.config.label
                  << " x_fraction=" << std::scientific << std::setprecision(9)
                  << p.dominant_physical.component_fraction[0]
                  << " y_fraction=" << p.dominant_physical.component_fraction[1]
                  << " z_fraction=" << p.dominant_physical.component_fraction[2]
                  << " x_centroid=" << p.dominant_physical.weighted_centroid[0]
                  << " bending_abs_cosine=" << p.dominant_physical.bending_abs_cosine
                  << '\n';
        print_selective_trace(p, levels);
        print_selective_baseline(p);
    }

    const double base_cycle_ms = points.front().baseline.solve.cycles > 0U
        ? points.front().baseline.solve.solve_ms /
          static_cast<double>(points.front().baseline.solve.cycles)
        : std::numeric_limits<double>::quiet_NaN();
    const double base_rho = std::abs(points.front().spectrum.eigenvalue.front());
    const double base_efficiency =
        std::isfinite(base_cycle_ms) && base_cycle_ms > 0.0 && base_rho > 0.0 && base_rho < 1.0
            ? -std::log(base_rho) / base_cycle_ms
            : std::numeric_limits<double>::quiet_NaN();

    std::cout << "\nSELECTIVE_GAMMA_SUMMARY\n";
    std::string first_accept = "none";
    double best_efficiency = -1.0;
    std::string best_efficiency_label = "none";
    double best_accepted_efficiency = -1.0;
    std::string best_accepted_label = "none";
    for (const auto& p : points) {
        const double cycle_ms = p.baseline.solve.cycles > 0U
            ? p.baseline.solve.solve_ms /
              static_cast<double>(p.baseline.solve.cycles)
            : std::numeric_limits<double>::quiet_NaN();
        const double relative_cycle_cost =
            std::isfinite(base_cycle_ms) && base_cycle_ms > 0.0
                ? cycle_ms / base_cycle_ms
                : std::numeric_limits<double>::quiet_NaN();
        const double rho = std::abs(p.spectrum.eigenvalue.front());
        const double efficiency =
            std::isfinite(cycle_ms) && cycle_ms > 0.0 && rho > 0.0 && rho < 1.0
                ? -std::log(rho) / cycle_ms
                : std::numeric_limits<double>::quiet_NaN();
        const double relative_efficiency =
            std::isfinite(base_efficiency) && base_efficiency > 0.0
                ? efficiency / base_efficiency
                : std::numeric_limits<double>::quiet_NaN();
        const double cycles_to_1e6 = rho > 0.0 && rho < 1.0
            ? std::log(1.0e-6) / std::log(rho)
            : std::numeric_limits<double>::infinity();
        const double estimated_ms_to_1e6 = cycles_to_1e6 * cycle_ms;
        const bool accepted = rho <= 0.4 && p.tail5_q <= 0.4;
        if (accepted && first_accept == "none") first_accept = p.config.label;
        if (std::isfinite(efficiency) && efficiency > best_efficiency) {
            best_efficiency = efficiency;
            best_efficiency_label = p.config.label;
        }
        if (accepted && std::isfinite(efficiency) &&
            efficiency > best_accepted_efficiency) {
            best_accepted_efficiency = efficiency;
            best_accepted_label = p.config.label;
        }

        const auto& visits = p.baseline.work.level_visits;
        const double cycles = std::max<std::size_t>(1U, p.baseline.solve.cycles);
        const double l1_visits_per_cycle = visits.size() > 1U
            ? static_cast<double>(visits[1]) / cycles : 0.0;
        const double l2_visits_per_cycle = visits.size() > 2U
            ? static_cast<double>(visits[2]) / cycles : 0.0;
        const double l3_visits_per_cycle = visits.size() > 3U
            ? static_cast<double>(visits[3]) / cycles : 0.0;

        std::cout << "config=" << p.config.label
                  << " l0_to_l1_repeats=" << p.config.repetitions[0]
                  << " l1_to_l2_repeats=" << p.config.repetitions[1]
                  << " l2_to_l3_repeats=" << p.config.repetitions[2]
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
                  << " log_reduction_per_ms=" << std::scientific << std::setprecision(9)
                  << efficiency
                  << " relative_log_reduction_efficiency=" << relative_efficiency
                  << " estimated_cycles_to_1e6=" << cycles_to_1e6
                  << " estimated_ms_to_1e6=" << estimated_ms_to_1e6
                  << " L1_visits_per_cycle=" << l1_visits_per_cycle
                  << " L2_visits_per_cycle=" << l2_visits_per_cycle
                  << " L3_visits_per_cycle=" << l3_visits_per_cycle
                  << " spectral_ms=" << std::fixed << std::setprecision(6)
                  << p.spectral_ms
                  << " acceptance_q_le_0p4=" << (accepted ? "true" : "false")
                  << '\n';
    }

    std::cout << "\nSELECTIVE_GAMMA_VERDICT\n"
              << "first_config_meeting_spectral_and_tail_q_le_0p4="
              << first_accept << '\n'
              << "best_log_reduction_per_ms_config=" << best_efficiency_label << '\n'
              << "best_accepted_log_reduction_per_ms_config="
              << best_accepted_label << '\n';

    const auto find_point = [&](const char* label) -> const SelectiveSweepPoint& {
        for (const auto& p : points) {
            if (std::string(p.config.label) == label) return p;
        }
        throw std::runtime_error("selective gamma comparison point missing");
    };
    const auto& p11 = find_point("1x1");
    const auto& p12 = find_point("1x2");
    const auto& p21 = find_point("2x1");
    const auto& p22 = find_point("2x2");
    const auto& p31 = find_point("3x1");
    const double rho11 = std::abs(p11.spectrum.eigenvalue.front());
    const double rho12 = std::abs(p12.spectrum.eigenvalue.front());
    const double rho21 = std::abs(p21.spectrum.eigenvalue.front());
    const double rho22 = std::abs(p22.spectrum.eigenvalue.front());
    const double rho31 = std::abs(p31.spectrum.eigenvalue.front());
    std::cout << std::scientific << std::setprecision(9)
              << "rho_1x1=" << rho11
              << " rho_1x2=" << rho12
              << " rho_2x1=" << rho21
              << " rho_2x2=" << rho22
              << " rho_3x1=" << rho31 << '\n'
              << "gain_deep_only_1x1_to_1x2=" << rho11 - rho12
              << " gain_first_coarse_only_1x1_to_2x1=" << rho11 - rho21
              << " extra_gain_2x1_to_2x2=" << rho21 - rho22
              << " extra_gain_2x1_to_3x1=" << rho21 - rho31 << '\n'
              << "total_reference_ms=" << std::fixed << std::setprecision(6)
              << elapsed_ms(total_start, Clock::now()) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t block_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 12U;
        const std::size_t block_width = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 4U;
        const std::size_t baseline_cycles = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 16U;
        const std::size_t target_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 12U;
        const std::size_t min_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 4U;
        const std::size_t pre_smooth = argc > 6
            ? static_cast<std::size_t>(std::stoull(argv[6])) : 3U;
        const std::size_t post_smooth = argc > 7
            ? static_cast<std::size_t>(std::stoull(argv[7])) : 3U;
        run_selective_gamma_sweep(
            block_iterations, block_width, baseline_cycles,
            target_nodes, min_nodes, pre_smooth, post_smooth);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_selective_gamma_sweep_bench "
                  << "[block_iterations=12 [block_width=4 [baseline_cycles=16 "
                  << "[target_nodes=12 [min_nodes=4 [pre_smooth=3 [post_smooth=3]]]]]]]\n";
        return 1;
    }
}
