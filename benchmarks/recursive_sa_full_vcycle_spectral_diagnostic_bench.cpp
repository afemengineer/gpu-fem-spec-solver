// M5 full-V-cycle spectral/error-propagation diagnostic.
// Diagnose the actual stationary multigrid error operator E_MG = I - B A,
// then trace the dominant mode through the recursive V-cycle. CPU/FP64 only.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "recursive_sa_cr_oracle_helpers.inc"
#include "recursive_sa_block_cr_spectral_helpers.inc"
#include "recursive_sa_full_vcycle_spectral_helpers.inc"

namespace {

void run_full_vcycle_spectral_diagnostic(std::size_t block_iterations,
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
    if (block_width < 2U || block_width > 12U || block_iterations < 4U ||
        baseline_cycles < 6U || target_nodes < 2U || min_nodes == 0U ||
        min_nodes > target_nodes || pre_smooth == 0U || post_smooth == 0U) {
        throw std::invalid_argument("invalid full-V-cycle spectral diagnostic options");
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

    const auto spectral_start = Clock::now();
    const auto spectrum = run_full_mg_spectrum(
        seeds, apply0, levels, block_iterations, pre_smooth, post_smooth);
    const double spectral_ms = elapsed_ms(spectral_start, Clock::now());
    if (spectrum.modes.empty()) {
        throw std::runtime_error("full-MG spectrum returned no modes");
    }

    std::vector<MgPhysicalSummary> physical;
    physical.reserve(spectrum.modes.size());
    for (const auto& mode : spectrum.modes) {
        physical.push_back(mg_summarize_fine_mode(mode, space0, bending_shape));
    }

    const auto dominant_trace = mg_trace_error_cycle(
        levels, spectrum.modes[0], pre_smooth, post_smooth);

    const auto rhs = make_rhs(mesh);
    const auto baseline = gfss::solve_reference_multilevel_vcycle(
        levels, rhs, 1.0e-30, baseline_cycles, pre_smooth, post_smooth);
    const double baseline_tail5_q = mg_tail_geomean_q(
        baseline.relative_residuals, 5U);
    const double baseline_tail10_q = mg_tail_geomean_q(
        baseline.relative_residuals, 10U);
    const auto final_residual = mg_true_residual(levels[0], rhs, baseline.x);
    const auto dominant_a_mode = apply0(spectrum.modes[0]);
    const double final_residual_alignment = mg_abs_cosine(
        final_residual, dominant_a_mode);
    const double spectral_tail_gap = std::abs(
        std::abs(spectrum.eigenvalue[0]) - baseline_tail5_q);

    std::cout << "GFSS M5 full-V-cycle spectral/error-propagation diagnostic\n"
              << "problem=thin_plate mesh=64x64x8\n"
              << "reference_execution=cpu_fp64\n"
              << "operator=E_MG=I-B*A_via_homogeneous_full_Vcycle\n"
              << "spectral_inner_product=A0_energy\n"
              << "strength_threshold=0.05\n"
              << "fixed_transfer_smoothing_steps=m0:1,m1:2,m2:1\n"
              << "block_iterations=" << block_iterations
              << " block_width=" << block_width
              << " baseline_cycles=" << baseline_cycles << '\n'
              << "pre_smooth_degree=" << pre_smooth
              << " post_smooth_degree=" << post_smooth << '\n'
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
              << " hierarchy_setup_ms=" << elapsed_ms(total_start, setup_stop)
              << " full_MG_block_spectral_ms=" << spectral_ms
              << " baseline_solve_ms=" << baseline.solve_ms << '\n'
              << std::scientific << std::setprecision(9)
              << "L1_local_block_vs_nested_oracle_relative_error=" << block1_oracle_error
              << " L2_local_block_vs_nested_oracle_relative_error=" << block2_oracle_error
              << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error << '\n'
              << "L1_transfer_adjoint_relative_error=" << transfer1_adjoint
              << " L2_transfer_adjoint_relative_error=" << transfer2_adjoint << '\n'
              << "full_MG_final_A_orthogonality_error="
              << spectrum.final_a_orthogonality_error
              << " full_MG_projected_operator_symmetry_defect="
              << spectrum.projected_operator_symmetry_defect << '\n';

    std::cout << "\nFULL_MG_SUBSPACE_HISTORY\n";
    for (std::size_t i = 0; i < spectrum.subspace_leakage.size(); ++i) {
        std::cout << "block_iteration=" << (i + 1U)
                  << " subspace_leakage=" << std::scientific << std::setprecision(9)
                  << spectrum.subspace_leakage[i] << '\n';
    }

    std::cout << "\nFULL_MG_SPECTRUM\n";
    for (std::size_t i = 0; i < spectrum.modes.size(); ++i) {
        std::cout << "mode=" << i
                  << " projected_eigenvalue=" << std::scientific << std::setprecision(9)
                  << spectrum.eigenvalue[i]
                  << " abs_eigenvalue=" << std::abs(spectrum.eigenvalue[i])
                  << " one_cycle_energy_q=" << spectrum.one_cycle_energy_q[i]
                  << " one_cycle_residual_q=" << spectrum.one_cycle_residual_q[i]
                  << " physical_x_fraction=" << physical[i].component_fraction[0]
                  << " physical_y_fraction=" << physical[i].component_fraction[1]
                  << " physical_z_fraction=" << physical[i].component_fraction[2]
                  << " physical_x_centroid=" << physical[i].weighted_centroid[0]
                  << " physical_bending_abs_cosine=" << physical[i].bending_abs_cosine
                  << '\n';
    }

    std::cout << "\nDOMINANT_MODE_VCYCLE_TRACE\n";
    for (std::size_t l = 0; l < dominant_trace.size(); ++l) {
        const auto& t = dominant_trace[l];
        const double denom = std::max(t.initial_residual_norm, 1.0e-300);
        std::cout << "level=" << l
                  << " label=" << t.label
                  << " initial_residual_norm=" << std::scientific << std::setprecision(9)
                  << t.initial_residual_norm
                  << " after_pre_over_initial=" << t.after_pre_residual_norm / denom
                  << " after_coarse_over_initial=" << t.after_coarse_residual_norm / denom
                  << " after_post_over_initial=" << t.after_post_residual_norm / denom
                  << " coarse_rhs_norm=" << t.coarse_rhs_norm
                  << " coarse_correction_norm=" << t.coarse_correction_norm
                  << '\n';
    }

    std::cout << "\nLONG_BASELINE\n"
              << std::fixed << std::setprecision(6)
              << "solve_ms=" << baseline.solve_ms
              << " converged=" << (baseline.converged ? "true" : "false")
              << " cycles=" << baseline.cycles << '\n';
    for (std::size_t i = 0; i < baseline.relative_residuals.size(); ++i) {
        std::cout << std::scientific << std::setprecision(9)
                  << "true_residual[" << i << "]=" << baseline.relative_residuals[i];
        if (i > 0U) {
            std::cout << " cycle_q="
                      << baseline.relative_residuals[i] /
                         baseline.relative_residuals[i - 1U];
        }
        std::cout << '\n';
    }
    std::cout << "baseline_tail5_geomean_q=" << std::scientific << std::setprecision(9)
              << baseline_tail5_q
              << " baseline_tail10_geomean_q=" << baseline_tail10_q << '\n';

    std::cout << "\nFULL_MG_SPECTRAL_VERDICT\n"
              << std::scientific << std::setprecision(9)
              << "dominant_abs_eigenvalue=" << std::abs(spectrum.eigenvalue[0])
              << " dominant_one_cycle_energy_q=" << spectrum.one_cycle_energy_q[0]
              << " dominant_one_cycle_residual_q=" << spectrum.one_cycle_residual_q[0] << '\n'
              << "baseline_tail5_q=" << baseline_tail5_q
              << " spectral_vs_tail5_abs_gap=" << spectral_tail_gap << '\n'
              << "final_residual_abs_cosine_with_A_dominant_mode="
              << final_residual_alignment << '\n'
              << "spectral_tail_consistent="
              << (std::isfinite(spectral_tail_gap) && spectral_tail_gap <= 0.05 ? "true" : "false")
              << '\n'
              << "total_reference_ms=" << std::fixed << std::setprecision(6)
              << elapsed_ms(total_start, Clock::now()) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t block_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 20U;
        const std::size_t block_width = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 6U;
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
        run_full_vcycle_spectral_diagnostic(
            block_iterations, block_width, baseline_cycles,
            target_nodes, min_nodes, pre_smooth, post_smooth);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_full_vcycle_spectral_diagnostic_bench "
                  << "[block_iterations=20 [block_width=6 [baseline_cycles=20 "
                  << "[target_nodes=12 [min_nodes=4 [pre_smooth=3 [post_smooth=3]]]]]]]\n";
        return 1;
    }
}
