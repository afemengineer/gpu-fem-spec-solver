// M5 block compatible-relaxation spectral diagnostic + nested oracle enrichment sweep.
// CPU/FP64 numerical reference only. No production construction is claimed.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "recursive_sa_cr_oracle_helpers.inc"
#include "recursive_sa_block_cr_spectral_helpers.inc"

namespace {

struct PhysicalModeSummary {
    double z_fraction{0.0};
    double x_centroid{0.0};
    double bending_abs_cosine{0.0};
};

struct SweepPoint {
    std::size_t k{0U};
    double postcheck_cr_q{std::numeric_limits<double>::quiet_NaN()};
    double vcycle_q{std::numeric_limits<double>::quiet_NaN()};
    double solve_ms{0.0};
    double adjoint_error{0.0};
    double bottom_oracle_error{0.0};
    double max_base_span_energy_fraction{0.0};
    double max_mode_energy_diag_defect{0.0};
    double max_mode_energy_offdiag{0.0};
    double bottom_min_pivot{0.0};
};

double spectral_residual_geomean_q(const std::vector<double>& residuals) {
    if (residuals.size() <= 2U) return std::numeric_limits<double>::quiet_NaN();
    double log_sum = 0.0;
    std::size_t count = 0U;
    for (std::size_t i = 2U; i < residuals.size(); ++i) {
        const double q = residuals[i] / residuals[i - 1U];
        if (q > 0.0 && std::isfinite(q)) {
            log_sum += std::log(q);
            ++count;
        }
    }
    return count > 0U
        ? std::exp(log_sum / static_cast<double>(count))
        : std::numeric_limits<double>::quiet_NaN();
}

double abs_cosine(const Vec& a, const Vec& b) {
    const double denom = norm(a) * norm(b);
    return denom > 0.0 ? std::abs(dot(a, b)) / denom : 0.0;
}

PhysicalModeSummary summarize_mode(
    const Vec& l2_mode,
    const FineSmoothedTransfer& transfer0,
    const L1BlockSmoothedTransfer& transfer1,
    const gfss::ElasticityAggregationCoarseSpace& space0,
    const Vec& bending_shape) {
    const auto l1 = transfer1.prolong(l2_mode);
    const auto fine = transfer0.prolong(l1);
    if (fine.size() != 3U * space0.graph.coordinates.size()) {
        throw std::runtime_error("spectral physical mode size mismatch");
    }
    PhysicalModeSummary out;
    double total = 0.0;
    double z2 = 0.0;
    double weighted_x = 0.0;
    for (std::size_t node = 0; node < space0.graph.coordinates.size(); ++node) {
        const double ux = fine[3U * node + 0U];
        const double uy = fine[3U * node + 1U];
        const double uz = fine[3U * node + 2U];
        const double amp2 = ux * ux + uy * uy + uz * uz;
        total += amp2;
        z2 += uz * uz;
        weighted_x += amp2 * space0.graph.coordinates[node][0];
    }
    if (total > 0.0) {
        out.z_fraction = z2 / total;
        out.x_centroid = weighted_x / total;
    }
    out.bending_abs_cosine = abs_cosine(fine, bending_shape);
    return out;
}

void print_solver_cycle_history(const char* tag,
                                std::size_t k,
                                const gfss::ReferenceMultilevelSolveResult& result) {
    std::cout << "\nVcycle_" << tag << " k=" << k << '\n'
              << std::fixed << std::setprecision(6)
              << "solve_ms=" << result.solve_ms
              << " converged=" << (result.converged ? "true" : "false")
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
    std::cout << "post_transient_geomean_q="
              << spectral_residual_geomean_q(result.relative_residuals) << '\n';
}

void run_block_cr_spectral_sweep(std::size_t block_iterations,
                                 std::size_t block_width,
                                 std::size_t postcheck_iterations,
                                 std::size_t max_cycles,
                                 std::size_t target_nodes,
                                 std::size_t min_nodes,
                                 std::size_t smoother_degree) {
    constexpr std::size_t m0 = 1U;
    constexpr std::size_t m1 = 2U;
    constexpr std::size_t m2 = 1U;
    constexpr double strength_threshold = 0.05;
    if (block_width < 8U) {
        throw std::invalid_argument("block width must be at least 8 for k={0,1,2,4,8}");
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
    const auto base_bottom = build_local_bottom(
        transfer2_tentative, block2, bottom_basis, local_a2_apply);
    const Apply apply3_base = [&](const Vec& x) {
        return transfer2.restrict_transpose(apply2(transfer2.prolong(x)));
    };
    const double base_bottom_oracle_error =
        bottom_local_oracle_error(base_bottom, apply3_base);

    const Vec zero2(transfer1_tentative.coarse_dofs, 0.0);
    const auto smooth2_error = [&](const Vec& e) {
        auto x = e;
        chebyshev_l1_block_smooth(
            apply2, block2, block_lambda2, zero2, x, smoother_degree);
        return x;
    };

    const auto bending_shape = oracle_make_plate_bending_shape(space0);
    const auto bending_l1 = transfer0.restrict_transpose(bending_shape);
    const auto bending_l2 = transfer1.restrict_transpose(bending_l1);
    std::vector<Vec> seeds;
    seeds.reserve(block_width);
    seeds.push_back(bending_l2);
    for (std::size_t j = 1U; j < block_width; ++j) {
        const double phase = 0.17 + 0.613 * static_cast<double>(j);
        seeds.push_back(oracle_deterministic_cr_seed(
            transfer1_tentative.coarse_dofs, phase));
    }

    const auto block_start = Clock::now();
    const auto spectrum = run_block_cr_spectrum(
        seeds, apply2, transfer2, base_bottom.factor,
        smooth2_error, block_iterations);
    const double block_cr_ms = elapsed_ms(block_start, Clock::now());

    std::vector<PhysicalModeSummary> mode_summary;
    mode_summary.reserve(spectrum.modes.size());
    for (const auto& mode : spectrum.modes) {
        mode_summary.push_back(summarize_mode(
            mode, transfer0, transfer1, space0, bending_shape));
    }

    std::cout << "GFSS M5 block-CR spectral enrichment sweep\n"
              << "problem=thin_plate mesh=64x64x8\n"
              << "reference_execution=cpu_fp64\n"
              << "strength_threshold=0.05\n"
              << "fixed_transfer_smoothing_steps=m0:1,m1:2,m2:1\n"
              << "diagnostic_level=L2_to_L3\n"
              << "block_CR_operator=Q_A*S\n"
              << "spectral_ordering=eigenvectors_of_(EV)^T_A(EV)\n"
              << "production_method_claim=false\n"
              << "enrichment_sweep_k=0,1,2,4,8\n"
              << "block_iterations=" << block_iterations
              << " block_width=" << block_width
              << " postcheck_iterations=" << postcheck_iterations
              << " smoother_degree=" << smoother_degree << '\n'
              << "pre_smooth_degree=3 post_smooth_degree=3\n"
              << "acceptance_target_post_transient_q<=0.4\n\n"
              << "L0_dofs=" << static_cast<std::size_t>(mesh.dof_count())
              << " L1_dofs=" << space0.coarse_dofs
              << " L2_dofs=" << transfer1_tentative.coarse_dofs
              << " base_L3_dofs=" << transfer2_tentative.coarse_dofs << '\n'
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
              << " base_bottom_local_assembly_ms=" << base_bottom.assembly_ms
              << " block_CR_ms=" << block_cr_ms << '\n'
              << std::scientific << std::setprecision(9)
              << "L1_local_block_vs_nested_oracle_relative_error=" << block1_oracle_error
              << " L2_local_block_vs_nested_oracle_relative_error=" << block2_oracle_error
              << " base_bottom_vs_nested_relative_error=" << base_bottom_oracle_error << '\n'
              << "block_final_A_orthogonality_error="
              << spectrum.final_a_orthogonality_error
              << " projected_CR_operator_symmetry_defect="
              << spectrum.projected_operator_symmetry_defect << '\n';

    std::cout << "\nBLOCK_CR_SUBSPACE_HISTORY\n";
    for (std::size_t i = 0; i < spectrum.subspace_leakage.size(); ++i) {
        std::cout << "block_iteration=" << (i + 1U)
                  << " subspace_leakage=" << std::scientific << std::setprecision(9)
                  << spectrum.subspace_leakage[i] << '\n';
    }

    std::cout << "\nBLOCK_CR_SPECTRUM\n";
    for (std::size_t i = 0; i < spectrum.modes.size(); ++i) {
        std::cout << "mode=" << i
                  << " one_step_energy_q=" << std::scientific << std::setprecision(9)
                  << spectrum.one_step_q[i]
                  << " projected_rayleigh_abs=" << spectrum.projected_rayleigh_abs[i]
                  << " physical_z_fraction=" << mode_summary[i].z_fraction
                  << " physical_x_centroid=" << mode_summary[i].x_centroid
                  << " physical_bending_abs_cosine=" << mode_summary[i].bending_abs_cosine
                  << '\n';
    }

    const std::array<std::size_t, 5> sweep_k{0U, 1U, 2U, 4U, 8U};
    std::vector<SweepPoint> sweep;
    sweep.reserve(sweep_k.size());
    const auto rhs = make_rhs(mesh);

    for (const std::size_t k : sweep_k) {
        if (k > spectrum.modes.size()) {
            throw std::runtime_error("spectral oracle sweep requests unavailable mode");
        }
        const MultiVectorEnrichedTransfer enriched_transfer{
            transfer2, spectrum.modes, k};
        const auto augmented = build_multi_vector_augmented_bottom(
            base_bottom, transfer2, apply2, spectrum.modes, k);
        const double adjoint_error = spectral_transfer_adjoint_error(
            enriched_transfer,
            transfer1_tentative.coarse_dofs,
            enriched_transfer.coarse_dofs());
        const Apply apply3_enriched = [&](const Vec& x) {
            return enriched_transfer.restrict_transpose(
                apply2(enriched_transfer.prolong(x)));
        };
        const double bottom_oracle_error =
            bottom_local_oracle_error(augmented.bottom, apply3_enriched);

        Vec post_seed;
        std::string post_seed_label;
        if (k < spectrum.modes.size()) {
            post_seed = spectrum.modes[k];
            post_seed_label = "next_block_spectral_mode";
        } else {
            post_seed = oracle_deterministic_cr_seed(
                transfer1_tentative.coarse_dofs, 5.371);
            post_seed_label = "fresh_deterministic_outside_block";
        }
        const auto postcheck = oracle_run_cr_probe(
            post_seed_label,
            post_seed,
            apply2, enriched_transfer, augmented.bottom.factor,
            smooth2_error, postcheck_iterations);
        if (!postcheck.valid) {
            throw std::runtime_error("spectral oracle post-enrichment CR probe invalid");
        }

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
            return enriched_transfer.restrict_transpose(r);
        };
        levels[2].prolong_from_coarse = [&](const Vec& c) {
            return enriched_transfer.prolong(c);
        };

        levels[3].dofs = enriched_transfer.coarse_dofs();
        levels[3].label = "L3_block_CR_spectral_oracle";
        levels[3].apply = [&](const Vec& x) {
            return apply_dense_bottom(augmented.bottom, x);
        };
        levels[3].bottom_solve = [&](const Vec& b) {
            return augmented.bottom.factor.solve(b);
        };

        const auto result = gfss::solve_reference_multilevel_vcycle(
            levels, rhs, 1.0e-6, max_cycles, 3U, 3U);
        print_solver_cycle_history("spectral_oracle", k, result);

        SweepPoint point;
        point.k = k;
        point.postcheck_cr_q = postcheck.post_transient_geomean_q;
        point.vcycle_q = spectral_residual_geomean_q(result.relative_residuals);
        point.solve_ms = result.solve_ms;
        point.adjoint_error = adjoint_error;
        point.bottom_oracle_error = bottom_oracle_error;
        point.max_base_span_energy_fraction = augmented.max_base_span_energy_fraction;
        point.max_mode_energy_diag_defect = augmented.max_mode_energy_diag_defect;
        point.max_mode_energy_offdiag = augmented.max_mode_energy_offdiag;
        point.bottom_min_pivot = augmented.bottom.factor.min_pivot;
        sweep.push_back(point);

        std::cout << "postcheck_CR k=" << k
                  << " seed=" << post_seed_label
                  << " q=" << std::scientific << std::setprecision(9)
                  << point.postcheck_cr_q
                  << " final_cycle_q=" << postcheck.cycle_q.back() << '\n';
    }

    std::cout << "\nSPECTRAL_ENRICHMENT_SWEEP\n";
    std::size_t first_accept_k = std::numeric_limits<std::size_t>::max();
    for (const auto& point : sweep) {
        const bool accepted = std::isfinite(point.vcycle_q) && point.vcycle_q <= 0.4;
        if (accepted && first_accept_k == std::numeric_limits<std::size_t>::max()) {
            first_accept_k = point.k;
        }
        std::cout << "k=" << point.k
                  << " L3_dofs=" << (transfer2_tentative.coarse_dofs + point.k)
                  << " postcheck_CR_q=" << std::scientific << std::setprecision(9)
                  << point.postcheck_cr_q
                  << " Vcycle_q=" << point.vcycle_q
                  << " solve_ms=" << std::fixed << std::setprecision(6) << point.solve_ms
                  << std::scientific << std::setprecision(9)
                  << " transfer_adjoint_error=" << point.adjoint_error
                  << " bottom_oracle_error=" << point.bottom_oracle_error
                  << " max_base_span_energy_fraction=" << point.max_base_span_energy_fraction
                  << " max_mode_energy_diag_defect=" << point.max_mode_energy_diag_defect
                  << " max_mode_energy_offdiag=" << point.max_mode_energy_offdiag
                  << " bottom_min_pivot=" << point.bottom_min_pivot
                  << " acceptance_q_le_0p4=" << (accepted ? "true" : "false")
                  << '\n';
    }

    std::cout << "\nBLOCK_CR_SPECTRAL_VERDICT\n"
              << "first_k_meeting_q_le_0p4=";
    if (first_accept_k == std::numeric_limits<std::size_t>::max()) {
        std::cout << "none\n";
    } else {
        std::cout << first_accept_k << '\n';
    }
    if (!sweep.empty()) {
        const double base_q = sweep.front().vcycle_q;
        const double k8_q = sweep.back().vcycle_q;
        std::cout << std::scientific << std::setprecision(9)
                  << "base_Vcycle_q=" << base_q
                  << " k8_Vcycle_q=" << k8_q
                  << " k8_relative_q_reduction="
                  << ((std::isfinite(base_q) && base_q > 0.0 && std::isfinite(k8_q))
                        ? (base_q - k8_q) / base_q
                        : std::numeric_limits<double>::quiet_NaN()) << '\n';
    }
    std::cout << std::fixed << std::setprecision(6)
              << "total_reference_ms=" << elapsed_ms(total_start, Clock::now()) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t block_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 32U;
        const std::size_t block_width = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 8U;
        const std::size_t postcheck_iterations = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t max_cycles = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 6U;
        const std::size_t target_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 12U;
        const std::size_t min_nodes = argc > 6
            ? static_cast<std::size_t>(std::stoull(argv[6])) : 4U;
        const std::size_t smoother_degree = argc > 7
            ? static_cast<std::size_t>(std::stoull(argv[7])) : 3U;
        if (block_iterations < 8U || block_width < 8U || block_width > 16U ||
            postcheck_iterations < 4U || max_cycles == 0U || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes ||
            smoother_degree == 0U || smoother_degree > 8U) {
            throw std::invalid_argument("invalid block-CR spectral sweep options");
        }
        run_block_cr_spectral_sweep(
            block_iterations, block_width, postcheck_iterations, max_cycles,
            target_nodes, min_nodes, smoother_degree);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_block_cr_spectral_enrichment_sweep_bench "
                  << "[block_iterations=32 [block_width=8 [postcheck_iterations=12 "
                  << "[max_cycles=6 [target_nodes=12 [min_nodes=4 [smoother_degree=3]]]]]]]\n";
        return 1;
    }
}
