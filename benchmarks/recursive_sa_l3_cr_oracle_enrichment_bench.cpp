// M5 one-vector oracle enrichment test.
//
// Rebuild the current theta=0.05 recursive hierarchy, converge the slowest
// sampled L2->L3 compatible-relaxation mode, re-project it into the exact
// A2-orthogonal complement of the existing L3 space, append exactly that one
// vector to P2, rebuild only the tiny bottom Galerkin solve, and rerun the same
// V-cycle. This is a causal numerical oracle, not a production construction.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "recursive_sa_cr_oracle_helpers.inc"

namespace {

struct OneVectorEnrichedTransfer {
    const L1BlockSmoothedTransfer& base;
    const Vec& mode;

    std::size_t coarse_dofs() const {
        return base.tentative.coarse_dofs + 1U;
    }

    Vec prolong(const Vec& coarse) const {
        const std::size_t base_n = base.tentative.coarse_dofs;
        if (coarse.size() != base_n + 1U || mode.size() != base.tentative.fine_dofs) {
            throw std::invalid_argument("L3 oracle-enriched prolongation size mismatch");
        }
        Vec base_coarse(base_n, 0.0);
        std::copy(coarse.begin(), coarse.begin() + static_cast<std::ptrdiff_t>(base_n),
                  base_coarse.begin());
        auto fine = base.prolong(base_coarse);
        const double alpha = coarse[base_n];
        for (std::size_t i = 0; i < fine.size(); ++i) fine[i] += alpha * mode[i];
        return fine;
    }

    Vec restrict_transpose(const Vec& fine) const {
        if (fine.size() != mode.size()) {
            throw std::invalid_argument("L3 oracle-enriched restriction size mismatch");
        }
        auto coarse = base.restrict_transpose(fine);
        coarse.push_back(dot(mode, fine));
        return coarse;
    }
};

struct AugmentedBottomResult {
    LocalBottomReference bottom;
    double enrichment_energy{0.0};
    double base_span_energy_fraction{0.0};
    double schur_complement{0.0};
};

void factor_dense_bottom_in_place(LocalBottomReference& bottom) {
    const std::size_t n = bottom.factor.n;
    if (bottom.values.size() != n * n || n == 0U) {
        throw std::invalid_argument("oracle augmented bottom size mismatch");
    }
    bottom.factor.lower.assign(n * n, 0.0);
    bottom.factor.symmetry_relative_defect = 0.0;
    bottom.factor.min_pivot = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = bottom.values[i * n + j];
            for (std::size_t k = 0; k < j; ++k) {
                value -= bottom.factor.lower[i * n + k] *
                         bottom.factor.lower[j * n + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("oracle augmented bottom Cholesky lost SPD");
                }
                bottom.factor.min_pivot = std::min(bottom.factor.min_pivot, value);
                bottom.factor.lower[i * n + j] = std::sqrt(value);
            } else {
                bottom.factor.lower[i * n + j] =
                    value / bottom.factor.lower[j * n + j];
            }
        }
    }
}

AugmentedBottomResult build_one_vector_augmented_bottom(
    const LocalBottomReference& base_bottom,
    const L1BlockSmoothedTransfer& base_transfer,
    const Apply& apply2,
    const Vec& mode) {
    const auto start = Clock::now();
    const std::size_t n = base_transfer.tentative.coarse_dofs;
    if (base_bottom.factor.n != n || base_bottom.values.size() != n * n ||
        mode.size() != base_transfer.tentative.fine_dofs) {
        throw std::invalid_argument("oracle augmented bottom input mismatch");
    }

    const auto av = apply2(mode);
    const auto cross = base_transfer.restrict_transpose(av);
    const double alpha = dot(mode, av);
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        throw std::runtime_error("oracle enrichment energy invalid");
    }
    const auto base_solve_cross = base_bottom.factor.solve(cross);
    const double captured = dot(cross, base_solve_cross);
    const double schur = alpha - captured;
    if (!(schur > 0.0) || !std::isfinite(schur)) {
        throw std::runtime_error("oracle enrichment Schur complement invalid");
    }

    AugmentedBottomResult result;
    result.enrichment_energy = alpha;
    result.base_span_energy_fraction = captured / alpha;
    result.schur_complement = schur;
    const std::size_t na = n + 1U;
    result.bottom.factor.n = na;
    result.bottom.values.assign(na * na, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result.bottom.values[i * na + j] = base_bottom.values[i * n + j];
        }
        result.bottom.values[i * na + n] = cross[i];
        result.bottom.values[n * na + i] = cross[i];
    }
    result.bottom.values[n * na + n] = alpha;
    factor_dense_bottom_in_place(result.bottom);
    result.bottom.assembly_ms = elapsed_ms(start, Clock::now());
    return result;
}

template <class Transfer>
double oracle_transfer_adjoint_error(const Transfer& transfer,
                                     std::size_t fine_dofs,
                                     std::size_t coarse_dofs) {
    const auto fine = oracle_deterministic_cr_seed(fine_dofs, 0.73);
    const auto coarse = oracle_deterministic_cr_seed(coarse_dofs, 0.29);
    const auto pc = transfer.prolong(coarse);
    const auto ptf = transfer.restrict_transpose(fine);
    const double lhs = dot(pc, fine);
    const double rhs = dot(coarse, ptf);
    return std::abs(lhs - rhs) / std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

double residual_post_transient_geomean_q(const std::vector<double>& residuals) {
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

void print_cr_probes(const char* tag,
                     const std::vector<OracleCrProbeResult>& probes) {
    for (const auto& probe : probes) {
        std::cout << "\nCR_" << tag
                  << " seed=" << probe.label
                  << " valid=" << (probe.valid ? "true" : "false") << '\n'
                  << std::scientific << std::setprecision(9)
                  << "projection_coarse_residual_ratio="
                  << probe.projection_coarse_residual_ratio
                  << " projection_idempotence_relative_error="
                  << probe.projection_idempotence_error << '\n';
        for (std::size_t i = 0; i < probe.cycle_q.size(); ++i) {
            std::cout << "cr_q[" << (i + 1U) << "]=" << probe.cycle_q[i] << '\n';
        }
        std::cout << "cr_post_transient_geomean_q="
                  << probe.post_transient_geomean_q << '\n';
    }
}

template <class Result>
void print_solver_result(const char* tag, const Result& result) {
    const double q = residual_post_transient_geomean_q(result.relative_residuals);
    std::cout << "\nVcycle_" << tag << '\n'
              << std::fixed << std::setprecision(6)
              << "solve_ms=" << result.solve_ms
              << " converged=" << (result.converged ? "true" : "false")
              << " cycles=" << result.cycles << '\n';
    for (std::size_t i = 0; i < result.relative_residuals.size(); ++i) {
        std::cout << std::scientific << std::setprecision(9)
                  << tag << "_true_residual[" << i << "]="
                  << result.relative_residuals[i];
        if (i > 0U) {
            std::cout << " cycle_q="
                      << result.relative_residuals[i] /
                         result.relative_residuals[i - 1U];
        }
        std::cout << '\n';
    }
    std::cout << tag << "_post_transient_geomean_q="
              << std::scientific << std::setprecision(9) << q << '\n';
}

void run_l3_cr_oracle_enrichment(std::size_t cr_iterations,
                                 std::size_t max_cycles,
                                 std::size_t target_nodes,
                                 std::size_t min_nodes,
                                 std::size_t smoother_degree) {
    constexpr std::size_t m0 = 1U;
    constexpr std::size_t m1 = 2U;
    constexpr std::size_t m2 = 1U;
    constexpr double strength_threshold = 0.05;

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
    const FineSmoothedTransfer transfer0{mesh, material, space0, fine_inverse, omega0, m0};
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
    const auto l2_setup_start = Clock::now();
    const auto l2_basis = build_smoothed_candidate_supports(
        transfer1_tentative, strength1.graph, block1,
        block_omega1, m1, local_a1_apply);
    const auto block2 = build_metric_from_local_supports(
        transfer1_tentative, block1, l2_basis, local_a1_apply);
    const double l2_local_setup_ms = elapsed_ms(l2_setup_start, Clock::now());
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

    const auto cr_start = Clock::now();
    std::vector<OracleCrProbeResult> base_cr;
    base_cr.push_back(oracle_run_cr_probe(
        "deterministic_phase_0p31",
        oracle_deterministic_cr_seed(transfer1_tentative.coarse_dofs, 0.31),
        apply2, transfer2, base_bottom.factor, smooth2_error, cr_iterations));
    base_cr.push_back(oracle_run_cr_probe(
        "deterministic_phase_1p19",
        oracle_deterministic_cr_seed(transfer1_tentative.coarse_dofs, 1.19),
        apply2, transfer2, base_bottom.factor, smooth2_error, cr_iterations));
    base_cr.push_back(oracle_run_cr_probe(
        "plate_bending_restricted", bending_l2,
        apply2, transfer2, base_bottom.factor, smooth2_error, cr_iterations));
    const auto* worst_base_cr = oracle_worst_valid_probe(base_cr);
    if (worst_base_cr == nullptr) {
        throw std::runtime_error("oracle enrichment found no valid base CR mode");
    }

    auto enrichment = oracle_project_a_complement(
        worst_base_cr->final_mode, apply2, transfer2, base_bottom.factor);
    oracle_normalize_energy(apply2, enrichment);
    const auto av = apply2(enrichment);
    const auto base_cross = transfer2.restrict_transpose(av);
    const double base_cross_norm = norm(base_cross);
    const double enrichment_energy = dot(enrichment, av);

    const OneVectorEnrichedTransfer enriched_transfer{transfer2, enrichment};
    const double enriched_transfer_adjoint = oracle_transfer_adjoint_error(
        enriched_transfer,
        transfer1_tentative.coarse_dofs,
        enriched_transfer.coarse_dofs());
    const auto augmented = build_one_vector_augmented_bottom(
        base_bottom, transfer2, apply2, enrichment);
    const Apply apply3_enriched = [&](const Vec& x) {
        return enriched_transfer.restrict_transpose(
            apply2(enriched_transfer.prolong(x)));
    };
    const double enriched_bottom_oracle_error =
        bottom_local_oracle_error(augmented.bottom, apply3_enriched);

    std::vector<OracleCrProbeResult> enriched_cr;
    enriched_cr.push_back(oracle_run_cr_probe(
        "deterministic_phase_0p31",
        oracle_deterministic_cr_seed(transfer1_tentative.coarse_dofs, 0.31),
        apply2, enriched_transfer, augmented.bottom.factor,
        smooth2_error, cr_iterations));
    enriched_cr.push_back(oracle_run_cr_probe(
        "deterministic_phase_1p19",
        oracle_deterministic_cr_seed(transfer1_tentative.coarse_dofs, 1.19),
        apply2, enriched_transfer, augmented.bottom.factor,
        smooth2_error, cr_iterations));
    enriched_cr.push_back(oracle_run_cr_probe(
        "plate_bending_restricted", bending_l2,
        apply2, enriched_transfer, augmented.bottom.factor,
        smooth2_error, cr_iterations));
    const auto* worst_enriched_cr = oracle_worst_valid_probe(enriched_cr);
    const double cr_total_ms = elapsed_ms(cr_start, Clock::now());
    if (worst_enriched_cr == nullptr) {
        throw std::runtime_error("oracle enrichment found no valid enriched CR mode");
    }

    const auto setup_stop = Clock::now();

    std::vector<gfss::ReferenceMultilevelLevel> base_levels(4U);
    base_levels[0].dofs = static_cast<std::size_t>(mesh.dof_count());
    base_levels[0].label = "L0_fine_matrix_free";
    base_levels[0].diagnostic_lambda_max = lambda0;
    base_levels[0].apply = apply0;
    base_levels[0].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_smooth(apply0, fine_inverse, lambda0, b, x, degree);
        clamp_x0(mesh, x);
    };
    base_levels[0].restrict_to_coarse = [&](const Vec& r) {
        return transfer0.restrict_transpose(r);
    };
    base_levels[0].prolong_from_coarse = [&](const Vec& c) {
        return transfer0.prolong(c);
    };

    base_levels[1].dofs = space0.coarse_dofs;
    base_levels[1].label = "L1_actual_block_metric_actual_strength_graph";
    base_levels[1].diagnostic_lambda_max = block_lambda1;
    base_levels[1].apply = apply1;
    base_levels[1].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply1, block1, block_lambda1, b, x, degree);
    };
    base_levels[1].restrict_to_coarse = [&](const Vec& r) {
        return transfer1.restrict_transpose(r);
    };
    base_levels[1].prolong_from_coarse = [&](const Vec& c) {
        return transfer1.prolong(c);
    };

    base_levels[2].dofs = transfer1_tentative.coarse_dofs;
    base_levels[2].label = "L2_local_exact_block_metric";
    base_levels[2].diagnostic_lambda_max = block_lambda2;
    base_levels[2].apply = apply2;
    base_levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply2, block2, block_lambda2, b, x, degree);
    };
    base_levels[2].restrict_to_coarse = [&](const Vec& r) {
        return transfer2.restrict_transpose(r);
    };
    base_levels[2].prolong_from_coarse = [&](const Vec& c) {
        return transfer2.prolong(c);
    };

    base_levels[3].dofs = transfer2_tentative.coarse_dofs;
    base_levels[3].label = "L3_base_local_dense_bottom";
    base_levels[3].apply = [&](const Vec& x) { return apply_dense_bottom(base_bottom, x); };
    base_levels[3].bottom_solve = [&](const Vec& b) { return base_bottom.factor.solve(b); };

    auto enriched_levels = base_levels;
    enriched_levels[2].restrict_to_coarse = [&](const Vec& r) {
        return enriched_transfer.restrict_transpose(r);
    };
    enriched_levels[2].prolong_from_coarse = [&](const Vec& c) {
        return enriched_transfer.prolong(c);
    };
    enriched_levels[3].dofs = enriched_transfer.coarse_dofs();
    enriched_levels[3].label = "L3_plus_one_CR_oracle_mode";
    enriched_levels[3].apply = [&](const Vec& x) {
        return apply_dense_bottom(augmented.bottom, x);
    };
    enriched_levels[3].bottom_solve = [&](const Vec& b) {
        return augmented.bottom.factor.solve(b);
    };

    const auto rhs = make_rhs(mesh);
    const auto base_result = gfss::solve_reference_multilevel_vcycle(
        base_levels, rhs, 1.0e-6, max_cycles, 3U, 3U);
    const auto enriched_result = gfss::solve_reference_multilevel_vcycle(
        enriched_levels, rhs, 1.0e-6, max_cycles, 3U, 3U);

    const double base_q = residual_post_transient_geomean_q(base_result.relative_residuals);
    const double enriched_q = residual_post_transient_geomean_q(
        enriched_result.relative_residuals);
    const double q_reduction = std::isfinite(base_q) && base_q > 0.0 && std::isfinite(enriched_q)
        ? (base_q - enriched_q) / base_q
        : std::numeric_limits<double>::quiet_NaN();

    std::cout << "GFSS M5 one-vector L3 CR oracle enrichment\n"
              << "problem=thin_plate mesh=64x64x8\n"
              << "reference_execution=cpu_fp64\n"
              << "strength_threshold=0.05\n"
              << "fixed_transfer_smoothing_steps=m0:1,m1:2,m2:1\n"
              << "oracle_mode_source=worst_sampled_L2_to_L3_projected_CR_mode\n"
              << "oracle_mode_count=1\n"
              << "production_method_claim=false\n"
              << "dense_A1_materialized=false dense_A2_materialized=false\n"
              << "bottom_direct_dense=true\n"
              << "pre_smooth_degree=3 post_smooth_degree=3\n"
              << "acceptance_target_enriched_post_transient_q<=0.4\n\n"
              << "L0_dofs=" << static_cast<std::size_t>(mesh.dof_count())
              << " L1_dofs=" << space0.coarse_dofs
              << " L2_dofs=" << transfer1_tentative.coarse_dofs
              << " base_L3_dofs=" << transfer2_tentative.coarse_dofs
              << " enriched_L3_dofs=" << enriched_transfer.coarse_dofs() << '\n'
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
              << " L2_local_support_block_setup_ms=" << l2_local_setup_ms
              << " base_bottom_local_assembly_ms=" << base_bottom.assembly_ms
              << " augmented_bottom_assembly_ms=" << augmented.bottom.assembly_ms
              << " CR_mode_and_postcheck_ms=" << cr_total_ms
              << " total_setup_and_diagnostic_ms=" << elapsed_ms(setup_start, setup_stop) << '\n'
              << std::scientific << std::setprecision(9)
              << "L1_local_block_vs_nested_oracle_relative_error=" << block1_oracle_error
              << " L2_local_block_vs_nested_oracle_relative_error=" << block2_oracle_error << '\n'
              << "base_bottom_vs_nested_relative_error=" << base_bottom_oracle_error
              << " enriched_bottom_vs_nested_relative_error=" << enriched_bottom_oracle_error << '\n'
              << "enriched_transfer_adjoint_relative_error=" << enriched_transfer_adjoint << '\n'
              << "selected_CR_seed=" << worst_base_cr->label
              << " selected_base_CR_post_transient_q="
              << worst_base_cr->post_transient_geomean_q << '\n'
              << "enrichment_A2_energy=" << enrichment_energy
              << " base_cross_norm=" << base_cross_norm
              << " base_span_energy_fraction=" << augmented.base_span_energy_fraction
              << " augmented_bottom_schur_complement=" << augmented.schur_complement << '\n'
              << "base_bottom_min_cholesky_pivot=" << base_bottom.factor.min_pivot
              << " augmented_bottom_min_cholesky_pivot="
              << augmented.bottom.factor.min_pivot << '\n';

    print_cr_probes("before_enrichment", base_cr);
    print_cr_probes("after_enrichment", enriched_cr);

    std::cout << "\nCR_oracle_summary\n"
              << std::scientific << std::setprecision(9)
              << "base_worst_post_transient_q="
              << worst_base_cr->post_transient_geomean_q
              << " enriched_worst_post_transient_q="
              << worst_enriched_cr->post_transient_geomean_q << '\n';

    print_solver_result("base", base_result);
    print_solver_result("enriched_one_mode", enriched_result);

    std::cout << "\nORACLE_ENRICHMENT_VERDICT\n"
              << std::scientific << std::setprecision(9)
              << "base_post_transient_geomean_q=" << base_q
              << " enriched_post_transient_geomean_q=" << enriched_q
              << " relative_q_reduction=" << q_reduction << '\n'
              << "acceptance_q_le_0p4="
              << (std::isfinite(enriched_q) && enriched_q <= 0.4 ? "true" : "false")
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t cr_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 12U;
        const std::size_t max_cycles = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 6U;
        const std::size_t target_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t min_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 4U;
        const std::size_t smoother_degree = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 3U;
        if (cr_iterations < 4U || max_cycles == 0U || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes ||
            smoother_degree == 0U || smoother_degree > 8U) {
            throw std::invalid_argument("invalid L3 CR oracle enrichment options");
        }
        run_l3_cr_oracle_enrichment(
            cr_iterations, max_cycles, target_nodes, min_nodes, smoother_degree);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_l3_cr_oracle_enrichment_bench "
                  << "[cr_iterations=12 [max_cycles=6 [target_nodes=12 [min_nodes=4 [smoother_degree=3]]]]]\n";
        return 1;
    }
}
