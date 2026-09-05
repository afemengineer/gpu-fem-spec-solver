// M5 projected compatible-relaxation diagnostic.
//
// Diagnose whether the current theta=0.05 recursive SA hierarchy misses
// slowly-relaxed error components. Because aggregation coarse variables are
// not a literal C-point subset, use an A-orthogonal projected generalization
// of compatible relaxation:
//
//   Q_A = I - P (P^T A P)^-1 P^T A
//   e_{k+1} = Q_A S e_k
//
// where S is the existing block-Chebyshev error smoother. Run this at both
// L1->L2 and L2->L3. Dense Galerkin factors used by Q_A are diagnostic-only
// CPU/FP64 oracles; they are not proposed production storage.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"

namespace {

struct CrProbeResult {
    std::string label;
    bool valid{false};
    std::vector<double> cycle_q;
    double post_transient_geomean_q{std::numeric_limits<double>::quiet_NaN()};
    double initial_projection_coarse_residual_ratio{std::numeric_limits<double>::quiet_NaN()};
    double initial_projection_idempotence_error{std::numeric_limits<double>::quiet_NaN()};
    Vec final_mode;
};

struct FineModeSummary {
    std::array<double, 3> component_fraction{};
    std::array<double, 3> weighted_centroid{};
    std::array<double, 4> x_quartile_fraction{};
    std::size_t max_node{0U};
    std::array<double, 3> max_node_xyz{};
    std::array<double, 3> max_node_u{};
    double bending_correlation{0.0};
    double transverse_linear_correlation{0.0};
};

Vec deterministic_cr_seed(std::size_t n, double phase) {
    Vec v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = std::sin(0.017 * t + phase)
             + 0.41 * std::cos(0.053 * t - 0.37 * phase)
             + 0.13 * std::sin(0.101 * t + 0.19 * phase);
    }
    return v;
}

Vec make_plate_bending_shape(const gfss::ElasticityAggregationCoarseSpace& space) {
    Vec fine(3U * space.graph.coordinates.size(), 0.0);
    constexpr double z_mid = 0.0625;
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        if (space.graph.constrained[node] != 0U) continue;
        const auto& p = space.graph.coordinates[node];
        const double x = p[0];
        const double z = p[2] - z_mid;
        // Euler-Bernoulli-like smooth bending field about the y axis.
        fine[3U * node + 0U] = -2.0 * x * z;
        fine[3U * node + 2U] = x * x;
    }
    return fine;
}

Vec make_transverse_linear_shape(const gfss::ElasticityAggregationCoarseSpace& space) {
    Vec fine(3U * space.graph.coordinates.size(), 0.0);
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        if (space.graph.constrained[node] != 0U) continue;
        fine[3U * node + 2U] = space.graph.coordinates[node][0];
    }
    return fine;
}

double absolute_cosine(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("CR cosine size mismatch");
    const double denom = norm(a) * norm(b);
    return denom > 0.0 ? std::abs(dot(a, b)) / denom : 0.0;
}

double energy_norm(const Apply& apply, const Vec& x) {
    const auto ax = apply(x);
    const double e2 = dot(x, ax);
    if (!(e2 >= 0.0) || !std::isfinite(e2)) {
        throw std::runtime_error("CR energy norm became invalid");
    }
    return std::sqrt(std::max(0.0, e2));
}

void normalize_energy(const Apply& apply, Vec& x) {
    const double en = energy_norm(apply, x);
    if (!(en > 0.0) || !std::isfinite(en)) {
        throw std::runtime_error("CR projected seed has zero/invalid energy");
    }
    for (double& v : x) v /= en;
}

template <class Transfer>
Vec project_a_complement(const Vec& x,
                         const Apply& apply,
                         const Transfer& transfer,
                         const DenseCholesky& coarse_factor) {
    const auto ax = apply(x);
    const auto coarse_residual = transfer.restrict_transpose(ax);
    const auto coarse_correction = coarse_factor.solve(coarse_residual);
    const auto represented = transfer.prolong(coarse_correction);
    if (represented.size() != x.size()) {
        throw std::runtime_error("CR projection size mismatch");
    }
    Vec q = x;
    for (std::size_t i = 0; i < q.size(); ++i) q[i] -= represented[i];
    return q;
}

template <class Transfer>
std::pair<double, double> audit_projection(const Vec& seed,
                                           const Apply& apply,
                                           const Transfer& transfer,
                                           const DenseCholesky& coarse_factor) {
    const auto aseed = apply(seed);
    const auto before = transfer.restrict_transpose(aseed);
    const auto q = project_a_complement(seed, apply, transfer, coarse_factor);
    const auto aq = apply(q);
    const auto after = transfer.restrict_transpose(aq);
    const double residual_ratio = norm(after) / std::max(norm(before), 1.0e-300);

    const auto qq = project_a_complement(q, apply, transfer, coarse_factor);
    Vec diff(q.size(), 0.0);
    for (std::size_t i = 0; i < q.size(); ++i) diff[i] = qq[i] - q[i];
    const double idempotence = norm(diff) / std::max(norm(q), 1.0e-300);
    return {residual_ratio, idempotence};
}

template <class Transfer, class SmoothError>
CrProbeResult run_cr_probe(const std::string& label,
                           const Vec& seed,
                           const Apply& apply,
                           const Transfer& transfer,
                           const DenseCholesky& coarse_factor,
                           const SmoothError& smooth_error,
                           std::size_t iterations) {
    CrProbeResult result;
    result.label = label;
    if (iterations == 0U) return result;

    const auto audit = audit_projection(seed, apply, transfer, coarse_factor);
    result.initial_projection_coarse_residual_ratio = audit.first;
    result.initial_projection_idempotence_error = audit.second;

    auto e = project_a_complement(seed, apply, transfer, coarse_factor);
    const double initial_energy = energy_norm(apply, e);
    if (!(initial_energy > 1.0e-24) || !std::isfinite(initial_energy)) {
        return result;
    }
    normalize_energy(apply, e);
    result.valid = true;
    result.cycle_q.reserve(iterations);

    for (std::size_t it = 0; it < iterations; ++it) {
        auto next = smooth_error(e);
        next = project_a_complement(next, apply, transfer, coarse_factor);
        const double q = energy_norm(apply, next); // current e has unit A-energy
        if (!(q > 0.0) || !std::isfinite(q)) {
            throw std::runtime_error("CR contraction factor invalid");
        }
        result.cycle_q.push_back(q);
        for (double& v : next) v /= q;
        e = std::move(next);
    }

    if (result.cycle_q.size() > 2U) {
        double log_sum = 0.0;
        std::size_t count = 0U;
        for (std::size_t i = 2U; i < result.cycle_q.size(); ++i) {
            const double q = result.cycle_q[i];
            if (q > 0.0 && std::isfinite(q)) {
                log_sum += std::log(q);
                ++count;
            }
        }
        if (count > 0U) {
            result.post_transient_geomean_q =
                std::exp(log_sum / static_cast<double>(count));
        }
    }
    result.final_mode = std::move(e);
    return result;
}

FineModeSummary summarize_fine_mode(
    const Vec& fine,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const Vec& bending_shape,
    const Vec& transverse_linear_shape) {
    if (fine.size() != 3U * space.graph.coordinates.size()) {
        throw std::invalid_argument("CR physical mode size mismatch");
    }
    FineModeSummary summary;
    double total = 0.0;
    double max_amp2 = -1.0;
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        const double ux = fine[3U * node + 0U];
        const double uy = fine[3U * node + 1U];
        const double uz = fine[3U * node + 2U];
        const std::array<double, 3> comp{ux * ux, uy * uy, uz * uz};
        const double amp2 = comp[0] + comp[1] + comp[2];
        total += amp2;
        for (std::size_t c = 0; c < 3U; ++c) summary.component_fraction[c] += comp[c];
        const auto& p = space.graph.coordinates[node];
        for (std::size_t c = 0; c < 3U; ++c) summary.weighted_centroid[c] += amp2 * p[c];
        const std::size_t q = std::min<std::size_t>(3U,
            static_cast<std::size_t>(std::floor(std::max(0.0, std::min(0.999999999, p[0])) * 4.0)));
        summary.x_quartile_fraction[q] += amp2;
        if (amp2 > max_amp2) {
            max_amp2 = amp2;
            summary.max_node = node;
            summary.max_node_xyz = p;
            summary.max_node_u = {ux, uy, uz};
        }
    }
    if (total > 0.0) {
        for (double& v : summary.component_fraction) v /= total;
        for (double& v : summary.weighted_centroid) v /= total;
        for (double& v : summary.x_quartile_fraction) v /= total;
    }
    summary.bending_correlation = absolute_cosine(fine, bending_shape);
    summary.transverse_linear_correlation = absolute_cosine(fine, transverse_linear_shape);
    return summary;
}

const CrProbeResult* worst_valid_probe(const std::vector<CrProbeResult>& probes) {
    const CrProbeResult* worst = nullptr;
    for (const auto& probe : probes) {
        if (!probe.valid || !std::isfinite(probe.post_transient_geomean_q)) continue;
        if (worst == nullptr ||
            probe.post_transient_geomean_q > worst->post_transient_geomean_q) {
            worst = &probe;
        }
    }
    return worst;
}

void print_probe_group(const char* level,
                       const std::vector<CrProbeResult>& probes) {
    for (const auto& probe : probes) {
        std::cout << "\nCR_probe level=" << level
                  << " seed=" << probe.label
                  << " valid=" << (probe.valid ? "true" : "false") << '\n'
                  << std::scientific << std::setprecision(9)
                  << "projection_coarse_residual_ratio="
                  << probe.initial_projection_coarse_residual_ratio
                  << " projection_idempotence_relative_error="
                  << probe.initial_projection_idempotence_error << '\n';
        for (std::size_t i = 0; i < probe.cycle_q.size(); ++i) {
            std::cout << "cr_q[" << (i + 1U) << "]=" << probe.cycle_q[i] << '\n';
        }
        std::cout << "cr_post_transient_geomean_q="
                  << probe.post_transient_geomean_q << '\n';
    }
}

void print_mode_summary(const char* level,
                        const CrProbeResult& probe,
                        const FineModeSummary& summary) {
    std::cout << "\nCR_worst_mode level=" << level
              << " seed=" << probe.label << '\n'
              << std::scientific << std::setprecision(9)
              << "worst_post_transient_geomean_q="
              << probe.post_transient_geomean_q << '\n'
              << "physical_component_fraction_x=" << summary.component_fraction[0]
              << " physical_component_fraction_y=" << summary.component_fraction[1]
              << " physical_component_fraction_z=" << summary.component_fraction[2] << '\n'
              << "physical_amplitude_weighted_centroid_x=" << summary.weighted_centroid[0]
              << " y=" << summary.weighted_centroid[1]
              << " z=" << summary.weighted_centroid[2] << '\n'
              << "physical_x_quartile_fraction_0_25=" << summary.x_quartile_fraction[0]
              << " q25_50=" << summary.x_quartile_fraction[1]
              << " q50_75=" << summary.x_quartile_fraction[2]
              << " q75_100=" << summary.x_quartile_fraction[3] << '\n'
              << "physical_max_node=" << summary.max_node
              << " max_xyz=" << summary.max_node_xyz[0] << ','
              << summary.max_node_xyz[1] << ',' << summary.max_node_xyz[2]
              << " max_u=" << summary.max_node_u[0] << ','
              << summary.max_node_u[1] << ',' << summary.max_node_u[2] << '\n'
              << "physical_abs_cosine_with_euler_bernoulli_bending="
              << summary.bending_correlation
              << " physical_abs_cosine_with_transverse_linear="
              << summary.transverse_linear_correlation << '\n';
}

void run_compatible_relaxation(std::size_t cr_iterations,
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

    const auto a2_factor_start = Clock::now();
    const auto a2_factor = materialize_and_factor_bottom(
        apply2, transfer1_tentative.coarse_dofs);
    const double a2_factor_ms = elapsed_ms(a2_factor_start, Clock::now());

    const auto local_a1_apply = [&](const LocalColumns& x) {
        return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
    };
    const auto l2_local_start = Clock::now();
    const auto l2_basis = build_smoothed_candidate_supports(
        transfer1_tentative, strength1.graph, block1,
        block_omega1, m1, local_a1_apply);
    const auto block2 = build_metric_from_local_supports(
        transfer1_tentative, block1, l2_basis, local_a1_apply);
    const double l2_local_setup_ms = elapsed_ms(l2_local_start, Clock::now());
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
    const Apply apply3 = [&](const Vec& x) {
        return transfer2.restrict_transpose(apply2(transfer2.prolong(x)));
    };
    const auto a3_factor_start = Clock::now();
    const auto a3_factor = materialize_and_factor_bottom(
        apply3, transfer2_tentative.coarse_dofs);
    const double a3_factor_ms = elapsed_ms(a3_factor_start, Clock::now());
    const auto setup_stop = Clock::now();

    const Vec zero1(space0.coarse_dofs, 0.0);
    const auto smooth1_error = [&](const Vec& e) {
        auto x = e;
        chebyshev_l1_block_smooth(
            apply1, block1, block_lambda1, zero1, x, smoother_degree);
        return x;
    };
    const Vec zero2(transfer1_tentative.coarse_dofs, 0.0);
    const auto smooth2_error = [&](const Vec& e) {
        auto x = e;
        chebyshev_l1_block_smooth(
            apply2, block2, block_lambda2, zero2, x, smoother_degree);
        return x;
    };

    const auto bending_shape = make_plate_bending_shape(space0);
    const auto transverse_linear_shape = make_transverse_linear_shape(space0);
    const auto bending_l1 = transfer0.restrict_transpose(bending_shape);
    const auto bending_l2 = transfer1.restrict_transpose(bending_l1);

    std::vector<CrProbeResult> l1_probes;
    l1_probes.push_back(run_cr_probe(
        "deterministic_phase_0p17", deterministic_cr_seed(space0.coarse_dofs, 0.17),
        apply1, transfer1, a2_factor, smooth1_error, cr_iterations));
    l1_probes.push_back(run_cr_probe(
        "deterministic_phase_1p03", deterministic_cr_seed(space0.coarse_dofs, 1.03),
        apply1, transfer1, a2_factor, smooth1_error, cr_iterations));
    l1_probes.push_back(run_cr_probe(
        "plate_bending", bending_l1,
        apply1, transfer1, a2_factor, smooth1_error, cr_iterations));

    std::vector<CrProbeResult> l2_probes;
    l2_probes.push_back(run_cr_probe(
        "deterministic_phase_0p31",
        deterministic_cr_seed(transfer1_tentative.coarse_dofs, 0.31),
        apply2, transfer2, a3_factor, smooth2_error, cr_iterations));
    l2_probes.push_back(run_cr_probe(
        "deterministic_phase_1p19",
        deterministic_cr_seed(transfer1_tentative.coarse_dofs, 1.19),
        apply2, transfer2, a3_factor, smooth2_error, cr_iterations));
    l2_probes.push_back(run_cr_probe(
        "plate_bending_restricted", bending_l2,
        apply2, transfer2, a3_factor, smooth2_error, cr_iterations));

    std::cout << "GFSS M5 projected compatible-relaxation diagnostic\n"
              << "problem=thin_plate mesh=64x64x8\n"
              << "reference_execution=cpu_fp64\n"
              << "diagnostic=projected_compatible_relaxation_A_orthogonal_complement\n"
              << "operator=Q_A*S with Q_A=I-P(P^TAP)^-1P^TA\n"
              << "strength_threshold=" << strength_threshold << '\n'
              << "fixed_transfer_smoothing_steps=m0:1,m1:2,m2:1\n"
              << "smoother=actual_block_Chebyshev degree=" << smoother_degree << '\n'
              << "cr_iterations=" << cr_iterations << '\n'
              << "diagnostic_projection_materializes_A2=true\n"
              << "diagnostic_projection_materializes_A3=true\n"
              << "production_storage_implication=none_reference_only\n"
              << "acceptance_threshold=none_diagnostic_only\n\n"
              << "L1_dofs=" << space0.coarse_dofs
              << " L1_nodes=" << graph1_tentative.nodes()
              << " L2_dofs=" << transfer1_tentative.coarse_dofs
              << " L2_nodes=" << transfer1_tentative.coarse_graph.nodes()
              << " L3_dofs=" << transfer2_tentative.coarse_dofs
              << " L3_nodes=" << transfer2_tentative.coarse_graph.nodes() << '\n'
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
              << " A2_projection_materialize_factor_ms=" << a2_factor_ms
              << " A3_projection_materialize_factor_ms=" << a3_factor_ms
              << " hierarchy_setup_ms=" << elapsed_ms(setup_start, setup_stop) << '\n'
              << std::scientific << std::setprecision(9)
              << "A2_projection_symmetry_relative_defect=" << a2_factor.symmetry_relative_defect
              << " A2_projection_min_cholesky_pivot=" << a2_factor.min_pivot << '\n'
              << "A3_projection_symmetry_relative_defect=" << a3_factor.symmetry_relative_defect
              << " A3_projection_min_cholesky_pivot=" << a3_factor.min_pivot << '\n';

    print_probe_group("L1_to_L2", l1_probes);
    print_probe_group("L2_to_L3", l2_probes);

    const auto* worst1 = worst_valid_probe(l1_probes);
    const auto* worst2 = worst_valid_probe(l2_probes);
    if (worst1 != nullptr) {
        const auto fine_mode = transfer0.prolong(worst1->final_mode);
        const auto summary = summarize_fine_mode(
            fine_mode, space0, bending_shape, transverse_linear_shape);
        print_mode_summary("L1_to_L2", *worst1, summary);
    }
    if (worst2 != nullptr) {
        const auto l1_mode = transfer1.prolong(worst2->final_mode);
        const auto fine_mode = transfer0.prolong(l1_mode);
        const auto summary = summarize_fine_mode(
            fine_mode, space0, bending_shape, transverse_linear_shape);
        print_mode_summary("L2_to_L3", *worst2, summary);
    }

    if (worst1 != nullptr && worst2 != nullptr) {
        std::cout << "\nCR_level_comparison\n"
                  << std::scientific << std::setprecision(9)
                  << "L1_to_L2_worst_post_transient_q="
                  << worst1->post_transient_geomean_q
                  << " L2_to_L3_worst_post_transient_q="
                  << worst2->post_transient_geomean_q << '\n'
                  << "slower_complement="
                  << (worst1->post_transient_geomean_q >= worst2->post_transient_geomean_q
                        ? "L1_to_L2" : "L2_to_L3") << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t cr_iterations = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 12U;
        const std::size_t target_nodes = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 12U;
        const std::size_t min_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 4U;
        const std::size_t smoother_degree = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 3U;
        if (cr_iterations < 4U || target_nodes < 2U || min_nodes == 0U ||
            min_nodes > target_nodes || smoother_degree == 0U || smoother_degree > 8U) {
            throw std::invalid_argument("invalid compatible-relaxation options");
        }
        run_compatible_relaxation(
            cr_iterations, target_nodes, min_nodes, smoother_degree);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_compatible_relaxation_bench "
                  << "[cr_iterations=12 [target_nodes=12 [min_nodes=4 [smoother_degree=3]]]]\n";
        return 1;
    }
}
