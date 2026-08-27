// Numerical-reference experiment: keep the validated actual-A2 hierarchy and
// replace only the L2 scalar Jacobi metric with exact algebraic-node block
// Jacobi. The diagonal blocks are taken directly from the materialized smoothed
// Galerkin A2. Reuse the actual-A2 helper implementation but suppress its entry
// point so this translation unit owns the benchmark main().
#define GFSS_RECURSIVE_SA_ACTUAL_A2_NO_MAIN
#include "recursive_sa_actual_a2_reference_bench.cpp"
#undef GFSS_RECURSIVE_SA_ACTUAL_A2_NO_MAIN

namespace {

struct BlockJacobiMetric {
    std::vector<std::size_t> dof_offsets;
    std::vector<std::size_t> value_offsets;
    std::vector<double> lower; // one row-major lower-triangular Cholesky block/node
    std::size_t min_rank{0};
    std::size_t max_rank{0};
    double min_cholesky_pivot{0.0};

    std::size_t nodes() const {
        return dof_offsets.empty() ? 0U : dof_offsets.size() - 1U;
    }

    std::size_t dofs() const {
        return dof_offsets.empty() ? 0U : dof_offsets.back();
    }

    std::size_t storage_bytes() const {
        return dof_offsets.size() * sizeof(std::size_t) +
               value_offsets.size() * sizeof(std::size_t) +
               lower.size() * sizeof(double);
    }

    Vec solve(const Vec& rhs) const {
        if (rhs.size() != dofs()) {
            throw std::invalid_argument("block-Jacobi solve size mismatch");
        }
        Vec result(rhs.size(), 0.0);
        for (std::size_t a = 0; a < nodes(); ++a) {
            const std::size_t begin = dof_offsets[a];
            const std::size_t end = dof_offsets[a + 1U];
            const std::size_t rank = end - begin;
            const double* l = lower.data() + value_offsets[a];
            std::array<double, kCandidates> y{};
            if (rank > y.size()) {
                throw std::runtime_error("block-Jacobi rank exceeds candidate capacity");
            }
            for (std::size_t i = 0; i < rank; ++i) {
                double value = rhs[begin + i];
                for (std::size_t j = 0; j < i; ++j) value -= l[i * rank + j] * y[j];
                y[i] = value / l[i * rank + i];
            }
            for (std::size_t ii = rank; ii-- > 0U;) {
                double value = y[ii];
                for (std::size_t j = ii + 1U; j < rank; ++j) {
                    value -= l[j * rank + ii] * result[begin + j];
                }
                result[begin + ii] = value / l[ii * rank + ii];
            }
        }
        return result;
    }

    double norm2(const Vec& x) const {
        if (x.size() != dofs()) {
            throw std::invalid_argument("block-Jacobi norm size mismatch");
        }
        double sum = 0.0;
        for (std::size_t a = 0; a < nodes(); ++a) {
            const std::size_t begin = dof_offsets[a];
            const std::size_t end = dof_offsets[a + 1U];
            const std::size_t rank = end - begin;
            const double* l = lower.data() + value_offsets[a];
            // x^T M x = ||L^T x||_2^2 for M=L L^T.
            for (std::size_t j = 0; j < rank; ++j) {
                double value = 0.0;
                for (std::size_t i = j; i < rank; ++i) {
                    value += l[i * rank + j] * x[begin + i];
                }
                sum += value * value;
            }
        }
        return sum;
    }
};

BlockJacobiMetric build_block_jacobi_metric(
    const DenseOperatorReference& dense,
    const AlgebraicNodeGraph& layout) {
    if (layout.nodes() == 0U || layout.dofs() != dense.n) {
        throw std::invalid_argument("block-Jacobi layout/operator mismatch");
    }

    BlockJacobiMetric metric;
    metric.dof_offsets = layout.dof_offsets;
    metric.value_offsets.resize(layout.nodes() + 1U, 0U);
    metric.min_rank = std::numeric_limits<std::size_t>::max();
    metric.max_rank = 0U;
    metric.min_cholesky_pivot = std::numeric_limits<double>::infinity();

    for (std::size_t a = 0; a < layout.nodes(); ++a) {
        const std::size_t rank = layout.dof_offsets[a + 1U] - layout.dof_offsets[a];
        if (rank == 0U || rank > kCandidates) {
            throw std::runtime_error("block-Jacobi algebraic-node rank outside [1,6]");
        }
        metric.min_rank = std::min(metric.min_rank, rank);
        metric.max_rank = std::max(metric.max_rank, rank);
        metric.value_offsets[a + 1U] = metric.value_offsets[a] + rank * rank;
    }
    metric.lower.assign(metric.value_offsets.back(), 0.0);

    for (std::size_t a = 0; a < layout.nodes(); ++a) {
        const std::size_t begin = layout.dof_offsets[a];
        const std::size_t rank = layout.dof_offsets[a + 1U] - begin;
        double* l = metric.lower.data() + metric.value_offsets[a];
        for (std::size_t i = 0; i < rank; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double value = dense.values[(begin + i) * dense.n + (begin + j)];
                for (std::size_t k = 0; k < j; ++k) {
                    value -= l[i * rank + k] * l[j * rank + k];
                }
                if (i == j) {
                    if (!(value > 0.0) || !std::isfinite(value)) {
                        throw std::runtime_error("actual-A2 diagonal block lost SPD");
                    }
                    metric.min_cholesky_pivot = std::min(metric.min_cholesky_pivot, value);
                    l[i * rank + j] = std::sqrt(value);
                } else {
                    l[i * rank + j] = value / l[j * rank + j];
                }
            }
        }
    }
    return metric;
}

double estimate_lambda_max_block_metric(
    const Apply& apply,
    const BlockJacobiMetric& metric,
    std::size_t power_iterations) {
    if (power_iterations == 0U) {
        throw std::invalid_argument("block-Jacobi power iterations must be positive");
    }
    Vec q = deterministic_actual_a2_probe(metric.dofs(), 0.61);
    double qnorm2 = metric.norm2(q);
    if (!(qnorm2 > 0.0) || !std::isfinite(qnorm2)) {
        throw std::runtime_error("block-Jacobi initial M-norm invalid");
    }
    for (double& v : q) v /= std::sqrt(qnorm2);

    double lambda = 0.0;
    for (std::size_t iteration = 0; iteration < power_iterations; ++iteration) {
        const auto aq = apply(q);
        const double rayleigh = dot(q, aq); // q^T M q == 1 by normalization.
        if (!(rayleigh > 0.0) || !std::isfinite(rayleigh)) {
            throw std::runtime_error("block-Jacobi generalized Rayleigh quotient invalid");
        }
        lambda = std::max(lambda, rayleigh);
        auto next = metric.solve(aq);
        const double next_norm2 = metric.norm2(next);
        if (!(next_norm2 > 0.0) || !std::isfinite(next_norm2)) {
            throw std::runtime_error("block-Jacobi power iteration M-norm invalid");
        }
        const double inv_norm = 1.0 / std::sqrt(next_norm2);
        for (double& v : next) v *= inv_norm;
        q = std::move(next);
    }
    return kLambdaSafety * lambda;
}

void chebyshev_block_jacobi_smooth(
    const Apply& apply,
    const BlockJacobiMetric& metric,
    double lambda_max,
    const Vec& b,
    Vec& x,
    std::size_t degree) {
    if (degree == 0U) return;
    if (b.size() != x.size() || b.size() != metric.dofs()) {
        throw std::invalid_argument("block-Jacobi Chebyshev size mismatch");
    }
    const double lambda_low = kChebyshevLowerFraction * lambda_max;
    const double theta = 0.5 * (lambda_max + lambda_low);
    const double delta = 0.5 * (lambda_max - lambda_low);
    for (std::size_t k = 0; k < degree; ++k) {
        const double angle = kPi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        const double root = theta + delta * std::cos(angle);
        if (!(root > 0.0)) {
            throw std::runtime_error("block-Jacobi Chebyshev root invalid");
        }
        const auto ax = apply(x);
        Vec residual(b.size(), 0.0);
        for (std::size_t i = 0; i < b.size(); ++i) residual[i] = b[i] - ax[i];
        const auto correction = metric.solve(residual);
        const double weight = 1.0 / root;
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += weight * correction[i];
    }
}

struct BlockSmoothedTransfer {
    const CandidateTransfer& tentative;
    const Apply& apply_fine;
    const BlockJacobiMetric& metric;
    double omega{0.0};
    std::size_t steps{0};

    Vec prolong(const Vec& coarse) const {
        auto fine = tentative.prolong(coarse);
        for (std::size_t step = 0; step < steps; ++step) {
            const auto af = apply_fine(fine);
            const auto scaled = metric.solve(af);
            for (std::size_t i = 0; i < fine.size(); ++i) {
                fine[i] -= omega * scaled[i];
            }
        }
        return fine;
    }

    Vec restrict_transpose(const Vec& fine) const {
        if (fine.size() != tentative.fine_dofs) {
            throw std::invalid_argument("block-smoothed transfer restriction size mismatch");
        }
        auto work = fine;
        for (std::size_t step = 0; step < steps; ++step) {
            const auto scaled = metric.solve(work);
            const auto a_scaled = apply_fine(scaled);
            for (std::size_t i = 0; i < work.size(); ++i) {
                work[i] -= omega * a_scaled[i];
            }
        }
        return tentative.restrict_transpose(work);
    }
};

double block_transfer_adjoint_error(const BlockSmoothedTransfer& transfer) {
    const auto coarse = deterministic_actual_a2_probe(transfer.tentative.coarse_dofs, 0.31);
    const auto fine = deterministic_actual_a2_probe(transfer.tentative.fine_dofs, 0.83);
    const auto pc = transfer.prolong(coarse);
    const auto ptf = transfer.restrict_transpose(fine);
    const double lhs = dot(pc, fine);
    const double rhs = dot(coarse, ptf);
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) / scale;
}

double block_smoothed_candidate_reproduction_error(
    const BlockSmoothedTransfer& transfer,
    const std::vector<double>& fine_candidates) {
    if (fine_candidates.size() != transfer.tentative.fine_dofs * kCandidates) {
        throw std::invalid_argument("block-smoothed candidate diagnostic size mismatch");
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

void run_block_jacobi_reference(std::size_t fine_transfer_steps,
                                std::size_t level1_transfer_steps,
                                std::size_t level2_transfer_steps,
                                std::size_t max_cycles,
                                double strength_threshold,
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

    const std::vector<double> inverse1 = tentative_a1.inverse_diagonal;
    if (inverse1.size() != graph1.dofs()) {
        throw std::runtime_error("block-Jacobi reference L1 diagonal size mismatch");
    }
    const double lambda1 = estimate_lambda_max(apply1, inverse1, 8U);
    const double omega1 = kSaDampingNumerator / lambda1;
    const AlgebraicSmoothedTransferActualA2 transfer1{
        tentative_transfer1, apply1, inverse1, omega1, level1_transfer_steps};
    const double transfer1_adjoint = smoothed_transfer_adjoint_error_actual_a2(transfer1);
    const double smoothed_candidate1_error =
        smoothed_candidate_reproduction_error_actual_a2(transfer1, candidates1);

    const Apply apply2_nested = [&](const Vec& x) {
        const auto l1 = transfer1.prolong(x);
        return transfer1.restrict_transpose(apply1(l1));
    };
    const auto actual_a2 = materialize_dense_symmetric_operator(
        apply2_nested, tentative_transfer1.coarse_dofs);
    const Apply apply2 = [&](const Vec& x) { return actual_a2.apply(x); };
    const double actual_a2_oracle_error = dense_vs_nested_relative_error(actual_a2, apply2_nested);

    StrengthGraphStats graph2_stats;
    const auto actual_graph2 = build_strength_graph_from_dense(
        actual_a2,
        tentative_transfer1.coarse_graph,
        strength_threshold,
        graph2_stats);

    const auto block_metric = build_block_jacobi_metric(actual_a2, actual_graph2);
    const double scalar_lambda2 = estimate_lambda_max(apply2, actual_a2.inverse_diagonal, 8U);
    const double block_lambda2 = estimate_lambda_max_block_metric(apply2, block_metric, 8U);
    const double block_omega2 = kSaDampingNumerator / block_lambda2;

    const auto tentative_transfer2 = build_candidate_transfer(
        actual_graph2,
        tentative_transfer1.coarse_candidates,
        target_nodes,
        min_nodes,
        1.0e-10);
    const double tentative_candidate2_error = candidate_reproduction_error(
        tentative_transfer2, tentative_transfer1.coarse_candidates);
    const BlockSmoothedTransfer transfer2{
        tentative_transfer2,
        apply2,
        block_metric,
        block_omega2,
        level2_transfer_steps};
    const double transfer2_adjoint = block_transfer_adjoint_error(transfer2);
    const double smoothed_candidate2_error = block_smoothed_candidate_reproduction_error(
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

    levels[2].dofs = actual_a2.n;
    levels[2].label = "L2_actual_A2_exact_block_Jacobi";
    levels[2].diagnostic_lambda_max = block_lambda2;
    levels[2].apply = apply2;
    levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_block_jacobi_smooth(apply2, block_metric, block_lambda2, b, x, degree);
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
              << "L2_operator_source=actual_smoothed_Galerkin_materialized_FP64\n"
              << "L2_relaxation=exact_algebraic_node_block_Jacobi\n"
              << "L2_transfer_smoothing_metric=exact_algebraic_node_block_Jacobi\n"
              << "L2_graph_source=actual_A2_block_strength\n"
              << "coarse_mesh_required_after_L0=false\n"
              << "L0_dofs=" << levels[0].dofs
              << " L1_dofs=" << levels[1].dofs
              << " L1_nodes=" << graph1.nodes()
              << " L2_dofs=" << levels[2].dofs
              << " L2_nodes=" << actual_graph2.nodes()
              << " L3_dofs=" << levels[3].dofs
              << " L3_nodes=" << tentative_transfer2.coarse_graph.nodes() << '\n'
              << std::scientific << std::setprecision(9)
              << "L1_tentative_candidate_reproduction_error=" << tentative_candidate1_error
              << " L1_smoothed_candidate_reproduction_error=" << smoothed_candidate1_error
              << " L1_transfer_adjoint_relative_error=" << transfer1_adjoint << '\n'
              << "L2_tentative_candidate_reproduction_error=" << tentative_candidate2_error
              << " L2_block_smoothed_candidate_reproduction_error=" << smoothed_candidate2_error
              << " L2_transfer_adjoint_relative_error=" << transfer2_adjoint << '\n'
              << "actual_A2_symmetry_relative_defect=" << actual_a2.symmetry_relative_defect
              << " actual_A2_dense_vs_nested_relative_error=" << actual_a2_oracle_error << '\n'
              << std::fixed << std::setprecision(6)
              << "actual_A2_dense_bytes=" << actual_a2.values.size() * sizeof(double)
              << " actual_A2_materialize_ms=" << actual_a2.materialize_ms
              << " actual_A2_diag_min=" << actual_a2.min_diagonal
              << " actual_A2_diag_max=" << actual_a2.max_diagonal << '\n'
              << "strength_threshold=" << graph2_stats.threshold
              << " strength_directed_edges=" << graph2_stats.directed_edges
              << " strength_forced_pairs=" << graph2_stats.forced_pairs
              << " strength_degree_min=" << graph2_stats.min_degree
              << " strength_degree_avg=" << graph2_stats.average_degree
              << " strength_degree_max=" << graph2_stats.max_degree << '\n'
              << "block_nodes=" << block_metric.nodes()
              << " block_rank_min=" << block_metric.min_rank
              << " block_rank_max=" << block_metric.max_rank
              << " block_metric_storage_bytes=" << block_metric.storage_bytes()
              << " block_min_cholesky_pivot=" << block_metric.min_cholesky_pivot << '\n'
              << "lambda0=" << lambda0 << " omega0=" << omega0
              << " lambda1=" << lambda1 << " omega1=" << omega1
              << " scalar_lambda2_reference=" << scalar_lambda2
              << " block_lambda2=" << block_lambda2
              << " block_omega2=" << block_omega2 << '\n'
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
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 2U;
        const std::size_t m2 = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 1U;
        const std::size_t max_cycles = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 6U;
        const double strength_threshold = argc > 5 ? std::stod(argv[5]) : 0.05;
        const std::size_t target_nodes = argc > 6
            ? static_cast<std::size_t>(std::stoull(argv[6])) : 12U;
        const std::size_t min_nodes = argc > 7
            ? static_cast<std::size_t>(std::stoull(argv[7])) : 4U;
        if (m0 > 4U || m1 > 4U || m2 > 4U || max_cycles == 0U ||
            !(strength_threshold > 0.0) || !(strength_threshold < 1.0) ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid block-Jacobi recursive SA reference options");
        }

        std::cout << "GFSS M5 actual-A2 block-Jacobi recursive SA numerical reference\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "purpose=isolate_exact_L2_algebraic_node_block_Jacobi_effect\n"
                  << "reference_execution=cpu_fp64\n"
                  << "L0_operator=matrix_free_exact_FEM\n"
                  << "L2_operator=explicit_actual_smoothed_Galerkin_reference_only\n"
                  << "L2_metric=exact_diagonal_blocks_of_actual_A2\n"
                  << "L2_graph=normalized_block_Frobenius_strength\n"
                  << "all_transfer_restrictions=exact_transpose\n"
                  << "L3_bottom=dense_materialized_Cholesky\n"
                  << "pre_smooth_degree=3 post_smooth_degree=3\n"
                  << "acceptance_target_post_transient_q<=0.4\n"
                  << "performance_status=numerical_reference_only\n";

        run_block_jacobi_reference(
            m0, m1, m2, max_cycles,
            strength_threshold, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_block_jacobi_reference_bench "
                  << "[m0 [m1 [m2 [max_cycles [strength_threshold [target_nodes [min_nodes]]]]]]]\n";
        return 1;
    }
}
