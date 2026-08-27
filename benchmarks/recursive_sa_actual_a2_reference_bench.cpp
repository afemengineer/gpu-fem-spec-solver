// Numerical-reference experiment for the next recursive-SA question:
// materialize the actual smoothed L2 Galerkin operator, use its true Jacobi
// diagonal and a strength graph derived from its block couplings, then rebuild
// and smooth the L2->L3 transfer. The raw recursive reference is included only
// to reuse its graph/candidate/multilevel helpers inside this standalone TU.
#define main gfss_recursive_sa_raw_reference_main
#include "recursive_sa_reference_bench.cpp"
#undef main

namespace {

struct AlgebraicSmoothedTransferActualA2 {
    const CandidateTransfer& tentative;
    const Apply& apply_fine;
    const std::vector<double>& inverse_diagonal;
    double omega{0.0};
    std::size_t steps{0};

    Vec prolong(const Vec& coarse) const {
        auto fine = tentative.prolong(coarse);
        if (inverse_diagonal.size() != fine.size()) {
            throw std::invalid_argument("actual-A2 SA prolongation diagonal size mismatch");
        }
        for (std::size_t step = 0; step < steps; ++step) {
            const auto af = apply_fine(fine);
            if (af.size() != fine.size()) {
                throw std::runtime_error("actual-A2 SA forward operator size mismatch");
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
            throw std::invalid_argument("actual-A2 SA restriction size mismatch");
        }
        auto work = fine;
        Vec scaled(fine.size(), 0.0);
        for (std::size_t step = 0; step < steps; ++step) {
            for (std::size_t i = 0; i < work.size(); ++i) {
                scaled[i] = inverse_diagonal[i] * work[i];
            }
            const auto a_scaled = apply_fine(scaled);
            if (a_scaled.size() != work.size()) {
                throw std::runtime_error("actual-A2 SA transpose operator size mismatch");
            }
            for (std::size_t i = 0; i < work.size(); ++i) {
                work[i] -= omega * a_scaled[i];
            }
        }
        return tentative.restrict_transpose(work);
    }
};

struct DenseOperatorReference {
    std::size_t n{0};
    std::vector<double> values; // row-major symmetric matrix
    std::vector<double> inverse_diagonal;
    double symmetry_relative_defect{0.0};
    double min_diagonal{0.0};
    double max_diagonal{0.0};
    double materialize_ms{0.0};

    Vec apply(const Vec& x) const {
        if (x.size() != n) {
            throw std::invalid_argument("actual-A2 dense matvec size mismatch");
        }
        Vec y(n, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(n); ++i64) {
            const auto i = static_cast<std::size_t>(i64);
            const double* row = values.data() + i * n;
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j) sum += row[j] * x[j];
            y[i] = sum;
        }
        return y;
    }
};

struct StrengthGraphStats {
    double threshold{0.0};
    std::size_t directed_edges{0};
    std::size_t forced_pairs{0};
    std::size_t min_degree{0};
    double average_degree{0.0};
    std::size_t max_degree{0};
};

Vec deterministic_actual_a2_probe(std::size_t n, double phase) {
    Vec v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = std::sin(0.019 * t + phase) + 0.27 * std::cos(0.041 * t - 0.37 * phase);
    }
    return v;
}

double smoothed_transfer_adjoint_error_actual_a2(
    const AlgebraicSmoothedTransferActualA2& transfer) {
    const auto coarse = deterministic_actual_a2_probe(transfer.tentative.coarse_dofs, 0.23);
    const auto fine = deterministic_actual_a2_probe(transfer.tentative.fine_dofs, 0.79);
    const auto pc = transfer.prolong(coarse);
    const auto ptf = transfer.restrict_transpose(fine);
    const double lhs = dot(pc, fine);
    const double rhs = dot(coarse, ptf);
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) / scale;
}

double smoothed_candidate_reproduction_error_actual_a2(
    const AlgebraicSmoothedTransferActualA2& transfer,
    const std::vector<double>& fine_candidates) {
    if (fine_candidates.size() != transfer.tentative.fine_dofs * kCandidates) {
        throw std::invalid_argument("actual-A2 candidate diagnostic size mismatch");
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

DenseOperatorReference materialize_dense_symmetric_operator(
    const Apply& apply,
    std::size_t n) {
    if (n == 0U) throw std::invalid_argument("actual-A2 operator has zero size");
    const auto start = Clock::now();

    DenseOperatorReference dense;
    dense.n = n;
    dense.values.assign(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        Vec e(n, 0.0);
        e[j] = 1.0;
        const auto col = apply(e);
        if (col.size() != n) {
            throw std::runtime_error("actual-A2 materialization column size mismatch");
        }
        for (std::size_t i = 0; i < n; ++i) dense.values[i * n + j] = col[i];
    }

    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        norm2 += dense.values[i * n + i] * dense.values[i * n + i];
        for (std::size_t j = i + 1U; j < n; ++j) {
            const double aij = dense.values[i * n + j];
            const double aji = dense.values[j * n + i];
            const double d = aij - aji;
            asym2 += 2.0 * d * d;
            norm2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            dense.values[i * n + j] = sym;
            dense.values[j * n + i] = sym;
        }
    }
    dense.symmetry_relative_defect = norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;

    dense.inverse_diagonal.resize(n, 0.0);
    dense.min_diagonal = std::numeric_limits<double>::infinity();
    dense.max_diagonal = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = dense.values[i * n + i];
        if (!(d > 0.0) || !std::isfinite(d)) {
            throw std::runtime_error("actual-A2 true diagonal is not positive finite");
        }
        dense.inverse_diagonal[i] = 1.0 / d;
        dense.min_diagonal = std::min(dense.min_diagonal, d);
        dense.max_diagonal = std::max(dense.max_diagonal, d);
    }
    dense.materialize_ms = elapsed_ms(start, Clock::now());
    return dense;
}

double dense_vs_nested_relative_error(const DenseOperatorReference& dense,
                                      const Apply& nested) {
    auto probe = deterministic_actual_a2_probe(dense.n, 0.47);
    const auto yd = dense.apply(probe);
    const auto yn = nested(probe);
    Vec diff(yd.size(), 0.0);
    for (std::size_t i = 0; i < yd.size(); ++i) diff[i] = yd[i] - yn[i];
    const double denom = norm(yn);
    if (!(denom > 0.0)) throw std::runtime_error("actual-A2 oracle norm is zero");
    return norm(diff) / denom;
}

AlgebraicNodeGraph build_strength_graph_from_dense(
    const DenseOperatorReference& dense,
    const AlgebraicNodeGraph& layout,
    double threshold,
    StrengthGraphStats& stats) {
    if (!(threshold > 0.0) || !(threshold < 1.0)) {
        throw std::invalid_argument("actual-A2 strength threshold must be in (0,1)");
    }
    if (layout.nodes() == 0U || layout.dofs() != dense.n ||
        layout.dof_offsets.size() != layout.nodes() + 1U) {
        throw std::invalid_argument("actual-A2 strength graph layout mismatch");
    }

    const std::size_t nodes = layout.nodes();
    std::vector<double> diag_block_norm(nodes, 0.0);
    for (std::size_t a = 0; a < nodes; ++a) {
        double sum2 = 0.0;
        for (std::size_t i = layout.dof_offsets[a]; i < layout.dof_offsets[a + 1U]; ++i) {
            for (std::size_t j = layout.dof_offsets[a]; j < layout.dof_offsets[a + 1U]; ++j) {
                const double v = dense.values[i * dense.n + j];
                sum2 += v * v;
            }
        }
        diag_block_norm[a] = std::sqrt(sum2);
        if (!(diag_block_norm[a] > 0.0) || !std::isfinite(diag_block_norm[a])) {
            throw std::runtime_error("actual-A2 diagonal block norm is invalid");
        }
    }

    std::vector<std::vector<std::uint32_t>> neighbors(nodes);
    std::vector<double> strongest(nodes, 0.0);
    std::vector<std::uint32_t> strongest_neighbor(nodes, kUnassigned);

    for (std::size_t a = 0; a < nodes; ++a) {
        for (std::size_t b = a + 1U; b < nodes; ++b) {
            double sum2 = 0.0;
            for (std::size_t i = layout.dof_offsets[a]; i < layout.dof_offsets[a + 1U]; ++i) {
                for (std::size_t j = layout.dof_offsets[b]; j < layout.dof_offsets[b + 1U]; ++j) {
                    const double v = dense.values[i * dense.n + j];
                    sum2 += v * v;
                }
            }
            const double block_norm = std::sqrt(sum2);
            const double scale = std::sqrt(diag_block_norm[a] * diag_block_norm[b]);
            const double strength = scale > 0.0 ? block_norm / scale : 0.0;
            if (strength > strongest[a]) {
                strongest[a] = strength;
                strongest_neighbor[a] = static_cast<std::uint32_t>(b);
            }
            if (strength > strongest[b]) {
                strongest[b] = strength;
                strongest_neighbor[b] = static_cast<std::uint32_t>(a);
            }
            if (strength >= threshold) {
                neighbors[a].push_back(static_cast<std::uint32_t>(b));
                neighbors[b].push_back(static_cast<std::uint32_t>(a));
            }
        }
    }

    std::size_t forced_pairs = 0U;
    for (std::size_t a = 0; a < nodes; ++a) {
        if (!neighbors[a].empty()) continue;
        const auto b = strongest_neighbor[a];
        if (b == kUnassigned || !(strongest[a] > 0.0)) {
            throw std::runtime_error("actual-A2 strength graph produced isolated zero-coupling node");
        }
        neighbors[a].push_back(b);
        neighbors[b].push_back(static_cast<std::uint32_t>(a));
        ++forced_pairs;
    }

    AlgebraicNodeGraph graph;
    graph.dof_offsets = layout.dof_offsets;
    graph.row_offsets.resize(nodes + 1U, 0U);
    std::size_t cursor = 0U;
    std::size_t min_degree = std::numeric_limits<std::size_t>::max();
    std::size_t max_degree = 0U;
    for (std::size_t a = 0; a < nodes; ++a) {
        auto& row = neighbors[a];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        row.erase(std::remove(row.begin(), row.end(), static_cast<std::uint32_t>(a)), row.end());
        min_degree = std::min(min_degree, row.size());
        max_degree = std::max(max_degree, row.size());
        graph.row_offsets[a] = static_cast<std::uint32_t>(cursor);
        graph.column_indices.insert(graph.column_indices.end(), row.begin(), row.end());
        cursor += row.size();
    }
    graph.row_offsets[nodes] = static_cast<std::uint32_t>(cursor);

    stats.threshold = threshold;
    stats.directed_edges = cursor;
    stats.forced_pairs = forced_pairs;
    stats.min_degree = nodes > 0U ? min_degree : 0U;
    stats.average_degree = nodes > 0U
        ? static_cast<double>(cursor) / static_cast<double>(nodes) : 0.0;
    stats.max_degree = max_degree;
    return graph;
}

void run_actual_a2_reference(std::size_t fine_transfer_steps,
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
        throw std::runtime_error("actual-A2 L1 diagonal size mismatch");
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

    // Numerical-reference-only: explicitly materialize the *actual* smoothed
    // Galerkin A2 so the L2 Jacobi metric and graph are derived from the same
    // operator used by the V-cycle.
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

    const double lambda2 = estimate_lambda_max(apply2, actual_a2.inverse_diagonal, 8U);
    const double omega2 = kSaDampingNumerator / lambda2;

    const auto tentative_transfer2 = build_candidate_transfer(
        actual_graph2,
        tentative_transfer1.coarse_candidates,
        target_nodes,
        min_nodes,
        1.0e-10);
    const double tentative_candidate2_error = candidate_reproduction_error(
        tentative_transfer2, tentative_transfer1.coarse_candidates);
    const AlgebraicSmoothedTransferActualA2 transfer2{
        tentative_transfer2,
        apply2,
        actual_a2.inverse_diagonal,
        omega2,
        level2_transfer_steps};
    const double transfer2_adjoint = smoothed_transfer_adjoint_error_actual_a2(transfer2);
    const double smoothed_candidate2_error =
        smoothed_candidate_reproduction_error_actual_a2(
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
    levels[2].label = "L2_actual_smoothed_Galerkin_dense_reference";
    levels[2].diagnostic_lambda_max = lambda2;
    levels[2].apply = apply2;
    levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_smooth(apply2, actual_a2.inverse_diagonal, lambda2, b, x, degree);
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
              << "L2_jacobi_source=true_diag_of_actual_A2\n"
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
              << " L2_smoothed_candidate_reproduction_error=" << smoothed_candidate2_error
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
        const double strength_threshold = argc > 5 ? std::stod(argv[5]) : 0.05;
        const std::size_t target_nodes = argc > 6
            ? static_cast<std::size_t>(std::stoull(argv[6])) : 12U;
        const std::size_t min_nodes = argc > 7
            ? static_cast<std::size_t>(std::stoull(argv[7])) : 4U;
        if (m0 > 4U || m1 > 4U || m2 > 4U || max_cycles == 0U ||
            !(strength_threshold > 0.0) || !(strength_threshold < 1.0) ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid actual-A2 recursive SA reference options");
        }

        std::cout << "GFSS M5 actual-A2 recursively smoothed-aggregation numerical reference\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "purpose=replace_stale_L2_metric_and_graph_with_actual_smoothed_Galerkin_A2\n"
                  << "reference_execution=cpu_fp64\n"
                  << "L0_operator=matrix_free_exact_FEM\n"
                  << "L2_operator=explicit_actual_smoothed_Galerkin_reference_only\n"
                  << "L2_jacobi=true_actual_A2_diagonal\n"
                  << "L2_graph=normalized_block_Frobenius_strength\n"
                  << "all_transfer_restrictions=exact_transpose\n"
                  << "L3_bottom=dense_materialized_Cholesky\n"
                  << "pre_smooth_degree=3 post_smooth_degree=3\n"
                  << "acceptance_target_post_transient_q<=0.4\n"
                  << "performance_status=numerical_reference_only\n";

        run_actual_a2_reference(
            m0, m1, m2, max_cycles,
            strength_threshold, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_actual_a2_reference_bench "
                  << "[m0 [m1 [m2 [max_cycles [strength_threshold [target_nodes [min_nodes]]]]]]]\n";
        return 1;
    }
}
