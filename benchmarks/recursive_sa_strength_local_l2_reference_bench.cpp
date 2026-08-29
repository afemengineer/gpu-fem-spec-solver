// Combined M5 numerical-reference consistency experiment.
//
// Hold the winning actual-A1 energy-strength graph at theta=0.05 fixed and
// combine it with the validated localized exact actual-L2 block metric and
// localized/direct bottom assembly. This intentionally composes two already
// audited ingredients while keeping dense A1/A2 materialization out of the
// hierarchy. The experiment is CPU/FP64 reference-only; production remains
// matrix-light and factorized.
#define main gfss_recursive_sa_local_l2_combined_helpers_only
#include "recursive_sa_local_l2_reference_bench.cpp"
#undef main

#include <unordered_set>

namespace {

using CombinedBlock = std::array<double, 36>; // <=6 x <=6, stride 6

std::uint64_t combined_pair_key(std::uint32_t a, std::uint32_t b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32U) |
           static_cast<std::uint64_t>(b);
}

std::pair<std::uint32_t, std::uint32_t> combined_decode_pair(std::uint64_t key) {
    return {static_cast<std::uint32_t>(key >> 32U),
            static_cast<std::uint32_t>(key & 0xffffffffULL)};
}

std::unordered_map<std::uint64_t, CombinedBlock>
accumulate_combined_actual_a1_offdiagonal_blocks(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const std::vector<FineBasisSupport>& supports,
    const std::vector<std::vector<std::uint32_t>>& element_supports) {
    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    std::unordered_map<std::uint64_t, CombinedBlock> blocks;
    blocks.reserve(supports.size() * 24U);

    struct ElementBasis {
        std::uint32_t aggregate{0};
        std::size_t rank{0};
        std::array<double, 24U * kCandidates> u{};
        std::array<double, 24U * kCandidates> ku{};
    };

    for (std::size_t element = 0; element < element_supports.size(); ++element) {
        const auto& active = element_supports[element];
        if (active.size() < 2U) continue;
        const auto ijk = decode_element(mesh, element);
        const auto nodes = mesh.element_nodes(ijk[0], ijk[1], ijk[2]);

        std::vector<ElementBasis> local;
        local.reserve(active.size());
        for (const auto aggregate : active) {
            ElementBasis basis;
            basis.aggregate = aggregate;
            basis.rank = supports[aggregate].rank;
            for (std::size_t n = 0; n < 8U; ++n) {
                const auto it = supports[aggregate].values.find(
                    static_cast<std::size_t>(nodes[n]));
                if (it == supports[aggregate].values.end()) continue;
                for (std::size_t c = 0; c < 3U; ++c) {
                    const std::size_t ldof = 3U * n + c;
                    for (std::size_t q = 0; q < basis.rank; ++q) {
                        basis.u[ldof * kCandidates + q] =
                            it->second[c * kCandidates + q];
                    }
                }
            }
            for (std::size_t i = 0; i < 24U; ++i) {
                for (std::size_t q = 0; q < basis.rank; ++q) {
                    double value = 0.0;
                    for (std::size_t j = 0; j < 24U; ++j) {
                        value += ke[i][j] * basis.u[j * kCandidates + q];
                    }
                    basis.ku[i * kCandidates + q] = value;
                }
            }
            local.push_back(std::move(basis));
        }

        for (std::size_t ia = 0; ia < local.size(); ++ia) {
            for (std::size_t ib = ia + 1U; ib < local.size(); ++ib) {
                const ElementBasis* left = &local[ia];
                const ElementBasis* right = &local[ib];
                if (left->aggregate > right->aggregate) std::swap(left, right);
                const auto key = combined_pair_key(left->aggregate, right->aggregate);
                auto [it, inserted] = blocks.try_emplace(key, CombinedBlock{});
                auto& block = it->second;
                for (std::size_t q = 0; q < left->rank; ++q) {
                    for (std::size_t r = 0; r < right->rank; ++r) {
                        double value = 0.0;
                        for (std::size_t i = 0; i < 24U; ++i) {
                            value += left->u[i * kCandidates + q] *
                                     right->ku[i * kCandidates + r];
                        }
                        block[q * kCandidates + r] += value;
                    }
                }
            }
        }
    }
    return blocks;
}

std::unordered_set<std::uint64_t> combined_pairs_from_graph(
    const AlgebraicNodeGraph& graph) {
    std::unordered_set<std::uint64_t> pairs;
    pairs.reserve(graph.column_indices.size());
    for (std::uint32_t a = 0; a < graph.nodes(); ++a) {
        for (std::uint32_t p = graph.row_offsets[a]; p < graph.row_offsets[a + 1U]; ++p) {
            const auto b = graph.column_indices[p];
            if (a != b) pairs.insert(combined_pair_key(a, b));
        }
    }
    return pairs;
}

double combined_sampled_cross_block_oracle_error(
    const std::unordered_map<std::uint64_t, CombinedBlock>& blocks,
    const AlgebraicNodeGraph& layout,
    const Apply& apply1) {
    if (blocks.empty()) return 0.0;
    std::vector<std::uint64_t> keys;
    keys.reserve(blocks.size());
    for (const auto& entry : blocks) keys.push_back(entry.first);
    std::sort(keys.begin(), keys.end());

    std::vector<std::size_t> samples{
        0U,
        keys.size() / 4U,
        keys.size() / 2U,
        (3U * keys.size()) / 4U,
        keys.size() - 1U,
    };
    sort_unique(samples);

    double diff2 = 0.0;
    double ref2 = 0.0;
    for (const auto sample : samples) {
        const auto [a, b] = combined_decode_pair(keys[sample]);
        const auto& block = blocks.at(keys[sample]);
        const std::size_t a_begin = layout.dof_offsets[a];
        const std::size_t b_begin = layout.dof_offsets[b];
        const std::size_t a_rank = layout.dof_offsets[a + 1U] - a_begin;
        const std::size_t b_rank = layout.dof_offsets[b + 1U] - b_begin;
        for (std::size_t r = 0; r < b_rank; ++r) {
            Vec e(layout.dofs(), 0.0);
            e[b_begin + r] = 1.0;
            const auto y = apply1(e);
            for (std::size_t q = 0; q < a_rank; ++q) {
                const double local = block[q * kCandidates + r];
                const double oracle = y[a_begin + q];
                const double d = local - oracle;
                diff2 += d * d;
                ref2 += oracle * oracle;
            }
        }
    }
    return ref2 > 0.0 ? std::sqrt(diff2 / ref2) : 0.0;
}

struct CombinedStrengthResult {
    AlgebraicNodeGraph graph;
    StrengthGraphStats stats;
    std::size_t candidate_undirected_pairs{0};
    std::size_t tentative_undirected_pairs{0};
    std::size_t shared_strong_tentative_pairs{0};
    double setup_ms{0.0};
};

CombinedStrengthResult build_combined_strength_graph(
    const AlgebraicNodeGraph& tentative_graph,
    const L1BlockMetric& block1,
    const std::unordered_map<std::uint64_t, CombinedBlock>& blocks,
    double threshold) {
    if (!(threshold > 0.0) || !(threshold < 1.0)) {
        throw std::invalid_argument("combined actual-A1 strength threshold must be in (0,1)");
    }
    const auto start = Clock::now();
    CombinedStrengthResult result;
    result.candidate_undirected_pairs = blocks.size();

    const std::size_t nodes = tentative_graph.nodes();
    std::vector<double> diag_norm(nodes, 0.0);
    for (std::size_t a = 0; a < nodes; ++a) {
        const std::size_t rank = tentative_graph.dof_offsets[a + 1U] -
                                 tentative_graph.dof_offsets[a];
        double sum2 = 0.0;
        for (std::size_t i = 0; i < rank; ++i) {
            for (std::size_t j = 0; j < rank; ++j) {
                const double v = block1.block_entry(a, i, j);
                sum2 += v * v;
            }
        }
        diag_norm[a] = std::sqrt(sum2);
        if (!(diag_norm[a] > 0.0) || !std::isfinite(diag_norm[a])) {
            throw std::runtime_error("combined actual-A1 diagonal block norm invalid");
        }
    }

    std::vector<std::vector<std::uint32_t>> neighbors(nodes);
    std::vector<double> strongest(nodes, 0.0);
    std::vector<std::uint32_t> strongest_neighbor(nodes, kUnassigned);
    std::unordered_set<std::uint64_t> strong_pairs;
    strong_pairs.reserve(blocks.size());

    for (const auto& entry : blocks) {
        const auto [a, b] = combined_decode_pair(entry.first);
        const std::size_t a_rank = tentative_graph.dof_offsets[a + 1U] -
                                   tentative_graph.dof_offsets[a];
        const std::size_t b_rank = tentative_graph.dof_offsets[b + 1U] -
                                   tentative_graph.dof_offsets[b];
        double sum2 = 0.0;
        for (std::size_t i = 0; i < a_rank; ++i) {
            for (std::size_t j = 0; j < b_rank; ++j) {
                const double v = entry.second[i * kCandidates + j];
                sum2 += v * v;
            }
        }
        const double block_norm = std::sqrt(sum2);
        const double scale = std::sqrt(diag_norm[a] * diag_norm[b]);
        const double strength = scale > 0.0 ? block_norm / scale : 0.0;
        if (strength > strongest[a]) {
            strongest[a] = strength;
            strongest_neighbor[a] = b;
        }
        if (strength > strongest[b]) {
            strongest[b] = strength;
            strongest_neighbor[b] = a;
        }
        if (strength >= threshold) {
            neighbors[a].push_back(b);
            neighbors[b].push_back(a);
            strong_pairs.insert(entry.first);
        }
    }

    std::size_t forced_pairs = 0U;
    for (std::size_t a = 0; a < nodes; ++a) {
        if (!neighbors[a].empty()) continue;
        const auto b = strongest_neighbor[a];
        if (b == kUnassigned || !(strongest[a] > 0.0)) {
            throw std::runtime_error("combined strength graph isolated zero-coupling node");
        }
        neighbors[a].push_back(b);
        neighbors[b].push_back(static_cast<std::uint32_t>(a));
        strong_pairs.insert(combined_pair_key(static_cast<std::uint32_t>(a), b));
        ++forced_pairs;
    }

    result.graph.dof_offsets = tentative_graph.dof_offsets;
    result.graph.row_offsets.resize(nodes + 1U, 0U);
    std::size_t cursor = 0U;
    std::size_t min_degree = std::numeric_limits<std::size_t>::max();
    std::size_t max_degree = 0U;
    for (std::size_t a = 0; a < nodes; ++a) {
        auto& row = neighbors[a];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        row.erase(std::remove(row.begin(), row.end(), static_cast<std::uint32_t>(a)), row.end());
        result.graph.row_offsets[a] = static_cast<std::uint32_t>(cursor);
        result.graph.column_indices.insert(result.graph.column_indices.end(), row.begin(), row.end());
        cursor += row.size();
        min_degree = std::min(min_degree, row.size());
        max_degree = std::max(max_degree, row.size());
    }
    result.graph.row_offsets[nodes] = static_cast<std::uint32_t>(cursor);

    result.stats.threshold = threshold;
    result.stats.directed_edges = cursor;
    result.stats.forced_pairs = forced_pairs;
    result.stats.min_degree = nodes > 0U ? min_degree : 0U;
    result.stats.average_degree = nodes > 0U
        ? static_cast<double>(cursor) / static_cast<double>(nodes) : 0.0;
    result.stats.max_degree = max_degree;

    const auto tentative_pairs = combined_pairs_from_graph(tentative_graph);
    result.tentative_undirected_pairs = tentative_pairs.size();
    for (const auto key : strong_pairs) {
        if (tentative_pairs.find(key) != tentative_pairs.end()) {
            ++result.shared_strong_tentative_pairs;
        }
    }
    result.setup_ms = elapsed_ms(start, Clock::now());
    return result;
}

void run_strength_local_l2_reference(std::size_t max_cycles,
                                     std::size_t target_nodes,
                                     std::size_t min_nodes) {
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
        const auto fine = transfer0.prolong(x);
        return transfer0.restrict_transpose(apply0(fine));
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
    const double strength_block_oracle_error = combined_sampled_cross_block_oracle_error(
        actual_a1_offdiagonal, graph1_tentative, apply1);
    const auto strength1 = build_combined_strength_graph(
        graph1_tentative, block1, actual_a1_offdiagonal, strength_threshold);

    const auto transfer1_tentative = build_candidate_transfer(
        strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
    const double candidate1_error = candidate_reproduction_error(
        transfer1_tentative, candidates1);
    const L1BlockSmoothedTransfer transfer1{
        transfer1_tentative, apply1, block1, block_omega1, m1};
    const double transfer1_adjoint = l1_block_transfer_adjoint_error(transfer1);
    const double smoothed_candidate1_error = l1_block_candidate_error(transfer1, candidates1);

    const Apply apply2 = [&](const Vec& x) {
        const auto l1 = transfer1.prolong(x);
        return transfer1.restrict_transpose(apply1(l1));
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
    const auto actual_inverse2 = block2.inverse_scalar_diagonal();
    const double actual_scalar_lambda2 = estimate_lambda_max(apply2, actual_inverse2, 8U);
    const double block_lambda2 = estimate_lambda_max_l1_block(apply2, block2, 8U);
    const double block_omega2 = kSaDampingNumerator / block_lambda2;

    const auto transfer2_tentative = build_candidate_transfer(
        transfer1_tentative.coarse_graph,
        transfer1_tentative.coarse_candidates,
        target_nodes,
        min_nodes,
        1.0e-10);
    const double candidate2_error = candidate_reproduction_error(
        transfer2_tentative, transfer1_tentative.coarse_candidates);
    const L1BlockSmoothedTransfer transfer2{
        transfer2_tentative, apply2, block2, block_omega2, m2};
    const double transfer2_adjoint = l1_block_transfer_adjoint_error(transfer2);
    const double smoothed_candidate2_error = l1_block_candidate_error(
        transfer2, transfer1_tentative.coarse_candidates);

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
        const auto l2 = transfer2.prolong(x);
        return transfer2.restrict_transpose(apply2(l2));
    };
    const double bottom_oracle_error = bottom_local_oracle_error(bottom, apply3_nested);
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
    levels[1].label = "L1_actual_block_metric_actual_strength_graph";
    levels[1].diagnostic_lambda_max = block_lambda1;
    levels[1].apply = apply1;
    levels[1].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply1, block1, block_lambda1, b, x, degree);
    };
    levels[1].restrict_to_coarse = [&](const Vec& r) { return transfer1.restrict_transpose(r); };
    levels[1].prolong_from_coarse = [&](const Vec& c) { return transfer1.prolong(c); };

    levels[2].dofs = transfer1_tentative.coarse_dofs;
    levels[2].label = "L2_local_exact_block_metric_from_strength_hierarchy";
    levels[2].diagnostic_lambda_max = block_lambda2;
    levels[2].apply = apply2;
    levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply2, block2, block_lambda2, b, x, degree);
    };
    levels[2].restrict_to_coarse = [&](const Vec& r) { return transfer2.restrict_transpose(r); };
    levels[2].prolong_from_coarse = [&](const Vec& c) { return transfer2.prolong(c); };

    levels[3].dofs = transfer2_tentative.coarse_dofs;
    levels[3].label = "L3_local_dense_bottom";
    levels[3].apply = [&](const Vec& x) { return apply_dense_bottom(bottom, x); };
    levels[3].bottom_solve = [&](const Vec& b) { return bottom.factor.solve(b); };

    const auto rhs = make_rhs(mesh);
    const auto result = gfss::solve_reference_multilevel_vcycle(
        levels, rhs, 1.0e-6, max_cycles, 3U, 3U);

    const std::size_t tentative_pairs = strength1.tentative_undirected_pairs;
    const std::size_t strong_pairs = strength1.stats.directed_edges / 2U;
    const double overlap_over_strong = strong_pairs > 0U
        ? static_cast<double>(strength1.shared_strong_tentative_pairs) /
          static_cast<double>(strong_pairs) : 0.0;
    const double overlap_over_tentative = tentative_pairs > 0U
        ? static_cast<double>(strength1.shared_strong_tentative_pairs) /
          static_cast<double>(tentative_pairs) : 0.0;

    double post_transient_q = std::numeric_limits<double>::quiet_NaN();
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
            post_transient_q = std::exp(log_sum / static_cast<double>(count));
        }
    }

    std::cout << "\n========================================\n"
              << "experiment=theta_0p05_plus_localized_actual_L2_metric\n"
              << "L0_transfer_smoothing_steps=" << m0
              << " L1_transfer_smoothing_steps=" << m1
              << " L2_transfer_smoothing_steps=" << m2 << '\n'
              << "hierarchy_levels=4\n"
              << "L1_metric=localized_exact_actual_block\n"
              << "L1_graph=localized_actual_smoothed_A1_block_strength\n"
              << "L2_metric=localized_exact_actual_block\n"
              << "bottom_operator=localized_exact_recursive_energy\n"
              << "dense_A1_materialized=false dense_A2_materialized=false\n"
              << "L0_dofs=" << levels[0].dofs
              << " L1_dofs=" << levels[1].dofs
              << " L1_nodes=" << graph1_tentative.nodes()
              << " L2_dofs=" << levels[2].dofs
              << " L2_nodes=" << transfer1_tentative.coarse_graph.nodes()
              << " L3_dofs=" << levels[3].dofs
              << " L3_nodes=" << transfer2_tentative.coarse_graph.nodes() << '\n'
              << std::scientific << std::setprecision(9)
              << "L1_candidate_reproduction_error=" << candidate1_error
              << " L1_block_smoothed_candidate_reproduction_error=" << smoothed_candidate1_error
              << " L1_transfer_adjoint_relative_error=" << transfer1_adjoint << '\n'
              << "L2_candidate_reproduction_error=" << candidate2_error
              << " L2_block_smoothed_candidate_reproduction_error=" << smoothed_candidate2_error
              << " L2_transfer_adjoint_relative_error=" << transfer2_adjoint << '\n'
              << "L1_local_block_vs_nested_oracle_relative_error=" << block1_oracle_error
              << " L1_strength_block_vs_nested_oracle_relative_error="
              << strength_block_oracle_error << '\n'
              << "L2_local_block_vs_nested_oracle_relative_error=" << block2_oracle_error
              << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error << '\n'
              << std::fixed << std::setprecision(6)
              << "strength_threshold=" << strength1.stats.threshold
              << " strength_candidate_undirected_pairs=" << strength1.candidate_undirected_pairs
              << " tentative_undirected_pairs=" << tentative_pairs
              << " strength_directed_edges=" << strength1.stats.directed_edges
              << " strength_forced_pairs=" << strength1.stats.forced_pairs
              << " strength_degree_min=" << strength1.stats.min_degree
              << " strength_degree_avg=" << strength1.stats.average_degree
              << " strength_degree_max=" << strength1.stats.max_degree << '\n'
              << "strength_shared_with_tentative_pairs=" << strength1.shared_strong_tentative_pairs
              << " strength_overlap_over_strong=" << overlap_over_strong
              << " strength_overlap_over_tentative=" << overlap_over_tentative << '\n'
              << "P0_smoothed_support_cache_ms=" << p0_support_cache_ms
              << " L1_block_setup_ms=" << block1.setup_ms
              << " actual_A1_offdiagonal_setup_ms=" << actual_a1_offdiagonal_ms
              << " strength_graph_setup_ms=" << strength1.setup_ms
              << " L2_local_support_block_setup_ms=" << l2_local_setup_ms
              << " bottom_local_assembly_ms=" << bottom.assembly_ms << '\n'
              << "L1_block_storage_bytes=" << block1.storage_bytes()
              << " L2_block_storage_bytes=" << block2.storage_bytes()
              << " L2_block_rank_min=" << block2.min_rank
              << " L2_block_rank_max=" << block2.max_rank
              << " L2_block_min_cholesky_pivot=" << block2.min_cholesky_pivot << '\n'
              << "lambda0=" << lambda0 << " omega0=" << omega0
              << " block_lambda1=" << block_lambda1
              << " block_omega1=" << block_omega1
              << " actual_scalar_lambda2_reference=" << actual_scalar_lambda2
              << " block_lambda2=" << block_lambda2
              << " block_omega2=" << block_omega2 << '\n'
              << std::scientific << std::setprecision(9)
              << "bottom_symmetry_relative_defect=" << bottom.factor.symmetry_relative_defect
              << " bottom_min_cholesky_pivot=" << bottom.factor.min_pivot << '\n'
              << std::fixed << std::setprecision(6)
              << "hierarchy_setup_ms=" << elapsed_ms(setup_start, setup_stop)
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
    if (std::isfinite(post_transient_q)) {
        std::cout << "post_transient_geomean_q="
                  << std::scientific << std::setprecision(9)
                  << post_transient_q << '\n'
                  << "acceptance_q_le_0p4="
                  << (post_transient_q <= 0.4 ? "true" : "false") << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t max_cycles = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 6U;
        const std::size_t target_nodes = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 12U;
        const std::size_t min_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 4U;
        if (max_cycles == 0U || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid combined recursive SA options");
        }
        std::cout << "GFSS M5 actual-strength + localized-L2 consistency reference\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "reference_execution=cpu_fp64\n"
                  << "fixed_strength_threshold=0.05\n"
                  << "fixed_transfer_smoothing_steps=m0:1,m1:2,m2:1\n"
                  << "L1_metric=localized_exact_actual_block\n"
                  << "L1_graph=actual_smoothed_A1_energy_strength\n"
                  << "L2_metric=localized_exact_actual_block\n"
                  << "bottom=localized_exact_recursive_energy_direct_Cholesky\n"
                  << "dense_A1_materialized=false\n"
                  << "dense_A2_materialized=false\n"
                  << "all_runtime_transfer_restrictions=exact_transpose\n"
                  << "pre_smooth_degree=3 post_smooth_degree=3\n"
                  << "acceptance_target_post_transient_q<=0.4\n"
                  << "performance_status=numerical_reference_only\n";
        run_strength_local_l2_reference(max_cycles, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_strength_local_l2_reference_bench "
                  << "[max_cycles=6 [target_nodes=12 [min_nodes=4]]]\n";
        return 1;
    }
}
