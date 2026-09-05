// Numerical-reference experiment: rebuild the first recursive aggregation graph
// from the actual smoothed Galerkin A1 = P0^T A0 P0 without materializing A1.
// Smoothed P0 basis support is cached locally; exact off-diagonal aggregate
// blocks are accumulated element-by-element and thresholded by normalized
// block-Frobenius strength. The validated exact local L1 block metric remains
// unchanged so this isolates aggregation topology as the main variable.
#define main gfss_recursive_sa_l1_block_strength_helpers_only
#include "recursive_sa_l1_block_jacobi_reference_bench.cpp"
#undef main

#include <unordered_map>
#include <unordered_set>

namespace {

using StrengthNodeColumns = std::array<double, 18>; // 3 components x <=6 basis columns

struct StrengthBasisSupport {
    std::size_t rank{0};
    std::unordered_map<std::size_t, StrengthNodeColumns> values;
    std::vector<std::size_t> energy_elements;
};

struct ActualA1StrengthResult {
    AlgebraicNodeGraph graph;
    StrengthGraphStats stats;
    std::size_t candidate_undirected_pairs{0};
    std::size_t tentative_undirected_pairs{0};
    std::size_t shared_strong_tentative_pairs{0};
    double setup_ms{0.0};
    double sampled_block_oracle_relative_error{0.0};
};

StrengthBasisSupport build_strength_basis_support(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<std::vector<std::uint32_t>>& aggregate_nodes,
    const std::vector<double>& fine_inverse,
    double omega0,
    std::size_t aggregate_id) {
    const auto& aggregate = space.aggregates[aggregate_id];
    StrengthBasisSupport support;
    support.rank = aggregate.rank;
    if (support.rank == 0U || support.rank > kCandidates) {
        throw std::runtime_error("actual-A1 strength basis rank invalid");
    }

    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    std::vector<std::size_t> first_ring_elements;
    for (const auto node : aggregate_nodes[aggregate_id]) {
        append_node_touching_elements(mesh, node, first_ring_elements);
    }
    sort_unique(first_ring_elements);

    std::unordered_map<std::size_t, StrengthNodeColumns> aphi;
    aphi.reserve(first_ring_elements.size() * 4U + 16U);
    for (const auto element : first_ring_elements) {
        const auto ijk = decode_element(mesh, element);
        const auto nodes = mesh.element_nodes(ijk[0], ijk[1], ijk[2]);
        std::array<double, 24U * kCandidates> phi{};
        for (std::size_t a = 0; a < 8U; ++a) {
            const std::size_t node = static_cast<std::size_t>(nodes[a]);
            if (space.aggregate_of_node[node] != aggregate_id) continue;
            for (std::size_t c = 0; c < 3U; ++c) {
                const std::size_t ldof = 3U * a + c;
                for (std::size_t q = 0; q < support.rank; ++q) {
                    phi[ldof * kCandidates + q] =
                        local_tentative_basis_value(space, aggregate_id, node, c, q);
                }
            }
        }
        for (std::size_t i = 0; i < 24U; ++i) {
            const std::size_t node = static_cast<std::size_t>(nodes[i / 3U]);
            if (space.graph.constrained[node] != 0U) continue;
            auto [it, inserted] = aphi.try_emplace(node);
            if (inserted) it->second.fill(0.0);
            const std::size_t c = i % 3U;
            for (std::size_t q = 0; q < support.rank; ++q) {
                double value = 0.0;
                for (std::size_t j = 0; j < 24U; ++j) {
                    value += ke[i][j] * phi[j * kCandidates + q];
                }
                it->second[c * kCandidates + q] += value;
            }
        }
    }

    support.values.reserve(aphi.size() + aggregate_nodes[aggregate_id].size() + 16U);
    for (const auto& entry : aphi) {
        StrengthNodeColumns values{};
        const std::size_t node = entry.first;
        for (std::size_t c = 0; c < 3U; ++c) {
            const double inv = fine_inverse[3U * node + c];
            for (std::size_t q = 0; q < support.rank; ++q) {
                values[c * kCandidates + q] =
                    -omega0 * inv * entry.second[c * kCandidates + q];
            }
        }
        support.values.emplace(node, values);
    }
    for (const auto node_u32 : aggregate_nodes[aggregate_id]) {
        const std::size_t node = node_u32;
        auto [it, inserted] = support.values.try_emplace(node);
        if (inserted) it->second.fill(0.0);
        for (std::size_t c = 0; c < 3U; ++c) {
            for (std::size_t q = 0; q < support.rank; ++q) {
                it->second[c * kCandidates + q] +=
                    local_tentative_basis_value(space, aggregate_id, node, c, q);
            }
        }
    }

    support.energy_elements.reserve(support.values.size() * 8U);
    for (const auto& entry : support.values) {
        append_node_touching_elements(mesh, entry.first, support.energy_elements);
    }
    sort_unique(support.energy_elements);
    return support;
}

std::vector<StrengthBasisSupport> build_strength_basis_cache(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& fine_inverse,
    double omega0,
    double& setup_ms) {
    const auto start = Clock::now();
    std::vector<std::vector<std::uint32_t>> aggregate_nodes(space.aggregates.size());
    for (std::size_t node = 0; node < space.aggregate_of_node.size(); ++node) {
        const auto a = space.aggregate_of_node[node];
        if (a != kUnassigned) aggregate_nodes[a].push_back(static_cast<std::uint32_t>(node));
    }

    std::vector<StrengthBasisSupport> supports(space.aggregates.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (std::int64_t a64 = 0; a64 < static_cast<std::int64_t>(supports.size()); ++a64) {
        const auto a = static_cast<std::size_t>(a64);
        supports[a] = build_strength_basis_support(
            mesh, material, space, aggregate_nodes, fine_inverse, omega0, a);
    }
    setup_ms = elapsed_ms(start, Clock::now());
    return supports;
}

std::vector<std::vector<std::uint32_t>> build_strength_element_index(
    const gfss::StructuredHexMesh& mesh,
    const std::vector<StrengthBasisSupport>& supports) {
    std::vector<std::vector<std::uint32_t>> index(
        static_cast<std::size_t>(mesh.element_count()));
    for (std::size_t a = 0; a < supports.size(); ++a) {
        for (const auto element : supports[a].energy_elements) {
            index[element].push_back(static_cast<std::uint32_t>(a));
        }
    }
    for (auto& row : index) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return index;
}

std::uint64_t pair_key(std::uint32_t a, std::uint32_t b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32U) |
           static_cast<std::uint64_t>(b);
}

std::pair<std::uint32_t, std::uint32_t> decode_pair_key(std::uint64_t key) {
    return {static_cast<std::uint32_t>(key >> 32U),
            static_cast<std::uint32_t>(key & 0xffffffffULL)};
}

std::unordered_set<std::uint64_t> undirected_pairs_from_graph(
    const AlgebraicNodeGraph& graph) {
    std::unordered_set<std::uint64_t> pairs;
    pairs.reserve(graph.column_indices.size());
    for (std::uint32_t a = 0; a < graph.nodes(); ++a) {
        for (std::uint32_t p = graph.row_offsets[a]; p < graph.row_offsets[a + 1U]; ++p) {
            const auto b = graph.column_indices[p];
            if (a == b) continue;
            pairs.insert(pair_key(a, b));
        }
    }
    return pairs;
}

std::unordered_map<std::uint64_t, std::array<double, 36>>
accumulate_actual_a1_offdiagonal_blocks(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const std::vector<StrengthBasisSupport>& supports,
    const std::vector<std::vector<std::uint32_t>>& element_supports) {
    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    std::unordered_map<std::uint64_t, std::array<double, 36>> blocks;
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
                const auto& left0 = local[ia];
                const auto& right0 = local[ib];
                const ElementBasis* left = &left0;
                const ElementBasis* right = &right0;
                if (left->aggregate > right->aggregate) std::swap(left, right);
                const auto key = pair_key(left->aggregate, right->aggregate);
                auto [it, inserted] = blocks.try_emplace(key, std::array<double, 36>{});
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

double sampled_cross_block_oracle_error(
    const std::unordered_map<std::uint64_t, std::array<double, 36>>& blocks,
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
        const auto key = keys[sample];
        const auto [a, b] = decode_pair_key(key);
        const auto& block = blocks.at(key);
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

ActualA1StrengthResult build_actual_a1_strength_graph(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const AlgebraicNodeGraph& tentative_graph,
    const L1BlockMetric& block1,
    const std::vector<double>& fine_inverse,
    double omega0,
    double threshold,
    const Apply& apply1) {
    if (!(threshold > 0.0) || !(threshold < 1.0)) {
        throw std::invalid_argument("actual-A1 strength threshold must be in (0,1)");
    }
    if (tentative_graph.nodes() != space.aggregates.size() ||
        tentative_graph.dofs() != block1.dofs()) {
        throw std::invalid_argument("actual-A1 strength layout mismatch");
    }
    const auto start = Clock::now();

    double support_setup_ms = 0.0;
    const auto supports = build_strength_basis_cache(
        mesh, material, space, fine_inverse, omega0, support_setup_ms);
    const auto element_supports = build_strength_element_index(mesh, supports);
    const auto blocks = accumulate_actual_a1_offdiagonal_blocks(
        mesh, material, supports, element_supports);

    ActualA1StrengthResult result;
    result.candidate_undirected_pairs = blocks.size();
    result.sampled_block_oracle_relative_error =
        sampled_cross_block_oracle_error(blocks, tentative_graph, apply1);

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
            throw std::runtime_error("actual-A1 diagonal block norm invalid");
        }
    }

    std::vector<std::vector<std::uint32_t>> neighbors(nodes);
    std::vector<double> strongest(nodes, 0.0);
    std::vector<std::uint32_t> strongest_neighbor(nodes, kUnassigned);
    std::unordered_set<std::uint64_t> strong_pairs;
    strong_pairs.reserve(blocks.size());

    for (const auto& entry : blocks) {
        const auto [a, b] = decode_pair_key(entry.first);
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
            throw std::runtime_error("actual-A1 strength graph isolated zero-coupling node");
        }
        neighbors[a].push_back(b);
        neighbors[b].push_back(static_cast<std::uint32_t>(a));
        strong_pairs.insert(pair_key(static_cast<std::uint32_t>(a), b));
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
        result.graph.column_indices.insert(
            result.graph.column_indices.end(), row.begin(), row.end());
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

    const auto tentative_pairs = undirected_pairs_from_graph(tentative_graph);
    result.tentative_undirected_pairs = tentative_pairs.size();
    for (const auto key : strong_pairs) {
        if (tentative_pairs.find(key) != tentative_pairs.end()) {
            ++result.shared_strong_tentative_pairs;
        }
    }
    result.setup_ms = elapsed_ms(start, Clock::now());
    return result;
}

void run_actual_a1_strength_reference(
    std::size_t m0,
    std::size_t m1,
    std::size_t m2,
    std::size_t max_cycles,
    double strength_threshold,
    std::size_t target_nodes,
    std::size_t min_nodes) {
    if (m0 != 1U) {
        throw std::invalid_argument("actual-A1 local strength reference currently requires m0=1");
    }
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
    const auto actual_inverse1 = block1.inverse_scalar_diagonal();
    const double block_lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
    const double block_omega1 = kSaDampingNumerator / block_lambda1;

    const auto strength1 = build_actual_a1_strength_graph(
        mesh, material, space0, graph1_tentative, block1,
        fine_inverse, omega0, strength_threshold, apply1);

    const auto transfer1_tentative = build_candidate_transfer(
        strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
    const double candidate1_error =
        candidate_reproduction_error(transfer1_tentative, candidates1);
    const L1BlockSmoothedTransfer transfer1{
        transfer1_tentative, apply1, block1, block_omega1, m1};
    const double transfer1_adjoint = l1_block_transfer_adjoint_error(transfer1);
    const double smoothed_candidate1_error = l1_block_candidate_error(transfer1, candidates1);

    const Apply apply2 = [&](const Vec& x) {
        const auto l1 = transfer1.prolong(x);
        return transfer1.restrict_transpose(apply1(l1));
    };
    const auto inverse2 =
        transfer1_tentative.approximate_inverse_coarse_diagonal(actual_inverse1);
    const double lambda2 = estimate_lambda_max(apply2, inverse2, 8U);
    const double omega2 = kSaDampingNumerator / lambda2;

    const auto transfer2_tentative = build_candidate_transfer(
        transfer1_tentative.coarse_graph,
        transfer1_tentative.coarse_candidates,
        target_nodes,
        min_nodes,
        1.0e-10);
    const double candidate2_error = candidate_reproduction_error(
        transfer2_tentative, transfer1_tentative.coarse_candidates);
    const AlgebraicSmoothedTransferActualA2 transfer2{
        transfer2_tentative, apply2, inverse2, omega2, m2};
    const double transfer2_adjoint =
        smoothed_transfer_adjoint_error_actual_a2(transfer2);

    const Apply apply3 = [&](const Vec& x) {
        const auto l2 = transfer2.prolong(x);
        return transfer2.restrict_transpose(apply2(l2));
    };
    const auto bottom_start = Clock::now();
    const auto bottom = materialize_and_factor_bottom(
        apply3, transfer2_tentative.coarse_dofs);
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
    levels[1].label = "L1_actual_block_metric_actual_strength_graph";
    levels[1].diagnostic_lambda_max = block_lambda1;
    levels[1].apply = apply1;
    levels[1].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply1, block1, block_lambda1, b, x, degree);
    };
    levels[1].restrict_to_coarse = [&](const Vec& r) { return transfer1.restrict_transpose(r); };
    levels[1].prolong_from_coarse = [&](const Vec& c) { return transfer1.prolong(c); };

    levels[2].dofs = transfer1_tentative.coarse_dofs;
    levels[2].label = "L2_from_actual_A1_strength_graph";
    levels[2].diagnostic_lambda_max = lambda2;
    levels[2].apply = apply2;
    levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_smooth(apply2, inverse2, lambda2, b, x, degree);
    };
    levels[2].restrict_to_coarse = [&](const Vec& r) { return transfer2.restrict_transpose(r); };
    levels[2].prolong_from_coarse = [&](const Vec& c) { return transfer2.prolong(c); };

    levels[3].dofs = transfer2_tentative.coarse_dofs;
    levels[3].label = "L3_dense_Cholesky_bottom";
    levels[3].apply = apply3;
    levels[3].bottom_solve = [&](const Vec& b) { return bottom.solve(b); };

    const auto rhs = make_rhs(mesh);
    const auto result = gfss::solve_reference_multilevel_vcycle(
        levels, rhs, 1.0e-6, max_cycles, 3U, 3U);

    const auto tentative_pairs = strength1.tentative_undirected_pairs;
    const auto strong_pairs = strength1.stats.directed_edges / 2U;
    const double overlap_over_strong = strong_pairs > 0U
        ? static_cast<double>(strength1.shared_strong_tentative_pairs) /
          static_cast<double>(strong_pairs) : 0.0;
    const double overlap_over_tentative = tentative_pairs > 0U
        ? static_cast<double>(strength1.shared_strong_tentative_pairs) /
          static_cast<double>(tentative_pairs) : 0.0;

    std::cout << "\n========================================\n"
              << "L0_transfer_smoothing_steps=" << m0
              << " L1_transfer_smoothing_steps=" << m1
              << " L2_transfer_smoothing_steps=" << m2 << '\n'
              << "hierarchy_levels=4\n"
              << "L1_metric=localized_exact_actual_block\n"
              << "L1_graph=localized_actual_smoothed_A1_block_strength\n"
              << "L2_metric=scalar_propagated_from_actual_L1_block_diagonal\n"
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
              << " L2_transfer_adjoint_relative_error=" << transfer2_adjoint << '\n'
              << "L1_local_block_vs_nested_oracle_relative_error=" << block1_oracle_error
              << " L1_strength_block_vs_nested_oracle_relative_error="
              << strength1.sampled_block_oracle_relative_error << '\n'
              << std::fixed << std::setprecision(6)
              << "strength_threshold=" << strength1.stats.threshold
              << " strength_candidate_undirected_pairs=" << strength1.candidate_undirected_pairs
              << " tentative_undirected_pairs=" << tentative_pairs
              << " strength_directed_edges=" << strength1.stats.directed_edges
              << " strength_forced_pairs=" << strength1.stats.forced_pairs
              << " strength_degree_min=" << strength1.stats.min_degree
              << " strength_degree_avg=" << strength1.stats.average_degree
              << " strength_degree_max=" << strength1.stats.max_degree << '\n'
              << "strength_shared_with_tentative_pairs="
              << strength1.shared_strong_tentative_pairs
              << " strength_overlap_over_strong=" << overlap_over_strong
              << " strength_overlap_over_tentative=" << overlap_over_tentative << '\n'
              << "L1_block_setup_ms=" << block1.setup_ms
              << " actual_A1_strength_setup_ms=" << strength1.setup_ms
              << " L1_block_storage_bytes=" << block1.storage_bytes() << '\n'
              << "lambda0=" << lambda0 << " omega0=" << omega0
              << " block_lambda1=" << block_lambda1
              << " block_omega1=" << block_omega1
              << " lambda2=" << lambda2
              << " omega2=" << omega2 << '\n'
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
            std::cout << "post_transient_geomean_q="
                      << std::scientific << std::setprecision(9)
                      << std::exp(log_sum / static_cast<double>(count)) << '\n';
        }
    }
}

} // namespace

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
        if (m0 != 1U || m1 > 4U || m2 > 4U || max_cycles == 0U ||
            !(strength_threshold > 0.0) || !(strength_threshold < 1.0) ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid actual-A1 strength reference options");
        }

        std::cout << "GFSS M5 actual-A1 local-strength recursive SA numerical reference\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "purpose=isolate_first_recursive_actual_strength_graph_effect\n"
                  << "reference_execution=cpu_fp64\n"
                  << "L0_transfer_smoothing_steps_required=1\n"
                  << "L1_metric=localized_exact_actual_block\n"
                  << "L1_graph=actual_smoothed_Galerkin_block_strength_from_local_element_energy\n"
                  << "L1_full_matrix=not_materialized\n"
                  << "L2_full_matrix=not_materialized\n"
                  << "all_transfer_restrictions=exact_transpose\n"
                  << "pre_smooth_degree=3 post_smooth_degree=3\n"
                  << "acceptance_target_post_transient_q<=0.4\n"
                  << "performance_status=numerical_reference_only\n";

        run_actual_a1_strength_reference(
            m0, m1, m2, max_cycles,
            strength_threshold, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_actual_a1_strength_reference_bench "
                  << "[m0=1 [m1 [m2 [max_cycles [strength_threshold [target_nodes [min_nodes]]]]]]]\n";
        return 1;
    }
}
