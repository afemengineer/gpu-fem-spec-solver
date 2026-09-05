// Numerical-reference sweep for the first recursive strength threshold.
// Build the exact smoothed Galerkin A1 = P0^T A0 P0 once as a sparse
// variable-block CPU reference from local HEX8 energies, verify it against the
// nested matrix-free operator, then reuse it across all strength thresholds.
// This is reference-only storage: production remains matrix-light.
#define main gfss_recursive_sa_l1_block_sweep_helpers_only
#include "recursive_sa_l1_block_jacobi_reference_bench.cpp"
#undef main

#include <unordered_map>
#include <unordered_set>

namespace {

using SweepNodeColumns = std::array<double, 18>; // 3 components x <=6 columns
using SweepBlock = std::array<double, 36>;       // <=6 x <=6, stride 6

struct SweepBasisSupport {
    std::size_t rank{0};
    std::unordered_map<std::size_t, SweepNodeColumns> values;
    std::vector<std::size_t> energy_elements;
};

std::uint64_t sweep_pair_key(std::uint32_t a, std::uint32_t b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(a) << 32U) |
           static_cast<std::uint64_t>(b);
}

std::pair<std::uint32_t, std::uint32_t> sweep_decode_pair(std::uint64_t key) {
    return {static_cast<std::uint32_t>(key >> 32U),
            static_cast<std::uint32_t>(key & 0xffffffffULL)};
}

SweepBasisSupport build_sweep_basis_support(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<std::vector<std::uint32_t>>& aggregate_nodes,
    const std::vector<double>& fine_inverse,
    double omega0,
    std::size_t aggregate_id) {
    const auto& aggregate = space.aggregates[aggregate_id];
    SweepBasisSupport support;
    support.rank = aggregate.rank;
    if (support.rank == 0U || support.rank > kCandidates) {
        throw std::runtime_error("strength sweep P0 basis rank invalid");
    }

    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    std::vector<std::size_t> first_ring;
    for (const auto node : aggregate_nodes[aggregate_id]) {
        append_node_touching_elements(mesh, node, first_ring);
    }
    sort_unique(first_ring);

    std::unordered_map<std::size_t, SweepNodeColumns> aphi;
    aphi.reserve(first_ring.size() * 4U + 16U);
    for (const auto element : first_ring) {
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
        SweepNodeColumns values{};
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

std::vector<SweepBasisSupport> build_sweep_basis_cache(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& fine_inverse,
    double omega0,
    double& setup_ms) {
    const auto start = Clock::now();
    std::vector<std::vector<std::uint32_t>> aggregate_nodes(space.aggregates.size());
    for (std::size_t node = 0; node < space.aggregate_of_node.size(); ++node) {
        const auto aggregate = space.aggregate_of_node[node];
        if (aggregate != kUnassigned) {
            aggregate_nodes[aggregate].push_back(static_cast<std::uint32_t>(node));
        }
    }

    std::vector<SweepBasisSupport> supports(space.aggregates.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (std::int64_t a64 = 0; a64 < static_cast<std::int64_t>(supports.size()); ++a64) {
        const auto a = static_cast<std::size_t>(a64);
        supports[a] = build_sweep_basis_support(
            mesh, material, space, aggregate_nodes, fine_inverse, omega0, a);
    }
    setup_ms = elapsed_ms(start, Clock::now());
    return supports;
}

std::vector<std::vector<std::uint32_t>> build_sweep_element_index(
    const gfss::StructuredHexMesh& mesh,
    const std::vector<SweepBasisSupport>& supports) {
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

std::unordered_map<std::uint64_t, SweepBlock> accumulate_sweep_offdiagonal_blocks(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const std::vector<SweepBasisSupport>& supports,
    const std::vector<std::vector<std::uint32_t>>& element_supports) {
    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    std::unordered_map<std::uint64_t, SweepBlock> blocks;
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
                const auto key = sweep_pair_key(left->aggregate, right->aggregate);
                auto [it, inserted] = blocks.try_emplace(key, SweepBlock{});
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

struct SparseBlockA1 {
    AlgebraicNodeGraph layout;
    const L1BlockMetric* diagonal{nullptr};
    const std::unordered_map<std::uint64_t, SweepBlock>* offdiagonal{nullptr};

    Vec apply(const Vec& x) const {
        if (diagonal == nullptr || offdiagonal == nullptr || x.size() != layout.dofs()) {
            throw std::invalid_argument("sparse actual-A1 apply size/state mismatch");
        }
        Vec y(x.size(), 0.0);
        for (std::size_t a = 0; a < layout.nodes(); ++a) {
            const std::size_t begin = layout.dof_offsets[a];
            const std::size_t rank = layout.dof_offsets[a + 1U] - begin;
            for (std::size_t i = 0; i < rank; ++i) {
                double value = 0.0;
                for (std::size_t j = 0; j < rank; ++j) {
                    value += diagonal->block_entry(a, i, j) * x[begin + j];
                }
                y[begin + i] += value;
            }
        }
        for (const auto& entry : *offdiagonal) {
            const auto [a, b] = sweep_decode_pair(entry.first);
            const std::size_t abegin = layout.dof_offsets[a];
            const std::size_t bbegin = layout.dof_offsets[b];
            const std::size_t arank = layout.dof_offsets[a + 1U] - abegin;
            const std::size_t brank = layout.dof_offsets[b + 1U] - bbegin;
            const auto& block = entry.second;
            for (std::size_t i = 0; i < arank; ++i) {
                double value = 0.0;
                for (std::size_t j = 0; j < brank; ++j) {
                    value += block[i * kCandidates + j] * x[bbegin + j];
                }
                y[abegin + i] += value;
            }
            for (std::size_t j = 0; j < brank; ++j) {
                double value = 0.0;
                for (std::size_t i = 0; i < arank; ++i) {
                    value += block[i * kCandidates + j] * x[abegin + i];
                }
                y[bbegin + j] += value;
            }
        }
        return y;
    }
};

double sparse_a1_oracle_error(const SparseBlockA1& sparse, const Apply& nested) {
    const auto x = deterministic_actual_a2_probe(sparse.layout.dofs(), 0.67);
    const auto ys = sparse.apply(x);
    const auto yn = nested(x);
    Vec diff(ys.size(), 0.0);
    for (std::size_t i = 0; i < ys.size(); ++i) diff[i] = ys[i] - yn[i];
    return norm(diff) / std::max(norm(yn), 1.0e-300);
}

std::unordered_set<std::uint64_t> sweep_pairs_from_graph(const AlgebraicNodeGraph& graph) {
    std::unordered_set<std::uint64_t> pairs;
    pairs.reserve(graph.column_indices.size());
    for (std::uint32_t a = 0; a < graph.nodes(); ++a) {
        for (std::uint32_t p = graph.row_offsets[a]; p < graph.row_offsets[a + 1U]; ++p) {
            const auto b = graph.column_indices[p];
            if (a != b) pairs.insert(sweep_pair_key(a, b));
        }
    }
    return pairs;
}

struct SweepStrengthCache {
    std::vector<double> diagonal_norm;
    std::unordered_map<std::uint64_t, double> strength;
    std::vector<double> strongest;
    std::vector<std::uint32_t> strongest_neighbor;
    std::unordered_set<std::uint64_t> tentative_pairs;
};

SweepStrengthCache build_strength_cache(
    const AlgebraicNodeGraph& layout,
    const L1BlockMetric& block1,
    const std::unordered_map<std::uint64_t, SweepBlock>& blocks,
    const AlgebraicNodeGraph& tentative_graph) {
    SweepStrengthCache cache;
    cache.diagonal_norm.resize(layout.nodes(), 0.0);
    cache.strongest.assign(layout.nodes(), 0.0);
    cache.strongest_neighbor.assign(layout.nodes(), kUnassigned);
    cache.strength.reserve(blocks.size());
    cache.tentative_pairs = sweep_pairs_from_graph(tentative_graph);

    for (std::size_t a = 0; a < layout.nodes(); ++a) {
        const std::size_t rank = layout.dof_offsets[a + 1U] - layout.dof_offsets[a];
        double sum2 = 0.0;
        for (std::size_t i = 0; i < rank; ++i) {
            for (std::size_t j = 0; j < rank; ++j) {
                const double v = block1.block_entry(a, i, j);
                sum2 += v * v;
            }
        }
        cache.diagonal_norm[a] = std::sqrt(sum2);
        if (!(cache.diagonal_norm[a] > 0.0)) {
            throw std::runtime_error("strength sweep diagonal norm invalid");
        }
    }

    for (const auto& entry : blocks) {
        const auto [a, b] = sweep_decode_pair(entry.first);
        const std::size_t arank = layout.dof_offsets[a + 1U] - layout.dof_offsets[a];
        const std::size_t brank = layout.dof_offsets[b + 1U] - layout.dof_offsets[b];
        double sum2 = 0.0;
        for (std::size_t i = 0; i < arank; ++i) {
            for (std::size_t j = 0; j < brank; ++j) {
                const double v = entry.second[i * kCandidates + j];
                sum2 += v * v;
            }
        }
        const double denom = std::sqrt(cache.diagonal_norm[a] * cache.diagonal_norm[b]);
        const double s = denom > 0.0 ? std::sqrt(sum2) / denom : 0.0;
        cache.strength.emplace(entry.first, s);
        if (s > cache.strongest[a]) {
            cache.strongest[a] = s;
            cache.strongest_neighbor[a] = b;
        }
        if (s > cache.strongest[b]) {
            cache.strongest[b] = s;
            cache.strongest_neighbor[b] = a;
        }
    }
    return cache;
}

struct ThresholdGraphResult {
    AlgebraicNodeGraph graph;
    StrengthGraphStats stats;
    std::size_t strong_undirected_pairs{0};
    std::size_t shared_tentative_pairs{0};
};

ThresholdGraphResult threshold_strength_graph(
    const AlgebraicNodeGraph& layout,
    const SweepStrengthCache& cache,
    double threshold) {
    ThresholdGraphResult result;
    const std::size_t nodes = layout.nodes();
    std::vector<std::vector<std::uint32_t>> neighbors(nodes);
    std::unordered_set<std::uint64_t> strong_pairs;
    strong_pairs.reserve(cache.strength.size());

    for (const auto& entry : cache.strength) {
        if (entry.second < threshold) continue;
        const auto [a, b] = sweep_decode_pair(entry.first);
        neighbors[a].push_back(b);
        neighbors[b].push_back(a);
        strong_pairs.insert(entry.first);
    }

    std::size_t forced = 0U;
    for (std::size_t a = 0; a < nodes; ++a) {
        if (!neighbors[a].empty()) continue;
        const auto b = cache.strongest_neighbor[a];
        if (b == kUnassigned || !(cache.strongest[a] > 0.0)) {
            throw std::runtime_error("strength sweep produced isolated zero-coupling node");
        }
        neighbors[a].push_back(b);
        neighbors[b].push_back(static_cast<std::uint32_t>(a));
        strong_pairs.insert(sweep_pair_key(static_cast<std::uint32_t>(a), b));
        ++forced;
    }

    result.graph.dof_offsets = layout.dof_offsets;
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
    result.stats.forced_pairs = forced;
    result.stats.min_degree = min_degree;
    result.stats.average_degree = static_cast<double>(cursor) / static_cast<double>(nodes);
    result.stats.max_degree = max_degree;
    result.strong_undirected_pairs = strong_pairs.size();
    for (const auto key : strong_pairs) {
        if (cache.tentative_pairs.find(key) != cache.tentative_pairs.end()) {
            ++result.shared_tentative_pairs;
        }
    }
    return result;
}

double post_transient_q(const gfss::ReferenceMultilevelResult& result) {
    if (result.relative_residuals.size() <= 2U) return std::numeric_limits<double>::quiet_NaN();
    double log_sum = 0.0;
    std::size_t count = 0U;
    for (std::size_t i = 2U; i < result.relative_residuals.size(); ++i) {
        const double q = result.relative_residuals[i] / result.relative_residuals[i - 1U];
        if (q > 0.0 && std::isfinite(q)) {
            log_sum += std::log(q);
            ++count;
        }
    }
    return count > 0U ? std::exp(log_sum / static_cast<double>(count))
                      : std::numeric_limits<double>::quiet_NaN();
}

struct SweepRow {
    double theta{0.0};
    ThresholdGraphResult graph;
    std::size_t l2_dofs{0};
    std::size_t l2_nodes{0};
    std::size_t l3_dofs{0};
    std::size_t l3_nodes{0};
    double lambda2{0.0};
    double omega2{0.0};
    double transfer1_adjoint{0.0};
    double transfer2_adjoint{0.0};
    double bottom_symmetry{0.0};
    double bottom_pivot{0.0};
    double bottom_ms{0.0};
    double solve_ms{0.0};
    double q{0.0};
    double final_residual{0.0};
};

void run_strength_sweep(std::size_t max_cycles,
                        std::size_t target_nodes,
                        std::size_t min_nodes) {
    constexpr std::size_t m0 = 1U;
    constexpr std::size_t m1 = 2U;
    constexpr std::size_t m2 = 1U;
    const std::array<double, 6> thresholds{{0.02, 0.035, 0.05, 0.075, 0.10, 0.15}};
    const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
    const gfss::Material material{210.0e9, 0.30};
    const auto shared_start = Clock::now();

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
    const Apply apply1_nested = [&](const Vec& x) {
        const auto fine = transfer0.prolong(x);
        return transfer0.restrict_transpose(apply0(fine));
    };

    const auto graph1_layout = graph_from_variable_blocks(tentative_a1);
    const auto candidates1 = make_level1_candidates(space0);
    const auto block1 = build_exact_l1_block_metric(
        mesh, material, space0, graph1_layout, fine_inverse, omega0);
    const double block1_oracle = audit_l1_block_metric(block1, apply1_nested);
    const auto actual_inverse1 = block1.inverse_scalar_diagonal();

    double support_ms = 0.0;
    const auto supports = build_sweep_basis_cache(
        mesh, material, space0, fine_inverse, omega0, support_ms);
    const auto element_supports = build_sweep_element_index(mesh, supports);
    const auto offdiag_start = Clock::now();
    const auto offdiag = accumulate_sweep_offdiagonal_blocks(
        mesh, material, supports, element_supports);
    const double offdiag_ms = elapsed_ms(offdiag_start, Clock::now());
    const SparseBlockA1 sparse_a1{graph1_layout, &block1, &offdiag};
    const Apply apply1 = [&](const Vec& x) { return sparse_a1.apply(x); };
    const double sparse_oracle = sparse_a1_oracle_error(sparse_a1, apply1_nested);
    const double block_lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
    const double block_omega1 = kSaDampingNumerator / block_lambda1;
    const auto strength_cache = build_strength_cache(
        graph1_layout, block1, offdiag, graph1_layout);
    const auto shared_stop = Clock::now();

    std::cout << "GFSS M5 actual-A1 strength threshold sweep\n"
              << "problem=thin_plate mesh=64x64x8\n"
              << "fixed_m0=1 fixed_m1=2 fixed_m2=1\n"
              << "L1_metric=localized_exact_actual_block\n"
              << "L1_operator=sparse_exact_actual_smoothed_Galerkin_reference\n"
              << "L2_metric=scalar_propagated_from_actual_L1_block_diagonal\n"
              << "thresholds=0.02,0.035,0.05,0.075,0.10,0.15\n"
              << "production_status=reference_only_sparse_A1_not_a_Pareto_candidate\n"
              << std::scientific << std::setprecision(9)
              << "L1_local_block_vs_nested_oracle_relative_error=" << block1_oracle
              << " sparse_A1_vs_nested_relative_error=" << sparse_oracle << '\n'
              << std::fixed << std::setprecision(6)
              << "shared_L1_block_setup_ms=" << block1.setup_ms
              << " shared_P0_support_setup_ms=" << support_ms
              << " shared_offdiagonal_block_setup_ms=" << offdiag_ms
              << " shared_total_setup_ms=" << elapsed_ms(shared_start, shared_stop) << '\n'
              << "candidate_undirected_pairs=" << offdiag.size()
              << " tentative_undirected_pairs=" << strength_cache.tentative_pairs.size()
              << " sparse_A1_reference_payload_bytes~="
              << offdiag.size() * sizeof(SweepBlock) + block1.storage_bytes() << '\n'
              << "lambda0=" << lambda0 << " omega0=" << omega0
              << " block_lambda1=" << block_lambda1
              << " block_omega1=" << block_omega1 << '\n';

    std::vector<SweepRow> rows;
    rows.reserve(thresholds.size());
    const auto rhs = make_rhs(mesh);

    for (const double theta : thresholds) {
        const auto case_start = Clock::now();
        SweepRow row;
        row.theta = theta;
        row.graph = threshold_strength_graph(graph1_layout, strength_cache, theta);
        const auto transfer1_tentative = build_candidate_transfer(
            row.graph.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
        const L1BlockSmoothedTransfer transfer1{
            transfer1_tentative, apply1, block1, block_omega1, m1};
        row.transfer1_adjoint = l1_block_transfer_adjoint_error(transfer1);

        const Apply apply2 = [&](const Vec& x) {
            const auto l1 = transfer1.prolong(x);
            return transfer1.restrict_transpose(apply1(l1));
        };
        const auto inverse2 =
            transfer1_tentative.approximate_inverse_coarse_diagonal(actual_inverse1);
        row.lambda2 = estimate_lambda_max(apply2, inverse2, 8U);
        row.omega2 = kSaDampingNumerator / row.lambda2;

        const auto transfer2_tentative = build_candidate_transfer(
            transfer1_tentative.coarse_graph,
            transfer1_tentative.coarse_candidates,
            target_nodes,
            min_nodes,
            1.0e-10);
        const AlgebraicSmoothedTransferActualA2 transfer2{
            transfer2_tentative, apply2, inverse2, row.omega2, m2};
        row.transfer2_adjoint = smoothed_transfer_adjoint_error_actual_a2(transfer2);
        const Apply apply3 = [&](const Vec& x) {
            const auto l2 = transfer2.prolong(x);
            return transfer2.restrict_transpose(apply2(l2));
        };

        const auto bottom_start = Clock::now();
        const auto bottom = materialize_and_factor_bottom(
            apply3, transfer2_tentative.coarse_dofs);
        row.bottom_ms = elapsed_ms(bottom_start, Clock::now());
        row.bottom_symmetry = bottom.symmetry_relative_defect;
        row.bottom_pivot = bottom.min_pivot;

        std::vector<gfss::ReferenceMultilevelLevel> levels(4U);
        levels[0].dofs = static_cast<std::size_t>(mesh.dof_count());
        levels[0].apply = apply0;
        levels[0].diagnostic_lambda_max = lambda0;
        levels[0].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
            chebyshev_smooth(apply0, fine_inverse, lambda0, b, x, degree);
            clamp_x0(mesh, x);
        };
        levels[0].restrict_to_coarse = [&](const Vec& r) { return transfer0.restrict_transpose(r); };
        levels[0].prolong_from_coarse = [&](const Vec& c) { return transfer0.prolong(c); };

        levels[1].dofs = space0.coarse_dofs;
        levels[1].apply = apply1;
        levels[1].diagnostic_lambda_max = block_lambda1;
        levels[1].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
            chebyshev_l1_block_smooth(apply1, block1, block_lambda1, b, x, degree);
        };
        levels[1].restrict_to_coarse = [&](const Vec& r) { return transfer1.restrict_transpose(r); };
        levels[1].prolong_from_coarse = [&](const Vec& c) { return transfer1.prolong(c); };

        levels[2].dofs = transfer1_tentative.coarse_dofs;
        levels[2].apply = apply2;
        levels[2].diagnostic_lambda_max = row.lambda2;
        levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
            chebyshev_smooth(apply2, inverse2, row.lambda2, b, x, degree);
        };
        levels[2].restrict_to_coarse = [&](const Vec& r) { return transfer2.restrict_transpose(r); };
        levels[2].prolong_from_coarse = [&](const Vec& c) { return transfer2.prolong(c); };

        levels[3].dofs = transfer2_tentative.coarse_dofs;
        levels[3].apply = apply3;
        levels[3].bottom_solve = [&](const Vec& b) { return bottom.solve(b); };

        const auto result = gfss::solve_reference_multilevel_vcycle(
            levels, rhs, 1.0e-6, max_cycles, 3U, 3U);
        row.solve_ms = result.solve_ms;
        row.q = post_transient_q(result);
        row.final_residual = result.relative_residuals.back();
        row.l2_dofs = levels[2].dofs;
        row.l2_nodes = transfer1_tentative.coarse_graph.nodes();
        row.l3_dofs = levels[3].dofs;
        row.l3_nodes = transfer2_tentative.coarse_graph.nodes();
        rows.push_back(row);

        const double overlap_strong = row.graph.strong_undirected_pairs > 0U
            ? static_cast<double>(row.graph.shared_tentative_pairs) /
              static_cast<double>(row.graph.strong_undirected_pairs) : 0.0;
        const double overlap_tentative = !strength_cache.tentative_pairs.empty()
            ? static_cast<double>(row.graph.shared_tentative_pairs) /
              static_cast<double>(strength_cache.tentative_pairs.size()) : 0.0;

        std::cout << std::fixed << std::setprecision(6)
                  << "theta=" << theta
                  << " edges=" << row.graph.stats.directed_edges
                  << " degree_avg=" << row.graph.stats.average_degree
                  << " degree_min=" << row.graph.stats.min_degree
                  << " degree_max=" << row.graph.stats.max_degree
                  << " forced=" << row.graph.stats.forced_pairs
                  << " L2_dofs=" << row.l2_dofs
                  << " L2_nodes=" << row.l2_nodes
                  << " L3_dofs=" << row.l3_dofs
                  << " L3_nodes=" << row.l3_nodes
                  << " lambda2=" << row.lambda2
                  << " omega2=" << row.omega2
                  << " overlap_strong=" << overlap_strong
                  << " overlap_tentative=" << overlap_tentative
                  << std::scientific << std::setprecision(9)
                  << " adj1=" << row.transfer1_adjoint
                  << " adj2=" << row.transfer2_adjoint
                  << " bottom_sym=" << row.bottom_symmetry
                  << " bottom_pivot=" << row.bottom_pivot
                  << " q=" << row.q
                  << " final_r=" << row.final_residual
                  << std::fixed << std::setprecision(3)
                  << " bottom_ms=" << row.bottom_ms
                  << " solve_ms=" << row.solve_ms
                  << " case_ms=" << elapsed_ms(case_start, Clock::now()) << '\n';
    }

    const auto best = std::min_element(rows.begin(), rows.end(),
        [](const SweepRow& a, const SweepRow& b) { return a.q < b.q; });
    if (best != rows.end()) {
        std::cout << std::scientific << std::setprecision(9)
                  << "best_theta=" << best->theta
                  << " best_post_transient_geomean_q=" << best->q
                  << " acceptance_q_le_0p4=" << (best->q <= 0.4 ? "true" : "false")
                  << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t max_cycles = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 6U;
        const std::size_t target_nodes = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 12U;
        const std::size_t min_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 4U;
        if (max_cycles == 0U || target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid actual-A1 strength sweep options");
        }
        run_strength_sweep(max_cycles, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_actual_a1_strength_sweep_bench "
                  << "[max_cycles [target_nodes [min_nodes]]]\n";
        return 1;
    }
}
