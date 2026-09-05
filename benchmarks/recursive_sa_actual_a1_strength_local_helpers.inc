// Reference-only exact actual-A1 off-diagonal block and strength-graph helpers.
// Requires recursive_sa_local_l2_helpers.inc to have provided the common
// recursive-SA types and cached smoothed-P0 FineBasisSupport representation.

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

}  // namespace
