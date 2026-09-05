// Numerical-reference experiment: extend the validated localized actual-L1
// block metric recursively. Cache the m0=1 smoothed P0 aggregate basis in fine
// space, use exact element-local energies as a local A1 action, build the actual
// L2 diagonal blocks after m1 smoothing without materializing A1/A2, and assemble
// the tiny bottom operator from the same local hierarchy.
#define main gfss_recursive_sa_l1_block_reference_main
#include "recursive_sa_l1_block_jacobi_reference_bench.cpp"
#undef main

namespace {

using FineNodeColumns = std::array<double, 18>;   // 3 fine components x <=6 columns
using LocalBlockColumns = std::array<double, 36>; // <=6 row dofs x <=6 columns

struct FineBasisSupport {
    std::size_t rank{0};
    std::unordered_map<std::size_t, FineNodeColumns> values;
    std::vector<std::size_t> energy_elements;
};

struct LocalColumns {
    std::size_t cols{0};
    std::unordered_map<std::uint32_t, LocalBlockColumns> values;
};

struct DofLayout {
    std::vector<std::uint32_t> node_of_dof;
    std::vector<std::uint8_t> local_of_dof;
};

struct LocalBottomReference {
    DenseCholesky factor;
    std::vector<double> values;
    double assembly_ms{0.0};
};

LocalBlockColumns& zero_block(std::unordered_map<std::uint32_t, LocalBlockColumns>& map,
                              std::uint32_t node) {
    auto [it, inserted] = map.try_emplace(node);
    if (inserted) it->second.fill(0.0);
    return it->second;
}

DofLayout make_dof_layout(const AlgebraicNodeGraph& graph) {
    DofLayout layout;
    layout.node_of_dof.resize(graph.dofs(), 0U);
    layout.local_of_dof.resize(graph.dofs(), 0U);
    for (std::size_t node = 0; node < graph.nodes(); ++node) {
        const std::size_t begin = graph.dof_offsets[node];
        const std::size_t end = graph.dof_offsets[node + 1U];
        if (end - begin > kCandidates) {
            throw std::runtime_error("local hierarchy node rank exceeds six");
        }
        for (std::size_t dof = begin; dof < end; ++dof) {
            layout.node_of_dof[dof] = static_cast<std::uint32_t>(node);
            layout.local_of_dof[dof] = static_cast<std::uint8_t>(dof - begin);
        }
    }
    return layout;
}

FineBasisSupport build_one_fine_basis_support(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<std::vector<std::uint32_t>>& aggregate_nodes,
    const std::vector<double>& fine_inverse,
    double omega0,
    std::size_t aggregate_id) {
    const auto& aggregate = space.aggregates[aggregate_id];
    FineBasisSupport support;
    support.rank = aggregate.rank;
    if (support.rank == 0U || support.rank > kCandidates) {
        throw std::runtime_error("local P0 basis rank invalid");
    }

    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    std::vector<std::size_t> first_ring_elements;
    for (const auto node : aggregate_nodes[aggregate_id]) {
        append_node_touching_elements(mesh, node, first_ring_elements);
    }
    sort_unique(first_ring_elements);

    std::unordered_map<std::size_t, FineNodeColumns> aphi;
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
            auto& out = aphi[node];
            const std::size_t c = i % 3U;
            for (std::size_t q = 0; q < support.rank; ++q) {
                double value = 0.0;
                for (std::size_t j = 0; j < 24U; ++j) {
                    value += ke[i][j] * phi[j * kCandidates + q];
                }
                out[c * kCandidates + q] += value;
            }
        }
    }

    support.values.reserve(aphi.size() + aggregate_nodes[aggregate_id].size() + 16U);
    for (const auto& entry : aphi) {
        FineNodeColumns values{};
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

std::vector<FineBasisSupport> build_fine_basis_support_cache(
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
    std::vector<FineBasisSupport> supports(space.aggregates.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (std::int64_t a64 = 0; a64 < static_cast<std::int64_t>(supports.size()); ++a64) {
        const std::size_t a = static_cast<std::size_t>(a64);
        supports[a] = build_one_fine_basis_support(
            mesh, material, space, aggregate_nodes, fine_inverse, omega0, a);
    }
    setup_ms = elapsed_ms(start, Clock::now());
    return supports;
}

std::vector<std::vector<std::uint32_t>> build_element_support_index(
    const gfss::StructuredHexMesh& mesh,
    const std::vector<FineBasisSupport>& supports) {
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

LocalColumns apply_local_a1_columns(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const std::vector<FineBasisSupport>& supports,
    const std::vector<std::vector<std::uint32_t>>& element_supports,
    const LocalColumns& input) {
    if (input.cols == 0U || input.cols > kCandidates) {
        throw std::invalid_argument("local A1 column count invalid");
    }
    std::vector<std::size_t> touched_elements;
    for (const auto& entry : input.values) {
        const auto node = static_cast<std::size_t>(entry.first);
        if (node >= supports.size()) throw std::out_of_range("local A1 input node out of range");
        touched_elements.insert(touched_elements.end(),
                                supports[node].energy_elements.begin(),
                                supports[node].energy_elements.end());
    }
    sort_unique(touched_elements);

    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    LocalColumns output;
    output.cols = input.cols;
    for (const auto element : touched_elements) {
        const auto ijk = decode_element(mesh, element);
        const auto nodes = mesh.element_nodes(ijk[0], ijk[1], ijk[2]);
        std::array<double, 24U * kCandidates> u{};
        std::array<double, 24U * kCandidates> f{};

        for (const auto a_u32 : element_supports[element]) {
            const auto in_it = input.values.find(a_u32);
            if (in_it == input.values.end()) continue;
            const auto& basis = supports[a_u32];
            const auto& coeff = in_it->second;
            for (std::size_t n = 0; n < 8U; ++n) {
                const auto p_it = basis.values.find(static_cast<std::size_t>(nodes[n]));
                if (p_it == basis.values.end()) continue;
                for (std::size_t c = 0; c < 3U; ++c) {
                    const std::size_t ldof = 3U * n + c;
                    for (std::size_t col = 0; col < input.cols; ++col) {
                        double value = 0.0;
                        for (std::size_t q = 0; q < basis.rank; ++q) {
                            value += p_it->second[c * kCandidates + q] *
                                     coeff[q * kCandidates + col];
                        }
                        u[ldof * kCandidates + col] += value;
                    }
                }
            }
        }

        for (std::size_t i = 0; i < 24U; ++i) {
            for (std::size_t col = 0; col < input.cols; ++col) {
                double value = 0.0;
                for (std::size_t j = 0; j < 24U; ++j) {
                    value += ke[i][j] * u[j * kCandidates + col];
                }
                f[i * kCandidates + col] = value;
            }
        }

        for (const auto b_u32 : element_supports[element]) {
            const auto& basis = supports[b_u32];
            auto& out = zero_block(output.values, b_u32);
            for (std::size_t n = 0; n < 8U; ++n) {
                const auto p_it = basis.values.find(static_cast<std::size_t>(nodes[n]));
                if (p_it == basis.values.end()) continue;
                for (std::size_t c = 0; c < 3U; ++c) {
                    const std::size_t ldof = 3U * n + c;
                    for (std::size_t q = 0; q < basis.rank; ++q) {
                        const double p = p_it->second[c * kCandidates + q];
                        for (std::size_t col = 0; col < input.cols; ++col) {
                            out[q * kCandidates + col] +=
                                p * f[ldof * kCandidates + col];
                        }
                    }
                }
            }
        }
    }
    return output;
}

LocalColumns solve_local_blocks(const L1BlockMetric& metric,
                                const LocalColumns& rhs) {
    LocalColumns result;
    result.cols = rhs.cols;
    for (const auto& entry : rhs.values) {
        const std::size_t node = entry.first;
        if (node >= metric.nodes()) throw std::out_of_range("local block solve node out of range");
        const std::size_t begin = metric.dof_offsets[node];
        const std::size_t rank = metric.dof_offsets[node + 1U] - begin;
        const double* l = metric.lower.data() + metric.value_offsets[node];
        auto& out = zero_block(result.values, entry.first);
        for (std::size_t col = 0; col < rhs.cols; ++col) {
            std::array<double, kCandidates> y{};
            for (std::size_t i = 0; i < rank; ++i) {
                double value = entry.second[i * kCandidates + col];
                for (std::size_t j = 0; j < i; ++j) value -= l[i * rank + j] * y[j];
                y[i] = value / l[i * rank + i];
            }
            for (std::size_t ii = rank; ii-- > 0U;) {
                double value = y[ii];
                for (std::size_t j = ii + 1U; j < rank; ++j) {
                    value -= l[j * rank + ii] * out[j * kCandidates + col];
                }
                out[ii * kCandidates + col] = value / l[ii * rank + ii];
            }
        }
    }
    return result;
}

void local_axpy(LocalColumns& x, const LocalColumns& y, double alpha) {
    if (x.cols != y.cols) throw std::invalid_argument("local axpy column mismatch");
    for (const auto& entry : y.values) {
        auto& out = zero_block(x.values, entry.first);
        for (std::size_t i = 0; i < kCandidates; ++i) {
            for (std::size_t col = 0; col < x.cols; ++col) {
                out[i * kCandidates + col] +=
                    alpha * entry.second[i * kCandidates + col];
            }
        }
    }
}

std::array<double, 36> local_cross_gram(const LocalColumns& left,
                                        const LocalColumns& right,
                                        const L1BlockMetric& row_metric) {
    std::array<double, 36> gram{};
    for (const auto& entry : left.values) {
        const auto r_it = right.values.find(entry.first);
        if (r_it == right.values.end()) continue;
        const std::size_t node = entry.first;
        const std::size_t rank = row_metric.dof_offsets[node + 1U] -
                                 row_metric.dof_offsets[node];
        for (std::size_t i = 0; i < left.cols; ++i) {
            for (std::size_t j = 0; j < right.cols; ++j) {
                double value = 0.0;
                for (std::size_t r = 0; r < rank; ++r) {
                    value += entry.second[r * kCandidates + i] *
                             r_it->second[r * kCandidates + j];
                }
                gram[i * kCandidates + j] += value;
            }
        }
    }
    return gram;
}

LocalColumns initial_candidate_columns(const CandidateAggregate& aggregate,
                                       const DofLayout& layout) {
    LocalColumns result;
    result.cols = aggregate.rank;
    for (std::size_t row = 0; row < aggregate.fine_dofs.size(); ++row) {
        const std::size_t dof = aggregate.fine_dofs[row];
        const auto node = layout.node_of_dof[dof];
        const std::size_t local = layout.local_of_dof[dof];
        auto& block = zero_block(result.values, node);
        for (std::size_t q = 0; q < aggregate.rank; ++q) {
            block[local * kCandidates + q] =
                aggregate.q_values[row * aggregate.rank + q];
        }
    }
    return result;
}

template <class LocalApply>
std::vector<LocalColumns> build_smoothed_candidate_supports(
    const CandidateTransfer& transfer,
    const AlgebraicNodeGraph& fine_graph,
    const L1BlockMetric& fine_metric,
    double omega,
    std::size_t steps,
    const LocalApply& local_apply) {
    const auto layout = make_dof_layout(fine_graph);
    std::vector<LocalColumns> supports(transfer.aggregates.size());
    for (std::size_t a = 0; a < transfer.aggregates.size(); ++a) {
        auto columns = initial_candidate_columns(transfer.aggregates[a], layout);
        for (std::size_t step = 0; step < steps; ++step) {
            const auto applied = local_apply(columns);
            const auto scaled = solve_local_blocks(fine_metric, applied);
            local_axpy(columns, scaled, -omega);
        }
        supports[a] = std::move(columns);
    }
    return supports;
}

template <class LocalApply>
L1BlockMetric build_metric_from_local_supports(
    const CandidateTransfer& transfer,
    const L1BlockMetric& row_metric,
    const std::vector<LocalColumns>& supports,
    const LocalApply& local_apply) {
    if (supports.size() != transfer.aggregates.size() ||
        transfer.coarse_graph.nodes() != supports.size()) {
        throw std::invalid_argument("local coarse metric support/layout mismatch");
    }
    L1BlockMetric metric;
    metric.dof_offsets = transfer.coarse_graph.dof_offsets;
    metric.value_offsets.resize(supports.size() + 1U, 0U);
    metric.min_rank = std::numeric_limits<std::size_t>::max();
    metric.max_rank = 0U;
    for (std::size_t a = 0; a < supports.size(); ++a) {
        const std::size_t rank = metric.dof_offsets[a + 1U] - metric.dof_offsets[a];
        if (rank == 0U || rank > kCandidates || supports[a].cols != rank) {
            throw std::runtime_error("local coarse metric rank mismatch");
        }
        metric.min_rank = std::min(metric.min_rank, rank);
        metric.max_rank = std::max(metric.max_rank, rank);
        metric.value_offsets[a + 1U] = metric.value_offsets[a] + rank * rank;
    }
    metric.lower.assign(metric.value_offsets.back(), 0.0);
    metric.min_cholesky_pivot = std::numeric_limits<double>::infinity();

    for (std::size_t a = 0; a < supports.size(); ++a) {
        const std::size_t rank = supports[a].cols;
        const auto applied = local_apply(supports[a]);
        const auto block = local_cross_gram(supports[a], applied, row_metric);
        double* l = metric.lower.data() + metric.value_offsets[a];
        for (std::size_t i = 0; i < rank; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double value = 0.5 * (block[i * kCandidates + j] +
                                      block[j * kCandidates + i]);
                for (std::size_t k = 0; k < j; ++k) {
                    value -= l[i * rank + k] * l[j * rank + k];
                }
                if (i == j) {
                    if (!(value > 0.0) || !std::isfinite(value)) {
                        throw std::runtime_error("localized coarse diagonal block lost SPD");
                    }
                    metric.min_cholesky_pivot =
                        std::min(metric.min_cholesky_pivot, value);
                    l[i * rank + j] = std::sqrt(value);
                } else {
                    l[i * rank + j] = value / l[j * rank + j];
                }
            }
        }
    }
    return metric;
}

LocalColumns prolong_l2_to_l1(const LocalColumns& input,
                              const std::vector<LocalColumns>& l2_basis,
                              const L1BlockMetric& l1_metric) {
    LocalColumns result;
    result.cols = input.cols;
    for (const auto& entry : input.values) {
        const std::size_t l2_node = entry.first;
        if (l2_node >= l2_basis.size()) throw std::out_of_range("local A2 input node out of range");
        const auto& basis = l2_basis[l2_node];
        const std::size_t rank2 = basis.cols;
        for (const auto& b_entry : basis.values) {
            const std::size_t l1_node = b_entry.first;
            const std::size_t rank1 = l1_metric.dof_offsets[l1_node + 1U] -
                                      l1_metric.dof_offsets[l1_node];
            auto& out = zero_block(result.values, b_entry.first);
            for (std::size_t r = 0; r < rank1; ++r) {
                for (std::size_t col = 0; col < input.cols; ++col) {
                    double value = 0.0;
                    for (std::size_t q = 0; q < rank2; ++q) {
                        value += b_entry.second[r * kCandidates + q] *
                                 entry.second[q * kCandidates + col];
                    }
                    out[r * kCandidates + col] += value;
                }
            }
        }
    }
    return result;
}

LocalColumns restrict_l1_to_l2(const LocalColumns& input,
                               const std::vector<LocalColumns>& l2_basis,
                               const L1BlockMetric& l1_metric) {
    LocalColumns result;
    result.cols = input.cols;
    for (std::size_t l2_node = 0; l2_node < l2_basis.size(); ++l2_node) {
        const auto& basis = l2_basis[l2_node];
        LocalBlockColumns block{};
        bool touched = false;
        for (const auto& b_entry : basis.values) {
            const auto in_it = input.values.find(b_entry.first);
            if (in_it == input.values.end()) continue;
            touched = true;
            const std::size_t l1_node = b_entry.first;
            const std::size_t rank1 = l1_metric.dof_offsets[l1_node + 1U] -
                                      l1_metric.dof_offsets[l1_node];
            for (std::size_t q = 0; q < basis.cols; ++q) {
                for (std::size_t col = 0; col < input.cols; ++col) {
                    double value = 0.0;
                    for (std::size_t r = 0; r < rank1; ++r) {
                        value += b_entry.second[r * kCandidates + q] *
                                 in_it->second[r * kCandidates + col];
                    }
                    block[q * kCandidates + col] += value;
                }
            }
        }
        if (touched) result.values.emplace(static_cast<std::uint32_t>(l2_node), block);
    }
    return result;
}

template <class LocalA1Apply>
LocalColumns apply_local_a2_columns(const LocalColumns& input,
                                    const std::vector<LocalColumns>& l2_basis,
                                    const L1BlockMetric& l1_metric,
                                    const LocalA1Apply& local_a1_apply) {
    const auto l1 = prolong_l2_to_l1(input, l2_basis, l1_metric);
    const auto a1_l1 = local_a1_apply(l1);
    return restrict_l1_to_l2(a1_l1, l2_basis, l1_metric);
}

LocalBottomReference build_local_bottom(
    const CandidateTransfer& transfer2,
    const L1BlockMetric& l2_metric,
    const std::vector<LocalColumns>& bottom_basis,
    const std::function<LocalColumns(const LocalColumns&)>& local_a2_apply) {
    const auto start = Clock::now();
    const std::size_t n = transfer2.coarse_dofs;
    LocalBottomReference bottom;
    bottom.values.assign(n * n, 0.0);

    for (std::size_t jnode = 0; jnode < bottom_basis.size(); ++jnode) {
        const auto applied = local_a2_apply(bottom_basis[jnode]);
        const std::size_t joff = transfer2.aggregates[jnode].coarse_offset;
        const std::size_t jrank = transfer2.aggregates[jnode].rank;
        for (std::size_t inode = 0; inode < bottom_basis.size(); ++inode) {
            const auto block = local_cross_gram(bottom_basis[inode], applied, l2_metric);
            const std::size_t ioff = transfer2.aggregates[inode].coarse_offset;
            const std::size_t irank = transfer2.aggregates[inode].rank;
            for (std::size_t i = 0; i < irank; ++i) {
                for (std::size_t j = 0; j < jrank; ++j) {
                    bottom.values[(ioff + i) * n + (joff + j)] =
                        block[i * kCandidates + j];
                }
            }
        }
    }

    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        norm2 += bottom.values[i * n + i] * bottom.values[i * n + i];
        for (std::size_t j = i + 1U; j < n; ++j) {
            const double aij = bottom.values[i * n + j];
            const double aji = bottom.values[j * n + i];
            const double d = aij - aji;
            asym2 += 2.0 * d * d;
            norm2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            bottom.values[i * n + j] = sym;
            bottom.values[j * n + i] = sym;
        }
    }

    bottom.factor.n = n;
    bottom.factor.symmetry_relative_defect =
        norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;
    bottom.factor.lower.assign(n * n, 0.0);
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
                    throw std::runtime_error("localized bottom Cholesky lost SPD");
                }
                bottom.factor.min_pivot = std::min(bottom.factor.min_pivot, value);
                bottom.factor.lower[i * n + j] = std::sqrt(value);
            } else {
                bottom.factor.lower[i * n + j] =
                    value / bottom.factor.lower[j * n + j];
            }
        }
    }
    bottom.assembly_ms = elapsed_ms(start, Clock::now());
    return bottom;
}

Vec apply_dense_bottom(const LocalBottomReference& bottom, const Vec& x) {
    if (x.size() != bottom.factor.n) throw std::invalid_argument("dense bottom apply size mismatch");
    Vec y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double* row = bottom.values.data() + i * x.size();
        for (std::size_t j = 0; j < x.size(); ++j) y[i] += row[j] * x[j];
    }
    return y;
}

double bottom_local_oracle_error(const LocalBottomReference& bottom,
                                 const Apply& nested_apply) {
    const auto x = deterministic_actual_a2_probe(bottom.factor.n, 0.91);
    const auto yl = apply_dense_bottom(bottom, x);
    const auto yn = nested_apply(x);
    Vec diff(yl.size(), 0.0);
    for (std::size_t i = 0; i < yl.size(); ++i) diff[i] = yl[i] - yn[i];
    return norm(diff) / std::max(norm(yn), 1.0e-300);
}

void run_local_l2_reference(std::size_t m0,
                            std::size_t m1,
                            std::size_t m2,
                            std::size_t max_cycles,
                            std::size_t target_nodes,
                            std::size_t min_nodes) {
    if (m0 != 1U) {
        throw std::invalid_argument("localized recursive reference currently requires m0=1");
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

    const auto graph1 = graph_from_variable_blocks(tentative_a1);
    const auto candidates1 = make_level1_candidates(space0);
    const auto tentative_transfer1 = build_candidate_transfer(
        graph1, candidates1, target_nodes, min_nodes, 1.0e-10);
    const double tentative_candidate1_error =
        candidate_reproduction_error(tentative_transfer1, candidates1);

    const auto block1 = build_exact_l1_block_metric(
        mesh, material, space0, graph1, fine_inverse, omega0);
    const double block1_oracle_error = audit_l1_block_metric(block1, apply1);
    const auto actual_inverse1 = block1.inverse_scalar_diagonal();
    const double block_lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
    const double block_omega1 = kSaDampingNumerator / block_lambda1;
    const L1BlockSmoothedTransfer transfer1{
        tentative_transfer1, apply1, block1, block_omega1, m1};
    const double transfer1_adjoint = l1_block_transfer_adjoint_error(transfer1);
    const double smoothed_candidate1_error = l1_block_candidate_error(transfer1, candidates1);

    double p0_support_cache_ms = 0.0;
    const auto fine_supports = build_fine_basis_support_cache(
        mesh, material, space0, fine_inverse, omega0, p0_support_cache_ms);
    const auto element_supports = build_element_support_index(mesh, fine_supports);
    const auto local_a1_apply = [&](const LocalColumns& x) {
        return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
    };

    const auto l2_setup_start = Clock::now();
    const auto l2_basis = build_smoothed_candidate_supports(
        tentative_transfer1, graph1, block1, block_omega1, m1, local_a1_apply);
    const auto block2 = build_metric_from_local_supports(
        tentative_transfer1, block1, l2_basis, local_a1_apply);
    const double l2_local_setup_ms = elapsed_ms(l2_setup_start, Clock::now());

    const Apply apply2 = [&](const Vec& x) {
        const auto l1 = transfer1.prolong(x);
        return transfer1.restrict_transpose(apply1(l1));
    };
    const double block2_oracle_error = audit_l1_block_metric(block2, apply2);
    const auto propagated_inverse2 =
        tentative_transfer1.approximate_inverse_coarse_diagonal(actual_inverse1);
    const auto actual_inverse2 = block2.inverse_scalar_diagonal();
    const double propagated_lambda2 = estimate_lambda_max(apply2, propagated_inverse2, 8U);
    const double actual_scalar_lambda2 = estimate_lambda_max(apply2, actual_inverse2, 8U);
    const double block_lambda2 = estimate_lambda_max_l1_block(apply2, block2, 8U);
    const double block_omega2 = kSaDampingNumerator / block_lambda2;

    const auto tentative_transfer2 = build_candidate_transfer(
        tentative_transfer1.coarse_graph,
        tentative_transfer1.coarse_candidates,
        target_nodes,
        min_nodes,
        1.0e-10);
    const L1BlockSmoothedTransfer transfer2{
        tentative_transfer2, apply2, block2, block_omega2, m2};
    const double transfer2_adjoint = l1_block_transfer_adjoint_error(transfer2);
    const double smoothed_candidate2_error = l1_block_candidate_error(
        transfer2, tentative_transfer1.coarse_candidates);

    const auto local_a2_apply_lambda = [&](const LocalColumns& x) {
        return apply_local_a2_columns(x, l2_basis, block1, local_a1_apply);
    };
    const std::function<LocalColumns(const LocalColumns&)> local_a2_apply =
        local_a2_apply_lambda;

    const auto bottom_basis = build_smoothed_candidate_supports(
        tentative_transfer2,
        tentative_transfer1.coarse_graph,
        block2,
        block_omega2,
        m2,
        local_a2_apply_lambda);
    const auto bottom = build_local_bottom(
        tentative_transfer2, block2, bottom_basis, local_a2_apply);

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
    levels[1].label = "L1_local_exact_block_Jacobi";
    levels[1].diagnostic_lambda_max = block_lambda1;
    levels[1].apply = apply1;
    levels[1].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply1, block1, block_lambda1, b, x, degree);
    };
    levels[1].restrict_to_coarse = [&](const Vec& r) { return transfer1.restrict_transpose(r); };
    levels[1].prolong_from_coarse = [&](const Vec& c) { return transfer1.prolong(c); };

    levels[2].dofs = tentative_transfer1.coarse_dofs;
    levels[2].label = "L2_local_exact_block_Jacobi";
    levels[2].diagnostic_lambda_max = block_lambda2;
    levels[2].apply = apply2;
    levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply2, block2, block_lambda2, b, x, degree);
    };
    levels[2].restrict_to_coarse = [&](const Vec& r) { return transfer2.restrict_transpose(r); };
    levels[2].prolong_from_coarse = [&](const Vec& c) { return transfer2.prolong(c); };

    levels[3].dofs = tentative_transfer2.coarse_dofs;
    levels[3].label = "L3_local_dense_bottom";
    levels[3].apply = [&](const Vec& x) { return apply_dense_bottom(bottom, x); };
    levels[3].bottom_solve = [&](const Vec& b) { return bottom.factor.solve(b); };

    const auto rhs = make_rhs(mesh);
    const auto result = gfss::solve_reference_multilevel_vcycle(
        levels, rhs, 1.0e-6, max_cycles, 3U, 3U);

    std::cout << "\n========================================\n"
              << "L0_transfer_smoothing_steps=" << m0
              << " L1_transfer_smoothing_steps=" << m1
              << " L2_transfer_smoothing_steps=" << m2 << '\n'
              << "hierarchy_levels=4\n"
              << "L1_metric=localized_exact_actual_block\n"
              << "L2_metric=localized_exact_actual_block\n"
              << "L2_graph=tentative_candidate_graph_control\n"
              << "bottom_operator=localized_exact_recursive_energy\n"
              << "dense_A1_materialized=false dense_A2_materialized=false\n"
              << "L0_dofs=" << levels[0].dofs
              << " L1_dofs=" << levels[1].dofs
              << " L1_nodes=" << graph1.nodes()
              << " L2_dofs=" << levels[2].dofs
              << " L2_nodes=" << tentative_transfer1.coarse_graph.nodes()
              << " L3_dofs=" << levels[3].dofs
              << " L3_nodes=" << tentative_transfer2.coarse_graph.nodes() << '\n'
              << std::scientific << std::setprecision(9)
              << "L1_tentative_candidate_reproduction_error=" << tentative_candidate1_error
              << " L1_block_smoothed_candidate_reproduction_error=" << smoothed_candidate1_error
              << " L1_transfer_adjoint_relative_error=" << transfer1_adjoint << '\n'
              << "L2_block_smoothed_candidate_reproduction_error=" << smoothed_candidate2_error
              << " L2_transfer_adjoint_relative_error=" << transfer2_adjoint << '\n'
              << "L1_local_block_vs_nested_oracle_relative_error=" << block1_oracle_error
              << " L2_local_block_vs_nested_oracle_relative_error=" << block2_oracle_error
              << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error << '\n'
              << std::fixed << std::setprecision(6)
              << "P0_smoothed_support_cache_ms=" << p0_support_cache_ms
              << " L1_block_setup_ms=" << block1.setup_ms
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
              << " propagated_lambda2_reference=" << propagated_lambda2
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

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t m0 = argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 1U;
        const std::size_t m1 = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 2U;
        const std::size_t m2 = argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 1U;
        const std::size_t max_cycles = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 6U;
        const std::size_t target_nodes = argc > 5 ? static_cast<std::size_t>(std::stoull(argv[5])) : 12U;
        const std::size_t min_nodes = argc > 6 ? static_cast<std::size_t>(std::stoull(argv[6])) : 4U;
        if (m0 != 1U || m1 > 4U || m2 > 4U || max_cycles == 0U ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid localized recursive SA options");
        }
        std::cout << "GFSS M5 fully localized recursive SA numerical reference\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "purpose=actual_L1_L2_metrics_and_bottom_without_dense_coarse_materialization\n"
                  << "reference_execution=cpu_fp64\n"
                  << "L0_transfer_smoothing_steps_required=1\n"
                  << "L1_operator_for_local_recursion=exact_element_energy_of_cached_smoothed_P0\n"
                  << "L1_full_matrix=not_materialized\n"
                  << "L2_full_matrix=not_materialized\n"
                  << "bottom_full_operator=assembled_directly_from_local_recursive_energy\n"
                  << "all_runtime_transfer_restrictions=exact_transpose\n"
                  << "pre_smooth_degree=3 post_smooth_degree=3\n"
                  << "acceptance_target_post_transient_q<=0.4\n"
                  << "performance_status=numerical_reference_only\n";
        run_local_l2_reference(m0, m1, m2, max_cycles, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_local_l2_reference_bench "
                  << "[m0=1 [m1 [m2 [max_cycles [target_nodes [min_nodes]]]]]]\n";
        return 1;
    }
}
