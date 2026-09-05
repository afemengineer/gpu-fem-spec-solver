// Reference-only reusable localized recursive-SA helpers.
// This mirrors the validated local-L2/bottom machinery from
// recursive_sa_local_l2_reference_bench.cpp without its standalone driver.
#define main gfss_recursive_sa_l1_block_local_l2_helpers_only
#include "recursive_sa_l1_block_jacobi_reference_bench.cpp"
#undef main

namespace {

using FineNodeColumns = std::array<double, 18>;
using LocalBlockColumns = std::array<double, 36>;

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

}  // namespace
