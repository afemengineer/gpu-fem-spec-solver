#include "gfss/aggregation_coarse_space.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gfss {
namespace {

constexpr std::uint32_t kUnassigned = std::numeric_limits<std::uint32_t>::max();

void validate_graph(const NodalGraph3D& graph) {
    const std::size_t n = graph.coordinates.size();
    if (n == 0U) {
        throw std::invalid_argument("aggregation graph has no nodes");
    }
    if (graph.constrained.size() != n) {
        throw std::invalid_argument("aggregation constraint vector size mismatch");
    }
    if (graph.row_offsets.size() != n + 1U) {
        throw std::invalid_argument("aggregation CSR row_offsets size mismatch");
    }
    if (graph.row_offsets.front() != 0U ||
        graph.row_offsets.back() != graph.column_indices.size()) {
        throw std::invalid_argument("aggregation CSR offsets are inconsistent");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (graph.row_offsets[i] > graph.row_offsets[i + 1U]) {
            throw std::invalid_argument("aggregation CSR row offsets are not monotone");
        }
    }
    for (const auto j : graph.column_indices) {
        if (j >= n) {
            throw std::invalid_argument("aggregation CSR column index out of range");
        }
    }
}

std::array<double, 18> rigid_rows(const std::array<double, 3>& xyz,
                                  const ElasticityAggregateInfo& aggregate) {
    const double scale = aggregate.coordinate_scale > 0.0
        ? aggregate.coordinate_scale
        : 1.0;
    const double x = (xyz[0] - aggregate.centroid[0]) / scale;
    const double y = (xyz[1] - aggregate.centroid[1]) / scale;
    const double z = (xyz[2] - aggregate.centroid[2]) / scale;

    // Rows are the x/y/z displacement components. Rotations use omega x r.
    return {
        1.0, 0.0, 0.0,  0.0,  z,   -y,
        0.0, 1.0, 0.0, -z,    0.0,  x,
        0.0, 0.0, 1.0,  y,   -x,    0.0,
    };
}

double metric_dot(const std::array<double, 36>& gram,
                  const std::array<double, 6>& a,
                  const std::array<double, 6>& b) {
    double value = 0.0;
    for (std::size_t i = 0; i < 6U; ++i) {
        double row = 0.0;
        for (std::size_t j = 0; j < 6U; ++j) {
            row += gram[6U * i + j] * b[j];
        }
        value += a[i] * row;
    }
    return value;
}

void build_rigid_transform(const NodalGraph3D& graph,
                           const std::vector<std::uint32_t>& nodes,
                           double rank_tolerance,
                           ElasticityAggregateInfo& aggregate) {
    if (nodes.empty()) {
        throw std::runtime_error("aggregation produced an empty aggregate");
    }

    aggregate.node_count = nodes.size();
    aggregate.centroid = {0.0, 0.0, 0.0};
    for (const auto node : nodes) {
        const auto& p = graph.coordinates[node];
        aggregate.centroid[0] += p[0];
        aggregate.centroid[1] += p[1];
        aggregate.centroid[2] += p[2];
    }
    const double inv_count = 1.0 / static_cast<double>(nodes.size());
    for (double& v : aggregate.centroid) v *= inv_count;

    double radius2 = 0.0;
    for (const auto node : nodes) {
        const auto& p = graph.coordinates[node];
        const double dx = p[0] - aggregate.centroid[0];
        const double dy = p[1] - aggregate.centroid[1];
        const double dz = p[2] - aggregate.centroid[2];
        radius2 += dx * dx + dy * dy + dz * dz;
    }
    aggregate.coordinate_scale = std::sqrt(radius2 * inv_count);
    if (!(aggregate.coordinate_scale > 0.0) ||
        !std::isfinite(aggregate.coordinate_scale)) {
        aggregate.coordinate_scale = 1.0;
    }

    std::array<double, 36> gram{};
    for (const auto node : nodes) {
        const auto rows = rigid_rows(graph.coordinates[node], aggregate);
        for (std::size_t component = 0; component < 3U; ++component) {
            const double* row = rows.data() + 6U * component;
            for (std::size_t i = 0; i < 6U; ++i) {
                for (std::size_t j = 0; j < 6U; ++j) {
                    gram[6U * i + j] += row[i] * row[j];
                }
            }
        }
    }

    double trace = 0.0;
    for (std::size_t i = 0; i < 6U; ++i) trace += gram[6U * i + i];
    const double threshold = std::max(
        1.0e-30,
        rank_tolerance * rank_tolerance * std::max(1.0, trace));

    std::array<std::array<double, 6>, 6> basis{};
    std::size_t rank = 0U;
    for (std::size_t candidate = 0; candidate < 6U; ++candidate) {
        std::array<double, 6> v{};
        v[candidate] = 1.0;

        for (std::size_t q = 0; q < rank; ++q) {
            const double projection = metric_dot(gram, basis[q], v);
            for (std::size_t j = 0; j < 6U; ++j) {
                v[j] -= projection * basis[q][j];
            }
        }

        // Re-orthogonalize once; 6x6 setup cost is negligible and this makes
        // the reference robust for flat/slender aggregates.
        for (std::size_t q = 0; q < rank; ++q) {
            const double projection = metric_dot(gram, basis[q], v);
            for (std::size_t j = 0; j < 6U; ++j) {
                v[j] -= projection * basis[q][j];
            }
        }

        const double norm2 = metric_dot(gram, v, v);
        if (!(norm2 > threshold) || !std::isfinite(norm2)) {
            continue;
        }
        const double inv_norm = 1.0 / std::sqrt(norm2);
        for (double& value : v) value *= inv_norm;
        basis[rank++] = v;
    }

    if (rank < 3U) {
        throw std::runtime_error(
            "elasticity aggregate failed to retain three translation modes");
    }

    aggregate.rank = rank;
    aggregate.rigid_transform.fill(0.0);
    for (std::size_t q = 0; q < rank; ++q) {
        for (std::size_t j = 0; j < 6U; ++j) {
            aggregate.rigid_transform[6U * q + j] = basis[q][j];
        }
    }
}

double tentative_basis_value(const ElasticityAggregationCoarseSpace& space,
                             std::size_t node,
                             std::size_t component,
                             std::size_t local_coarse) {
    const auto aggregate_id = space.aggregate_of_node[node];
    if (aggregate_id == kUnassigned) return 0.0;
    const auto& aggregate = space.aggregates[aggregate_id];
    const auto rows = rigid_rows(space.graph.coordinates[node], aggregate);
    const double* row = rows.data() + 6U * component;
    const double* transform =
        aggregate.rigid_transform.data() + 6U * local_coarse;
    double value = 0.0;
    for (std::size_t j = 0; j < 6U; ++j) {
        value += row[j] * transform[j];
    }
    return value;
}

std::vector<double> make_global_rigid_mode(
    const ElasticityAggregationCoarseSpace& space,
    std::size_t mode) {
    std::vector<double> v(3U * space.graph.coordinates.size(), 0.0);
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        if (space.graph.constrained[node] != 0U) continue;
        const auto& p = space.graph.coordinates[node];
        const std::size_t base = 3U * node;
        switch (mode) {
            case 0U: v[base + 0U] = 1.0; break;
            case 1U: v[base + 1U] = 1.0; break;
            case 2U: v[base + 2U] = 1.0; break;
            case 3U:
                v[base + 1U] = -p[2];
                v[base + 2U] = p[1];
                break;
            case 4U:
                v[base + 0U] = p[2];
                v[base + 2U] = -p[0];
                break;
            case 5U:
                v[base + 0U] = -p[1];
                v[base + 1U] = p[0];
                break;
            default:
                throw std::invalid_argument("rigid-body mode index out of range");
        }
    }
    return v;
}

double vector_norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (const double x : v) sum += x * x;
    return std::sqrt(sum);
}

}  // namespace

NodalGraph3D build_structured_hex_nodal_graph_x0(
    const StructuredHexMesh& mesh) {
    if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
        throw std::invalid_argument("structured aggregation graph requires non-empty mesh");
    }

    NodalGraph3D graph;
    const std::size_t nodes = static_cast<std::size_t>(mesh.node_count());
    graph.coordinates.resize(nodes);
    graph.constrained.assign(nodes, 0U);
    graph.row_offsets.resize(nodes + 1U, 0U);

    const double hx = mesh.lx / static_cast<double>(mesh.nx);
    const double hy = mesh.ly / static_cast<double>(mesh.ny);
    const double hz = mesh.lz / static_cast<double>(mesh.nz);

    std::size_t cursor = 0U;
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            for (std::uint32_t i = 0; i <= mesh.nx; ++i) {
                const auto node = static_cast<std::size_t>(mesh.node_index(i, j, k));
                graph.coordinates[node] = {
                    hx * static_cast<double>(i),
                    hy * static_cast<double>(j),
                    hz * static_cast<double>(k)};
                graph.constrained[node] = i == 0U ? 1U : 0U;
                graph.row_offsets[node] = static_cast<std::uint32_t>(cursor);

                for (int dz = -1; dz <= 1; ++dz) {
                    const int nk = static_cast<int>(k) + dz;
                    if (nk < 0 || nk > static_cast<int>(mesh.nz)) continue;
                    for (int dy = -1; dy <= 1; ++dy) {
                        const int nj = static_cast<int>(j) + dy;
                        if (nj < 0 || nj > static_cast<int>(mesh.ny)) continue;
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int ni = static_cast<int>(i) + dx;
                            if (ni < 0 || ni > static_cast<int>(mesh.nx)) continue;
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            graph.column_indices.push_back(static_cast<std::uint32_t>(
                                mesh.node_index(static_cast<std::uint32_t>(ni),
                                                static_cast<std::uint32_t>(nj),
                                                static_cast<std::uint32_t>(nk))));
                            ++cursor;
                        }
                    }
                }
            }
        }
    }
    graph.row_offsets[nodes] = static_cast<std::uint32_t>(cursor);
    return graph;
}

ElasticityAggregationCoarseSpace build_elasticity_aggregation_coarse_space(
    NodalGraph3D graph,
    const ElasticityAggregationOptions& options) {
    validate_graph(graph);
    if (options.target_nodes_per_aggregate < 2U) {
        throw std::invalid_argument("aggregation target size must be >= 2");
    }
    if (options.min_nodes_per_aggregate == 0U ||
        options.min_nodes_per_aggregate > options.target_nodes_per_aggregate) {
        throw std::invalid_argument("aggregation minimum size must be in [1,target]");
    }
    if (!(options.rank_tolerance > 0.0)) {
        throw std::invalid_argument("aggregation rank tolerance must be positive");
    }

    const std::size_t n = graph.coordinates.size();
    std::vector<std::uint32_t> aggregate_of_node(n, kUnassigned);
    std::vector<std::vector<std::uint32_t>> aggregate_nodes;

    for (std::uint32_t seed = 0U; seed < n; ++seed) {
        if (graph.constrained[seed] != 0U ||
            aggregate_of_node[seed] != kUnassigned) {
            continue;
        }

        const auto aggregate_id = static_cast<std::uint32_t>(aggregate_nodes.size());
        aggregate_nodes.emplace_back();
        auto& nodes = aggregate_nodes.back();
        std::queue<std::uint32_t> frontier;
        aggregate_of_node[seed] = aggregate_id;
        nodes.push_back(seed);
        frontier.push(seed);

        while (!frontier.empty() &&
               nodes.size() < options.target_nodes_per_aggregate) {
            const auto u = frontier.front();
            frontier.pop();
            for (std::uint32_t p = graph.row_offsets[u];
                 p < graph.row_offsets[u + 1U] &&
                 nodes.size() < options.target_nodes_per_aggregate;
                 ++p) {
                const auto v = graph.column_indices[p];
                if (graph.constrained[v] != 0U ||
                    aggregate_of_node[v] != kUnassigned) {
                    continue;
                }
                aggregate_of_node[v] = aggregate_id;
                nodes.push_back(v);
                frontier.push(v);
            }
        }
    }

    // Merge undersized leftovers into the neighboring aggregate with the most
    // graph connections. This keeps tentative rigid-body bases away from tiny
    // rank-deficient islands without requiring geometric remeshing.
    std::vector<std::uint8_t> active(aggregate_nodes.size(), 1U);
    for (std::size_t a = 0; a < aggregate_nodes.size(); ++a) {
        if (!active[a] ||
            aggregate_nodes[a].size() >= options.min_nodes_per_aggregate) {
            continue;
        }
        std::vector<std::size_t> edge_counts(aggregate_nodes.size(), 0U);
        for (const auto u : aggregate_nodes[a]) {
            for (std::uint32_t p = graph.row_offsets[u]; p < graph.row_offsets[u + 1U]; ++p) {
                const auto v = graph.column_indices[p];
                const auto other = aggregate_of_node[v];
                if (other != kUnassigned && other != a && active[other]) {
                    ++edge_counts[other];
                }
            }
        }
        std::size_t best = aggregate_nodes.size();
        std::size_t best_edges = 0U;
        for (std::size_t other = 0; other < aggregate_nodes.size(); ++other) {
            if (active[other] && edge_counts[other] > best_edges) {
                best = other;
                best_edges = edge_counts[other];
            }
        }
        if (best == aggregate_nodes.size()) continue;
        for (const auto node : aggregate_nodes[a]) {
            aggregate_of_node[node] = static_cast<std::uint32_t>(best);
            aggregate_nodes[best].push_back(node);
        }
        aggregate_nodes[a].clear();
        active[a] = 0U;
    }

    std::vector<std::uint32_t> compact(aggregate_nodes.size(), kUnassigned);
    std::vector<std::vector<std::uint32_t>> compact_nodes;
    for (std::size_t a = 0; a < aggregate_nodes.size(); ++a) {
        if (!active[a] || aggregate_nodes[a].empty()) continue;
        compact[a] = static_cast<std::uint32_t>(compact_nodes.size());
        compact_nodes.push_back(std::move(aggregate_nodes[a]));
    }
    for (std::size_t node = 0; node < n; ++node) {
        if (aggregate_of_node[node] != kUnassigned) {
            aggregate_of_node[node] = compact[aggregate_of_node[node]];
        }
    }

    ElasticityAggregationCoarseSpace space;
    space.graph = std::move(graph);
    space.aggregate_of_node = std::move(aggregate_of_node);
    space.aggregates.resize(compact_nodes.size());

    std::size_t coarse_offset = 0U;
    for (std::size_t a = 0; a < compact_nodes.size(); ++a) {
        auto& info = space.aggregates[a];
        build_rigid_transform(space.graph,
                              compact_nodes[a],
                              options.rank_tolerance,
                              info);
        info.coarse_offset = coarse_offset;
        coarse_offset += info.rank;
        space.tentative_p_nnz += 3U * info.rank * info.node_count;
    }
    space.coarse_dofs = coarse_offset;

    for (std::size_t node = 0; node < n; ++node) {
        if (space.graph.constrained[node] == 0U) ++space.free_nodes;
    }
    space.fine_free_dofs = 3U * space.free_nodes;

    // Production-oriented logical payload, excluding coordinates/CSR already
    // owned by the mesh: uint32 aggregate id per node plus per-aggregate
    // FP32 centroid(3), scale(1), 6x6 transform, coarse offset, rank and count.
    constexpr std::size_t per_aggregate_payload =
        (3U + 1U + 36U) * sizeof(float) +
        3U * sizeof(std::uint32_t);
    space.estimated_matrix_free_transfer_payload_bytes =
        n * sizeof(std::uint32_t) +
        space.aggregates.size() * per_aggregate_payload;

    return space;
}

std::vector<double> apply_elasticity_tentative_prolongation(
    const ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& coarse) {
    if (coarse.size() != space.coarse_dofs) {
        throw std::invalid_argument("tentative prolongation coarse size mismatch");
    }
    std::vector<double> fine(3U * space.graph.coordinates.size(), 0.0);
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        const auto aggregate_id = space.aggregate_of_node[node];
        if (aggregate_id == kUnassigned) continue;
        const auto& aggregate = space.aggregates[aggregate_id];
        const std::size_t base = 3U * node;
        for (std::size_t component = 0; component < 3U; ++component) {
            double value = 0.0;
            for (std::size_t q = 0; q < aggregate.rank; ++q) {
                value += tentative_basis_value(space, node, component, q) *
                         coarse[aggregate.coarse_offset + q];
            }
            fine[base + component] = value;
        }
    }
    return fine;
}

std::vector<double> apply_elasticity_tentative_restriction(
    const ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& fine) {
    if (fine.size() != 3U * space.graph.coordinates.size()) {
        throw std::invalid_argument("tentative restriction fine size mismatch");
    }
    std::vector<double> coarse(space.coarse_dofs, 0.0);
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        const auto aggregate_id = space.aggregate_of_node[node];
        if (aggregate_id == kUnassigned) continue;
        const auto& aggregate = space.aggregates[aggregate_id];
        const std::size_t base = 3U * node;
        for (std::size_t q = 0; q < aggregate.rank; ++q) {
            double value = 0.0;
            for (std::size_t component = 0; component < 3U; ++component) {
                value += tentative_basis_value(space, node, component, q) *
                         fine[base + component];
            }
            coarse[aggregate.coarse_offset + q] += value;
        }
    }
    return coarse;
}

double audit_elasticity_rigid_body_reproduction(
    const ElasticityAggregationCoarseSpace& space) {
    double worst = 0.0;
    for (std::size_t mode = 0; mode < 6U; ++mode) {
        const auto exact = make_global_rigid_mode(space, mode);
        const double exact_norm = vector_norm(exact);
        if (!(exact_norm > 0.0)) continue;
        const auto coarse = apply_elasticity_tentative_restriction(space, exact);
        const auto projected = apply_elasticity_tentative_prolongation(space, coarse);
        std::vector<double> diff(exact.size(), 0.0);
        for (std::size_t i = 0; i < exact.size(); ++i) {
            diff[i] = exact[i] - projected[i];
        }
        worst = std::max(worst, vector_norm(diff) / exact_norm);
    }
    return worst;
}

}  // namespace gfss
