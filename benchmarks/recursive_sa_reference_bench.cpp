#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/aggregation_two_grid_reference.hpp"
#include "gfss/cpu_elasticity.hpp"
#include "gfss/hex8.hpp"
#include "gfss/multilevel_reference.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Vec = std::vector<double>;
using Apply = std::function<Vec(const Vec&)>;

constexpr std::size_t kCandidates = 6U;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kChebyshevLowerFraction = 0.10;
constexpr double kLambdaSafety = 1.25;
constexpr double kSaDampingNumerator = 4.0 / 3.0;
constexpr std::uint32_t kUnassigned = std::numeric_limits<std::uint32_t>::max();

struct AlgebraicNodeGraph {
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
    std::vector<std::size_t> dof_offsets;

    std::size_t nodes() const {
        return dof_offsets.empty() ? 0U : dof_offsets.size() - 1U;
    }
    std::size_t dofs() const {
        return dof_offsets.empty() ? 0U : dof_offsets.back();
    }
};

struct CandidateAggregate {
    std::vector<std::uint32_t> nodes;
    std::vector<std::size_t> fine_dofs;
    std::size_t coarse_offset{0};
    std::size_t rank{0};
    std::vector<double> q_values;
};

struct CandidateTransfer {
    std::size_t fine_dofs{0};
    std::size_t coarse_dofs{0};
    std::vector<CandidateAggregate> aggregates;
    std::vector<double> coarse_candidates;
    AlgebraicNodeGraph coarse_graph;

    Vec prolong(const Vec& coarse) const {
        if (coarse.size() != coarse_dofs) {
            throw std::invalid_argument("recursive SA prolongation coarse size mismatch");
        }
        Vec fine(fine_dofs, 0.0);
        for (const auto& aggregate : aggregates) {
            for (std::size_t row = 0; row < aggregate.fine_dofs.size(); ++row) {
                double value = 0.0;
                for (std::size_t q = 0; q < aggregate.rank; ++q) {
                    value += aggregate.q_values[row * aggregate.rank + q] *
                             coarse[aggregate.coarse_offset + q];
                }
                fine[aggregate.fine_dofs[row]] = value;
            }
        }
        return fine;
    }

    Vec restrict_transpose(const Vec& fine) const {
        if (fine.size() != fine_dofs) {
            throw std::invalid_argument("recursive SA restriction fine size mismatch");
        }
        Vec coarse(coarse_dofs, 0.0);
        for (const auto& aggregate : aggregates) {
            for (std::size_t q = 0; q < aggregate.rank; ++q) {
                double value = 0.0;
                for (std::size_t row = 0; row < aggregate.fine_dofs.size(); ++row) {
                    value += aggregate.q_values[row * aggregate.rank + q] *
                             fine[aggregate.fine_dofs[row]];
                }
                coarse[aggregate.coarse_offset + q] = value;
            }
        }
        return coarse;
    }

    std::vector<double> approximate_inverse_coarse_diagonal(
        const std::vector<double>& fine_inverse_diagonal) const {
        if (fine_inverse_diagonal.size() != fine_dofs) {
            throw std::invalid_argument("recursive SA diagonal propagation size mismatch");
        }
        std::vector<double> diagonal(coarse_dofs, 0.0);
        for (const auto& aggregate : aggregates) {
            for (std::size_t row = 0; row < aggregate.fine_dofs.size(); ++row) {
                const std::size_t fine_dof = aggregate.fine_dofs[row];
                const double inv = fine_inverse_diagonal[fine_dof];
                if (!(inv > 0.0) || !std::isfinite(inv)) continue;
                const double d = 1.0 / inv;
                for (std::size_t q = 0; q < aggregate.rank; ++q) {
                    const double v = aggregate.q_values[row * aggregate.rank + q];
                    diagonal[aggregate.coarse_offset + q] += v * v * d;
                }
            }
        }
        std::vector<double> inverse(diagonal.size(), 0.0);
        for (std::size_t i = 0; i < diagonal.size(); ++i) {
            if (!(diagonal[i] > 0.0) || !std::isfinite(diagonal[i])) {
                throw std::runtime_error("recursive SA propagated coarse diagonal invalid");
            }
            inverse[i] = 1.0 / diagonal[i];
        }
        return inverse;
    }
};

struct FineSmoothedTransfer {
    const gfss::StructuredHexMesh& mesh;
    const gfss::Material& material;
    const gfss::ElasticityAggregationCoarseSpace& space;
    const std::vector<double>& inverse_diagonal;
    double omega{0.0};
    std::size_t steps{0};

    Vec apply_fine(const Vec& x) const;

    Vec prolong(const Vec& coarse) const {
        auto fine = gfss::apply_elasticity_tentative_prolongation(space, coarse);
        for (std::size_t step = 0; step < steps; ++step) {
            const auto af = apply_fine(fine);
            for (std::size_t i = 0; i < fine.size(); ++i) {
                if (inverse_diagonal[i] > 0.0) {
                    fine[i] -= omega * inverse_diagonal[i] * af[i];
                }
            }
            for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
                for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
                    const auto node = static_cast<std::size_t>(mesh.node_index(0U, j, k));
                    fine[3U * node + 0U] = 0.0;
                    fine[3U * node + 1U] = 0.0;
                    fine[3U * node + 2U] = 0.0;
                }
            }
        }
        return fine;
    }

    Vec restrict_transpose(const Vec& fine) const {
        if (fine.size() != inverse_diagonal.size()) {
            throw std::invalid_argument("recursive SA fine restriction size mismatch");
        }
        auto work = fine;
        for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
            for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
                const auto node = static_cast<std::size_t>(mesh.node_index(0U, j, k));
                work[3U * node + 0U] = 0.0;
                work[3U * node + 1U] = 0.0;
                work[3U * node + 2U] = 0.0;
            }
        }
        Vec scaled(work.size(), 0.0);
        for (std::size_t step = 0; step < steps; ++step) {
            for (std::size_t i = 0; i < work.size(); ++i) {
                scaled[i] = inverse_diagonal[i] * work[i];
            }
            const auto a_scaled = apply_fine(scaled);
            for (std::size_t i = 0; i < work.size(); ++i) {
                work[i] -= omega * a_scaled[i];
            }
            for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
                for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
                    const auto node = static_cast<std::size_t>(mesh.node_index(0U, j, k));
                    work[3U * node + 0U] = 0.0;
                    work[3U * node + 1U] = 0.0;
                    work[3U * node + 2U] = 0.0;
                }
            }
        }
        return gfss::apply_elasticity_tentative_restriction(space, work);
    }
};

struct DenseCholesky {
    std::size_t n{0};
    std::vector<double> lower;
    double symmetry_relative_defect{0.0};
    double min_pivot{0.0};

    Vec solve(const Vec& b) const {
        if (b.size() != n) throw std::invalid_argument("bottom solve size mismatch");
        Vec y(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            double value = b[i];
            for (std::size_t j = 0; j < i; ++j) {
                value -= lower[i * n + j] * y[j];
            }
            y[i] = value / lower[i * n + i];
        }
        Vec x(n, 0.0);
        for (std::size_t ii = n; ii-- > 0U;) {
            double value = y[ii];
            for (std::size_t j = ii + 1U; j < n; ++j) {
                value -= lower[j * n + ii] * x[j];
            }
            x[ii] = value / lower[ii * n + ii];
        }
        return x;
    }
};

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double dot(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("recursive SA dot size mismatch");
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}

double norm(const Vec& v) {
    return std::sqrt(std::max(0.0, dot(v, v)));
}

void clamp_x0(const gfss::StructuredHexMesh& mesh, Vec& v) {
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = static_cast<std::size_t>(mesh.node_index(0U, j, k));
            v[3U * node + 0U] = 0.0;
            v[3U * node + 1U] = 0.0;
            v[3U * node + 2U] = 0.0;
        }
    }
}

Vec apply_fine_clamped(const gfss::StructuredHexMesh& mesh,
                       const gfss::Material& material,
                       const Vec& x) {
    if (x.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("recursive SA fine operator size mismatch");
    }
    auto free_x = x;
    clamp_x0(mesh, free_x);
    auto y = gfss::apply_matrix_free_openmp(mesh, material, free_x);
    clamp_x0(mesh, y);
    return y;
}

Vec FineSmoothedTransfer::apply_fine(const Vec& x) const {
    return apply_fine_clamped(mesh, material, x);
}

std::vector<double> build_fine_inverse_diagonal(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space) {
    std::vector<double> diagonal(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                for (std::size_t a = 0; a < 8U; ++a) {
                    const auto node = static_cast<std::size_t>(nodes[a]);
                    if (space.graph.constrained[node] != 0U) continue;
                    for (std::size_t c = 0; c < 3U; ++c) {
                        const std::size_t local = 3U * a + c;
                        diagonal[3U * node + c] += ke[local][local];
                    }
                }
            }
        }
    }
    std::vector<double> inverse(diagonal.size(), 0.0);
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        if (space.graph.constrained[node] != 0U) continue;
        for (std::size_t c = 0; c < 3U; ++c) {
            const std::size_t dof = 3U * node + c;
            if (!(diagonal[dof] > 0.0) || !std::isfinite(diagonal[dof])) {
                throw std::runtime_error("recursive SA fine Jacobi diagonal invalid");
            }
            inverse[dof] = 1.0 / diagonal[dof];
        }
    }
    return inverse;
}

double estimate_lambda_max(const Apply& apply,
                           const std::vector<double>& inverse_diagonal,
                           std::size_t power_iterations) {
    if (power_iterations == 0U) throw std::invalid_argument("power iterations must be positive");
    Vec q(inverse_diagonal.size(), 0.0);
    Vec scaled(q.size(), 0.0);
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (inverse_diagonal[i] > 0.0) {
            const double t = static_cast<double>((i % 251U) + 1U);
            q[i] = std::sin(0.173 * t) + 0.37 * std::cos(0.071 * t);
        }
    }
    double qnorm = norm(q);
    if (!(qnorm > 0.0)) throw std::runtime_error("recursive SA lambda probe is zero");
    for (double& v : q) v /= qnorm;

    double lambda = 0.0;
    for (std::size_t it = 0; it < power_iterations; ++it) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            scaled[i] = inverse_diagonal[i] > 0.0
                ? std::sqrt(inverse_diagonal[i]) * q[i] : 0.0;
        }
        auto y = apply(scaled);
        for (std::size_t i = 0; i < y.size(); ++i) {
            y[i] = inverse_diagonal[i] > 0.0
                ? std::sqrt(inverse_diagonal[i]) * y[i] : 0.0;
        }
        const double rq = dot(q, y);
        const double ynorm = norm(y);
        if (!(rq > 0.0) || !(ynorm > 0.0) || !std::isfinite(rq) || !std::isfinite(ynorm)) {
            throw std::runtime_error("recursive SA lambda estimate invalid");
        }
        lambda = std::max(lambda, rq);
        q = std::move(y);
        for (double& v : q) v /= ynorm;
    }
    return kLambdaSafety * lambda;
}

void chebyshev_smooth(const Apply& apply,
                      const std::vector<double>& inverse_diagonal,
                      double lambda_max,
                      const Vec& b,
                      Vec& x,
                      std::size_t degree) {
    if (degree == 0U) return;
    if (b.size() != x.size() || b.size() != inverse_diagonal.size()) {
        throw std::invalid_argument("recursive SA smoother size mismatch");
    }
    const double lambda_low = kChebyshevLowerFraction * lambda_max;
    const double theta = 0.5 * (lambda_max + lambda_low);
    const double delta = 0.5 * (lambda_max - lambda_low);
    for (std::size_t k = 0; k < degree; ++k) {
        const double angle = kPi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        const double root = theta + delta * std::cos(angle);
        if (!(root > 0.0)) throw std::runtime_error("recursive SA Chebyshev root invalid");
        const auto ax = apply(x);
        const double weight = 1.0 / root;
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (inverse_diagonal[i] > 0.0) {
                x[i] += weight * inverse_diagonal[i] * (b[i] - ax[i]);
            }
        }
    }
}

AlgebraicNodeGraph graph_from_variable_blocks(
    const gfss::AggregationVariableBlockMatrix& matrix) {
    const std::size_t nodes = matrix.aggregate_ranks.size();
    if (matrix.aggregate_offsets.size() != nodes ||
        matrix.block_row_offsets.size() != nodes + 1U) {
        throw std::invalid_argument("recursive SA variable-block graph metadata size mismatch");
    }

    AlgebraicNodeGraph graph;
    // AggregationVariableBlockMatrix stores one start offset per aggregate.
    // AlgebraicNodeGraph uses CSR-style DOF offsets and therefore requires the
    // terminal total-DOF sentinel as entry N+1.
    graph.dof_offsets.resize(nodes + 1U, 0U);
    for (std::size_t a = 0; a < nodes; ++a) {
        graph.dof_offsets[a] = matrix.aggregate_offsets[a];
    }
    graph.dof_offsets[nodes] = matrix.coarse_dofs;
    for (std::size_t a = 0; a < nodes; ++a) {
        if (graph.dof_offsets[a] > graph.dof_offsets[a + 1U] ||
            graph.dof_offsets[a + 1U] - graph.dof_offsets[a] != matrix.aggregate_ranks[a]) {
            throw std::runtime_error("recursive SA variable-rank DOF offsets are inconsistent");
        }
    }

    graph.row_offsets.resize(nodes + 1U, 0U);
    std::size_t cursor = 0U;
    for (std::size_t row = 0; row < nodes; ++row) {
        std::vector<std::uint32_t> neighbors;
        for (std::size_t p = matrix.block_row_offsets[row];
             p < matrix.block_row_offsets[row + 1U]; ++p) {
            const auto col = matrix.block_columns[p];
            if (col == row) continue;
            neighbors.push_back(col);
        }
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        graph.row_offsets[row] = static_cast<std::uint32_t>(cursor);
        graph.column_indices.insert(graph.column_indices.end(), neighbors.begin(), neighbors.end());
        cursor += neighbors.size();
    }
    graph.row_offsets[nodes] = static_cast<std::uint32_t>(cursor);
    return graph;
}

std::vector<double> make_level1_candidates(
    const gfss::ElasticityAggregationCoarseSpace& space) {
    std::vector<double> candidates(space.coarse_dofs * kCandidates, 0.0);
    for (std::size_t mode = 0; mode < kCandidates; ++mode) {
        Vec fine(3U * space.graph.coordinates.size(), 0.0);
        for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
            if (space.graph.constrained[node] != 0U) continue;
            const auto& p = space.graph.coordinates[node];
            const std::size_t base = 3U * node;
            switch (mode) {
                case 0U: fine[base + 0U] = 1.0; break;
                case 1U: fine[base + 1U] = 1.0; break;
                case 2U: fine[base + 2U] = 1.0; break;
                case 3U: fine[base + 1U] = -p[2]; fine[base + 2U] = p[1]; break;
                case 4U: fine[base + 0U] = p[2]; fine[base + 2U] = -p[0]; break;
                case 5U: fine[base + 0U] = -p[1]; fine[base + 1U] = p[0]; break;
                default: break;
            }
        }
        const auto coarse = gfss::apply_elasticity_tentative_restriction(space, fine);
        for (std::size_t dof = 0; dof < coarse.size(); ++dof) {
            candidates[dof * kCandidates + mode] = coarse[dof];
        }
    }
    return candidates;
}

std::vector<std::vector<std::uint32_t>> aggregate_graph_nodes(
    const AlgebraicNodeGraph& graph,
    std::size_t target_nodes,
    std::size_t min_nodes) {
    if (graph.nodes() == 0U || graph.row_offsets.size() != graph.nodes() + 1U) {
        throw std::invalid_argument("recursive SA algebraic graph invalid");
    }
    if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
        throw std::invalid_argument("recursive SA aggregate sizes invalid");
    }

    const std::size_t n = graph.nodes();
    std::vector<std::uint32_t> owner(n, kUnassigned);
    std::vector<std::vector<std::uint32_t>> groups;
    for (std::uint32_t seed = 0U; seed < n; ++seed) {
        if (owner[seed] != kUnassigned) continue;
        const auto id = static_cast<std::uint32_t>(groups.size());
        groups.emplace_back();
        std::queue<std::uint32_t> frontier;
        owner[seed] = id;
        groups.back().push_back(seed);
        frontier.push(seed);
        while (!frontier.empty() && groups.back().size() < target_nodes) {
            const auto u = frontier.front();
            frontier.pop();
            for (std::uint32_t p = graph.row_offsets[u];
                 p < graph.row_offsets[u + 1U] && groups.back().size() < target_nodes; ++p) {
                const auto v = graph.column_indices[p];
                if (owner[v] != kUnassigned) continue;
                owner[v] = id;
                groups.back().push_back(v);
                frontier.push(v);
            }
        }
    }

    std::vector<std::uint8_t> active(groups.size(), 1U);
    for (std::size_t a = 0; a < groups.size(); ++a) {
        if (!active[a] || groups[a].size() >= min_nodes) continue;
        std::vector<std::size_t> edges(groups.size(), 0U);
        for (const auto u : groups[a]) {
            for (std::uint32_t p = graph.row_offsets[u]; p < graph.row_offsets[u + 1U]; ++p) {
                const auto v = graph.column_indices[p];
                const auto other = owner[v];
                if (other != kUnassigned && other != a && active[other]) ++edges[other];
            }
        }
        std::size_t best = groups.size();
        std::size_t best_edges = 0U;
        for (std::size_t b = 0; b < groups.size(); ++b) {
            if (!active[b] || b == a) continue;
            if (edges[b] > best_edges) {
                best = b;
                best_edges = edges[b];
            }
        }
        if (best == groups.size()) {
            for (std::size_t b = 0; b < groups.size(); ++b) {
                if (active[b] && b != a) { best = b; break; }
            }
        }
        if (best == groups.size()) continue;
        for (const auto node : groups[a]) {
            owner[node] = static_cast<std::uint32_t>(best);
            groups[best].push_back(node);
        }
        groups[a].clear();
        active[a] = 0U;
    }

    std::vector<std::vector<std::uint32_t>> compact;
    compact.reserve(groups.size());
    for (std::size_t a = 0; a < groups.size(); ++a) {
        if (active[a] && !groups[a].empty()) compact.push_back(std::move(groups[a]));
    }
    return compact;
}

AlgebraicNodeGraph build_coarse_graph_from_groups(
    const AlgebraicNodeGraph& fine_graph,
    const std::vector<std::vector<std::uint32_t>>& groups,
    const std::vector<CandidateAggregate>& aggregates,
    const std::vector<std::uint32_t>& owner) {
    AlgebraicNodeGraph coarse;
    coarse.dof_offsets.resize(groups.size() + 1U, 0U);
    for (std::size_t a = 0; a < aggregates.size(); ++a) {
        coarse.dof_offsets[a] = aggregates[a].coarse_offset;
    }
    coarse.dof_offsets.back() = aggregates.empty()
        ? 0U : aggregates.back().coarse_offset + aggregates.back().rank;
    coarse.row_offsets.resize(groups.size() + 1U, 0U);

    std::size_t cursor = 0U;
    for (std::size_t a = 0; a < groups.size(); ++a) {
        std::vector<std::uint32_t> neighbors;
        for (const auto u : groups[a]) {
            for (std::uint32_t p = fine_graph.row_offsets[u]; p < fine_graph.row_offsets[u + 1U]; ++p) {
                const auto v = fine_graph.column_indices[p];
                const auto other = owner[v];
                if (other != a && other != kUnassigned) neighbors.push_back(other);
            }
        }
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        coarse.row_offsets[a] = static_cast<std::uint32_t>(cursor);
        coarse.column_indices.insert(coarse.column_indices.end(), neighbors.begin(), neighbors.end());
        cursor += neighbors.size();
    }
    coarse.row_offsets[groups.size()] = static_cast<std::uint32_t>(cursor);
    return coarse;
}

CandidateTransfer build_candidate_transfer(
    const AlgebraicNodeGraph& graph,
    const std::vector<double>& candidates,
    std::size_t target_nodes,
    std::size_t min_nodes,
    double rank_tolerance) {
    if (candidates.size() != graph.dofs() * kCandidates) {
        throw std::invalid_argument("recursive SA candidate matrix size mismatch");
    }
    const auto groups = aggregate_graph_nodes(graph, target_nodes, min_nodes);
    std::vector<std::uint32_t> owner(graph.nodes(), kUnassigned);
    for (std::size_t a = 0; a < groups.size(); ++a) {
        for (const auto node : groups[a]) owner[node] = static_cast<std::uint32_t>(a);
    }

    CandidateTransfer transfer;
    transfer.fine_dofs = graph.dofs();
    transfer.aggregates.resize(groups.size());
    std::size_t coarse_cursor = 0U;

    for (std::size_t a = 0; a < groups.size(); ++a) {
        auto& aggregate = transfer.aggregates[a];
        aggregate.nodes = groups[a];
        aggregate.coarse_offset = coarse_cursor;
        for (const auto node : groups[a]) {
            for (std::size_t dof = graph.dof_offsets[node]; dof < graph.dof_offsets[node + 1U]; ++dof) {
                aggregate.fine_dofs.push_back(dof);
            }
        }
        if (aggregate.fine_dofs.empty()) {
            throw std::runtime_error("recursive SA candidate aggregate has no DOFs");
        }

        std::vector<Vec> qcols;
        double max_candidate_norm = 0.0;
        for (std::size_t c = 0; c < kCandidates; ++c) {
            double n2 = 0.0;
            for (const auto dof : aggregate.fine_dofs) {
                const double v = candidates[dof * kCandidates + c];
                n2 += v * v;
            }
            max_candidate_norm = std::max(max_candidate_norm, std::sqrt(n2));
        }
        const double threshold = rank_tolerance * std::max(1.0, max_candidate_norm);

        for (std::size_t candidate = 0; candidate < kCandidates; ++candidate) {
            Vec v(aggregate.fine_dofs.size(), 0.0);
            for (std::size_t row = 0; row < v.size(); ++row) {
                v[row] = candidates[aggregate.fine_dofs[row] * kCandidates + candidate];
            }
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& q : qcols) {
                    const double projection = dot(q, v);
                    for (std::size_t i = 0; i < v.size(); ++i) v[i] -= projection * q[i];
                }
            }
            const double vnorm = norm(v);
            if (!(vnorm > threshold) || !std::isfinite(vnorm)) continue;
            for (double& x : v) x /= vnorm;
            qcols.push_back(std::move(v));
        }
        if (qcols.empty()) {
            throw std::runtime_error("recursive SA candidate QR produced rank zero");
        }
        aggregate.rank = qcols.size();
        aggregate.q_values.assign(aggregate.fine_dofs.size() * aggregate.rank, 0.0);
        for (std::size_t row = 0; row < aggregate.fine_dofs.size(); ++row) {
            for (std::size_t q = 0; q < aggregate.rank; ++q) {
                aggregate.q_values[row * aggregate.rank + q] = qcols[q][row];
            }
        }
        coarse_cursor += aggregate.rank;
    }

    transfer.coarse_dofs = coarse_cursor;
    transfer.coarse_candidates.assign(coarse_cursor * kCandidates, 0.0);
    for (const auto& aggregate : transfer.aggregates) {
        for (std::size_t q = 0; q < aggregate.rank; ++q) {
            for (std::size_t c = 0; c < kCandidates; ++c) {
                double value = 0.0;
                for (std::size_t row = 0; row < aggregate.fine_dofs.size(); ++row) {
                    value += aggregate.q_values[row * aggregate.rank + q] *
                             candidates[aggregate.fine_dofs[row] * kCandidates + c];
                }
                transfer.coarse_candidates[(aggregate.coarse_offset + q) * kCandidates + c] = value;
            }
        }
    }
    transfer.coarse_graph = build_coarse_graph_from_groups(
        graph, groups, transfer.aggregates, owner);
    return transfer;
}

double candidate_reproduction_error(const CandidateTransfer& transfer,
                                    const std::vector<double>& fine_candidates) {
    double worst = 0.0;
    for (std::size_t c = 0; c < kCandidates; ++c) {
        Vec fine(transfer.fine_dofs, 0.0);
        for (std::size_t i = 0; i < fine.size(); ++i) {
            fine[i] = fine_candidates[i * kCandidates + c];
        }
        const auto coarse = transfer.restrict_transpose(fine);
        const auto projected = transfer.prolong(coarse);
        Vec diff(fine.size(), 0.0);
        for (std::size_t i = 0; i < fine.size(); ++i) diff[i] = projected[i] - fine[i];
        const double fn = norm(fine);
        if (fn > 0.0) worst = std::max(worst, norm(diff) / fn);
    }
    return worst;
}

DenseCholesky materialize_and_factor_bottom(const Apply& apply,
                                            std::size_t n) {
    if (n == 0U) throw std::invalid_argument("bottom operator has zero size");
    std::vector<double> a(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        Vec e(n, 0.0);
        e[j] = 1.0;
        const auto col = apply(e);
        if (col.size() != n) throw std::runtime_error("bottom operator column size mismatch");
        for (std::size_t i = 0; i < n; ++i) a[i * n + j] = col[i];
    }

    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1U; j < n; ++j) {
            const double d = a[i * n + j] - a[j * n + i];
            asym2 += 2.0 * d * d;
            norm2 += a[i * n + j] * a[i * n + j] +
                     a[j * n + i] * a[j * n + i];
            const double sym = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = sym;
            a[j * n + i] = sym;
        }
        norm2 += a[i * n + i] * a[i * n + i];
    }

    DenseCholesky result;
    result.n = n;
    result.symmetry_relative_defect = norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;
    result.lower.assign(n * n, 0.0);
    result.min_pivot = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k) {
                value -= result.lower[i * n + k] * result.lower[j * n + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("recursive SA bottom Cholesky lost SPD");
                }
                result.min_pivot = std::min(result.min_pivot, value);
                result.lower[i * n + j] = std::sqrt(value);
            } else {
                result.lower[i * n + j] = value / result.lower[j * n + j];
            }
        }
    }
    return result;
}

Vec make_rhs(const gfss::StructuredHexMesh& mesh) {
    Vec rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const double count = static_cast<double>(mesh.ny + 1U) *
                         static_cast<double>(mesh.nz + 1U);
    const double magnitude = 1.0 / count;
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = -magnitude;
        }
    }
    return rhs;
}

void run_reference(std::size_t fine_transfer_steps,
                   std::size_t max_cycles,
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
    const auto transfer1 = build_candidate_transfer(
        graph1, candidates1, target_nodes, min_nodes, 1.0e-10);
    const double candidate1_error = candidate_reproduction_error(transfer1, candidates1);

    std::vector<double> inverse1 = tentative_a1.inverse_diagonal;
    if (inverse1.size() != graph1.dofs()) {
        throw std::runtime_error("recursive SA L1 diagonal size mismatch");
    }
    const double lambda1 = estimate_lambda_max(apply1, inverse1, 8U);

    const Apply apply2 = [&](const Vec& x) {
        const auto l1 = transfer1.prolong(x);
        return transfer1.restrict_transpose(apply1(l1));
    };
    const auto inverse2 = transfer1.approximate_inverse_coarse_diagonal(inverse1);
    const double lambda2 = estimate_lambda_max(apply2, inverse2, 8U);

    const auto transfer2 = build_candidate_transfer(
        transfer1.coarse_graph,
        transfer1.coarse_candidates,
        target_nodes,
        min_nodes,
        1.0e-10);
    const double candidate2_error = candidate_reproduction_error(
        transfer2, transfer1.coarse_candidates);

    const Apply apply3 = [&](const Vec& x) {
        const auto l2 = transfer2.prolong(x);
        return transfer2.restrict_transpose(apply2(l2));
    };

    const auto bottom_start = Clock::now();
    const auto bottom = materialize_and_factor_bottom(apply3, transfer2.coarse_dofs);
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

    levels[2].dofs = transfer1.coarse_dofs;
    levels[2].label = "L2_recursive_candidate_Galerkin";
    levels[2].diagnostic_lambda_max = lambda2;
    levels[2].apply = apply2;
    levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_smooth(apply2, inverse2, lambda2, b, x, degree);
    };
    levels[2].restrict_to_coarse = [&](const Vec& r) { return transfer2.restrict_transpose(r); };
    levels[2].prolong_from_coarse = [&](const Vec& c) { return transfer2.prolong(c); };

    levels[3].dofs = transfer2.coarse_dofs;
    levels[3].label = "L3_dense_Cholesky_bottom";
    levels[3].apply = apply3;
    levels[3].bottom_solve = [&](const Vec& b) { return bottom.solve(b); };

    const auto rhs = make_rhs(mesh);
    const auto result = gfss::solve_reference_multilevel_vcycle(
        levels, rhs, 1.0e-6, max_cycles, 3U, 3U);

    std::cout << "\n========================================\n"
              << "fine_transfer_smoothing_steps=" << fine_transfer_steps << '\n'
              << "hierarchy_levels=4\n"
              << "higher_level_transfer=tentative_candidate_QR\n"
              << "coarse_mesh_required_after_L0=false\n"
              << "L0_dofs=" << levels[0].dofs
              << " L1_dofs=" << levels[1].dofs
              << " L1_nodes=" << graph1.nodes()
              << " L2_dofs=" << levels[2].dofs
              << " L2_nodes=" << transfer1.coarse_graph.nodes()
              << " L3_dofs=" << levels[3].dofs
              << " L3_nodes=" << transfer2.coarse_graph.nodes() << '\n'
              << std::scientific << std::setprecision(9)
              << "L1_candidate_reproduction_error=" << candidate1_error
              << " L2_candidate_reproduction_error=" << candidate2_error << '\n'
              << std::fixed << std::setprecision(6)
              << "lambda0=" << lambda0
              << " omega0=" << omega0
              << " lambda1=" << lambda1
              << " lambda2=" << lambda2 << '\n'
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
        const std::string selector = argc > 1 ? argv[1] : "all";
        const std::size_t max_cycles = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 8U;
        const std::size_t target_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t min_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 4U;
        if (max_cycles == 0U || target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid recursive SA reference options");
        }

        std::cout << "GFSS M5 recursive smoothed-aggregation numerical reference\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "purpose=test_recursive_SA_before_recursive_GPU_representation\n"
                  << "reference_execution=cpu_fp64\n"
                  << "L0_operator=matrix_free_exact_FEM\n"
                  << "L0_transfer=factorized_smoothed_P0_exact_transpose\n"
                  << "L1_L2_transfer=candidate_based_tentative_QR\n"
                  << "L1_L2_graph=algebraic_block_adjacency\n"
                  << "L3_bottom=dense_materialized_Cholesky\n"
                  << "pre_smooth_degree=3 post_smooth_degree=3\n"
                  << "acceptance_target_post_transient_q<=0.4\n"
                  << "performance_status=numerical_reference_only\n";

        if (selector == "all" || selector == "1") {
            run_reference(1U, max_cycles, target_nodes, min_nodes);
        }
        if (selector == "all" || selector == "2") {
            run_reference(2U, max_cycles, target_nodes, min_nodes);
        }
        if (selector != "all" && selector != "1" && selector != "2") {
            throw std::invalid_argument("selector must be 1, 2, or all");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_reference_bench "
                  << "[1|2|all [max_cycles [target_nodes [min_nodes]]]]\n";
        return 1;
    }
}
