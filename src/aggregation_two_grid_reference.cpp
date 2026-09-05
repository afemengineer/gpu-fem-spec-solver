#include "gfss/aggregation_two_grid_reference.hpp"

#include "gfss/aggregation_coarse_operator.hpp"
#include "gfss/cpu_elasticity.hpp"
#include "gfss/hex8.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gfss {
namespace {

using Clock = std::chrono::steady_clock;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kChebyshevLowerFraction = 0.10;
constexpr double kLambdaSafety = 1.25;

struct CoarsePcgResult {
    bool converged{false};
    std::size_t iterations{0};
    double relative_residual{0.0};
    std::vector<double> x;
};

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("aggregation two-grid dot size mismatch");
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
    return sum;
}

double norm(const std::vector<double>& v) {
    return std::sqrt(std::max(0.0, dot(v, v)));
}

void clamp_x0(const StructuredHexMesh& mesh, std::vector<double>& v) {
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(0U, j, k);
            const auto base = static_cast<std::size_t>(3ULL * node);
            v[base + 0U] = 0.0;
            v[base + 1U] = 0.0;
            v[base + 2U] = 0.0;
        }
    }
}

std::vector<double> apply_clamped_openmp(const StructuredHexMesh& mesh,
                                          const Material& material,
                                          const std::vector<double>& x) {
    if (x.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("aggregation two-grid fine operator size mismatch");
    }
    auto free_x = x;
    clamp_x0(mesh, free_x);
    auto y = apply_matrix_free_openmp(mesh, material, free_x);
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(0U, j, k);
            const auto base = static_cast<std::size_t>(3ULL * node);
            y[base + 0U] = x[base + 0U];
            y[base + 1U] = x[base + 1U];
            y[base + 2U] = x[base + 2U];
        }
    }
    return y;
}

std::array<double, 6> basis_row(
    const ElasticityAggregationCoarseSpace& space,
    std::size_t node,
    std::size_t component) {
    std::array<double, 6> values{};
    if (node >= space.aggregate_of_node.size() || component >= 3U) return values;
    const auto aggregate_id = space.aggregate_of_node[node];
    if (aggregate_id >= space.aggregates.size()) return values;

    const auto& aggregate = space.aggregates[aggregate_id];
    const auto& xyz = space.graph.coordinates[node];
    const double scale = aggregate.coordinate_scale > 0.0
        ? aggregate.coordinate_scale
        : 1.0;
    const double x = (xyz[0] - aggregate.centroid[0]) / scale;
    const double y = (xyz[1] - aggregate.centroid[1]) / scale;
    const double z = (xyz[2] - aggregate.centroid[2]) / scale;

    std::array<double, 6> raw{};
    if (component == 0U) raw = {1.0, 0.0, 0.0, 0.0, z, -y};
    if (component == 1U) raw = {0.0, 1.0, 0.0, -z, 0.0, x};
    if (component == 2U) raw = {0.0, 0.0, 1.0, y, -x, 0.0};

    for (std::size_t q = 0; q < aggregate.rank; ++q) {
        const double* transform = aggregate.rigid_transform.data() + 6U * q;
        double value = 0.0;
        for (std::size_t j = 0; j < 6U; ++j) value += raw[j] * transform[j];
        values[q] = value;
    }
    return values;
}

std::size_t block_index(const AggregationVariableBlockMatrix& matrix,
                        std::size_t row_aggregate,
                        std::uint32_t column_aggregate) {
    const auto first = matrix.block_row_offsets[row_aggregate];
    const auto last = matrix.block_row_offsets[row_aggregate + 1U];
    const auto begin = matrix.block_columns.begin() + static_cast<std::ptrdiff_t>(first);
    const auto end = matrix.block_columns.begin() + static_cast<std::ptrdiff_t>(last);
    const auto it = std::lower_bound(begin, end, column_aggregate);
    if (it == end || *it != column_aggregate) {
        throw std::runtime_error("aggregation Galerkin assembly missing coarse block");
    }
    return first + static_cast<std::size_t>(it - begin);
}

std::vector<double> build_fine_inverse_diagonal(
    const StructuredHexMesh& mesh,
    const Material& material,
    const ElasticityAggregationCoarseSpace& space) {
    std::vector<double> diagonal(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const auto ke = hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);

    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                for (std::size_t a = 0; a < 8U; ++a) {
                    const auto node = static_cast<std::size_t>(nodes[a]);
                    if (space.graph.constrained[node] != 0U) continue;
                    for (std::size_t c = 0; c < 3U; ++c) {
                        const std::size_t local = 3U * a + c;
                        const std::size_t global = 3U * node + c;
                        diagonal[global] += ke[local][local];
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
                throw std::runtime_error("aggregation two-grid fine Jacobi diagonal is invalid");
            }
            inverse[dof] = 1.0 / diagonal[dof];
        }
    }
    return inverse;
}

double estimate_lambda_max(const StructuredHexMesh& mesh,
                           const Material& material,
                           const std::vector<double>& inverse_diagonal,
                           std::size_t power_iterations) {
    if (power_iterations == 0U) {
        throw std::invalid_argument("aggregation two-grid power_iterations must be positive");
    }
    std::vector<double> q(inverse_diagonal.size(), 0.0);
    std::vector<double> scaled(inverse_diagonal.size(), 0.0);
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (inverse_diagonal[i] > 0.0) {
            const double t = static_cast<double>((i % 251U) + 1U);
            q[i] = std::sin(0.173 * t) + 0.37 * std::cos(0.071 * t);
        }
    }
    double qnorm = norm(q);
    if (!(qnorm > 0.0)) {
        throw std::runtime_error("aggregation two-grid lambda probe is zero");
    }
    for (double& value : q) value /= qnorm;

    double lambda = 0.0;
    for (std::size_t it = 0; it < power_iterations; ++it) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            scaled[i] = inverse_diagonal[i] > 0.0
                ? std::sqrt(inverse_diagonal[i]) * q[i]
                : 0.0;
        }
        auto y = apply_clamped_openmp(mesh, material, scaled);
        for (std::size_t i = 0; i < y.size(); ++i) {
            y[i] = inverse_diagonal[i] > 0.0
                ? std::sqrt(inverse_diagonal[i]) * y[i]
                : 0.0;
        }
        const double rayleigh = dot(q, y);
        const double ynorm = norm(y);
        if (!(rayleigh > 0.0) || !(ynorm > 0.0) ||
            !std::isfinite(rayleigh) || !std::isfinite(ynorm)) {
            throw std::runtime_error("aggregation two-grid lambda estimate became invalid");
        }
        lambda = std::max(lambda, rayleigh);
        q = std::move(y);
        for (double& value : q) value /= ynorm;
    }
    return kLambdaSafety * lambda;
}

void chebyshev_jacobi_smooth(const StructuredHexMesh& mesh,
                             const Material& material,
                             const std::vector<double>& inverse_diagonal,
                             double lambda_max,
                             const std::vector<double>& b,
                             std::vector<double>& x,
                             std::size_t degree) {
    if (degree == 0U) return;
    if (!(lambda_max > 0.0)) {
        throw std::runtime_error("aggregation two-grid smoother missing lambda_max");
    }
    const double lambda_low = kChebyshevLowerFraction * lambda_max;
    const double theta = 0.5 * (lambda_max + lambda_low);
    const double delta = 0.5 * (lambda_max - lambda_low);

    for (std::size_t k = 0; k < degree; ++k) {
        const double angle = kPi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        const double root = theta + delta * std::cos(angle);
        if (!(root > 0.0)) {
            throw std::runtime_error("aggregation two-grid Chebyshev root is invalid");
        }
        const double omega = 1.0 / root;
        const auto ax = apply_clamped_openmp(mesh, material, x);
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (inverse_diagonal[i] > 0.0) {
                x[i] += omega * inverse_diagonal[i] * (b[i] - ax[i]);
            }
        }
        clamp_x0(mesh, x);
    }
}

CoarsePcgResult solve_coarse_pcg(const AggregationVariableBlockMatrix& matrix,
                                 const std::vector<double>& b,
                                 double relative_tolerance,
                                 std::size_t max_iterations) {
    if (b.size() != matrix.coarse_dofs) {
        throw std::invalid_argument("aggregation coarse PCG RHS size mismatch");
    }
    if (!(relative_tolerance > 0.0) || max_iterations == 0U) {
        throw std::invalid_argument("aggregation coarse PCG options are invalid");
    }

    CoarsePcgResult result;
    result.x.assign(b.size(), 0.0);
    const double bnorm = norm(b);
    if (bnorm == 0.0) {
        result.converged = true;
        result.relative_residual = 0.0;
        return result;
    }

    std::vector<double> r = b;
    std::vector<double> z(b.size(), 0.0);
    std::vector<double> p(b.size(), 0.0);
    for (std::size_t i = 0; i < b.size(); ++i) {
        z[i] = matrix.inverse_diagonal[i] * r[i];
        p[i] = z[i];
    }
    double rho = dot(r, z);
    if (!(rho > 0.0) || !std::isfinite(rho)) {
        throw std::runtime_error("aggregation coarse PCG initial rho is invalid");
    }

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        const auto q = apply_aggregation_variable_block_matrix(matrix, p);
        const double pq = dot(p, q);
        if (!(pq > 0.0) || !std::isfinite(pq)) {
            throw std::runtime_error("aggregation coarse PCG lost positive curvature");
        }
        const double alpha = rho / pq;
        for (std::size_t i = 0; i < b.size(); ++i) {
            result.x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        result.iterations = iteration + 1U;
        result.relative_residual = norm(r) / bnorm;
        if (!std::isfinite(result.relative_residual)) {
            throw std::runtime_error("aggregation coarse PCG residual became invalid");
        }
        if (result.relative_residual <= relative_tolerance) {
            result.converged = true;
            return result;
        }

        for (std::size_t i = 0; i < b.size(); ++i) {
            z[i] = matrix.inverse_diagonal[i] * r[i];
        }
        const double rho_new = dot(r, z);
        if (!(rho_new > 0.0) || !std::isfinite(rho_new)) {
            throw std::runtime_error("aggregation coarse PCG rho became invalid");
        }
        const double beta = rho_new / rho;
        for (std::size_t i = 0; i < b.size(); ++i) p[i] = z[i] + beta * p[i];
        rho = rho_new;
    }
    return result;
}

std::vector<double> deterministic_probe(std::size_t n, double phase) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = std::sin(0.017 * t + phase) + 0.29 * std::cos(0.043 * t - phase);
    }
    const double nrm = norm(v);
    if (!(nrm > 0.0)) throw std::runtime_error("aggregation probe vector is zero");
    for (double& value : v) value /= nrm;
    return v;
}

}  // namespace

AggregationVariableBlockMatrix assemble_structured_hex_aggregation_galerkin(
    const StructuredHexMesh& mesh,
    const Material& material,
    const ElasticityAggregationCoarseSpace& space) {
    if (space.graph.coordinates.size() != static_cast<std::size_t>(mesh.node_count())) {
        throw std::invalid_argument("aggregation Galerkin space/mesh node mismatch");
    }
    if (space.aggregates.empty() || space.coarse_dofs == 0U) {
        throw std::invalid_argument("aggregation Galerkin coarse space is empty");
    }

    const std::size_t aggregate_count = space.aggregates.size();
    std::vector<std::vector<std::uint32_t>> neighbors(aggregate_count);

    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                std::array<std::uint32_t, 8> ids{};
                std::size_t count = 0U;
                for (const auto node64 : nodes) {
                    const auto node = static_cast<std::size_t>(node64);
                    const auto id = space.aggregate_of_node[node];
                    if (id >= aggregate_count) continue;
                    bool present = false;
                    for (std::size_t i = 0; i < count; ++i) present = present || ids[i] == id;
                    if (!present) ids[count++] = id;
                }
                for (std::size_t i = 0; i < count; ++i) {
                    for (std::size_t j = 0; j < count; ++j) {
                        neighbors[ids[i]].push_back(ids[j]);
                    }
                }
            }
        }
    }

    AggregationVariableBlockMatrix matrix;
    matrix.coarse_dofs = space.coarse_dofs;
    matrix.aggregate_offsets.resize(aggregate_count);
    matrix.aggregate_ranks.resize(aggregate_count);
    matrix.block_row_offsets.resize(aggregate_count + 1U, 0U);

    for (std::size_t a = 0; a < aggregate_count; ++a) {
        matrix.aggregate_offsets[a] = space.aggregates[a].coarse_offset;
        matrix.aggregate_ranks[a] = space.aggregates[a].rank;
        auto& row = neighbors[a];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        if (!std::binary_search(row.begin(), row.end(), static_cast<std::uint32_t>(a))) {
            row.push_back(static_cast<std::uint32_t>(a));
            std::sort(row.begin(), row.end());
        }
        matrix.block_row_offsets[a + 1U] = matrix.block_row_offsets[a] + row.size();
        matrix.block_columns.insert(matrix.block_columns.end(), row.begin(), row.end());
    }

    matrix.block_value_offsets.resize(matrix.block_columns.size() + 1U, 0U);
    for (std::size_t a = 0; a < aggregate_count; ++a) {
        const std::size_t ra = matrix.aggregate_ranks[a];
        for (std::size_t bi = matrix.block_row_offsets[a];
             bi < matrix.block_row_offsets[a + 1U]; ++bi) {
            const auto b = static_cast<std::size_t>(matrix.block_columns[bi]);
            const std::size_t rb = matrix.aggregate_ranks[b];
            matrix.block_value_offsets[bi + 1U] =
                matrix.block_value_offsets[bi] + ra * rb;
        }
    }
    matrix.values.assign(matrix.block_value_offsets.back(), 0.0);

    const auto ke = hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                std::array<std::uint32_t, 8> aggregate_ids{};
                std::array<std::array<std::array<double, 6>, 3>, 8> basis{};
                for (std::size_t a = 0; a < 8U; ++a) {
                    const auto node = static_cast<std::size_t>(nodes[a]);
                    aggregate_ids[a] = space.aggregate_of_node[node];
                    if (aggregate_ids[a] >= aggregate_count) continue;
                    for (std::size_t c = 0; c < 3U; ++c) {
                        basis[a][c] = basis_row(space, node, c);
                    }
                }

                for (std::size_t a = 0; a < 8U; ++a) {
                    const auto aggregate_a = static_cast<std::size_t>(aggregate_ids[a]);
                    if (aggregate_a >= aggregate_count) continue;
                    const std::size_t ra = matrix.aggregate_ranks[aggregate_a];
                    for (std::size_t b = 0; b < 8U; ++b) {
                        const auto aggregate_b = static_cast<std::size_t>(aggregate_ids[b]);
                        if (aggregate_b >= aggregate_count) continue;
                        const std::size_t rb = matrix.aggregate_ranks[aggregate_b];
                        const std::size_t bi = block_index(
                            matrix, aggregate_a, static_cast<std::uint32_t>(aggregate_b));
                        double* block = matrix.values.data() + matrix.block_value_offsets[bi];

                        std::array<std::array<double, 6>, 3> temp{};
                        for (std::size_t ca = 0; ca < 3U; ++ca) {
                            for (std::size_t cb = 0; cb < 3U; ++cb) {
                                const double kij = ke[3U * a + ca][3U * b + cb];
                                if (kij == 0.0) continue;
                                for (std::size_t s = 0; s < rb; ++s) {
                                    temp[ca][s] += kij * basis[b][cb][s];
                                }
                            }
                        }
                        for (std::size_t q = 0; q < ra; ++q) {
                            for (std::size_t s = 0; s < rb; ++s) {
                                double value = 0.0;
                                for (std::size_t ca = 0; ca < 3U; ++ca) {
                                    value += basis[a][ca][q] * temp[ca][s];
                                }
                                block[q * rb + s] += value;
                            }
                        }
                    }
                }
            }
        }
    }

    matrix.inverse_diagonal.assign(matrix.coarse_dofs, 0.0);
    for (std::size_t a = 0; a < aggregate_count; ++a) {
        const std::size_t rank = matrix.aggregate_ranks[a];
        const std::size_t bi = block_index(matrix, a, static_cast<std::uint32_t>(a));
        const double* block = matrix.values.data() + matrix.block_value_offsets[bi];
        for (std::size_t q = 0; q < rank; ++q) {
            const double diagonal = block[q * rank + q];
            if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
                throw std::runtime_error("aggregation Galerkin coarse diagonal is invalid");
            }
            matrix.inverse_diagonal[matrix.aggregate_offsets[a] + q] = 1.0 / diagonal;
        }
    }

    matrix.storage_bytes =
        matrix.aggregate_offsets.size() * sizeof(std::size_t) +
        matrix.aggregate_ranks.size() * sizeof(std::size_t) +
        matrix.block_row_offsets.size() * sizeof(std::size_t) +
        matrix.block_columns.size() * sizeof(std::uint32_t) +
        matrix.block_value_offsets.size() * sizeof(std::size_t) +
        matrix.values.size() * sizeof(double) +
        matrix.inverse_diagonal.size() * sizeof(double);
    return matrix;
}

std::vector<double> apply_aggregation_variable_block_matrix(
    const AggregationVariableBlockMatrix& matrix,
    const std::vector<double>& x) {
    if (x.size() != matrix.coarse_dofs) {
        throw std::invalid_argument("aggregation variable-block matvec size mismatch");
    }
    std::vector<double> y(matrix.coarse_dofs, 0.0);
    const std::int64_t rows = static_cast<std::int64_t>(matrix.aggregate_ranks.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t row64 = 0; row64 < rows; ++row64) {
        const auto a = static_cast<std::size_t>(row64);
        const std::size_t ra = matrix.aggregate_ranks[a];
        const std::size_t oa = matrix.aggregate_offsets[a];
        for (std::size_t bi = matrix.block_row_offsets[a];
             bi < matrix.block_row_offsets[a + 1U]; ++bi) {
            const auto b = static_cast<std::size_t>(matrix.block_columns[bi]);
            const std::size_t rb = matrix.aggregate_ranks[b];
            const std::size_t ob = matrix.aggregate_offsets[b];
            const double* block = matrix.values.data() + matrix.block_value_offsets[bi];
            for (std::size_t q = 0; q < ra; ++q) {
                double sum = 0.0;
                for (std::size_t s = 0; s < rb; ++s) {
                    sum += block[q * rb + s] * x[ob + s];
                }
                y[oa + q] += sum;
            }
        }
    }
    return y;
}

AggregationTwoGridResult solve_aggregation_two_grid_reference(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    const AggregationTwoGridOptions& options) {
    if (rhs.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("aggregation two-grid RHS size mismatch");
    }
    if (!(options.true_relative_tolerance > 0.0) ||
        !(options.coarse_relative_tolerance > 0.0) ||
        options.max_cycles == 0U || options.coarse_max_iterations == 0U ||
        options.power_iterations == 0U) {
        throw std::invalid_argument("aggregation two-grid options are invalid");
    }

    AggregationTwoGridResult result;
    result.fine_dofs = static_cast<std::size_t>(mesh.dof_count());
    const auto total_start = Clock::now();

    const auto aggregation_start = Clock::now();
    auto graph = build_structured_hex_nodal_graph_x0(mesh);
    const auto space = build_elasticity_aggregation_coarse_space(
        std::move(graph),
        {options.target_nodes_per_aggregate,
         options.min_nodes_per_aggregate,
         options.rank_tolerance});
    const auto aggregation_stop = Clock::now();
    result.aggregation_setup_ms = elapsed_ms(aggregation_start, aggregation_stop);
    result.fine_free_dofs = space.fine_free_dofs;
    result.coarse_dofs = space.coarse_dofs;
    result.aggregates = space.aggregates.size();
    result.matrix_free_transfer_bytes_per_fine_free_dof =
        static_cast<double>(space.estimated_matrix_free_transfer_payload_bytes) /
        static_cast<double>(space.fine_free_dofs);

    const auto assembly_start = Clock::now();
    const auto coarse_matrix = assemble_structured_hex_aggregation_galerkin(
        mesh, material, space);
    const auto assembly_stop = Clock::now();
    result.coarse_assembly_ms = elapsed_ms(assembly_start, assembly_stop);
    result.explicit_coarse_matrix_bytes = coarse_matrix.storage_bytes;
    result.explicit_coarse_matrix_bytes_per_fine_free_dof =
        static_cast<double>(coarse_matrix.storage_bytes) /
        static_cast<double>(space.fine_free_dofs);

    const auto apply_fine = [&](const std::vector<double>& x) {
        return apply_clamped_openmp(mesh, material, x);
    };
    const auto u = deterministic_probe(space.coarse_dofs, 0.17);
    const auto v = deterministic_probe(space.coarse_dofs, 0.73);
    const auto explicit_u = apply_aggregation_variable_block_matrix(coarse_matrix, u);
    const auto explicit_v = apply_aggregation_variable_block_matrix(coarse_matrix, v);
    const auto oracle_u = apply_elasticity_aggregation_coarse_operator(space, apply_fine, u);
    std::vector<double> oracle_diff(oracle_u.size(), 0.0);
    for (std::size_t i = 0; i < oracle_u.size(); ++i) {
        oracle_diff[i] = explicit_u[i] - oracle_u[i];
    }
    result.coarse_operator_oracle_relative_error = norm(oracle_diff) / norm(oracle_u);
    const double uv = dot(u, explicit_v);
    const double vu = dot(v, explicit_u);
    result.coarse_symmetry_relative_defect =
        std::abs(uv - vu) / std::max({std::abs(uv), std::abs(vu), 1.0});
    result.coarse_spd_probe = dot(u, explicit_u) > 0.0;

    const auto smoother_start = Clock::now();
    const auto fine_inverse_diagonal = build_fine_inverse_diagonal(mesh, material, space);
    result.fine_lambda_max = estimate_lambda_max(
        mesh, material, fine_inverse_diagonal, options.power_iterations);
    const auto smoother_stop = Clock::now();
    result.smoother_setup_ms = elapsed_ms(smoother_start, smoother_stop);

    const double bnorm = norm(rhs);
    if (!(bnorm > 0.0) || !std::isfinite(bnorm)) {
        throw std::invalid_argument("aggregation two-grid requires a finite non-zero RHS");
    }
    std::vector<double> x(rhs.size(), 0.0);
    const auto solve_start = Clock::now();

    auto ax = apply_clamped_openmp(mesh, material, x);
    std::vector<double> residual(rhs.size(), 0.0);
    for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
    result.true_relative_residuals.push_back(norm(residual) / bnorm);

    for (std::size_t cycle = 0; cycle < options.max_cycles; ++cycle) {
        chebyshev_jacobi_smooth(
            mesh, material, fine_inverse_diagonal, result.fine_lambda_max,
            rhs, x, options.pre_smooth_degree);

        ax = apply_clamped_openmp(mesh, material, x);
        for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
        const auto coarse_rhs = apply_elasticity_tentative_restriction(space, residual);
        const auto coarse = solve_coarse_pcg(
            coarse_matrix,
            coarse_rhs,
            options.coarse_relative_tolerance,
            options.coarse_max_iterations);
        result.coarse_iterations.push_back(coarse.iterations);
        result.coarse_final_relative_residuals.push_back(coarse.relative_residual);
        if (!coarse.converged) {
            throw std::runtime_error("aggregation two-grid reference coarse PCG did not converge");
        }

        const auto fine_correction = apply_elasticity_tentative_prolongation(space, coarse.x);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += fine_correction[i];
        clamp_x0(mesh, x);

        chebyshev_jacobi_smooth(
            mesh, material, fine_inverse_diagonal, result.fine_lambda_max,
            rhs, x, options.post_smooth_degree);

        ax = apply_clamped_openmp(mesh, material, x);
        for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
        const double relative = norm(residual) / bnorm;
        const double previous = result.true_relative_residuals.back();
        result.true_relative_residuals.push_back(relative);
        result.cycle_contractions.push_back(relative / previous);
        result.cycles = cycle + 1U;
        if (!std::isfinite(relative)) {
            throw std::runtime_error("aggregation two-grid true residual became invalid");
        }
        if (relative <= options.true_relative_tolerance) {
            result.converged = true;
            break;
        }
    }

    const auto solve_stop = Clock::now();
    result.solve_ms = elapsed_ms(solve_start, solve_stop);
    result.total_ms = elapsed_ms(total_start, solve_stop);
    return result;
}

}  // namespace gfss
