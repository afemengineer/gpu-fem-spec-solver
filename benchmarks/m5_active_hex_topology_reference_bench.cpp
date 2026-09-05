// M5 topology-generalization front-end gate.
//
// This deliberately stops before the production CUDA path.  It removes the
// StructuredHexMesh topology assumption by compacting arbitrary subsets of a
// Cartesian HEX8 parent grid into orphan-style coordinates/connectivity, builds
// NodalGraph3D from the surviving element connectivity, and exercises the exact
// elasticity aggregation + one-step smoothed P0 coarse space.  An exact dense
// A1 solve is used only because these topology cases are intentionally small;
// performance claims are out of scope for this reference gate.

#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/hex8.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Vec = std::vector<double>;
using Apply = std::function<Vec(const Vec&)>;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kLambdaSafety = 1.25;
constexpr double kSaDampingNumerator = 4.0 / 3.0;
constexpr double kChebyshevLowerFraction = 0.10;

struct ActiveHexDomain {
    gfss::StructuredHexMesh parent;
    std::vector<gfss::Vec3> coordinates;
    std::vector<std::array<std::uint32_t, 8>> elements;
    std::vector<std::vector<std::uint32_t>> node_elements;
    std::vector<std::uint8_t> constrained;
    gfss::Hex8Matrix ke{};
    std::size_t parent_elements{0U};
    std::size_t removed_elements{0U};
    double xmax{0.0};

    std::size_t nodes() const noexcept { return coordinates.size(); }
    std::size_t dofs() const noexcept { return 3U * coordinates.size(); }
};

struct DenseFactor {
    std::size_t n{0U};
    std::vector<double> a;
    std::vector<double> lower;
    double symmetry_relative_defect{0.0};
    double min_pivot{0.0};

    Vec solve(const Vec& b) const {
        if (b.size() != n) throw std::invalid_argument("topology bottom solve size mismatch");
        Vec y(n, 0.0);
        for (std::size_t i = 0U; i < n; ++i) {
            double value = b[i];
            for (std::size_t j = 0U; j < i; ++j) value -= lower[i * n + j] * y[j];
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

struct CaseDef {
    const char* name;
    const char* description;
    gfss::StructuredHexMesh parent;
};

double dot(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("topology dot size mismatch");
    double value = 0.0;
    for (std::size_t i = 0U; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}

double norm(const Vec& x) { return std::sqrt(std::max(0.0, dot(x, x))); }

void clamp(const ActiveHexDomain& domain, Vec& x) {
    if (x.size() != domain.dofs()) throw std::invalid_argument("topology clamp size mismatch");
    for (std::size_t node = 0U; node < domain.nodes(); ++node) {
        if (domain.constrained[node] == 0U) continue;
        x[3U * node + 0U] = 0.0;
        x[3U * node + 1U] = 0.0;
        x[3U * node + 2U] = 0.0;
    }
}

bool active_element(const std::string& name,
                    const gfss::StructuredHexMesh& mesh,
                    std::uint32_t ex,
                    std::uint32_t ey,
                    std::uint32_t ez) {
    (void)ez;
    if (name == "box_control") return true;
    if (name == "l_solid") {
        return !(ex >= mesh.nx / 2U && ey >= mesh.ny / 2U);
    }
    if (name == "through_hole") {
        const std::uint32_t x0 = mesh.nx / 2U - 2U;
        const std::uint32_t x1 = mesh.nx / 2U + 1U;
        const std::uint32_t y0 = mesh.ny / 2U - 2U;
        const std::uint32_t y1 = mesh.ny / 2U + 1U;
        return !(ex >= x0 && ex <= x1 && ey >= y0 && ey <= y1);
    }
    if (name == "notched_beam") {
        const std::uint32_t x0 = mesh.nx / 2U - 1U;
        const std::uint32_t x1 = mesh.nx / 2U + 1U;
        return !(ex >= x0 && ex <= x1 && ey >= mesh.ny / 2U);
    }
    throw std::invalid_argument("unknown active topology case");
}

ActiveHexDomain make_domain(const CaseDef& test) {
    ActiveHexDomain out;
    out.parent = test.parent;
    out.parent_elements = static_cast<std::size_t>(test.parent.element_count());
    out.ke = gfss::hex8_stiffness(test.parent.element_coordinates(0U, 0U, 0U),
                                  {210.0e9, 0.30});

    const std::size_t parent_nodes = static_cast<std::size_t>(test.parent.node_count());
    std::vector<std::int64_t> remap(parent_nodes, -1);
    out.coordinates.reserve(parent_nodes);
    out.elements.reserve(out.parent_elements);

    for (std::uint32_t ez = 0U; ez < test.parent.nz; ++ez) {
        for (std::uint32_t ey = 0U; ey < test.parent.ny; ++ey) {
            for (std::uint32_t ex = 0U; ex < test.parent.nx; ++ex) {
                if (!active_element(test.name, test.parent, ex, ey, ez)) {
                    ++out.removed_elements;
                    continue;
                }
                const auto parent_ids = test.parent.element_nodes(ex, ey, ez);
                const auto xyz = test.parent.element_coordinates(ex, ey, ez);
                std::array<std::uint32_t, 8> compact{};
                for (std::size_t a = 0U; a < 8U; ++a) {
                    const std::size_t parent_node = static_cast<std::size_t>(parent_ids[a]);
                    if (remap[parent_node] < 0) {
                        if (out.coordinates.size() >= std::numeric_limits<std::uint32_t>::max()) {
                            throw std::runtime_error("topology active node count exceeds uint32");
                        }
                        remap[parent_node] = static_cast<std::int64_t>(out.coordinates.size());
                        out.coordinates.push_back(xyz[a]);
                    }
                    compact[a] = static_cast<std::uint32_t>(remap[parent_node]);
                }
                out.elements.push_back(compact);
            }
        }
    }
    if (out.elements.empty() || out.coordinates.empty()) {
        throw std::runtime_error("topology case produced empty domain");
    }

    out.node_elements.assign(out.nodes(), {});
    for (std::size_t e = 0U; e < out.elements.size(); ++e) {
        for (const auto node : out.elements[e]) {
            out.node_elements[node].push_back(static_cast<std::uint32_t>(e));
        }
    }
    for (const auto& row : out.node_elements) {
        if (row.empty()) throw std::runtime_error("topology compaction left orphan node");
    }

    out.xmax = 0.0;
    for (const auto& xyz : out.coordinates) out.xmax = std::max(out.xmax, xyz[0]);
    const double tol = std::max(1.0, out.xmax) * 1.0e-12;
    out.constrained.assign(out.nodes(), 0U);
    std::size_t constrained_nodes = 0U;
    for (std::size_t node = 0U; node < out.nodes(); ++node) {
        if (std::abs(out.coordinates[node][0]) <= tol) {
            out.constrained[node] = 1U;
            ++constrained_nodes;
        }
    }
    if (constrained_nodes == 0U || constrained_nodes == out.nodes()) {
        throw std::runtime_error("topology case has invalid support set");
    }
    return out;
}

gfss::NodalGraph3D build_graph(const ActiveHexDomain& domain) {
    gfss::NodalGraph3D graph;
    graph.coordinates = domain.coordinates;
    graph.constrained = domain.constrained;
    std::vector<std::unordered_set<std::uint32_t>> neighbors(domain.nodes());
    for (const auto& elem : domain.elements) {
        for (std::size_t a = 0U; a < 8U; ++a) {
            for (std::size_t b = 0U; b < 8U; ++b) {
                if (a == b) continue;
                neighbors[elem[a]].insert(elem[b]);
            }
        }
    }
    graph.row_offsets.resize(domain.nodes() + 1U, 0U);
    std::size_t cursor = 0U;
    for (std::size_t node = 0U; node < domain.nodes(); ++node) {
        graph.row_offsets[node] = static_cast<std::uint32_t>(cursor);
        std::vector<std::uint32_t> row(neighbors[node].begin(), neighbors[node].end());
        std::sort(row.begin(), row.end());
        graph.column_indices.insert(graph.column_indices.end(), row.begin(), row.end());
        cursor += row.size();
        if (cursor > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("topology graph exceeds uint32 indexing");
        }
    }
    graph.row_offsets[domain.nodes()] = static_cast<std::uint32_t>(cursor);
    return graph;
}

std::size_t graph_components(const gfss::NodalGraph3D& graph) {
    const std::size_t nodes = graph.coordinates.size();
    std::vector<std::uint8_t> seen(nodes, 0U);
    std::size_t components = 0U;
    std::queue<std::uint32_t> q;
    for (std::size_t seed = 0U; seed < nodes; ++seed) {
        if (seen[seed] != 0U) continue;
        ++components;
        seen[seed] = 1U;
        q.push(static_cast<std::uint32_t>(seed));
        while (!q.empty()) {
            const auto node = q.front();
            q.pop();
            for (std::size_t p = graph.row_offsets[node]; p < graph.row_offsets[node + 1U]; ++p) {
                const auto nbr = graph.column_indices[p];
                if (seen[nbr] != 0U) continue;
                seen[nbr] = 1U;
                q.push(nbr);
            }
        }
    }
    return components;
}

Vec apply_fine(const ActiveHexDomain& domain, const Vec& x) {
    if (x.size() != domain.dofs()) throw std::invalid_argument("topology fine apply size mismatch");
    Vec free_x = x;
    clamp(domain, free_x);
    Vec y(x.size(), 0.0);
    for (const auto& elem : domain.elements) {
        gfss::Hex8Vector xe{};
        for (std::size_t a = 0U; a < 8U; ++a) {
            for (std::size_t c = 0U; c < 3U; ++c) {
                xe[3U * a + c] = free_x[3U * elem[a] + c];
            }
        }
        gfss::Hex8Vector ye{};
        for (std::size_t i = 0U; i < 24U; ++i) {
            double value = 0.0;
            for (std::size_t j = 0U; j < 24U; ++j) value += domain.ke[i][j] * xe[j];
            ye[i] = value;
        }
        for (std::size_t a = 0U; a < 8U; ++a) {
            for (std::size_t c = 0U; c < 3U; ++c) {
                y[3U * elem[a] + c] += ye[3U * a + c];
            }
        }
    }
    clamp(domain, y);
    return y;
}

std::vector<double> inverse_diagonal(const ActiveHexDomain& domain) {
    Vec diagonal(domain.dofs(), 0.0);
    for (const auto& elem : domain.elements) {
        for (std::size_t a = 0U; a < 8U; ++a) {
            const std::size_t node = elem[a];
            if (domain.constrained[node] != 0U) continue;
            for (std::size_t c = 0U; c < 3U; ++c) {
                const std::size_t local = 3U * a + c;
                diagonal[3U * node + c] += domain.ke[local][local];
            }
        }
    }
    Vec inverse(diagonal.size(), 0.0);
    for (std::size_t node = 0U; node < domain.nodes(); ++node) {
        if (domain.constrained[node] != 0U) continue;
        for (std::size_t c = 0U; c < 3U; ++c) {
            const std::size_t dof = 3U * node + c;
            if (!(diagonal[dof] > 0.0) || !std::isfinite(diagonal[dof])) {
                throw std::runtime_error("topology Jacobi diagonal invalid");
            }
            inverse[dof] = 1.0 / diagonal[dof];
        }
    }
    return inverse;
}

double estimate_lambda(const Apply& apply,
                       const Vec& inverse,
                       std::size_t iterations = 8U) {
    Vec q(inverse.size(), 0.0);
    for (std::size_t i = 0U; i < q.size(); ++i) {
        if (inverse[i] > 0.0) {
            const double t = static_cast<double>((i % 251U) + 1U);
            q[i] = std::sin(0.173 * t) + 0.37 * std::cos(0.071 * t);
        }
    }
    double qn = norm(q);
    if (!(qn > 0.0)) throw std::runtime_error("topology lambda seed is zero");
    for (double& v : q) v /= qn;
    double lambda = 0.0;
    Vec scaled(q.size(), 0.0);
    for (std::size_t it = 0U; it < iterations; ++it) {
        for (std::size_t i = 0U; i < q.size(); ++i) {
            scaled[i] = inverse[i] > 0.0 ? std::sqrt(inverse[i]) * q[i] : 0.0;
        }
        auto y = apply(scaled);
        for (std::size_t i = 0U; i < y.size(); ++i) {
            y[i] = inverse[i] > 0.0 ? std::sqrt(inverse[i]) * y[i] : 0.0;
        }
        const double rq = dot(q, y);
        const double yn = norm(y);
        if (!(rq > 0.0) || !(yn > 0.0) || !std::isfinite(rq) || !std::isfinite(yn)) {
            throw std::runtime_error("topology lambda estimate invalid");
        }
        lambda = std::max(lambda, rq);
        q = std::move(y);
        for (double& v : q) v /= yn;
    }
    return kLambdaSafety * lambda;
}

struct GenericSmoothedTransfer {
    const ActiveHexDomain& domain;
    const gfss::ElasticityAggregationCoarseSpace& space;
    const Apply& apply;
    const Vec& inverse;
    double omega{0.0};

    Vec prolong(const Vec& coarse) const {
        auto fine = gfss::apply_elasticity_tentative_prolongation(space, coarse);
        clamp(domain, fine);
        const auto af = apply(fine);
        for (std::size_t i = 0U; i < fine.size(); ++i) {
            if (inverse[i] > 0.0) fine[i] -= omega * inverse[i] * af[i];
        }
        clamp(domain, fine);
        return fine;
    }

    Vec restrict_transpose(const Vec& fine_in) const {
        if (fine_in.size() != domain.dofs()) throw std::invalid_argument("topology PT size mismatch");
        Vec work = fine_in;
        clamp(domain, work);
        Vec scaled(work.size(), 0.0);
        for (std::size_t i = 0U; i < work.size(); ++i) scaled[i] = inverse[i] * work[i];
        const auto as = apply(scaled);
        for (std::size_t i = 0U; i < work.size(); ++i) work[i] -= omega * as[i];
        clamp(domain, work);
        return gfss::apply_elasticity_tentative_restriction(space, work);
    }
};

double adjoint_error(const GenericSmoothedTransfer& transfer) {
    Vec coarse(transfer.space.coarse_dofs, 0.0);
    Vec fine(transfer.domain.dofs(), 0.0);
    for (std::size_t i = 0U; i < coarse.size(); ++i) {
        const double t = static_cast<double>(i + 1U);
        coarse[i] = std::sin(0.019 * t + 0.13) + 0.17 * std::cos(0.041 * t);
    }
    for (std::size_t i = 0U; i < fine.size(); ++i) {
        const double t = static_cast<double>(i + 1U);
        fine[i] = std::sin(0.011 * t - 0.31) + 0.23 * std::cos(0.037 * t);
    }
    clamp(transfer.domain, fine);
    const auto pc = transfer.prolong(coarse);
    const auto ptf = transfer.restrict_transpose(fine);
    const double lhs = dot(pc, fine);
    const double rhs = dot(coarse, ptf);
    return std::abs(lhs - rhs) / std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

DenseFactor materialize_and_factor_a1(const GenericSmoothedTransfer& transfer,
                                      const Apply& fine_apply) {
    DenseFactor out;
    out.n = transfer.space.coarse_dofs;
    if (out.n == 0U) throw std::runtime_error("topology A1 has zero dimension");
    out.a.assign(out.n * out.n, 0.0);
    for (std::size_t col = 0U; col < out.n; ++col) {
        Vec e(out.n, 0.0);
        e[col] = 1.0;
        const auto p = transfer.prolong(e);
        const auto ap = fine_apply(p);
        const auto c = transfer.restrict_transpose(ap);
        for (std::size_t row = 0U; row < out.n; ++row) out.a[row * out.n + col] = c[row];
    }

    double asym2 = 0.0;
    double n2 = 0.0;
    for (std::size_t i = 0U; i < out.n; ++i) {
        n2 += out.a[i * out.n + i] * out.a[i * out.n + i];
        for (std::size_t j = i + 1U; j < out.n; ++j) {
            const double aij = out.a[i * out.n + j];
            const double aji = out.a[j * out.n + i];
            const double d = aij - aji;
            asym2 += 2.0 * d * d;
            n2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            out.a[i * out.n + j] = sym;
            out.a[j * out.n + i] = sym;
        }
    }
    out.symmetry_relative_defect = n2 > 0.0 ? std::sqrt(asym2 / n2) : 0.0;

    out.lower.assign(out.n * out.n, 0.0);
    out.min_pivot = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0U; i < out.n; ++i) {
        for (std::size_t j = 0U; j <= i; ++j) {
            double value = out.a[i * out.n + j];
            for (std::size_t k = 0U; k < j; ++k) {
                value -= out.lower[i * out.n + k] * out.lower[j * out.n + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("topology A1 lost SPD");
                }
                out.min_pivot = std::min(out.min_pivot, value);
                out.lower[i * out.n + j] = std::sqrt(value);
            } else {
                out.lower[i * out.n + j] = value / out.lower[j * out.n + j];
            }
        }
    }
    return out;
}

std::vector<double> chebyshev_weights(double lambda_max, std::size_t degree) {
    std::vector<double> weights(degree, 0.0);
    const double low = kChebyshevLowerFraction * lambda_max;
    const double theta = 0.5 * (lambda_max + low);
    const double delta = 0.5 * (lambda_max - low);
    for (std::size_t k = 0U; k < degree; ++k) {
        const double angle = kPi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        weights[k] = 1.0 / (theta + delta * std::cos(angle));
    }
    return weights;
}

void smooth(const ActiveHexDomain& domain,
            const Apply& apply,
            const Vec& inverse,
            const std::vector<double>& weights,
            const Vec& b,
            Vec& x) {
    for (const double w : weights) {
        const auto ax = apply(x);
        for (std::size_t i = 0U; i < x.size(); ++i) {
            if (inverse[i] > 0.0) x[i] += w * inverse[i] * (b[i] - ax[i]);
        }
        clamp(domain, x);
    }
}

Vec make_rhs(const ActiveHexDomain& domain) {
    Vec rhs(domain.dofs(), 0.0);
    const double tol = std::max(1.0, domain.xmax) * 1.0e-12;
    std::vector<std::size_t> loaded;
    for (std::size_t node = 0U; node < domain.nodes(); ++node) {
        if (domain.constrained[node] == 0U &&
            std::abs(domain.coordinates[node][0] - domain.xmax) <= tol) {
            loaded.push_back(node);
        }
    }
    if (loaded.empty()) throw std::runtime_error("topology xmax load face is empty");
    const double value = -1.0 / static_cast<double>(loaded.size());
    for (const auto node : loaded) rhs[3U * node + 2U] = value;
    return rhs;
}

double two_grid_contraction(const ActiveHexDomain& domain,
                            const Apply& apply,
                            const Vec& inverse,
                            double lambda0,
                            const GenericSmoothedTransfer& transfer,
                            const DenseFactor& a1) {
    const auto b = make_rhs(domain);
    const double bnorm = norm(b);
    Vec x(domain.dofs(), 0.0);
    const auto weights = chebyshev_weights(lambda0, 5U);
    smooth(domain, apply, inverse, weights, b, x);
    auto ax = apply(x);
    Vec r(b.size(), 0.0);
    for (std::size_t i = 0U; i < b.size(); ++i) r[i] = b[i] - ax[i];
    const auto b1 = transfer.restrict_transpose(r);
    const auto x1 = a1.solve(b1);
    const auto corr = transfer.prolong(x1);
    for (std::size_t i = 0U; i < x.size(); ++i) x[i] += corr[i];
    smooth(domain, apply, inverse, weights, b, x);
    ax = apply(x);
    for (std::size_t i = 0U; i < b.size(); ++i) r[i] = b[i] - ax[i];
    return norm(r) / bnorm;
}

std::vector<CaseDef> cases() {
    return {
        {"box_control", "compact orphan-connectivity control box",
         {12U, 12U, 6U, 1.0, 1.0, 0.5}},
        {"l_solid", "extruded L-shaped solid with re-entrant corner",
         {12U, 12U, 6U, 1.0, 1.0, 0.5}},
        {"through_hole", "solid with rectangular through-thickness hole",
         {14U, 12U, 6U, 1.2, 1.0, 0.5}},
        {"notched_beam", "beam with deep top-surface notch",
         {18U, 8U, 6U, 2.0, 0.8, 0.6}},
    };
}

bool run_case(const CaseDef& test, std::size_t target_nodes, std::size_t min_nodes) {
    const auto domain = make_domain(test);
    auto graph = build_graph(domain);
    const std::size_t components = graph_components(graph);
    const std::size_t graph_edges = graph.column_indices.size();
    const auto space = gfss::build_elasticity_aggregation_coarse_space(
        std::move(graph), {target_nodes, min_nodes, 1.0e-10});
    const double rb_error = gfss::audit_elasticity_rigid_body_reproduction(space);

    const Apply apply = [&](const Vec& x) { return apply_fine(domain, x); };
    const auto inverse = inverse_diagonal(domain);
    const double lambda0 = estimate_lambda(apply, inverse, 8U);
    const double omega0 = kSaDampingNumerator / lambda0;
    const GenericSmoothedTransfer transfer{domain, space, apply, inverse, omega0};
    const double pt_error = adjoint_error(transfer);
    const auto a1 = materialize_and_factor_a1(transfer, apply);
    const double q = two_grid_contraction(domain, apply, inverse, lambda0, transfer, a1);

    std::size_t min_rank = 7U;
    std::size_t max_rank = 0U;
    for (const auto& agg : space.aggregates) {
        min_rank = std::min(min_rank, agg.rank);
        max_rank = std::max(max_rank, agg.rank);
    }
    std::size_t constrained_nodes = 0U;
    for (const auto v : domain.constrained) constrained_nodes += v != 0U ? 1U : 0U;

    const bool accept = components == 1U && rb_error <= 1.0e-10 &&
                        pt_error <= 1.0e-10 &&
                        a1.symmetry_relative_defect <= 1.0e-10 &&
                        a1.min_pivot > 0.0 && std::isfinite(a1.min_pivot) &&
                        q < 0.5;

    std::cout << "case=" << test.name << '\n'
              << "description=" << test.description << '\n'
              << "parent_mesh=" << test.parent.nx << 'x' << test.parent.ny << 'x' << test.parent.nz
              << " parent_elements=" << domain.parent_elements
              << " active_elements=" << domain.elements.size()
              << " removed_elements=" << domain.removed_elements
              << " active_nodes=" << domain.nodes()
              << " dofs=" << domain.dofs() << '\n'
              << "graph_components=" << components
              << " directed_graph_edges=" << graph_edges
              << " constrained_nodes=" << constrained_nodes
              << " orphan_nodes=0\n"
              << "free_nodes=" << space.free_nodes
              << " L1_dofs=" << space.coarse_dofs
              << " aggregates=" << space.aggregates.size()
              << " aggregate_rank_min=" << min_rank
              << " aggregate_rank_max=" << max_rank
              << " fine_free_to_L1_ratio="
              << static_cast<double>(space.fine_free_dofs) /
                 static_cast<double>(std::max<std::size_t>(1U, space.coarse_dofs)) << '\n'
              << std::scientific << std::setprecision(12)
              << "rigid_body_reproduction_error=" << rb_error
              << " smoothed_transfer_adjoint_error=" << pt_error
              << " A1_symmetry_relative_defect=" << a1.symmetry_relative_defect
              << " A1_min_cholesky_pivot=" << a1.min_pivot << '\n'
              << std::fixed << std::setprecision(6)
              << "lambda0=" << lambda0 << " omega0=" << omega0
              << " two_grid_true_residual_contraction=" << q
              << " accept=" << (accept ? "true" : "false") << '\n';
    return accept;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "all";
        const std::size_t target_nodes = argc > 2 ? std::stoull(argv[2]) : 12U;
        const std::size_t min_nodes = argc > 3 ? std::stoull(argv[3]) : 4U;
        if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid topology aggregation options");
        }
        std::cout << "GFSS M5 active-HEX topology generalization front-end gate\n"
                  << "representation=compact_orphan_hex_connectivity\n"
                  << "constraint=x0_zero_dirichlet\n"
                  << "transfer=one_step_smoothed_rigid_body_aggregation\n"
                  << "coarse_solver=exact_dense_A1_reference_only\n"
                  << "target_nodes=" << target_nodes << " min_nodes=" << min_nodes << '\n'
                  << "selector=" << selector << '\n';

        std::size_t selected = 0U;
        std::size_t passed = 0U;
        for (const auto& test : cases()) {
            if (selector != "all" && selector != test.name) continue;
            ++selected;
            std::cout << "\n========================================\n";
            if (run_case(test, target_nodes, min_nodes)) ++passed;
        }
        if (selected == 0U) throw std::invalid_argument("unknown topology case");
        const std::size_t failed = selected - passed;
        std::cout << "\n========================================\n"
                  << "suite_selected=" << selected
                  << " suite_passed=" << passed
                  << " suite_failed=" << failed << '\n';
        return failed == 0U ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_m5_active_hex_topology_reference_bench "
                  << "[all|box_control|l_solid|through_hole|notched_beam "
                  << "[target_nodes=12 [min_nodes=4]]]\n";
        return 1;
    }
}
