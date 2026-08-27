// Numerical-reference experiment: derive exact diagonal blocks of the actual
// smoothed first coarse operator A1 = P0^T A0 P0 without materializing A1.
// For m0=1 each smoothed aggregate basis has one-ring nodal support, so its
// diagonal Galerkin block can be assembled exactly from local HEX8 energies.
#define GFSS_RECURSIVE_SA_ACTUAL_A2_NO_MAIN
#include "recursive_sa_actual_a2_reference_bench.cpp"
#undef GFSS_RECURSIVE_SA_ACTUAL_A2_NO_MAIN

#include <unordered_map>

namespace {

struct L1BlockMetric {
    std::vector<std::size_t> dof_offsets;
    std::vector<std::size_t> value_offsets;
    std::vector<double> lower;
    std::size_t min_rank{0};
    std::size_t max_rank{0};
    double min_cholesky_pivot{0.0};
    double setup_ms{0.0};

    std::size_t nodes() const {
        return dof_offsets.empty() ? 0U : dof_offsets.size() - 1U;
    }
    std::size_t dofs() const {
        return dof_offsets.empty() ? 0U : dof_offsets.back();
    }
    std::size_t storage_bytes() const {
        return dof_offsets.size() * sizeof(std::size_t) +
               value_offsets.size() * sizeof(std::size_t) +
               lower.size() * sizeof(double);
    }

    Vec solve(const Vec& rhs) const {
        if (rhs.size() != dofs()) throw std::invalid_argument("L1 block solve size mismatch");
        Vec result(rhs.size(), 0.0);
        for (std::size_t a = 0; a < nodes(); ++a) {
            const std::size_t begin = dof_offsets[a];
            const std::size_t rank = dof_offsets[a + 1U] - begin;
            const double* l = lower.data() + value_offsets[a];
            std::array<double, kCandidates> y{};
            for (std::size_t i = 0; i < rank; ++i) {
                double value = rhs[begin + i];
                for (std::size_t j = 0; j < i; ++j) value -= l[i * rank + j] * y[j];
                y[i] = value / l[i * rank + i];
            }
            for (std::size_t ii = rank; ii-- > 0U;) {
                double value = y[ii];
                for (std::size_t j = ii + 1U; j < rank; ++j) {
                    value -= l[j * rank + ii] * result[begin + j];
                }
                result[begin + ii] = value / l[ii * rank + ii];
            }
        }
        return result;
    }

    double norm2(const Vec& x) const {
        if (x.size() != dofs()) throw std::invalid_argument("L1 block norm size mismatch");
        double sum = 0.0;
        for (std::size_t a = 0; a < nodes(); ++a) {
            const std::size_t begin = dof_offsets[a];
            const std::size_t rank = dof_offsets[a + 1U] - begin;
            const double* l = lower.data() + value_offsets[a];
            for (std::size_t j = 0; j < rank; ++j) {
                double value = 0.0;
                for (std::size_t i = j; i < rank; ++i) {
                    value += l[i * rank + j] * x[begin + i];
                }
                sum += value * value;
            }
        }
        return sum;
    }

    std::vector<double> inverse_scalar_diagonal() const {
        std::vector<double> inverse(dofs(), 0.0);
        for (std::size_t a = 0; a < nodes(); ++a) {
            const std::size_t begin = dof_offsets[a];
            const std::size_t rank = dof_offsets[a + 1U] - begin;
            const double* l = lower.data() + value_offsets[a];
            for (std::size_t i = 0; i < rank; ++i) {
                double d = 0.0;
                for (std::size_t k = 0; k <= i; ++k) {
                    const double v = l[i * rank + k];
                    d += v * v;
                }
                if (!(d > 0.0) || !std::isfinite(d)) {
                    throw std::runtime_error("L1 block scalar diagonal invalid");
                }
                inverse[begin + i] = 1.0 / d;
            }
        }
        return inverse;
    }

    double block_entry(std::size_t a, std::size_t i, std::size_t j) const {
        const std::size_t rank = dof_offsets[a + 1U] - dof_offsets[a];
        const double* l = lower.data() + value_offsets[a];
        double value = 0.0;
        const std::size_t stop = std::min(i, j);
        for (std::size_t k = 0; k <= stop; ++k) {
            value += l[i * rank + k] * l[j * rank + k];
        }
        return value;
    }
};

std::array<double, 18> local_rigid_rows(
    const std::array<double, 3>& xyz,
    const gfss::ElasticityAggregateInfo& aggregate) {
    const double scale = aggregate.coordinate_scale > 0.0 ? aggregate.coordinate_scale : 1.0;
    const double x = (xyz[0] - aggregate.centroid[0]) / scale;
    const double y = (xyz[1] - aggregate.centroid[1]) / scale;
    const double z = (xyz[2] - aggregate.centroid[2]) / scale;
    return {
        1.0, 0.0, 0.0,  0.0,  z,   -y,
        0.0, 1.0, 0.0, -z,    0.0,  x,
        0.0, 0.0, 1.0,  y,   -x,    0.0,
    };
}

double local_tentative_basis_value(
    const gfss::ElasticityAggregationCoarseSpace& space,
    std::size_t aggregate_id,
    std::size_t node,
    std::size_t component,
    std::size_t q) {
    if (space.aggregate_of_node[node] != aggregate_id) return 0.0;
    const auto& aggregate = space.aggregates[aggregate_id];
    const auto rows = local_rigid_rows(space.graph.coordinates[node], aggregate);
    const double* row = rows.data() + 6U * component;
    const double* transform = aggregate.rigid_transform.data() + 6U * q;
    double value = 0.0;
    for (std::size_t j = 0; j < 6U; ++j) value += row[j] * transform[j];
    return value;
}

void append_node_touching_elements(
    const gfss::StructuredHexMesh& mesh,
    std::size_t node,
    std::vector<std::size_t>& elements) {
    const std::size_t sx = static_cast<std::size_t>(mesh.nx) + 1U;
    const std::size_t sy = static_cast<std::size_t>(mesh.ny) + 1U;
    const std::size_t i = node % sx;
    const std::size_t t = node / sx;
    const std::size_t j = t % sy;
    const std::size_t k = t / sy;

    const std::size_t ex0 = i == 0U ? 0U : i - 1U;
    const std::size_t ex1 = std::min<std::size_t>(i, mesh.nx - 1U);
    const std::size_t ey0 = j == 0U ? 0U : j - 1U;
    const std::size_t ey1 = std::min<std::size_t>(j, mesh.ny - 1U);
    const std::size_t ez0 = k == 0U ? 0U : k - 1U;
    const std::size_t ez1 = std::min<std::size_t>(k, mesh.nz - 1U);

    for (std::size_t ez = ez0; ez <= ez1; ++ez) {
        for (std::size_t ey = ey0; ey <= ey1; ++ey) {
            for (std::size_t ex = ex0; ex <= ex1; ++ex) {
                elements.push_back(ex + static_cast<std::size_t>(mesh.nx) *
                    (ey + static_cast<std::size_t>(mesh.ny) * ez));
            }
        }
    }
}

void sort_unique(std::vector<std::size_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

std::array<std::uint32_t, 3> decode_element(
    const gfss::StructuredHexMesh& mesh,
    std::size_t element) {
    const std::size_t ex = element % mesh.nx;
    const std::size_t t = element / mesh.nx;
    const std::size_t ey = t % mesh.ny;
    const std::size_t ez = t / mesh.ny;
    return {static_cast<std::uint32_t>(ex),
            static_cast<std::uint32_t>(ey),
            static_cast<std::uint32_t>(ez)};
}

std::array<double, 36> exact_smoothed_aggregate_block(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<std::vector<std::uint32_t>>& aggregate_nodes,
    const std::vector<double>& fine_inverse,
    double omega0,
    std::size_t aggregate_id) {
    const auto& aggregate = space.aggregates[aggregate_id];
    const std::size_t rank = aggregate.rank;
    if (rank == 0U || rank > kCandidates) {
        throw std::runtime_error("L1 localized block aggregate rank invalid");
    }

    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    std::vector<std::size_t> first_ring_elements;
    for (const auto node : aggregate_nodes[aggregate_id]) {
        append_node_touching_elements(mesh, node, first_ring_elements);
    }
    sort_unique(first_ring_elements);

    using NodeColumns = std::array<double, 18>;
    std::unordered_map<std::size_t, NodeColumns> aphi;
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
                for (std::size_t q = 0; q < rank; ++q) {
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
            for (std::size_t q = 0; q < rank; ++q) {
                double value = 0.0;
                for (std::size_t j = 0; j < 24U; ++j) {
                    value += ke[i][j] * phi[j * kCandidates + q];
                }
                out[c * kCandidates + q] += value;
            }
        }
    }

    std::unordered_map<std::size_t, NodeColumns> psi;
    psi.reserve(aphi.size() + aggregate_nodes[aggregate_id].size() + 16U);
    for (const auto& entry : aphi) {
        const std::size_t node = entry.first;
        auto& values = psi[node];
        for (std::size_t c = 0; c < 3U; ++c) {
            const double inv = fine_inverse[3U * node + c];
            for (std::size_t q = 0; q < rank; ++q) {
                values[c * kCandidates + q] =
                    -omega0 * inv * entry.second[c * kCandidates + q];
            }
        }
    }
    for (const auto node_u32 : aggregate_nodes[aggregate_id]) {
        const std::size_t node = node_u32;
        auto& values = psi[node];
        for (std::size_t c = 0; c < 3U; ++c) {
            for (std::size_t q = 0; q < rank; ++q) {
                values[c * kCandidates + q] +=
                    local_tentative_basis_value(space, aggregate_id, node, c, q);
            }
        }
    }

    std::vector<std::size_t> energy_elements;
    energy_elements.reserve(psi.size() * 8U);
    for (const auto& entry : psi) append_node_touching_elements(mesh, entry.first, energy_elements);
    sort_unique(energy_elements);

    std::array<double, 36> block{};
    for (const auto element : energy_elements) {
        const auto ijk = decode_element(mesh, element);
        const auto nodes = mesh.element_nodes(ijk[0], ijk[1], ijk[2]);
        std::array<double, 24U * kCandidates> local{};
        for (std::size_t a = 0; a < 8U; ++a) {
            const auto it = psi.find(static_cast<std::size_t>(nodes[a]));
            if (it == psi.end()) continue;
            for (std::size_t c = 0; c < 3U; ++c) {
                const std::size_t ldof = 3U * a + c;
                for (std::size_t q = 0; q < rank; ++q) {
                    local[ldof * kCandidates + q] = it->second[c * kCandidates + q];
                }
            }
        }
        std::array<double, 24U * kCandidates> klocal{};
        for (std::size_t i = 0; i < 24U; ++i) {
            for (std::size_t q = 0; q < rank; ++q) {
                double value = 0.0;
                for (std::size_t j = 0; j < 24U; ++j) {
                    value += ke[i][j] * local[j * kCandidates + q];
                }
                klocal[i * kCandidates + q] = value;
            }
        }
        for (std::size_t q = 0; q < rank; ++q) {
            for (std::size_t r = 0; r < rank; ++r) {
                double value = 0.0;
                for (std::size_t i = 0; i < 24U; ++i) {
                    value += local[i * kCandidates + q] * klocal[i * kCandidates + r];
                }
                block[6U * q + r] += value;
            }
        }
    }
    return block;
}

L1BlockMetric build_exact_l1_block_metric(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const AlgebraicNodeGraph& graph1,
    const std::vector<double>& fine_inverse,
    double omega0) {
    if (graph1.nodes() != space.aggregates.size() || graph1.dofs() != space.coarse_dofs) {
        throw std::invalid_argument("L1 localized block graph/space mismatch");
    }
    const auto start = Clock::now();
    std::vector<std::vector<std::uint32_t>> aggregate_nodes(space.aggregates.size());
    for (std::size_t node = 0; node < space.aggregate_of_node.size(); ++node) {
        const auto a = space.aggregate_of_node[node];
        if (a != kUnassigned) aggregate_nodes[a].push_back(static_cast<std::uint32_t>(node));
    }

    L1BlockMetric metric;
    metric.dof_offsets = graph1.dof_offsets;
    metric.value_offsets.resize(graph1.nodes() + 1U, 0U);
    metric.min_rank = std::numeric_limits<std::size_t>::max();
    metric.max_rank = 0U;
    for (std::size_t a = 0; a < graph1.nodes(); ++a) {
        const std::size_t rank = graph1.dof_offsets[a + 1U] - graph1.dof_offsets[a];
        if (rank == 0U || rank > kCandidates || rank != space.aggregates[a].rank) {
            throw std::runtime_error("L1 localized block rank mismatch");
        }
        metric.min_rank = std::min(metric.min_rank, rank);
        metric.max_rank = std::max(metric.max_rank, rank);
        metric.value_offsets[a + 1U] = metric.value_offsets[a] + rank * rank;
    }
    metric.lower.assign(metric.value_offsets.back(), 0.0);
    std::vector<double> min_pivots(graph1.nodes(), std::numeric_limits<double>::infinity());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (std::int64_t a64 = 0; a64 < static_cast<std::int64_t>(graph1.nodes()); ++a64) {
        const std::size_t a = static_cast<std::size_t>(a64);
        const std::size_t rank = graph1.dof_offsets[a + 1U] - graph1.dof_offsets[a];
        const auto block = exact_smoothed_aggregate_block(
            mesh, material, space, aggregate_nodes, fine_inverse, omega0, a);
        double* l = metric.lower.data() + metric.value_offsets[a];
        for (std::size_t i = 0; i < rank; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                const double bij = 0.5 * (block[6U * i + j] + block[6U * j + i]);
                double value = bij;
                for (std::size_t k = 0; k < j; ++k) {
                    value -= l[i * rank + k] * l[j * rank + k];
                }
                if (i == j) {
                    if (!(value > 0.0) || !std::isfinite(value)) {
                        throw std::runtime_error("exact L1 diagonal block lost SPD");
                    }
                    min_pivots[a] = std::min(min_pivots[a], value);
                    l[i * rank + j] = std::sqrt(value);
                } else {
                    l[i * rank + j] = value / l[j * rank + j];
                }
            }
        }
    }
    metric.min_cholesky_pivot = *std::min_element(min_pivots.begin(), min_pivots.end());
    metric.setup_ms = elapsed_ms(start, Clock::now());
    return metric;
}

double audit_l1_block_metric(
    const L1BlockMetric& metric,
    const Apply& apply1) {
    if (metric.nodes() == 0U) return 0.0;
    std::vector<std::size_t> samples{
        0U,
        metric.nodes() / 4U,
        metric.nodes() / 2U,
        (3U * metric.nodes()) / 4U,
        metric.nodes() - 1U,
    };
    sort_unique(samples);
    double diff2 = 0.0;
    double ref2 = 0.0;
    for (const auto a : samples) {
        const std::size_t begin = metric.dof_offsets[a];
        const std::size_t rank = metric.dof_offsets[a + 1U] - begin;
        for (std::size_t q = 0; q < rank; ++q) {
            Vec e(metric.dofs(), 0.0);
            e[begin + q] = 1.0;
            const auto y = apply1(e);
            for (std::size_t r = 0; r < rank; ++r) {
                const double oracle = y[begin + r];
                const double local = metric.block_entry(a, r, q);
                const double d = local - oracle;
                diff2 += d * d;
                ref2 += oracle * oracle;
            }
        }
    }
    return ref2 > 0.0 ? std::sqrt(diff2 / ref2) : 0.0;
}

double estimate_lambda_max_l1_block(
    const Apply& apply,
    const L1BlockMetric& metric,
    std::size_t power_iterations) {
    Vec q = deterministic_actual_a2_probe(metric.dofs(), 0.53);
    double n2 = metric.norm2(q);
    if (!(n2 > 0.0)) throw std::runtime_error("L1 block power probe invalid");
    for (double& v : q) v /= std::sqrt(n2);
    double lambda = 0.0;
    for (std::size_t it = 0; it < power_iterations; ++it) {
        const auto aq = apply(q);
        const double rq = dot(q, aq);
        if (!(rq > 0.0) || !std::isfinite(rq)) {
            throw std::runtime_error("L1 block Rayleigh quotient invalid");
        }
        lambda = std::max(lambda, rq);
        auto next = metric.solve(aq);
        n2 = metric.norm2(next);
        if (!(n2 > 0.0) || !std::isfinite(n2)) {
            throw std::runtime_error("L1 block power norm invalid");
        }
        for (double& v : next) v /= std::sqrt(n2);
        q = std::move(next);
    }
    return kLambdaSafety * lambda;
}

void chebyshev_l1_block_smooth(
    const Apply& apply,
    const L1BlockMetric& metric,
    double lambda_max,
    const Vec& b,
    Vec& x,
    std::size_t degree) {
    if (degree == 0U) return;
    const double lambda_low = kChebyshevLowerFraction * lambda_max;
    const double theta = 0.5 * (lambda_max + lambda_low);
    const double delta = 0.5 * (lambda_max - lambda_low);
    for (std::size_t k = 0; k < degree; ++k) {
        const double angle = kPi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        const double root = theta + delta * std::cos(angle);
        const auto ax = apply(x);
        Vec residual(b.size(), 0.0);
        for (std::size_t i = 0; i < b.size(); ++i) residual[i] = b[i] - ax[i];
        const auto correction = metric.solve(residual);
        const double weight = 1.0 / root;
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += weight * correction[i];
    }
}

struct L1BlockSmoothedTransfer {
    const CandidateTransfer& tentative;
    const Apply& apply_fine;
    const L1BlockMetric& metric;
    double omega{0.0};
    std::size_t steps{0};

    Vec prolong(const Vec& coarse) const {
        auto fine = tentative.prolong(coarse);
        for (std::size_t step = 0; step < steps; ++step) {
            const auto af = apply_fine(fine);
            const auto scaled = metric.solve(af);
            for (std::size_t i = 0; i < fine.size(); ++i) fine[i] -= omega * scaled[i];
        }
        return fine;
    }
    Vec restrict_transpose(const Vec& fine) const {
        auto work = fine;
        for (std::size_t step = 0; step < steps; ++step) {
            const auto scaled = metric.solve(work);
            const auto a_scaled = apply_fine(scaled);
            for (std::size_t i = 0; i < work.size(); ++i) work[i] -= omega * a_scaled[i];
        }
        return tentative.restrict_transpose(work);
    }
};

double l1_block_transfer_adjoint_error(const L1BlockSmoothedTransfer& transfer) {
    const auto coarse = deterministic_actual_a2_probe(transfer.tentative.coarse_dofs, 0.17);
    const auto fine = deterministic_actual_a2_probe(transfer.tentative.fine_dofs, 0.73);
    const auto pc = transfer.prolong(coarse);
    const auto ptf = transfer.restrict_transpose(fine);
    const double lhs = dot(pc, fine);
    const double rhs = dot(coarse, ptf);
    return std::abs(lhs - rhs) / std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

double l1_block_candidate_error(
    const L1BlockSmoothedTransfer& transfer,
    const std::vector<double>& fine_candidates) {
    double worst = 0.0;
    for (std::size_t c = 0; c < kCandidates; ++c) {
        Vec fine(transfer.tentative.fine_dofs, 0.0);
        Vec coarse(transfer.tentative.coarse_dofs, 0.0);
        for (std::size_t i = 0; i < fine.size(); ++i) fine[i] = fine_candidates[i * kCandidates + c];
        for (std::size_t i = 0; i < coarse.size(); ++i) {
            coarse[i] = transfer.tentative.coarse_candidates[i * kCandidates + c];
        }
        const auto reproduced = transfer.prolong(coarse);
        Vec diff(fine.size(), 0.0);
        for (std::size_t i = 0; i < fine.size(); ++i) diff[i] = reproduced[i] - fine[i];
        const double fn = norm(fine);
        if (fn > 0.0) worst = std::max(worst, norm(diff) / fn);
    }
    return worst;
}

void run_l1_block_reference(
    std::size_t m0,
    std::size_t m1,
    std::size_t m2,
    std::size_t max_cycles,
    std::size_t target_nodes,
    std::size_t min_nodes) {
    if (m0 != 1U) {
        throw std::invalid_argument("localized exact L1 block reference currently requires m0=1");
    }
    const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
    const gfss::Material material{210.0e9, 0.30};
    const auto setup_start = Clock::now();

    auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
    const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
        std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
    const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(mesh, material, space0);
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
    const double tentative_candidate1_error = candidate_reproduction_error(tentative_transfer1, candidates1);

    const auto block1 = build_exact_l1_block_metric(
        mesh, material, space0, graph1, fine_inverse, omega0);
    const double block1_oracle_error = audit_l1_block_metric(block1, apply1);
    const auto actual_inverse1 = block1.inverse_scalar_diagonal();
    const double tentative_scalar_lambda1 = estimate_lambda_max(apply1, tentative_a1.inverse_diagonal, 8U);
    const double actual_scalar_lambda1 = estimate_lambda_max(apply1, actual_inverse1, 8U);
    const double block_lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
    const double block_omega1 = kSaDampingNumerator / block_lambda1;

    const L1BlockSmoothedTransfer transfer1{
        tentative_transfer1, apply1, block1, block_omega1, m1};
    const double transfer1_adjoint = l1_block_transfer_adjoint_error(transfer1);
    const double smoothed_candidate1_error = l1_block_candidate_error(transfer1, candidates1);

    const Apply apply2 = [&](const Vec& x) {
        const auto l1 = transfer1.prolong(x);
        return transfer1.restrict_transpose(apply1(l1));
    };
    const auto inverse2 = tentative_transfer1.approximate_inverse_coarse_diagonal(actual_inverse1);
    const double lambda2 = estimate_lambda_max(apply2, inverse2, 8U);
    const double omega2 = kSaDampingNumerator / lambda2;
    const auto tentative_transfer2 = build_candidate_transfer(
        tentative_transfer1.coarse_graph,
        tentative_transfer1.coarse_candidates,
        target_nodes,
        min_nodes,
        1.0e-10);
    const AlgebraicSmoothedTransferActualA2 transfer2{
        tentative_transfer2, apply2, inverse2, omega2, m2};
    const double transfer2_adjoint = smoothed_transfer_adjoint_error_actual_a2(transfer2);

    const Apply apply3 = [&](const Vec& x) {
        const auto l2 = transfer2.prolong(x);
        return transfer2.restrict_transpose(apply2(l2));
    };
    const auto bottom_start = Clock::now();
    const auto bottom = materialize_and_factor_bottom(apply3, tentative_transfer2.coarse_dofs);
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
    levels[1].label = "L1_exact_actual_block_Jacobi";
    levels[1].diagnostic_lambda_max = block_lambda1;
    levels[1].apply = apply1;
    levels[1].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_l1_block_smooth(apply1, block1, block_lambda1, b, x, degree);
    };
    levels[1].restrict_to_coarse = [&](const Vec& r) { return transfer1.restrict_transpose(r); };
    levels[1].prolong_from_coarse = [&](const Vec& c) { return transfer1.prolong(c); };

    levels[2].dofs = tentative_transfer1.coarse_dofs;
    levels[2].label = "L2_propagated_from_actual_L1_diagonal";
    levels[2].diagnostic_lambda_max = lambda2;
    levels[2].apply = apply2;
    levels[2].smooth = [&](const Vec& b, Vec& x, std::size_t degree) {
        chebyshev_smooth(apply2, inverse2, lambda2, b, x, degree);
    };
    levels[2].restrict_to_coarse = [&](const Vec& r) { return transfer2.restrict_transpose(r); };
    levels[2].prolong_from_coarse = [&](const Vec& c) { return transfer2.prolong(c); };

    levels[3].dofs = tentative_transfer2.coarse_dofs;
    levels[3].label = "L3_dense_Cholesky_bottom";
    levels[3].apply = apply3;
    levels[3].bottom_solve = [&](const Vec& b) { return bottom.solve(b); };

    const auto rhs = make_rhs(mesh);
    const auto result = gfss::solve_reference_multilevel_vcycle(
        levels, rhs, 1.0e-6, max_cycles, 3U, 3U);

    std::cout << "\n========================================\n"
              << "L0_transfer_smoothing_steps=" << m0
              << " L1_transfer_smoothing_steps=" << m1
              << " L2_transfer_smoothing_steps=" << m2 << '\n'
              << "hierarchy_levels=4\n"
              << "L1_metric_source=localized_exact_blocks_of_actual_smoothed_A1\n"
              << "L1_relaxation=rank_aware_block_Jacobi\n"
              << "L1_transfer_smoothing_metric=rank_aware_block_Jacobi\n"
              << "L2_metric=scalar_propagated_from_actual_L1_block_diagonal\n"
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
              << "L1_local_block_vs_nested_oracle_relative_error=" << block1_oracle_error
              << " L2_transfer_adjoint_relative_error=" << transfer2_adjoint << '\n'
              << std::fixed << std::setprecision(6)
              << "L1_block_setup_ms=" << block1.setup_ms
              << " L1_block_nodes=" << block1.nodes()
              << " L1_block_rank_min=" << block1.min_rank
              << " L1_block_rank_max=" << block1.max_rank
              << " L1_block_storage_bytes=" << block1.storage_bytes()
              << " L1_block_min_cholesky_pivot=" << block1.min_cholesky_pivot << '\n'
              << "lambda0=" << lambda0 << " omega0=" << omega0
              << " tentative_scalar_lambda1_reference=" << tentative_scalar_lambda1
              << " actual_scalar_lambda1_reference=" << actual_scalar_lambda1
              << " block_lambda1=" << block_lambda1
              << " block_omega1=" << block_omega1
              << " lambda2=" << lambda2 << " omega2=" << omega2 << '\n'
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
            std::cout << " cycle_q=" << result.relative_residuals[i] / result.relative_residuals[i - 1U];
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
        const std::size_t m0 = argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 1U;
        const std::size_t m1 = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 2U;
        const std::size_t m2 = argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 1U;
        const std::size_t max_cycles = argc > 4 ? static_cast<std::size_t>(std::stoull(argv[4])) : 6U;
        const std::size_t target_nodes = argc > 5 ? static_cast<std::size_t>(std::stoull(argv[5])) : 12U;
        const std::size_t min_nodes = argc > 6 ? static_cast<std::size_t>(std::stoull(argv[6])) : 4U;
        if (m0 != 1U || m1 > 4U || m2 > 4U || max_cycles == 0U ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid localized L1 block-Jacobi reference options");
        }
        std::cout << "GFSS M5 localized actual-L1 block-Jacobi recursive SA reference\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "purpose=test_actual_L1_block_metric_without_dense_coarse_materialization\n"
                  << "reference_execution=cpu_fp64\n"
                  << "L0_transfer_smoothing_steps_required=1\n"
                  << "L1_blocks=exact_local_energy_of_smoothed_aggregate_basis\n"
                  << "L1_full_matrix=not_materialized\n"
                  << "L2_full_matrix=not_materialized\n"
                  << "all_transfer_restrictions=exact_transpose\n"
                  << "pre_smooth_degree=3 post_smooth_degree=3\n"
                  << "acceptance_target_post_transient_q<=0.4\n"
                  << "performance_status=numerical_reference_only\n";
        run_l1_block_reference(m0, m1, m2, max_cycles, target_nodes, min_nodes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_recursive_sa_l1_block_jacobi_reference_bench "
                  << "[m0=1 [m1 [m2 [max_cycles [target_nodes [min_nodes]]]]]]\n";
        return 1;
    }
}
