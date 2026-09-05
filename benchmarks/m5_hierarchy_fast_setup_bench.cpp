// M5 stage 15: production hierarchy setup optimization.
// Reuse A1*P1 local columns, parallelize independent aggregate work, then use
// the exact dense A2 already required by runtime to build lambda2, dense P2 and
// A3 directly instead of recursively re-applying localized A1/A2 machinery.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_p1_block6_setup.hpp"
#include "m5_l2_dense_setup.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using FastClock = std::chrono::steady_clock;

double fast_ms(FastClock::time_point a, FastClock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double rel_error(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("fast setup oracle size mismatch");
    Vec d(a.size(), 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) d[i] = a[i] - b[i];
    return norm(d) / std::max(norm(b), 1.0e-300);
}

std::vector<float> fast_symmetric_inverse_col_major(const DenseCholesky& factor) {
    const std::size_t n = factor.n;
    std::vector<double> inverse(n * n, 0.0);
    for (std::size_t j = 0U; j < n; ++j) {
        Vec e(n, 0.0);
        e[j] = 1.0;
        const auto column = factor.solve(e);
        for (std::size_t i = 0U; i < n; ++i) inverse[i * n + j] = column[i];
    }
    std::vector<float> out(n * n, 0.0f);
    for (std::size_t i = 0U; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            const float value = static_cast<float>(
                0.5 * (inverse[i * n + j] + inverse[j * n + i]));
            out[j * n + i] = value;
            out[i * n + j] = value;
        }
    }
    return out;
}

template <class LocalApply>
std::vector<LocalColumns> build_smoothed_supports_parallel(
    const CandidateTransfer& transfer,
    const AlgebraicNodeGraph& fine_graph,
    const L1BlockMetric& fine_metric,
    double omega,
    std::size_t steps,
    const LocalApply& local_apply) {
    const auto layout = make_dof_layout(fine_graph);
    std::vector<LocalColumns> supports(transfer.aggregates.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::int64_t a64 = 0; a64 < static_cast<std::int64_t>(transfer.aggregates.size()); ++a64) {
        const std::size_t a = static_cast<std::size_t>(a64);
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
std::vector<LocalColumns> apply_supports_parallel(
    const std::vector<LocalColumns>& supports,
    const LocalApply& local_apply) {
    std::vector<LocalColumns> applied(supports.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::int64_t a64 = 0; a64 < static_cast<std::int64_t>(supports.size()); ++a64) {
        const std::size_t a = static_cast<std::size_t>(a64);
        applied[a] = local_apply(supports[a]);
    }
    return applied;
}

L1BlockMetric metric_from_cached_applied(
    const CandidateTransfer& transfer,
    const L1BlockMetric& row_metric,
    const std::vector<LocalColumns>& supports,
    const std::vector<LocalColumns>& applied) {
    if (supports.size() != transfer.aggregates.size() || applied.size() != supports.size()) {
        throw std::invalid_argument("cached metric support size mismatch");
    }
    L1BlockMetric metric;
    metric.dof_offsets = transfer.coarse_graph.dof_offsets;
    metric.value_offsets.resize(supports.size() + 1U, 0U);
    metric.min_rank = std::numeric_limits<std::size_t>::max();
    metric.max_rank = 0U;
    for (std::size_t a = 0; a < supports.size(); ++a) {
        const std::size_t rank = metric.dof_offsets[a + 1U] - metric.dof_offsets[a];
        if (rank == 0U || rank > kCandidates || supports[a].cols != rank || applied[a].cols != rank) {
            throw std::runtime_error("cached metric rank mismatch");
        }
        metric.min_rank = std::min(metric.min_rank, rank);
        metric.max_rank = std::max(metric.max_rank, rank);
        metric.value_offsets[a + 1U] = metric.value_offsets[a] + rank * rank;
    }
    metric.lower.assign(metric.value_offsets.back(), 0.0);
    std::vector<double> pivots(supports.size(), std::numeric_limits<double>::infinity());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t a64 = 0; a64 < static_cast<std::int64_t>(supports.size()); ++a64) {
        const std::size_t a = static_cast<std::size_t>(a64);
        const std::size_t rank = supports[a].cols;
        const auto block = local_cross_gram(supports[a], applied[a], row_metric);
        double* l = metric.lower.data() + metric.value_offsets[a];
        double min_pivot = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < rank; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double value = 0.5 * (block[i * kCandidates + j] + block[j * kCandidates + i]);
                for (std::size_t k = 0; k < j; ++k) value -= l[i * rank + k] * l[j * rank + k];
                if (i == j) {
                    if (!(value > 0.0) || !std::isfinite(value)) {
                        throw std::runtime_error("cached L2 metric lost SPD");
                    }
                    min_pivot = std::min(min_pivot, value);
                    l[i * rank + j] = std::sqrt(value);
                } else {
                    l[i * rank + j] = value / l[j * rank + j];
                }
            }
        }
        pivots[a] = min_pivot;
    }
    metric.min_cholesky_pivot = *std::min_element(pivots.begin(), pivots.end());
    return metric;
}

m5_l2_setup::DenseA2 dense_a2_from_cached_applied(
    const CandidateTransfer& transfer1,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis,
    const std::vector<LocalColumns>& applied_l2_basis) {
    m5_l2_setup::DenseA2 out;
    const auto start = FastClock::now();
    out.n = transfer1.coarse_dofs;
    out.fp64.assign(out.n * out.n, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::int64_t j64 = 0; j64 < static_cast<std::int64_t>(l2_basis.size()); ++j64) {
        const std::size_t jnode = static_cast<std::size_t>(j64);
        const auto& applied = applied_l2_basis[jnode];
        const std::size_t joff = transfer1.aggregates[jnode].coarse_offset;
        const std::size_t jrank = transfer1.aggregates[jnode].rank;
        for (std::size_t inode = 0; inode < l2_basis.size(); ++inode) {
            const auto block = local_cross_gram(l2_basis[inode], applied, block1);
            const std::size_t ioff = transfer1.aggregates[inode].coarse_offset;
            const std::size_t irank = transfer1.aggregates[inode].rank;
            for (std::size_t i = 0; i < irank; ++i) {
                for (std::size_t j = 0; j < jrank; ++j) {
                    out.fp64[(ioff + i) * out.n + (joff + j)] = block[i * kCandidates + j];
                }
            }
        }
    }
    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0; i < out.n; ++i) {
        const double d = out.fp64[i * out.n + i];
        if (!(d > 0.0) || !std::isfinite(d)) throw std::runtime_error("cached A2 diagonal invalid");
        norm2 += d * d;
        for (std::size_t j = i + 1U; j < out.n; ++j) {
            const double aij = out.fp64[i * out.n + j];
            const double aji = out.fp64[j * out.n + i];
            const double diff = aij - aji;
            asym2 += 2.0 * diff * diff;
            norm2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            out.fp64[i * out.n + j] = sym;
            out.fp64[j * out.n + i] = sym;
        }
    }
    out.symmetry_relative_defect = norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;
    out.assembly_ms = fast_ms(start, FastClock::now());
    return out;
}

void dense_apply_matrix(const m5_l2_setup::DenseA2& a,
                        const std::vector<double>& x,
                        std::size_t cols,
                        std::vector<double>& y) {
    if (cols == 0U || x.size() != a.n * cols) throw std::invalid_argument("dense matrix apply shape");
    y.assign(a.n * cols, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(a.n); ++i64) {
        const std::size_t i = static_cast<std::size_t>(i64);
        double* yi = y.data() + i * cols;
        const double* ai = a.fp64.data() + i * a.n;
        for (std::size_t k = 0; k < a.n; ++k) {
            const double aik = ai[k];
            if (aik == 0.0) continue;
            const double* xk = x.data() + k * cols;
            for (std::size_t j = 0; j < cols; ++j) yi[j] += aik * xk[j];
        }
    }
}

void block_solve_matrix(const L1BlockMetric& metric,
                        const std::vector<double>& rhs,
                        std::size_t cols,
                        std::vector<double>& x) {
    if (rhs.size() != metric.dofs() * cols) throw std::invalid_argument("block matrix solve shape");
    x.assign(rhs.size(), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t a64 = 0; a64 < static_cast<std::int64_t>(metric.nodes()); ++a64) {
        const std::size_t a = static_cast<std::size_t>(a64);
        const std::size_t begin = metric.dof_offsets[a];
        const std::size_t rank = metric.dof_offsets[a + 1U] - begin;
        const double* l = metric.lower.data() + metric.value_offsets[a];
        for (std::size_t col = 0; col < cols; ++col) {
            std::array<double, kCandidates> y{};
            std::array<double, kCandidates> z{};
            for (std::size_t i = 0; i < rank; ++i) {
                double value = rhs[(begin + i) * cols + col];
                for (std::size_t j = 0; j < i; ++j) value -= l[i * rank + j] * y[j];
                y[i] = value / l[i * rank + i];
            }
            for (std::size_t ii = rank; ii-- > 0U;) {
                double value = y[ii];
                for (std::size_t j = ii + 1U; j < rank; ++j) value -= l[j * rank + ii] * z[j];
                z[ii] = value / l[ii * rank + ii];
            }
            for (std::size_t i = 0; i < rank; ++i) x[(begin + i) * cols + col] = z[i];
        }
    }
}

m5_l2_setup::DenseP2 dense_smoothed_p2_from_a2(
    const CandidateTransfer& transfer2,
    const L1BlockMetric& block2,
    const m5_l2_setup::DenseA2& a2,
    double omega2) {
    const auto start = FastClock::now();
    m5_l2_setup::DenseP2 p;
    p.rows = block2.dofs();
    p.cols = transfer2.coarse_dofs;
    p.fp64.assign(p.rows * p.cols, 0.0);
    for (const auto& aggregate : transfer2.aggregates) {
        for (std::size_t row = 0; row < aggregate.fine_dofs.size(); ++row) {
            const std::size_t fine = aggregate.fine_dofs[row];
            for (std::size_t q = 0; q < aggregate.rank; ++q) {
                p.fp64[fine * p.cols + aggregate.coarse_offset + q] =
                    aggregate.q_values[row * aggregate.rank + q];
            }
        }
    }
    std::vector<double> ap;
    std::vector<double> scaled;
    dense_apply_matrix(a2, p.fp64, p.cols, ap);
    block_solve_matrix(block2, ap, p.cols, scaled);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(p.fp64.size()); ++i64) {
        const std::size_t i = static_cast<std::size_t>(i64);
        p.fp64[i] -= omega2 * scaled[i];
    }
    p.assembly_ms = fast_ms(start, FastClock::now());
    return p;
}

LocalBottomReference dense_bottom_from_a2_p2(
    const m5_l2_setup::DenseA2& a2,
    const m5_l2_setup::DenseP2& p2) {
    const auto start = FastClock::now();
    if (p2.rows != a2.n) throw std::invalid_argument("dense bottom P2/A2 shape mismatch");
    std::vector<double> ap;
    dense_apply_matrix(a2, p2.fp64, p2.cols, ap);
    LocalBottomReference bottom;
    bottom.values.assign(p2.cols * p2.cols, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(p2.cols); ++i64) {
        const std::size_t i = static_cast<std::size_t>(i64);
        for (std::size_t j = 0; j < p2.cols; ++j) {
            double value = 0.0;
            for (std::size_t r = 0; r < p2.rows; ++r) {
                value += p2.fp64[r * p2.cols + i] * ap[r * p2.cols + j];
            }
            bottom.values[i * p2.cols + j] = value;
        }
    }
    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0; i < p2.cols; ++i) {
        norm2 += bottom.values[i * p2.cols + i] * bottom.values[i * p2.cols + i];
        for (std::size_t j = i + 1U; j < p2.cols; ++j) {
            const double aij = bottom.values[i * p2.cols + j];
            const double aji = bottom.values[j * p2.cols + i];
            const double d = aij - aji;
            asym2 += 2.0 * d * d;
            norm2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            bottom.values[i * p2.cols + j] = sym;
            bottom.values[j * p2.cols + i] = sym;
        }
    }
    bottom.factor.n = p2.cols;
    bottom.factor.symmetry_relative_defect = norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;
    bottom.factor.lower.assign(p2.cols * p2.cols, 0.0);
    bottom.factor.min_pivot = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < p2.cols; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = bottom.values[i * p2.cols + j];
            for (std::size_t k = 0; k < j; ++k) {
                value -= bottom.factor.lower[i * p2.cols + k] * bottom.factor.lower[j * p2.cols + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) throw std::runtime_error("dense bottom lost SPD");
                bottom.factor.min_pivot = std::min(bottom.factor.min_pivot, value);
                bottom.factor.lower[i * p2.cols + j] = std::sqrt(value);
            } else {
                bottom.factor.lower[i * p2.cols + j] = value / bottom.factor.lower[j * p2.cols + j];
            }
        }
    }
    bottom.assembly_ms = fast_ms(start, FastClock::now());
    return bottom;
}

Vec dense_p2_apply(const m5_l2_setup::DenseP2& p, const Vec& x) {
    return m5_l2_setup::apply_p2(p, x);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t target_nodes = argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 12U;
        const std::size_t min_nodes = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 4U;
        if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid fast hierarchy options");
        }
        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto total_start = FastClock::now();

        auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
        const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(mesh, material, space0);
        const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
        const Apply apply0 = [&](const Vec& x) { return apply_fine_clamped(mesh, material, x); };
        const double lambda0 = estimate_lambda_max(apply0, fine_inverse, 8U);
        const double omega0 = kSaDampingNumerator / lambda0;
        const FineSmoothedTransfer transfer0{mesh, material, space0, fine_inverse, omega0, m0};
        const Apply apply1 = [&](const Vec& x) { return transfer0.restrict_transpose(apply0(transfer0.prolong(x))); };
        const auto graph1_tentative = graph_from_variable_blocks(tentative_a1);
        const auto candidates1 = make_level1_candidates(space0);
        const auto block1 = build_exact_l1_block_metric(mesh, material, space0, graph1_tentative, fine_inverse, omega0);
        const double lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
        const double omega1 = kSaDampingNumerator / lambda1;
        double p0_support_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(mesh, material, space0, fine_inverse, omega0, p0_support_ms);
        const auto element_supports = build_element_support_index(mesh, fine_supports);
        const auto actual_a1_offdiagonal = accumulate_combined_actual_a1_offdiagonal_blocks(mesh, material, fine_supports, element_supports);
        const auto strength1 = build_combined_strength_graph(graph1_tentative, block1, actual_a1_offdiagonal, strength_threshold);
        const auto transfer1 = build_candidate_transfer(strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
        const L1BlockSmoothedTransfer transfer1_nested{transfer1, apply1, block1, omega1, m1};
        const Apply apply2_nested = [&](const Vec& x) { return transfer1_nested.restrict_transpose(apply1(transfer1_nested.prolong(x))); };
        const auto local_a1 = [&](const LocalColumns& x) { return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x); };

        const auto l2_basis_start = FastClock::now();
        const auto l2_basis = build_smoothed_supports_parallel(transfer1, strength1.graph, block1, omega1, m1, local_a1);
        const double l2_basis_ms = fast_ms(l2_basis_start, FastClock::now());

        const auto cached_apply_start = FastClock::now();
        const auto applied_l2_basis = apply_supports_parallel(l2_basis, local_a1);
        const double cached_apply_ms = fast_ms(cached_apply_start, FastClock::now());

        const auto metric_start = FastClock::now();
        const auto block2 = metric_from_cached_applied(transfer1, block1, l2_basis, applied_l2_basis);
        const double block2_ms = fast_ms(metric_start, FastClock::now());

        const auto payload1_start = FastClock::now();
        const auto p1 = m5_p1_setup::assemble_dual_order_block6(transfer1, block1, l2_basis);
        const auto inverse1 = m5_l2_setup::inverse_blocks_6x6_fp32(block1);
        const auto inverse2 = m5_l2_setup::inverse_blocks_6x6_fp32(block2);
        const double p1_payload_ms = fast_ms(payload1_start, FastClock::now());

        const auto a2_start = FastClock::now();
        const auto a2 = dense_a2_from_cached_applied(transfer1, block1, l2_basis, applied_l2_basis);
        const double a2_ms = fast_ms(a2_start, FastClock::now());

        const Apply apply2_dense = [&](const Vec& x) { return m5_l2_setup::apply_dense_a2(a2, x); };
        const auto lambda2_start = FastClock::now();
        const double lambda2 = estimate_lambda_max_l1_block(apply2_dense, block2, 8U);
        const double omega2 = kSaDampingNumerator / lambda2;
        const double lambda2_ms = fast_ms(lambda2_start, FastClock::now());

        const auto transfer2 = build_candidate_transfer(transfer1.coarse_graph, transfer1.coarse_candidates,
                                                        target_nodes, min_nodes, 1.0e-10);
        const auto p2_start = FastClock::now();
        const auto p2 = dense_smoothed_p2_from_a2(transfer2, block2, a2, omega2);
        const double p2_ms = fast_ms(p2_start, FastClock::now());
        const auto bottom_start = FastClock::now();
        const auto bottom = dense_bottom_from_a2_p2(a2, p2);
        const double bottom_ms = fast_ms(bottom_start, FastClock::now());

        const auto final_payload_start = FastClock::now();
        const auto bottom_inverse = fast_symmetric_inverse_col_major(bottom.factor);
        const auto a2_fp32 = m5_l2_setup::to_float(a2.fp64);
        const auto p2_fp32 = m5_l2_setup::to_float(p2.fp64);
        const double final_payload_ms = fast_ms(final_payload_start, FastClock::now());
        const double production_total_ms = fast_ms(total_start, FastClock::now());

        // Independent probes against the established nested hierarchy.
        const auto oracle_start = FastClock::now();
        const double block2_error = audit_l1_block_metric(block2, apply2_nested);
        const auto probe2 = deterministic_actual_a2_probe(a2.n, 0.63);
        const double a2_error = rel_error(apply2_dense(probe2), apply2_nested(probe2));
        const L1BlockSmoothedTransfer transfer2_nested{transfer2, apply2_nested, block2, omega2, m2};
        const auto probe3 = deterministic_actual_a2_probe(transfer2.coarse_dofs, 0.91);
        const double p2_error = rel_error(dense_p2_apply(p2, probe3), transfer2_nested.prolong(probe3));
        const Apply apply3_nested = [&](const Vec& x) {
            return transfer2_nested.restrict_transpose(apply2_nested(transfer2_nested.prolong(x)));
        };
        const double bottom_error = bottom_local_oracle_error(bottom, apply3_nested);
        const double oracle_ms = fast_ms(oracle_start, FastClock::now());
        const bool oracle_ok = block2_error <= 1.0e-10 && a2_error <= 1.0e-10 &&
                               p2_error <= 1.0e-10 && bottom_error <= 1.0e-10;

        std::cout << "GFSS M5 cached/parallel exact hierarchy setup\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "method=parallel_L2_basis_cached_A1P1_dense_A2_driven_L3\n"
#ifdef _OPENMP
                  << "openmp_enabled=true\n"
#else
                  << "openmp_enabled=false\n"
#endif
                  << "fine_dofs=" << mesh.dof_count() << " L1_dofs=" << space0.coarse_dofs
                  << " L2_dofs=" << block2.dofs() << " L3_dofs=" << bottom.factor.n << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L2_block_vs_nested_relative_error=" << block2_error
                  << " A2_dense_vs_nested_relative_error=" << a2_error
                  << " P2_dense_vs_factorized_relative_error=" << p2_error
                  << " bottom_dense_vs_nested_relative_error=" << bottom_error
                  << " oracle_accept_1e-10=" << (oracle_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "production_required_setup_ms=" << production_total_ms
                  << " validation_oracle_ms=" << oracle_ms << '\n'
                  << "fast_L2_basis_ms=" << l2_basis_ms
                  << " cached_final_A1P1_ms=" << cached_apply_ms
                  << " fast_L2_block_metric_ms=" << block2_ms
                  << " P1_plus_block_inverses_ms=" << p1_payload_ms
                  << " fast_A2_dense_ms=" << a2_ms
                  << " fast_lambda2_ms=" << lambda2_ms
                  << " fast_P2_dense_ms=" << p2_ms
                  << " fast_bottom_A3_ms=" << bottom_ms
                  << " final_FP32_payload_ms=" << final_payload_ms << '\n'
                  << "payload_P1_nnz=" << p1.forward_column_indices.size()
                  << " inverse1_values=" << inverse1.size()
                  << " inverse2_values=" << inverse2.size()
                  << " A2_fp32_values=" << a2_fp32.size()
                  << " P2_fp32_values=" << p2_fp32.size()
                  << " bottom_inverse_values=" << bottom_inverse.size() << '\n'
                  << "A2_symmetry_relative_defect=" << std::scientific << a2.symmetry_relative_defect
                  << " bottom_symmetry_relative_defect=" << bottom.factor.symmetry_relative_defect << '\n';
        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_hierarchy_fast_setup_bench [target_nodes=12 [min_nodes=4]]\n";
        return 1;
    }
}
