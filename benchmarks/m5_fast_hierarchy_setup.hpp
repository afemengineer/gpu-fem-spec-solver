#pragma once

// Exact setup accelerators for the frozen M5 hierarchy. Include after
// recursive_sa_local_l2_helpers.inc and m5_l2_dense_setup.hpp so the reference
// hierarchy types are visible. These helpers change only construction strategy:
// they cache A1*P1 local columns, parallelize independent aggregate work, and
// use exact dense A2 to build lambda2, P2 and A3.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace m5_fast_setup {

using Clock = std::chrono::steady_clock;

inline double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

template <class LocalApply>
inline std::vector<LocalColumns> build_smoothed_supports_parallel(
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
inline std::vector<LocalColumns> apply_supports_parallel(
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

inline L1BlockMetric metric_from_cached_applied(
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

inline m5_l2_setup::DenseA2 dense_a2_from_cached_applied(
    const CandidateTransfer& transfer1,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis,
    const std::vector<LocalColumns>& applied_l2_basis) {
    m5_l2_setup::DenseA2 out;
    const auto start = Clock::now();
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
    out.assembly_ms = elapsed_ms(start, Clock::now());
    return out;
}

inline void dense_apply_matrix(const m5_l2_setup::DenseA2& a,
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

inline void block_solve_matrix(const L1BlockMetric& metric,
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

inline m5_l2_setup::DenseP2 dense_smoothed_p2_from_a2(
    const CandidateTransfer& transfer2,
    const L1BlockMetric& block2,
    const m5_l2_setup::DenseA2& a2,
    double omega2) {
    const auto start = Clock::now();
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
    p.assembly_ms = elapsed_ms(start, Clock::now());
    return p;
}

inline LocalBottomReference dense_bottom_from_a2_p2(
    const m5_l2_setup::DenseA2& a2,
    const m5_l2_setup::DenseP2& p2) {
    const auto start = Clock::now();
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
                value -= bottom.factor.lower[i * p2.cols + k] *
                         bottom.factor.lower[j * p2.cols + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("dense bottom lost SPD");
                }
                bottom.factor.min_pivot = std::min(bottom.factor.min_pivot, value);
                bottom.factor.lower[i * p2.cols + j] = std::sqrt(value);
            } else {
                bottom.factor.lower[i * p2.cols + j] = value / bottom.factor.lower[j * p2.cols + j];
            }
        }
    }
    bottom.assembly_ms = elapsed_ms(start, Clock::now());
    return bottom;
}

inline std::vector<float> symmetric_inverse_col_major(const DenseCholesky& factor) {
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

}  // namespace m5_fast_setup
