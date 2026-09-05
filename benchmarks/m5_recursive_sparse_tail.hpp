#pragma once

// Recursive exact sparse coarse tail for M5.  Starting from any variable-rank
// algebraic level represented by CandidateTransfer/L1BlockMetric and an exact
// block-sparse Galerkin operator, repeatedly build the same one-step smoothed
// aggregation transfer and exact support-pruned Galerkin operator until the
// coarse dimension is small enough for a dense direct bottom solve.
//
// This is setup/reference infrastructure.  Runtime representation selection is
// recorded per level, but the CUDA V-cycle is not changed by this header.

#include "m5_sparse_a2_setup.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace m5_recursive_tail {

using Clock = std::chrono::steady_clock;

struct LevelTelemetry {
    std::size_t level{0U};
    std::size_t nodes{0U};
    std::size_t dofs{0U};
    std::size_t scalar_nnz{0U};
    std::size_t directed_blocks{0U};
    double scalar_density{0.0};
    std::size_t dense_fp32_bytes{0U};
    std::size_t sparse_fp32_bytes{0U};
    std::string memory_preferred_runtime_representation;

    double lambda{0.0};
    double omega{0.0};
    double lambda_ms{0.0};
    double transfer_ms{0.0};
    double smoothed_support_ms{0.0};
    double applied_support_ms{0.0};
    double metric_ms{0.0};
    double galerkin_ms{0.0};
    std::size_t next_nodes{0U};
    std::size_t next_dofs{0U};
    std::size_t transfer_support_blocks{0U};
};

struct SparseCoarseLevel {
    std::size_t level{0U};
    CandidateTransfer layout;
    L1BlockMetric metric;
    m5_sparse_a2::ExactSparseA2 op;
    double lambda{0.0};
    double omega{0.0};

    // Present only for a non-bottom level.  These localized block columns are
    // the exact one-step smoothed prolongation from level+1 to this level.
    CandidateTransfer transfer_to_next;
    std::vector<LocalColumns> prolong_supports;
};

struct RecursiveSparseTail {
    std::vector<SparseCoarseLevel> levels;
    std::vector<LevelTelemetry> telemetry;
    LocalBottomReference bottom;
    std::size_t bottom_level{0U};
    std::size_t bottom_dofs{0U};
    std::size_t dense_bottom_threshold{0U};
    double bottom_materialize_factor_ms{0.0};
    double bottom_sparse_apply_relative_error{0.0};
    double bottom_solve_relative_residual{0.0};
    double total_ms{0.0};
};

inline std::size_t support_block_count(const std::vector<LocalColumns>& supports) {
    std::size_t count = 0U;
    for (const auto& support : supports) count += support.values.size();
    return count;
}

class SparseLocalApply {
public:
    SparseLocalApply(const m5_sparse_a2::ExactSparseA2& op,
                     const CandidateTransfer& layout)
        : op_(op), layout_(layout), by_column_(op.nodes + 1U, 0U) {
        if (layout_.aggregates.size() != op_.nodes ||
            op_.block_row_offsets.size() != op_.nodes + 1U) {
            throw std::invalid_argument("recursive sparse local apply layout mismatch");
        }
        std::vector<std::size_t> counts(op_.nodes, 0U);
        for (const auto& block : op_.blocks) {
            if (block.col_node >= op_.nodes || block.row_node >= op_.nodes) {
                throw std::out_of_range("recursive sparse local block node out of range");
            }
            ++counts[block.col_node];
        }
        std::size_t cursor = 0U;
        for (std::size_t col = 0U; col < op_.nodes; ++col) {
            if (cursor > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("recursive sparse local index exceeds uint32");
            }
            by_column_[col] = static_cast<std::uint32_t>(cursor);
            cursor += counts[col];
        }
        if (cursor > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("recursive sparse local index exceeds uint32");
        }
        by_column_[op_.nodes] = static_cast<std::uint32_t>(cursor);
        block_indices_.resize(cursor);
        std::vector<std::size_t> write(op_.nodes, 0U);
        for (std::size_t col = 0U; col < op_.nodes; ++col) write[col] = by_column_[col];
        for (std::size_t p = 0U; p < op_.blocks.size(); ++p) {
            const std::size_t col = op_.blocks[p].col_node;
            block_indices_[write[col]++] = static_cast<std::uint32_t>(p);
        }
    }

    LocalColumns operator()(const LocalColumns& input) const {
        if (input.cols == 0U || input.cols > kCandidates) {
            throw std::invalid_argument("recursive sparse local column count invalid");
        }
        LocalColumns output;
        output.cols = input.cols;
        output.values.reserve(input.values.size() * 4U + 8U);
        for (const auto& in_entry : input.values) {
            const std::size_t col_node = static_cast<std::size_t>(in_entry.first);
            if (col_node >= op_.nodes) {
                throw std::out_of_range("recursive sparse local input node out of range");
            }
            const auto& cagg = layout_.aggregates[col_node];
            for (std::size_t q = by_column_[col_node]; q < by_column_[col_node + 1U]; ++q) {
                const auto& block = op_.blocks[block_indices_[q]];
                const std::size_t row_node = static_cast<std::size_t>(block.row_node);
                const auto& ragg = layout_.aggregates[row_node];
                auto& dst = zero_block(output.values, static_cast<std::uint32_t>(row_node));
                for (std::size_t r = 0U; r < ragg.rank; ++r) {
                    for (std::size_t c = 0U; c < cagg.rank; ++c) {
                        const double a = block.values[r * kCandidates + c];
                        if (a == 0.0) continue;
                        for (std::size_t j = 0U; j < input.cols; ++j) {
                            dst[r * kCandidates + j] +=
                                a * in_entry.second[c * kCandidates + j];
                        }
                    }
                }
            }
        }
        return output;
    }

private:
    const m5_sparse_a2::ExactSparseA2& op_;
    const CandidateTransfer& layout_;
    std::vector<std::uint32_t> by_column_;
    std::vector<std::uint32_t> block_indices_;
};

inline m5_l2_setup::DenseA2 materialize_dense(
    const m5_sparse_a2::ExactSparseA2& op,
    const CandidateTransfer& layout) {
    if (layout.aggregates.size() != op.nodes) {
        throw std::invalid_argument("recursive sparse dense materialization layout mismatch");
    }
    const auto start = Clock::now();
    m5_l2_setup::DenseA2 dense;
    dense.n = op.n;
    dense.fp64.assign(op.n * op.n, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t p64 = 0; p64 < static_cast<std::int64_t>(op.blocks.size()); ++p64) {
        const auto& block = op.blocks[static_cast<std::size_t>(p64)];
        const auto& ragg = layout.aggregates[block.row_node];
        const auto& cagg = layout.aggregates[block.col_node];
        for (std::size_t r = 0U; r < ragg.rank; ++r) {
            for (std::size_t c = 0U; c < cagg.rank; ++c) {
                dense.fp64[(ragg.coarse_offset + r) * dense.n +
                           (cagg.coarse_offset + c)] =
                    block.values[r * kCandidates + c];
            }
        }
    }
    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0U; i < dense.n; ++i) {
        const double di = dense.fp64[i * dense.n + i];
        if (!(di > 0.0) || !std::isfinite(di)) {
            throw std::runtime_error("recursive sparse dense bottom diagonal invalid");
        }
        norm2 += di * di;
        for (std::size_t j = i + 1U; j < dense.n; ++j) {
            const double aij = dense.fp64[i * dense.n + j];
            const double aji = dense.fp64[j * dense.n + i];
            const double d = aij - aji;
            asym2 += 2.0 * d * d;
            norm2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            dense.fp64[i * dense.n + j] = sym;
            dense.fp64[j * dense.n + i] = sym;
        }
    }
    dense.symmetry_relative_defect = norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;
    dense.assembly_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return dense;
}

inline LocalBottomReference factor_dense_bottom(const m5_l2_setup::DenseA2& dense) {
    const auto start = Clock::now();
    LocalBottomReference bottom;
    bottom.values = dense.fp64;
    bottom.factor.n = dense.n;
    bottom.factor.symmetry_relative_defect = dense.symmetry_relative_defect;
    bottom.factor.lower.assign(dense.n * dense.n, 0.0);
    bottom.factor.min_pivot = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0U; i < dense.n; ++i) {
        for (std::size_t j = 0U; j <= i; ++j) {
            double value = dense.fp64[i * dense.n + j];
            for (std::size_t k = 0U; k < j; ++k) {
                value -= bottom.factor.lower[i * dense.n + k] *
                         bottom.factor.lower[j * dense.n + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("recursive sparse dense bottom lost SPD");
                }
                bottom.factor.min_pivot = std::min(bottom.factor.min_pivot, value);
                bottom.factor.lower[i * dense.n + j] = std::sqrt(value);
            } else {
                bottom.factor.lower[i * dense.n + j] =
                    value / bottom.factor.lower[j * dense.n + j];
            }
        }
    }
    bottom.assembly_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return bottom;
}

inline double relative_error(const std::vector<double>& a,
                             const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("recursive sparse oracle size mismatch");
    }
    double d2 = 0.0;
    double b2 = 0.0;
    for (std::size_t i = 0U; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        d2 += d * d;
        b2 += b[i] * b[i];
    }
    return std::sqrt(d2 / std::max(b2, 1.0e-300));
}

inline RecursiveSparseTail build(
    CandidateTransfer initial_layout,
    L1BlockMetric initial_metric,
    m5_sparse_a2::ExactSparseA2 initial_operator,
    std::size_t initial_level,
    std::size_t target_nodes,
    std::size_t min_nodes,
    std::size_t dense_bottom_threshold,
    std::size_t max_levels = 12U) {
    if (initial_level < 1U || target_nodes < 2U || min_nodes == 0U ||
        min_nodes > target_nodes || dense_bottom_threshold == 0U || max_levels == 0U) {
        throw std::invalid_argument("recursive sparse tail options invalid");
    }
    if (initial_layout.coarse_dofs != initial_operator.n ||
        initial_metric.dofs() != initial_operator.n ||
        initial_layout.aggregates.size() != initial_operator.nodes) {
        throw std::invalid_argument("recursive sparse tail initial layout mismatch");
    }

    RecursiveSparseTail out;
    out.dense_bottom_threshold = dense_bottom_threshold;
    const auto total_start = Clock::now();

    CandidateTransfer layout = std::move(initial_layout);
    L1BlockMetric metric = std::move(initial_metric);
    m5_sparse_a2::ExactSparseA2 op = std::move(initial_operator);
    std::size_t level_index = initial_level;

    for (std::size_t depth = 0U; depth < max_levels; ++depth) {
        LevelTelemetry tel;
        tel.level = level_index;
        tel.nodes = op.nodes;
        tel.dofs = op.n;
        tel.scalar_nnz = op.values_fp32.size();
        tel.directed_blocks = op.blocks.size();
        tel.scalar_density = op.n > 0U
            ? static_cast<double>(op.values_fp32.size()) /
              (static_cast<double>(op.n) * static_cast<double>(op.n))
            : 0.0;
        tel.dense_fp32_bytes = op.n * op.n * sizeof(float);
        tel.sparse_fp32_bytes = op.fp32_csr_logical_bytes();
        tel.memory_preferred_runtime_representation =
            tel.sparse_fp32_bytes < tel.dense_fp32_bytes
                ? "structural_scalar_CSR_fp32" : "dense_fp32";

        const Apply apply_current = [&](const Vec& x) {
            return m5_sparse_a2::apply_fp64(op, layout, x);
        };
        auto stage = Clock::now();
        const double lambda = estimate_lambda_max_l1_block(apply_current, metric, 8U);
        tel.lambda_ms = std::chrono::duration<double, std::milli>(Clock::now() - stage).count();
        tel.lambda = lambda;
        tel.omega = kSaDampingNumerator / lambda;

        SparseCoarseLevel current;
        current.level = level_index;
        current.layout = layout;
        current.metric = metric;
        current.op = std::move(op);
        current.lambda = tel.lambda;
        current.omega = tel.omega;

        if (current.op.n <= dense_bottom_threshold) {
            const auto bottom_start = Clock::now();
            const auto dense = materialize_dense(current.op, current.layout);
            out.bottom = factor_dense_bottom(dense);
            out.bottom_materialize_factor_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - bottom_start).count();
            out.bottom_level = level_index;
            out.bottom_dofs = current.op.n;

            Vec probe(current.op.n, 0.0);
            for (std::size_t i = 0U; i < probe.size(); ++i) {
                const double t = static_cast<double>(i + 1U);
                probe[i] = std::sin(0.021 * t + 0.37) + 0.19 * std::cos(0.043 * t);
            }
            const auto ys = m5_sparse_a2::apply_fp64(current.op, current.layout, probe);
            const auto yd = m5_l2_setup::apply_dense_a2(dense, probe);
            out.bottom_sparse_apply_relative_error = relative_error(ys, yd);
            const auto x = out.bottom.factor.solve(probe);
            const auto ax = m5_l2_setup::apply_dense_a2(dense, x);
            out.bottom_solve_relative_residual = relative_error(ax, probe);

            tel.next_nodes = 0U;
            tel.next_dofs = 0U;
            out.telemetry.push_back(std::move(tel));
            out.levels.push_back(std::move(current));
            out.total_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - total_start).count();
            return out;
        }

        stage = Clock::now();
        auto next_transfer = build_candidate_transfer(
            current.layout.coarse_graph,
            current.layout.coarse_candidates,
            target_nodes,
            min_nodes,
            1.0e-10);
        tel.transfer_ms = std::chrono::duration<double, std::milli>(Clock::now() - stage).count();
        if (next_transfer.coarse_dofs == 0U || next_transfer.coarse_dofs >= current.op.n) {
            throw std::runtime_error("recursive sparse tail failed to reduce coarse dimension");
        }
        tel.next_nodes = next_transfer.aggregates.size();
        tel.next_dofs = next_transfer.coarse_dofs;

        SparseLocalApply local_apply(current.op, current.layout);
        stage = Clock::now();
        auto supports = m5_fast_setup::build_smoothed_supports_parallel(
            next_transfer,
            current.layout.coarse_graph,
            current.metric,
            current.omega,
            1U,
            local_apply);
        tel.smoothed_support_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - stage).count();
        tel.transfer_support_blocks = support_block_count(supports);

        stage = Clock::now();
        auto applied = m5_fast_setup::apply_supports_parallel(supports, local_apply);
        tel.applied_support_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - stage).count();

        stage = Clock::now();
        auto next_metric = m5_fast_setup::metric_from_cached_applied(
            next_transfer, current.metric, supports, applied);
        tel.metric_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - stage).count();

        stage = Clock::now();
        auto next_op = m5_sparse_a2::assemble_from_cached_applied(
            next_transfer, current.metric, supports, applied);
        tel.galerkin_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - stage).count();

        current.transfer_to_next = next_transfer;
        current.prolong_supports = std::move(supports);
        out.telemetry.push_back(std::move(tel));
        out.levels.push_back(std::move(current));

        layout = std::move(next_transfer);
        metric = std::move(next_metric);
        op = std::move(next_op);
        ++level_index;
    }

    throw std::runtime_error("recursive sparse tail exceeded max_levels before dense bottom");
}

}  // namespace m5_recursive_tail
