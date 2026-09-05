#pragma once

#include "gfss/gpu_m5_recursive_tail.hpp"
#include "m5_recursive_sparse_tail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace m5_recursive_tail_gpu_bridge {

struct TransferCsr {
    std::vector<std::uint32_t> p_rows;
    std::vector<std::uint32_t> p_cols;
    std::vector<float> p_values;
    std::vector<std::uint32_t> pt_rows;
    std::vector<std::uint32_t> pt_cols;
    std::vector<float> pt_values;
};

inline std::vector<float> inverse_metric_padded(const L1BlockMetric& metric) {
    std::vector<float> out(metric.nodes() * 36U, 0.0f);
    for (std::size_t node = 0U; node < metric.nodes(); ++node) {
        const std::size_t begin = metric.dof_offsets[node];
        const std::size_t rank = metric.dof_offsets[node + 1U] - begin;
        if (rank == 0U || rank > kCandidates) {
            throw std::runtime_error("recursive tail GPU bridge invalid metric rank");
        }
        const double* l = metric.lower.data() + metric.value_offsets[node];
        for (std::size_t col = 0U; col < rank; ++col) {
            std::array<double, kCandidates> y{};
            std::array<double, kCandidates> x{};
            for (std::size_t i = 0U; i < rank; ++i) {
                double value = i == col ? 1.0 : 0.0;
                for (std::size_t j = 0U; j < i; ++j) value -= l[i * rank + j] * y[j];
                y[i] = value / l[i * rank + i];
            }
            for (std::size_t ii = rank; ii-- > 0U;) {
                double value = y[ii];
                for (std::size_t j = ii + 1U; j < rank; ++j) {
                    value -= l[j * rank + ii] * x[j];
                }
                x[ii] = value / l[ii * rank + ii];
            }
            for (std::size_t row = 0U; row < rank; ++row) {
                out[node * 36U + row * 6U + col] = static_cast<float>(x[row]);
            }
        }
    }
    return out;
}

inline TransferCsr export_transfer(const m5_recursive_tail::SparseCoarseLevel& level) {
    const auto& supports = level.prolong_supports;
    const auto& next = level.transfer_to_next;
    if (supports.size() != next.aggregates.size() || next.coarse_dofs == 0U) {
        throw std::invalid_argument("recursive tail GPU bridge transfer size mismatch");
    }
    const std::size_t rows = level.op.n;
    const std::size_t cols = next.coarse_dofs;
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        cols > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("recursive tail GPU bridge transfer exceeds uint32");
    }

    std::vector<std::size_t> p_count(rows, 0U);
    std::vector<std::size_t> pt_count(cols, 0U);
    std::size_t nnz = 0U;
    for (std::size_t a = 0U; a < supports.size(); ++a) {
        const auto& support = supports[a];
        const auto& cagg = next.aggregates[a];
        if (support.cols != cagg.rank) {
            throw std::runtime_error("recursive tail GPU bridge transfer rank mismatch");
        }
        for (const auto& entry : support.values) {
            const std::size_t node = static_cast<std::size_t>(entry.first);
            if (node >= level.layout.aggregates.size()) {
                throw std::out_of_range("recursive tail GPU bridge transfer node out of range");
            }
            const auto& ragg = level.layout.aggregates[node];
            for (std::size_t r = 0U; r < ragg.rank; ++r) {
                const std::size_t row = ragg.coarse_offset + r;
                for (std::size_t q = 0U; q < cagg.rank; ++q) {
                    const double value = entry.second[r * kCandidates + q];
                    if (value == 0.0) continue;
                    ++p_count[row];
                    ++pt_count[cagg.coarse_offset + q];
                    ++nnz;
                }
            }
        }
    }
    if (nnz > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("recursive tail GPU bridge transfer nnz exceeds uint32");
    }

    TransferCsr out;
    out.p_rows.resize(rows + 1U, 0U);
    out.pt_rows.resize(cols + 1U, 0U);
    std::size_t cursor = 0U;
    for (std::size_t row = 0U; row < rows; ++row) {
        out.p_rows[row] = static_cast<std::uint32_t>(cursor);
        cursor += p_count[row];
    }
    out.p_rows[rows] = static_cast<std::uint32_t>(cursor);
    cursor = 0U;
    for (std::size_t row = 0U; row < cols; ++row) {
        out.pt_rows[row] = static_cast<std::uint32_t>(cursor);
        cursor += pt_count[row];
    }
    out.pt_rows[cols] = static_cast<std::uint32_t>(cursor);

    out.p_cols.resize(nnz);
    out.p_values.resize(nnz);
    out.pt_cols.resize(nnz);
    out.pt_values.resize(nnz);
    std::vector<std::size_t> p_write(rows, 0U);
    std::vector<std::size_t> pt_write(cols, 0U);
    for (std::size_t row = 0U; row < rows; ++row) p_write[row] = out.p_rows[row];
    for (std::size_t row = 0U; row < cols; ++row) pt_write[row] = out.pt_rows[row];

    for (std::size_t a = 0U; a < supports.size(); ++a) {
        const auto& support = supports[a];
        const auto& cagg = next.aggregates[a];
        for (const auto& entry : support.values) {
            const std::size_t node = static_cast<std::size_t>(entry.first);
            const auto& ragg = level.layout.aggregates[node];
            for (std::size_t r = 0U; r < ragg.rank; ++r) {
                const std::size_t row = ragg.coarse_offset + r;
                for (std::size_t q = 0U; q < cagg.rank; ++q) {
                    const double value = entry.second[r * kCandidates + q];
                    if (value == 0.0) continue;
                    const std::size_t col = cagg.coarse_offset + q;
                    const std::size_t pp = p_write[row]++;
                    out.p_cols[pp] = static_cast<std::uint32_t>(col);
                    out.p_values[pp] = static_cast<float>(value);
                    const std::size_t tp = pt_write[col]++;
                    out.pt_cols[tp] = static_cast<std::uint32_t>(row);
                    out.pt_values[tp] = static_cast<float>(value);
                }
            }
        }
    }
    return out;
}

inline Vec prolong(const m5_recursive_tail::SparseCoarseLevel& level, const Vec& coarse) {
    if (coarse.size() != level.transfer_to_next.coarse_dofs) {
        throw std::invalid_argument("recursive tail CPU prolong size mismatch");
    }
    Vec fine(level.op.n, 0.0);
    for (std::size_t a = 0U; a < level.prolong_supports.size(); ++a) {
        const auto& support = level.prolong_supports[a];
        const auto& cagg = level.transfer_to_next.aggregates[a];
        for (const auto& entry : support.values) {
            const auto& ragg = level.layout.aggregates[entry.first];
            for (std::size_t r = 0U; r < ragg.rank; ++r) {
                double value = 0.0;
                for (std::size_t q = 0U; q < cagg.rank; ++q) {
                    value += entry.second[r * kCandidates + q] *
                             coarse[cagg.coarse_offset + q];
                }
                fine[ragg.coarse_offset + r] += value;
            }
        }
    }
    return fine;
}

inline Vec restrict_transpose(const m5_recursive_tail::SparseCoarseLevel& level,
                              const Vec& fine) {
    if (fine.size() != level.op.n) {
        throw std::invalid_argument("recursive tail CPU restrict size mismatch");
    }
    Vec coarse(level.transfer_to_next.coarse_dofs, 0.0);
    for (std::size_t a = 0U; a < level.prolong_supports.size(); ++a) {
        const auto& support = level.prolong_supports[a];
        const auto& cagg = level.transfer_to_next.aggregates[a];
        for (const auto& entry : support.values) {
            const auto& ragg = level.layout.aggregates[entry.first];
            for (std::size_t q = 0U; q < cagg.rank; ++q) {
                double value = 0.0;
                for (std::size_t r = 0U; r < ragg.rank; ++r) {
                    value += entry.second[r * kCandidates + q] *
                             fine[ragg.coarse_offset + r];
                }
                coarse[cagg.coarse_offset + q] += value;
            }
        }
    }
    return coarse;
}

inline Vec cpu_vcycle(const m5_recursive_tail::RecursiveSparseTail& tail,
                      std::size_t index,
                      const Vec& b) {
    if (index >= tail.levels.size() || b.size() != tail.levels[index].op.n) {
        throw std::invalid_argument("recursive tail CPU V-cycle level mismatch");
    }
    const auto& level = tail.levels[index];
    if (index + 1U == tail.levels.size()) {
        return tail.bottom.factor.solve(b);
    }
    const double weight = 1.0 / (0.55 * level.lambda);
    auto x = level.metric.solve(b);
    for (double& value : x) value *= weight;

    auto ax = m5_sparse_a2::apply_fp64(level.op, level.layout, x);
    Vec residual(b.size(), 0.0);
    for (std::size_t i = 0U; i < b.size(); ++i) residual[i] = b[i] - ax[i];
    const auto coarse_b = restrict_transpose(level, residual);
    const auto coarse_x = cpu_vcycle(tail, index + 1U, coarse_b);
    const auto correction = prolong(level, coarse_x);
    for (std::size_t i = 0U; i < x.size(); ++i) x[i] += correction[i];

    ax = m5_sparse_a2::apply_fp64(level.op, level.layout, x);
    for (std::size_t i = 0U; i < b.size(); ++i) residual[i] = b[i] - ax[i];
    auto post = level.metric.solve(residual);
    for (std::size_t i = 0U; i < x.size(); ++i) x[i] += weight * post[i];
    return x;
}

inline std::vector<gfss::M5RecursiveTailLevelPayload> export_payloads(
    const m5_recursive_tail::RecursiveSparseTail& tail) {
    std::vector<gfss::M5RecursiveTailLevelPayload> out;
    out.reserve(tail.levels.size());
    for (std::size_t i = 0U; i < tail.levels.size(); ++i) {
        const auto& level = tail.levels[i];
        gfss::M5RecursiveTailLevelPayload payload;
        payload.level = level.level;
        payload.dofs = level.op.n;
        payload.nodes = level.op.nodes;
        payload.lambda = level.lambda;
        payload.block_dof_offsets.resize(level.metric.dof_offsets.size());
        for (std::size_t j = 0U; j < level.metric.dof_offsets.size(); ++j) {
            if (level.metric.dof_offsets[j] > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("recursive tail GPU bridge block offset exceeds uint32");
            }
            payload.block_dof_offsets[j] =
                static_cast<std::uint32_t>(level.metric.dof_offsets[j]);
        }

        if (i + 1U < tail.levels.size()) {
            payload.inverse_blocks_padded_6x6 = inverse_metric_padded(level.metric);
            const auto& tel = tail.telemetry[i];
            if (tel.sparse_fp32_bytes < tel.dense_fp32_bytes) {
                payload.operator_kind = gfss::M5RecursiveTailOperatorKind::structural_scalar_csr_fp32;
                payload.csr_row_offsets = level.op.row_offsets;
                payload.csr_column_indices = level.op.column_indices;
                payload.csr_values = level.op.values_fp32;
            } else {
                payload.operator_kind = gfss::M5RecursiveTailOperatorKind::dense_fp32;
                const auto dense = m5_recursive_tail::materialize_dense(level.op, level.layout);
                payload.dense_row_major.resize(dense.fp64.size());
                std::transform(dense.fp64.begin(), dense.fp64.end(),
                               payload.dense_row_major.begin(),
                               [](double value) { return static_cast<float>(value); });
            }
            const auto transfer = export_transfer(level);
            payload.next_dofs = tail.levels[i + 1U].op.n;
            payload.p_row_offsets = transfer.p_rows;
            payload.p_column_indices = transfer.p_cols;
            payload.p_values = transfer.p_values;
            payload.pt_row_offsets = transfer.pt_rows;
            payload.pt_column_indices = transfer.pt_cols;
            payload.pt_values = transfer.pt_values;
        }
        out.push_back(std::move(payload));
    }
    return out;
}

inline std::vector<float> bottom_inverse_fp32(
    const m5_recursive_tail::RecursiveSparseTail& tail) {
    const std::size_t n = tail.bottom.factor.n;
    std::vector<float> inverse(n * n, 0.0f); // column-major
    for (std::size_t col = 0U; col < n; ++col) {
        Vec e(n, 0.0);
        e[col] = 1.0;
        const auto x = tail.bottom.factor.solve(e);
        for (std::size_t row = 0U; row < n; ++row) {
            inverse[col * n + row] = static_cast<float>(x[row]);
        }
    }
    return inverse;
}

inline double gpu_vs_cpu_relative_error(const std::vector<float>& gpu, const Vec& cpu) {
    if (gpu.size() != cpu.size()) {
        throw std::invalid_argument("recursive tail GPU/CPU oracle size mismatch");
    }
    double d2 = 0.0;
    double c2 = 0.0;
    for (std::size_t i = 0U; i < gpu.size(); ++i) {
        const double d = static_cast<double>(gpu[i]) - cpu[i];
        d2 += d * d;
        c2 += cpu[i] * cpu[i];
    }
    return std::sqrt(d2 / std::max(c2, 1.0e-300));
}

}  // namespace m5_recursive_tail_gpu_bridge
