#pragma once

// Exact temporary A1 materialization for hierarchy setup only.
// Include after recursive_sa_local_l2_helpers.inc and
// recursive_sa_actual_a1_strength_local_helpers.inc so CombinedBlock,
// LocalColumns and L1BlockMetric are available.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace m5_materialized_a1 {

using Clock = std::chrono::steady_clock;

struct ColumnEntry {
    std::uint32_t row{0U};
    CombinedBlock block{};  // A(row,col), stride 6
};

struct Operator {
    std::vector<std::vector<ColumnEntry>> columns;
    std::size_t directed_blocks{0U};
    std::size_t logical_bytes{0U};
    double setup_ms{0.0};
};

inline CombinedBlock transpose_block6(const CombinedBlock& a) {
    CombinedBlock t{};
    for (std::size_t i = 0U; i < kCandidates; ++i) {
        for (std::size_t j = 0U; j < kCandidates; ++j) {
            t[i * kCandidates + j] = a[j * kCandidates + i];
        }
    }
    return t;
}

inline Operator build(
    const L1BlockMetric& block1,
    const std::unordered_map<std::uint64_t, CombinedBlock>& offdiagonal) {
    const auto start = Clock::now();
    Operator out;
    out.columns.resize(block1.nodes());

    for (std::size_t node = 0U; node < block1.nodes(); ++node) {
        const std::size_t rank = block1.dof_offsets[node + 1U] - block1.dof_offsets[node];
        CombinedBlock diag{};
        for (std::size_t i = 0U; i < rank; ++i) {
            for (std::size_t j = 0U; j < rank; ++j) {
                diag[i * kCandidates + j] = block1.block_entry(node, i, j);
            }
        }
        out.columns[node].push_back({static_cast<std::uint32_t>(node), diag});
    }

    for (const auto& item : offdiagonal) {
        const auto [a, b] = combined_decode_pair(item.first);
        out.columns[b].push_back({a, item.second});
        out.columns[a].push_back({b, transpose_block6(item.second)});
    }

    for (auto& column : out.columns) {
        std::sort(column.begin(), column.end(), [](const auto& x, const auto& y) {
            return x.row < y.row;
        });
        out.directed_blocks += column.size();
    }
    out.logical_bytes = out.directed_blocks *
        (sizeof(std::uint32_t) + sizeof(CombinedBlock));
    out.setup_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    return out;
}

inline LocalColumns apply_columns(
    const Operator& a1,
    const L1BlockMetric& layout,
    const LocalColumns& input) {
    if (input.cols == 0U || input.cols > kCandidates) {
        throw std::invalid_argument("materialized A1 local column count invalid");
    }
    LocalColumns out;
    out.cols = input.cols;
    for (const auto& xentry : input.values) {
        const std::size_t col_node = xentry.first;
        if (col_node >= a1.columns.size()) {
            throw std::out_of_range("materialized A1 input node out of range");
        }
        const std::size_t col_rank = layout.dof_offsets[col_node + 1U] -
                                     layout.dof_offsets[col_node];
        for (const auto& entry : a1.columns[col_node]) {
            const std::size_t row_node = entry.row;
            const std::size_t row_rank = layout.dof_offsets[row_node + 1U] -
                                         layout.dof_offsets[row_node];
            auto& y = zero_block(out.values, entry.row);
            for (std::size_t i = 0U; i < row_rank; ++i) {
                for (std::size_t q = 0U; q < col_rank; ++q) {
                    const double a = entry.block[i * kCandidates + q];
                    if (a == 0.0) continue;
                    for (std::size_t c = 0U; c < input.cols; ++c) {
                        y[i * kCandidates + c] +=
                            a * xentry.second[q * kCandidates + c];
                    }
                }
            }
        }
    }
    return out;
}

// Reusable setup-only accumulator for localized A1*X products. A dense 6x6
// block is provisioned per L1 node, but only rows touched by the current input
// are zeroed and exported. This removes repeated output unordered_map probes
// from the sparse block matmul while preserving the existing LocalColumns API.
struct ApplyScratch {
    std::vector<CombinedBlock> blocks;
    std::vector<std::uint32_t> marks;
    std::vector<std::uint32_t> touched;
    std::uint32_t generation{1U};

    explicit ApplyScratch(std::size_t nodes)
        : blocks(nodes), marks(nodes, 0U) {
        touched.reserve(std::min<std::size_t>(nodes, 1024U));
    }

    void begin_apply() {
        touched.clear();
        ++generation;
        if (generation == 0U) {
            std::fill(marks.begin(), marks.end(), 0U);
            generation = 1U;
        }
    }

    CombinedBlock& touch(std::size_t row) {
        if (row >= blocks.size()) {
            throw std::out_of_range("materialized A1 scratch row out of range");
        }
        if (marks[row] != generation) {
            marks[row] = generation;
            blocks[row].fill(0.0);
            touched.push_back(static_cast<std::uint32_t>(row));
        }
        return blocks[row];
    }

    std::size_t logical_bytes() const {
        return blocks.size() * sizeof(CombinedBlock) +
               marks.size() * sizeof(std::uint32_t) +
               touched.capacity() * sizeof(std::uint32_t);
    }
};

inline LocalColumns apply_columns_dense_scratch(
    const Operator& a1,
    const L1BlockMetric& layout,
    const LocalColumns& input,
    ApplyScratch& scratch) {
    if (input.cols == 0U || input.cols > kCandidates) {
        throw std::invalid_argument("materialized A1 dense-scratch column count invalid");
    }
    if (scratch.blocks.size() != a1.columns.size()) {
        throw std::invalid_argument("materialized A1 dense-scratch node count mismatch");
    }
    scratch.begin_apply();

    for (const auto& xentry : input.values) {
        const std::size_t col_node = xentry.first;
        if (col_node >= a1.columns.size()) {
            throw std::out_of_range("materialized A1 dense-scratch input node out of range");
        }
        const std::size_t col_rank = layout.dof_offsets[col_node + 1U] -
                                     layout.dof_offsets[col_node];
        for (const auto& entry : a1.columns[col_node]) {
            const std::size_t row_node = entry.row;
            const std::size_t row_rank = layout.dof_offsets[row_node + 1U] -
                                         layout.dof_offsets[row_node];
            auto& y = scratch.touch(row_node);
            for (std::size_t i = 0U; i < row_rank; ++i) {
                for (std::size_t q = 0U; q < col_rank; ++q) {
                    const double a = entry.block[i * kCandidates + q];
                    if (a == 0.0) continue;
                    for (std::size_t c = 0U; c < input.cols; ++c) {
                        y[i * kCandidates + c] +=
                            a * xentry.second[q * kCandidates + c];
                    }
                }
            }
        }
    }

    LocalColumns out;
    out.cols = input.cols;
    out.values.reserve(scratch.touched.size());
    for (const auto row : scratch.touched) {
        out.values.emplace(row, scratch.blocks[row]);
    }
    return out;
}

inline std::vector<LocalColumns> apply_supports_parallel_dense_scratch(
    const Operator& a1,
    const L1BlockMetric& layout,
    const std::vector<LocalColumns>& supports,
    std::size_t* scratch_bytes_per_worker = nullptr) {
    std::vector<LocalColumns> applied(supports.size());
    if (scratch_bytes_per_worker != nullptr) {
        ApplyScratch sample(a1.columns.size());
        *scratch_bytes_per_worker = sample.logical_bytes();
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        ApplyScratch scratch(a1.columns.size());
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::int64_t a64 = 0;
             a64 < static_cast<std::int64_t>(supports.size()); ++a64) {
            const std::size_t a = static_cast<std::size_t>(a64);
            applied[a] = apply_columns_dense_scratch(a1, layout, supports[a], scratch);
        }
    }
    return applied;
}

inline Vec apply_vector(
    const Operator& a1,
    const L1BlockMetric& layout,
    const Vec& x) {
    if (x.size() != layout.dofs()) {
        throw std::invalid_argument("materialized A1 vector size mismatch");
    }
    Vec y(x.size(), 0.0);
    for (std::size_t col_node = 0U; col_node < a1.columns.size(); ++col_node) {
        const std::size_t cb = layout.dof_offsets[col_node];
        const std::size_t cr = layout.dof_offsets[col_node + 1U] - cb;
        for (const auto& entry : a1.columns[col_node]) {
            const std::size_t rb = layout.dof_offsets[entry.row];
            const std::size_t rr = layout.dof_offsets[entry.row + 1U] - rb;
            for (std::size_t i = 0U; i < rr; ++i) {
                double sum = 0.0;
                for (std::size_t q = 0U; q < cr; ++q) {
                    sum += entry.block[i * kCandidates + q] * x[cb + q];
                }
                y[rb + i] += sum;
            }
        }
    }
    return y;
}

}  // namespace m5_materialized_a1
