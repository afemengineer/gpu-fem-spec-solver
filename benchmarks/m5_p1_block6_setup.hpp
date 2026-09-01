#pragma once

// Reusable staging setup for the frozen explicit L1<->L2 transfer. Include this
// only after recursive_sa_local_l2_helpers.inc so CandidateTransfer,
// L1BlockMetric and LocalColumns are visible in the benchmark translation unit.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace m5_p1_setup {

using Clock = std::chrono::steady_clock;
using DenseBlock6 = std::array<double, 36U>;
using Entry = std::pair<std::uint32_t, DenseBlock6>;

struct DualOrderBlock6Transfer {
    std::size_t block_rows{0U};
    std::size_t block_cols{0U};
    std::vector<std::uint32_t> forward_row_offsets;
    std::vector<std::uint32_t> forward_column_indices;
    std::vector<float> forward_values_row_major;
    std::vector<std::uint32_t> transpose_column_offsets;
    std::vector<std::uint32_t> transpose_row_indices;
    std::vector<float> transpose_values_q_r_entry;
    double assembly_ms{0.0};

    std::size_t block_nnz() const noexcept {
        return forward_column_indices.size();
    }
    std::size_t forward_index_bytes() const noexcept {
        return (forward_row_offsets.size() + forward_column_indices.size()) *
               sizeof(std::uint32_t);
    }
    std::size_t transpose_index_bytes() const noexcept {
        return (transpose_column_offsets.size() + transpose_row_indices.size()) *
               sizeof(std::uint32_t);
    }
    std::size_t one_value_payload_bytes() const noexcept {
        return block_nnz() * 36U * sizeof(float);
    }
    std::size_t matrix_bytes_fp32() const noexcept {
        return forward_index_bytes() + transpose_index_bytes() +
               2U * one_value_payload_bytes();
    }
};

inline DualOrderBlock6Transfer assemble_dual_order_block6(
    const CandidateTransfer& transfer,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis) {
    const auto start = Clock::now();
    DualOrderBlock6Transfer out;
    out.block_rows = block1.nodes();
    out.block_cols = l2_basis.size();
    if (out.block_rows == 0U || out.block_cols == 0U ||
        transfer.aggregates.size() != out.block_cols ||
        transfer.coarse_graph.nodes() != out.block_cols) {
        throw std::invalid_argument("M5 dual block6 transfer layout mismatch");
    }

    std::vector<std::vector<Entry>> rows(out.block_rows);
    std::vector<std::vector<Entry>> columns(out.block_cols);

    for (std::size_t col = 0; col < out.block_cols; ++col) {
        const auto& basis = l2_basis[col];
        const std::size_t rank2 = transfer.aggregates[col].rank;
        if (rank2 == 0U || rank2 > 6U || basis.cols != rank2) {
            throw std::runtime_error("M5 dual block6 L2 rank mismatch");
        }
        for (const auto& item : basis.values) {
            const std::size_t row = item.first;
            if (row >= out.block_rows) {
                throw std::out_of_range("M5 dual block6 L1 node out of range");
            }
            const std::size_t rank1 =
                block1.dof_offsets[row + 1U] - block1.dof_offsets[row];
            if (rank1 == 0U || rank1 > 6U) {
                throw std::runtime_error("M5 dual block6 L1 rank mismatch");
            }
            DenseBlock6 block{};
            bool nonzero = false;
            for (std::size_t r = 0; r < rank1; ++r) {
                for (std::size_t q = 0; q < rank2; ++q) {
                    const double value = item.second[r * 6U + q];
                    block[r * 6U + q] = value;
                    nonzero = nonzero || (value != 0.0);
                }
            }
            if (!nonzero) continue;
            rows[row].emplace_back(static_cast<std::uint32_t>(col), block);
            columns[col].emplace_back(static_cast<std::uint32_t>(row), block);
        }
    }

    out.forward_row_offsets.resize(out.block_rows + 1U, 0U);
    for (std::size_t row = 0; row < out.block_rows; ++row) {
        auto& entries = rows[row];
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return a.first < b.first;
        });
        out.forward_row_offsets[row] =
            static_cast<std::uint32_t>(out.forward_column_indices.size());
        std::uint32_t previous = std::numeric_limits<std::uint32_t>::max();
        for (const auto& e : entries) {
            if (e.first == previous) {
                throw std::runtime_error("M5 dual block6 duplicate forward block");
            }
            previous = e.first;
            out.forward_column_indices.push_back(e.first);
            for (const double value : e.second) {
                out.forward_values_row_major.push_back(static_cast<float>(value));
            }
        }
    }
    out.forward_row_offsets[out.block_rows] =
        static_cast<std::uint32_t>(out.forward_column_indices.size());

    out.transpose_column_offsets.resize(out.block_cols + 1U, 0U);
    for (std::size_t col = 0; col < out.block_cols; ++col) {
        auto& entries = columns[col];
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return a.first < b.first;
        });
        out.transpose_column_offsets[col] =
            static_cast<std::uint32_t>(out.transpose_row_indices.size());
        std::uint32_t previous = std::numeric_limits<std::uint32_t>::max();
        for (const auto& e : entries) {
            if (e.first == previous) {
                throw std::runtime_error("M5 dual block6 duplicate transpose block");
            }
            previous = e.first;
            out.transpose_row_indices.push_back(e.first);
        }

        const std::size_t count = entries.size();
        // Per-column layout consumed by the optimized CUDA transpose:
        // [output q][input r][incident entry].
        for (std::size_t q = 0; q < 6U; ++q) {
            for (std::size_t r = 0; r < 6U; ++r) {
                for (std::size_t k = 0; k < count; ++k) {
                    out.transpose_values_q_r_entry.push_back(
                        static_cast<float>(entries[k].second[r * 6U + q]));
                }
            }
        }
    }
    out.transpose_column_offsets[out.block_cols] =
        static_cast<std::uint32_t>(out.transpose_row_indices.size());

    if (out.transpose_row_indices.size() != out.block_nnz() ||
        out.forward_values_row_major.size() != out.block_nnz() * 36U ||
        out.transpose_values_q_r_entry.size() != out.block_nnz() * 36U) {
        throw std::runtime_error("M5 dual block6 final payload mismatch");
    }
    out.assembly_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    return out;
}

inline std::vector<double> pad_l1(const L1BlockMetric& block1,
                                  const std::vector<double>& packed) {
    if (packed.size() != block1.dofs()) {
        throw std::invalid_argument("M5 dual block6 L1 pad size mismatch");
    }
    std::vector<double> padded(block1.nodes() * 6U, 0.0);
    for (std::size_t node = 0; node < block1.nodes(); ++node) {
        const std::size_t begin = block1.dof_offsets[node];
        const std::size_t rank = block1.dof_offsets[node + 1U] - begin;
        for (std::size_t r = 0; r < rank; ++r) {
            padded[node * 6U + r] = packed[begin + r];
        }
    }
    return padded;
}

inline std::vector<double> unpad_l1(const L1BlockMetric& block1,
                                    const std::vector<double>& padded) {
    if (padded.size() != block1.nodes() * 6U) {
        throw std::invalid_argument("M5 dual block6 L1 unpad size mismatch");
    }
    std::vector<double> packed(block1.dofs(), 0.0);
    for (std::size_t node = 0; node < block1.nodes(); ++node) {
        const std::size_t begin = block1.dof_offsets[node];
        const std::size_t rank = block1.dof_offsets[node + 1U] - begin;
        for (std::size_t r = 0; r < rank; ++r) {
            packed[begin + r] = padded[node * 6U + r];
        }
    }
    return packed;
}

inline std::vector<double> pad_l2(const CandidateTransfer& transfer,
                                  const std::vector<double>& packed) {
    if (packed.size() != transfer.coarse_dofs) {
        throw std::invalid_argument("M5 dual block6 L2 pad size mismatch");
    }
    std::vector<double> padded(transfer.aggregates.size() * 6U, 0.0);
    for (std::size_t node = 0; node < transfer.aggregates.size(); ++node) {
        const std::size_t begin = transfer.aggregates[node].coarse_offset;
        const std::size_t rank = transfer.aggregates[node].rank;
        for (std::size_t q = 0; q < rank; ++q) {
            padded[node * 6U + q] = packed[begin + q];
        }
    }
    return padded;
}

inline std::vector<double> unpad_l2(const CandidateTransfer& transfer,
                                    const std::vector<double>& padded) {
    if (padded.size() != transfer.aggregates.size() * 6U) {
        throw std::invalid_argument("M5 dual block6 L2 unpad size mismatch");
    }
    std::vector<double> packed(transfer.coarse_dofs, 0.0);
    for (std::size_t node = 0; node < transfer.aggregates.size(); ++node) {
        const std::size_t begin = transfer.aggregates[node].coarse_offset;
        const std::size_t rank = transfer.aggregates[node].rank;
        for (std::size_t q = 0; q < rank; ++q) {
            packed[begin + q] = padded[node * 6U + q];
        }
    }
    return packed;
}

inline std::vector<double> forward_cpu(const DualOrderBlock6Transfer& p,
                                       const std::vector<double>& x_padded) {
    if (x_padded.size() != p.block_cols * 6U) {
        throw std::invalid_argument("M5 dual block6 forward x size mismatch");
    }
    std::vector<double> y(p.block_rows * 6U, 0.0);
    for (std::size_t row = 0; row < p.block_rows; ++row) {
        for (std::size_t bid = p.forward_row_offsets[row];
             bid < p.forward_row_offsets[row + 1U]; ++bid) {
            const std::size_t col = p.forward_column_indices[bid];
            const float* block = p.forward_values_row_major.data() + bid * 36U;
            for (std::size_t r = 0; r < 6U; ++r) {
                double sum = y[row * 6U + r];
                for (std::size_t q = 0; q < 6U; ++q) {
                    sum += static_cast<double>(block[r * 6U + q]) *
                           x_padded[col * 6U + q];
                }
                y[row * 6U + r] = sum;
            }
        }
    }
    return y;
}

inline std::vector<double> transpose_cpu(const DualOrderBlock6Transfer& p,
                                         const std::vector<double>& x_padded) {
    if (x_padded.size() != p.block_rows * 6U) {
        throw std::invalid_argument("M5 dual block6 transpose x size mismatch");
    }
    std::vector<double> y(p.block_cols * 6U, 0.0);
    for (std::size_t col = 0; col < p.block_cols; ++col) {
        const std::size_t first = p.transpose_column_offsets[col];
        const std::size_t last = p.transpose_column_offsets[col + 1U];
        const std::size_t count = last - first;
        const std::size_t base = first * 36U;
        for (std::size_t q = 0; q < 6U; ++q) {
            double sum = 0.0;
            for (std::size_t local = 0; local < count; ++local) {
                const std::size_t row = p.transpose_row_indices[first + local];
                for (std::size_t r = 0; r < 6U; ++r) {
                    const std::size_t coeff = base + (q * 6U + r) * count + local;
                    sum += static_cast<double>(p.transpose_values_q_r_entry[coeff]) *
                           x_padded[row * 6U + r];
                }
            }
            y[col * 6U + q] = sum;
        }
    }
    return y;
}

inline std::vector<float> to_float(const std::vector<double>& values) {
    std::vector<float> out(values.size(), 0.0f);
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = static_cast<float>(values[i]);
    }
    return out;
}

}  // namespace m5_p1_setup
