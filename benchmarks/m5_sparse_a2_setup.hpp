#pragma once

// Exact setup-only sparse A2 construction. Candidate aggregate pairs are
// discovered from localized support intersection, evaluated once in the upper
// block triangle, mirrored by Galerkin symmetry, and stored as compact directed
// variable-rank block entries. No dense n^2 A2 payload is allocated.

#include "m5_l2_dense_setup.hpp"
#include "m5_fast_hierarchy_setup.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace m5_sparse_a2 {

using Clock = std::chrono::steady_clock;

struct BlockEntry {
    std::uint32_t row_node{0U};
    std::uint32_t col_node{0U};
    std::array<double, 36> values{};
};

struct ExactSparseA2 {
    std::size_t n{0U};
    std::size_t nodes{0U};
    std::vector<BlockEntry> blocks;
    std::vector<std::uint32_t> block_row_offsets;
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
    std::vector<float> values_fp32;
    std::size_t all_block_pairs{0U};
    std::size_t candidate_upper_block_pairs{0U};
    double support_index_ms{0.0};
    double block_assembly_ms{0.0};
    double csr_export_ms{0.0};
    double total_ms{0.0};

    std::size_t fp64_block_logical_bytes() const noexcept {
        return blocks.size() * (2U * sizeof(std::uint32_t) + 36U * sizeof(double)) +
               block_row_offsets.size() * sizeof(std::uint32_t);
    }

    std::size_t fp32_csr_logical_bytes() const noexcept {
        return row_offsets.size() * sizeof(std::uint32_t) +
               column_indices.size() * sizeof(std::uint32_t) +
               values_fp32.size() * sizeof(float);
    }
};

inline std::array<double, 36> transpose_block6(const std::array<double, 36>& in) {
    std::array<double, 36> out{};
    for (std::size_t i = 0U; i < kCandidates; ++i) {
        for (std::size_t j = 0U; j < kCandidates; ++j) {
            out[i * kCandidates + j] = in[j * kCandidates + i];
        }
    }
    return out;
}

inline ExactSparseA2 assemble_from_cached_applied(
    const CandidateTransfer& transfer1,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis,
    const std::vector<LocalColumns>& applied_l2_basis) {
    if (l2_basis.size() != transfer1.aggregates.size() ||
        applied_l2_basis.size() != l2_basis.size()) {
        throw std::invalid_argument("sparse A2 cached support size mismatch");
    }
    if (transfer1.coarse_dofs > std::numeric_limits<std::uint32_t>::max() ||
        l2_basis.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("sparse A2 exceeds uint32 indexing");
    }

    ExactSparseA2 out;
    out.n = transfer1.coarse_dofs;
    out.nodes = l2_basis.size();
    out.all_block_pairs = out.nodes * (out.nodes + 1U) / 2U;
    const auto total_start = Clock::now();

    const auto index_start = Clock::now();
    std::vector<std::vector<std::uint32_t>> basis_by_l1_row(block1.nodes());
    for (std::size_t inode = 0U; inode < l2_basis.size(); ++inode) {
        for (const auto& entry : l2_basis[inode].values) {
            const std::size_t row = static_cast<std::size_t>(entry.first);
            if (row >= basis_by_l1_row.size()) {
                throw std::out_of_range("sparse A2 L1 row out of range");
            }
            basis_by_l1_row[row].push_back(static_cast<std::uint32_t>(inode));
        }
    }
    out.support_index_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - index_start).count();

    const auto assembly_start = Clock::now();
    std::vector<std::vector<BlockEntry>> per_column(out.nodes);
    std::uint64_t candidate_count = 0U;
#ifdef _OPENMP
#pragma omp parallel reduction(+:candidate_count)
#endif
    {
        std::vector<std::uint32_t> marks(out.nodes, 0U);
        std::vector<std::uint32_t> candidates;
        candidates.reserve(std::min<std::size_t>(out.nodes, 128U));
        std::uint32_t generation = 0U;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::int64_t j64 = 0;
             j64 < static_cast<std::int64_t>(out.nodes); ++j64) {
            const std::size_t jnode = static_cast<std::size_t>(j64);
            ++generation;
            if (generation == 0U) {
                std::fill(marks.begin(), marks.end(), 0U);
                generation = 1U;
            }
            candidates.clear();
            const auto& applied = applied_l2_basis[jnode];
            for (const auto& entry : applied.values) {
                const std::size_t row = static_cast<std::size_t>(entry.first);
                if (row >= basis_by_l1_row.size()) {
                    throw std::out_of_range("sparse A2 applied row out of range");
                }
                for (const auto inode_u32 : basis_by_l1_row[row]) {
                    const std::size_t inode = static_cast<std::size_t>(inode_u32);
                    if (inode > jnode || marks[inode] == generation) continue;
                    marks[inode] = generation;
                    candidates.push_back(inode_u32);
                }
            }
            if (marks[jnode] != generation) {
                marks[jnode] = generation;
                candidates.push_back(static_cast<std::uint32_t>(jnode));
            }
            std::sort(candidates.begin(), candidates.end());
            candidate_count += static_cast<std::uint64_t>(candidates.size());

            auto& column_entries = per_column[jnode];
            column_entries.reserve(candidates.size() * 2U);
            const auto& jagg = transfer1.aggregates[jnode];
            const std::size_t jrank = jagg.rank;
            for (const auto inode_u32 : candidates) {
                const std::size_t inode = static_cast<std::size_t>(inode_u32);
                const auto& iagg = transfer1.aggregates[inode];
                const std::size_t irank = iagg.rank;
                auto block = local_cross_gram(l2_basis[inode], applied, block1);
                if (inode == jnode) {
                    if (irank != jrank) {
                        throw std::runtime_error("sparse A2 diagonal block rank mismatch");
                    }
                    for (std::size_t i = 0U; i < irank; ++i) {
                        for (std::size_t j = i + 1U; j < jrank; ++j) {
                            const double sym = 0.5 * (
                                block[i * kCandidates + j] +
                                block[j * kCandidates + i]);
                            block[i * kCandidates + j] = sym;
                            block[j * kCandidates + i] = sym;
                        }
                    }
                    BlockEntry diag;
                    diag.row_node = static_cast<std::uint32_t>(inode);
                    diag.col_node = static_cast<std::uint32_t>(jnode);
                    diag.values = block;
                    column_entries.push_back(std::move(diag));
                } else {
                    BlockEntry upper;
                    upper.row_node = static_cast<std::uint32_t>(inode);
                    upper.col_node = static_cast<std::uint32_t>(jnode);
                    upper.values = block;
                    column_entries.push_back(std::move(upper));

                    BlockEntry lower;
                    lower.row_node = static_cast<std::uint32_t>(jnode);
                    lower.col_node = static_cast<std::uint32_t>(inode);
                    lower.values = transpose_block6(block);
                    column_entries.push_back(std::move(lower));
                }
            }
        }
    }
    out.candidate_upper_block_pairs = static_cast<std::size_t>(candidate_count);

    std::size_t directed_blocks = 0U;
    for (const auto& v : per_column) directed_blocks += v.size();
    if (directed_blocks > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("sparse A2 block nnz exceeds uint32");
    }
    out.blocks.reserve(directed_blocks);
    for (auto& v : per_column) {
        out.blocks.insert(out.blocks.end(),
                          std::make_move_iterator(v.begin()),
                          std::make_move_iterator(v.end()));
    }
    std::sort(out.blocks.begin(), out.blocks.end(), [](const BlockEntry& a, const BlockEntry& b) {
        if (a.row_node != b.row_node) return a.row_node < b.row_node;
        return a.col_node < b.col_node;
    });
    out.block_row_offsets.assign(out.nodes + 1U, 0U);
    std::size_t block_cursor = 0U;
    for (std::size_t row_node = 0U; row_node < out.nodes; ++row_node) {
        out.block_row_offsets[row_node] = static_cast<std::uint32_t>(block_cursor);
        while (block_cursor < out.blocks.size() &&
               static_cast<std::size_t>(out.blocks[block_cursor].row_node) == row_node) {
            ++block_cursor;
        }
    }
    out.block_row_offsets[out.nodes] = static_cast<std::uint32_t>(out.blocks.size());
    out.block_assembly_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - assembly_start).count();

    const auto csr_start = Clock::now();
    out.row_offsets.resize(out.n + 1U, 0U);
    std::vector<std::size_t> row_counts(out.n, 0U);
    for (const auto& entry : out.blocks) {
        const auto& ragg = transfer1.aggregates[entry.row_node];
        const auto& cagg = transfer1.aggregates[entry.col_node];
        for (std::size_t r = 0U; r < ragg.rank; ++r) {
            const std::size_t scalar_row = ragg.coarse_offset + r;
            for (std::size_t c = 0U; c < cagg.rank; ++c) {
                if (entry.values[r * kCandidates + c] != 0.0) ++row_counts[scalar_row];
            }
        }
    }
    std::size_t nnz = 0U;
    for (std::size_t row = 0U; row < out.n; ++row) {
        if (nnz > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("sparse A2 scalar nnz exceeds uint32");
        }
        out.row_offsets[row] = static_cast<std::uint32_t>(nnz);
        nnz += row_counts[row];
    }
    if (nnz > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("sparse A2 scalar nnz exceeds uint32");
    }
    out.row_offsets[out.n] = static_cast<std::uint32_t>(nnz);
    out.column_indices.resize(nnz);
    out.values_fp32.resize(nnz);
    std::vector<std::size_t> cursor(out.n, 0U);
    for (std::size_t row = 0U; row < out.n; ++row) cursor[row] = out.row_offsets[row];
    for (const auto& entry : out.blocks) {
        const auto& ragg = transfer1.aggregates[entry.row_node];
        const auto& cagg = transfer1.aggregates[entry.col_node];
        for (std::size_t r = 0U; r < ragg.rank; ++r) {
            const std::size_t scalar_row = ragg.coarse_offset + r;
            for (std::size_t c = 0U; c < cagg.rank; ++c) {
                const double value = entry.values[r * kCandidates + c];
                if (value == 0.0) continue;
                const std::size_t p = cursor[scalar_row]++;
                out.column_indices[p] = static_cast<std::uint32_t>(cagg.coarse_offset + c);
                out.values_fp32[p] = static_cast<float>(value);
            }
        }
    }
    out.csr_export_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - csr_start).count();
    out.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
    return out;
}

inline std::vector<double> apply_fp64(
    const ExactSparseA2& a,
    const CandidateTransfer& transfer1,
    const std::vector<double>& x) {
    if (x.size() != a.n || transfer1.aggregates.size() != a.nodes ||
        a.block_row_offsets.size() != a.nodes + 1U) {
        throw std::invalid_argument("sparse A2 apply size/layout mismatch");
    }
    std::vector<double> y(a.n, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t rn64 = 0; rn64 < static_cast<std::int64_t>(a.nodes); ++rn64) {
        const std::size_t row_node = static_cast<std::size_t>(rn64);
        const auto& ragg = transfer1.aggregates[row_node];
        for (std::size_t bp = a.block_row_offsets[row_node];
             bp < a.block_row_offsets[row_node + 1U]; ++bp) {
            const auto& entry = a.blocks[bp];
            const auto& cagg = transfer1.aggregates[entry.col_node];
            for (std::size_t r = 0U; r < ragg.rank; ++r) {
                double value = 0.0;
                for (std::size_t c = 0U; c < cagg.rank; ++c) {
                    value += entry.values[r * kCandidates + c] * x[cagg.coarse_offset + c];
                }
                y[ragg.coarse_offset + r] += value;
            }
        }
    }
    return y;
}

inline void apply_matrix_fp64(
    const ExactSparseA2& a,
    const CandidateTransfer& transfer1,
    const std::vector<double>& x,
    std::size_t cols,
    std::vector<double>& y) {
    if (cols == 0U || x.size() != a.n * cols ||
        transfer1.aggregates.size() != a.nodes ||
        a.block_row_offsets.size() != a.nodes + 1U) {
        throw std::invalid_argument("sparse A2 matrix apply size/layout mismatch");
    }
    y.assign(a.n * cols, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t rn64 = 0; rn64 < static_cast<std::int64_t>(a.nodes); ++rn64) {
        const std::size_t row_node = static_cast<std::size_t>(rn64);
        const auto& ragg = transfer1.aggregates[row_node];
        for (std::size_t bp = a.block_row_offsets[row_node];
             bp < a.block_row_offsets[row_node + 1U]; ++bp) {
            const auto& entry = a.blocks[bp];
            const auto& cagg = transfer1.aggregates[entry.col_node];
            for (std::size_t r = 0U; r < ragg.rank; ++r) {
                double* dst = y.data() + (ragg.coarse_offset + r) * cols;
                for (std::size_t c = 0U; c < cagg.rank; ++c) {
                    const double coeff = entry.values[r * kCandidates + c];
                    if (coeff == 0.0) continue;
                    const double* src = x.data() + (cagg.coarse_offset + c) * cols;
                    for (std::size_t q = 0U; q < cols; ++q) {
                        dst[q] += coeff * src[q];
                    }
                }
            }
        }
    }
}

inline m5_l2_setup::DenseP2 smoothed_p2_from_sparse_a2(
    const ExactSparseA2& a,
    const CandidateTransfer& transfer1,
    const CandidateTransfer& transfer2,
    const L1BlockMetric& block2,
    double omega2) {
    const auto start = Clock::now();
    m5_l2_setup::DenseP2 p;
    p.rows = block2.dofs();
    p.cols = transfer2.coarse_dofs;
    if (p.rows != a.n || p.cols == 0U) {
        throw std::invalid_argument("sparse A2 P2 shape mismatch");
    }
    p.fp64.assign(p.rows * p.cols, 0.0);
    for (const auto& aggregate : transfer2.aggregates) {
        for (std::size_t row = 0U; row < aggregate.fine_dofs.size(); ++row) {
            const std::size_t fine = aggregate.fine_dofs[row];
            for (std::size_t q = 0U; q < aggregate.rank; ++q) {
                p.fp64[fine * p.cols + aggregate.coarse_offset + q] =
                    aggregate.q_values[row * aggregate.rank + q];
            }
        }
    }
    std::vector<double> ap;
    std::vector<double> scaled;
    apply_matrix_fp64(a, transfer1, p.fp64, p.cols, ap);
    m5_fast_setup::block_solve_matrix(block2, ap, p.cols, scaled);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(p.fp64.size()); ++i64) {
        const std::size_t i = static_cast<std::size_t>(i64);
        p.fp64[i] -= omega2 * scaled[i];
    }
    p.assembly_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return p;
}

inline LocalBottomReference bottom_from_sparse_a2_p2(
    const ExactSparseA2& a,
    const CandidateTransfer& transfer1,
    const m5_l2_setup::DenseP2& p2) {
    const auto start = Clock::now();
    if (p2.rows != a.n || p2.cols == 0U) {
        throw std::invalid_argument("sparse bottom P2/A2 shape mismatch");
    }
    std::vector<double> ap;
    apply_matrix_fp64(a, transfer1, p2.fp64, p2.cols, ap);
    LocalBottomReference bottom;
    bottom.values.assign(p2.cols * p2.cols, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(p2.cols); ++i64) {
        const std::size_t i = static_cast<std::size_t>(i64);
        for (std::size_t j = 0U; j < p2.cols; ++j) {
            double value = 0.0;
            for (std::size_t r = 0U; r < p2.rows; ++r) {
                value += p2.fp64[r * p2.cols + i] * ap[r * p2.cols + j];
            }
            bottom.values[i * p2.cols + j] = value;
        }
    }
    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0U; i < p2.cols; ++i) {
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
    for (std::size_t i = 0U; i < p2.cols; ++i) {
        for (std::size_t j = 0U; j <= i; ++j) {
            double value = bottom.values[i * p2.cols + j];
            for (std::size_t k = 0U; k < j; ++k) {
                value -= bottom.factor.lower[i * p2.cols + k] *
                         bottom.factor.lower[j * p2.cols + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("sparse bottom lost SPD");
                }
                bottom.factor.min_pivot = std::min(bottom.factor.min_pivot, value);
                bottom.factor.lower[i * p2.cols + j] = std::sqrt(value);
            } else {
                bottom.factor.lower[i * p2.cols + j] =
                    value / bottom.factor.lower[j * p2.cols + j];
            }
        }
    }
    bottom.assembly_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return bottom;
}

}  // namespace m5_sparse_a2
