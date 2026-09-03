#pragma once

// Exact setup-only dense A2 assembly exploiting Galerkin symmetry.
// Each aggregate pair (i,j) is evaluated once as P_i^T A1 P_j and its
// transpose is written directly. The established two-sided assembly remains
// available in m5_fast_hierarchy_setup.hpp as a reference implementation.

#include "m5_l2_dense_setup.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace m5_symmetric_a2 {

using Clock = std::chrono::steady_clock;

inline m5_l2_setup::DenseA2 assemble_from_cached_applied(
    const CandidateTransfer& transfer1,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis,
    const std::vector<LocalColumns>& applied_l2_basis) {
    if (l2_basis.size() != transfer1.aggregates.size() ||
        applied_l2_basis.size() != l2_basis.size()) {
        throw std::invalid_argument("symmetric A2 cached support size mismatch");
    }

    const auto start = Clock::now();
    m5_l2_setup::DenseA2 out;
    out.n = transfer1.coarse_dofs;
    out.fp64.assign(out.n * out.n, 0.0);

    // jnode owns all blocks (inode,jnode) with inode <= jnode. Distinct
    // aggregate pairs write disjoint dense entries, including their mirrored
    // transpose, so this loop is race-free.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::int64_t j64 = 0;
         j64 < static_cast<std::int64_t>(l2_basis.size()); ++j64) {
        const std::size_t jnode = static_cast<std::size_t>(j64);
        const auto& applied = applied_l2_basis[jnode];
        const auto& jagg = transfer1.aggregates[jnode];
        const std::size_t joff = jagg.coarse_offset;
        const std::size_t jrank = jagg.rank;

        for (std::size_t inode = 0U; inode <= jnode; ++inode) {
            const auto block = local_cross_gram(l2_basis[inode], applied, block1);
            const auto& iagg = transfer1.aggregates[inode];
            const std::size_t ioff = iagg.coarse_offset;
            const std::size_t irank = iagg.rank;

            if (inode == jnode) {
                if (irank != jrank) {
                    throw std::runtime_error("symmetric A2 diagonal block rank mismatch");
                }
                // The diagonal block is itself symmetric. Average its two
                // floating-point accumulation orders exactly as the reference
                // metric path does, then store one exact symmetric block.
                for (std::size_t i = 0U; i < irank; ++i) {
                    for (std::size_t j = i; j < jrank; ++j) {
                        const double value = i == j
                            ? block[i * kCandidates + j]
                            : 0.5 * (block[i * kCandidates + j] +
                                     block[j * kCandidates + i]);
                        out.fp64[(ioff + i) * out.n + (joff + j)] = value;
                        out.fp64[(joff + j) * out.n + (ioff + i)] = value;
                    }
                }
            } else {
                // One exact cross block determines the opposite block by
                // Galerkin symmetry A2_ji = A2_ij^T.
                for (std::size_t i = 0U; i < irank; ++i) {
                    for (std::size_t j = 0U; j < jrank; ++j) {
                        const double value = block[i * kCandidates + j];
                        out.fp64[(ioff + i) * out.n + (joff + j)] = value;
                        out.fp64[(joff + j) * out.n + (ioff + i)] = value;
                    }
                }
            }
        }
    }

    for (std::size_t i = 0U; i < out.n; ++i) {
        const double d = out.fp64[i * out.n + i];
        if (!(d > 0.0) || !std::isfinite(d)) {
            throw std::runtime_error("symmetric A2 diagonal invalid");
        }
    }

    // Symmetry is enforced by construction rather than repaired afterward.
    out.symmetry_relative_defect = 0.0;
    out.assembly_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    return out;
}

// Decision-probe result for an exact support-pruned dense A2 assembly. The
// final representation is intentionally still dense; this only tests whether
// setup can skip block pairs that are algebraically guaranteed to be zero.
struct SupportPrunedResult {
    m5_l2_setup::DenseA2 dense;
    std::size_t all_block_pairs{0U};
    std::size_t candidate_block_pairs{0U};
    double support_index_ms{0.0};
    double assembly_ms{0.0};
    double total_ms{0.0};
};

inline SupportPrunedResult assemble_from_cached_applied_support_pruned(
    const CandidateTransfer& transfer1,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis,
    const std::vector<LocalColumns>& applied_l2_basis) {
    if (l2_basis.size() != transfer1.aggregates.size() ||
        applied_l2_basis.size() != l2_basis.size()) {
        throw std::invalid_argument("support-pruned A2 cached support size mismatch");
    }

    SupportPrunedResult result;
    const auto total_start = Clock::now();
    const auto index_start = Clock::now();

    // local_cross_gram(left,right) can only be non-zero when left and right
    // contain at least one common L1 block row. Invert the L2 basis supports so
    // each applied A1*P1 column can enumerate only potentially coupled bases.
    std::vector<std::vector<std::uint32_t>> basis_by_l1_row(block1.nodes());
    for (std::size_t inode = 0U; inode < l2_basis.size(); ++inode) {
        for (const auto& entry : l2_basis[inode].values) {
            const std::size_t row = static_cast<std::size_t>(entry.first);
            if (row >= basis_by_l1_row.size()) {
                throw std::out_of_range("support-pruned A2 L1 row out of range");
            }
            basis_by_l1_row[row].push_back(static_cast<std::uint32_t>(inode));
        }
    }
    result.support_index_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - index_start).count();

    auto& out = result.dense;
    out.n = transfer1.coarse_dofs;
    out.fp64.assign(out.n * out.n, 0.0);
    const std::size_t basis_count = l2_basis.size();
    result.all_block_pairs = basis_count * (basis_count + 1U) / 2U;

    std::uint64_t candidate_pair_count = 0U;
    const auto assembly_start = Clock::now();
#ifdef _OPENMP
#pragma omp parallel reduction(+:candidate_pair_count)
#endif
    {
        std::vector<std::uint32_t> marks(basis_count, 0U);
        std::vector<std::uint32_t> candidates;
        candidates.reserve(std::min<std::size_t>(basis_count, 128U));
        std::uint32_t generation = 0U;

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
        for (std::int64_t j64 = 0;
             j64 < static_cast<std::int64_t>(basis_count); ++j64) {
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
                    throw std::out_of_range("support-pruned A2 applied row out of range");
                }
                for (const auto inode_u32 : basis_by_l1_row[row]) {
                    const std::size_t inode = static_cast<std::size_t>(inode_u32);
                    if (inode > jnode || marks[inode] == generation) continue;
                    marks[inode] = generation;
                    candidates.push_back(inode_u32);
                }
            }

            // SPD requires the diagonal block; include it defensively even if
            // a pathological support container failed to expose an overlap.
            if (marks[jnode] != generation) {
                marks[jnode] = generation;
                candidates.push_back(static_cast<std::uint32_t>(jnode));
            }
            candidate_pair_count += static_cast<std::uint64_t>(candidates.size());

            const auto& jagg = transfer1.aggregates[jnode];
            const std::size_t joff = jagg.coarse_offset;
            const std::size_t jrank = jagg.rank;

            for (const auto inode_u32 : candidates) {
                const std::size_t inode = static_cast<std::size_t>(inode_u32);
                const auto block = local_cross_gram(l2_basis[inode], applied, block1);
                const auto& iagg = transfer1.aggregates[inode];
                const std::size_t ioff = iagg.coarse_offset;
                const std::size_t irank = iagg.rank;

                if (inode == jnode) {
                    if (irank != jrank) {
                        throw std::runtime_error("support-pruned A2 diagonal block rank mismatch");
                    }
                    for (std::size_t i = 0U; i < irank; ++i) {
                        for (std::size_t j = i; j < jrank; ++j) {
                            const double value = i == j
                                ? block[i * kCandidates + j]
                                : 0.5 * (block[i * kCandidates + j] +
                                         block[j * kCandidates + i]);
                            out.fp64[(ioff + i) * out.n + (joff + j)] = value;
                            out.fp64[(joff + j) * out.n + (ioff + i)] = value;
                        }
                    }
                } else {
                    for (std::size_t i = 0U; i < irank; ++i) {
                        for (std::size_t j = 0U; j < jrank; ++j) {
                            const double value = block[i * kCandidates + j];
                            out.fp64[(ioff + i) * out.n + (joff + j)] = value;
                            out.fp64[(joff + j) * out.n + (ioff + i)] = value;
                        }
                    }
                }
            }
        }
    }
    result.candidate_block_pairs = static_cast<std::size_t>(candidate_pair_count);
    result.assembly_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - assembly_start).count();

    for (std::size_t i = 0U; i < out.n; ++i) {
        const double d = out.fp64[i * out.n + i];
        if (!(d > 0.0) || !std::isfinite(d)) {
            throw std::runtime_error("support-pruned A2 diagonal invalid");
        }
    }
    out.symmetry_relative_defect = 0.0;
    out.assembly_ms = result.assembly_ms;
    result.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
    return result;
}

}  // namespace m5_symmetric_a2
