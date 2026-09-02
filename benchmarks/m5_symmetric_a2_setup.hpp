#pragma once

// Exact setup-only dense A2 assembly exploiting Galerkin symmetry.
// Each aggregate pair (i,j) is evaluated once as P_i^T A1 P_j and its
// transpose is written directly. The established two-sided assembly remains
// available in m5_fast_hierarchy_setup.hpp as a reference implementation.

#include "m5_l2_dense_setup.hpp"

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

}  // namespace m5_symmetric_a2
