#pragma once

// Setup-only exact actual-A1 assembly using a preindexed aggregate-pair plan.
// The hot element loop performs no hash-table lookup or insertion: each worker
// writes into a compact contiguous block array indexed before accumulation.
// Reduction is then performed over stable global pair IDs before exporting the
// established BlockMap representation required by the downstream hierarchy.
//
// Requires recursive_sa_local_l2_helpers.inc and
// recursive_sa_actual_a1_strength_local_helpers.inc.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace m5_indexed_a1 {

using Clock = std::chrono::steady_clock;
using BlockMap = std::unordered_map<std::uint64_t, CombinedBlock>;

struct Result {
    BlockMap blocks;
    double plan_ms{0.0};
    double local_accumulation_ms{0.0};
    double indexed_reduction_ms{0.0};
    double export_map_ms{0.0};
    double total_ms{0.0};
    std::size_t unique_pairs{0U};
    std::size_t summed_thread_pairs{0U};
    std::size_t pair_contributions{0U};
    std::size_t local_block_bytes{0U};
    std::size_t index_bytes{0U};
    int threads{1};
};

inline std::uint64_t ordered_pair_key(std::uint32_t a, std::uint32_t b) {
    if (a > b) std::swap(a, b);
    return combined_pair_key(a, b);
}

inline Result assemble(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const std::vector<FineBasisSupport>& supports,
    const std::vector<std::vector<std::uint32_t>>& element_supports) {
    const auto total_start = Clock::now();
    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    const std::size_t element_count = element_supports.size();

    int thread_count = 1;
#ifdef _OPENMP
    thread_count = std::max(1, omp_get_max_threads());
#endif
    const std::size_t threads = static_cast<std::size_t>(thread_count);

    const auto plan_start = Clock::now();
    std::vector<std::vector<std::uint64_t>> thread_keys(threads);

    // Use an explicit contiguous partition so the precomputed pair IDs and the
    // accumulation pass have identical element ownership independent of the
    // OpenMP runtime's schedule(static) implementation details.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t tid64 = 0; tid64 < static_cast<std::int64_t>(threads); ++tid64) {
        const std::size_t tid = static_cast<std::size_t>(tid64);
        const std::size_t begin = element_count * tid / threads;
        const std::size_t end = element_count * (tid + 1U) / threads;
        auto& keys = thread_keys[tid];
        for (std::size_t element = begin; element < end; ++element) {
            const auto& active = element_supports[element];
            if (active.size() < 2U) continue;
            const std::size_t pairs = active.size() * (active.size() - 1U) / 2U;
            keys.reserve(keys.size() + pairs);
            for (std::size_t ia = 0U; ia < active.size(); ++ia) {
                for (std::size_t ib = ia + 1U; ib < active.size(); ++ib) {
                    keys.push_back(ordered_pair_key(active[ia], active[ib]));
                }
            }
        }
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    }

    std::size_t summed_thread_pairs = 0U;
    std::vector<std::uint64_t> global_keys;
    for (const auto& keys : thread_keys) {
        summed_thread_pairs += keys.size();
        global_keys.insert(global_keys.end(), keys.begin(), keys.end());
    }
    std::sort(global_keys.begin(), global_keys.end());
    global_keys.erase(std::unique(global_keys.begin(), global_keys.end()), global_keys.end());
    const std::size_t global_pairs = global_keys.size();
    if (global_pairs > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("indexed A1 pair count exceeds int32 index range");
    }

    // Map each global pair ID to its compact thread-local pair ID. This table is
    // small compared with the block payload (threads * ~100k int32 entries).
    std::vector<std::int32_t> global_to_local(
        threads * global_pairs, static_cast<std::int32_t>(-1));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t tid64 = 0; tid64 < static_cast<std::int64_t>(threads); ++tid64) {
        const std::size_t tid = static_cast<std::size_t>(tid64);
        const auto& keys = thread_keys[tid];
        for (std::size_t local = 0U; local < keys.size(); ++local) {
            const auto it = std::lower_bound(global_keys.begin(), global_keys.end(), keys[local]);
            if (it == global_keys.end() || *it != keys[local]) {
                throw std::runtime_error("indexed A1 global pair lookup failed");
            }
            const std::size_t global = static_cast<std::size_t>(it - global_keys.begin());
            global_to_local[tid * global_pairs + global] = static_cast<std::int32_t>(local);
        }
    }

    // Flatten each element's pair list to the compact local pair IDs used by its
    // owner thread. The hot loop can then consume these IDs sequentially.
    std::vector<std::size_t> element_pair_offsets(element_count + 1U, 0U);
    for (std::size_t element = 0U; element < element_count; ++element) {
        const std::size_t n = element_supports[element].size();
        element_pair_offsets[element + 1U] =
            element_pair_offsets[element] + (n < 2U ? 0U : n * (n - 1U) / 2U);
    }
    const std::size_t pair_contributions = element_pair_offsets.back();
    std::vector<std::uint32_t> element_pair_local(pair_contributions, 0U);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t tid64 = 0; tid64 < static_cast<std::int64_t>(threads); ++tid64) {
        const std::size_t tid = static_cast<std::size_t>(tid64);
        const std::size_t begin = element_count * tid / threads;
        const std::size_t end = element_count * (tid + 1U) / threads;
        const auto& keys = thread_keys[tid];
        for (std::size_t element = begin; element < end; ++element) {
            const auto& active = element_supports[element];
            std::size_t cursor = element_pair_offsets[element];
            for (std::size_t ia = 0U; ia < active.size(); ++ia) {
                for (std::size_t ib = ia + 1U; ib < active.size(); ++ib) {
                    const auto key = ordered_pair_key(active[ia], active[ib]);
                    const auto it = std::lower_bound(keys.begin(), keys.end(), key);
                    if (it == keys.end() || *it != key) {
                        throw std::runtime_error("indexed A1 local pair lookup failed");
                    }
                    const std::size_t local = static_cast<std::size_t>(it - keys.begin());
                    if (local > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                        throw std::runtime_error("indexed A1 local pair index exceeds uint32 range");
                    }
                    element_pair_local[cursor++] = static_cast<std::uint32_t>(local);
                }
            }
        }
    }

    std::vector<std::vector<CombinedBlock>> local_blocks(threads);
    for (std::size_t tid = 0U; tid < threads; ++tid) {
        local_blocks[tid].resize(thread_keys[tid].size());
    }
    const double plan_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - plan_start).count();

    struct ElementBasis {
        std::uint32_t aggregate{0U};
        std::size_t rank{0U};
        std::array<double, 24U * kCandidates> u{};
        std::array<double, 24U * kCandidates> ku{};
    };

    const auto local_start = Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t tid64 = 0; tid64 < static_cast<std::int64_t>(threads); ++tid64) {
        const std::size_t tid = static_cast<std::size_t>(tid64);
        const std::size_t begin = element_count * tid / threads;
        const std::size_t end = element_count * (tid + 1U) / threads;
        auto& blocks = local_blocks[tid];

        for (std::size_t element = begin; element < end; ++element) {
            const auto& active = element_supports[element];
            if (active.size() < 2U) continue;
            const auto ijk = decode_element(mesh, element);
            const auto nodes = mesh.element_nodes(ijk[0], ijk[1], ijk[2]);

            std::vector<ElementBasis> local;
            local.reserve(active.size());
            for (const auto aggregate : active) {
                ElementBasis basis;
                basis.aggregate = aggregate;
                basis.rank = supports[aggregate].rank;
                for (std::size_t n = 0U; n < 8U; ++n) {
                    const auto it = supports[aggregate].values.find(
                        static_cast<std::size_t>(nodes[n]));
                    if (it == supports[aggregate].values.end()) continue;
                    for (std::size_t c = 0U; c < 3U; ++c) {
                        const std::size_t ldof = 3U * n + c;
                        for (std::size_t q = 0U; q < basis.rank; ++q) {
                            basis.u[ldof * kCandidates + q] =
                                it->second[c * kCandidates + q];
                        }
                    }
                }
                for (std::size_t i = 0U; i < 24U; ++i) {
                    for (std::size_t q = 0U; q < basis.rank; ++q) {
                        double value = 0.0;
                        for (std::size_t j = 0U; j < 24U; ++j) {
                            value += ke[i][j] * basis.u[j * kCandidates + q];
                        }
                        basis.ku[i * kCandidates + q] = value;
                    }
                }
                local.push_back(std::move(basis));
            }

            std::size_t cursor = element_pair_offsets[element];
            for (std::size_t ia = 0U; ia < local.size(); ++ia) {
                for (std::size_t ib = ia + 1U; ib < local.size(); ++ib) {
                    const ElementBasis* left = &local[ia];
                    const ElementBasis* right = &local[ib];
                    if (left->aggregate > right->aggregate) std::swap(left, right);
                    const std::size_t local_pair = element_pair_local[cursor++];
                    auto& block = blocks[local_pair];
                    for (std::size_t q = 0U; q < left->rank; ++q) {
                        for (std::size_t r = 0U; r < right->rank; ++r) {
                            double value = 0.0;
                            for (std::size_t i = 0U; i < 24U; ++i) {
                                value += left->u[i * kCandidates + q] *
                                         right->ku[i * kCandidates + r];
                            }
                            block[q * kCandidates + r] += value;
                        }
                    }
                }
            }
        }
    }
    const double local_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - local_start).count();

    const auto reduction_start = Clock::now();
    std::vector<CombinedBlock> global_blocks(global_pairs);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t global64 = 0;
         global64 < static_cast<std::int64_t>(global_pairs); ++global64) {
        const std::size_t global = static_cast<std::size_t>(global64);
        auto& dst = global_blocks[global];
        for (std::size_t tid = 0U; tid < threads; ++tid) {
            const std::int32_t local = global_to_local[tid * global_pairs + global];
            if (local < 0) continue;
            const auto& src = local_blocks[tid][static_cast<std::size_t>(local)];
            for (std::size_t k = 0U; k < dst.size(); ++k) dst[k] += src[k];
        }
    }
    const double reduction_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - reduction_start).count();

    const auto export_start = Clock::now();
    BlockMap reduced;
    reduced.reserve(std::max<std::size_t>(supports.size() * 24U, global_pairs));
    for (std::size_t global = 0U; global < global_pairs; ++global) {
        reduced.emplace(global_keys[global], std::move(global_blocks[global]));
    }
    const double export_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - export_start).count();

    Result out;
    out.blocks = std::move(reduced);
    out.plan_ms = plan_ms;
    out.local_accumulation_ms = local_ms;
    out.indexed_reduction_ms = reduction_ms;
    out.export_map_ms = export_ms;
    out.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
    out.unique_pairs = global_pairs;
    out.summed_thread_pairs = summed_thread_pairs;
    out.pair_contributions = pair_contributions;
    out.local_block_bytes = summed_thread_pairs * sizeof(CombinedBlock);
    out.index_bytes = global_to_local.size() * sizeof(std::int32_t) +
                      element_pair_local.size() * sizeof(std::uint32_t) +
                      element_pair_offsets.size() * sizeof(std::size_t);
    out.threads = thread_count;
    return out;
}

}  // namespace m5_indexed_a1
