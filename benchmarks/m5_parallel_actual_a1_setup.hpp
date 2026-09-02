#pragma once

// Setup-only parallel exact actual-A1 off-diagonal assembly.
// Requires recursive_sa_local_l2_helpers.inc and
// recursive_sa_actual_a1_strength_local_helpers.inc.
//
// Each OpenMP worker accumulates a private block map over a static contiguous
// element range. The private maps are then reduced in worker-index order. This
// removes synchronization from the expensive element loop while keeping the
// reduction deterministic for a fixed thread count.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace m5_parallel_a1 {

using Clock = std::chrono::steady_clock;
using BlockMap = std::unordered_map<std::uint64_t, CombinedBlock>;

struct Result {
    BlockMap blocks;
    double local_accumulation_ms{0.0};
    double deterministic_reduction_ms{0.0};
    double total_ms{0.0};
    std::size_t summed_thread_entries{0U};
    int threads{1};
};

inline Result assemble(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const std::vector<FineBasisSupport>& supports,
    const std::vector<std::vector<std::uint32_t>>& element_supports) {
    const auto total_start = Clock::now();
    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);

    int thread_count = 1;
#ifdef _OPENMP
    thread_count = std::max(1, omp_get_max_threads());
#endif
    std::vector<BlockMap> local_maps(static_cast<std::size_t>(thread_count));
    const std::size_t reserve_each = std::max<std::size_t>(
        1024U, (supports.size() * 28U) / static_cast<std::size_t>(thread_count));
    for (auto& map : local_maps) map.reserve(reserve_each);

    struct ElementBasis {
        std::uint32_t aggregate{0U};
        std::size_t rank{0U};
        std::array<double, 24U * kCandidates> u{};
        std::array<double, 24U * kCandidates> ku{};
    };

    const auto local_start = Clock::now();
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& blocks = local_maps[static_cast<std::size_t>(tid)];

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (std::int64_t e64 = 0;
             e64 < static_cast<std::int64_t>(element_supports.size()); ++e64) {
            const std::size_t element = static_cast<std::size_t>(e64);
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

            for (std::size_t ia = 0U; ia < local.size(); ++ia) {
                for (std::size_t ib = ia + 1U; ib < local.size(); ++ib) {
                    const ElementBasis* left = &local[ia];
                    const ElementBasis* right = &local[ib];
                    if (left->aggregate > right->aggregate) std::swap(left, right);
                    const auto key = combined_pair_key(left->aggregate, right->aggregate);
                    auto [it, inserted] = blocks.try_emplace(key, CombinedBlock{});
                    auto& block = it->second;
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

    std::size_t total_entries = 0U;
    for (const auto& map : local_maps) total_entries += map.size();

    const auto reduction_start = Clock::now();
    BlockMap reduced;
    reduced.reserve(std::max<std::size_t>(supports.size() * 24U, total_entries));
    for (std::size_t tid = 0U; tid < local_maps.size(); ++tid) {
        std::vector<std::uint64_t> keys;
        keys.reserve(local_maps[tid].size());
        for (const auto& item : local_maps[tid]) keys.push_back(item.first);
        std::sort(keys.begin(), keys.end());
        for (const auto key : keys) {
            const auto& src = local_maps[tid].at(key);
            auto [it, inserted] = reduced.try_emplace(key, CombinedBlock{});
            auto& dst = it->second;
            for (std::size_t k = 0U; k < dst.size(); ++k) dst[k] += src[k];
        }
    }
    const double reduction_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - reduction_start).count();

    Result out;
    out.blocks = std::move(reduced);
    out.local_accumulation_ms = local_ms;
    out.deterministic_reduction_ms = reduction_ms;
    out.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
    out.summed_thread_entries = total_entries;
    out.threads = thread_count;
    return out;
}

inline double block_map_relative_error(const BlockMap& candidate, const BlockMap& reference) {
    double diff2 = 0.0;
    double ref2 = 0.0;
    for (const auto& item : reference) {
        const auto it = candidate.find(item.first);
        for (std::size_t k = 0U; k < item.second.size(); ++k) {
            const double ref = item.second[k];
            const double got = it == candidate.end() ? 0.0 : it->second[k];
            const double d = got - ref;
            diff2 += d * d;
            ref2 += ref * ref;
        }
    }
    for (const auto& item : candidate) {
        if (reference.find(item.first) != reference.end()) continue;
        for (const double value : item.second) diff2 += value * value;
    }
    return ref2 > 0.0 ? std::sqrt(diff2 / ref2) : std::sqrt(diff2);
}

}  // namespace m5_parallel_a1
