#pragma once

// Experimental exact A1 assembly retaining the proven thread-local unordered-map
// element accumulation while replacing the final global hash reduction with a
// k-way merge of sorted thread-local entry streams. This isolates reduction
// overhead without requiring a preindexed element plan.
//
// Requires recursive_sa_local_l2_helpers.inc and
// recursive_sa_actual_a1_strength_local_helpers.inc.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace m5_sorted_reduce_a1 {

using Clock = std::chrono::steady_clock;
using BlockMap = std::unordered_map<std::uint64_t, CombinedBlock>;

struct Result {
    BlockMap blocks;
    double local_accumulation_ms{0.0};
    double stream_sort_ms{0.0};
    double merge_ms{0.0};
    double export_map_ms{0.0};
    double total_ms{0.0};
    std::size_t summed_thread_entries{0U};
    std::size_t unique_pairs{0U};
    int threads{1};
};

struct EntryRef {
    std::uint64_t key{0U};
    const CombinedBlock* block{nullptr};
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
    const std::size_t threads = static_cast<std::size_t>(thread_count);
    std::vector<BlockMap> local_maps(threads);
    const std::size_t reserve_each = std::max<std::size_t>(
        1024U, (supports.size() * 28U) / threads);
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

    std::size_t summed_entries = 0U;
    for (const auto& map : local_maps) summed_entries += map.size();

    const auto sort_start = Clock::now();
    std::vector<std::vector<EntryRef>> streams(threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t tid64 = 0; tid64 < static_cast<std::int64_t>(threads); ++tid64) {
        const std::size_t tid = static_cast<std::size_t>(tid64);
        auto& stream = streams[tid];
        stream.reserve(local_maps[tid].size());
        for (const auto& item : local_maps[tid]) {
            stream.push_back({item.first, &item.second});
        }
        std::sort(stream.begin(), stream.end(), [](const EntryRef& a, const EntryRef& b) {
            return a.key < b.key;
        });
    }
    const double sort_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - sort_start).count();

    struct Cursor {
        std::uint64_t key{0U};
        std::size_t tid{0U};
        std::size_t pos{0U};
    };
    struct Greater {
        bool operator()(const Cursor& a, const Cursor& b) const {
            if (a.key != b.key) return a.key > b.key;
            return a.tid > b.tid;
        }
    };

    const auto merge_start = Clock::now();
    std::priority_queue<Cursor, std::vector<Cursor>, Greater> heap;
    for (std::size_t tid = 0U; tid < threads; ++tid) {
        if (!streams[tid].empty()) heap.push({streams[tid][0U].key, tid, 0U});
    }

    std::vector<std::pair<std::uint64_t, CombinedBlock>> merged;
    merged.reserve(summed_entries);
    while (!heap.empty()) {
        const std::uint64_t key = heap.top().key;
        CombinedBlock block{};
        while (!heap.empty() && heap.top().key == key) {
            const Cursor cur = heap.top();
            heap.pop();
            const auto& entry = streams[cur.tid][cur.pos];
            const auto& src = *entry.block;
            for (std::size_t k = 0U; k < block.size(); ++k) block[k] += src[k];
            const std::size_t next = cur.pos + 1U;
            if (next < streams[cur.tid].size()) {
                heap.push({streams[cur.tid][next].key, cur.tid, next});
            }
        }
        merged.emplace_back(key, std::move(block));
    }
    const double merge_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - merge_start).count();

    const auto export_start = Clock::now();
    BlockMap reduced;
    reduced.reserve(std::max<std::size_t>(supports.size() * 24U, merged.size()));
    for (auto& item : merged) reduced.emplace(item.first, std::move(item.second));
    const double export_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - export_start).count();

    Result out;
    out.blocks = std::move(reduced);
    out.local_accumulation_ms = local_ms;
    out.stream_sort_ms = sort_ms;
    out.merge_ms = merge_ms;
    out.export_map_ms = export_ms;
    out.total_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - total_start).count();
    out.summed_thread_entries = summed_entries;
    out.unique_pairs = merged.size();
    out.threads = thread_count;
    return out;
}

}  // namespace m5_sorted_reduce_a1
