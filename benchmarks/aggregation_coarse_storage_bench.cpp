#include "gfss/aggregation_coarse_space.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parse_size(const char* text, const char* name) {
    const auto value = std::stoull(text);
    if (value == 0ULL) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

struct StorageAudit {
    std::size_t directed_blocks{0};
    std::size_t upper_blocks{0};
    std::size_t rank_aware_scalar_values{0};
    std::size_t symmetric_rank_aware_scalar_values{0};
    std::size_t min_blocks_per_row{0};
    std::size_t max_blocks_per_row{0};
    double avg_blocks_per_row{0.0};
};

StorageAudit audit_storage(const gfss::ElasticityAggregationCoarseSpace& space) {
    const std::size_t n = space.aggregates.size();
    if (n == 0U) throw std::runtime_error("coarse space has no aggregates");

    std::vector<std::vector<std::uint32_t>> rows(n);
    for (std::size_t a = 0; a < n; ++a) {
        rows[a].push_back(static_cast<std::uint32_t>(a));
    }

    const auto& graph = space.graph;
    for (std::size_t node = 0; node < graph.coordinates.size(); ++node) {
        const auto a = space.aggregate_of_node[node];
        if (a == std::numeric_limits<std::uint32_t>::max()) continue;
        for (std::uint32_t p = graph.row_offsets[node]; p < graph.row_offsets[node + 1U]; ++p) {
            const auto other_node = graph.column_indices[p];
            const auto b = space.aggregate_of_node[other_node];
            if (b == std::numeric_limits<std::uint32_t>::max()) continue;
            rows[a].push_back(b);
        }
    }

    StorageAudit audit;
    audit.min_blocks_per_row = std::numeric_limits<std::size_t>::max();
    double row_sum = 0.0;
    for (std::size_t a = 0; a < n; ++a) {
        auto& row = rows[a];
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        audit.directed_blocks += row.size();
        audit.min_blocks_per_row = std::min(audit.min_blocks_per_row, row.size());
        audit.max_blocks_per_row = std::max(audit.max_blocks_per_row, row.size());
        row_sum += static_cast<double>(row.size());

        const std::size_t ra = space.aggregates[a].rank;
        for (const auto b : row) {
            const std::size_t rb = space.aggregates[b].rank;
            audit.rank_aware_scalar_values += ra * rb;
            if (a < b) {
                ++audit.upper_blocks;
                audit.symmetric_rank_aware_scalar_values += ra * rb;
            } else if (a == b) {
                ++audit.upper_blocks;
                audit.symmetric_rank_aware_scalar_values += ra * (ra + 1U) / 2U;
            }
        }
    }
    audit.avg_blocks_per_row = row_sum / static_cast<double>(n);
    return audit;
}

std::size_t bsr_bytes(std::size_t block_rows,
                      std::size_t blocks,
                      std::size_t scalar_values,
                      std::size_t value_bytes) {
    return scalar_values * value_bytes +
           blocks * sizeof(std::uint32_t) +
           (block_rows + 1U) * sizeof(std::uint32_t);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t target = argc > 1 ? parse_size(argv[1], "target aggregate nodes") : 12U;
        const std::size_t minimum = argc > 2 ? parse_size(argv[2], "minimum aggregate nodes") : 4U;

        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        auto graph = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph), {target, minimum, 1.0e-10});
        const auto audit = audit_storage(space);

        const std::size_t rows = space.aggregates.size();
        const std::size_t full6_values = audit.directed_blocks * 36U;
        const std::size_t fp32_full6 = bsr_bytes(rows, audit.directed_blocks, full6_values, 4U);
        const std::size_t fp16_full6 = bsr_bytes(rows, audit.directed_blocks, full6_values, 2U);
        const std::size_t fp32_rank = bsr_bytes(
            rows, audit.directed_blocks, audit.rank_aware_scalar_values, 4U);
        const std::size_t fp16_rank = bsr_bytes(
            rows, audit.directed_blocks, audit.rank_aware_scalar_values, 2U);
        const std::size_t fp32_sym_rank = bsr_bytes(
            rows, audit.upper_blocks, audit.symmetric_rank_aware_scalar_values, 4U);

        const double fine_dofs = static_cast<double>(space.fine_free_dofs);
        const double transfer_bpd = static_cast<double>(
            space.estimated_matrix_free_transfer_payload_bytes) / fine_dofs;
        const double coarse_vec_bpd = 4.0 * static_cast<double>(space.coarse_dofs) / fine_dofs;

        std::cout << "GFSS M5 aggregation coarse-storage audit\n"
                  << "fine_stiffness_matrix=not_assembled\n"
                  << "coarse_mesh_required=false\n"
                  << "fine_mesh=64x64x8 fine_free_dofs=" << space.fine_free_dofs
                  << " coarse_dofs=" << space.coarse_dofs
                  << " aggregates=" << rows << '\n'
                  << std::fixed << std::setprecision(6)
                  << "fine_to_coarse_dof_ratio="
                  << static_cast<double>(space.fine_free_dofs) / space.coarse_dofs << '\n'
                  << "coarse_blocks_directed=" << audit.directed_blocks
                  << " coarse_blocks_upper=" << audit.upper_blocks << '\n'
                  << "blocks_per_row_min=" << audit.min_blocks_per_row
                  << " blocks_per_row_avg=" << audit.avg_blocks_per_row
                  << " blocks_per_row_max=" << audit.max_blocks_per_row << '\n'
                  << "rank_aware_scalar_values=" << audit.rank_aware_scalar_values << '\n'
                  << "fp32_full6_bsr_bytes_per_fine_dof="
                  << static_cast<double>(fp32_full6) / fine_dofs << '\n'
                  << "fp16_full6_bsr_bytes_per_fine_dof="
                  << static_cast<double>(fp16_full6) / fine_dofs << '\n'
                  << "fp32_rank_aware_bsr_bytes_per_fine_dof="
                  << static_cast<double>(fp32_rank) / fine_dofs << '\n'
                  << "fp16_rank_aware_bsr_bytes_per_fine_dof="
                  << static_cast<double>(fp16_rank) / fine_dofs << '\n'
                  << "fp32_symmetric_rank_aware_lower_bound_bytes_per_fine_dof="
                  << static_cast<double>(fp32_sym_rank) / fine_dofs << '\n'
                  << "matrix_free_transfer_payload_bytes_per_fine_dof=" << transfer_bpd << '\n'
                  << "one_fp32_coarse_vector_bytes_per_fine_dof=" << coarse_vec_bpd << '\n'
                  << "matrix_free_transfer_plus_3_coarse_vectors_bytes_per_fine_dof="
                  << transfer_bpd + 3.0 * coarse_vec_bpd << '\n'
                  << "fp32_rank_aware_matrix_plus_transfer_plus_3_vectors_bytes_per_fine_dof="
                  << static_cast<double>(fp32_rank) / fine_dofs +
                         transfer_bpd + 3.0 * coarse_vec_bpd << '\n'
                  << "storage_decision_status=audit_only_no_coarse_matrix_materialized\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
