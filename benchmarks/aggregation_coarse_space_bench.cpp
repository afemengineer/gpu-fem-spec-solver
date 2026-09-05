#include "gfss/aggregation_coarse_space.hpp"

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CaseDef {
    const char* name;
    const char* description;
    gfss::StructuredHexMesh mesh;
};

std::size_t parse_size(const char* text, const char* name) {
    const auto value = std::stoull(text);
    if (value == 0ULL) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

std::vector<CaseDef> make_cases() {
    return {
        {"baseline_cube",
         "64^3 cube graph; structured geometry used only as fine-mesh adapter",
         {64U, 64U, 64U, 1.0, 1.0, 1.0}},
        {"slender_beam",
         "4:1:1 beam graph with near-isotropic fine cells",
         {128U, 32U, 32U, 4.0, 1.0, 1.0}},
        {"thin_plate",
         "thin 1:1:0.125 graph with near-isotropic fine cells",
         {64U, 64U, 8U, 1.0, 1.0, 0.125}},
    };
}

void run_case(const CaseDef& test,
              std::size_t target_nodes,
              std::size_t min_nodes) {
    const auto graph = gfss::build_structured_hex_nodal_graph_x0(test.mesh);
    const gfss::ElasticityAggregationOptions options{
        target_nodes,
        min_nodes,
        1.0e-10};
    const auto space = gfss::build_elasticity_aggregation_coarse_space(
        graph, options);

    std::size_t min_size = std::numeric_limits<std::size_t>::max();
    std::size_t max_size = 0U;
    std::size_t min_rank = std::numeric_limits<std::size_t>::max();
    std::size_t max_rank = 0U;
    double size_sum = 0.0;
    double rank_sum = 0.0;
    for (const auto& aggregate : space.aggregates) {
        min_size = std::min(min_size, aggregate.node_count);
        max_size = std::max(max_size, aggregate.node_count);
        min_rank = std::min(min_rank, aggregate.rank);
        max_rank = std::max(max_rank, aggregate.rank);
        size_sum += static_cast<double>(aggregate.node_count);
        rank_sum += static_cast<double>(aggregate.rank);
    }

    const double aggregate_count = static_cast<double>(space.aggregates.size());
    const double avg_size = aggregate_count > 0.0 ? size_sum / aggregate_count : 0.0;
    const double avg_rank = aggregate_count > 0.0 ? rank_sum / aggregate_count : 0.0;
    const double coarsening_ratio = space.coarse_dofs > 0U
        ? static_cast<double>(space.fine_free_dofs) /
              static_cast<double>(space.coarse_dofs)
        : 0.0;
    const double matrix_free_payload_bpd = space.fine_free_dofs > 0U
        ? static_cast<double>(space.estimated_matrix_free_transfer_payload_bytes) /
              static_cast<double>(space.fine_free_dofs)
        : 0.0;
    const double explicit_p_value_bpd = space.fine_free_dofs > 0U
        ? static_cast<double>(space.tentative_p_nnz * sizeof(float)) /
              static_cast<double>(space.fine_free_dofs)
        : 0.0;
    const double rigid_error =
        gfss::audit_elasticity_rigid_body_reproduction(space);

    std::cout << "case=" << test.name << '\n'
              << "description=" << test.description << '\n'
              << "fine_mesh=" << test.mesh.nx << 'x' << test.mesh.ny << 'x'
              << test.mesh.nz
              << " fine_nodes=" << test.mesh.node_count()
              << " free_nodes=" << space.free_nodes
              << " fine_free_dofs=" << space.fine_free_dofs << '\n'
              << "graph_source=fine_nodal_stiffness_connectivity\n"
              << "cad_required=false\n"
              << "coarse_mesh_required=false\n"
              << "near_nullspace=3_translations_plus_3_rotations\n"
              << "target_nodes_per_aggregate=" << target_nodes
              << " min_nodes_per_aggregate=" << min_nodes << '\n'
              << "aggregates=" << space.aggregates.size()
              << " coarse_dofs=" << space.coarse_dofs
              << std::fixed << std::setprecision(6)
              << " fine_to_coarse_dof_ratio=" << coarsening_ratio << '\n'
              << "aggregate_nodes_min=" << min_size
              << " aggregate_nodes_avg=" << avg_size
              << " aggregate_nodes_max=" << max_size << '\n'
              << "aggregate_rank_min=" << min_rank
              << " aggregate_rank_avg=" << avg_rank
              << " aggregate_rank_max=" << max_rank << '\n'
              << "tentative_P_nnz=" << space.tentative_p_nnz << '\n'
              << "explicit_P_value_bytes_per_fine_free_dof="
              << explicit_p_value_bpd << '\n'
              << "matrix_free_transfer_payload_bytes_per_fine_free_dof="
              << matrix_free_payload_bpd << '\n'
              << std::scientific << std::setprecision(9)
              << "rigid_body_projection_max_relative_error=" << rigid_error << '\n'
              << "coarse_operator_status=not_built_yet\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "all";
        const std::size_t target_nodes = argc > 2
            ? parse_size(argv[2], "target aggregate nodes")
            : 12U;
        const std::size_t min_nodes = argc > 3
            ? parse_size(argv[3], "minimum aggregate nodes")
            : 4U;

        std::cout << "GFSS M5 mesh-independent elasticity aggregation prototype\n"
                  << "purpose=build_coarse_space_without_coarse_CAD_or_remeshing\n"
                  << "tentative_transfer=matrix_free_local_rigid_body_basis\n";

        std::size_t selected = 0U;
        for (const auto& test : make_cases()) {
            if (selector != "all" && selector != test.name) continue;
            ++selected;
            std::cout << "\n========================================\n";
            run_case(test, target_nodes, min_nodes);
        }
        if (selected == 0U) {
            throw std::invalid_argument("unknown aggregation benchmark case");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_aggregation_coarse_space_bench "
                  << "[all|baseline_cube|slender_beam|thin_plate "
                  << "[target_nodes [min_nodes]]]\n";
        return 1;
    }
}
