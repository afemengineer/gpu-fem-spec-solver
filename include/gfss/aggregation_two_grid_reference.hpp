#pragma once

#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfss {

// Reference-only explicit Galerkin coarse matrix. Rows are aggregates and each
// block has the actual row_rank x col_rank size, so rank-deficient aggregates
// do not create artificial zero coarse DOFs.
struct AggregationVariableBlockMatrix {
    std::size_t coarse_dofs{0};
    std::vector<std::size_t> aggregate_offsets;
    std::vector<std::size_t> aggregate_ranks;
    std::vector<std::size_t> block_row_offsets;
    std::vector<std::uint32_t> block_columns;
    std::vector<std::size_t> block_value_offsets;
    std::vector<double> values;
    std::vector<double> inverse_diagonal;
    std::size_t storage_bytes{0};
};

struct AggregationTwoGridOptions {
    std::size_t target_nodes_per_aggregate{12};
    std::size_t min_nodes_per_aggregate{4};
    double rank_tolerance{1.0e-10};

    std::size_t pre_smooth_degree{3};
    std::size_t post_smooth_degree{3};
    std::size_t power_iterations{8};

    double coarse_relative_tolerance{1.0e-10};
    std::size_t coarse_max_iterations{5000};

    double true_relative_tolerance{1.0e-6};
    std::size_t max_cycles{12};
};

struct AggregationTwoGridResult {
    bool converged{false};
    std::size_t cycles{0};
    std::size_t fine_dofs{0};
    std::size_t fine_free_dofs{0};
    std::size_t coarse_dofs{0};
    std::size_t aggregates{0};

    double fine_lambda_max{0.0};
    double coarse_operator_oracle_relative_error{0.0};
    double coarse_symmetry_relative_defect{0.0};
    bool coarse_spd_probe{false};

    std::vector<double> true_relative_residuals;
    std::vector<double> cycle_contractions;
    std::vector<std::size_t> coarse_iterations;
    std::vector<double> coarse_final_relative_residuals;

    double aggregation_setup_ms{0.0};
    double coarse_assembly_ms{0.0};
    double smoother_setup_ms{0.0};
    double solve_ms{0.0};
    double total_ms{0.0};

    std::size_t explicit_coarse_matrix_bytes{0};
    double explicit_coarse_matrix_bytes_per_fine_free_dof{0.0};
    double matrix_free_transfer_bytes_per_fine_free_dof{0.0};
};

AggregationVariableBlockMatrix assemble_structured_hex_aggregation_galerkin(
    const StructuredHexMesh& mesh,
    const Material& material,
    const ElasticityAggregationCoarseSpace& space);

std::vector<double> apply_aggregation_variable_block_matrix(
    const AggregationVariableBlockMatrix& matrix,
    const std::vector<double>& x);

AggregationTwoGridResult solve_aggregation_two_grid_reference(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    const AggregationTwoGridOptions& options = {});

}  // namespace gfss
