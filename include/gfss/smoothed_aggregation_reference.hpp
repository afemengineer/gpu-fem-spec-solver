#pragma once

#include "gfss/structured_hex_mesh.hpp"

#include <cstddef>
#include <vector>

namespace gfss {

struct SmoothedAggregationOptions {
    std::size_t target_nodes_per_aggregate{12};
    std::size_t min_nodes_per_aggregate{4};
    double rank_tolerance{1.0e-10};

    // P = (I - omega D^{-1} A)^m P0. Restriction applies the exact transpose
    // P^T = P0^T (I - omega A D^{-1})^m.
    std::size_t transfer_smoothing_steps{1};
    std::size_t power_iterations{8};

    std::size_t pre_smooth_degree{3};
    std::size_t post_smooth_degree{3};

    // The smoothed coarse operator remains matrix-free in this numerical
    // reference. A moderate coarse tolerance is sufficient to diagnose
    // two-grid contraction without paying for an unnecessarily exact solve.
    double coarse_relative_tolerance{1.0e-6};
    std::size_t coarse_max_iterations{2500};

    double true_relative_tolerance{1.0e-6};
    std::size_t max_cycles{4};
};

struct SmoothedAggregationResult {
    bool converged{false};
    std::size_t cycles{0};
    std::size_t fine_dofs{0};
    std::size_t fine_free_dofs{0};
    std::size_t coarse_dofs{0};
    std::size_t aggregates{0};

    std::size_t transfer_smoothing_steps{0};
    double fine_lambda_max{0.0};
    double transfer_omega{0.0};
    double transfer_adjoint_relative_error{0.0};
    double coarse_symmetry_relative_defect{0.0};
    bool coarse_spd_probe{false};
    std::size_t fine_operator_applies_per_coarse_apply{0};

    std::vector<double> true_relative_residuals;
    std::vector<double> cycle_contractions;
    std::vector<std::size_t> coarse_iterations;
    std::vector<double> coarse_final_relative_residuals;

    double aggregation_setup_ms{0.0};
    double tentative_coarse_preconditioner_setup_ms{0.0};
    double spectral_setup_ms{0.0};
    double solve_ms{0.0};
    double total_ms{0.0};

    double matrix_free_transfer_bytes_per_fine_free_dof{0.0};
    double fp32_factorized_transfer_extra_bytes_per_fine_free_dof{0.0};
};

SmoothedAggregationResult solve_smoothed_aggregation_two_grid_reference(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<double>& rhs,
    const SmoothedAggregationOptions& options = {});

}  // namespace gfss
