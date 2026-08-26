#include "gfss/smoothed_aggregation_reference.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class LoadKind { UniformZ, CheckerboardZ };

struct CaseDef {
    const char* name;
    const char* description;
    gfss::StructuredHexMesh mesh;
    gfss::Material material;
    LoadKind load;
};

std::size_t parse_size(const char* text, const char* name, bool allow_zero = false) {
    const auto value = std::stoull(text);
    if (!allow_zero && value == 0ULL) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

double parse_double(const char* text, const char* name) {
    const double value = std::stod(text);
    if (!(value > 0.0)) throw std::invalid_argument(std::string(name) + " must be positive");
    return value;
}

std::vector<CaseDef> make_cases() {
    constexpr double E = 210.0e9;
    return {
        {"thin_plate",
         "thin 1:1:0.125 cantilever; hard low-frequency bending case",
         {64U, 64U, 8U, 1.0, 1.0, 0.125}, {E, 0.30}, LoadKind::UniformZ},
        {"slender_beam",
         "4:1:1 cantilever; hard global bending case",
         {128U, 32U, 32U, 4.0, 1.0, 1.0}, {E, 0.30}, LoadKind::UniformZ},
        {"checkerboard_face",
         "high-spatial-frequency face-load control case",
         {48U, 48U, 48U, 1.0, 1.0, 1.0}, {E, 0.30}, LoadKind::CheckerboardZ},
    };
}

std::vector<double> make_rhs(const CaseDef& test) {
    const auto& mesh = test.mesh;
    std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const double count = static_cast<double>(mesh.ny + 1U) *
                         static_cast<double>(mesh.nz + 1U);
    const double magnitude = 1.0 / count;
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            const double sign = test.load == LoadKind::CheckerboardZ && ((j + k) & 1U)
                ? -1.0 : 1.0;
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = -sign * magnitude;
        }
    }
    return rhs;
}

void run_case(const CaseDef& test, const gfss::SmoothedAggregationOptions& options) {
    std::cout << "\n========================================\n"
              << "case=" << test.name << '\n'
              << "description=" << test.description << '\n'
              << "mesh=" << test.mesh.nx << 'x' << test.mesh.ny << 'x' << test.mesh.nz
              << " physical=" << test.mesh.lx << 'x' << test.mesh.ly << 'x' << test.mesh.lz
              << " dofs=" << test.mesh.dof_count() << '\n'
              << std::scientific << std::setprecision(6)
              << "material_E=" << test.material.young_modulus
              << " poisson=" << test.material.poisson_ratio << '\n'
              << "target_true_relative_residual=" << options.true_relative_tolerance << '\n'
              << std::fixed
              << "transfer_smoothing_steps=" << options.transfer_smoothing_steps
              << " aggregation_target_nodes=" << options.target_nodes_per_aggregate
              << " aggregation_min_nodes=" << options.min_nodes_per_aggregate << '\n'
              << "pre_smooth_degree=" << options.pre_smooth_degree
              << " post_smooth_degree=" << options.post_smooth_degree
              << " power_iterations=" << options.power_iterations << '\n'
              << std::scientific
              << "coarse_relative_tolerance=" << options.coarse_relative_tolerance
              << std::fixed
              << " coarse_max_iterations=" << options.coarse_max_iterations << '\n';

    const auto result = gfss::solve_smoothed_aggregation_two_grid_reference(
        test.mesh, test.material, make_rhs(test), options);

    std::cout << "reference_execution=cpu_fp64_factorized_smoothed_aggregation\n"
              << "production_status=numerical_hypothesis_only\n"
              << "prolongation=(I-omega_DinvA)^m_P0\n"
              << "restriction=P0T_(I-omega_A_Dinv)^m\n"
              << "coarse_operator=P^T_A_f_P_matrix_free\n"
              << "coarse_preconditioner=tentative_Ac_diagonal_reference\n"
              << "coarse_mesh_required=false\n"
              << "coarse_dofs=" << result.coarse_dofs
              << " aggregates=" << result.aggregates
              << std::fixed << std::setprecision(6)
              << " fine_to_coarse_dof_ratio="
              << static_cast<double>(result.fine_free_dofs) /
                     static_cast<double>(result.coarse_dofs) << '\n'
              << "fine_lambda_max_est=" << result.fine_lambda_max
              << " transfer_omega=" << result.transfer_omega << '\n'
              << "fine_operator_applies_per_coarse_apply="
              << result.fine_operator_applies_per_coarse_apply << '\n'
              << std::scientific << std::setprecision(9)
              << "transfer_adjoint_relative_error="
              << result.transfer_adjoint_relative_error << '\n'
              << "coarse_symmetry_relative_defect="
              << result.coarse_symmetry_relative_defect << '\n'
              << "coarse_spd_probe=" << (result.coarse_spd_probe ? "true" : "false") << '\n'
              << std::fixed << std::setprecision(6)
              << "matrix_free_tentative_transfer_bytes_per_fine_free_dof="
              << result.matrix_free_transfer_bytes_per_fine_free_dof << '\n'
              << "fp32_factorized_transfer_extra_bytes_per_fine_free_dof="
              << result.fp32_factorized_transfer_extra_bytes_per_fine_free_dof << '\n'
              << "aggregation_setup_ms=" << result.aggregation_setup_ms
              << " tentative_coarse_preconditioner_setup_ms="
              << result.tentative_coarse_preconditioner_setup_ms
              << " spectral_setup_ms=" << result.spectral_setup_ms
              << " solve_ms=" << result.solve_ms
              << " total_ms=" << result.total_ms << '\n'
              << "converged=" << (result.converged ? "true" : "false")
              << " cycles=" << result.cycles << '\n';

    for (std::size_t i = 0; i < result.true_relative_residuals.size(); ++i) {
        std::cout << std::scientific << std::setprecision(9)
                  << "true_residual[" << i << "]=" << result.true_relative_residuals[i];
        if (i > 0U) {
            std::cout << " cycle_q=" << result.cycle_contractions[i - 1U]
                      << std::fixed
                      << " coarse_iterations=" << result.coarse_iterations[i - 1U]
                      << std::scientific
                      << " coarse_rel=" << result.coarse_final_relative_residuals[i - 1U];
        }
        std::cout << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "thin_plate";
        gfss::SmoothedAggregationOptions options;
        if (argc > 2) options.transfer_smoothing_steps = parse_size(argv[2], "transfer smoothing steps", true);
        if (argc > 3) options.true_relative_tolerance = parse_double(argv[3], "true tolerance");
        if (argc > 4) options.max_cycles = parse_size(argv[4], "max cycles");
        if (argc > 5) options.pre_smooth_degree = parse_size(argv[5], "pre degree", true);
        if (argc > 6) options.post_smooth_degree = parse_size(argv[6], "post degree", true);
        if (argc > 7) options.target_nodes_per_aggregate = parse_size(argv[7], "target aggregate nodes");
        if (argc > 8) options.min_nodes_per_aggregate = parse_size(argv[8], "minimum aggregate nodes");
        if (argc > 9) options.coarse_relative_tolerance = parse_double(argv[9], "coarse tolerance");
        if (argc > 10) options.coarse_max_iterations = parse_size(argv[10], "coarse max iterations");

        std::cout << "GFSS M5 factorized smoothed-aggregation numerical reference\n"
                  << "purpose=test_smoothed_interpolation_before_GPU_optimization\n"
                  << "damping=4_over_3_divided_by_safety_lambda_max\n"
                  << "explicit_smoothed_P=false\n"
                  << "exact_transpose_restriction=true\n";

        std::size_t selected = 0U;
        for (const auto& test : make_cases()) {
            if (selector != "all" && selector != test.name) continue;
            ++selected;
            run_case(test, options);
        }
        if (selected == 0U) {
            throw std::invalid_argument("unknown case; use thin_plate, slender_beam, checkerboard_face, or all");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_smoothed_aggregation_reference_bench "
                  << "[case [transfer_steps [true_tol [max_cycles [pre [post [target [min [coarse_tol [coarse_max]]]]]]]]]]\n";
        return 1;
    }
}
