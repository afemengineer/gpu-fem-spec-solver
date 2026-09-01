#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/cpu_elasticity.hpp"
#include "gfss/gpu_m5_fine_level.hpp"
#include "gfss/hex8.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kChebyshevLowerFraction = 0.10;
constexpr double kFrozenLambda0 = 3.379863;
constexpr double kFrozenOmega0 = 0.394493;
constexpr double kCorrectionScale = 1.0e-9;

struct CpuFullShellResult {
    std::vector<double> coarse_residual;
    std::vector<double> fine_correction;
};

void clamp_x0(const gfss::StructuredHexMesh& mesh, std::vector<double>& v) {
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = static_cast<std::size_t>(mesh.node_index(0U, j, k));
            v[3U * node + 0U] = 0.0;
            v[3U * node + 1U] = 0.0;
            v[3U * node + 2U] = 0.0;
        }
    }
}

std::vector<double> apply_clamped(const gfss::StructuredHexMesh& mesh,
                                  const gfss::Material& material,
                                  const std::vector<double>& x) {
    auto free_x = x;
    clamp_x0(mesh, free_x);
    auto y = gfss::apply_matrix_free_openmp(mesh, material, free_x);
    clamp_x0(mesh, y);
    return y;
}

std::vector<double> build_inverse_diagonal(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space) {
    std::vector<double> diagonal(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                for (std::size_t a = 0; a < 8U; ++a) {
                    const auto node = static_cast<std::size_t>(nodes[a]);
                    if (space.graph.constrained[node] != 0U) continue;
                    for (std::size_t c = 0; c < 3U; ++c) {
                        const std::size_t local = 3U * a + c;
                        diagonal[3U * node + c] += ke[local][local];
                    }
                }
            }
        }
    }
    std::vector<double> inverse(diagonal.size(), 0.0);
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        if (space.graph.constrained[node] != 0U) continue;
        for (std::size_t c = 0; c < 3U; ++c) {
            const std::size_t dof = 3U * node + c;
            if (!(diagonal[dof] > 0.0) || !std::isfinite(diagonal[dof])) {
                throw std::runtime_error("M5 full-shell oracle diagonal invalid");
            }
            inverse[dof] = 1.0 / diagonal[dof];
        }
    }
    return inverse;
}

std::vector<double> make_rhs(const gfss::StructuredHexMesh& mesh) {
    std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const double count = static_cast<double>(mesh.ny + 1U) *
                         static_cast<double>(mesh.nz + 1U);
    const double magnitude = 1.0 / count;
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = -magnitude;
        }
    }
    return rhs;
}

std::vector<double> make_deterministic_coarse_correction(std::size_t n) {
    std::vector<double> correction(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        correction[i] = kCorrectionScale *
            (std::sin(0.013 * t + 0.17) + 0.31 * std::cos(0.037 * t - 0.23));
    }
    return correction;
}

std::vector<double> chebyshev_weights(double lambda_max, std::size_t degree) {
    std::vector<double> weights(degree, 0.0);
    const double lambda_low = kChebyshevLowerFraction * lambda_max;
    const double theta = 0.5 * (lambda_max + lambda_low);
    const double delta = 0.5 * (lambda_max - lambda_low);
    for (std::size_t k = 0; k < degree; ++k) {
        const double angle = kPi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        const double root = theta + delta * std::cos(angle);
        if (!(root > 0.0)) throw std::runtime_error("M5 full-shell oracle root invalid");
        weights[k] = 1.0 / root;
    }
    return weights;
}

void chebyshev_smooth(const gfss::StructuredHexMesh& mesh,
                      const gfss::Material& material,
                      const std::vector<double>& inverse_diagonal,
                      const std::vector<double>& rhs,
                      const std::vector<double>& weights,
                      std::vector<double>& x) {
    for (const double weight : weights) {
        const auto ax = apply_clamped(mesh, material, x);
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (inverse_diagonal[i] > 0.0) {
                x[i] += weight * inverse_diagonal[i] * (rhs[i] - ax[i]);
            }
        }
        clamp_x0(mesh, x);
    }
}

std::vector<double> smoothed_restrict(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& inverse_diagonal,
    std::vector<double> work,
    double omega,
    std::size_t steps) {
    std::vector<double> scaled(work.size(), 0.0);
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t i = 0; i < work.size(); ++i) {
            scaled[i] = inverse_diagonal[i] * work[i];
        }
        const auto a_scaled = apply_clamped(mesh, material, scaled);
        for (std::size_t i = 0; i < work.size(); ++i) {
            work[i] -= omega * a_scaled[i];
        }
        clamp_x0(mesh, work);
    }
    return gfss::apply_elasticity_tentative_restriction(space, work);
}

std::vector<double> smoothed_prolong(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& inverse_diagonal,
    const std::vector<double>& coarse,
    double omega,
    std::size_t steps) {
    auto fine = gfss::apply_elasticity_tentative_prolongation(space, coarse);
    clamp_x0(mesh, fine);
    for (std::size_t step = 0; step < steps; ++step) {
        const auto af = apply_clamped(mesh, material, fine);
        for (std::size_t i = 0; i < fine.size(); ++i) {
            if (inverse_diagonal[i] > 0.0) {
                fine[i] -= omega * inverse_diagonal[i] * af[i];
            }
        }
        clamp_x0(mesh, fine);
    }
    return fine;
}

CpuFullShellResult cpu_full_shell(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& inverse_diagonal,
    const std::vector<double>& rhs,
    const std::vector<double>& coarse_correction,
    double lambda_max,
    double transfer_omega,
    std::size_t degree,
    std::size_t transfer_steps) {
    const auto weights = chebyshev_weights(lambda_max, degree);
    std::vector<double> x(rhs.size(), 0.0);
    chebyshev_smooth(mesh, material, inverse_diagonal, rhs, weights, x);

    const auto ax = apply_clamped(mesh, material, x);
    std::vector<double> residual(rhs.size(), 0.0);
    for (std::size_t i = 0; i < residual.size(); ++i) residual[i] = rhs[i] - ax[i];
    clamp_x0(mesh, residual);

    CpuFullShellResult result;
    result.coarse_residual = smoothed_restrict(
        mesh, material, space, inverse_diagonal,
        std::move(residual), transfer_omega, transfer_steps);

    const auto fine_coarse_correction = smoothed_prolong(
        mesh, material, space, inverse_diagonal,
        coarse_correction, transfer_omega, transfer_steps);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += fine_coarse_correction[i];
    clamp_x0(mesh, x);

    chebyshev_smooth(mesh, material, inverse_diagonal, rhs, weights, x);
    result.fine_correction = std::move(x);
    return result;
}

double relative_error(const std::vector<float>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("M5 full-shell oracle size mismatch");
    }
    double diff2 = 0.0;
    double ref2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - reference[i];
        diff2 += d * d;
        ref2 += reference[i] * reference[i];
    }
    if (!(ref2 > 0.0)) throw std::runtime_error("M5 full-shell oracle norm zero");
    return std::sqrt(diff2 / ref2);
}

void print_timing(const char* prefix,
                  const gfss::GpuM5FineFullShellTiming& t,
                  std::size_t fine_operator_applies,
                  std::size_t fine_dofs) {
    const double gdof_s = t.total_ms > 0.0
        ? static_cast<double>(fine_operator_applies) * static_cast<double>(fine_dofs) /
              (t.total_ms * 1.0e6)
        : 0.0;
    std::cout << prefix << "_total_ms=" << t.total_ms
              << " " << prefix << "_zero_ms=" << t.zero_ms
              << " " << prefix << "_pre_smooth_ms=" << t.pre_smooth_ms
              << " " << prefix << "_residual_ms=" << t.residual_ms
              << " " << prefix << "_transfer_transpose_ms=" << t.transfer_transpose_ms
              << " " << prefix << "_P0T_ms=" << t.p0t_ms
              << " " << prefix << "_P0_ms=" << t.p0_ms
              << " " << prefix << "_transfer_forward_ms=" << t.transfer_forward_ms
              << " " << prefix << "_correction_ms=" << t.correction_ms
              << " " << prefix << "_post_smooth_ms=" << t.post_smooth_ms
              << " " << prefix << "_fine_equiv_GDOF_s=" << gdof_s << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t degree = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 5U;
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 50;
        const int block_y = argc > 3 ? std::stoi(argv[3]) : 4;
        const std::size_t target_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 12U;
        const std::size_t min_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 4U;
        const double lambda0 = argc > 6 ? std::stod(argv[6]) : kFrozenLambda0;
        const double omega0 = argc > 7 ? std::stod(argv[7]) : kFrozenOmega0;
        const std::size_t transfer_steps = argc > 8
            ? static_cast<std::size_t>(std::stoull(argv[8])) : 1U;
        if (degree == 0U || repeats <= 0 || block_y <= 0 || target_nodes == 0U ||
            min_nodes == 0U || !(lambda0 > 0.0) || !(omega0 > 0.0)) {
            throw std::invalid_argument("invalid M5 fine full-shell benchmark options");
        }

        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        auto graph = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph), {target_nodes, min_nodes, 1.0e-10});
        const auto inverse = build_inverse_diagonal(mesh, material, space);
        const auto rhs_double = make_rhs(mesh);
        const auto correction_double = make_deterministic_coarse_correction(space.coarse_dofs);

        std::vector<float> rhs_float(rhs_double.size(), 0.0f);
        for (std::size_t i = 0; i < rhs_double.size(); ++i) {
            rhs_float[i] = static_cast<float>(rhs_double[i]);
        }
        std::vector<float> correction_float(correction_double.size(), 0.0f);
        for (std::size_t i = 0; i < correction_double.size(); ++i) {
            correction_float[i] = static_cast<float>(correction_double[i]);
        }

        const auto cpu = cpu_full_shell(
            mesh, material, space, inverse, rhs_double, correction_double,
            lambda0, omega0, degree, transfer_steps);

        gfss::GpuM5FineLevelContext context(
            mesh, material, space, omega0, lambda0, block_y);
        const auto gpu = context.full_shell(
            rhs_float, correction_float, degree, transfer_steps, repeats);

        const double coarse_oracle_error = relative_error(
            gpu.coarse_residual, cpu.coarse_residual);
        const double fine_oracle_error = relative_error(
            gpu.fine_correction_aos, cpu.fine_correction);
        const bool oracle_accept = coarse_oracle_error <= 1.0e-4 &&
                                   fine_oracle_error <= 1.0e-4;
        const double free_dofs = static_cast<double>(space.fine_free_dofs);

        std::cout << "GFSS M5 persistent CUDA complete fine-level V-cycle shell\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "shell=degree_n_pre_then_P0T_then_external_L1_correction_then_P0_then_degree_n_post\n"
                  << "recursive_schedule_target=1x1\n"
                  << "production_policy_target=5x1x1\n"
                  << "fine_layout_device=SoA\n"
                  << "host_round_trips_inside_timed_shell=false\n"
                  << "external_L1_correction=deterministic_oracle_probe\n"
                  << "external_L1_correction_scale=" << std::scientific << kCorrectionScale << '\n'
                  << "transpose_transfer=exact_unfused_Dinv_then_A_then_update\n"
                  << "forward_transfer=exact_A_then_Dinv_update\n"
                  << "timing_excludes_context_setup_RHS_H2D_correction_H2D_all_D2H_CPU_oracle\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " fine_free_dofs=" << space.fine_free_dofs
                  << " coarse_dofs=" << space.coarse_dofs
                  << " aggregates=" << space.aggregates.size() << '\n'
                  << "smoother_degree=" << degree
                  << " transfer_smoothing_steps=" << transfer_steps
                  << " fine_operator_applies=" << gpu.fine_operator_applies
                  << " repeats=" << repeats << " block_y=" << block_y << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0 << " omega0=" << omega0 << '\n'
                  << std::scientific << std::setprecision(9)
                  << "gpu_vs_cpu_fp64_coarse_residual_relative_error=" << coarse_oracle_error << '\n'
                  << "gpu_vs_cpu_fp64_final_fine_correction_relative_error=" << fine_oracle_error << '\n'
                  << "oracle_accept_1e-4=" << (oracle_accept ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "device_bytes_total=" << gpu.device_bytes_total
                  << " device_bytes_per_fine_free_dof="
                  << static_cast<double>(gpu.device_bytes_total) / free_dofs << '\n'
                  << "fine_vector_bytes=" << gpu.fine_vector_bytes
                  << " coarse_vector_bytes=" << gpu.coarse_vector_bytes
                  << " aggregation_metadata_bytes=" << gpu.aggregation_metadata_bytes
                  << " model_coordinate_bytes=" << gpu.model_coordinate_bytes << '\n';

        print_timing("median", gpu.median_timing, gpu.fine_operator_applies, mesh.dof_count());
        print_timing("best", gpu.best_timing, gpu.fine_operator_applies, mesh.dof_count());

        if (gpu.median_timing.total_ms > 0.0) {
            const double total = gpu.median_timing.total_ms;
            std::cout << "median_fraction_zero=" << gpu.median_timing.zero_ms / total
                      << " median_fraction_pre_smooth=" << gpu.median_timing.pre_smooth_ms / total
                      << " median_fraction_residual=" << gpu.median_timing.residual_ms / total
                      << " median_fraction_transfer_transpose="
                      << gpu.median_timing.transfer_transpose_ms / total
                      << " median_fraction_P0T=" << gpu.median_timing.p0t_ms / total
                      << " median_fraction_P0=" << gpu.median_timing.p0_ms / total
                      << " median_fraction_transfer_forward="
                      << gpu.median_timing.transfer_forward_ms / total
                      << " median_fraction_correction=" << gpu.median_timing.correction_ms / total
                      << " median_fraction_post_smooth=" << gpu.median_timing.post_smooth_ms / total
                      << '\n';
        }

        const std::size_t expected = 2U * degree + 1U + 2U * transfer_steps;
        std::cout << "expected_fine_operator_applies_for_frozen_full_shell="
                  << (degree == 5U && transfer_steps == 1U ? 13U : expected) << '\n';
        return oracle_accept ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_fine_full_shell_bench "
                  << "[degree=5 [repeats=50 [block_y=4 [target_nodes=12 [min_nodes=4 "
                  << "[lambda0=3.379863 [omega0=0.394493 [transfer_steps=1]]]]]]]]\n";
        return 1;
    }
}
