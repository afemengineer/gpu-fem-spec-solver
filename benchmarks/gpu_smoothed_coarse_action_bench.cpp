#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_gold.hpp"
#include "gfss/gpu_smoothed_aggregation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kThinPlateReferenceOmega = 0.394493;

int axis_class(std::uint32_t coordinate, std::uint32_t maximum) {
    return coordinate == 0U ? 0 : (coordinate == maximum ? 2 : 1);
}

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

std::vector<double> apply_clamped_openmp(const gfss::StructuredHexMesh& mesh,
                                          const gfss::Material& material,
                                          const std::vector<double>& x) {
    auto free_x = x;
    clamp_x0(mesh, free_x);
    auto y = gfss::apply_matrix_free_openmp(mesh, material, free_x);
    clamp_x0(mesh, y);
    return y;
}

std::vector<double> build_inverse_diagonal(const gfss::StructuredHexMesh& mesh,
                                           const gfss::Material& material) {
    const auto stencil = gfss::build_cpu_gold_stencil_fp32(mesh, material);
    std::vector<double> inverse(static_cast<std::size_t>(mesh.dof_count()), 0.0);

    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        const int cz = axis_class(k, mesh.nz);
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const int cy = axis_class(j, mesh.ny);
            for (std::uint32_t i = 1U; i <= mesh.nx; ++i) {
                const int cx = axis_class(i, mesh.nx);
                const std::size_t cls = static_cast<std::size_t>(cx + 3 * (cy + 3 * cz));
                const std::size_t count = stencil.regular.counts[cls];
                const auto node = static_cast<std::size_t>(mesh.node_index(i, j, k));
                bool found = false;
                for (std::size_t e = 0; e < count; ++e) {
                    const auto& entry = stencil.regular.entries[cls][e];
                    if (entry.dx == 0 && entry.dy == 0 && entry.dz == 0) {
                        if (!(entry.block[0] > 0.0f) ||
                            !(entry.block[4] > 0.0f) ||
                            !(entry.block[8] > 0.0f)) {
                            throw std::runtime_error("GPU SA benchmark found invalid fine diagonal");
                        }
                        inverse[3U * node + 0U] = 1.0 / static_cast<double>(entry.block[0]);
                        inverse[3U * node + 1U] = 1.0 / static_cast<double>(entry.block[4]);
                        inverse[3U * node + 2U] = 1.0 / static_cast<double>(entry.block[8]);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw std::runtime_error("GPU SA benchmark could not locate fine diagonal");
                }
            }
        }
    }
    return inverse;
}

std::vector<double> apply_cpu_factorized_coarse(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& inverse_diagonal,
    double omega,
    std::size_t m,
    const std::vector<double>& coarse) {
    auto u = gfss::apply_elasticity_tentative_prolongation(space, coarse);
    for (std::size_t step = 0; step < m; ++step) {
        const auto au = apply_clamped_openmp(mesh, material, u);
        for (std::size_t i = 0; i < u.size(); ++i) {
            u[i] -= omega * inverse_diagonal[i] * au[i];
        }
        clamp_x0(mesh, u);
    }

    auto y = apply_clamped_openmp(mesh, material, u);
    std::vector<double> scaled(y.size(), 0.0);
    for (std::size_t step = 0; step < m; ++step) {
        for (std::size_t i = 0; i < y.size(); ++i) {
            scaled[i] = inverse_diagonal[i] * y[i];
        }
        const auto z = apply_clamped_openmp(mesh, material, scaled);
        for (std::size_t i = 0; i < y.size(); ++i) {
            y[i] -= omega * z[i];
        }
        clamp_x0(mesh, y);
    }
    return gfss::apply_elasticity_tentative_restriction(space, y);
}

double relative_error(const std::vector<float>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("GPU SA benchmark oracle size mismatch");
    }
    double diff2 = 0.0;
    double ref2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - reference[i];
        diff2 += d * d;
        ref2 += reference[i] * reference[i];
    }
    if (!(ref2 > 0.0)) throw std::runtime_error("GPU SA benchmark oracle norm is zero");
    return std::sqrt(diff2 / ref2);
}

std::vector<double> make_probe_double(std::size_t n) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = std::sin(0.017 * t) + 0.29 * std::cos(0.043 * t + 0.31);
    }
    return v;
}

void print_timing(const char* prefix,
                  const gfss::GpuSmoothedAggregationTiming& t,
                  std::size_t smoothing_steps,
                  std::size_t fine_operator_applies,
                  std::size_t fine_dofs) {
    const double forward_per_step = smoothing_steps > 0U
        ? t.jacobi_ms / static_cast<double>(smoothing_steps)
        : 0.0;
    const double transpose_per_step = smoothing_steps > 0U
        ? t.vector_update_ms / static_cast<double>(smoothing_steps)
        : 0.0;
    const double fine_equiv_gdof_s = t.total_ms > 0.0
        ? static_cast<double>(fine_operator_applies) * static_cast<double>(fine_dofs) /
              (t.total_ms * 1.0e6)
        : 0.0;

    std::cout << prefix
              << "_total_ms=" << t.total_ms
              << " " << prefix << "_P0_ms=" << t.p0_ms
              << " " << prefix << "_center_A_ms=" << t.fine_operator_ms
              << " " << prefix << "_forward_fused_total_ms=" << t.jacobi_ms
              << " " << prefix << "_forward_fused_per_step_ms=" << forward_per_step
              << " " << prefix << "_transpose_fused_total_ms=" << t.vector_update_ms
              << " " << prefix << "_transpose_fused_per_step_ms=" << transpose_per_step
              << " " << prefix << "_P0T_ms=" << t.p0t_ms
              << " " << prefix << "_fine_equiv_GDOF_s=" << fine_equiv_gdof_s
              << '\n';
}

void run_one(gfss::GpuSmoothedAggregationContext& context,
             const gfss::StructuredHexMesh& mesh,
             const gfss::Material& material,
             const gfss::ElasticityAggregationCoarseSpace& space,
             const std::vector<double>& inverse_diagonal,
             const std::vector<double>& coarse_double,
             std::size_t m,
             int repeats) {
    std::vector<float> coarse_float(coarse_double.size(), 0.0f);
    for (std::size_t i = 0; i < coarse_double.size(); ++i) {
        coarse_float[i] = static_cast<float>(coarse_double[i]);
    }

    const auto cpu_oracle = apply_cpu_factorized_coarse(
        mesh, material, space, inverse_diagonal,
        context.omega(), m, coarse_double);
    const auto gpu = context.apply(coarse_float, m, repeats);
    const double oracle_rel = relative_error(gpu.coarse_y, cpu_oracle);
    const double fine_free = static_cast<double>(space.fine_free_dofs);

    std::cout << "\n----------------------------------------\n"
              << "transfer_smoothing_steps=" << m << '\n'
              << "fine_operator_applies_per_coarse_apply=" << gpu.fine_operator_applies << '\n'
              << std::scientific << std::setprecision(9)
              << "gpu_vs_cpu_fp64_coarse_action_relative_error=" << oracle_rel << '\n'
              << std::fixed << std::setprecision(6)
              << "device_bytes_total=" << gpu.device_bytes_total
              << " device_bytes_per_fine_free_dof="
              << static_cast<double>(gpu.device_bytes_total) / fine_free << '\n'
              << "fine_workspace_bytes_per_fine_free_dof="
              << static_cast<double>(gpu.fine_workspace_bytes) / fine_free
              << " coarse_workspace_bytes_per_fine_free_dof="
              << static_cast<double>(gpu.coarse_workspace_bytes) / fine_free << '\n'
              << "aggregation_metadata_bytes_per_fine_free_dof="
              << static_cast<double>(gpu.aggregation_metadata_bytes) / fine_free
              << " model_coordinate_bytes_per_fine_free_dof="
              << static_cast<double>(gpu.model_coordinate_bytes) / fine_free << '\n';

    print_timing("median", gpu.median_timing, m,
                 gpu.fine_operator_applies, mesh.dof_count());
    print_timing("best", gpu.best_timing, m,
                 gpu.fine_operator_applies, mesh.dof_count());

    const double total = gpu.median_timing.total_ms;
    if (total > 0.0) {
        std::cout << "median_fraction_P0=" << gpu.median_timing.p0_ms / total
                  << " median_fraction_center_A=" << gpu.median_timing.fine_operator_ms / total
                  << " median_fraction_forward_fused=" << gpu.median_timing.jacobi_ms / total
                  << " median_fraction_transpose_fused=" << gpu.median_timing.vector_update_ms / total
                  << " median_fraction_P0T=" << gpu.median_timing.p0t_ms / total
                  << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "all";
        const int repeats = argc > 2 ? std::stoi(argv[2]) : 50;
        const int block_y = argc > 3 ? std::stoi(argv[3]) : 4;
        const std::size_t target_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 12U;
        const std::size_t min_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 4U;
        const double omega = argc > 6 ? std::stod(argv[6]) : kThinPlateReferenceOmega;
        if (repeats <= 0 || block_y <= 0 || target_nodes == 0U || min_nodes == 0U ||
            !(omega > 0.0)) {
            throw std::invalid_argument("invalid GPU smoothed coarse-action benchmark options");
        }

        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        auto graph = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph), {target_nodes, min_nodes, 1.0e-10});
        const auto inverse_diagonal = build_inverse_diagonal(mesh, material);
        const auto coarse = make_probe_double(space.coarse_dofs);

        std::cout << "GFSS M5 persistent CUDA factorized smoothed coarse action\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "coarse_action=P_m^T_A_f_P_m\n"
                  << "P_m=(I-omega_DinvA)^m_P0\n"
                  << "smoothed_P_materialized=false\n"
                  << "coarse_matrix_materialized=false\n"
                  << "transfer_layout=aggregate_CSR_one_warp_per_aggregate\n"
                  << "restriction_global_atomics=false\n"
                  << "restriction_coarse_memset=false\n"
                  << "smoothing_kernels=fused_forward_and_exact_transpose\n"
                  << "forward_kernel=I_minus_omega_DinvA\n"
                  << "transpose_kernel=I_minus_omega_A_Dinv\n"
                  << "fine_workspace_vectors=2\n"
                  << "fine_operator=existing_GoldSparse_FP32_center_action\n"
                  << "timing_excludes_context_setup_H2D_D2H_and_CPU_oracle\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " fine_free_dofs=" << space.fine_free_dofs
                  << " coarse_dofs=" << space.coarse_dofs
                  << " aggregates=" << space.aggregates.size() << '\n'
                  << std::fixed << std::setprecision(6)
                  << "omega=" << omega
                  << " repeats=" << repeats
                  << " block_y=" << block_y
                  << " target_nodes=" << target_nodes
                  << " min_nodes=" << min_nodes << '\n';

        gfss::GpuSmoothedAggregationContext context(
            mesh, material, space, omega, block_y);

        if (selector == "all") {
            run_one(context, mesh, material, space, inverse_diagonal, coarse, 0U, repeats);
            run_one(context, mesh, material, space, inverse_diagonal, coarse, 1U, repeats);
            run_one(context, mesh, material, space, inverse_diagonal, coarse, 2U, repeats);
        } else {
            const auto m = static_cast<std::size_t>(std::stoull(selector));
            run_one(context, mesh, material, space, inverse_diagonal, coarse, m, repeats);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_smoothed_coarse_action_bench "
                  << "[m|all [repeats [block_y [target_nodes [min_nodes [omega]]]]]]\n";
        return 1;
    }
}
