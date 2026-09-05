// M5 GPU productionization stage 2.
//
// This standalone CUDA TU deliberately includes the persistent smoothed-
// aggregation implementation so the experimental L1 block-Chebyshev method can
// access its device-resident coarse vectors and factorized A1 pipeline without
// exposing internal device pointers in the production API. Do not link
// gfss_cuda_operator into this target: that would duplicate the included CUDA
// symbols.
#include "../src/gpu_smoothed_aggregation.cu"
#include "m5_l1_block_setup.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

__global__ void m5_l1_block_chebyshev_update_kernel(
    std::uint32_t aggregate_count,
    const DeviceAggregateSa* __restrict__ aggregates,
    const float* __restrict__ inverse_blocks,
    float weight,
    const float* __restrict__ rhs,
    const float* __restrict__ ax,
    float* __restrict__ x) {
    const std::uint32_t aggregate_id = blockIdx.x;
    if (aggregate_id >= aggregate_count) return;
    const DeviceAggregateSa* aggregate = aggregates + aggregate_id;
    const std::uint32_t q = threadIdx.x;
    if (q >= aggregate->rank) return;

    const std::uint32_t offset = aggregate->coarse_offset;
    const float* inverse = inverse_blocks + 36U * aggregate_id;
    float value = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0U; j < 6U; ++j) {
        if (j < aggregate->rank) {
            value = fmaf(inverse[6U * q + j],
                         rhs[offset + j] - ax[offset + j], value);
        }
    }
    x[offset + q] = fmaf(weight, value, x[offset + q]);
}

void launch_m5_l1_block_update(
    std::uint32_t aggregate_count,
    const DeviceAggregateSa* aggregates,
    const float* inverse_blocks,
    float weight,
    const float* rhs,
    const float* ax,
    float* x) {
    constexpr unsigned int threads = 32U;
    m5_l1_block_chebyshev_update_kernel<<<aggregate_count, threads>>>(
        aggregate_count, aggregates, inverse_blocks, weight, rhs, ax, x);
    check_cuda_pcg(cudaGetLastError(), "M5 L1 block-Chebyshev update launch");
}

double m5_l1_median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

float m5_l1_elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    check_cuda_pcg(cudaEventElapsedTime(&ms, a, b),
                   "cudaEventElapsedTime(M5 L1)");
    return ms;
}

}  // namespace

GpuSmoothedAggregationL1BlockStepResult
GpuSmoothedAggregationContext::l1_block_chebyshev_step(
    const std::vector<float>& rhs,
    const std::vector<float>& initial_x,
    const std::vector<float>& inverse_blocks_6x6,
    double lambda_max,
    std::size_t transfer_smoothing_steps,
    int repeats) {
    if (!impl_) throw std::runtime_error("M5 L1 context is empty");
    if (rhs.size() != impl_->coarse_dof_count ||
        initial_x.size() != impl_->coarse_dof_count) {
        throw std::invalid_argument("M5 L1 block smoother vector size mismatch");
    }
    if (inverse_blocks_6x6.size() !=
        static_cast<std::size_t>(impl_->aggregate_count) * 36U) {
        throw std::invalid_argument("M5 L1 inverse block payload size mismatch");
    }
    if (!(lambda_max > 0.0) || !std::isfinite(lambda_max)) {
        throw std::invalid_argument("M5 L1 lambda_max must be positive finite");
    }
    if (transfer_smoothing_steps > 8U) {
        throw std::invalid_argument("M5 L1 transfer smoothing limited to 8");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("M5 L1 repeats must be positive");
    }

    for (float value : inverse_blocks_6x6) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("M5 L1 inverse block contains non-finite value");
        }
    }

    // For degree 1, the single Chebyshev root is theta because cos(pi/2)=0.
    // lambda_low=0.1*lambda_max, therefore theta=0.55*lambda_max.
    const float weight = static_cast<float>(1.0 / (0.55 * lambda_max));
    const std::size_t coarse_bytes = impl_->coarse_dof_count * sizeof(float);
    const std::size_t inverse_bytes = inverse_blocks_6x6.size() * sizeof(float);

    float* d_rhs = nullptr;
    float* d_initial = nullptr;
    float* d_inverse = nullptr;
    cudaEvent_t e0{};
    cudaEvent_t e1{};
    cudaEvent_t e2{};

    try {
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_rhs), coarse_bytes),
                       "cudaMalloc(M5 L1 rhs)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_initial), coarse_bytes),
                       "cudaMalloc(M5 L1 initial x)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_inverse), inverse_bytes),
                       "cudaMalloc(M5 L1 inverse blocks)");
        check_cuda_pcg(cudaMemcpy(d_rhs, rhs.data(), coarse_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 rhs H2D)");
        check_cuda_pcg(cudaMemcpy(d_initial, initial_x.data(), coarse_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 initial x H2D)");
        check_cuda_pcg(cudaMemcpy(d_inverse, inverse_blocks_6x6.data(), inverse_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 inverse blocks H2D)");

        // Warm the exact same device-resident path before collecting events.
        check_cuda_pcg(cudaMemcpyAsync(impl_->d_coarse_x, d_initial, coarse_bytes,
                                       cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(M5 L1 warm x)");
        impl_->run_pipeline(transfer_smoothing_steps, nullptr, nullptr);
        launch_m5_l1_block_update(impl_->aggregate_count, impl_->d_aggregates,
                                   d_inverse, weight, d_rhs, impl_->d_coarse_y,
                                   impl_->d_coarse_x);
        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(M5 L1 warmup)");

        check_cuda_pcg(cudaEventCreate(&e0), "cudaEventCreate(M5 L1 start)");
        check_cuda_pcg(cudaEventCreate(&e1), "cudaEventCreate(M5 L1 A1)");
        check_cuda_pcg(cudaEventCreate(&e2), "cudaEventCreate(M5 L1 update)");

        std::vector<double> a1_samples;
        std::vector<double> update_samples;
        std::vector<double> total_samples;
        a1_samples.reserve(static_cast<std::size_t>(repeats));
        update_samples.reserve(static_cast<std::size_t>(repeats));
        total_samples.reserve(static_cast<std::size_t>(repeats));

        for (int repeat = 0; repeat < repeats; ++repeat) {
            // Reset is deliberately excluded from the interval: the production
            // V-cycle will already own the current L1 iterate on device.
            check_cuda_pcg(cudaMemcpyAsync(impl_->d_coarse_x, d_initial, coarse_bytes,
                                           cudaMemcpyDeviceToDevice),
                           "cudaMemcpyAsync(M5 L1 reset x)");
            check_cuda_pcg(cudaEventRecord(e0), "cudaEventRecord(M5 L1 start)");
            impl_->run_pipeline(transfer_smoothing_steps, nullptr, nullptr);
            check_cuda_pcg(cudaEventRecord(e1), "cudaEventRecord(M5 L1 A1)");
            launch_m5_l1_block_update(impl_->aggregate_count, impl_->d_aggregates,
                                       d_inverse, weight, d_rhs, impl_->d_coarse_y,
                                       impl_->d_coarse_x);
            check_cuda_pcg(cudaEventRecord(e2), "cudaEventRecord(M5 L1 update)");
            check_cuda_pcg(cudaEventSynchronize(e2), "cudaEventSynchronize(M5 L1)");

            a1_samples.push_back(static_cast<double>(m5_l1_elapsed(e0, e1)));
            update_samples.push_back(static_cast<double>(m5_l1_elapsed(e1, e2)));
            total_samples.push_back(static_cast<double>(m5_l1_elapsed(e0, e2)));
        }

        GpuSmoothedAggregationL1BlockStepResult result;
        result.x.resize(impl_->coarse_dof_count, 0.0f);
        check_cuda_pcg(cudaMemcpy(result.x.data(), impl_->d_coarse_x, coarse_bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 L1 result D2H)");
        result.median_a1_ms = m5_l1_median(a1_samples);
        result.median_block_update_ms = m5_l1_median(update_samples);
        result.median_total_ms = m5_l1_median(total_samples);
        result.best_a1_ms = *std::min_element(a1_samples.begin(), a1_samples.end());
        result.best_block_update_ms =
            *std::min_element(update_samples.begin(), update_samples.end());
        result.best_total_ms = *std::min_element(total_samples.begin(), total_samples.end());
        result.fine_operator_applies = 2U * transfer_smoothing_steps + 1U;
        result.persistent_l1_bytes = 2U * coarse_bytes + inverse_bytes;

        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        cudaEventDestroy(e2);
        cudaFree(d_rhs);
        cudaFree(d_initial);
        cudaFree(d_inverse);
        return result;
    } catch (...) {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (e2) cudaEventDestroy(e2);
        if (d_rhs) cudaFree(d_rhs);
        if (d_initial) cudaFree(d_initial);
        if (d_inverse) cudaFree(d_inverse);
        throw;
    }
}

}  // namespace gfss

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kFrozenOmega0 = 0.394493;
constexpr double kFrozenLambda1 = 1.554198;

std::vector<double> deterministic_probe(std::size_t n, double scale, double phase) {
    std::vector<double> v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = scale * (std::sin(0.017 * t + phase) +
                        0.31 * std::cos(0.043 * t - 0.7 * phase));
    }
    return v;
}

double relative_error(const std::vector<float>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("M5 L1 oracle size mismatch");
    }
    double diff2 = 0.0;
    double ref2 = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - reference[i];
        diff2 += d * d;
        ref2 += reference[i] * reference[i];
    }
    if (!(ref2 > 0.0)) throw std::runtime_error("M5 L1 oracle norm is zero");
    return std::sqrt(diff2 / ref2);
}

std::vector<float> to_float(const std::vector<double>& v) {
    std::vector<float> out(v.size(), 0.0f);
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = static_cast<float>(v[i]);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 50;
        const int block_y = argc > 2 ? std::stoi(argv[2]) : 4;
        const std::size_t target_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t min_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 4U;
        const double omega0 = argc > 5 ? std::stod(argv[5]) : kFrozenOmega0;
        const double lambda1 = argc > 6 ? std::stod(argv[6]) : kFrozenLambda1;
        const std::size_t transfer_steps = argc > 7
            ? static_cast<std::size_t>(std::stoull(argv[7])) : 1U;
        if (repeats <= 0 || block_y <= 0 || target_nodes == 0U || min_nodes == 0U ||
            !(omega0 > 0.0) || !(lambda1 > 0.0) || transfer_steps > 8U) {
            throw std::invalid_argument("invalid M5 L1 block-smoother benchmark options");
        }

        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        auto graph = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph), {target_nodes, min_nodes, 1.0e-10});

        const auto setup_start = Clock::now();
        const auto fine_inverse =
            m5_l1_setup::build_fine_inverse_diagonal(mesh, material, space);
        const auto blocks = m5_l1_setup::build_exact_l1_blocks(
            mesh, material, space, fine_inverse, omega0);
        const auto setup_stop = Clock::now();
        const double block_setup_ms =
            std::chrono::duration<double, std::milli>(setup_stop - setup_start).count();

        const auto x0 = deterministic_probe(space.coarse_dofs, 1.0e-9, 0.17);
        const auto z = deterministic_probe(space.coarse_dofs, 4.0e-10, 0.83);
        const auto cpu_a1 = m5_l1_setup::apply_factorized_a1(
            mesh, material, space, fine_inverse, omega0, transfer_steps, x0);
        const auto bz = m5_l1_setup::apply_block_matrix(space, blocks, z);

        std::vector<double> rhs(space.coarse_dofs, 0.0);
        for (std::size_t i = 0; i < rhs.size(); ++i) rhs[i] = cpu_a1[i] + bz[i];

        const double weight = 1.0 / (0.55 * lambda1);
        std::vector<double> expected(space.coarse_dofs, 0.0);
        for (std::size_t i = 0; i < expected.size(); ++i) {
            expected[i] = x0[i] + weight * z[i];
        }

        const auto x0f = to_float(x0);
        const auto rhsf = to_float(rhs);
        gfss::GpuSmoothedAggregationContext context(
            mesh, material, space, omega0, block_y);

        const auto gpu_a1 = context.apply(x0f, transfer_steps, repeats);
        const double a1_oracle_error = relative_error(gpu_a1.coarse_y, cpu_a1);

        const auto gpu_step = context.l1_block_chebyshev_step(
            rhsf, x0f, blocks.inverse_blocks_6x6_fp32,
            lambda1, transfer_steps, repeats);
        const double step_oracle_error = relative_error(gpu_step.x, expected);
        const bool accepted = a1_oracle_error <= 1.0e-4 && step_oracle_error <= 1.0e-4;

        std::cout << "GFSS M5 persistent CUDA L1 actual-block smoother stage\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "operator=A1=P0T_A0_P0_factorized\n"
                  << "transfer_smoothing_steps=" << transfer_steps << '\n'
                  << "L1_preconditioner=actual_diagonal_blocks\n"
                  << "L1_block_storage=padded_dense_inverse_6x6_fp32\n"
                  << "L1_smoother=degree1_block_Chebyshev\n"
                  << "L1_lambda_max=" << std::fixed << std::setprecision(6) << lambda1 << '\n'
                  << "L1_chebyshev_weight=" << weight << '\n'
                  << "host_round_trips_inside_timed_step=false\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space.coarse_dofs
                  << " aggregates=" << space.aggregates.size() << '\n'
                  << "block_setup_ms=" << block_setup_ms
                  << " block_inverse_bytes_fp32=" << blocks.storage_bytes_fp32()
                  << " min_block_cholesky_pivot=" << std::scientific
                  << blocks.min_cholesky_pivot << '\n'
                  << "gpu_vs_cpu_fp64_A1_relative_error=" << a1_oracle_error << '\n'
                  << "gpu_vs_constructed_exact_block_step_relative_error="
                  << step_oracle_error << '\n'
                  << "oracle_accept_1e-4=" << (accepted ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "A1_median_total_ms=" << gpu_a1.median_timing.total_ms
                  << " A1_best_total_ms=" << gpu_a1.best_timing.total_ms
                  << " A1_fine_operator_applies=" << gpu_a1.fine_operator_applies << '\n'
                  << "step_median_total_ms=" << gpu_step.median_total_ms
                  << " step_median_A1_ms=" << gpu_step.median_a1_ms
                  << " step_median_block_update_ms=" << gpu_step.median_block_update_ms
                  << " step_best_total_ms=" << gpu_step.best_total_ms
                  << " step_best_A1_ms=" << gpu_step.best_a1_ms
                  << " step_best_block_update_ms=" << gpu_step.best_block_update_ms << '\n'
                  << "step_fine_operator_applies=" << gpu_step.fine_operator_applies
                  << " persistent_L1_extra_bytes=" << gpu_step.persistent_l1_bytes << '\n';

        if (gpu_step.median_total_ms > 0.0) {
            std::cout << "median_fraction_A1="
                      << gpu_step.median_a1_ms / gpu_step.median_total_ms
                      << " median_fraction_block_update="
                      << gpu_step.median_block_update_ms / gpu_step.median_total_ms << '\n';
        }
        std::cout << "expected_fine_operator_applies_for_m0_1_A1=3\n";
        return accepted ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_l1_block_smoother_bench "
                  << "[repeats=50 [block_y=4 [target_nodes=12 [min_nodes=4 "
                  << "[omega0=0.394493 [lambda1=1.554198 [m0=1]]]]]]]\n";
        return 1;
    }
}
