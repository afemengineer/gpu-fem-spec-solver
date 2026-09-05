#include "gfss/gpu_m5_l2_full_shell.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

void check_cuda_l2_shell(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas_l2_shell(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

__global__ void m5_l2_zero_start_block_step_kernel(
    std::uint32_t blocks,
    const float* __restrict__ inverse_blocks,
    float weight,
    const float* __restrict__ rhs,
    float* __restrict__ x) {
    const std::uint32_t block_id = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (block_id >= blocks || q >= 6U) return;
    const float* inverse = inverse_blocks + static_cast<std::size_t>(block_id) * 36U;
    const float* b = rhs + static_cast<std::size_t>(block_id) * 6U;
    float value = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0U; j < 6U; ++j) {
        value = fmaf(inverse[6U * q + j], b[j], value);
    }
    x[static_cast<std::size_t>(block_id) * 6U + q] = weight * value;
}

__global__ void m5_l2_residual_kernel(
    std::size_t n,
    const float* __restrict__ rhs,
    const float* __restrict__ ax,
    float* __restrict__ residual) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) residual[i] = rhs[i] - ax[i];
}

__global__ void m5_l2_add_kernel(
    std::size_t n,
    const float* __restrict__ correction,
    float* __restrict__ x) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] += correction[i];
}

__global__ void m5_l2_post_block_step_kernel(
    std::uint32_t blocks,
    const float* __restrict__ inverse_blocks,
    float weight,
    const float* __restrict__ rhs,
    const float* __restrict__ ax,
    float* __restrict__ x) {
    const std::uint32_t block_id = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (block_id >= blocks || q >= 6U) return;
    const float* inverse = inverse_blocks + static_cast<std::size_t>(block_id) * 36U;
    const float* b = rhs + static_cast<std::size_t>(block_id) * 6U;
    const float* a = ax + static_cast<std::size_t>(block_id) * 6U;
    float value = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0U; j < 6U; ++j) {
        value = fmaf(inverse[6U * q + j], b[j] - a[j], value);
    }
    x[static_cast<std::size_t>(block_id) * 6U + q] =
        fmaf(weight, value, x[static_cast<std::size_t>(block_id) * 6U + q]);
}

double median_l2_shell(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

double elapsed_l2_shell(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    check_cuda_l2_shell(cudaEventElapsedTime(&ms, a, b),
                        "cudaEventElapsedTime(M5 L2 shell)");
    return static_cast<double>(ms);
}

GpuM5L2FullShellTiming median_timing(
    const std::vector<GpuM5L2FullShellTiming>& samples) {
    std::vector<double> pre, ra2, rup, p2t, p2, corr, pa2, post, total;
    pre.reserve(samples.size()); ra2.reserve(samples.size()); rup.reserve(samples.size());
    p2t.reserve(samples.size()); p2.reserve(samples.size()); corr.reserve(samples.size());
    pa2.reserve(samples.size()); post.reserve(samples.size()); total.reserve(samples.size());
    for (const auto& s : samples) {
        pre.push_back(s.pre_smooth_ms);
        ra2.push_back(s.residual_a2_ms);
        rup.push_back(s.residual_update_ms);
        p2t.push_back(s.p2t_ms);
        p2.push_back(s.p2_ms);
        corr.push_back(s.correction_ms);
        pa2.push_back(s.post_a2_ms);
        post.push_back(s.post_smooth_ms);
        total.push_back(s.total_ms);
    }
    return {median_l2_shell(std::move(pre)),
            median_l2_shell(std::move(ra2)),
            median_l2_shell(std::move(rup)),
            median_l2_shell(std::move(p2t)),
            median_l2_shell(std::move(p2)),
            median_l2_shell(std::move(corr)),
            median_l2_shell(std::move(pa2)),
            median_l2_shell(std::move(post)),
            median_l2_shell(std::move(total))};
}

GpuM5L2FullShellTiming best_timing(
    const std::vector<GpuM5L2FullShellTiming>& samples) {
    GpuM5L2FullShellTiming out;
    out.pre_smooth_ms = out.residual_a2_ms = out.residual_update_ms =
        out.p2t_ms = out.p2_ms = out.correction_ms = out.post_a2_ms =
        out.post_smooth_ms = out.total_ms = std::numeric_limits<double>::infinity();
    for (const auto& s : samples) {
        out.pre_smooth_ms = std::min(out.pre_smooth_ms, s.pre_smooth_ms);
        out.residual_a2_ms = std::min(out.residual_a2_ms, s.residual_a2_ms);
        out.residual_update_ms = std::min(out.residual_update_ms, s.residual_update_ms);
        out.p2t_ms = std::min(out.p2t_ms, s.p2t_ms);
        out.p2_ms = std::min(out.p2_ms, s.p2_ms);
        out.correction_ms = std::min(out.correction_ms, s.correction_ms);
        out.post_a2_ms = std::min(out.post_a2_ms, s.post_a2_ms);
        out.post_smooth_ms = std::min(out.post_smooth_ms, s.post_smooth_ms);
        out.total_ms = std::min(out.total_ms, s.total_ms);
    }
    return out;
}

}  // namespace

GpuM5L2FullShellResult benchmark_m5_l2_full_shell(
    const std::vector<float>& a2_dense_row_major,
    std::size_t l2_dofs,
    const std::vector<float>& inverse_blocks_6x6,
    const std::vector<float>& p2_dense_row_major,
    std::size_t l3_dofs,
    const std::vector<float>& rhs_l2,
    const std::vector<float>& external_l3_correction,
    double lambda2,
    int repeats) {
    if (l2_dofs == 0U || l3_dofs == 0U || l2_dofs % 6U != 0U ||
        l2_dofs > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        l3_dofs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("M5 L2 shell unsupported dimensions");
    }
    const std::size_t blocks = l2_dofs / 6U;
    if (a2_dense_row_major.size() != l2_dofs * l2_dofs ||
        inverse_blocks_6x6.size() != blocks * 36U ||
        p2_dense_row_major.size() != l2_dofs * l3_dofs ||
        rhs_l2.size() != l2_dofs ||
        external_l3_correction.size() != l3_dofs) {
        throw std::invalid_argument("M5 L2 shell payload shape mismatch");
    }
    if (!(lambda2 > 0.0) || !std::isfinite(lambda2) || repeats <= 0) {
        throw std::invalid_argument("M5 L2 shell lambda/repeats invalid");
    }
    for (float v : inverse_blocks_6x6) {
        if (!std::isfinite(v)) throw std::invalid_argument("M5 L2 inverse block non-finite");
    }

    const std::size_t a2_bytes = a2_dense_row_major.size() * sizeof(float);
    const std::size_t inv_bytes = inverse_blocks_6x6.size() * sizeof(float);
    const std::size_t p2_bytes = p2_dense_row_major.size() * sizeof(float);
    const std::size_t l2_vec_bytes = l2_dofs * sizeof(float);
    const std::size_t l3_vec_bytes = l3_dofs * sizeof(float);

    float* d_a2 = nullptr;
    float* d_inv = nullptr;
    float* d_p2 = nullptr;
    float* d_rhs = nullptr;
    float* d_e3 = nullptr;
    float* d_x = nullptr;
    float* d_ax = nullptr;
    float* d_residual = nullptr;
    float* d_r3 = nullptr;
    float* d_e2 = nullptr;
    cublasHandle_t handle{};
    cudaEvent_t events[9]{};

    try {
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_a2), a2_bytes),
                            "cudaMalloc(M5 L2 shell A2)");
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_inv), inv_bytes),
                            "cudaMalloc(M5 L2 shell inverse blocks)");
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_p2), p2_bytes),
                            "cudaMalloc(M5 L2 shell P2)");
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_rhs), l2_vec_bytes),
                            "cudaMalloc(M5 L2 shell rhs)");
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_e3), l3_vec_bytes),
                            "cudaMalloc(M5 L2 shell e3)");
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_x), l2_vec_bytes),
                            "cudaMalloc(M5 L2 shell x)");
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_ax), l2_vec_bytes),
                            "cudaMalloc(M5 L2 shell ax)");
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_residual), l2_vec_bytes),
                            "cudaMalloc(M5 L2 shell residual)");
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_r3), l3_vec_bytes),
                            "cudaMalloc(M5 L2 shell r3)");
        check_cuda_l2_shell(cudaMalloc(reinterpret_cast<void**>(&d_e2), l2_vec_bytes),
                            "cudaMalloc(M5 L2 shell e2)");

        check_cuda_l2_shell(cudaMemcpy(d_a2, a2_dense_row_major.data(), a2_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 L2 shell A2 H2D)");
        check_cuda_l2_shell(cudaMemcpy(d_inv, inverse_blocks_6x6.data(), inv_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 L2 shell inverse H2D)");
        check_cuda_l2_shell(cudaMemcpy(d_p2, p2_dense_row_major.data(), p2_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 L2 shell P2 H2D)");
        check_cuda_l2_shell(cudaMemcpy(d_rhs, rhs_l2.data(), l2_vec_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 L2 shell rhs H2D)");
        check_cuda_l2_shell(cudaMemcpy(d_e3, external_l3_correction.data(), l3_vec_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 L2 shell e3 H2D)");
        check_cublas_l2_shell(cublasCreate(&handle), "cublasCreate(M5 L2 shell)");

        const int n2 = static_cast<int>(l2_dofs);
        const int n3 = static_cast<int>(l3_dofs);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const float weight = static_cast<float>(1.0 / (0.55 * lambda2));
        constexpr unsigned int vector_threads = 256U;
        const unsigned int vector_blocks = static_cast<unsigned int>(
            (l2_dofs + vector_threads - 1U) / vector_threads);

        auto launch_a2 = [&]() {
            // Row-major symmetric A2 is column-major A2^T, which equals A2.
            check_cublas_l2_shell(cublasSgemv(handle, CUBLAS_OP_N, n2, n2,
                                              &alpha, d_a2, n2, d_x, 1,
                                              &beta, d_ax, 1),
                                  "cublasSgemv(M5 L2 shell A2)");
        };
        auto launch_p2t = [&]() {
            // Row-major P2(n2 x n3) is column-major P2^T(n3 x n2).
            check_cublas_l2_shell(cublasSgemv(handle, CUBLAS_OP_N, n3, n2,
                                              &alpha, d_p2, n3, d_residual, 1,
                                              &beta, d_r3, 1),
                                  "cublasSgemv(M5 L2 shell P2T)");
        };
        auto launch_p2 = [&]() {
            check_cublas_l2_shell(cublasSgemv(handle, CUBLAS_OP_T, n3, n2,
                                              &alpha, d_p2, n3, d_e3, 1,
                                              &beta, d_e2, 1),
                                  "cublasSgemv(M5 L2 shell P2)");
        };
        auto run_once = [&](bool mark) {
            if (mark) check_cuda_l2_shell(cudaEventRecord(events[0]), "M5 L2 shell event 0");
            m5_l2_zero_start_block_step_kernel<<<static_cast<unsigned int>(blocks), 32U>>>(
                static_cast<std::uint32_t>(blocks), d_inv, weight, d_rhs, d_x);
            check_cuda_l2_shell(cudaGetLastError(), "M5 L2 zero-start pre launch");
            if (mark) check_cuda_l2_shell(cudaEventRecord(events[1]), "M5 L2 shell event 1");

            launch_a2();
            if (mark) check_cuda_l2_shell(cudaEventRecord(events[2]), "M5 L2 shell event 2");
            m5_l2_residual_kernel<<<vector_blocks, vector_threads>>>(
                l2_dofs, d_rhs, d_ax, d_residual);
            check_cuda_l2_shell(cudaGetLastError(), "M5 L2 residual update launch");
            if (mark) check_cuda_l2_shell(cudaEventRecord(events[3]), "M5 L2 shell event 3");

            launch_p2t();
            if (mark) check_cuda_l2_shell(cudaEventRecord(events[4]), "M5 L2 shell event 4");
            launch_p2();
            if (mark) check_cuda_l2_shell(cudaEventRecord(events[5]), "M5 L2 shell event 5");
            m5_l2_add_kernel<<<vector_blocks, vector_threads>>>(l2_dofs, d_e2, d_x);
            check_cuda_l2_shell(cudaGetLastError(), "M5 L2 correction add launch");
            if (mark) check_cuda_l2_shell(cudaEventRecord(events[6]), "M5 L2 shell event 6");

            launch_a2();
            if (mark) check_cuda_l2_shell(cudaEventRecord(events[7]), "M5 L2 shell event 7");
            m5_l2_post_block_step_kernel<<<static_cast<unsigned int>(blocks), 32U>>>(
                static_cast<std::uint32_t>(blocks), d_inv, weight, d_rhs, d_ax, d_x);
            check_cuda_l2_shell(cudaGetLastError(), "M5 L2 post smoother launch");
            if (mark) check_cuda_l2_shell(cudaEventRecord(events[8]), "M5 L2 shell event 8");
        };

        run_once(false);
        check_cuda_l2_shell(cudaDeviceSynchronize(), "M5 L2 shell warmup sync");
        for (auto& event : events) {
            check_cuda_l2_shell(cudaEventCreate(&event), "cudaEventCreate(M5 L2 shell)");
        }

        std::vector<GpuM5L2FullShellTiming> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            run_once(true);
            check_cuda_l2_shell(cudaEventSynchronize(events[8]),
                                "cudaEventSynchronize(M5 L2 shell)");
            GpuM5L2FullShellTiming t;
            t.pre_smooth_ms = elapsed_l2_shell(events[0], events[1]);
            t.residual_a2_ms = elapsed_l2_shell(events[1], events[2]);
            t.residual_update_ms = elapsed_l2_shell(events[2], events[3]);
            t.p2t_ms = elapsed_l2_shell(events[3], events[4]);
            t.p2_ms = elapsed_l2_shell(events[4], events[5]);
            t.correction_ms = elapsed_l2_shell(events[5], events[6]);
            t.post_a2_ms = elapsed_l2_shell(events[6], events[7]);
            t.post_smooth_ms = elapsed_l2_shell(events[7], events[8]);
            t.total_ms = elapsed_l2_shell(events[0], events[8]);
            samples.push_back(t);
        }

        GpuM5L2FullShellResult result;
        result.l2_dofs = l2_dofs;
        result.l3_dofs = l3_dofs;
        result.l2_blocks = blocks;
        result.executed_a2_applies = 2U;
        result.l3_residual.resize(l3_dofs, 0.0f);
        result.final_l2_correction.resize(l2_dofs, 0.0f);
        check_cuda_l2_shell(cudaMemcpy(result.l3_residual.data(), d_r3, l3_vec_bytes,
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy(M5 L2 shell r3 D2H)");
        check_cuda_l2_shell(cudaMemcpy(result.final_l2_correction.data(), d_x, l2_vec_bytes,
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy(M5 L2 shell x D2H)");
        result.median_timing = median_timing(samples);
        result.best_timing = best_timing(samples);
        result.device_bytes = a2_bytes + inv_bytes + p2_bytes +
                              5U * l2_vec_bytes + 2U * l3_vec_bytes;

        for (auto event : events) cudaEventDestroy(event);
        cublasDestroy(handle);
        cudaFree(d_a2); cudaFree(d_inv); cudaFree(d_p2); cudaFree(d_rhs);
        cudaFree(d_e3); cudaFree(d_x); cudaFree(d_ax); cudaFree(d_residual);
        cudaFree(d_r3); cudaFree(d_e2);
        return result;
    } catch (...) {
        for (auto event : events) if (event) cudaEventDestroy(event);
        if (handle) cublasDestroy(handle);
        if (d_a2) cudaFree(d_a2);
        if (d_inv) cudaFree(d_inv);
        if (d_p2) cudaFree(d_p2);
        if (d_rhs) cudaFree(d_rhs);
        if (d_e3) cudaFree(d_e3);
        if (d_x) cudaFree(d_x);
        if (d_ax) cudaFree(d_ax);
        if (d_residual) cudaFree(d_residual);
        if (d_r3) cudaFree(d_r3);
        if (d_e2) cudaFree(d_e2);
        throw;
    }
}

}  // namespace gfss
