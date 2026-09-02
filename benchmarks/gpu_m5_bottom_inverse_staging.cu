#include "m5_bottom_inverse_staging.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed");
    }
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return (n & 1U) ? values[n / 2U]
                    : 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

__global__ void m5_bottom_inverse_matvec_kernel(
    std::uint32_t n,
    const float* __restrict__ inverse_col_major,
    const float* __restrict__ rhs,
    float* __restrict__ x) {
    const std::uint32_t i = threadIdx.x;
    if (i >= n) return;
    float sum = 0.0f;
    // Column-major inverse means, for fixed j, adjacent output threads i read
    // adjacent coefficients inverse[j*n+i]. This is coalesced across the warp.
    for (std::uint32_t j = 0U; j < n; ++j) {
        sum = fmaf(inverse_col_major[static_cast<std::size_t>(j) * n + i], rhs[j], sum);
    }
    x[i] = sum;
}

}  // namespace

M5BottomInverseStagingResult benchmark_m5_bottom_inverse_apply(
    const std::vector<float>& inverse_col_major,
    std::size_t n,
    const std::vector<float>& rhs,
    int repeats) {
    if (n == 0U || n > 256U || rhs.size() != n ||
        inverse_col_major.size() != n * n || repeats <= 0) {
        throw std::invalid_argument("M5 bottom inverse staging shape/options invalid");
    }
    for (const float v : inverse_col_major) {
        if (!std::isfinite(v)) throw std::invalid_argument("M5 bottom inverse contains non-finite value");
    }

    const std::size_t inverse_bytes = inverse_col_major.size() * sizeof(float);
    const std::size_t vector_bytes = n * sizeof(float);
    float* d_inverse = nullptr;
    float* d_rhs = nullptr;
    float* d_cublas_x = nullptr;
    float* d_custom_x = nullptr;
    cublasHandle_t handle{};
    cudaEvent_t start{};
    cudaEvent_t stop{};

    auto cleanup = [&]() noexcept {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (handle) cublasDestroy(handle);
        if (d_inverse) cudaFree(d_inverse);
        if (d_rhs) cudaFree(d_rhs);
        if (d_cublas_x) cudaFree(d_cublas_x);
        if (d_custom_x) cudaFree(d_custom_x);
    };

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_inverse), inverse_bytes),
                   "cudaMalloc(M5 bottom inverse)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_rhs), vector_bytes),
                   "cudaMalloc(M5 bottom inverse rhs)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cublas_x), vector_bytes),
                   "cudaMalloc(M5 bottom inverse cublas x)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_custom_x), vector_bytes),
                   "cudaMalloc(M5 bottom inverse custom x)");
        check_cuda(cudaMemcpy(d_inverse, inverse_col_major.data(), inverse_bytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(M5 bottom inverse)");
        check_cuda(cudaMemcpy(d_rhs, rhs.data(), vector_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy(M5 bottom inverse rhs)");
        check_cublas(cublasCreate(&handle), "cublasCreate(M5 bottom inverse)");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(M5 bottom inverse start)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(M5 bottom inverse stop)");

        const int ni = static_cast<int>(n);
        const float alpha = 1.0f;
        const float beta = 0.0f;
        auto launch_cublas = [&]() {
            check_cublas(cublasSgemv(handle, CUBLAS_OP_N, ni, ni,
                                     &alpha, d_inverse, ni, d_rhs, 1,
                                     &beta, d_cublas_x, 1),
                         "cublasSgemv(M5 bottom inverse)");
        };
        auto launch_custom = [&]() {
            m5_bottom_inverse_matvec_kernel<<<1U, 256U>>>(
                static_cast<std::uint32_t>(n), d_inverse, d_rhs, d_custom_x);
            check_cuda(cudaGetLastError(), "M5 bottom inverse custom launch");
        };

        launch_cublas();
        launch_custom();
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(M5 bottom inverse warmup)");

        std::vector<double> cublas_samples;
        std::vector<double> custom_samples;
        cublas_samples.reserve(static_cast<std::size_t>(repeats));
        custom_samples.reserve(static_cast<std::size_t>(repeats));

        for (int r = 0; r < repeats; ++r) {
            check_cuda(cudaEventRecord(start), "cudaEventRecord(M5 bottom inverse cublas start)");
            launch_cublas();
            check_cuda(cudaEventRecord(stop), "cudaEventRecord(M5 bottom inverse cublas stop)");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 bottom inverse cublas)");
            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, stop),
                       "cudaEventElapsedTime(M5 bottom inverse cublas)");
            cublas_samples.push_back(static_cast<double>(ms));
        }
        for (int r = 0; r < repeats; ++r) {
            check_cuda(cudaEventRecord(start), "cudaEventRecord(M5 bottom inverse custom start)");
            launch_custom();
            check_cuda(cudaEventRecord(stop), "cudaEventRecord(M5 bottom inverse custom stop)");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 bottom inverse custom)");
            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, stop),
                       "cudaEventElapsedTime(M5 bottom inverse custom)");
            custom_samples.push_back(static_cast<double>(ms));
        }

        M5BottomInverseStagingResult result;
        result.cublas_x.assign(n, 0.0f);
        result.custom_x.assign(n, 0.0f);
        check_cuda(cudaMemcpy(result.cublas_x.data(), d_cublas_x, vector_bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(M5 bottom inverse cublas result)");
        check_cuda(cudaMemcpy(result.custom_x.data(), d_custom_x, vector_bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy(M5 bottom inverse custom result)");
        result.cublas_median_ms = median(cublas_samples);
        result.cublas_best_ms = *std::min_element(cublas_samples.begin(), cublas_samples.end());
        result.custom_median_ms = median(custom_samples);
        result.custom_best_ms = *std::min_element(custom_samples.begin(), custom_samples.end());
        result.inverse_bytes = inverse_bytes;
        result.device_bytes = inverse_bytes + 3U * vector_bytes;
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

}  // namespace gfss
