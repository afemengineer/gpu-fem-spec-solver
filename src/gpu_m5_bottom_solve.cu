#include "gfss/gpu_m5_bottom_solve.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

void check_cuda_bottom(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ float warp_sum_bottom(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, delta);
    }
    return value;
}

// A 132-DOF bottom level is small enough that a single warp can keep the
// sequential triangular dependency chain local while distributing each row dot
// product across 32 lanes. The solution is kept in shared memory. No global
// synchronization or host traffic is required between the forward/back solves.
__global__ void m5_bottom_cholesky_solve_kernel(
    std::uint32_t n,
    const float* __restrict__ lower,
    const float* __restrict__ rhs,
    float* __restrict__ x) {
    extern __shared__ float solution[];
    const std::uint32_t lane = threadIdx.x & 31U;
    if (threadIdx.x >= 32U) return;

    // Forward solve: L y = b. solution[] stores y as it is produced.
    for (std::uint32_t i = 0U; i < n; ++i) {
        float partial = 0.0f;
        const float* row = lower + static_cast<std::size_t>(i) * n;
        for (std::uint32_t k = lane; k < i; k += 32U) {
            partial = fmaf(row[k], solution[k], partial);
        }
        partial = warp_sum_bottom(partial);
        if (lane == 0U) {
            solution[i] = (rhs[i] - partial) / row[i];
        }
        __syncwarp();
    }

    // Backward solve: L^T x = y. Entries k>i in solution[] have already been
    // overwritten by x[k]; solution[i] still contains y[i] until this row ends.
    for (std::uint32_t ii = n; ii-- > 0U;) {
        float partial = 0.0f;
        for (std::uint32_t k = ii + 1U + lane; k < n; k += 32U) {
            partial = fmaf(lower[static_cast<std::size_t>(k) * n + ii],
                           solution[k], partial);
        }
        partial = warp_sum_bottom(partial);
        if (lane == 0U) {
            const float diagonal = lower[static_cast<std::size_t>(ii) * n + ii];
            solution[ii] = (solution[ii] - partial) / diagonal;
        }
        __syncwarp();
    }

    for (std::uint32_t i = lane; i < n; i += 32U) {
        x[i] = solution[i];
    }
}

double median_bottom(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

}  // namespace

GpuM5BottomSolveResult benchmark_m5_bottom_cholesky_solve(
    const std::vector<float>& lower_row_major,
    std::size_t n,
    const std::vector<float>& rhs,
    int repeats) {
    if (n == 0U || n > 256U ||
        n > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
        lower_row_major.size() != n * n || rhs.size() != n) {
        throw std::invalid_argument("M5 bottom solve shape unsupported");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("M5 bottom solve repeats must be positive");
    }
    for (std::size_t i = 0; i < n; ++i) {
        const float d = lower_row_major[i * n + i];
        if (!(d > 0.0f) || !std::isfinite(d)) {
            throw std::invalid_argument("M5 bottom solve diagonal invalid");
        }
    }

    const std::size_t factor_bytes = lower_row_major.size() * sizeof(float);
    const std::size_t vector_bytes = n * sizeof(float);
    float* d_lower = nullptr;
    float* d_rhs = nullptr;
    float* d_x = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    try {
        check_cuda_bottom(cudaMalloc(reinterpret_cast<void**>(&d_lower), factor_bytes),
                          "cudaMalloc(M5 bottom factor)");
        check_cuda_bottom(cudaMalloc(reinterpret_cast<void**>(&d_rhs), vector_bytes),
                          "cudaMalloc(M5 bottom rhs)");
        check_cuda_bottom(cudaMalloc(reinterpret_cast<void**>(&d_x), vector_bytes),
                          "cudaMalloc(M5 bottom x)");
        check_cuda_bottom(cudaMemcpy(d_lower, lower_row_major.data(), factor_bytes,
                                     cudaMemcpyHostToDevice),
                          "cudaMemcpy(M5 bottom factor H2D)");
        check_cuda_bottom(cudaMemcpy(d_rhs, rhs.data(), vector_bytes,
                                     cudaMemcpyHostToDevice),
                          "cudaMemcpy(M5 bottom rhs H2D)");

        m5_bottom_cholesky_solve_kernel<<<1U, 32U, vector_bytes>>>(
            static_cast<std::uint32_t>(n), d_lower, d_rhs, d_x);
        check_cuda_bottom(cudaGetLastError(), "M5 bottom warmup launch");
        check_cuda_bottom(cudaDeviceSynchronize(), "M5 bottom warmup sync");

        check_cuda_bottom(cudaEventCreate(&start), "cudaEventCreate(M5 bottom start)");
        check_cuda_bottom(cudaEventCreate(&stop), "cudaEventCreate(M5 bottom stop)");
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_bottom(cudaEventRecord(start), "cudaEventRecord(M5 bottom start)");
            m5_bottom_cholesky_solve_kernel<<<1U, 32U, vector_bytes>>>(
                static_cast<std::uint32_t>(n), d_lower, d_rhs, d_x);
            check_cuda_bottom(cudaGetLastError(), "M5 bottom solve launch");
            check_cuda_bottom(cudaEventRecord(stop), "cudaEventRecord(M5 bottom stop)");
            check_cuda_bottom(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 bottom)");
            float ms = 0.0f;
            check_cuda_bottom(cudaEventElapsedTime(&ms, start, stop),
                              "cudaEventElapsedTime(M5 bottom)");
            samples.push_back(static_cast<double>(ms));
        }

        GpuM5BottomSolveResult result;
        result.x.resize(n, 0.0f);
        check_cuda_bottom(cudaMemcpy(result.x.data(), d_x, vector_bytes,
                                     cudaMemcpyDeviceToHost),
                          "cudaMemcpy(M5 bottom x D2H)");
        result.median_ms = median_bottom(samples);
        result.best_ms = *std::min_element(samples.begin(), samples.end());
        result.dofs = n;
        result.factor_bytes = factor_bytes;
        result.device_bytes = factor_bytes + 2U * vector_bytes;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_lower);
        cudaFree(d_rhs);
        cudaFree(d_x);
        return result;
    } catch (...) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (d_lower) cudaFree(d_lower);
        if (d_rhs) cudaFree(d_rhs);
        if (d_x) cudaFree(d_x);
        throw;
    }
}

}  // namespace gfss
