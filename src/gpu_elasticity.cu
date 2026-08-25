#include "gfss/gpu_elasticity.hpp"

#include "gfss/hex8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

__constant__ float kElementStiffness[24 * 24];

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__global__ void hex8_atomic_apply_kernel(std::uint32_t nx,
                                         std::uint32_t ny,
                                         std::uint32_t nz,
                                         const float* __restrict__ x,
                                         float* __restrict__ y) {
    const std::uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint64_t ne = static_cast<std::uint64_t>(nx) * ny * nz;
    if (element >= ne) {
        return;
    }

    const std::uint32_t ex = element % nx;
    const std::uint32_t q = element / nx;
    const std::uint32_t ey = q % ny;
    const std::uint32_t ez = q / ny;

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::uint32_t plane = sx * sy;
    const std::uint32_t base = ex + sx * (ey + sy * ez);

    const std::uint32_t nodes[8] = {
        base,
        base + 1U,
        base + 1U + sx,
        base + sx,
        base + plane,
        base + plane + 1U,
        base + plane + 1U + sx,
        base + plane + sx,
    };

    float xe[24];
#pragma unroll
    for (int a = 0; a < 8; ++a) {
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            xe[3 * a + c] = x[3U * nodes[a] + static_cast<std::uint32_t>(c)];
        }
    }

#pragma unroll
    for (int row = 0; row < 24; ++row) {
        float sum = 0.0f;
#pragma unroll
        for (int col = 0; col < 24; ++col) {
            sum = fmaf(kElementStiffness[row * 24 + col], xe[col], sum);
        }
        const int a = row / 3;
        const int c = row % 3;
        atomicAdd(&y[3U * nodes[a] + static_cast<std::uint32_t>(c)], sum);
    }
}

std::array<float, 24 * 24> regular_element_stiffness_float(const StructuredHexMesh& mesh,
                                                           const Material& material) {
    const auto coordinates = mesh.element_coordinates(0, 0, 0);
    const auto ke = hex8_stiffness(coordinates, material);
    std::array<float, 24 * 24> result{};
    for (int row = 0; row < 24; ++row) {
        for (int col = 0; col < 24; ++col) {
            result[static_cast<std::size_t>(row * 24 + col)] =
                static_cast<float>(ke[row][col]);
        }
    }
    return result;
}

struct TimingSummary {
    double best{0.0};
    double median{0.0};
    double mean{0.0};
    double p95{0.0};
};

TimingSummary summarize(const std::vector<double>& samples) {
    if (samples.empty()) {
        throw std::invalid_argument("cannot summarize an empty timing sample set");
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    const auto percentile = [&](double p) {
        const double position = p * static_cast<double>(sorted.size() - 1);
        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, sorted.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    };

    TimingSummary summary;
    summary.best = sorted.front();
    summary.median = percentile(0.50);
    summary.p95 = percentile(0.95);
    summary.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                   static_cast<double>(samples.size());
    return summary;
}

}  // namespace

CudaOperatorResult apply_matrix_free_cuda_atomic(const StructuredHexMesh& mesh,
                                                 const Material& material,
                                                 const std::vector<float>& x,
                                                 int repeats) {
    if (mesh.nx == 0 || mesh.ny == 0 || mesh.nz == 0) {
        throw std::invalid_argument("CUDA operator requires non-empty mesh dimensions");
    }
    if (x.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("CUDA input vector size does not match mesh DOF count");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("CUDA repeat count must be positive");
    }
    if (mesh.node_count() > std::numeric_limits<std::uint32_t>::max() / 3ULL) {
        throw std::invalid_argument("baseline CUDA kernel currently requires 32-bit global DOF indexing");
    }
    if (mesh.element_count() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("baseline CUDA kernel currently requires 32-bit element indexing");
    }

    const auto ke = regular_element_stiffness_float(mesh, material);
    check_cuda(cudaMemcpyToSymbol(kElementStiffness,
                                  ke.data(),
                                  ke.size() * sizeof(float)),
               "cudaMemcpyToSymbol(element stiffness)");

    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaEvent_t start{};
    cudaEvent_t after_zero{};
    cudaEvent_t stop{};

    const std::size_t bytes = x.size() * sizeof(float);
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes), "cudaMalloc(x)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_y), bytes), "cudaMalloc(y)");
        check_cuda(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(x H2D)");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&after_zero), "cudaEventCreate(after_zero)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

        constexpr int threads = 128;
        const std::uint32_t elements = static_cast<std::uint32_t>(mesh.element_count());
        const int blocks = static_cast<int>((elements + threads - 1U) / threads);

        // Unmeasured warmup to populate code/data caches and allow the GPU clock state
        // to settle before collecting benchmark samples.
        check_cuda(cudaMemsetAsync(d_y, 0, bytes), "cudaMemsetAsync(y warmup)");
        hex8_atomic_apply_kernel<<<blocks, threads>>>(mesh.nx, mesh.ny, mesh.nz, d_x, d_y);
        check_cuda(cudaGetLastError(), "hex8_atomic_apply_kernel warmup launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

        std::vector<double> total_samples;
        std::vector<double> zero_samples;
        std::vector<double> kernel_samples;
        total_samples.reserve(static_cast<std::size_t>(repeats));
        zero_samples.reserve(static_cast<std::size_t>(repeats));
        kernel_samples.reserve(static_cast<std::size_t>(repeats));

        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
            check_cuda(cudaMemsetAsync(d_y, 0, bytes), "cudaMemsetAsync(y)");
            check_cuda(cudaEventRecord(after_zero), "cudaEventRecord(after_zero)");
            hex8_atomic_apply_kernel<<<blocks, threads>>>(mesh.nx, mesh.ny, mesh.nz, d_x, d_y);
            check_cuda(cudaGetLastError(), "hex8_atomic_apply_kernel launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

            float total_ms = 0.0f;
            float zero_ms = 0.0f;
            float kernel_ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "cudaEventElapsedTime(total)");
            check_cuda(cudaEventElapsedTime(&zero_ms, start, after_zero), "cudaEventElapsedTime(zero)");
            check_cuda(cudaEventElapsedTime(&kernel_ms, after_zero, stop), "cudaEventElapsedTime(kernel)");

            total_samples.push_back(static_cast<double>(total_ms));
            zero_samples.push_back(static_cast<double>(zero_ms));
            kernel_samples.push_back(static_cast<double>(kernel_ms));
        }

        const auto total = summarize(total_samples);
        const auto zero = summarize(zero_samples);
        const auto kernel = summarize(kernel_samples);

        CudaOperatorResult result;
        result.y.resize(x.size());
        check_cuda(cudaMemcpy(result.y.data(), d_y, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(y D2H)");
        result.timing.best_ms = total.best;
        result.timing.median_ms = total.median;
        result.timing.mean_ms = total.mean;
        result.timing.p95_ms = total.p95;
        result.timing.best_zero_ms = zero.best;
        result.timing.median_zero_ms = zero.median;
        result.timing.mean_zero_ms = zero.mean;
        result.timing.best_kernel_ms = kernel.best;
        result.timing.median_kernel_ms = kernel.median;
        result.timing.mean_kernel_ms = kernel.mean;
        result.device_bytes = 2 * bytes;

        cudaEventDestroy(start);
        cudaEventDestroy(after_zero);
        cudaEventDestroy(stop);
        cudaFree(d_x);
        cudaFree(d_y);
        return result;
    } catch (...) {
        if (start) cudaEventDestroy(start);
        if (after_zero) cudaEventDestroy(after_zero);
        if (stop) cudaEventDestroy(stop);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        throw;
    }
}

}  // namespace gfss
