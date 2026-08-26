#include "gfss/gpu_elasticity.hpp"

#include "gfss/cpu_stencil.hpp"
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

struct DeviceNodeStencilEntry {
    std::int8_t dx{0};
    std::int8_t dy{0};
    std::int8_t dz{0};
    std::int8_t pad{0};
    float block[9]{};
};

static_assert(sizeof(DeviceNodeStencilEntry) == 40,
              "unexpected CUDA stencil entry layout");

__constant__ DeviceNodeStencilEntry kNodeStencilEntries[27 * 27];
__constant__ std::uint8_t kNodeStencilCounts[27];

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

__device__ __forceinline__ int axis_class_device(std::uint32_t coordinate,
                                                  std::uint32_t max_coordinate) {
    return coordinate == 0U ? 0 : (coordinate == max_coordinate ? 2 : 1);
}

__global__ void node_stencil_soa_apply_kernel(std::uint32_t nx,
                                              std::uint32_t ny,
                                              std::uint32_t nz,
                                              const float* __restrict__ ux,
                                              const float* __restrict__ uy,
                                              const float* __restrict__ uz,
                                              float* __restrict__ yx,
                                              float* __restrict__ yy,
                                              float* __restrict__ yz) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::uint64_t nn64 = static_cast<std::uint64_t>(sx) * sy * (nz + 1U);
    if (static_cast<std::uint64_t>(node) >= nn64) {
        return;
    }

    const std::uint32_t i = node % sx;
    const std::uint32_t q = node / sx;
    const std::uint32_t j = q % sy;
    const std::uint32_t k = q / sy;

    const int cls = axis_class_device(i, nx) +
                    3 * (axis_class_device(j, ny) +
                         3 * axis_class_device(k, nz));
    const int count = static_cast<int>(kNodeStencilCounts[cls]);

    float out_x = 0.0f;
    float out_y = 0.0f;
    float out_z = 0.0f;

#pragma unroll 1
    for (int e = 0; e < count; ++e) {
        const DeviceNodeStencilEntry& entry = kNodeStencilEntries[cls * 27 + e];
        const int neighbor =
            static_cast<int>(node) + static_cast<int>(entry.dx) +
            static_cast<int>(sx) *
                (static_cast<int>(entry.dy) +
                 static_cast<int>(sy) * static_cast<int>(entry.dz));
        const float x0 = ux[neighbor];
        const float x1 = uy[neighbor];
        const float x2 = uz[neighbor];
        const float* b = entry.block;

        out_x = fmaf(b[0], x0, out_x);
        out_x = fmaf(b[1], x1, out_x);
        out_x = fmaf(b[2], x2, out_x);
        out_y = fmaf(b[3], x0, out_y);
        out_y = fmaf(b[4], x1, out_y);
        out_y = fmaf(b[5], x2, out_y);
        out_z = fmaf(b[6], x0, out_z);
        out_z = fmaf(b[7], x1, out_z);
        out_z = fmaf(b[8], x2, out_z);
    }

    yx[node] = out_x;
    yy[node] = out_y;
    yz[node] = out_z;
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

void upload_regular_node_stencil(const StructuredHexMesh& mesh,
                                 const Material& material) {
    const auto host = build_regular_node_stencil_fp32(mesh, material);
    std::array<DeviceNodeStencilEntry, 27 * 27> flat{};

    for (std::size_t cls = 0; cls < 27; ++cls) {
        for (std::size_t e = 0; e < 27; ++e) {
            const auto& src = host.entries[cls][e];
            auto& dst = flat[cls * 27 + e];
            dst.dx = src.dx;
            dst.dy = src.dy;
            dst.dz = src.dz;
            for (std::size_t q = 0; q < 9; ++q) {
                dst.block[q] = src.block[q];
            }
        }
    }

    check_cuda(cudaMemcpyToSymbol(kNodeStencilEntries,
                                  flat.data(),
                                  flat.size() * sizeof(DeviceNodeStencilEntry)),
               "cudaMemcpyToSymbol(node stencil entries)");
    check_cuda(cudaMemcpyToSymbol(kNodeStencilCounts,
                                  host.counts.data(),
                                  host.counts.size() * sizeof(std::uint8_t)),
               "cudaMemcpyToSymbol(node stencil counts)");
}

void aos_to_soa_host(const std::vector<float>& x,
                     std::vector<float>& ux,
                     std::vector<float>& uy,
                     std::vector<float>& uz) {
    const std::size_t nodes = x.size() / 3U;
    ux.resize(nodes);
    uy.resize(nodes);
    uz.resize(nodes);
    for (std::size_t node = 0; node < nodes; ++node) {
        ux[node] = x[3U * node + 0U];
        uy[node] = x[3U * node + 1U];
        uz[node] = x[3U * node + 2U];
    }
}

void soa_to_aos_host(const std::vector<float>& yx,
                     const std::vector<float>& yy,
                     const std::vector<float>& yz,
                     std::vector<float>& y) {
    const std::size_t nodes = yx.size();
    y.resize(3U * nodes);
    for (std::size_t node = 0; node < nodes; ++node) {
        y[3U * node + 0U] = yx[node];
        y[3U * node + 1U] = yy[node];
        y[3U * node + 2U] = yz[node];
    }
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

void validate_common_inputs(const StructuredHexMesh& mesh,
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
        throw std::invalid_argument("CUDA kernels currently require 32-bit global indexing");
    }
}

}  // namespace

CudaOperatorResult apply_matrix_free_cuda_atomic(const StructuredHexMesh& mesh,
                                                 const Material& material,
                                                 const std::vector<float>& x,
                                                 int repeats) {
    validate_common_inputs(mesh, x, repeats);
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

CudaOperatorResult apply_node_stencil_cuda_soa(const StructuredHexMesh& mesh,
                                               const Material& material,
                                               const std::vector<float>& x,
                                               int repeats,
                                               int threads_per_block) {
    validate_common_inputs(mesh, x, repeats);
    if (threads_per_block < 32 || threads_per_block > 1024 ||
        threads_per_block % 32 != 0) {
        throw std::invalid_argument("CUDA node-stencil threads_per_block must be a warp multiple in [32, 1024]");
    }
    if (mesh.node_count() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA node stencil currently requires 32-bit node indexing");
    }

    upload_regular_node_stencil(mesh, material);

    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<float> uz;
    aos_to_soa_host(x, ux, uy, uz);
    const std::size_t nodes = ux.size();
    const std::size_t node_bytes = nodes * sizeof(float);

    float* d_ux = nullptr;
    float* d_uy = nullptr;
    float* d_uz = nullptr;
    float* d_yx = nullptr;
    float* d_yy = nullptr;
    float* d_yz = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_ux), node_bytes), "cudaMalloc(ux)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_uy), node_bytes), "cudaMalloc(uy)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_uz), node_bytes), "cudaMalloc(uz)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_yx), node_bytes), "cudaMalloc(yx)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_yy), node_bytes), "cudaMalloc(yy)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_yz), node_bytes), "cudaMalloc(yz)");

        check_cuda(cudaMemcpy(d_ux, ux.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(ux H2D)");
        check_cuda(cudaMemcpy(d_uy, uy.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(uy H2D)");
        check_cuda(cudaMemcpy(d_uz, uz.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(uz H2D)");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(node start)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(node stop)");

        const std::uint32_t node_count = static_cast<std::uint32_t>(mesh.node_count());
        const int blocks = static_cast<int>(
            (node_count + static_cast<std::uint32_t>(threads_per_block) - 1U) /
            static_cast<std::uint32_t>(threads_per_block));

        node_stencil_soa_apply_kernel<<<blocks, threads_per_block>>>(
            mesh.nx, mesh.ny, mesh.nz,
            d_ux, d_uy, d_uz,
            d_yx, d_yy, d_yz);
        check_cuda(cudaGetLastError(), "node_stencil_soa_apply_kernel warmup launch");
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(node warmup)");

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda(cudaEventRecord(start), "cudaEventRecord(node start)");
            node_stencil_soa_apply_kernel<<<blocks, threads_per_block>>>(
                mesh.nx, mesh.ny, mesh.nz,
                d_ux, d_uy, d_uz,
                d_yx, d_yy, d_yz);
            check_cuda(cudaGetLastError(), "node_stencil_soa_apply_kernel launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord(node stop)");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(node stop)");

            float kernel_ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&kernel_ms, start, stop),
                       "cudaEventElapsedTime(node kernel)");
            samples.push_back(static_cast<double>(kernel_ms));
        }

        const auto timing = summarize(samples);

        std::vector<float> yx(nodes);
        std::vector<float> yy(nodes);
        std::vector<float> yz(nodes);
        check_cuda(cudaMemcpy(yx.data(), d_yx, node_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(yx D2H)");
        check_cuda(cudaMemcpy(yy.data(), d_yy, node_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(yy D2H)");
        check_cuda(cudaMemcpy(yz.data(), d_yz, node_bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(yz D2H)");

        CudaOperatorResult result;
        soa_to_aos_host(yx, yy, yz, result.y);
        result.timing.best_ms = timing.best;
        result.timing.median_ms = timing.median;
        result.timing.mean_ms = timing.mean;
        result.timing.p95_ms = timing.p95;
        result.timing.best_kernel_ms = timing.best;
        result.timing.median_kernel_ms = timing.median;
        result.timing.mean_kernel_ms = timing.mean;
        result.device_bytes = 6U * node_bytes;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_ux);
        cudaFree(d_uy);
        cudaFree(d_uz);
        cudaFree(d_yx);
        cudaFree(d_yy);
        cudaFree(d_yz);
        return result;
    } catch (...) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (d_ux) cudaFree(d_ux);
        if (d_uy) cudaFree(d_uy);
        if (d_uz) cudaFree(d_uz);
        if (d_yx) cudaFree(d_yx);
        if (d_yy) cudaFree(d_yy);
        if (d_yz) cudaFree(d_yz);
        throw;
    }
}

}  // namespace gfss
