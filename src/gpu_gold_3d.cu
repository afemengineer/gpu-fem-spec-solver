#include "gfss/gpu_elasticity.hpp"

#include "gfss/cpu_stencil.hpp"

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

struct DeviceNodeStencilEntry3D {
    std::int8_t dx{0};
    std::int8_t dy{0};
    std::int8_t dz{0};
    std::int8_t pad{0};
    float block[9]{};
};

static_assert(sizeof(DeviceNodeStencilEntry3D) == 40,
              "unexpected CUDA Gold3D stencil entry layout");

constexpr int kInteriorClass = 13;

__constant__ DeviceNodeStencilEntry3D kGold3DEntries[27 * 27];
__constant__ std::uint8_t kGold3DCounts[27];
__constant__ int kGold3DInteriorOffsets[27];
__constant__ float kGold3DInteriorBlocks[27 * 9];

void check_cuda_gold3d(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ int axis_class_gold3d(std::uint32_t coordinate,
                                                 std::uint32_t max_coordinate) {
    return coordinate == 0U ? 0 : (coordinate == max_coordinate ? 2 : 1);
}

__global__ void node_stencil_gold3d_kernel(std::uint32_t nx,
                                           std::uint32_t ny,
                                           std::uint32_t nz,
                                           const float* __restrict__ ux,
                                           const float* __restrict__ uy,
                                           const float* __restrict__ uz,
                                           float* __restrict__ yx,
                                           float* __restrict__ yy,
                                           float* __restrict__ yz) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;

    if (i > nx || j > ny || k > nz) {
        return;
    }

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::uint32_t node = i + sx * (j + sy * k);

    float out_x = 0.0f;
    float out_y = 0.0f;
    float out_z = 0.0f;

    if (i != 0U && i != nx && j != 0U && j != ny && k != 0U && k != nz) {
#pragma unroll 1
        for (int e = 0; e < 27; ++e) {
            const int neighbor = static_cast<int>(node) + kGold3DInteriorOffsets[e];
            const float x0 = ux[neighbor];
            const float x1 = uy[neighbor];
            const float x2 = uz[neighbor];
            const float* b = &kGold3DInteriorBlocks[e * 9];

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
    } else {
        const int cls = axis_class_gold3d(i, nx) +
                        3 * (axis_class_gold3d(j, ny) +
                             3 * axis_class_gold3d(k, nz));
        const int count = static_cast<int>(kGold3DCounts[cls]);
#pragma unroll 1
        for (int e = 0; e < count; ++e) {
            const DeviceNodeStencilEntry3D& entry = kGold3DEntries[cls * 27 + e];
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
    }

    yx[node] = out_x;
    yy[node] = out_y;
    yz[node] = out_z;
}

void upload_gold3d_stencil(const StructuredHexMesh& mesh,
                           const Material& material) {
    const auto host = build_regular_node_stencil_fp32(mesh, material);
    std::array<DeviceNodeStencilEntry3D, 27 * 27> flat{};

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

    const std::uint32_t sx = mesh.nx + 1U;
    const std::uint32_t sy = mesh.ny + 1U;
    std::array<int, 27> offsets{};
    std::array<float, 27 * 9> blocks{};
    for (std::size_t e = 0; e < 27; ++e) {
        const auto& src = host.entries[kInteriorClass][e];
        offsets[e] = static_cast<int>(src.dx) +
                     static_cast<int>(sx) *
                         (static_cast<int>(src.dy) +
                          static_cast<int>(sy) * static_cast<int>(src.dz));
        for (std::size_t q = 0; q < 9; ++q) {
            blocks[e * 9 + q] = src.block[q];
        }
    }

    check_cuda_gold3d(cudaMemcpyToSymbol(kGold3DEntries,
                                         flat.data(),
                                         flat.size() * sizeof(DeviceNodeStencilEntry3D)),
                      "cudaMemcpyToSymbol(Gold3D entries)");
    check_cuda_gold3d(cudaMemcpyToSymbol(kGold3DCounts,
                                         host.counts.data(),
                                         host.counts.size() * sizeof(std::uint8_t)),
                      "cudaMemcpyToSymbol(Gold3D counts)");
    check_cuda_gold3d(cudaMemcpyToSymbol(kGold3DInteriorOffsets,
                                         offsets.data(),
                                         offsets.size() * sizeof(int)),
                      "cudaMemcpyToSymbol(Gold3D interior offsets)");
    check_cuda_gold3d(cudaMemcpyToSymbol(kGold3DInteriorBlocks,
                                         blocks.data(),
                                         blocks.size() * sizeof(float)),
                      "cudaMemcpyToSymbol(Gold3D interior blocks)");
}

void aos_to_soa_gold3d(const std::vector<float>& x,
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

void soa_to_aos_gold3d(const std::vector<float>& yx,
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

struct Gold3DTimingSummary {
    double best{0.0};
    double median{0.0};
    double mean{0.0};
    double p95{0.0};
};

Gold3DTimingSummary summarize_gold3d(const std::vector<double>& samples) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) {
        const double position = p * static_cast<double>(sorted.size() - 1);
        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, sorted.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    };

    Gold3DTimingSummary summary;
    summary.best = sorted.front();
    summary.median = percentile(0.50);
    summary.p95 = percentile(0.95);
    summary.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                   static_cast<double>(samples.size());
    return summary;
}

}  // namespace

CudaOperatorResult apply_node_stencil_cuda_gold3d(const StructuredHexMesh& mesh,
                                                  const Material& material,
                                                  const std::vector<float>& x,
                                                  int repeats,
                                                  int block_y) {
    if (mesh.nx == 0 || mesh.ny == 0 || mesh.nz == 0) {
        throw std::invalid_argument("CUDA Gold3D requires non-empty mesh dimensions");
    }
    if (x.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("CUDA Gold3D input vector size does not match mesh DOF count");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("CUDA Gold3D repeat count must be positive");
    }
    if (block_y <= 0 || block_y > 32 || 32 * block_y > 1024) {
        throw std::invalid_argument("CUDA Gold3D block_y must produce a valid 32 x block_y block");
    }
    if (mesh.node_count() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA Gold3D currently requires 32-bit node indexing");
    }

    upload_gold3d_stencil(mesh, material);

    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<float> uz;
    aos_to_soa_gold3d(x, ux, uy, uz);
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
        check_cuda_gold3d(cudaMalloc(reinterpret_cast<void**>(&d_ux), node_bytes), "cudaMalloc(Gold3D ux)");
        check_cuda_gold3d(cudaMalloc(reinterpret_cast<void**>(&d_uy), node_bytes), "cudaMalloc(Gold3D uy)");
        check_cuda_gold3d(cudaMalloc(reinterpret_cast<void**>(&d_uz), node_bytes), "cudaMalloc(Gold3D uz)");
        check_cuda_gold3d(cudaMalloc(reinterpret_cast<void**>(&d_yx), node_bytes), "cudaMalloc(Gold3D yx)");
        check_cuda_gold3d(cudaMalloc(reinterpret_cast<void**>(&d_yy), node_bytes), "cudaMalloc(Gold3D yy)");
        check_cuda_gold3d(cudaMalloc(reinterpret_cast<void**>(&d_yz), node_bytes), "cudaMalloc(Gold3D yz)");

        check_cuda_gold3d(cudaMemcpy(d_ux, ux.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(Gold3D ux H2D)");
        check_cuda_gold3d(cudaMemcpy(d_uy, uy.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(Gold3D uy H2D)");
        check_cuda_gold3d(cudaMemcpy(d_uz, uz.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(Gold3D uz H2D)");
        check_cuda_gold3d(cudaEventCreate(&start), "cudaEventCreate(Gold3D start)");
        check_cuda_gold3d(cudaEventCreate(&stop), "cudaEventCreate(Gold3D stop)");

        const dim3 block(32U, static_cast<unsigned int>(block_y), 1U);
        const dim3 grid((mesh.nx + 1U + block.x - 1U) / block.x,
                        (mesh.ny + 1U + block.y - 1U) / block.y,
                        mesh.nz + 1U);

        node_stencil_gold3d_kernel<<<grid, block>>>(
            mesh.nx, mesh.ny, mesh.nz,
            d_ux, d_uy, d_uz,
            d_yx, d_yy, d_yz);
        check_cuda_gold3d(cudaGetLastError(), "node_stencil_gold3d_kernel warmup launch");
        check_cuda_gold3d(cudaDeviceSynchronize(), "cudaDeviceSynchronize(Gold3D warmup)");

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_gold3d(cudaEventRecord(start), "cudaEventRecord(Gold3D start)");
            node_stencil_gold3d_kernel<<<grid, block>>>(
                mesh.nx, mesh.ny, mesh.nz,
                d_ux, d_uy, d_uz,
                d_yx, d_yy, d_yz);
            check_cuda_gold3d(cudaGetLastError(), "node_stencil_gold3d_kernel launch");
            check_cuda_gold3d(cudaEventRecord(stop), "cudaEventRecord(Gold3D stop)");
            check_cuda_gold3d(cudaEventSynchronize(stop), "cudaEventSynchronize(Gold3D stop)");

            float kernel_ms = 0.0f;
            check_cuda_gold3d(cudaEventElapsedTime(&kernel_ms, start, stop),
                              "cudaEventElapsedTime(Gold3D kernel)");
            samples.push_back(static_cast<double>(kernel_ms));
        }

        const auto timing = summarize_gold3d(samples);

        std::vector<float> yx(nodes);
        std::vector<float> yy(nodes);
        std::vector<float> yz(nodes);
        check_cuda_gold3d(cudaMemcpy(yx.data(), d_yx, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(Gold3D yx D2H)");
        check_cuda_gold3d(cudaMemcpy(yy.data(), d_yy, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(Gold3D yy D2H)");
        check_cuda_gold3d(cudaMemcpy(yz.data(), d_yz, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(Gold3D yz D2H)");

        CudaOperatorResult result;
        soa_to_aos_gold3d(yx, yy, yz, result.y);
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
