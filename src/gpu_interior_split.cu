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

struct DeviceNodeStencilEntrySplit {
    std::int8_t dx{0};
    std::int8_t dy{0};
    std::int8_t dz{0};
    std::int8_t pad{0};
    float block[9]{};
};

static_assert(sizeof(DeviceNodeStencilEntrySplit) == 40,
              "unexpected CUDA split stencil entry layout");

constexpr int kInteriorClass = 13;

__constant__ DeviceNodeStencilEntrySplit kSplitEntries[27 * 27];
__constant__ std::uint8_t kSplitCounts[27];
__constant__ int kSplitInteriorOffsets[27];
__constant__ float kSplitInteriorBlocks[27 * 9];

void check_cuda_split(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ int axis_class_split(std::uint32_t coordinate,
                                                std::uint32_t max_coordinate) {
    return coordinate == 0U ? 0 : (coordinate == max_coordinate ? 2 : 1);
}

__device__ __forceinline__ void apply_dense_block_split(
    const float* b,
    float x0,
    float x1,
    float x2,
    float& out_x,
    float& out_y,
    float& out_z) {
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

__global__ void node_stencil_split_interior_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float* __restrict__ yx,
    float* __restrict__ yy,
    float* __restrict__ yz) {
    const std::uint32_t i = 1U + blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = 1U + blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = 1U + blockIdx.z;

    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::uint32_t node = i + sx * (j + sy * k);

    float out_x = 0.0f;
    float out_y = 0.0f;
    float out_z = 0.0f;

#pragma unroll 1
    for (int e = 0; e < 27; ++e) {
        const int neighbor = static_cast<int>(node) + kSplitInteriorOffsets[e];
        const float x0 = ux[neighbor];
        const float x1 = uy[neighbor];
        const float x2 = uz[neighbor];
        const float* b = &kSplitInteriorBlocks[e * 9];
        apply_dense_block_split(b, x0, x1, x2, out_x, out_y, out_z);
    }

    yx[node] = out_x;
    yy[node] = out_y;
    yz[node] = out_z;
}

__global__ void node_stencil_split_boundary_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::uint32_t boundary_count,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float* __restrict__ yx,
    float* __restrict__ yy,
    float* __restrict__ yz) {
    std::uint32_t bidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bidx >= boundary_count) {
        return;
    }

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::uint32_t sz = nz + 1U;

    const std::uint32_t x_faces = 2U * sy * sz;
    const std::uint32_t y_faces = 2U * (nx - 1U) * sz;

    std::uint32_t i = 0U;
    std::uint32_t j = 0U;
    std::uint32_t k = 0U;

    if (bidx < x_faces) {
        const std::uint32_t face = bidx & 1U;
        const std::uint32_t q = bidx >> 1U;
        i = face ? nx : 0U;
        j = q % sy;
        k = q / sy;
    } else if ((bidx -= x_faces) < y_faces) {
        const std::uint32_t face = bidx & 1U;
        const std::uint32_t q = bidx >> 1U;
        const std::uint32_t interior_x = nx - 1U;
        i = 1U + q % interior_x;
        j = face ? ny : 0U;
        k = q / interior_x;
    } else {
        bidx -= y_faces;
        const std::uint32_t face = bidx & 1U;
        const std::uint32_t q = bidx >> 1U;
        const std::uint32_t interior_x = nx - 1U;
        i = 1U + q % interior_x;
        j = 1U + q / interior_x;
        k = face ? nz : 0U;
    }

    const std::uint32_t node = i + sx * (j + sy * k);
    const int cls = axis_class_split(i, nx) +
                    3 * (axis_class_split(j, ny) +
                         3 * axis_class_split(k, nz));
    const int count = static_cast<int>(kSplitCounts[cls]);

    float out_x = 0.0f;
    float out_y = 0.0f;
    float out_z = 0.0f;

#pragma unroll 1
    for (int e = 0; e < count; ++e) {
        const DeviceNodeStencilEntrySplit& entry = kSplitEntries[cls * 27 + e];
        const int neighbor =
            static_cast<int>(node) + static_cast<int>(entry.dx) +
            static_cast<int>(sx) *
                (static_cast<int>(entry.dy) +
                 static_cast<int>(sy) * static_cast<int>(entry.dz));
        const float x0 = ux[neighbor];
        const float x1 = uy[neighbor];
        const float x2 = uz[neighbor];
        apply_dense_block_split(entry.block, x0, x1, x2, out_x, out_y, out_z);
    }

    yx[node] = out_x;
    yy[node] = out_y;
    yz[node] = out_z;
}

void upload_split_stencil(const StructuredHexMesh& mesh,
                          const Material& material) {
    const auto host = build_regular_node_stencil_fp32(mesh, material);
    std::array<DeviceNodeStencilEntrySplit, 27 * 27> flat{};

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

    check_cuda_split(cudaMemcpyToSymbol(kSplitEntries,
                                        flat.data(),
                                        flat.size() * sizeof(DeviceNodeStencilEntrySplit)),
                     "cudaMemcpyToSymbol(split entries)");
    check_cuda_split(cudaMemcpyToSymbol(kSplitCounts,
                                        host.counts.data(),
                                        host.counts.size() * sizeof(std::uint8_t)),
                     "cudaMemcpyToSymbol(split counts)");
    check_cuda_split(cudaMemcpyToSymbol(kSplitInteriorOffsets,
                                        offsets.data(),
                                        offsets.size() * sizeof(int)),
                     "cudaMemcpyToSymbol(split offsets)");
    check_cuda_split(cudaMemcpyToSymbol(kSplitInteriorBlocks,
                                        blocks.data(),
                                        blocks.size() * sizeof(float)),
                     "cudaMemcpyToSymbol(split blocks)");
}

void aos_to_soa_split(const std::vector<float>& x,
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

void soa_to_aos_split(const std::vector<float>& yx,
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

struct SplitTimingSummary {
    double best{0.0};
    double median{0.0};
    double mean{0.0};
    double p95{0.0};
};

SplitTimingSummary summarize_split(const std::vector<double>& samples) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) {
        const double position = p * static_cast<double>(sorted.size() - 1);
        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, sorted.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    };

    SplitTimingSummary summary;
    summary.best = sorted.front();
    summary.median = percentile(0.50);
    summary.p95 = percentile(0.95);
    summary.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                   static_cast<double>(samples.size());
    return summary;
}

}  // namespace

CudaOperatorResult apply_node_stencil_cuda_interior_split(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<float>& x,
    int repeats) {
    if (mesh.nx < 2U || mesh.ny < 2U || mesh.nz < 2U) {
        return apply_node_stencil_cuda_gold3d(mesh, material, x, repeats, 16);
    }
    if (x.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("CUDA interior-split input vector size does not match mesh DOF count");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("CUDA interior-split repeat count must be positive");
    }
    if (mesh.node_count() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA interior-split currently requires 32-bit node indexing");
    }

    const std::uint64_t boundary64 =
        mesh.node_count() -
        static_cast<std::uint64_t>(mesh.nx - 1U) *
            (mesh.ny - 1U) * (mesh.nz - 1U);
    if (boundary64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA interior-split boundary count exceeds 32-bit indexing");
    }
    const std::uint32_t boundary_count = static_cast<std::uint32_t>(boundary64);

    upload_split_stencil(mesh, material);

    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<float> uz;
    aos_to_soa_split(x, ux, uy, uz);
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
        check_cuda_split(cudaMalloc(reinterpret_cast<void**>(&d_ux), node_bytes), "cudaMalloc(split ux)");
        check_cuda_split(cudaMalloc(reinterpret_cast<void**>(&d_uy), node_bytes), "cudaMalloc(split uy)");
        check_cuda_split(cudaMalloc(reinterpret_cast<void**>(&d_uz), node_bytes), "cudaMalloc(split uz)");
        check_cuda_split(cudaMalloc(reinterpret_cast<void**>(&d_yx), node_bytes), "cudaMalloc(split yx)");
        check_cuda_split(cudaMalloc(reinterpret_cast<void**>(&d_yy), node_bytes), "cudaMalloc(split yy)");
        check_cuda_split(cudaMalloc(reinterpret_cast<void**>(&d_yz), node_bytes), "cudaMalloc(split yz)");

        check_cuda_split(cudaMemcpy(d_ux, ux.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(split ux H2D)");
        check_cuda_split(cudaMemcpy(d_uy, uy.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(split uy H2D)");
        check_cuda_split(cudaMemcpy(d_uz, uz.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(split uz H2D)");
        check_cuda_split(cudaEventCreate(&start), "cudaEventCreate(split start)");
        check_cuda_split(cudaEventCreate(&stop), "cudaEventCreate(split stop)");

        const dim3 interior_block(32U, 16U, 1U);
        const dim3 interior_grid(
            ((mesh.nx - 1U) + interior_block.x - 1U) / interior_block.x,
            ((mesh.ny - 1U) + interior_block.y - 1U) / interior_block.y,
            mesh.nz - 1U);
        constexpr std::uint32_t boundary_threads = 256U;
        const std::uint32_t boundary_blocks =
            (boundary_count + boundary_threads - 1U) / boundary_threads;

        node_stencil_split_interior_kernel<<<interior_grid, interior_block>>>(
            mesh.nx, mesh.ny, mesh.nz,
            d_ux, d_uy, d_uz,
            d_yx, d_yy, d_yz);
        node_stencil_split_boundary_kernel<<<boundary_blocks, boundary_threads>>>(
            mesh.nx, mesh.ny, mesh.nz, boundary_count,
            d_ux, d_uy, d_uz,
            d_yx, d_yy, d_yz);
        check_cuda_split(cudaGetLastError(), "interior-split warmup launch");
        check_cuda_split(cudaDeviceSynchronize(), "cudaDeviceSynchronize(interior-split warmup)");

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_split(cudaEventRecord(start), "cudaEventRecord(split start)");
            node_stencil_split_interior_kernel<<<interior_grid, interior_block>>>(
                mesh.nx, mesh.ny, mesh.nz,
                d_ux, d_uy, d_uz,
                d_yx, d_yy, d_yz);
            node_stencil_split_boundary_kernel<<<boundary_blocks, boundary_threads>>>(
                mesh.nx, mesh.ny, mesh.nz, boundary_count,
                d_ux, d_uy, d_uz,
                d_yx, d_yy, d_yz);
            check_cuda_split(cudaGetLastError(), "interior-split launch");
            check_cuda_split(cudaEventRecord(stop), "cudaEventRecord(split stop)");
            check_cuda_split(cudaEventSynchronize(stop), "cudaEventSynchronize(split stop)");

            float kernel_ms = 0.0f;
            check_cuda_split(cudaEventElapsedTime(&kernel_ms, start, stop),
                             "cudaEventElapsedTime(interior-split)");
            samples.push_back(static_cast<double>(kernel_ms));
        }

        const auto timing = summarize_split(samples);

        std::vector<float> yx(nodes);
        std::vector<float> yy(nodes);
        std::vector<float> yz(nodes);
        check_cuda_split(cudaMemcpy(yx.data(), d_yx, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(split yx D2H)");
        check_cuda_split(cudaMemcpy(yy.data(), d_yy, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(split yy D2H)");
        check_cuda_split(cudaMemcpy(yz.data(), d_yz, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(split yz D2H)");

        CudaOperatorResult result;
        soa_to_aos_split(yx, yy, yz, result.y);
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
