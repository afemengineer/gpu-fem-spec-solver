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

struct DeviceBoundaryEntryShared {
    std::int8_t dx{0};
    std::int8_t dy{0};
    std::int8_t dz{0};
    std::int8_t pad{0};
    float block[9]{};
};

static_assert(sizeof(DeviceBoundaryEntryShared) == 40,
              "unexpected CUDA shared-tile stencil entry layout");

constexpr int kInteriorClassShared = 13;

__constant__ DeviceBoundaryEntryShared kSharedBoundaryEntries[27 * 27];
__constant__ std::uint8_t kSharedBoundaryCounts[27];
__constant__ std::int8_t kSharedInteriorDx[27];
__constant__ std::int8_t kSharedInteriorDy[27];
__constant__ std::int8_t kSharedInteriorDz[27];
__constant__ float kSharedInteriorBlocks[27 * 9];

void check_cuda_shared(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ int axis_class_shared(std::uint32_t coordinate,
                                                 std::uint32_t max_coordinate) {
    return coordinate == 0U ? 0 : (coordinate == max_coordinate ? 2 : 1);
}

__global__ void node_stencil_shared_tile_kernel(std::uint32_t nx,
                                                std::uint32_t ny,
                                                std::uint32_t nz,
                                                const float* __restrict__ ux,
                                                const float* __restrict__ uy,
                                                const float* __restrict__ uz,
                                                float* __restrict__ yx,
                                                float* __restrict__ yy,
                                                float* __restrict__ yz) {
    const int tile_x = static_cast<int>(blockDim.x) + 2;
    const int tile_y = static_cast<int>(blockDim.y) + 2;
    const int tile_z = static_cast<int>(blockDim.z) + 2;
    const int tile_nodes = tile_x * tile_y * tile_z;

    extern __shared__ float shared[];
    float* const sux = shared;
    float* const suy = sux + tile_nodes;
    float* const suz = suy + tile_nodes;

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const int thread_linear =
        tx + static_cast<int>(blockDim.x) *
                 (ty + static_cast<int>(blockDim.y) * tz);
    const int block_threads =
        static_cast<int>(blockDim.x * blockDim.y * blockDim.z);

    const int origin_x = static_cast<int>(blockIdx.x * blockDim.x);
    const int origin_y = static_cast<int>(blockIdx.y * blockDim.y);
    const int origin_z = static_cast<int>(blockIdx.z * blockDim.z);
    const int sx = static_cast<int>(nx) + 1;
    const int sy = static_cast<int>(ny) + 1;

    // Cooperative halo load. Each output block stages one-node halos in all
    // three directions and all three displacement components.
    for (int s = thread_linear; s < tile_nodes; s += block_threads) {
        const int lx = s % tile_x;
        const int q = s / tile_x;
        const int ly = q % tile_y;
        const int lz = q / tile_y;

        const int gx = origin_x + lx - 1;
        const int gy = origin_y + ly - 1;
        const int gz = origin_z + lz - 1;

        float vx = 0.0f;
        float vy = 0.0f;
        float vz = 0.0f;
        if (gx >= 0 && gx <= static_cast<int>(nx) &&
            gy >= 0 && gy <= static_cast<int>(ny) &&
            gz >= 0 && gz <= static_cast<int>(nz)) {
            const int node = gx + sx * (gy + sy * gz);
            vx = ux[node];
            vy = uy[node];
            vz = uz[node];
        }
        sux[s] = vx;
        suy[s] = vy;
        suz[s] = vz;
    }

    __syncthreads();

    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i > nx || j > ny || k > nz) {
        return;
    }

    const int node = static_cast<int>(i) +
                     sx * (static_cast<int>(j) + sy * static_cast<int>(k));

    float out_x = 0.0f;
    float out_y = 0.0f;
    float out_z = 0.0f;

    if (i != 0U && i != nx && j != 0U && j != ny && k != 0U && k != nz) {
        const int local_center =
            (tx + 1) + tile_x * ((ty + 1) + tile_y * (tz + 1));
#pragma unroll 1
        for (int e = 0; e < 27; ++e) {
            const int sn =
                local_center + static_cast<int>(kSharedInteriorDx[e]) +
                tile_x * (static_cast<int>(kSharedInteriorDy[e]) +
                          tile_y * static_cast<int>(kSharedInteriorDz[e]));
            const float x0 = sux[sn];
            const float x1 = suy[sn];
            const float x2 = suz[sn];
            const float* b = &kSharedInteriorBlocks[e * 9];

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
        const int cls = axis_class_shared(i, nx) +
                        3 * (axis_class_shared(j, ny) +
                             3 * axis_class_shared(k, nz));
        const int count = static_cast<int>(kSharedBoundaryCounts[cls]);
#pragma unroll 1
        for (int e = 0; e < count; ++e) {
            const DeviceBoundaryEntryShared& entry =
                kSharedBoundaryEntries[cls * 27 + e];
            const int neighbor =
                node + static_cast<int>(entry.dx) +
                sx * (static_cast<int>(entry.dy) +
                      sy * static_cast<int>(entry.dz));
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

void upload_shared_stencil(const StructuredHexMesh& mesh,
                           const Material& material) {
    const auto host = build_regular_node_stencil_fp32(mesh, material);
    std::array<DeviceBoundaryEntryShared, 27 * 27> flat{};

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

    std::array<std::int8_t, 27> dx{};
    std::array<std::int8_t, 27> dy{};
    std::array<std::int8_t, 27> dz{};
    std::array<float, 27 * 9> blocks{};
    for (std::size_t e = 0; e < 27; ++e) {
        const auto& src = host.entries[kInteriorClassShared][e];
        dx[e] = src.dx;
        dy[e] = src.dy;
        dz[e] = src.dz;
        for (std::size_t q = 0; q < 9; ++q) {
            blocks[e * 9 + q] = src.block[q];
        }
    }

    check_cuda_shared(cudaMemcpyToSymbol(kSharedBoundaryEntries,
                                         flat.data(),
                                         flat.size() * sizeof(DeviceBoundaryEntryShared)),
                      "cudaMemcpyToSymbol(shared boundary entries)");
    check_cuda_shared(cudaMemcpyToSymbol(kSharedBoundaryCounts,
                                         host.counts.data(),
                                         host.counts.size() * sizeof(std::uint8_t)),
                      "cudaMemcpyToSymbol(shared boundary counts)");
    check_cuda_shared(cudaMemcpyToSymbol(kSharedInteriorDx,
                                         dx.data(),
                                         dx.size() * sizeof(std::int8_t)),
                      "cudaMemcpyToSymbol(shared interior dx)");
    check_cuda_shared(cudaMemcpyToSymbol(kSharedInteriorDy,
                                         dy.data(),
                                         dy.size() * sizeof(std::int8_t)),
                      "cudaMemcpyToSymbol(shared interior dy)");
    check_cuda_shared(cudaMemcpyToSymbol(kSharedInteriorDz,
                                         dz.data(),
                                         dz.size() * sizeof(std::int8_t)),
                      "cudaMemcpyToSymbol(shared interior dz)");
    check_cuda_shared(cudaMemcpyToSymbol(kSharedInteriorBlocks,
                                         blocks.data(),
                                         blocks.size() * sizeof(float)),
                      "cudaMemcpyToSymbol(shared interior blocks)");
}

void aos_to_soa_shared(const std::vector<float>& x,
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

void soa_to_aos_shared(const std::vector<float>& yx,
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

struct SharedTimingSummary {
    double best{0.0};
    double median{0.0};
    double mean{0.0};
    double p95{0.0};
};

SharedTimingSummary summarize_shared(const std::vector<double>& samples) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) {
        const double position = p * static_cast<double>(sorted.size() - 1);
        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, sorted.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    };

    SharedTimingSummary summary;
    summary.best = sorted.front();
    summary.median = percentile(0.50);
    summary.p95 = percentile(0.95);
    summary.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                   static_cast<double>(samples.size());
    return summary;
}

}  // namespace

CudaOperatorResult apply_node_stencil_cuda_shared_tile(const StructuredHexMesh& mesh,
                                                       const Material& material,
                                                       const std::vector<float>& x,
                                                       int repeats,
                                                       int block_y,
                                                       int block_z) {
    if (mesh.nx == 0 || mesh.ny == 0 || mesh.nz == 0) {
        throw std::invalid_argument("CUDA shared tile requires non-empty mesh dimensions");
    }
    if (x.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("CUDA shared-tile input vector size does not match mesh DOF count");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("CUDA shared-tile repeat count must be positive");
    }
    if (block_y <= 0 || block_z <= 0 || 32 * block_y * block_z > 1024) {
        throw std::invalid_argument("CUDA shared-tile block must fit within 1024 threads");
    }
    if (mesh.node_count() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA shared tile currently requires 32-bit node indexing");
    }

    upload_shared_stencil(mesh, material);

    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<float> uz;
    aos_to_soa_shared(x, ux, uy, uz);
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
        check_cuda_shared(cudaMalloc(reinterpret_cast<void**>(&d_ux), node_bytes), "cudaMalloc(shared ux)");
        check_cuda_shared(cudaMalloc(reinterpret_cast<void**>(&d_uy), node_bytes), "cudaMalloc(shared uy)");
        check_cuda_shared(cudaMalloc(reinterpret_cast<void**>(&d_uz), node_bytes), "cudaMalloc(shared uz)");
        check_cuda_shared(cudaMalloc(reinterpret_cast<void**>(&d_yx), node_bytes), "cudaMalloc(shared yx)");
        check_cuda_shared(cudaMalloc(reinterpret_cast<void**>(&d_yy), node_bytes), "cudaMalloc(shared yy)");
        check_cuda_shared(cudaMalloc(reinterpret_cast<void**>(&d_yz), node_bytes), "cudaMalloc(shared yz)");

        check_cuda_shared(cudaMemcpy(d_ux, ux.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(shared ux H2D)");
        check_cuda_shared(cudaMemcpy(d_uy, uy.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(shared uy H2D)");
        check_cuda_shared(cudaMemcpy(d_uz, uz.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(shared uz H2D)");
        check_cuda_shared(cudaEventCreate(&start), "cudaEventCreate(shared start)");
        check_cuda_shared(cudaEventCreate(&stop), "cudaEventCreate(shared stop)");

        const dim3 block(32U,
                         static_cast<unsigned int>(block_y),
                         static_cast<unsigned int>(block_z));
        const dim3 grid((mesh.nx + 1U + block.x - 1U) / block.x,
                        (mesh.ny + 1U + block.y - 1U) / block.y,
                        (mesh.nz + 1U + block.z - 1U) / block.z);
        const std::size_t tile_nodes =
            static_cast<std::size_t>(block.x + 2U) *
            static_cast<std::size_t>(block.y + 2U) *
            static_cast<std::size_t>(block.z + 2U);
        const std::size_t shared_bytes = 3U * tile_nodes * sizeof(float);

        node_stencil_shared_tile_kernel<<<grid, block, shared_bytes>>>(
            mesh.nx, mesh.ny, mesh.nz,
            d_ux, d_uy, d_uz,
            d_yx, d_yy, d_yz);
        check_cuda_shared(cudaGetLastError(), "node_stencil_shared_tile_kernel warmup launch");
        check_cuda_shared(cudaDeviceSynchronize(), "cudaDeviceSynchronize(shared warmup)");

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_shared(cudaEventRecord(start), "cudaEventRecord(shared start)");
            node_stencil_shared_tile_kernel<<<grid, block, shared_bytes>>>(
                mesh.nx, mesh.ny, mesh.nz,
                d_ux, d_uy, d_uz,
                d_yx, d_yy, d_yz);
            check_cuda_shared(cudaGetLastError(), "node_stencil_shared_tile_kernel launch");
            check_cuda_shared(cudaEventRecord(stop), "cudaEventRecord(shared stop)");
            check_cuda_shared(cudaEventSynchronize(stop), "cudaEventSynchronize(shared stop)");

            float kernel_ms = 0.0f;
            check_cuda_shared(cudaEventElapsedTime(&kernel_ms, start, stop),
                              "cudaEventElapsedTime(shared kernel)");
            samples.push_back(static_cast<double>(kernel_ms));
        }

        const auto timing = summarize_shared(samples);

        std::vector<float> yx(nodes);
        std::vector<float> yy(nodes);
        std::vector<float> yz(nodes);
        check_cuda_shared(cudaMemcpy(yx.data(), d_yx, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(shared yx D2H)");
        check_cuda_shared(cudaMemcpy(yy.data(), d_yy, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(shared yy D2H)");
        check_cuda_shared(cudaMemcpy(yz.data(), d_yz, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(shared yz D2H)");

        CudaOperatorResult result;
        soa_to_aos_shared(yx, yy, yz, result.y);
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
