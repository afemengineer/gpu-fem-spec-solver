#include "gfss/gpu_elasticity.hpp"

#include "gfss/cpu_gold.hpp"

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

struct DeviceNodeStencilEntrySparse {
    std::int8_t dx{0};
    std::int8_t dy{0};
    std::int8_t dz{0};
    std::int8_t pad{0};
    float block[9]{};
};

struct DeviceDiagEntry {
    int offset{0};
    float b00{0.0f};
    float b11{0.0f};
    float b22{0.0f};
};

struct DeviceEdgeXYEntry {
    int offset{0};
    float b00{0.0f};
    float b01{0.0f};
    float b10{0.0f};
    float b11{0.0f};
    float b22{0.0f};
};

struct DeviceEdgeXZEntry {
    int offset{0};
    float b00{0.0f};
    float b02{0.0f};
    float b11{0.0f};
    float b20{0.0f};
    float b22{0.0f};
};

struct DeviceEdgeYZEntry {
    int offset{0};
    float b00{0.0f};
    float b11{0.0f};
    float b12{0.0f};
    float b21{0.0f};
    float b22{0.0f};
};

struct DeviceCornerEntry {
    int offset{0};
    float block[9]{};
};

static_assert(sizeof(DeviceNodeStencilEntrySparse) == 40,
              "unexpected CUDA sparse boundary entry layout");

__constant__ DeviceNodeStencilEntrySparse kSparseBoundaryEntries[27 * 27];
__constant__ std::uint8_t kSparseBoundaryCounts[27];
__constant__ DeviceDiagEntry kSparseDiag[7];
__constant__ DeviceEdgeXYEntry kSparseEdgeXY[4];
__constant__ DeviceEdgeXZEntry kSparseEdgeXZ[4];
__constant__ DeviceEdgeYZEntry kSparseEdgeYZ[4];
__constant__ DeviceCornerEntry kSparseCorner[8];

void check_cuda_sparse(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ int axis_class_sparse(std::uint32_t coordinate,
                                                  std::uint32_t max_coordinate) {
    return coordinate == 0U ? 0 : (coordinate == max_coordinate ? 2 : 1);
}

__global__ void node_stencil_gold_sparse_kernel(
    std::uint32_t nx,
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
#pragma unroll
        for (int e = 0; e < 7; ++e) {
            const DeviceDiagEntry entry = kSparseDiag[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            out_x = fmaf(entry.b00, ux[neighbor], out_x);
            out_y = fmaf(entry.b11, uy[neighbor], out_y);
            out_z = fmaf(entry.b22, uz[neighbor], out_z);
        }

#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const DeviceEdgeXYEntry entry = kSparseEdgeXY[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            const float x0 = ux[neighbor];
            const float x1 = uy[neighbor];
            const float x2 = uz[neighbor];
            out_x = fmaf(entry.b00, x0, out_x);
            out_x = fmaf(entry.b01, x1, out_x);
            out_y = fmaf(entry.b10, x0, out_y);
            out_y = fmaf(entry.b11, x1, out_y);
            out_z = fmaf(entry.b22, x2, out_z);
        }

#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const DeviceEdgeXZEntry entry = kSparseEdgeXZ[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            const float x0 = ux[neighbor];
            const float x1 = uy[neighbor];
            const float x2 = uz[neighbor];
            out_x = fmaf(entry.b00, x0, out_x);
            out_x = fmaf(entry.b02, x2, out_x);
            out_y = fmaf(entry.b11, x1, out_y);
            out_z = fmaf(entry.b20, x0, out_z);
            out_z = fmaf(entry.b22, x2, out_z);
        }

#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const DeviceEdgeYZEntry entry = kSparseEdgeYZ[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            const float x0 = ux[neighbor];
            const float x1 = uy[neighbor];
            const float x2 = uz[neighbor];
            out_x = fmaf(entry.b00, x0, out_x);
            out_y = fmaf(entry.b11, x1, out_y);
            out_y = fmaf(entry.b12, x2, out_y);
            out_z = fmaf(entry.b21, x1, out_z);
            out_z = fmaf(entry.b22, x2, out_z);
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const DeviceCornerEntry entry = kSparseCorner[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
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
    } else {
        const int cls = axis_class_sparse(i, nx) +
                        3 * (axis_class_sparse(j, ny) +
                             3 * axis_class_sparse(k, nz));
        const int count = static_cast<int>(kSparseBoundaryCounts[cls]);
#pragma unroll 1
        for (int e = 0; e < count; ++e) {
            const DeviceNodeStencilEntrySparse& entry =
                kSparseBoundaryEntries[cls * 27 + e];
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

void upload_gold_sparse_stencil(const StructuredHexMesh& mesh,
                                const Material& material) {
    const auto host = build_cpu_gold_stencil_fp32(mesh, material);

    std::array<DeviceNodeStencilEntrySparse, 27 * 27> boundary{};
    for (std::size_t cls = 0; cls < 27; ++cls) {
        for (std::size_t e = 0; e < 27; ++e) {
            const auto& src = host.regular.entries[cls][e];
            auto& dst = boundary[cls * 27 + e];
            dst.dx = src.dx;
            dst.dy = src.dy;
            dst.dz = src.dz;
            for (std::size_t q = 0; q < 9; ++q) {
                dst.block[q] = src.block[q];
            }
        }
    }

    std::array<DeviceDiagEntry, 7> diag{};
    std::array<DeviceEdgeXYEntry, 4> edge_xy{};
    std::array<DeviceEdgeXZEntry, 4> edge_xz{};
    std::array<DeviceEdgeYZEntry, 4> edge_yz{};
    std::array<DeviceCornerEntry, 8> corner{};

    for (std::size_t e = 0; e < diag.size(); ++e) {
        const auto& src = host.diag[e];
        diag[e] = {static_cast<int>(src.offset), src.b00, src.b11, src.b22};
    }
    for (std::size_t e = 0; e < edge_xy.size(); ++e) {
        const auto& src = host.edge_xy[e];
        edge_xy[e] = {static_cast<int>(src.offset), src.b00, src.b01,
                      src.b10, src.b11, src.b22};
    }
    for (std::size_t e = 0; e < edge_xz.size(); ++e) {
        const auto& src = host.edge_xz[e];
        edge_xz[e] = {static_cast<int>(src.offset), src.b00, src.b02,
                      src.b11, src.b20, src.b22};
    }
    for (std::size_t e = 0; e < edge_yz.size(); ++e) {
        const auto& src = host.edge_yz[e];
        edge_yz[e] = {static_cast<int>(src.offset), src.b00, src.b11,
                      src.b12, src.b21, src.b22};
    }
    for (std::size_t e = 0; e < corner.size(); ++e) {
        const auto& src = host.corner[e];
        corner[e].offset = static_cast<int>(src.offset);
        for (std::size_t q = 0; q < 9; ++q) {
            corner[e].block[q] = src.block[q];
        }
    }

    check_cuda_sparse(cudaMemcpyToSymbol(kSparseBoundaryEntries,
                                         boundary.data(),
                                         boundary.size() * sizeof(DeviceNodeStencilEntrySparse)),
                      "cudaMemcpyToSymbol(sparse boundary entries)");
    check_cuda_sparse(cudaMemcpyToSymbol(kSparseBoundaryCounts,
                                         host.regular.counts.data(),
                                         host.regular.counts.size() * sizeof(std::uint8_t)),
                      "cudaMemcpyToSymbol(sparse boundary counts)");
    check_cuda_sparse(cudaMemcpyToSymbol(kSparseDiag,
                                         diag.data(),
                                         diag.size() * sizeof(DeviceDiagEntry)),
                      "cudaMemcpyToSymbol(sparse diag)");
    check_cuda_sparse(cudaMemcpyToSymbol(kSparseEdgeXY,
                                         edge_xy.data(),
                                         edge_xy.size() * sizeof(DeviceEdgeXYEntry)),
                      "cudaMemcpyToSymbol(sparse edge xy)");
    check_cuda_sparse(cudaMemcpyToSymbol(kSparseEdgeXZ,
                                         edge_xz.data(),
                                         edge_xz.size() * sizeof(DeviceEdgeXZEntry)),
                      "cudaMemcpyToSymbol(sparse edge xz)");
    check_cuda_sparse(cudaMemcpyToSymbol(kSparseEdgeYZ,
                                         edge_yz.data(),
                                         edge_yz.size() * sizeof(DeviceEdgeYZEntry)),
                      "cudaMemcpyToSymbol(sparse edge yz)");
    check_cuda_sparse(cudaMemcpyToSymbol(kSparseCorner,
                                         corner.data(),
                                         corner.size() * sizeof(DeviceCornerEntry)),
                      "cudaMemcpyToSymbol(sparse corner)");
}

void aos_to_soa_sparse(const std::vector<float>& x,
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

void soa_to_aos_sparse(const std::vector<float>& yx,
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

struct SparseTimingSummary {
    double best{0.0};
    double median{0.0};
    double mean{0.0};
    double p95{0.0};
};

SparseTimingSummary summarize_sparse(const std::vector<double>& samples) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) {
        const double position = p * static_cast<double>(sorted.size() - 1);
        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, sorted.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    };

    SparseTimingSummary summary;
    summary.best = sorted.front();
    summary.median = percentile(0.50);
    summary.p95 = percentile(0.95);
    summary.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                   static_cast<double>(samples.size());
    return summary;
}

}  // namespace

CudaOperatorResult apply_node_stencil_cuda_gold_sparse(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<float>& x,
    int repeats,
    int block_y) {
    if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
        throw std::invalid_argument("CUDA GoldSparse requires non-empty mesh dimensions");
    }
    if (x.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("CUDA GoldSparse input vector size does not match mesh DOF count");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("CUDA GoldSparse repeat count must be positive");
    }
    if (block_y <= 0 || block_y > 32 || 32 * block_y > 1024) {
        throw std::invalid_argument("CUDA GoldSparse block_y must produce a valid 32 x block_y block");
    }
    if (mesh.node_count() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA GoldSparse currently requires 32-bit node indexing");
    }

    upload_gold_sparse_stencil(mesh, material);

    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<float> uz;
    aos_to_soa_sparse(x, ux, uy, uz);
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
        check_cuda_sparse(cudaMalloc(reinterpret_cast<void**>(&d_ux), node_bytes), "cudaMalloc(GoldSparse ux)");
        check_cuda_sparse(cudaMalloc(reinterpret_cast<void**>(&d_uy), node_bytes), "cudaMalloc(GoldSparse uy)");
        check_cuda_sparse(cudaMalloc(reinterpret_cast<void**>(&d_uz), node_bytes), "cudaMalloc(GoldSparse uz)");
        check_cuda_sparse(cudaMalloc(reinterpret_cast<void**>(&d_yx), node_bytes), "cudaMalloc(GoldSparse yx)");
        check_cuda_sparse(cudaMalloc(reinterpret_cast<void**>(&d_yy), node_bytes), "cudaMalloc(GoldSparse yy)");
        check_cuda_sparse(cudaMalloc(reinterpret_cast<void**>(&d_yz), node_bytes), "cudaMalloc(GoldSparse yz)");

        check_cuda_sparse(cudaMemcpy(d_ux, ux.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(GoldSparse ux H2D)");
        check_cuda_sparse(cudaMemcpy(d_uy, uy.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(GoldSparse uy H2D)");
        check_cuda_sparse(cudaMemcpy(d_uz, uz.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(GoldSparse uz H2D)");
        check_cuda_sparse(cudaEventCreate(&start), "cudaEventCreate(GoldSparse start)");
        check_cuda_sparse(cudaEventCreate(&stop), "cudaEventCreate(GoldSparse stop)");

        const dim3 block(32U, static_cast<unsigned int>(block_y), 1U);
        const dim3 grid((mesh.nx + 1U + block.x - 1U) / block.x,
                        (mesh.ny + 1U + block.y - 1U) / block.y,
                        mesh.nz + 1U);

        node_stencil_gold_sparse_kernel<<<grid, block>>>(
            mesh.nx, mesh.ny, mesh.nz,
            d_ux, d_uy, d_uz,
            d_yx, d_yy, d_yz);
        check_cuda_sparse(cudaGetLastError(), "node_stencil_gold_sparse_kernel warmup launch");
        check_cuda_sparse(cudaDeviceSynchronize(), "cudaDeviceSynchronize(GoldSparse warmup)");

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_sparse(cudaEventRecord(start), "cudaEventRecord(GoldSparse start)");
            node_stencil_gold_sparse_kernel<<<grid, block>>>(
                mesh.nx, mesh.ny, mesh.nz,
                d_ux, d_uy, d_uz,
                d_yx, d_yy, d_yz);
            check_cuda_sparse(cudaGetLastError(), "node_stencil_gold_sparse_kernel launch");
            check_cuda_sparse(cudaEventRecord(stop), "cudaEventRecord(GoldSparse stop)");
            check_cuda_sparse(cudaEventSynchronize(stop), "cudaEventSynchronize(GoldSparse stop)");

            float kernel_ms = 0.0f;
            check_cuda_sparse(cudaEventElapsedTime(&kernel_ms, start, stop),
                              "cudaEventElapsedTime(GoldSparse kernel)");
            samples.push_back(static_cast<double>(kernel_ms));
        }

        const auto timing = summarize_sparse(samples);

        std::vector<float> yx(nodes);
        std::vector<float> yy(nodes);
        std::vector<float> yz(nodes);
        check_cuda_sparse(cudaMemcpy(yx.data(), d_yx, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(GoldSparse yx D2H)");
        check_cuda_sparse(cudaMemcpy(yy.data(), d_yy, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(GoldSparse yy D2H)");
        check_cuda_sparse(cudaMemcpy(yz.data(), d_yz, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(GoldSparse yz D2H)");

        CudaOperatorResult result;
        soa_to_aos_sparse(yx, yy, yz, result.y);
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

