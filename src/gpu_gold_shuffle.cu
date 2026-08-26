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

struct DeviceNodeStencilEntryShuffle {
    std::int8_t dx{0};
    std::int8_t dy{0};
    std::int8_t dz{0};
    std::int8_t pad{0};
    float block[9]{};
};

struct DeviceDiagEntryShuffle {
    int offset{0};
    float b00{0.0f};
    float b11{0.0f};
    float b22{0.0f};
};

struct DeviceEdgeXYEntryShuffle {
    int offset{0};
    float b00{0.0f};
    float b01{0.0f};
    float b10{0.0f};
    float b11{0.0f};
    float b22{0.0f};
};

struct DeviceEdgeXZEntryShuffle {
    int offset{0};
    float b00{0.0f};
    float b02{0.0f};
    float b11{0.0f};
    float b20{0.0f};
    float b22{0.0f};
};

struct DeviceEdgeYZEntryShuffle {
    int offset{0};
    float b00{0.0f};
    float b11{0.0f};
    float b12{0.0f};
    float b21{0.0f};
    float b22{0.0f};
};

struct DeviceCornerEntryShuffle {
    int offset{0};
    float block[9]{};
};

static_assert(sizeof(DeviceNodeStencilEntryShuffle) == 40,
              "unexpected CUDA shuffle boundary entry layout");

__constant__ DeviceNodeStencilEntryShuffle kShuffleBoundaryEntries[27 * 27];
__constant__ std::uint8_t kShuffleBoundaryCounts[27];
__constant__ DeviceDiagEntryShuffle kShuffleDiag[7];
__constant__ DeviceEdgeXYEntryShuffle kShuffleEdgeXY[4];
__constant__ DeviceEdgeXZEntryShuffle kShuffleEdgeXZ[4];
__constant__ DeviceEdgeYZEntryShuffle kShuffleEdgeYZ[4];
__constant__ DeviceCornerEntryShuffle kShuffleCorner[8];
__constant__ float kShuffleInteriorBlocks[27][9];

void check_cuda_shuffle(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ __forceinline__ int axis_class_shuffle(std::uint32_t coordinate,
                                                   std::uint32_t max_coordinate) {
    return coordinate == 0U ? 0 : (coordinate == max_coordinate ? 2 : 1);
}

constexpr int shuffle_slot(int dx, int dy, int dz) {
    return (dx + 1) + 3 * ((dy + 1) + 3 * (dz + 1));
}

template <int DX, int DY, int DZ>
__device__ __forceinline__ void accumulate_structural_shuffle(
    const float* b,
    float x0,
    float x1,
    float x2,
    float& out_x,
    float& out_y,
    float& out_z) {
    constexpr int active_axes = (DX != 0 ? 1 : 0) +
                                (DY != 0 ? 1 : 0) +
                                (DZ != 0 ? 1 : 0);
    if constexpr (active_axes <= 1) {
        out_x = fmaf(b[0], x0, out_x);
        out_y = fmaf(b[4], x1, out_y);
        out_z = fmaf(b[8], x2, out_z);
    } else if constexpr (DZ == 0) {
        out_x = fmaf(b[0], x0, out_x);
        out_x = fmaf(b[1], x1, out_x);
        out_y = fmaf(b[3], x0, out_y);
        out_y = fmaf(b[4], x1, out_y);
        out_z = fmaf(b[8], x2, out_z);
    } else if constexpr (DY == 0) {
        out_x = fmaf(b[0], x0, out_x);
        out_x = fmaf(b[2], x2, out_x);
        out_y = fmaf(b[4], x1, out_y);
        out_z = fmaf(b[6], x0, out_z);
        out_z = fmaf(b[8], x2, out_z);
    } else if constexpr (DX == 0) {
        out_x = fmaf(b[0], x0, out_x);
        out_y = fmaf(b[4], x1, out_y);
        out_y = fmaf(b[5], x2, out_y);
        out_z = fmaf(b[7], x1, out_z);
        out_z = fmaf(b[8], x2, out_z);
    } else {
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

template <int DY, int DZ>
__device__ __forceinline__ void accumulate_shuffle_plane(
    std::uint32_t node,
    std::uint32_t sx,
    std::uint32_t sy,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float& out_x,
    float& out_y,
    float& out_z) {
    constexpr unsigned int kFullMask = 0xffffffffU;
    constexpr int kLeftSlot = shuffle_slot(-1, DY, DZ);
    constexpr int kCenterSlot = shuffle_slot(0, DY, DZ);
    constexpr int kRightSlot = shuffle_slot(1, DY, DZ);

    const int plane_offset = static_cast<int>(sx) *
                             (DY + static_cast<int>(sy) * DZ);
    const int center_neighbor = static_cast<int>(node) + plane_offset;

    const float c0 = ux[center_neighbor];
    const float c1 = uy[center_neighbor];
    const float c2 = uz[center_neighbor];

    float l0 = __shfl_up_sync(kFullMask, c0, 1);
    float l1 = __shfl_up_sync(kFullMask, c1, 1);
    float l2 = __shfl_up_sync(kFullMask, c2, 1);
    float r0 = __shfl_down_sync(kFullMask, c0, 1);
    float r1 = __shfl_down_sync(kFullMask, c1, 1);
    float r2 = __shfl_down_sync(kFullMask, c2, 1);

    const unsigned int lane = threadIdx.x;
    if (lane == 0U) {
        l0 = ux[center_neighbor - 1];
        l1 = uy[center_neighbor - 1];
        l2 = uz[center_neighbor - 1];
    }
    if (lane == 31U) {
        r0 = ux[center_neighbor + 1];
        r1 = uy[center_neighbor + 1];
        r2 = uz[center_neighbor + 1];
    }

    accumulate_structural_shuffle<-1, DY, DZ>(
        kShuffleInteriorBlocks[kLeftSlot], l0, l1, l2,
        out_x, out_y, out_z);
    accumulate_structural_shuffle<0, DY, DZ>(
        kShuffleInteriorBlocks[kCenterSlot], c0, c1, c2,
        out_x, out_y, out_z);
    accumulate_structural_shuffle<1, DY, DZ>(
        kShuffleInteriorBlocks[kRightSlot], r0, r1, r2,
        out_x, out_y, out_z);
}

__device__ __forceinline__ void accumulate_sparse_fallback_shuffle(
    int node,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float& out_x,
    float& out_y,
    float& out_z) {
#pragma unroll
    for (int e = 0; e < 7; ++e) {
        const DeviceDiagEntryShuffle entry = kShuffleDiag[e];
        const int neighbor = node + entry.offset;
        out_x = fmaf(entry.b00, ux[neighbor], out_x);
        out_y = fmaf(entry.b11, uy[neighbor], out_y);
        out_z = fmaf(entry.b22, uz[neighbor], out_z);
    }

#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const DeviceEdgeXYEntryShuffle entry = kShuffleEdgeXY[e];
        const int neighbor = node + entry.offset;
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
        const DeviceEdgeXZEntryShuffle entry = kShuffleEdgeXZ[e];
        const int neighbor = node + entry.offset;
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
        const DeviceEdgeYZEntryShuffle entry = kShuffleEdgeYZ[e];
        const int neighbor = node + entry.offset;
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
        const DeviceCornerEntryShuffle entry = kShuffleCorner[e];
        const int neighbor = node + entry.offset;
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

__global__ void node_stencil_gold_shuffle_kernel(
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

    const bool node_interior =
        i != 0U && i != nx && j != 0U && j != ny && k != 0U && k != nz;
    const std::uint32_t warp_x0 = blockIdx.x * 32U;
    const bool full_interior_x_warp =
        warp_x0 > 0U && (warp_x0 + 31U) < nx;

    if (node_interior && full_interior_x_warp) {
        accumulate_shuffle_plane<-1, -1>(node, sx, sy, ux, uy, uz,
                                         out_x, out_y, out_z);
        accumulate_shuffle_plane<0, -1>(node, sx, sy, ux, uy, uz,
                                        out_x, out_y, out_z);
        accumulate_shuffle_plane<1, -1>(node, sx, sy, ux, uy, uz,
                                        out_x, out_y, out_z);
        accumulate_shuffle_plane<-1, 0>(node, sx, sy, ux, uy, uz,
                                        out_x, out_y, out_z);
        accumulate_shuffle_plane<0, 0>(node, sx, sy, ux, uy, uz,
                                       out_x, out_y, out_z);
        accumulate_shuffle_plane<1, 0>(node, sx, sy, ux, uy, uz,
                                       out_x, out_y, out_z);
        accumulate_shuffle_plane<-1, 1>(node, sx, sy, ux, uy, uz,
                                        out_x, out_y, out_z);
        accumulate_shuffle_plane<0, 1>(node, sx, sy, ux, uy, uz,
                                       out_x, out_y, out_z);
        accumulate_shuffle_plane<1, 1>(node, sx, sy, ux, uy, uz,
                                       out_x, out_y, out_z);
    } else if (node_interior) {
        accumulate_sparse_fallback_shuffle(static_cast<int>(node),
                                           ux, uy, uz,
                                           out_x, out_y, out_z);
    } else {
        const int cls = axis_class_shuffle(i, nx) +
                        3 * (axis_class_shuffle(j, ny) +
                             3 * axis_class_shuffle(k, nz));
        const int count = static_cast<int>(kShuffleBoundaryCounts[cls]);
#pragma unroll 1
        for (int e = 0; e < count; ++e) {
            const DeviceNodeStencilEntryShuffle& entry =
                kShuffleBoundaryEntries[cls * 27 + e];
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

void upload_gold_shuffle_stencil(const StructuredHexMesh& mesh,
                                 const Material& material) {
    const auto host = build_cpu_gold_stencil_fp32(mesh, material);

    std::array<DeviceNodeStencilEntryShuffle, 27 * 27> boundary{};
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

    std::array<DeviceDiagEntryShuffle, 7> diag{};
    std::array<DeviceEdgeXYEntryShuffle, 4> edge_xy{};
    std::array<DeviceEdgeXZEntryShuffle, 4> edge_xz{};
    std::array<DeviceEdgeYZEntryShuffle, 4> edge_yz{};
    std::array<DeviceCornerEntryShuffle, 8> corner{};

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

    std::array<std::array<float, 9>, 27> interior_blocks{};
    constexpr std::size_t kInteriorClass = 13U;
    const auto interior_count = host.regular.counts[kInteriorClass];
    for (std::uint8_t e = 0; e < interior_count; ++e) {
        const auto& src = host.regular.entries[kInteriorClass][e];
        const int slot = shuffle_slot(static_cast<int>(src.dx),
                                      static_cast<int>(src.dy),
                                      static_cast<int>(src.dz));
        interior_blocks[static_cast<std::size_t>(slot)] = src.block;
    }

    check_cuda_shuffle(cudaMemcpyToSymbol(kShuffleBoundaryEntries,
                                          boundary.data(),
                                          boundary.size() * sizeof(DeviceNodeStencilEntryShuffle)),
                       "cudaMemcpyToSymbol(shuffle boundary entries)");
    check_cuda_shuffle(cudaMemcpyToSymbol(kShuffleBoundaryCounts,
                                          host.regular.counts.data(),
                                          host.regular.counts.size() * sizeof(std::uint8_t)),
                       "cudaMemcpyToSymbol(shuffle boundary counts)");
    check_cuda_shuffle(cudaMemcpyToSymbol(kShuffleDiag,
                                          diag.data(),
                                          diag.size() * sizeof(DeviceDiagEntryShuffle)),
                       "cudaMemcpyToSymbol(shuffle diag)");
    check_cuda_shuffle(cudaMemcpyToSymbol(kShuffleEdgeXY,
                                          edge_xy.data(),
                                          edge_xy.size() * sizeof(DeviceEdgeXYEntryShuffle)),
                       "cudaMemcpyToSymbol(shuffle edge xy)");
    check_cuda_shuffle(cudaMemcpyToSymbol(kShuffleEdgeXZ,
                                          edge_xz.data(),
                                          edge_xz.size() * sizeof(DeviceEdgeXZEntryShuffle)),
                       "cudaMemcpyToSymbol(shuffle edge xz)");
    check_cuda_shuffle(cudaMemcpyToSymbol(kShuffleEdgeYZ,
                                          edge_yz.data(),
                                          edge_yz.size() * sizeof(DeviceEdgeYZEntryShuffle)),
                       "cudaMemcpyToSymbol(shuffle edge yz)");
    check_cuda_shuffle(cudaMemcpyToSymbol(kShuffleCorner,
                                          corner.data(),
                                          corner.size() * sizeof(DeviceCornerEntryShuffle)),
                       "cudaMemcpyToSymbol(shuffle corner)");
    check_cuda_shuffle(cudaMemcpyToSymbol(kShuffleInteriorBlocks,
                                          interior_blocks.data(),
                                          interior_blocks.size() * sizeof(interior_blocks[0])),
                       "cudaMemcpyToSymbol(shuffle interior blocks)");
}

void aos_to_soa_shuffle(const std::vector<float>& x,
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

void soa_to_aos_shuffle(const std::vector<float>& yx,
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

struct ShuffleTimingSummary {
    double best{0.0};
    double median{0.0};
    double mean{0.0};
    double p95{0.0};
};

ShuffleTimingSummary summarize_shuffle(const std::vector<double>& samples) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) {
        const double position = p * static_cast<double>(sorted.size() - 1);
        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, sorted.size() - 1);
        const double fraction = position - static_cast<double>(lower);
        return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
    };

    ShuffleTimingSummary summary;
    summary.best = sorted.front();
    summary.median = percentile(0.50);
    summary.p95 = percentile(0.95);
    summary.mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                   static_cast<double>(samples.size());
    return summary;
}

}  // namespace

CudaOperatorResult apply_node_stencil_cuda_gold_shuffle(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<float>& x,
    int repeats,
    int block_y) {
    if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
        throw std::invalid_argument("CUDA GoldShuffle requires non-empty mesh dimensions");
    }
    if (x.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("CUDA GoldShuffle input vector size does not match mesh DOF count");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("CUDA GoldShuffle repeat count must be positive");
    }
    if (block_y <= 0 || block_y > 32 || 32 * block_y > 1024) {
        throw std::invalid_argument("CUDA GoldShuffle block_y must produce a valid 32 x block_y block");
    }
    if (mesh.node_count() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA GoldShuffle currently requires 32-bit node indexing");
    }

    upload_gold_shuffle_stencil(mesh, material);

    std::vector<float> ux;
    std::vector<float> uy;
    std::vector<float> uz;
    aos_to_soa_shuffle(x, ux, uy, uz);
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
        check_cuda_shuffle(cudaMalloc(reinterpret_cast<void**>(&d_ux), node_bytes), "cudaMalloc(GoldShuffle ux)");
        check_cuda_shuffle(cudaMalloc(reinterpret_cast<void**>(&d_uy), node_bytes), "cudaMalloc(GoldShuffle uy)");
        check_cuda_shuffle(cudaMalloc(reinterpret_cast<void**>(&d_uz), node_bytes), "cudaMalloc(GoldShuffle uz)");
        check_cuda_shuffle(cudaMalloc(reinterpret_cast<void**>(&d_yx), node_bytes), "cudaMalloc(GoldShuffle yx)");
        check_cuda_shuffle(cudaMalloc(reinterpret_cast<void**>(&d_yy), node_bytes), "cudaMalloc(GoldShuffle yy)");
        check_cuda_shuffle(cudaMalloc(reinterpret_cast<void**>(&d_yz), node_bytes), "cudaMalloc(GoldShuffle yz)");

        check_cuda_shuffle(cudaMemcpy(d_ux, ux.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(GoldShuffle ux H2D)");
        check_cuda_shuffle(cudaMemcpy(d_uy, uy.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(GoldShuffle uy H2D)");
        check_cuda_shuffle(cudaMemcpy(d_uz, uz.data(), node_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(GoldShuffle uz H2D)");
        check_cuda_shuffle(cudaEventCreate(&start), "cudaEventCreate(GoldShuffle start)");
        check_cuda_shuffle(cudaEventCreate(&stop), "cudaEventCreate(GoldShuffle stop)");

        const dim3 block(32U, static_cast<unsigned int>(block_y), 1U);
        const dim3 grid((mesh.nx + 1U + block.x - 1U) / block.x,
                        (mesh.ny + 1U + block.y - 1U) / block.y,
                        mesh.nz + 1U);

        node_stencil_gold_shuffle_kernel<<<grid, block>>>(
            mesh.nx, mesh.ny, mesh.nz,
            d_ux, d_uy, d_uz,
            d_yx, d_yy, d_yz);
        check_cuda_shuffle(cudaGetLastError(), "node_stencil_gold_shuffle_kernel warmup launch");
        check_cuda_shuffle(cudaDeviceSynchronize(), "cudaDeviceSynchronize(GoldShuffle warmup)");

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_shuffle(cudaEventRecord(start), "cudaEventRecord(GoldShuffle start)");
            node_stencil_gold_shuffle_kernel<<<grid, block>>>(
                mesh.nx, mesh.ny, mesh.nz,
                d_ux, d_uy, d_uz,
                d_yx, d_yy, d_yz);
            check_cuda_shuffle(cudaGetLastError(), "node_stencil_gold_shuffle_kernel launch");
            check_cuda_shuffle(cudaEventRecord(stop), "cudaEventRecord(GoldShuffle stop)");
            check_cuda_shuffle(cudaEventSynchronize(stop), "cudaEventSynchronize(GoldShuffle stop)");

            float kernel_ms = 0.0f;
            check_cuda_shuffle(cudaEventElapsedTime(&kernel_ms, start, stop),
                               "cudaEventElapsedTime(GoldShuffle kernel)");
            samples.push_back(static_cast<double>(kernel_ms));
        }

        const auto timing = summarize_shuffle(samples);

        std::vector<float> yx(nodes);
        std::vector<float> yy(nodes);
        std::vector<float> yz(nodes);
        check_cuda_shuffle(cudaMemcpy(yx.data(), d_yx, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(GoldShuffle yx D2H)");
        check_cuda_shuffle(cudaMemcpy(yy.data(), d_yy, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(GoldShuffle yy D2H)");
        check_cuda_shuffle(cudaMemcpy(yz.data(), d_yz, node_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(GoldShuffle yz D2H)");

        CudaOperatorResult result;
        soa_to_aos_shuffle(yx, yy, yz, result.y);
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
