// Reuse the exact persistent-PCG GoldSparse/Jacobi stencil data and launchers.
// The public PCG entry point is renamed so this TU contributes only the
// smoothed-aggregation API.
#define solve_pcg_cuda_gold_sparse_x0 solve_pcg_cuda_gold_sparse_x0_sa_internal_unused
#include "gpu_pcg.cu"
#undef solve_pcg_cuda_gold_sparse_x0

#include "gfss/gpu_smoothed_aggregation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gfss {
namespace {

struct DeviceAggregateSa {
    std::uint32_t coarse_offset{0};
    std::uint32_t rank{0};
    float centroid[3]{};
    float inv_scale{1.0f};
    float transform[36]{};
};

static_assert(sizeof(DeviceAggregateSa) == 168,
              "unexpected smoothed-aggregation aggregate layout");

enum class TimedStage {
    P0,
    ForwardSmooth,
    FineA,
    TransposeSmooth,
    P0T,
};

__device__ __forceinline__ void aggregate_basis_values(
    const DeviceAggregateSa* aggregate,
    float x,
    float y,
    float z,
    std::uint32_t q,
    float& bx,
    float& by,
    float& bz) {
    const float* t = aggregate->transform + 6U * q;
    bx = t[0] + z * t[4] - y * t[5];
    by = t[1] - z * t[3] + x * t[5];
    bz = t[2] + y * t[3] - x * t[4];
}

__global__ void tentative_prolongation_aggregate_sa_kernel(
    std::size_t nodes,
    const std::uint32_t* __restrict__ aggregate_offsets,
    const std::uint32_t* __restrict__ aggregate_nodes,
    const DeviceAggregateSa* __restrict__ aggregates,
    const float* __restrict__ coordinates,
    const float* __restrict__ coarse,
    float* __restrict__ fine) {
    const std::uint32_t aggregate_id = blockIdx.x;
    const DeviceAggregateSa* aggregate = aggregates + aggregate_id;
    const std::uint32_t first = aggregate_offsets[aggregate_id];
    const std::uint32_t last = aggregate_offsets[aggregate_id + 1U];

    __shared__ float coarse_local[6];
    if (threadIdx.x < aggregate->rank) {
        coarse_local[threadIdx.x] = coarse[aggregate->coarse_offset + threadIdx.x];
    }
    __syncwarp();

    for (std::uint32_t p = first + threadIdx.x; p < last; p += blockDim.x) {
        const std::size_t node = aggregate_nodes[p];
        const float x = (coordinates[node] - aggregate->centroid[0]) * aggregate->inv_scale;
        const float y = (coordinates[nodes + node] - aggregate->centroid[1]) * aggregate->inv_scale;
        const float z = (coordinates[2U * nodes + node] - aggregate->centroid[2]) * aggregate->inv_scale;

        float ux = 0.0f;
        float uy = 0.0f;
        float uz = 0.0f;
        for (std::uint32_t q = 0; q < aggregate->rank; ++q) {
            float bx = 0.0f;
            float by = 0.0f;
            float bz = 0.0f;
            aggregate_basis_values(aggregate, x, y, z, q, bx, by, bz);
            const float c = coarse_local[q];
            ux = fmaf(bx, c, ux);
            uy = fmaf(by, c, uy);
            uz = fmaf(bz, c, uz);
        }
        fine[node] = ux;
        fine[nodes + node] = uy;
        fine[2U * nodes + node] = uz;
    }
}

__global__ void tentative_restriction_aggregate_sa_kernel(
    std::size_t nodes,
    const std::uint32_t* __restrict__ aggregate_offsets,
    const std::uint32_t* __restrict__ aggregate_nodes,
    const DeviceAggregateSa* __restrict__ aggregates,
    const float* __restrict__ coordinates,
    const float* __restrict__ fine,
    float* __restrict__ coarse) {
    const std::uint32_t aggregate_id = blockIdx.x;
    const DeviceAggregateSa* aggregate = aggregates + aggregate_id;
    const std::uint32_t first = aggregate_offsets[aggregate_id];
    const std::uint32_t last = aggregate_offsets[aggregate_id + 1U];

    float sums[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (std::uint32_t p = first + threadIdx.x; p < last; p += blockDim.x) {
        const std::size_t node = aggregate_nodes[p];
        const float x = (coordinates[node] - aggregate->centroid[0]) * aggregate->inv_scale;
        const float y = (coordinates[nodes + node] - aggregate->centroid[1]) * aggregate->inv_scale;
        const float z = (coordinates[2U * nodes + node] - aggregate->centroid[2]) * aggregate->inv_scale;
        const float fx = fine[node];
        const float fy = fine[nodes + node];
        const float fz = fine[2U * nodes + node];

        for (std::uint32_t q = 0; q < aggregate->rank; ++q) {
            float bx = 0.0f;
            float by = 0.0f;
            float bz = 0.0f;
            aggregate_basis_values(aggregate, x, y, z, q, bx, by, bz);
            sums[q] = fmaf(bx, fx, sums[q]);
            sums[q] = fmaf(by, fy, sums[q]);
            sums[q] = fmaf(bz, fz, sums[q]);
        }
    }

    constexpr unsigned int mask = 0xffffffffU;
    for (int delta = 16; delta > 0; delta >>= 1) {
#pragma unroll
        for (int q = 0; q < 6; ++q) {
            sums[q] += __shfl_down_sync(mask, sums[q], delta);
        }
    }

    if (threadIdx.x == 0U) {
        for (std::uint32_t q = 0; q < aggregate->rank; ++q) {
            coarse[aggregate->coarse_offset + q] = sums[q];
        }
    }
}

__device__ __forceinline__ void apply_gold_sparse_sa(
    std::uint32_t i,
    std::uint32_t j,
    std::uint32_t k,
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float& out_x,
    float& out_y,
    float& out_z) {
    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::uint32_t node = i + sx * (j + sy * k);
    out_x = 0.0f;
    out_y = 0.0f;
    out_z = 0.0f;

    if (i != 0U && i != nx && j != 0U && j != ny && k != 0U && k != nz) {
#pragma unroll
        for (int e = 0; e < 7; ++e) {
            const DeviceDiagEntryPcg entry = kPcgDiag[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            out_x = fmaf(entry.b00, ux[neighbor], out_x);
            out_y = fmaf(entry.b11, uy[neighbor], out_y);
            out_z = fmaf(entry.b22, uz[neighbor], out_z);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const DeviceEdgeXYEntryPcg entry = kPcgEdgeXY[e];
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
            const DeviceEdgeXZEntryPcg entry = kPcgEdgeXZ[e];
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
            const DeviceEdgeYZEntryPcg entry = kPcgEdgeYZ[e];
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
            const DeviceCornerEntryPcg entry = kPcgCorner[e];
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
        const int cls = axis_class_pcg(i, nx) +
                        3 * (axis_class_pcg(j, ny) + 3 * axis_class_pcg(k, nz));
        const int count = static_cast<int>(kPcgBoundaryCounts[cls]);
#pragma unroll 1
        for (int e = 0; e < count; ++e) {
            const DeviceNodeStencilEntryPcg& entry = kPcgBoundaryEntries[cls * 27 + e];
            const int neighbor = static_cast<int>(node) + static_cast<int>(entry.dx) +
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
}

__device__ __forceinline__ void apply_gold_sparse_scaled_input_sa(
    std::uint32_t i,
    std::uint32_t j,
    std::uint32_t k,
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    const float* __restrict__ ux,
    const float* __restrict__ uy,
    const float* __restrict__ uz,
    float& out_x,
    float& out_y,
    float& out_z) {
    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::uint32_t node = i + sx * (j + sy * k);
    out_x = 0.0f;
    out_y = 0.0f;
    out_z = 0.0f;

    const bool deep_interior =
        i > 1U && i + 1U < nx &&
        j > 1U && j + 1U < ny &&
        k > 1U && k + 1U < nz;

    if (deep_interior) {
        constexpr int interior_cls = 13;
        const float sx0 = kPcgInvDiag[interior_cls][0];
        const float sx1 = kPcgInvDiag[interior_cls][1];
        const float sx2 = kPcgInvDiag[interior_cls][2];
#pragma unroll
        for (int e = 0; e < 7; ++e) {
            const DeviceDiagEntryPcg entry = kPcgDiag[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            out_x = fmaf(entry.b00, sx0 * ux[neighbor], out_x);
            out_y = fmaf(entry.b11, sx1 * uy[neighbor], out_y);
            out_z = fmaf(entry.b22, sx2 * uz[neighbor], out_z);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const DeviceEdgeXYEntryPcg entry = kPcgEdgeXY[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            const float x0 = sx0 * ux[neighbor];
            const float x1 = sx1 * uy[neighbor];
            const float x2 = sx2 * uz[neighbor];
            out_x = fmaf(entry.b00, x0, out_x);
            out_x = fmaf(entry.b01, x1, out_x);
            out_y = fmaf(entry.b10, x0, out_y);
            out_y = fmaf(entry.b11, x1, out_y);
            out_z = fmaf(entry.b22, x2, out_z);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const DeviceEdgeXZEntryPcg entry = kPcgEdgeXZ[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            const float x0 = sx0 * ux[neighbor];
            const float x1 = sx1 * uy[neighbor];
            const float x2 = sx2 * uz[neighbor];
            out_x = fmaf(entry.b00, x0, out_x);
            out_x = fmaf(entry.b02, x2, out_x);
            out_y = fmaf(entry.b11, x1, out_y);
            out_z = fmaf(entry.b20, x0, out_z);
            out_z = fmaf(entry.b22, x2, out_z);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const DeviceEdgeYZEntryPcg entry = kPcgEdgeYZ[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            const float x0 = sx0 * ux[neighbor];
            const float x1 = sx1 * uy[neighbor];
            const float x2 = sx2 * uz[neighbor];
            out_x = fmaf(entry.b00, x0, out_x);
            out_y = fmaf(entry.b11, x1, out_y);
            out_y = fmaf(entry.b12, x2, out_y);
            out_z = fmaf(entry.b21, x1, out_z);
            out_z = fmaf(entry.b22, x2, out_z);
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const DeviceCornerEntryPcg entry = kPcgCorner[e];
            const int neighbor = static_cast<int>(node) + entry.offset;
            const float x0 = sx0 * ux[neighbor];
            const float x1 = sx1 * uy[neighbor];
            const float x2 = sx2 * uz[neighbor];
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
        return;
    }

    const int cls = axis_class_pcg(i, nx) +
                    3 * (axis_class_pcg(j, ny) + 3 * axis_class_pcg(k, nz));
    const int count = static_cast<int>(kPcgBoundaryCounts[cls]);
#pragma unroll 1
    for (int e = 0; e < count; ++e) {
        const DeviceNodeStencilEntryPcg& entry = kPcgBoundaryEntries[cls * 27 + e];
        const int ni = static_cast<int>(i) + static_cast<int>(entry.dx);
        const int nj = static_cast<int>(j) + static_cast<int>(entry.dy);
        const int nk = static_cast<int>(k) + static_cast<int>(entry.dz);
        const int neighbor = ni + static_cast<int>(sx) *
                                  (nj + static_cast<int>(sy) * nk);

        float x0 = 0.0f;
        float x1 = 0.0f;
        float x2 = 0.0f;
        if (ni != 0) {
            const int ncls = axis_class_pcg(static_cast<std::uint32_t>(ni), nx) +
                             3 * (axis_class_pcg(static_cast<std::uint32_t>(nj), ny) +
                                  3 * axis_class_pcg(static_cast<std::uint32_t>(nk), nz));
            x0 = kPcgInvDiag[ncls][0] * ux[neighbor];
            x1 = kPcgInvDiag[ncls][1] * uy[neighbor];
            x2 = kPcgInvDiag[ncls][2] * uz[neighbor];
        }
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

__global__ void forward_smoothed_step_sa_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::size_t nodes,
    float omega,
    const float* __restrict__ input,
    float* __restrict__ output) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::size_t node = static_cast<std::size_t>(i + sx * (j + sy * k));
    if (i == 0U) {
        output[node] = 0.0f;
        output[nodes + node] = 0.0f;
        output[2U * nodes + node] = 0.0f;
        return;
    }

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    apply_gold_sparse_sa(i, j, k, nx, ny, nz,
                         input, input + nodes, input + 2U * nodes,
                         ax, ay, az);
    const int cls = axis_class_pcg(i, nx) +
                    3 * (axis_class_pcg(j, ny) + 3 * axis_class_pcg(k, nz));
    output[node] = fmaf(-omega * kPcgInvDiag[cls][0], ax, input[node]);
    output[nodes + node] = fmaf(-omega * kPcgInvDiag[cls][1], ay,
                                input[nodes + node]);
    output[2U * nodes + node] = fmaf(-omega * kPcgInvDiag[cls][2], az,
                                     input[2U * nodes + node]);
}

__global__ void transpose_smoothed_step_sa_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::size_t nodes,
    float omega,
    const float* __restrict__ input,
    float* __restrict__ output) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::size_t node = static_cast<std::size_t>(i + sx * (j + sy * k));
    if (i == 0U) {
        output[node] = 0.0f;
        output[nodes + node] = 0.0f;
        output[2U * nodes + node] = 0.0f;
        return;
    }

    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    apply_gold_sparse_scaled_input_sa(i, j, k, nx, ny, nz,
                                      input, input + nodes, input + 2U * nodes,
                                      ax, ay, az);
    output[node] = fmaf(-omega, ax, input[node]);
    output[nodes + node] = fmaf(-omega, ay, input[nodes + node]);
    output[2U * nodes + node] = fmaf(-omega, az, input[2U * nodes + node]);
}

void launch_p0(const StructuredHexMesh& mesh,
               std::size_t nodes,
               std::uint32_t aggregate_count,
               const std::uint32_t* aggregate_offsets,
               const std::uint32_t* aggregate_nodes,
               const DeviceAggregateSa* aggregates,
               const float* coordinates,
               const float* coarse,
               float* fine) {
    constexpr unsigned int threads = 32U;
    tentative_prolongation_aggregate_sa_kernel<<<aggregate_count, threads>>>(
        nodes, aggregate_offsets, aggregate_nodes, aggregates,
        coordinates, coarse, fine);
    check_cuda_pcg(cudaGetLastError(), "smoothed aggregation aggregate P0 launch");

    const std::uint32_t face_nodes = (mesh.ny + 1U) * (mesh.nz + 1U);
    constexpr unsigned int clamp_threads = 256U;
    const unsigned int clamp_blocks = (face_nodes + clamp_threads - 1U) / clamp_threads;
    zero_x0_face_pcg_kernel<<<clamp_blocks, clamp_threads>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, fine);
    check_cuda_pcg(cudaGetLastError(), "smoothed aggregation P0 clamp launch");
}

void launch_p0t(std::size_t nodes,
                std::uint32_t aggregate_count,
                const std::uint32_t* aggregate_offsets,
                const std::uint32_t* aggregate_nodes,
                const DeviceAggregateSa* aggregates,
                const float* coordinates,
                const float* fine,
                float* coarse) {
    constexpr unsigned int threads = 32U;
    tentative_restriction_aggregate_sa_kernel<<<aggregate_count, threads>>>(
        nodes, aggregate_offsets, aggregate_nodes, aggregates,
        coordinates, fine, coarse);
    check_cuda_pcg(cudaGetLastError(), "smoothed aggregation aggregate P0T launch");
}

void launch_forward_smooth(const StructuredHexMesh& mesh,
                           int block_y,
                           std::size_t nodes,
                           float omega,
                           const float* input,
                           float* output) {
    const dim3 block(32U, static_cast<unsigned int>(block_y), 1U);
    const dim3 grid((mesh.nx + 1U + block.x - 1U) / block.x,
                    (mesh.ny + 1U + block.y - 1U) / block.y,
                    mesh.nz + 1U);
    forward_smoothed_step_sa_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, omega, input, output);
    check_cuda_pcg(cudaGetLastError(), "smoothed aggregation fused forward launch");
}

void launch_transpose_smooth(const StructuredHexMesh& mesh,
                             int block_y,
                             std::size_t nodes,
                             float omega,
                             const float* input,
                             float* output) {
    const dim3 block(32U, static_cast<unsigned int>(block_y), 1U);
    const dim3 grid((mesh.nx + 1U + block.x - 1U) / block.x,
                    (mesh.ny + 1U + block.y - 1U) / block.y,
                    mesh.nz + 1U);
    transpose_smoothed_step_sa_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, omega, input, output);
    check_cuda_pcg(cudaGetLastError(), "smoothed aggregation fused transpose launch");
}

GpuSmoothedAggregationTiming summarize_fieldwise_min(
    const std::vector<GpuSmoothedAggregationTiming>& samples) {
    GpuSmoothedAggregationTiming out;
    out.p0_ms = std::numeric_limits<double>::infinity();
    out.fine_operator_ms = std::numeric_limits<double>::infinity();
    out.jacobi_ms = std::numeric_limits<double>::infinity();
    out.vector_update_ms = std::numeric_limits<double>::infinity();
    out.p0t_ms = std::numeric_limits<double>::infinity();
    out.total_ms = std::numeric_limits<double>::infinity();
    for (const auto& s : samples) {
        out.p0_ms = std::min(out.p0_ms, s.p0_ms);
        out.fine_operator_ms = std::min(out.fine_operator_ms, s.fine_operator_ms);
        out.jacobi_ms = std::min(out.jacobi_ms, s.jacobi_ms);
        out.vector_update_ms = std::min(out.vector_update_ms, s.vector_update_ms);
        out.p0t_ms = std::min(out.p0t_ms, s.p0t_ms);
        out.total_ms = std::min(out.total_ms, s.total_ms);
    }
    return out;
}

double median_scalar(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

GpuSmoothedAggregationTiming summarize_fieldwise_median(
    const std::vector<GpuSmoothedAggregationTiming>& samples) {
    std::vector<double> p0, center_a, forward, transpose, p0t, total;
    p0.reserve(samples.size());
    center_a.reserve(samples.size());
    forward.reserve(samples.size());
    transpose.reserve(samples.size());
    p0t.reserve(samples.size());
    total.reserve(samples.size());
    for (const auto& s : samples) {
        p0.push_back(s.p0_ms);
        center_a.push_back(s.fine_operator_ms);
        forward.push_back(s.jacobi_ms);
        transpose.push_back(s.vector_update_ms);
        p0t.push_back(s.p0t_ms);
        total.push_back(s.total_ms);
    }
    return {median_scalar(std::move(p0)),
            median_scalar(std::move(center_a)),
            median_scalar(std::move(forward)),
            median_scalar(std::move(transpose)),
            median_scalar(std::move(p0t)),
            median_scalar(std::move(total))};
}

float elapsed_event(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    check_cuda_pcg(cudaEventElapsedTime(&ms, a, b),
                   "cudaEventElapsedTime(smoothed aggregation)");
    return ms;
}

}  // namespace

struct GpuSmoothedAggregationContext::Impl {
    StructuredHexMesh mesh;
    Material material;
    double omega_value{0.0};
    int block_y{4};
    std::size_t nodes{0};
    std::size_t ndof{0};
    std::size_t coarse_dof_count{0};
    std::uint32_t aggregate_count{0};

    std::uint32_t* d_aggregate_offsets{nullptr};
    std::uint32_t* d_aggregate_nodes{nullptr};
    DeviceAggregateSa* d_aggregates{nullptr};
    float* d_coordinates{nullptr};
    float* d_coarse_x{nullptr};
    float* d_coarse_y{nullptr};
    float* d_f0{nullptr};
    float* d_f1{nullptr};

    std::size_t aggregation_metadata_bytes{0};
    std::size_t model_coordinate_bytes{0};
    std::size_t fine_workspace_bytes{0};
    std::size_t coarse_workspace_bytes{0};

    Impl(const StructuredHexMesh& mesh_in,
         const Material& material_in,
         const ElasticityAggregationCoarseSpace& space,
         double omega_in,
         int block_y_in)
        : mesh(mesh_in), material(material_in), omega_value(omega_in), block_y(block_y_in) {
        if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
            throw std::invalid_argument("GPU smoothed aggregation requires non-empty mesh");
        }
        if (!(omega_value > 0.0) || !std::isfinite(omega_value)) {
            throw std::invalid_argument("GPU smoothed aggregation omega must be finite and positive");
        }
        if (block_y <= 0 || block_y > 32 || 32 * block_y > 1024) {
            throw std::invalid_argument("GPU smoothed aggregation block_y is invalid");
        }
        nodes = static_cast<std::size_t>(mesh.node_count());
        ndof = static_cast<std::size_t>(mesh.dof_count());
        coarse_dof_count = space.coarse_dofs;
        if (space.graph.coordinates.size() != nodes ||
            space.aggregate_of_node.size() != nodes ||
            coarse_dof_count == 0U || space.aggregates.empty()) {
            throw std::invalid_argument("GPU smoothed aggregation coarse space/mesh mismatch");
        }
        if (space.aggregates.size() > std::numeric_limits<std::uint32_t>::max() ||
            coarse_dof_count > std::numeric_limits<std::uint32_t>::max() ||
            nodes > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("GPU smoothed aggregation currently requires 32-bit indexing");
        }
        aggregate_count = static_cast<std::uint32_t>(space.aggregates.size());

        std::vector<DeviceAggregateSa> host_aggregates(space.aggregates.size());
        for (std::size_t a = 0; a < space.aggregates.size(); ++a) {
            const auto& src = space.aggregates[a];
            if (src.rank == 0U || src.rank > 6U ||
                src.coarse_offset + src.rank > coarse_dof_count ||
                src.coarse_offset > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("GPU smoothed aggregation aggregate metadata is invalid");
            }
            auto& dst = host_aggregates[a];
            dst.coarse_offset = static_cast<std::uint32_t>(src.coarse_offset);
            dst.rank = static_cast<std::uint32_t>(src.rank);
            dst.centroid[0] = static_cast<float>(src.centroid[0]);
            dst.centroid[1] = static_cast<float>(src.centroid[1]);
            dst.centroid[2] = static_cast<float>(src.centroid[2]);
            dst.inv_scale = static_cast<float>(1.0 / src.coordinate_scale);
            for (std::size_t q = 0; q < 36U; ++q) {
                dst.transform[q] = static_cast<float>(src.rigid_transform[q]);
            }
        }

        std::vector<std::uint32_t> host_aggregate_offsets(
            static_cast<std::size_t>(aggregate_count) + 1U, 0U);
        std::size_t owned_nodes = 0U;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto a = space.aggregate_of_node[node];
            if (a >= aggregate_count) continue;
            ++host_aggregate_offsets[static_cast<std::size_t>(a) + 1U];
            ++owned_nodes;
        }
        for (std::size_t a = 0; a < aggregate_count; ++a) {
            host_aggregate_offsets[a + 1U] += host_aggregate_offsets[a];
        }
        if (owned_nodes != space.free_nodes ||
            host_aggregate_offsets.back() != owned_nodes) {
            throw std::runtime_error("GPU smoothed aggregation aggregate ownership count mismatch");
        }

        std::vector<std::uint32_t> host_aggregate_nodes(owned_nodes, 0U);
        auto cursor = host_aggregate_offsets;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto a = space.aggregate_of_node[node];
            if (a >= aggregate_count) continue;
            host_aggregate_nodes[cursor[a]++] = static_cast<std::uint32_t>(node);
        }
        for (std::size_t a = 0; a < aggregate_count; ++a) {
            const std::size_t count = host_aggregate_offsets[a + 1U] - host_aggregate_offsets[a];
            if (count == 0U) {
                throw std::runtime_error("GPU smoothed aggregation found empty aggregate");
            }
        }

        std::vector<float> host_coordinates(3U * nodes, 0.0f);
        for (std::size_t node = 0; node < nodes; ++node) {
            host_coordinates[node] = static_cast<float>(space.graph.coordinates[node][0]);
            host_coordinates[nodes + node] = static_cast<float>(space.graph.coordinates[node][1]);
            host_coordinates[2U * nodes + node] = static_cast<float>(space.graph.coordinates[node][2]);
        }

        const std::size_t aggregate_offset_bytes = host_aggregate_offsets.size() * sizeof(std::uint32_t);
        const std::size_t aggregate_node_bytes = host_aggregate_nodes.size() * sizeof(std::uint32_t);
        const std::size_t aggregate_bytes = host_aggregates.size() * sizeof(DeviceAggregateSa);
        aggregation_metadata_bytes = aggregate_offset_bytes + aggregate_node_bytes + aggregate_bytes;
        model_coordinate_bytes = host_coordinates.size() * sizeof(float);
        fine_workspace_bytes = 2U * ndof * sizeof(float);
        coarse_workspace_bytes = 2U * coarse_dof_count * sizeof(float);

        try {
            upload_pcg_stencil(mesh, material);
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_aggregate_offsets), aggregate_offset_bytes),
                           "cudaMalloc(SA aggregate offsets)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_aggregate_nodes), aggregate_node_bytes),
                           "cudaMalloc(SA aggregate nodes)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_aggregates), aggregate_bytes),
                           "cudaMalloc(SA aggregate metadata)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_coordinates), model_coordinate_bytes),
                           "cudaMalloc(SA coordinates)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_coarse_x), coarse_dof_count * sizeof(float)),
                           "cudaMalloc(SA coarse x)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_coarse_y), coarse_dof_count * sizeof(float)),
                           "cudaMalloc(SA coarse y)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_f0), ndof * sizeof(float)),
                           "cudaMalloc(SA fine f0)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_f1), ndof * sizeof(float)),
                           "cudaMalloc(SA fine f1)");

            check_cuda_pcg(cudaMemcpy(d_aggregate_offsets,
                                      host_aggregate_offsets.data(),
                                      aggregate_offset_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(SA aggregate offsets H2D)");
            check_cuda_pcg(cudaMemcpy(d_aggregate_nodes,
                                      host_aggregate_nodes.data(),
                                      aggregate_node_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(SA aggregate nodes H2D)");
            check_cuda_pcg(cudaMemcpy(d_aggregates,
                                      host_aggregates.data(),
                                      aggregate_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(SA aggregate metadata H2D)");
            check_cuda_pcg(cudaMemcpy(d_coordinates,
                                      host_coordinates.data(),
                                      model_coordinate_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(SA coordinates H2D)");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Impl() { cleanup(); }

    void cleanup() noexcept {
        if (d_aggregate_offsets) cudaFree(d_aggregate_offsets);
        if (d_aggregate_nodes) cudaFree(d_aggregate_nodes);
        if (d_aggregates) cudaFree(d_aggregates);
        if (d_coordinates) cudaFree(d_coordinates);
        if (d_coarse_x) cudaFree(d_coarse_x);
        if (d_coarse_y) cudaFree(d_coarse_y);
        if (d_f0) cudaFree(d_f0);
        if (d_f1) cudaFree(d_f1);
        d_aggregate_offsets = nullptr;
        d_aggregate_nodes = nullptr;
        d_aggregates = nullptr;
        d_coordinates = nullptr;
        d_coarse_x = nullptr;
        d_coarse_y = nullptr;
        d_f0 = d_f1 = nullptr;
    }

    void run_pipeline(std::size_t m,
                      std::vector<cudaEvent_t>* markers,
                      std::vector<TimedStage>* stages) {
        std::size_t marker_index = 0U;
        auto mark = [&](TimedStage stage) {
            if (markers && stages) {
                check_cuda_pcg(cudaEventRecord((*markers)[++marker_index]),
                               "cudaEventRecord(SA marker)");
                stages->push_back(stage);
            }
        };
        if (markers && stages) {
            stages->clear();
            marker_index = 0U;
            check_cuda_pcg(cudaEventRecord((*markers)[0]),
                           "cudaEventRecord(SA start)");
        }

        launch_p0(mesh, nodes, aggregate_count, d_aggregate_offsets,
                  d_aggregate_nodes, d_aggregates,
                  d_coordinates, d_coarse_x, d_f0);
        mark(TimedStage::P0);

        float* current = d_f0;
        float* other = d_f1;
        const float omega = static_cast<float>(omega_value);
        for (std::size_t step = 0; step < m; ++step) {
            launch_forward_smooth(mesh, block_y, nodes, omega, current, other);
            mark(TimedStage::ForwardSmooth);
            std::swap(current, other);
        }

        launch_pcg_matvec(mesh, block_y, nodes, current, other);
        mark(TimedStage::FineA);
        std::swap(current, other);

        for (std::size_t step = 0; step < m; ++step) {
            launch_transpose_smooth(mesh, block_y, nodes, omega, current, other);
            mark(TimedStage::TransposeSmooth);
            std::swap(current, other);
        }

        launch_p0t(nodes, aggregate_count, d_aggregate_offsets,
                   d_aggregate_nodes, d_aggregates,
                   d_coordinates, current, d_coarse_y);
        mark(TimedStage::P0T);
    }
};

GpuSmoothedAggregationContext::GpuSmoothedAggregationContext(
    const StructuredHexMesh& mesh,
    const Material& material,
    const ElasticityAggregationCoarseSpace& space,
    double omega,
    int block_y)
    : impl_(std::make_unique<Impl>(mesh, material, space, omega, block_y)) {}

GpuSmoothedAggregationContext::~GpuSmoothedAggregationContext() = default;
GpuSmoothedAggregationContext::GpuSmoothedAggregationContext(
    GpuSmoothedAggregationContext&&) noexcept = default;
GpuSmoothedAggregationContext& GpuSmoothedAggregationContext::operator=(
    GpuSmoothedAggregationContext&&) noexcept = default;

GpuSmoothedAggregationApplyResult GpuSmoothedAggregationContext::apply(
    const std::vector<float>& coarse_x,
    std::size_t transfer_smoothing_steps,
    int repeats) {
    if (!impl_) throw std::runtime_error("GPU smoothed aggregation context is empty");
    if (coarse_x.size() != impl_->coarse_dof_count) {
        throw std::invalid_argument("GPU smoothed aggregation coarse input size mismatch");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("GPU smoothed aggregation repeats must be positive");
    }
    if (transfer_smoothing_steps > 8U) {
        throw std::invalid_argument("GPU smoothed aggregation reference limits m to 8");
    }

    check_cuda_pcg(cudaMemcpy(impl_->d_coarse_x,
                              coarse_x.data(),
                              coarse_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(SA coarse x H2D)");

    impl_->run_pipeline(transfer_smoothing_steps, nullptr, nullptr);
    check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(SA warmup)");

    const std::size_t intervals = 2U * transfer_smoothing_steps + 3U;
    std::vector<cudaEvent_t> markers(intervals + 1U, nullptr);
    try {
        for (auto& event : markers) {
            check_cuda_pcg(cudaEventCreate(&event), "cudaEventCreate(SA marker)");
        }

        std::vector<GpuSmoothedAggregationTiming> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        std::vector<TimedStage> stages;
        stages.reserve(intervals);

        for (int repeat = 0; repeat < repeats; ++repeat) {
            impl_->run_pipeline(transfer_smoothing_steps, &markers, &stages);
            check_cuda_pcg(cudaEventSynchronize(markers.back()),
                           "cudaEventSynchronize(SA final marker)");
            if (stages.size() != intervals) {
                throw std::runtime_error("GPU smoothed aggregation timing marker mismatch");
            }

            GpuSmoothedAggregationTiming sample;
            for (std::size_t i = 0; i < intervals; ++i) {
                const double ms = static_cast<double>(elapsed_event(markers[i], markers[i + 1U]));
                switch (stages[i]) {
                    case TimedStage::P0: sample.p0_ms += ms; break;
                    case TimedStage::ForwardSmooth: sample.jacobi_ms += ms; break;
                    case TimedStage::FineA: sample.fine_operator_ms += ms; break;
                    case TimedStage::TransposeSmooth: sample.vector_update_ms += ms; break;
                    case TimedStage::P0T: sample.p0t_ms += ms; break;
                }
            }
            sample.total_ms = static_cast<double>(elapsed_event(markers.front(), markers.back()));
            samples.push_back(sample);
        }

        GpuSmoothedAggregationApplyResult result;
        result.coarse_y.resize(impl_->coarse_dof_count);
        check_cuda_pcg(cudaMemcpy(result.coarse_y.data(),
                                  impl_->d_coarse_y,
                                  result.coarse_y.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(SA coarse y D2H)");
        result.median_timing = summarize_fieldwise_median(samples);
        result.best_timing = summarize_fieldwise_min(samples);
        result.transfer_smoothing_steps = transfer_smoothing_steps;
        result.fine_operator_applies = 2U * transfer_smoothing_steps + 1U;
        result.fine_workspace_bytes = impl_->fine_workspace_bytes;
        result.coarse_workspace_bytes = impl_->coarse_workspace_bytes;
        result.aggregation_metadata_bytes = impl_->aggregation_metadata_bytes;
        result.model_coordinate_bytes = impl_->model_coordinate_bytes;
        result.device_bytes_total = impl_->fine_workspace_bytes +
                                    impl_->coarse_workspace_bytes +
                                    impl_->aggregation_metadata_bytes +
                                    impl_->model_coordinate_bytes;

        for (auto event : markers) cudaEventDestroy(event);
        return result;
    } catch (...) {
        for (auto event : markers) {
            if (event) cudaEventDestroy(event);
        }
        throw;
    }
}

std::size_t GpuSmoothedAggregationContext::fine_dofs() const noexcept {
    return impl_ ? impl_->ndof : 0U;
}

std::size_t GpuSmoothedAggregationContext::coarse_dofs() const noexcept {
    return impl_ ? impl_->coarse_dof_count : 0U;
}

double GpuSmoothedAggregationContext::omega() const noexcept {
    return impl_ ? impl_->omega_value : 0.0;
}

}  // namespace gfss
