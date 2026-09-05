// M5 productionization: persistent GPU implementation of the fine-level
// symmetric V-cycle shell. Reuse the validated GoldSparse stencil upload and
// matvec launchers without exporting another PCG entry point.
#define solve_pcg_cuda_gold_sparse_x0 solve_pcg_cuda_gold_sparse_x0_m5_fine_internal_unused
#include "gpu_pcg.cu"
#undef solve_pcg_cuda_gold_sparse_x0

#include "gfss/gpu_m5_fine_level.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gfss {
namespace {

constexpr double kM5Pi = 3.141592653589793238462643383279502884;
constexpr double kM5ChebyshevLowerFraction = 0.10;

struct DeviceAggregateM5 {
    std::uint32_t coarse_offset{0};
    std::uint32_t rank{0};
    float centroid[3]{};
    float inv_scale{1.0f};
    float transform[36]{};
};

static_assert(sizeof(DeviceAggregateM5) == 168,
              "unexpected M5 aggregate device layout");

__device__ __forceinline__ void m5_aggregate_basis_values(
    const DeviceAggregateM5* aggregate,
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

__global__ void m5_tentative_prolongation_kernel(
    std::size_t nodes,
    const std::uint32_t* __restrict__ aggregate_offsets,
    const std::uint32_t* __restrict__ aggregate_nodes,
    const DeviceAggregateM5* __restrict__ aggregates,
    const float* __restrict__ coordinates,
    const float* __restrict__ coarse,
    float* __restrict__ fine) {
    const std::uint32_t aggregate_id = blockIdx.x;
    const DeviceAggregateM5* aggregate = aggregates + aggregate_id;
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
            m5_aggregate_basis_values(aggregate, x, y, z, q, bx, by, bz);
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

__global__ void m5_tentative_restriction_kernel(
    std::size_t nodes,
    const std::uint32_t* __restrict__ aggregate_offsets,
    const std::uint32_t* __restrict__ aggregate_nodes,
    const DeviceAggregateM5* __restrict__ aggregates,
    const float* __restrict__ coordinates,
    const float* __restrict__ fine,
    float* __restrict__ coarse) {
    const std::uint32_t aggregate_id = blockIdx.x;
    const DeviceAggregateM5* aggregate = aggregates + aggregate_id;
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
            m5_aggregate_basis_values(aggregate, x, y, z, q, bx, by, bz);
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

__global__ void m5_chebyshev_update_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::size_t nodes,
    float weight,
    const float* __restrict__ rhs,
    const float* __restrict__ ax,
    float* __restrict__ x) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::size_t node = static_cast<std::size_t>(i + sx * (j + sy * k));
    if (i == 0U) {
        x[node] = 0.0f;
        x[nodes + node] = 0.0f;
        x[2U * nodes + node] = 0.0f;
        return;
    }
    const int cls = axis_class_pcg(i, nx) +
                    3 * (axis_class_pcg(j, ny) + 3 * axis_class_pcg(k, nz));
    x[node] = fmaf(weight * kPcgInvDiag[cls][0], rhs[node] - ax[node], x[node]);
    x[nodes + node] = fmaf(weight * kPcgInvDiag[cls][1],
                           rhs[nodes + node] - ax[nodes + node],
                           x[nodes + node]);
    x[2U * nodes + node] = fmaf(weight * kPcgInvDiag[cls][2],
                                rhs[2U * nodes + node] - ax[2U * nodes + node],
                                x[2U * nodes + node]);
}

__global__ void m5_residual_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::size_t nodes,
    const float* __restrict__ rhs,
    float* __restrict__ ax_inout) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::size_t node = static_cast<std::size_t>(i + sx * (j + sy * k));
    if (i == 0U) {
        ax_inout[node] = 0.0f;
        ax_inout[nodes + node] = 0.0f;
        ax_inout[2U * nodes + node] = 0.0f;
        return;
    }
    ax_inout[node] = rhs[node] - ax_inout[node];
    ax_inout[nodes + node] = rhs[nodes + node] - ax_inout[nodes + node];
    ax_inout[2U * nodes + node] = rhs[2U * nodes + node] - ax_inout[2U * nodes + node];
}

__global__ void m5_scale_inverse_diagonal_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::size_t nodes,
    const float* __restrict__ input,
    float* __restrict__ scaled) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::size_t node = static_cast<std::size_t>(i + sx * (j + sy * k));
    if (i == 0U) {
        scaled[node] = 0.0f;
        scaled[nodes + node] = 0.0f;
        scaled[2U * nodes + node] = 0.0f;
        return;
    }
    const int cls = axis_class_pcg(i, nx) +
                    3 * (axis_class_pcg(j, ny) + 3 * axis_class_pcg(k, nz));
    scaled[node] = kPcgInvDiag[cls][0] * input[node];
    scaled[nodes + node] = kPcgInvDiag[cls][1] * input[nodes + node];
    scaled[2U * nodes + node] = kPcgInvDiag[cls][2] * input[2U * nodes + node];
}

__global__ void m5_transpose_update_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::size_t nodes,
    float omega,
    const float* __restrict__ a_scaled,
    float* __restrict__ work) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::size_t node = static_cast<std::size_t>(i + sx * (j + sy * k));
    if (i == 0U) {
        work[node] = 0.0f;
        work[nodes + node] = 0.0f;
        work[2U * nodes + node] = 0.0f;
        return;
    }
    work[node] = fmaf(-omega, a_scaled[node], work[node]);
    work[nodes + node] = fmaf(-omega, a_scaled[nodes + node], work[nodes + node]);
    work[2U * nodes + node] = fmaf(-omega, a_scaled[2U * nodes + node],
                                   work[2U * nodes + node]);
}

__global__ void m5_forward_transfer_update_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::size_t nodes,
    float omega,
    const float* __restrict__ ax,
    float* __restrict__ fine) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::size_t node = static_cast<std::size_t>(i + sx * (j + sy * k));
    if (i == 0U) {
        fine[node] = 0.0f;
        fine[nodes + node] = 0.0f;
        fine[2U * nodes + node] = 0.0f;
        return;
    }
    const int cls = axis_class_pcg(i, nx) +
                    3 * (axis_class_pcg(j, ny) + 3 * axis_class_pcg(k, nz));
    fine[node] = fmaf(-omega * kPcgInvDiag[cls][0], ax[node], fine[node]);
    fine[nodes + node] = fmaf(-omega * kPcgInvDiag[cls][1],
                              ax[nodes + node], fine[nodes + node]);
    fine[2U * nodes + node] = fmaf(-omega * kPcgInvDiag[cls][2],
                                   ax[2U * nodes + node], fine[2U * nodes + node]);
}

__global__ void m5_add_correction_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::size_t nodes,
    const float* __restrict__ correction,
    float* __restrict__ x) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;
    if (i > nx || j > ny || k > nz) return;

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::size_t node = static_cast<std::size_t>(i + sx * (j + sy * k));
    if (i == 0U) {
        x[node] = 0.0f;
        x[nodes + node] = 0.0f;
        x[2U * nodes + node] = 0.0f;
        return;
    }
    x[node] += correction[node];
    x[nodes + node] += correction[nodes + node];
    x[2U * nodes + node] += correction[2U * nodes + node];
}

void m5_launch_vector_kernel_geometry(const StructuredHexMesh& mesh,
                                       dim3& grid,
                                       dim3& block,
                                       int block_y) {
    block = dim3(32U, static_cast<unsigned int>(block_y), 1U);
    grid = dim3((mesh.nx + 1U + block.x - 1U) / block.x,
                (mesh.ny + 1U + block.y - 1U) / block.y,
                mesh.nz + 1U);
}

void m5_launch_chebyshev_update(const StructuredHexMesh& mesh,
                                int block_y,
                                std::size_t nodes,
                                float weight,
                                const float* rhs,
                                const float* ax,
                                float* x) {
    dim3 grid, block;
    m5_launch_vector_kernel_geometry(mesh, grid, block, block_y);
    m5_chebyshev_update_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, weight, rhs, ax, x);
    check_cuda_pcg(cudaGetLastError(), "M5 Chebyshev update launch");
}

void m5_launch_residual(const StructuredHexMesh& mesh,
                        int block_y,
                        std::size_t nodes,
                        const float* rhs,
                        float* ax_inout) {
    dim3 grid, block;
    m5_launch_vector_kernel_geometry(mesh, grid, block, block_y);
    m5_residual_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, rhs, ax_inout);
    check_cuda_pcg(cudaGetLastError(), "M5 residual update launch");
}

void m5_launch_inverse_scale(const StructuredHexMesh& mesh,
                             int block_y,
                             std::size_t nodes,
                             const float* input,
                             float* scaled) {
    dim3 grid, block;
    m5_launch_vector_kernel_geometry(mesh, grid, block, block_y);
    m5_scale_inverse_diagonal_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, input, scaled);
    check_cuda_pcg(cudaGetLastError(), "M5 transpose inverse-diagonal scale launch");
}

void m5_launch_transpose_update(const StructuredHexMesh& mesh,
                                int block_y,
                                std::size_t nodes,
                                float omega,
                                const float* a_scaled,
                                float* work) {
    dim3 grid, block;
    m5_launch_vector_kernel_geometry(mesh, grid, block, block_y);
    m5_transpose_update_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, omega, a_scaled, work);
    check_cuda_pcg(cudaGetLastError(), "M5 transpose update launch");
}

void m5_launch_forward_transfer_update(const StructuredHexMesh& mesh,
                                       int block_y,
                                       std::size_t nodes,
                                       float omega,
                                       const float* ax,
                                       float* fine) {
    dim3 grid, block;
    m5_launch_vector_kernel_geometry(mesh, grid, block, block_y);
    m5_forward_transfer_update_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, omega, ax, fine);
    check_cuda_pcg(cudaGetLastError(), "M5 forward transfer update launch");
}

void m5_launch_add_correction(const StructuredHexMesh& mesh,
                              int block_y,
                              std::size_t nodes,
                              const float* correction,
                              float* x) {
    dim3 grid, block;
    m5_launch_vector_kernel_geometry(mesh, grid, block, block_y);
    m5_add_correction_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, correction, x);
    check_cuda_pcg(cudaGetLastError(), "M5 fine correction add launch");
}

void m5_launch_p0(const StructuredHexMesh& mesh,
                   std::size_t nodes,
                   std::uint32_t aggregate_count,
                   const std::uint32_t* aggregate_offsets,
                   const std::uint32_t* aggregate_nodes,
                   const DeviceAggregateM5* aggregates,
                   const float* coordinates,
                   const float* coarse,
                   float* fine) {
    constexpr unsigned int threads = 32U;
    m5_tentative_prolongation_kernel<<<aggregate_count, threads>>>(
        nodes, aggregate_offsets, aggregate_nodes, aggregates,
        coordinates, coarse, fine);
    check_cuda_pcg(cudaGetLastError(), "M5 tentative P0 launch");

    const std::uint32_t face_nodes = (mesh.ny + 1U) * (mesh.nz + 1U);
    constexpr unsigned int clamp_threads = 256U;
    const unsigned int clamp_blocks = (face_nodes + clamp_threads - 1U) / clamp_threads;
    zero_x0_face_pcg_kernel<<<clamp_blocks, clamp_threads>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, fine);
    check_cuda_pcg(cudaGetLastError(), "M5 tentative P0 clamp launch");
}

void m5_launch_p0t(std::size_t nodes,
                    std::uint32_t aggregate_count,
                    const std::uint32_t* aggregate_offsets,
                    const std::uint32_t* aggregate_nodes,
                    const DeviceAggregateM5* aggregates,
                    const float* coordinates,
                    const float* fine,
                    float* coarse) {
    constexpr unsigned int threads = 32U;
    m5_tentative_restriction_kernel<<<aggregate_count, threads>>>(
        nodes, aggregate_offsets, aggregate_nodes, aggregates,
        coordinates, fine, coarse);
    check_cuda_pcg(cudaGetLastError(), "M5 tentative P0T launch");
}

double m5_median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

GpuM5FinePreRestrictTiming m5_summarize_median(
    const std::vector<GpuM5FinePreRestrictTiming>& samples) {
    std::vector<double> zero, smooth, residual, transpose, p0t, total;
    zero.reserve(samples.size()); smooth.reserve(samples.size());
    residual.reserve(samples.size()); transpose.reserve(samples.size());
    p0t.reserve(samples.size()); total.reserve(samples.size());
    for (const auto& s : samples) {
        zero.push_back(s.zero_ms);
        smooth.push_back(s.pre_smooth_ms);
        residual.push_back(s.residual_ms);
        transpose.push_back(s.transfer_transpose_ms);
        p0t.push_back(s.p0t_ms);
        total.push_back(s.total_ms);
    }
    GpuM5FinePreRestrictTiming out;
    out.zero_ms = m5_median(std::move(zero));
    out.pre_smooth_ms = m5_median(std::move(smooth));
    out.residual_ms = m5_median(std::move(residual));
    out.transfer_transpose_ms = m5_median(std::move(transpose));
    out.p0t_ms = m5_median(std::move(p0t));
    out.total_ms = m5_median(std::move(total));
    return out;
}

GpuM5FinePreRestrictTiming m5_summarize_best(
    const std::vector<GpuM5FinePreRestrictTiming>& samples) {
    GpuM5FinePreRestrictTiming out;
    out.zero_ms = out.pre_smooth_ms = out.residual_ms =
        out.transfer_transpose_ms = out.p0t_ms = out.total_ms =
            std::numeric_limits<double>::infinity();
    for (const auto& s : samples) {
        out.zero_ms = std::min(out.zero_ms, s.zero_ms);
        out.pre_smooth_ms = std::min(out.pre_smooth_ms, s.pre_smooth_ms);
        out.residual_ms = std::min(out.residual_ms, s.residual_ms);
        out.transfer_transpose_ms = std::min(out.transfer_transpose_ms, s.transfer_transpose_ms);
        out.p0t_ms = std::min(out.p0t_ms, s.p0t_ms);
        out.total_ms = std::min(out.total_ms, s.total_ms);
    }
    return out;
}

GpuM5FineFullShellTiming m5_summarize_full_median(
    const std::vector<GpuM5FineFullShellTiming>& samples) {
    std::vector<double> zero, pre, residual, transpose, p0t, p0, forward, correction, post, total;
    zero.reserve(samples.size()); pre.reserve(samples.size()); residual.reserve(samples.size());
    transpose.reserve(samples.size()); p0t.reserve(samples.size()); p0.reserve(samples.size());
    forward.reserve(samples.size()); correction.reserve(samples.size()); post.reserve(samples.size());
    total.reserve(samples.size());
    for (const auto& s : samples) {
        zero.push_back(s.zero_ms);
        pre.push_back(s.pre_smooth_ms);
        residual.push_back(s.residual_ms);
        transpose.push_back(s.transfer_transpose_ms);
        p0t.push_back(s.p0t_ms);
        p0.push_back(s.p0_ms);
        forward.push_back(s.transfer_forward_ms);
        correction.push_back(s.correction_ms);
        post.push_back(s.post_smooth_ms);
        total.push_back(s.total_ms);
    }
    GpuM5FineFullShellTiming out;
    out.zero_ms = m5_median(std::move(zero));
    out.pre_smooth_ms = m5_median(std::move(pre));
    out.residual_ms = m5_median(std::move(residual));
    out.transfer_transpose_ms = m5_median(std::move(transpose));
    out.p0t_ms = m5_median(std::move(p0t));
    out.p0_ms = m5_median(std::move(p0));
    out.transfer_forward_ms = m5_median(std::move(forward));
    out.correction_ms = m5_median(std::move(correction));
    out.post_smooth_ms = m5_median(std::move(post));
    out.total_ms = m5_median(std::move(total));
    return out;
}

GpuM5FineFullShellTiming m5_summarize_full_best(
    const std::vector<GpuM5FineFullShellTiming>& samples) {
    GpuM5FineFullShellTiming out;
    out.zero_ms = out.pre_smooth_ms = out.residual_ms =
        out.transfer_transpose_ms = out.p0t_ms = out.p0_ms =
        out.transfer_forward_ms = out.correction_ms = out.post_smooth_ms =
        out.total_ms = std::numeric_limits<double>::infinity();
    for (const auto& s : samples) {
        out.zero_ms = std::min(out.zero_ms, s.zero_ms);
        out.pre_smooth_ms = std::min(out.pre_smooth_ms, s.pre_smooth_ms);
        out.residual_ms = std::min(out.residual_ms, s.residual_ms);
        out.transfer_transpose_ms = std::min(out.transfer_transpose_ms, s.transfer_transpose_ms);
        out.p0t_ms = std::min(out.p0t_ms, s.p0t_ms);
        out.p0_ms = std::min(out.p0_ms, s.p0_ms);
        out.transfer_forward_ms = std::min(out.transfer_forward_ms, s.transfer_forward_ms);
        out.correction_ms = std::min(out.correction_ms, s.correction_ms);
        out.post_smooth_ms = std::min(out.post_smooth_ms, s.post_smooth_ms);
        out.total_ms = std::min(out.total_ms, s.total_ms);
    }
    return out;
}

float m5_elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    check_cuda_pcg(cudaEventElapsedTime(&ms, a, b), "M5 cudaEventElapsedTime");
    return ms;
}

}  // namespace

struct GpuM5FineLevelContext::Impl {
    StructuredHexMesh mesh;
    Material material;
    double transfer_omega_value{0.0};
    double smoother_lambda_max_value{0.0};
    int block_y{4};
    std::size_t nodes{0U};
    std::size_t ndof{0U};
    std::size_t coarse_dof_count{0U};
    std::uint32_t aggregate_count{0U};

    std::uint32_t* d_aggregate_offsets{nullptr};
    std::uint32_t* d_aggregate_nodes{nullptr};
    DeviceAggregateM5* d_aggregates{nullptr};
    float* d_coordinates{nullptr};
    float* d_rhs{nullptr};
    float* d_x{nullptr};
    float* d_work0{nullptr};
    float* d_work1{nullptr};
    float* d_work2{nullptr};
    float* d_coarse{nullptr};
    float* d_coarse_correction{nullptr};

    std::size_t aggregation_metadata_bytes{0U};
    std::size_t model_coordinate_bytes{0U};
    std::size_t fine_vector_bytes{0U};
    std::size_t coarse_vector_bytes{0U};

    Impl(const StructuredHexMesh& mesh_in,
         const Material& material_in,
         const ElasticityAggregationCoarseSpace& space,
         double transfer_omega_in,
         double smoother_lambda_max_in,
         int block_y_in)
        : mesh(mesh_in), material(material_in),
          transfer_omega_value(transfer_omega_in),
          smoother_lambda_max_value(smoother_lambda_max_in),
          block_y(block_y_in) {
        if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
            throw std::invalid_argument("M5 GPU fine-level context requires non-empty mesh");
        }
        if (!(transfer_omega_value > 0.0) || !std::isfinite(transfer_omega_value) ||
            !(smoother_lambda_max_value > 0.0) || !std::isfinite(smoother_lambda_max_value)) {
            throw std::invalid_argument("M5 GPU fine-level spectral parameters invalid");
        }
        if (block_y <= 0 || block_y > 32 || 32 * block_y > 1024) {
            throw std::invalid_argument("M5 GPU fine-level block_y invalid");
        }

        nodes = static_cast<std::size_t>(mesh.node_count());
        ndof = static_cast<std::size_t>(mesh.dof_count());
        coarse_dof_count = space.coarse_dofs;
        if (space.graph.coordinates.size() != nodes ||
            space.aggregate_of_node.size() != nodes ||
            coarse_dof_count == 0U || space.aggregates.empty()) {
            throw std::invalid_argument("M5 GPU fine-level coarse space/mesh mismatch");
        }
        if (space.aggregates.size() > std::numeric_limits<std::uint32_t>::max() ||
            coarse_dof_count > std::numeric_limits<std::uint32_t>::max() ||
            nodes > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("M5 GPU fine-level context requires 32-bit indexing");
        }
        aggregate_count = static_cast<std::uint32_t>(space.aggregates.size());

        std::vector<DeviceAggregateM5> host_aggregates(space.aggregates.size());
        for (std::size_t a = 0; a < space.aggregates.size(); ++a) {
            const auto& src = space.aggregates[a];
            if (src.rank == 0U || src.rank > 6U ||
                src.coarse_offset + src.rank > coarse_dof_count ||
                src.coarse_offset > std::numeric_limits<std::uint32_t>::max() ||
                !(src.coordinate_scale > 0.0)) {
                throw std::invalid_argument("M5 GPU aggregate metadata invalid");
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

        std::vector<std::uint32_t> host_offsets(
            static_cast<std::size_t>(aggregate_count) + 1U, 0U);
        std::size_t owned_nodes = 0U;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto a = space.aggregate_of_node[node];
            if (a >= aggregate_count) continue;
            ++host_offsets[static_cast<std::size_t>(a) + 1U];
            ++owned_nodes;
        }
        for (std::size_t a = 0; a < aggregate_count; ++a) {
            host_offsets[a + 1U] += host_offsets[a];
        }
        if (owned_nodes != space.free_nodes || host_offsets.back() != owned_nodes) {
            throw std::runtime_error("M5 GPU aggregate ownership mismatch");
        }
        std::vector<std::uint32_t> host_nodes(owned_nodes, 0U);
        auto cursor = host_offsets;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto a = space.aggregate_of_node[node];
            if (a >= aggregate_count) continue;
            host_nodes[cursor[a]++] = static_cast<std::uint32_t>(node);
        }

        std::vector<float> host_coordinates(3U * nodes, 0.0f);
        for (std::size_t node = 0; node < nodes; ++node) {
            host_coordinates[node] = static_cast<float>(space.graph.coordinates[node][0]);
            host_coordinates[nodes + node] = static_cast<float>(space.graph.coordinates[node][1]);
            host_coordinates[2U * nodes + node] = static_cast<float>(space.graph.coordinates[node][2]);
        }

        const std::size_t offset_bytes = host_offsets.size() * sizeof(std::uint32_t);
        const std::size_t node_bytes = host_nodes.size() * sizeof(std::uint32_t);
        const std::size_t aggregate_bytes = host_aggregates.size() * sizeof(DeviceAggregateM5);
        aggregation_metadata_bytes = offset_bytes + node_bytes + aggregate_bytes;
        model_coordinate_bytes = host_coordinates.size() * sizeof(float);
        fine_vector_bytes = 5U * ndof * sizeof(float);
        coarse_vector_bytes = 2U * coarse_dof_count * sizeof(float);
        const std::size_t one_fine_bytes = ndof * sizeof(float);
        const std::size_t one_coarse_bytes = coarse_dof_count * sizeof(float);

        try {
            upload_pcg_stencil(mesh, material);
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_aggregate_offsets), offset_bytes),
                           "cudaMalloc(M5 aggregate offsets)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_aggregate_nodes), node_bytes),
                           "cudaMalloc(M5 aggregate nodes)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_aggregates), aggregate_bytes),
                           "cudaMalloc(M5 aggregate metadata)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_coordinates), model_coordinate_bytes),
                           "cudaMalloc(M5 coordinates)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_rhs), one_fine_bytes),
                           "cudaMalloc(M5 rhs)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_x), one_fine_bytes),
                           "cudaMalloc(M5 x)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_work0), one_fine_bytes),
                           "cudaMalloc(M5 work0)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_work1), one_fine_bytes),
                           "cudaMalloc(M5 work1)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_work2), one_fine_bytes),
                           "cudaMalloc(M5 work2)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_coarse), one_coarse_bytes),
                           "cudaMalloc(M5 coarse residual)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_coarse_correction), one_coarse_bytes),
                           "cudaMalloc(M5 coarse correction)");

            check_cuda_pcg(cudaMemcpy(d_aggregate_offsets, host_offsets.data(), offset_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(M5 aggregate offsets H2D)");
            check_cuda_pcg(cudaMemcpy(d_aggregate_nodes, host_nodes.data(), node_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(M5 aggregate nodes H2D)");
            check_cuda_pcg(cudaMemcpy(d_aggregates, host_aggregates.data(), aggregate_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(M5 aggregate metadata H2D)");
            check_cuda_pcg(cudaMemcpy(d_coordinates, host_coordinates.data(), model_coordinate_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(M5 coordinates H2D)");
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
        if (d_rhs) cudaFree(d_rhs);
        if (d_x) cudaFree(d_x);
        if (d_work0) cudaFree(d_work0);
        if (d_work1) cudaFree(d_work1);
        if (d_work2) cudaFree(d_work2);
        if (d_coarse) cudaFree(d_coarse);
        if (d_coarse_correction) cudaFree(d_coarse_correction);
        d_aggregate_offsets = nullptr;
        d_aggregate_nodes = nullptr;
        d_aggregates = nullptr;
        d_coordinates = nullptr;
        d_rhs = d_x = d_work0 = d_work1 = d_work2 = nullptr;
        d_coarse = d_coarse_correction = nullptr;
    }

    std::vector<float> chebyshev_weights(std::size_t degree) const {
        std::vector<float> weights(degree, 0.0f);
        const double lambda_low = kM5ChebyshevLowerFraction * smoother_lambda_max_value;
        const double theta = 0.5 * (smoother_lambda_max_value + lambda_low);
        const double delta = 0.5 * (smoother_lambda_max_value - lambda_low);
        for (std::size_t k = 0; k < degree; ++k) {
            const double angle = kM5Pi * (2.0 * static_cast<double>(k) + 1.0) /
                                 (2.0 * static_cast<double>(degree));
            const double root = theta + delta * std::cos(angle);
            if (!(root > 0.0) || !std::isfinite(root)) {
                throw std::runtime_error("M5 GPU Chebyshev root invalid");
            }
            weights[k] = static_cast<float>(1.0 / root);
        }
        return weights;
    }

    void run_pre_once(const std::vector<float>& weights,
                      std::size_t transfer_steps,
                      const std::array<cudaEvent_t, 6>* events) {
        const std::size_t bytes = ndof * sizeof(float);
        if (events) check_cuda_pcg(cudaEventRecord((*events)[0]), "M5 record start");
        check_cuda_pcg(cudaMemsetAsync(d_x, 0, bytes), "cudaMemsetAsync(M5 x=0)");
        if (events) check_cuda_pcg(cudaEventRecord((*events)[1]), "M5 record zero");

        for (const float weight : weights) {
            launch_pcg_matvec(mesh, block_y, nodes, d_x, d_work0);
            m5_launch_chebyshev_update(mesh, block_y, nodes, weight, d_rhs, d_work0, d_x);
        }
        if (events) check_cuda_pcg(cudaEventRecord((*events)[2]), "M5 record pre-smooth");

        launch_pcg_matvec(mesh, block_y, nodes, d_x, d_work0);
        m5_launch_residual(mesh, block_y, nodes, d_rhs, d_work0);
        if (events) check_cuda_pcg(cudaEventRecord((*events)[3]), "M5 record residual");

        const float omega = static_cast<float>(transfer_omega_value);
        for (std::size_t step = 0; step < transfer_steps; ++step) {
            m5_launch_inverse_scale(mesh, block_y, nodes, d_work0, d_work1);
            launch_pcg_matvec(mesh, block_y, nodes, d_work1, d_work2);
            m5_launch_transpose_update(mesh, block_y, nodes, omega, d_work2, d_work0);
        }
        if (events) check_cuda_pcg(cudaEventRecord((*events)[4]), "M5 record transpose transfer");

        m5_launch_p0t(nodes, aggregate_count, d_aggregate_offsets, d_aggregate_nodes,
                       d_aggregates, d_coordinates, d_work0, d_coarse);
        if (events) check_cuda_pcg(cudaEventRecord((*events)[5]), "M5 record P0T");
    }

    void run_full_shell_once(const std::vector<float>& weights,
                             std::size_t transfer_steps,
                             const std::array<cudaEvent_t, 10>* events) {
        const std::size_t bytes = ndof * sizeof(float);
        if (events) check_cuda_pcg(cudaEventRecord((*events)[0]), "M5 full record start");
        check_cuda_pcg(cudaMemsetAsync(d_x, 0, bytes), "cudaMemsetAsync(M5 full x=0)");
        if (events) check_cuda_pcg(cudaEventRecord((*events)[1]), "M5 full record zero");

        for (const float weight : weights) {
            launch_pcg_matvec(mesh, block_y, nodes, d_x, d_work0);
            m5_launch_chebyshev_update(mesh, block_y, nodes, weight, d_rhs, d_work0, d_x);
        }
        if (events) check_cuda_pcg(cudaEventRecord((*events)[2]), "M5 full record pre-smooth");

        launch_pcg_matvec(mesh, block_y, nodes, d_x, d_work0);
        m5_launch_residual(mesh, block_y, nodes, d_rhs, d_work0);
        if (events) check_cuda_pcg(cudaEventRecord((*events)[3]), "M5 full record residual");

        const float omega = static_cast<float>(transfer_omega_value);
        for (std::size_t step = 0; step < transfer_steps; ++step) {
            m5_launch_inverse_scale(mesh, block_y, nodes, d_work0, d_work1);
            launch_pcg_matvec(mesh, block_y, nodes, d_work1, d_work2);
            m5_launch_transpose_update(mesh, block_y, nodes, omega, d_work2, d_work0);
        }
        if (events) check_cuda_pcg(cudaEventRecord((*events)[4]), "M5 full record transpose");

        m5_launch_p0t(nodes, aggregate_count, d_aggregate_offsets, d_aggregate_nodes,
                       d_aggregates, d_coordinates, d_work0, d_coarse);
        if (events) check_cuda_pcg(cudaEventRecord((*events)[5]), "M5 full record P0T");

        m5_launch_p0(mesh, nodes, aggregate_count, d_aggregate_offsets, d_aggregate_nodes,
                      d_aggregates, d_coordinates, d_coarse_correction, d_work0);
        if (events) check_cuda_pcg(cudaEventRecord((*events)[6]), "M5 full record P0");

        for (std::size_t step = 0; step < transfer_steps; ++step) {
            launch_pcg_matvec(mesh, block_y, nodes, d_work0, d_work1);
            m5_launch_forward_transfer_update(mesh, block_y, nodes, omega, d_work1, d_work0);
        }
        if (events) check_cuda_pcg(cudaEventRecord((*events)[7]), "M5 full record forward transfer");

        m5_launch_add_correction(mesh, block_y, nodes, d_work0, d_x);
        if (events) check_cuda_pcg(cudaEventRecord((*events)[8]), "M5 full record correction");

        for (const float weight : weights) {
            launch_pcg_matvec(mesh, block_y, nodes, d_x, d_work0);
            m5_launch_chebyshev_update(mesh, block_y, nodes, weight, d_rhs, d_work0, d_x);
        }
        if (events) check_cuda_pcg(cudaEventRecord((*events)[9]), "M5 full record post-smooth");
    }
};

GpuM5FineLevelContext::GpuM5FineLevelContext(
    const StructuredHexMesh& mesh,
    const Material& material,
    const ElasticityAggregationCoarseSpace& space,
    double transfer_omega,
    double smoother_lambda_max,
    int block_y)
    : impl_(std::make_unique<Impl>(
          mesh, material, space, transfer_omega, smoother_lambda_max, block_y)) {}

GpuM5FineLevelContext::~GpuM5FineLevelContext() = default;
GpuM5FineLevelContext::GpuM5FineLevelContext(GpuM5FineLevelContext&&) noexcept = default;
GpuM5FineLevelContext& GpuM5FineLevelContext::operator=(GpuM5FineLevelContext&&) noexcept = default;

GpuM5FinePreRestrictResult GpuM5FineLevelContext::pre_smooth_restrict(
    const std::vector<float>& rhs_aos,
    std::size_t smoother_degree,
    std::size_t transfer_smoothing_steps,
    int repeats) {
    if (!impl_) throw std::runtime_error("M5 GPU fine-level context is empty");
    if (rhs_aos.size() != impl_->ndof) {
        throw std::invalid_argument("M5 GPU fine-level RHS size mismatch");
    }
    if (smoother_degree == 0U || smoother_degree > 32U) {
        throw std::invalid_argument("M5 GPU fine-level smoother degree must be in [1,32]");
    }
    if (transfer_smoothing_steps > 8U) {
        throw std::invalid_argument("M5 GPU fine-level transfer smoothing limited to 8");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("M5 GPU fine-level repeats must be positive");
    }

    std::vector<float> rhs_soa(impl_->ndof, 0.0f);
    for (std::size_t node = 0; node < impl_->nodes; ++node) {
        rhs_soa[node] = rhs_aos[3U * node + 0U];
        rhs_soa[impl_->nodes + node] = rhs_aos[3U * node + 1U];
        rhs_soa[2U * impl_->nodes + node] = rhs_aos[3U * node + 2U];
    }
    check_cuda_pcg(cudaMemcpy(impl_->d_rhs, rhs_soa.data(), impl_->ndof * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(M5 RHS H2D)");

    const auto weights = impl_->chebyshev_weights(smoother_degree);
    impl_->run_pre_once(weights, transfer_smoothing_steps, nullptr);
    check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(M5 warmup)");

    std::array<cudaEvent_t, 6> events{};
    try {
        for (auto& event : events) {
            check_cuda_pcg(cudaEventCreate(&event), "cudaEventCreate(M5 fine slice)");
        }
        std::vector<GpuM5FinePreRestrictTiming> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            impl_->run_pre_once(weights, transfer_smoothing_steps, &events);
            check_cuda_pcg(cudaEventSynchronize(events[5]), "cudaEventSynchronize(M5 fine slice)");
            GpuM5FinePreRestrictTiming sample;
            sample.zero_ms = static_cast<double>(m5_elapsed(events[0], events[1]));
            sample.pre_smooth_ms = static_cast<double>(m5_elapsed(events[1], events[2]));
            sample.residual_ms = static_cast<double>(m5_elapsed(events[2], events[3]));
            sample.transfer_transpose_ms = static_cast<double>(m5_elapsed(events[3], events[4]));
            sample.p0t_ms = static_cast<double>(m5_elapsed(events[4], events[5]));
            sample.total_ms = static_cast<double>(m5_elapsed(events[0], events[5]));
            samples.push_back(sample);
        }

        GpuM5FinePreRestrictResult result;
        result.coarse_residual.resize(impl_->coarse_dof_count, 0.0f);
        check_cuda_pcg(cudaMemcpy(result.coarse_residual.data(), impl_->d_coarse,
                                  impl_->coarse_dof_count * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 coarse residual D2H)");
        result.median_timing = m5_summarize_median(samples);
        result.best_timing = m5_summarize_best(samples);
        result.smoother_degree = smoother_degree;
        result.transfer_smoothing_steps = transfer_smoothing_steps;
        result.fine_operator_applies = smoother_degree + 1U + transfer_smoothing_steps;
        result.fine_vector_bytes = impl_->fine_vector_bytes;
        result.coarse_vector_bytes = impl_->coarse_vector_bytes;
        result.aggregation_metadata_bytes = impl_->aggregation_metadata_bytes;
        result.model_coordinate_bytes = impl_->model_coordinate_bytes;
        result.device_bytes_total = impl_->fine_vector_bytes + impl_->coarse_vector_bytes +
                                    impl_->aggregation_metadata_bytes +
                                    impl_->model_coordinate_bytes;
        for (auto event : events) cudaEventDestroy(event);
        return result;
    } catch (...) {
        for (auto event : events) {
            if (event) cudaEventDestroy(event);
        }
        throw;
    }
}

GpuM5FineFullShellResult GpuM5FineLevelContext::full_shell(
    const std::vector<float>& rhs_aos,
    const std::vector<float>& coarse_correction,
    std::size_t smoother_degree,
    std::size_t transfer_smoothing_steps,
    int repeats) {
    if (!impl_) throw std::runtime_error("M5 GPU fine-level context is empty");
    if (rhs_aos.size() != impl_->ndof) {
        throw std::invalid_argument("M5 GPU full shell RHS size mismatch");
    }
    if (coarse_correction.size() != impl_->coarse_dof_count) {
        throw std::invalid_argument("M5 GPU full shell coarse correction size mismatch");
    }
    if (smoother_degree == 0U || smoother_degree > 32U) {
        throw std::invalid_argument("M5 GPU full shell smoother degree must be in [1,32]");
    }
    if (transfer_smoothing_steps > 8U) {
        throw std::invalid_argument("M5 GPU full shell transfer smoothing limited to 8");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("M5 GPU full shell repeats must be positive");
    }

    std::vector<float> rhs_soa(impl_->ndof, 0.0f);
    for (std::size_t node = 0; node < impl_->nodes; ++node) {
        rhs_soa[node] = rhs_aos[3U * node + 0U];
        rhs_soa[impl_->nodes + node] = rhs_aos[3U * node + 1U];
        rhs_soa[2U * impl_->nodes + node] = rhs_aos[3U * node + 2U];
    }
    check_cuda_pcg(cudaMemcpy(impl_->d_rhs, rhs_soa.data(), impl_->ndof * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(M5 full RHS H2D)");
    check_cuda_pcg(cudaMemcpy(impl_->d_coarse_correction, coarse_correction.data(),
                              impl_->coarse_dof_count * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(M5 full coarse correction H2D)");

    const auto weights = impl_->chebyshev_weights(smoother_degree);
    impl_->run_full_shell_once(weights, transfer_smoothing_steps, nullptr);
    check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(M5 full warmup)");

    std::array<cudaEvent_t, 10> events{};
    try {
        for (auto& event : events) {
            check_cuda_pcg(cudaEventCreate(&event), "cudaEventCreate(M5 full shell)");
        }
        std::vector<GpuM5FineFullShellTiming> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            impl_->run_full_shell_once(weights, transfer_smoothing_steps, &events);
            check_cuda_pcg(cudaEventSynchronize(events[9]), "cudaEventSynchronize(M5 full shell)");
            GpuM5FineFullShellTiming sample;
            sample.zero_ms = static_cast<double>(m5_elapsed(events[0], events[1]));
            sample.pre_smooth_ms = static_cast<double>(m5_elapsed(events[1], events[2]));
            sample.residual_ms = static_cast<double>(m5_elapsed(events[2], events[3]));
            sample.transfer_transpose_ms = static_cast<double>(m5_elapsed(events[3], events[4]));
            sample.p0t_ms = static_cast<double>(m5_elapsed(events[4], events[5]));
            sample.p0_ms = static_cast<double>(m5_elapsed(events[5], events[6]));
            sample.transfer_forward_ms = static_cast<double>(m5_elapsed(events[6], events[7]));
            sample.correction_ms = static_cast<double>(m5_elapsed(events[7], events[8]));
            sample.post_smooth_ms = static_cast<double>(m5_elapsed(events[8], events[9]));
            sample.total_ms = static_cast<double>(m5_elapsed(events[0], events[9]));
            samples.push_back(sample);
        }

        GpuM5FineFullShellResult result;
        result.coarse_residual.resize(impl_->coarse_dof_count, 0.0f);
        check_cuda_pcg(cudaMemcpy(result.coarse_residual.data(), impl_->d_coarse,
                                  impl_->coarse_dof_count * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 full coarse residual D2H)");

        std::vector<float> fine_soa(impl_->ndof, 0.0f);
        check_cuda_pcg(cudaMemcpy(fine_soa.data(), impl_->d_x,
                                  impl_->ndof * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 full fine correction D2H)");
        result.fine_correction_aos.resize(impl_->ndof, 0.0f);
        for (std::size_t node = 0; node < impl_->nodes; ++node) {
            result.fine_correction_aos[3U * node + 0U] = fine_soa[node];
            result.fine_correction_aos[3U * node + 1U] = fine_soa[impl_->nodes + node];
            result.fine_correction_aos[3U * node + 2U] = fine_soa[2U * impl_->nodes + node];
        }

        result.median_timing = m5_summarize_full_median(samples);
        result.best_timing = m5_summarize_full_best(samples);
        result.smoother_degree = smoother_degree;
        result.transfer_smoothing_steps = transfer_smoothing_steps;
        result.fine_operator_applies = 2U * smoother_degree + 1U +
                                       2U * transfer_smoothing_steps;
        result.fine_vector_bytes = impl_->fine_vector_bytes;
        result.coarse_vector_bytes = impl_->coarse_vector_bytes;
        result.aggregation_metadata_bytes = impl_->aggregation_metadata_bytes;
        result.model_coordinate_bytes = impl_->model_coordinate_bytes;
        result.device_bytes_total = impl_->fine_vector_bytes + impl_->coarse_vector_bytes +
                                    impl_->aggregation_metadata_bytes +
                                    impl_->model_coordinate_bytes;
        for (auto event : events) cudaEventDestroy(event);
        return result;
    } catch (...) {
        for (auto event : events) {
            if (event) cudaEventDestroy(event);
        }
        throw;
    }
}

std::size_t GpuM5FineLevelContext::fine_dofs() const noexcept {
    return impl_ ? impl_->ndof : 0U;
}

std::size_t GpuM5FineLevelContext::coarse_dofs() const noexcept {
    return impl_ ? impl_->coarse_dof_count : 0U;
}

double GpuM5FineLevelContext::transfer_omega() const noexcept {
    return impl_ ? impl_->transfer_omega_value : 0.0;
}

double GpuM5FineLevelContext::smoother_lambda_max() const noexcept {
    return impl_ ? impl_->smoother_lambda_max_value : 0.0;
}

}  // namespace gfss
