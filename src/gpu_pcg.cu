#include "gfss/gpu_solver.hpp"

#include "gfss/cpu_gold.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

using Clock = std::chrono::steady_clock;

struct DeviceNodeStencilEntryPcg {
    std::int8_t dx{0};
    std::int8_t dy{0};
    std::int8_t dz{0};
    std::int8_t pad{0};
    float block[9]{};
};

struct DeviceDiagEntryPcg {
    int offset{0};
    float b00{0.0f};
    float b11{0.0f};
    float b22{0.0f};
};

struct DeviceEdgeXYEntryPcg {
    int offset{0};
    float b00{0.0f};
    float b01{0.0f};
    float b10{0.0f};
    float b11{0.0f};
    float b22{0.0f};
};

struct DeviceEdgeXZEntryPcg {
    int offset{0};
    float b00{0.0f};
    float b02{0.0f};
    float b11{0.0f};
    float b20{0.0f};
    float b22{0.0f};
};

struct DeviceEdgeYZEntryPcg {
    int offset{0};
    float b00{0.0f};
    float b11{0.0f};
    float b12{0.0f};
    float b21{0.0f};
    float b22{0.0f};
};

struct DeviceCornerEntryPcg {
    int offset{0};
    float block[9]{};
};

static_assert(sizeof(DeviceNodeStencilEntryPcg) == 40,
              "unexpected CUDA PCG boundary entry layout");

__constant__ DeviceNodeStencilEntryPcg kPcgBoundaryEntries[27 * 27];
__constant__ std::uint8_t kPcgBoundaryCounts[27];
__constant__ DeviceDiagEntryPcg kPcgDiag[7];
__constant__ DeviceEdgeXYEntryPcg kPcgEdgeXY[4];
__constant__ DeviceEdgeXZEntryPcg kPcgEdgeXZ[4];
__constant__ DeviceEdgeYZEntryPcg kPcgEdgeYZ[4];
__constant__ DeviceCornerEntryPcg kPcgCorner[8];
__constant__ float kPcgInvDiag[27][3];

void check_cuda_pcg(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_cublas_pcg(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) +
                                 ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

__host__ __device__ constexpr int class_index_pcg(int cx, int cy, int cz) noexcept {
    return cx + 3 * (cy + 3 * cz);
}

__device__ __forceinline__ int axis_class_pcg(std::uint32_t coordinate,
                                               std::uint32_t max_coordinate) {
    return coordinate == 0U ? 0 : (coordinate == max_coordinate ? 2 : 1);
}

__global__ void node_stencil_gold_sparse_pcg_kernel(
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
                        3 * (axis_class_pcg(j, ny) +
                             3 * axis_class_pcg(k, nz));
        const int count = static_cast<int>(kPcgBoundaryCounts[cls]);
#pragma unroll 1
        for (int e = 0; e < count; ++e) {
            const DeviceNodeStencilEntryPcg& entry =
                kPcgBoundaryEntries[cls * 27 + e];
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

__global__ void jacobi_pcg_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    const float* __restrict__ rx,
    const float* __restrict__ ry,
    const float* __restrict__ rz,
    float* __restrict__ zx,
    float* __restrict__ zy,
    float* __restrict__ zz) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const std::uint32_t k = blockIdx.z;

    if (i > nx || j > ny || k > nz) {
        return;
    }

    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::uint32_t node = i + sx * (j + sy * k);

    if (i == 0U) {
        zx[node] = 0.0f;
        zy[node] = 0.0f;
        zz[node] = 0.0f;
        return;
    }

    const int cls = axis_class_pcg(i, nx) +
                    3 * (axis_class_pcg(j, ny) +
                         3 * axis_class_pcg(k, nz));
    zx[node] = kPcgInvDiag[cls][0] * rx[node];
    zy[node] = kPcgInvDiag[cls][1] * ry[node];
    zz[node] = kPcgInvDiag[cls][2] * rz[node];
}

__global__ void zero_x0_face_pcg_kernel(
    std::uint32_t nx,
    std::uint32_t ny,
    std::uint32_t nz,
    std::size_t nodes,
    float* __restrict__ v) {
    const std::uint32_t face_index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t face_nodes = (ny + 1U) * (nz + 1U);
    if (face_index >= face_nodes) {
        return;
    }

    const std::uint32_t j = face_index % (ny + 1U);
    const std::uint32_t k = face_index / (ny + 1U);
    const std::uint32_t sx = nx + 1U;
    const std::uint32_t sy = ny + 1U;
    const std::uint32_t node = sx * (j + sy * k);
    v[node] = 0.0f;
    v[nodes + node] = 0.0f;
    v[2U * nodes + node] = 0.0f;
}

__global__ void update_x_r_pcg_kernel(
    int n,
    float alpha,
    const float* __restrict__ p,
    const float* __restrict__ q,
    float* __restrict__ x,
    float* __restrict__ r) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        x[idx] = fmaf(alpha, p[idx], x[idx]);
        r[idx] = fmaf(-alpha, q[idx], r[idx]);
    }
}

__global__ void update_p_pcg_kernel(
    int n,
    float beta,
    const float* __restrict__ z,
    float* __restrict__ p) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < n) {
        p[idx] = fmaf(beta, p[idx], z[idx]);
    }
}

void upload_pcg_stencil(const StructuredHexMesh& mesh,
                        const Material& material) {
    const auto host = build_cpu_gold_stencil_fp32(mesh, material);

    std::array<DeviceNodeStencilEntryPcg, 27 * 27> boundary{};
    std::array<DeviceDiagEntryPcg, 7> diag{};
    std::array<DeviceEdgeXYEntryPcg, 4> edge_xy{};
    std::array<DeviceEdgeXZEntryPcg, 4> edge_xz{};
    std::array<DeviceEdgeYZEntryPcg, 4> edge_yz{};
    std::array<DeviceCornerEntryPcg, 8> corner{};
    std::array<std::array<float, 3>, 27> inv_diag{};

    for (std::size_t cls = 0; cls < 27; ++cls) {
        bool found_center = false;
        const std::size_t count = host.regular.counts[cls];
        for (std::size_t e = 0; e < 27; ++e) {
            const auto& src = host.regular.entries[cls][e];
            auto& dst = boundary[cls * 27 + e];
            dst.dx = src.dx;
            dst.dy = src.dy;
            dst.dz = src.dz;
            for (std::size_t q = 0; q < 9; ++q) {
                dst.block[q] = src.block[q];
            }

            if (e < count && src.dx == 0 && src.dy == 0 && src.dz == 0) {
                if (!(src.block[0] > 0.0f) ||
                    !(src.block[4] > 0.0f) ||
                    !(src.block[8] > 0.0f)) {
                    throw std::runtime_error("PCG Jacobi encountered non-positive diagonal");
                }
                inv_diag[cls][0] = 1.0f / src.block[0];
                inv_diag[cls][1] = 1.0f / src.block[4];
                inv_diag[cls][2] = 1.0f / src.block[8];
                found_center = true;
            }
        }
        if (!found_center) {
            throw std::runtime_error("PCG Jacobi could not locate stencil center entry");
        }
    }

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

    check_cuda_pcg(cudaMemcpyToSymbol(kPcgBoundaryEntries,
                                      boundary.data(),
                                      boundary.size() * sizeof(DeviceNodeStencilEntryPcg)),
                   "cudaMemcpyToSymbol(PCG boundary entries)");
    check_cuda_pcg(cudaMemcpyToSymbol(kPcgBoundaryCounts,
                                      host.regular.counts.data(),
                                      host.regular.counts.size() * sizeof(std::uint8_t)),
                   "cudaMemcpyToSymbol(PCG boundary counts)");
    check_cuda_pcg(cudaMemcpyToSymbol(kPcgDiag, diag.data(),
                                      diag.size() * sizeof(DeviceDiagEntryPcg)),
                   "cudaMemcpyToSymbol(PCG diag)");
    check_cuda_pcg(cudaMemcpyToSymbol(kPcgEdgeXY, edge_xy.data(),
                                      edge_xy.size() * sizeof(DeviceEdgeXYEntryPcg)),
                   "cudaMemcpyToSymbol(PCG edge XY)");
    check_cuda_pcg(cudaMemcpyToSymbol(kPcgEdgeXZ, edge_xz.data(),
                                      edge_xz.size() * sizeof(DeviceEdgeXZEntryPcg)),
                   "cudaMemcpyToSymbol(PCG edge XZ)");
    check_cuda_pcg(cudaMemcpyToSymbol(kPcgEdgeYZ, edge_yz.data(),
                                      edge_yz.size() * sizeof(DeviceEdgeYZEntryPcg)),
                   "cudaMemcpyToSymbol(PCG edge YZ)");
    check_cuda_pcg(cudaMemcpyToSymbol(kPcgCorner, corner.data(),
                                      corner.size() * sizeof(DeviceCornerEntryPcg)),
                   "cudaMemcpyToSymbol(PCG corner)");
    check_cuda_pcg(cudaMemcpyToSymbol(kPcgInvDiag, inv_diag.data(),
                                      inv_diag.size() * sizeof(inv_diag[0])),
                   "cudaMemcpyToSymbol(PCG inverse diagonal)");
}

float dot_pcg(cublasHandle_t handle,
              int n,
              const float* a,
              const float* b) {
    float result = 0.0f;
    check_cublas_pcg(cublasSdot(handle, n, a, 1, b, 1, &result),
                     "cublasSdot(PCG)");
    return result;
}

void launch_pcg_matvec(const StructuredHexMesh& mesh,
                       int block_y,
                       std::size_t nodes,
                       const float* p,
                       float* q) {
    const dim3 block(32U, static_cast<unsigned int>(block_y), 1U);
    const dim3 grid((mesh.nx + 1U + block.x - 1U) / block.x,
                    (mesh.ny + 1U + block.y - 1U) / block.y,
                    mesh.nz + 1U);

    node_stencil_gold_sparse_pcg_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz,
        p, p + nodes, p + 2U * nodes,
        q, q + nodes, q + 2U * nodes);
    check_cuda_pcg(cudaGetLastError(), "PCG GoldSparse matvec launch");

    const std::uint32_t face_nodes = (mesh.ny + 1U) * (mesh.nz + 1U);
    constexpr unsigned int threads = 256U;
    const unsigned int blocks = (face_nodes + threads - 1U) / threads;
    zero_x0_face_pcg_kernel<<<blocks, threads>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, q);
    check_cuda_pcg(cudaGetLastError(), "PCG clamp output launch");
}

void launch_pcg_jacobi(const StructuredHexMesh& mesh,
                       int block_y,
                       std::size_t nodes,
                       const float* r,
                       float* z) {
    const dim3 block(32U, static_cast<unsigned int>(block_y), 1U);
    const dim3 grid((mesh.nx + 1U + block.x - 1U) / block.x,
                    (mesh.ny + 1U + block.y - 1U) / block.y,
                    mesh.nz + 1U);
    jacobi_pcg_kernel<<<grid, block>>>(
        mesh.nx, mesh.ny, mesh.nz,
        r, r + nodes, r + 2U * nodes,
        z, z + nodes, z + 2U * nodes);
    check_cuda_pcg(cudaGetLastError(), "PCG Jacobi launch");
}

}  // namespace

GpuPcgResult solve_pcg_cuda_gold_sparse_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<float>& rhs,
    double relative_tolerance,
    std::size_t max_iterations,
    int block_y) {
    if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
        throw std::invalid_argument("GPU PCG requires non-empty mesh dimensions");
    }
    if (rhs.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("GPU PCG RHS size does not match mesh DOF count");
    }
    if (!(relative_tolerance > 0.0)) {
        throw std::invalid_argument("GPU PCG relative tolerance must be positive");
    }
    if (max_iterations == 0U) {
        throw std::invalid_argument("GPU PCG max_iterations must be positive");
    }
    if (block_y <= 0 || block_y > 32 || 32 * block_y > 1024) {
        throw std::invalid_argument("GPU PCG block_y must produce a valid 32 x block_y block");
    }
    if (mesh.node_count() > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("GPU PCG currently requires node_count <= INT_MAX");
    }
    if (mesh.dof_count() > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("GPU PCG currently requires dof_count <= INT_MAX for cuBLAS reductions");
    }

    upload_pcg_stencil(mesh, material);

    const std::size_t nodes = static_cast<std::size_t>(mesh.node_count());
    const std::size_t ndof = static_cast<std::size_t>(mesh.dof_count());
    const std::size_t vector_bytes = ndof * sizeof(float);
    const int n = static_cast<int>(ndof);

    std::vector<float> host_b(ndof, 0.0f);
    for (std::size_t node = 0; node < nodes; ++node) {
        host_b[node] = rhs[3U * node + 0U];
        host_b[nodes + node] = rhs[3U * node + 1U];
        host_b[2U * nodes + node] = rhs[3U * node + 2U];
    }
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const std::size_t node = static_cast<std::size_t>(mesh.node_index(0U, j, k));
            host_b[node] = 0.0f;
            host_b[nodes + node] = 0.0f;
            host_b[2U * nodes + node] = 0.0f;
        }
    }

    float* d_b = nullptr;
    float* d_x = nullptr;
    float* d_r = nullptr;
    float* d_z = nullptr;
    float* d_p = nullptr;
    float* d_q = nullptr;
    cublasHandle_t handle = nullptr;

    try {
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_b), vector_bytes), "cudaMalloc(PCG b)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_x), vector_bytes), "cudaMalloc(PCG x)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_r), vector_bytes), "cudaMalloc(PCG r)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_z), vector_bytes), "cudaMalloc(PCG z)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_p), vector_bytes), "cudaMalloc(PCG p)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_q), vector_bytes), "cudaMalloc(PCG q)");
        check_cuda_pcg(cudaMemcpy(d_b, host_b.data(), vector_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(PCG b H2D)");

        check_cublas_pcg(cublasCreate(&handle), "cublasCreate(PCG)");
        check_cublas_pcg(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                         "cublasSetPointerMode(PCG)");

        // Warm the exact matvec path without changing timed PCG state.
        check_cuda_pcg(cudaMemset(d_p, 0, vector_bytes), "cudaMemset(PCG warm p)");
        launch_pcg_matvec(mesh, block_y, nodes, d_p, d_q);
        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(PCG warmup)");

        const auto wall_start = Clock::now();

        check_cuda_pcg(cudaMemsetAsync(d_x, 0, vector_bytes), "cudaMemsetAsync(PCG x)");
        check_cuda_pcg(cudaMemcpyAsync(d_r, d_b, vector_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(PCG r=b)");
        launch_pcg_jacobi(mesh, block_y, nodes, d_r, d_z);
        check_cuda_pcg(cudaMemcpyAsync(d_p, d_z, vector_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(PCG p=z)");

        const float bnorm2 = dot_pcg(handle, n, d_b, d_b);
        float rho = dot_pcg(handle, n, d_r, d_z);
        float rr = dot_pcg(handle, n, d_r, d_r);

        GpuPcgResult result;
        result.explicit_device_bytes = 6U * vector_bytes;
        result.reported_relative_residual =
            bnorm2 > 0.0f ? std::sqrt(static_cast<double>(rr) / static_cast<double>(bnorm2)) : 0.0;

        if (!(bnorm2 >= 0.0f) || !std::isfinite(bnorm2)) {
            throw std::runtime_error("GPU PCG RHS norm became invalid");
        }

        if (bnorm2 == 0.0f) {
            result.converged = true;
        } else {
            constexpr int vector_threads = 256;
            const int vector_blocks = (n + vector_threads - 1) / vector_threads;

            for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
                launch_pcg_matvec(mesh, block_y, nodes, d_p, d_q);
                ++result.matvecs;

                const float pq = dot_pcg(handle, n, d_p, d_q);
                if (!(pq > 0.0f) || !std::isfinite(pq) ||
                    !(rho > 0.0f) || !std::isfinite(rho)) {
                    throw std::runtime_error("GPU PCG breakdown: non-positive or non-finite scalar");
                }

                const float alpha = rho / pq;
                update_x_r_pcg_kernel<<<vector_blocks, vector_threads>>>(
                    n, alpha, d_p, d_q, d_x, d_r);
                check_cuda_pcg(cudaGetLastError(), "GPU PCG x/r update launch");

                rr = dot_pcg(handle, n, d_r, d_r);
                result.iterations = iteration + 1U;
                result.reported_relative_residual =
                    std::sqrt(static_cast<double>(rr) / static_cast<double>(bnorm2));

                if (!std::isfinite(rr) || !std::isfinite(result.reported_relative_residual)) {
                    throw std::runtime_error("GPU PCG residual became non-finite");
                }
                if (result.reported_relative_residual <= relative_tolerance) {
                    result.converged = true;
                    break;
                }

                launch_pcg_jacobi(mesh, block_y, nodes, d_r, d_z);
                const float rho_new = dot_pcg(handle, n, d_r, d_z);
                if (!(rho_new > 0.0f) || !std::isfinite(rho_new)) {
                    throw std::runtime_error("GPU PCG breakdown: preconditioned residual became invalid");
                }
                const float beta = rho_new / rho;
                update_p_pcg_kernel<<<vector_blocks, vector_threads>>>(
                    n, beta, d_z, d_p);
                check_cuda_pcg(cudaGetLastError(), "GPU PCG p update launch");
                rho = rho_new;
            }
        }

        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(PCG solve)");
        const auto wall_stop = Clock::now();
        result.solve_ms =
            std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();

        std::vector<float> host_x(ndof);
        check_cuda_pcg(cudaMemcpy(host_x.data(), d_x, vector_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(PCG x D2H)");
        result.x.resize(ndof);
        for (std::size_t node = 0; node < nodes; ++node) {
            result.x[3U * node + 0U] = host_x[node];
            result.x[3U * node + 1U] = host_x[nodes + node];
            result.x[3U * node + 2U] = host_x[2U * nodes + node];
        }

        cublasDestroy(handle);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_z);
        cudaFree(d_p);
        cudaFree(d_q);
        return result;
    } catch (...) {
        if (handle) cublasDestroy(handle);
        if (d_b) cudaFree(d_b);
        if (d_x) cudaFree(d_x);
        if (d_r) cudaFree(d_r);
        if (d_z) cudaFree(d_z);
        if (d_p) cudaFree(d_p);
        if (d_q) cudaFree(d_q);
        throw;
    }
}

}  // namespace gfss
