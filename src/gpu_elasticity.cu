#include "gfss/gpu_elasticity.hpp"

#include "gfss/hex8.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

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
    cudaEvent_t stop{};

    const std::size_t bytes = x.size() * sizeof(float);
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes), "cudaMalloc(x)");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_y), bytes), "cudaMalloc(y)");
        check_cuda(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(x H2D)");
        check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
        check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

        constexpr int threads = 128;
        const std::uint32_t elements = static_cast<std::uint32_t>(mesh.element_count());
        const int blocks = static_cast<int>((elements + threads - 1U) / threads);

        double best_ms = 1.0e300;
        double total_ms = 0.0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
            check_cuda(cudaMemsetAsync(d_y, 0, bytes), "cudaMemsetAsync(y)");
            hex8_atomic_apply_kernel<<<blocks, threads>>>(mesh.nx, mesh.ny, mesh.nz, d_x, d_y);
            check_cuda(cudaGetLastError(), "hex8_atomic_apply_kernel launch");
            check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
            check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

            float ms = 0.0f;
            check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
            best_ms = std::min(best_ms, static_cast<double>(ms));
            total_ms += static_cast<double>(ms);
        }

        CudaOperatorResult result;
        result.y.resize(x.size());
        check_cuda(cudaMemcpy(result.y.data(), d_y, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy(y D2H)");
        result.timing.best_ms = best_ms;
        result.timing.mean_ms = total_ms / static_cast<double>(repeats);
        result.device_bytes = 2 * bytes;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_x);
        cudaFree(d_y);
        return result;
    } catch (...) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        throw;
    }
}

}  // namespace gfss
