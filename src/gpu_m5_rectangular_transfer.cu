#include "gfss/gpu_m5_rectangular_transfer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

void check_cuda_transfer(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__global__ void m5_rectangular_csr_warp_kernel(
    std::uint32_t rows,
    const std::uint32_t* __restrict__ row_offsets,
    const std::uint32_t* __restrict__ column_indices,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y) {
    const std::uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t warp = global_thread >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    if (warp >= rows) return;

    const std::uint32_t first = row_offsets[warp];
    const std::uint32_t last = row_offsets[warp + 1U];
    float sum = 0.0f;
    for (std::uint32_t p = first + lane; p < last; p += 32U) {
        sum = fmaf(values[p], x[column_indices[p]], sum);
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, delta);
    }
    if (lane == 0U) y[warp] = sum;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

}  // namespace

GpuM5RectangularTransferResult benchmark_m5_rectangular_csr(
    std::size_t rows,
    std::size_t cols,
    const std::vector<std::uint32_t>& row_offsets,
    const std::vector<std::uint32_t>& column_indices,
    const std::vector<float>& values,
    const std::vector<float>& x,
    int repeats) {
    if (rows == 0U || cols == 0U ||
        rows > std::numeric_limits<std::uint32_t>::max() ||
        cols > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("M5 rectangular transfer unsupported dimensions");
    }
    if (row_offsets.size() != rows + 1U || x.size() != cols) {
        throw std::invalid_argument("M5 rectangular transfer shape mismatch");
    }
    if (column_indices.size() != values.size() ||
        row_offsets.back() != values.size()) {
        throw std::invalid_argument("M5 rectangular transfer payload mismatch");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("M5 rectangular transfer repeats must be positive");
    }
    for (std::size_t r = 0; r < rows; ++r) {
        if (row_offsets[r] > row_offsets[r + 1U]) {
            throw std::invalid_argument("M5 rectangular transfer row offsets not monotone");
        }
    }
    for (const auto c : column_indices) {
        if (c >= cols) {
            throw std::invalid_argument("M5 rectangular transfer column out of range");
        }
    }

    const std::size_t row_bytes = row_offsets.size() * sizeof(std::uint32_t);
    const std::size_t col_bytes = column_indices.size() * sizeof(std::uint32_t);
    const std::size_t val_bytes = values.size() * sizeof(float);
    const std::size_t x_bytes = cols * sizeof(float);
    const std::size_t y_bytes = rows * sizeof(float);

    std::uint32_t* d_rows = nullptr;
    std::uint32_t* d_cols = nullptr;
    float* d_values = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    try {
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_rows), row_bytes),
                            "cudaMalloc(M5 transfer rows)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_cols), col_bytes),
                            "cudaMalloc(M5 transfer columns)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_values), val_bytes),
                            "cudaMalloc(M5 transfer values)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_x), x_bytes),
                            "cudaMalloc(M5 transfer x)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_y), y_bytes),
                            "cudaMalloc(M5 transfer y)");
        check_cuda_transfer(cudaMemcpy(d_rows, row_offsets.data(), row_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 transfer rows H2D)");
        check_cuda_transfer(cudaMemcpy(d_cols, column_indices.data(), col_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 transfer columns H2D)");
        check_cuda_transfer(cudaMemcpy(d_values, values.data(), val_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 transfer values H2D)");
        check_cuda_transfer(cudaMemcpy(d_x, x.data(), x_bytes, cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 transfer x H2D)");

        constexpr unsigned int threads = 256U;
        const unsigned int warps_per_block = threads / 32U;
        const unsigned int blocks = static_cast<unsigned int>(
            (rows + warps_per_block - 1U) / warps_per_block);

        m5_rectangular_csr_warp_kernel<<<blocks, threads>>>(
            static_cast<std::uint32_t>(rows), d_rows, d_cols, d_values, d_x, d_y);
        check_cuda_transfer(cudaGetLastError(), "M5 transfer warmup launch");
        check_cuda_transfer(cudaDeviceSynchronize(), "M5 transfer warmup sync");

        check_cuda_transfer(cudaEventCreate(&start), "cudaEventCreate(M5 transfer start)");
        check_cuda_transfer(cudaEventCreate(&stop), "cudaEventCreate(M5 transfer stop)");
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_transfer(cudaEventRecord(start), "cudaEventRecord(M5 transfer start)");
            m5_rectangular_csr_warp_kernel<<<blocks, threads>>>(
                static_cast<std::uint32_t>(rows), d_rows, d_cols, d_values, d_x, d_y);
            check_cuda_transfer(cudaGetLastError(), "M5 transfer launch");
            check_cuda_transfer(cudaEventRecord(stop), "cudaEventRecord(M5 transfer stop)");
            check_cuda_transfer(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 transfer)");
            float ms = 0.0f;
            check_cuda_transfer(cudaEventElapsedTime(&ms, start, stop),
                                "cudaEventElapsedTime(M5 transfer)");
            samples.push_back(static_cast<double>(ms));
        }

        GpuM5RectangularTransferResult result;
        result.rows = rows;
        result.cols = cols;
        result.nnz = values.size();
        result.y.resize(rows, 0.0f);
        check_cuda_transfer(cudaMemcpy(result.y.data(), d_y, y_bytes, cudaMemcpyDeviceToHost),
                            "cudaMemcpy(M5 transfer y D2H)");
        result.timing.median_ms = median(samples);
        result.timing.best_ms = *std::min_element(samples.begin(), samples.end());
        result.device_bytes = row_bytes + col_bytes + val_bytes + x_bytes + y_bytes;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_rows);
        cudaFree(d_cols);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        return result;
    } catch (...) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (d_rows) cudaFree(d_rows);
        if (d_cols) cudaFree(d_cols);
        if (d_values) cudaFree(d_values);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        throw;
    }
}

}  // namespace gfss
