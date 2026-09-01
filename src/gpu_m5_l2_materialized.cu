#include "gfss/gpu_m5_l2_materialized.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

void check_cuda_l2(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__global__ void m5_l2_csr_spmv_warp_kernel(
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
    if (n & 1U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

}  // namespace

GpuM5L2CsrResult benchmark_m5_l2_csr(
    const std::vector<std::uint32_t>& row_offsets,
    const std::vector<std::uint32_t>& column_indices,
    const std::vector<float>& values,
    const std::vector<float>& x,
    int repeats) {
    if (row_offsets.size() < 2U) {
        throw std::invalid_argument("M5 L2 CSR requires at least one row");
    }
    const std::size_t rows = row_offsets.size() - 1U;
    if (rows > std::numeric_limits<std::uint32_t>::max() || x.size() != rows) {
        throw std::invalid_argument("M5 L2 CSR row/vector size mismatch");
    }
    if (column_indices.size() != values.size() || row_offsets.back() != values.size()) {
        throw std::invalid_argument("M5 L2 CSR payload size mismatch");
    }
    if (repeats <= 0) throw std::invalid_argument("M5 L2 CSR repeats must be positive");
    for (std::size_t r = 0; r < rows; ++r) {
        if (row_offsets[r] > row_offsets[r + 1U]) {
            throw std::invalid_argument("M5 L2 CSR row offsets not monotone");
        }
    }
    for (const auto c : column_indices) {
        if (c >= rows) throw std::invalid_argument("M5 L2 CSR column out of range");
    }

    const std::size_t row_bytes = row_offsets.size() * sizeof(std::uint32_t);
    const std::size_t col_bytes = column_indices.size() * sizeof(std::uint32_t);
    const std::size_t val_bytes = values.size() * sizeof(float);
    const std::size_t vec_bytes = rows * sizeof(float);

    std::uint32_t* d_rows = nullptr;
    std::uint32_t* d_cols = nullptr;
    float* d_values = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    try {
        check_cuda_l2(cudaMalloc(reinterpret_cast<void**>(&d_rows), row_bytes),
                      "cudaMalloc(M5 L2 rows)");
        check_cuda_l2(cudaMalloc(reinterpret_cast<void**>(&d_cols), col_bytes),
                      "cudaMalloc(M5 L2 columns)");
        check_cuda_l2(cudaMalloc(reinterpret_cast<void**>(&d_values), val_bytes),
                      "cudaMalloc(M5 L2 values)");
        check_cuda_l2(cudaMalloc(reinterpret_cast<void**>(&d_x), vec_bytes),
                      "cudaMalloc(M5 L2 x)");
        check_cuda_l2(cudaMalloc(reinterpret_cast<void**>(&d_y), vec_bytes),
                      "cudaMalloc(M5 L2 y)");
        check_cuda_l2(cudaMemcpy(d_rows, row_offsets.data(), row_bytes, cudaMemcpyHostToDevice),
                      "cudaMemcpy(M5 L2 rows H2D)");
        check_cuda_l2(cudaMemcpy(d_cols, column_indices.data(), col_bytes, cudaMemcpyHostToDevice),
                      "cudaMemcpy(M5 L2 columns H2D)");
        check_cuda_l2(cudaMemcpy(d_values, values.data(), val_bytes, cudaMemcpyHostToDevice),
                      "cudaMemcpy(M5 L2 values H2D)");
        check_cuda_l2(cudaMemcpy(d_x, x.data(), vec_bytes, cudaMemcpyHostToDevice),
                      "cudaMemcpy(M5 L2 x H2D)");

        constexpr unsigned int threads = 256U;
        const unsigned int warps_per_block = threads / 32U;
        const unsigned int blocks = static_cast<unsigned int>(
            (rows + warps_per_block - 1U) / warps_per_block);
        m5_l2_csr_spmv_warp_kernel<<<blocks, threads>>>(
            static_cast<std::uint32_t>(rows), d_rows, d_cols, d_values, d_x, d_y);
        check_cuda_l2(cudaGetLastError(), "M5 L2 CSR warmup launch");
        check_cuda_l2(cudaDeviceSynchronize(), "M5 L2 CSR warmup sync");

        check_cuda_l2(cudaEventCreate(&start), "cudaEventCreate(M5 L2 start)");
        check_cuda_l2(cudaEventCreate(&stop), "cudaEventCreate(M5 L2 stop)");
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_l2(cudaEventRecord(start), "cudaEventRecord(M5 L2 start)");
            m5_l2_csr_spmv_warp_kernel<<<blocks, threads>>>(
                static_cast<std::uint32_t>(rows), d_rows, d_cols, d_values, d_x, d_y);
            check_cuda_l2(cudaGetLastError(), "M5 L2 CSR launch");
            check_cuda_l2(cudaEventRecord(stop), "cudaEventRecord(M5 L2 stop)");
            check_cuda_l2(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 L2)");
            float ms = 0.0f;
            check_cuda_l2(cudaEventElapsedTime(&ms, start, stop),
                          "cudaEventElapsedTime(M5 L2)");
            samples.push_back(static_cast<double>(ms));
        }

        GpuM5L2CsrResult result;
        result.rows = rows;
        result.nnz = values.size();
        result.y.resize(rows, 0.0f);
        check_cuda_l2(cudaMemcpy(result.y.data(), d_y, vec_bytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy(M5 L2 y D2H)");
        result.timing.median_ms = median(samples);
        result.timing.best_ms = *std::min_element(samples.begin(), samples.end());
        result.device_bytes = row_bytes + col_bytes + val_bytes + 2U * vec_bytes;

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
