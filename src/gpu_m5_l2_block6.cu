#include "gfss/gpu_m5_l2_block6.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

void check_cuda_block6(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__global__ void m5_l2_block6_csr_spmv_kernel(
    std::uint32_t block_rows,
    const std::uint32_t* __restrict__ row_offsets,
    const std::uint32_t* __restrict__ column_indices,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y) {
    const std::uint32_t scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t scalar_rows = block_rows * 6U;
    if (scalar_row >= scalar_rows) return;

    const std::uint32_t block_row = scalar_row / 6U;
    const std::uint32_t local_row = scalar_row - block_row * 6U;
    const std::uint32_t first = row_offsets[block_row];
    const std::uint32_t last = row_offsets[block_row + 1U];

    float sum = 0.0f;
    for (std::uint32_t p = first; p < last; ++p) {
        const std::uint32_t block_col = column_indices[p];
        const float* __restrict__ block = values + static_cast<std::size_t>(p) * 36U + local_row * 6U;
        const float* __restrict__ xv = x + static_cast<std::size_t>(block_col) * 6U;
#pragma unroll
        for (std::uint32_t c = 0U; c < 6U; ++c) {
            sum = fmaf(block[c], xv[c], sum);
        }
    }
    y[scalar_row] = sum;
}

double block6_median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n & 1U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

}  // namespace

GpuM5L2Block6Result benchmark_m5_l2_block6_csr(
    const std::vector<std::uint32_t>& block_row_offsets,
    const std::vector<std::uint32_t>& block_column_indices,
    const std::vector<float>& block_values_row_major_6x6,
    const std::vector<float>& x,
    int repeats) {
    if (block_row_offsets.size() < 2U) {
        throw std::invalid_argument("M5 L2 block6 CSR requires at least one block row");
    }
    const std::size_t block_rows = block_row_offsets.size() - 1U;
    if (block_rows > static_cast<std::size_t>(UINT32_MAX) || x.size() != block_rows * 6U) {
        throw std::invalid_argument("M5 L2 block6 CSR row/vector size mismatch");
    }
    if (block_values_row_major_6x6.size() != block_column_indices.size() * 36U ||
        block_row_offsets.back() != block_column_indices.size()) {
        throw std::invalid_argument("M5 L2 block6 CSR payload size mismatch");
    }
    if (repeats <= 0) throw std::invalid_argument("M5 L2 block6 repeats must be positive");
    for (std::size_t r = 0U; r < block_rows; ++r) {
        if (block_row_offsets[r] > block_row_offsets[r + 1U]) {
            throw std::invalid_argument("M5 L2 block6 row offsets not monotone");
        }
    }
    for (const auto c : block_column_indices) {
        if (c >= block_rows) throw std::invalid_argument("M5 L2 block6 column out of range");
    }

    const std::size_t row_bytes = block_row_offsets.size() * sizeof(std::uint32_t);
    const std::size_t col_bytes = block_column_indices.size() * sizeof(std::uint32_t);
    const std::size_t val_bytes = block_values_row_major_6x6.size() * sizeof(float);
    const std::size_t vec_bytes = x.size() * sizeof(float);

    std::uint32_t* d_rows = nullptr;
    std::uint32_t* d_cols = nullptr;
    float* d_values = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    try {
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_rows), row_bytes),
                          "cudaMalloc(M5 L2 block6 rows)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_cols), col_bytes),
                          "cudaMalloc(M5 L2 block6 columns)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_values), val_bytes),
                          "cudaMalloc(M5 L2 block6 values)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_x), vec_bytes),
                          "cudaMalloc(M5 L2 block6 x)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_y), vec_bytes),
                          "cudaMalloc(M5 L2 block6 y)");
        check_cuda_block6(cudaMemcpy(d_rows, block_row_offsets.data(), row_bytes, cudaMemcpyHostToDevice),
                          "cudaMemcpy(M5 L2 block6 rows H2D)");
        check_cuda_block6(cudaMemcpy(d_cols, block_column_indices.data(), col_bytes, cudaMemcpyHostToDevice),
                          "cudaMemcpy(M5 L2 block6 columns H2D)");
        check_cuda_block6(cudaMemcpy(d_values, block_values_row_major_6x6.data(), val_bytes, cudaMemcpyHostToDevice),
                          "cudaMemcpy(M5 L2 block6 values H2D)");
        check_cuda_block6(cudaMemcpy(d_x, x.data(), vec_bytes, cudaMemcpyHostToDevice),
                          "cudaMemcpy(M5 L2 block6 x H2D)");

        constexpr unsigned int threads = 256U;
        const std::size_t scalar_rows = block_rows * 6U;
        const unsigned int blocks = static_cast<unsigned int>((scalar_rows + threads - 1U) / threads);
        m5_l2_block6_csr_spmv_kernel<<<blocks, threads>>>(
            static_cast<std::uint32_t>(block_rows), d_rows, d_cols, d_values, d_x, d_y);
        check_cuda_block6(cudaGetLastError(), "M5 L2 block6 warmup launch");
        check_cuda_block6(cudaDeviceSynchronize(), "M5 L2 block6 warmup sync");

        check_cuda_block6(cudaEventCreate(&start), "cudaEventCreate(M5 L2 block6 start)");
        check_cuda_block6(cudaEventCreate(&stop), "cudaEventCreate(M5 L2 block6 stop)");
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_block6(cudaEventRecord(start), "cudaEventRecord(M5 L2 block6 start)");
            m5_l2_block6_csr_spmv_kernel<<<blocks, threads>>>(
                static_cast<std::uint32_t>(block_rows), d_rows, d_cols, d_values, d_x, d_y);
            check_cuda_block6(cudaGetLastError(), "M5 L2 block6 launch");
            check_cuda_block6(cudaEventRecord(stop), "cudaEventRecord(M5 L2 block6 stop)");
            check_cuda_block6(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 L2 block6)");
            float ms = 0.0f;
            check_cuda_block6(cudaEventElapsedTime(&ms, start, stop),
                              "cudaEventElapsedTime(M5 L2 block6)");
            samples.push_back(static_cast<double>(ms));
        }

        GpuM5L2Block6Result result;
        result.block_rows = block_rows;
        result.block_nnz = block_column_indices.size();
        result.y.resize(scalar_rows, 0.0f);
        check_cuda_block6(cudaMemcpy(result.y.data(), d_y, vec_bytes, cudaMemcpyDeviceToHost),
                          "cudaMemcpy(M5 L2 block6 y D2H)");
        result.timing.median_ms = block6_median(samples);
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
