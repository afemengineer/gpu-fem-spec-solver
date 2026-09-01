#include "gfss/gpu_m5_rectangular_transfer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
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

__global__ void m5_block6_forward_kernel(
    std::uint32_t block_rows,
    const std::uint32_t* __restrict__ row_offsets,
    const std::uint32_t* __restrict__ column_indices,
    const float* __restrict__ block_values,
    const float* __restrict__ x,
    float* __restrict__ y) {
    const std::uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = global_thread >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    if (row >= block_rows || lane >= 6U) return;

    float sum = 0.0f;
    const std::uint32_t first = row_offsets[row];
    const std::uint32_t last = row_offsets[row + 1U];
    for (std::uint32_t p = first; p < last; ++p) {
        const std::uint32_t col = column_indices[p];
        const float* block = block_values + static_cast<std::size_t>(p) * 36U;
        const float* xv = x + static_cast<std::size_t>(col) * 6U;
#pragma unroll
        for (std::uint32_t q = 0U; q < 6U; ++q) {
            sum = fmaf(block[lane * 6U + q], xv[q], sum);
        }
    }
    y[static_cast<std::size_t>(row) * 6U + lane] = sum;
}

// Optimized transpose mapping for the long L2 rows. Six warps own each output
// algebraic node: one warp per output component. The ~hundreds of incident
// blocks are striped across all 32 lanes and reduced in-warp. This preserves
// the shared 6x6 value payload, deterministic gather semantics, and zero atomics
// while exposing the long reduction dimension to the GPU.
__global__ void m5_block6_transpose_component_warp_kernel(
    std::uint32_t block_cols,
    const std::uint32_t* __restrict__ column_offsets,
    const std::uint32_t* __restrict__ row_indices,
    const std::uint32_t* __restrict__ block_ids,
    const float* __restrict__ block_values,
    const float* __restrict__ x,
    float* __restrict__ y) {
    const std::uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t warp = global_thread >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t col = warp / 6U;
    const std::uint32_t component = warp - col * 6U;
    if (col >= block_cols) return;

    const std::uint32_t first = column_offsets[col];
    const std::uint32_t last = column_offsets[col + 1U];
    float sum = 0.0f;
    for (std::uint32_t p = first + lane; p < last; p += 32U) {
        const std::uint32_t row = row_indices[p];
        const std::uint32_t block_id = block_ids[p];
        const float* block = block_values + static_cast<std::size_t>(block_id) * 36U;
        const float* xv = x + static_cast<std::size_t>(row) * 6U;
#pragma unroll
        for (std::uint32_t r = 0U; r < 6U; ++r) {
            sum = fmaf(block[r * 6U + component], xv[r], sum);
        }
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, delta);
    }
    if (lane == 0U) {
        y[static_cast<std::size_t>(col) * 6U + component] = sum;
    }
}

double median_block6(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

}  // namespace

GpuM5Block6TransferResult benchmark_m5_block6_transfer(
    std::size_t block_rows,
    std::size_t block_cols,
    const std::vector<std::uint32_t>& forward_row_offsets,
    const std::vector<std::uint32_t>& forward_column_indices,
    const std::vector<float>& block_values_row_major_6x6,
    const std::vector<std::uint32_t>& transpose_column_offsets,
    const std::vector<std::uint32_t>& transpose_row_indices,
    const std::vector<std::uint32_t>& transpose_block_ids,
    const std::vector<float>& forward_x_padded,
    const std::vector<float>& transpose_x_padded,
    int repeats) {
    if (block_rows == 0U || block_cols == 0U ||
        block_rows > std::numeric_limits<std::uint32_t>::max() ||
        block_cols > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("M5 block6 transfer unsupported dimensions");
    }
    if (forward_row_offsets.size() != block_rows + 1U ||
        transpose_column_offsets.size() != block_cols + 1U ||
        forward_x_padded.size() != block_cols * 6U ||
        transpose_x_padded.size() != block_rows * 6U) {
        throw std::invalid_argument("M5 block6 transfer shape mismatch");
    }
    const std::size_t block_nnz = forward_column_indices.size();
    if (block_values_row_major_6x6.size() != block_nnz * 36U ||
        forward_row_offsets.back() != block_nnz ||
        transpose_row_indices.size() != block_nnz ||
        transpose_block_ids.size() != block_nnz ||
        transpose_column_offsets.back() != block_nnz) {
        throw std::invalid_argument("M5 block6 transfer payload mismatch");
    }
    if (repeats <= 0) throw std::invalid_argument("M5 block6 repeats must be positive");
    for (std::size_t r = 0; r < block_rows; ++r) {
        if (forward_row_offsets[r] > forward_row_offsets[r + 1U]) {
            throw std::invalid_argument("M5 block6 forward offsets not monotone");
        }
    }
    for (std::size_t c = 0; c < block_cols; ++c) {
        if (transpose_column_offsets[c] > transpose_column_offsets[c + 1U]) {
            throw std::invalid_argument("M5 block6 transpose offsets not monotone");
        }
    }
    for (const auto c : forward_column_indices) {
        if (c >= block_cols) throw std::invalid_argument("M5 block6 column out of range");
    }
    for (const auto r : transpose_row_indices) {
        if (r >= block_rows) throw std::invalid_argument("M5 block6 transpose row out of range");
    }
    for (const auto id : transpose_block_ids) {
        if (id >= block_nnz) throw std::invalid_argument("M5 block6 transpose block id out of range");
    }

    const std::size_t frow_bytes = forward_row_offsets.size() * sizeof(std::uint32_t);
    const std::size_t fcol_bytes = forward_column_indices.size() * sizeof(std::uint32_t);
    const std::size_t value_bytes = block_values_row_major_6x6.size() * sizeof(float);
    const std::size_t toff_bytes = transpose_column_offsets.size() * sizeof(std::uint32_t);
    const std::size_t trow_bytes = transpose_row_indices.size() * sizeof(std::uint32_t);
    const std::size_t tid_bytes = transpose_block_ids.size() * sizeof(std::uint32_t);
    const std::size_t forward_x_bytes = block_cols * 6U * sizeof(float);
    const std::size_t forward_y_bytes = block_rows * 6U * sizeof(float);
    const std::size_t transpose_x_bytes = block_rows * 6U * sizeof(float);
    const std::size_t transpose_y_bytes = block_cols * 6U * sizeof(float);

    std::uint32_t* d_frows = nullptr;
    std::uint32_t* d_fcols = nullptr;
    std::uint32_t* d_toffs = nullptr;
    std::uint32_t* d_trows = nullptr;
    std::uint32_t* d_tids = nullptr;
    float* d_values = nullptr;
    float* d_fx = nullptr;
    float* d_fy = nullptr;
    float* d_tx = nullptr;
    float* d_ty = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    try {
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_frows), frow_bytes), "cudaMalloc(M5 block6 forward rows)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_fcols), fcol_bytes), "cudaMalloc(M5 block6 forward columns)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_toffs), toff_bytes), "cudaMalloc(M5 block6 transpose offsets)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_trows), trow_bytes), "cudaMalloc(M5 block6 transpose rows)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_tids), tid_bytes), "cudaMalloc(M5 block6 transpose ids)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_values), value_bytes), "cudaMalloc(M5 block6 values)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_fx), forward_x_bytes), "cudaMalloc(M5 block6 forward x)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_fy), forward_y_bytes), "cudaMalloc(M5 block6 forward y)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_tx), transpose_x_bytes), "cudaMalloc(M5 block6 transpose x)");
        check_cuda_block6(cudaMalloc(reinterpret_cast<void**>(&d_ty), transpose_y_bytes), "cudaMalloc(M5 block6 transpose y)");

        check_cuda_block6(cudaMemcpy(d_frows, forward_row_offsets.data(), frow_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(M5 block6 forward rows H2D)");
        check_cuda_block6(cudaMemcpy(d_fcols, forward_column_indices.data(), fcol_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(M5 block6 forward columns H2D)");
        check_cuda_block6(cudaMemcpy(d_toffs, transpose_column_offsets.data(), toff_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(M5 block6 transpose offsets H2D)");
        check_cuda_block6(cudaMemcpy(d_trows, transpose_row_indices.data(), trow_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(M5 block6 transpose rows H2D)");
        check_cuda_block6(cudaMemcpy(d_tids, transpose_block_ids.data(), tid_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(M5 block6 transpose ids H2D)");
        check_cuda_block6(cudaMemcpy(d_values, block_values_row_major_6x6.data(), value_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(M5 block6 values H2D)");
        check_cuda_block6(cudaMemcpy(d_fx, forward_x_padded.data(), forward_x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(M5 block6 forward x H2D)");
        check_cuda_block6(cudaMemcpy(d_tx, transpose_x_padded.data(), transpose_x_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(M5 block6 transpose x H2D)");

        constexpr unsigned int threads = 256U;
        constexpr unsigned int warps_per_block = threads / 32U;
        const unsigned int forward_blocks = static_cast<unsigned int>(
            (block_rows + warps_per_block - 1U) / warps_per_block);
        const std::size_t transpose_warps = block_cols * 6U;
        const unsigned int transpose_blocks = static_cast<unsigned int>(
            (transpose_warps + warps_per_block - 1U) / warps_per_block);

        m5_block6_forward_kernel<<<forward_blocks, threads>>>(
            static_cast<std::uint32_t>(block_rows), d_frows, d_fcols, d_values, d_fx, d_fy);
        check_cuda_block6(cudaGetLastError(), "M5 block6 forward warmup launch");
        m5_block6_transpose_component_warp_kernel<<<transpose_blocks, threads>>>(
            static_cast<std::uint32_t>(block_cols), d_toffs, d_trows, d_tids,
            d_values, d_tx, d_ty);
        check_cuda_block6(cudaGetLastError(), "M5 block6 transpose warmup launch");
        check_cuda_block6(cudaDeviceSynchronize(), "M5 block6 warmup sync");

        check_cuda_block6(cudaEventCreate(&start), "cudaEventCreate(M5 block6 start)");
        check_cuda_block6(cudaEventCreate(&stop), "cudaEventCreate(M5 block6 stop)");
        std::vector<double> forward_samples;
        std::vector<double> transpose_samples;
        forward_samples.reserve(static_cast<std::size_t>(repeats));
        transpose_samples.reserve(static_cast<std::size_t>(repeats));

        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_block6(cudaEventRecord(start), "cudaEventRecord(M5 block6 forward start)");
            m5_block6_forward_kernel<<<forward_blocks, threads>>>(
                static_cast<std::uint32_t>(block_rows), d_frows, d_fcols, d_values, d_fx, d_fy);
            check_cuda_block6(cudaGetLastError(), "M5 block6 forward launch");
            check_cuda_block6(cudaEventRecord(stop), "cudaEventRecord(M5 block6 forward stop)");
            check_cuda_block6(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 block6 forward)");
            float ms = 0.0f;
            check_cuda_block6(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime(M5 block6 forward)");
            forward_samples.push_back(static_cast<double>(ms));

            check_cuda_block6(cudaEventRecord(start), "cudaEventRecord(M5 block6 transpose start)");
            m5_block6_transpose_component_warp_kernel<<<transpose_blocks, threads>>>(
                static_cast<std::uint32_t>(block_cols), d_toffs, d_trows, d_tids,
                d_values, d_tx, d_ty);
            check_cuda_block6(cudaGetLastError(), "M5 block6 transpose launch");
            check_cuda_block6(cudaEventRecord(stop), "cudaEventRecord(M5 block6 transpose stop)");
            check_cuda_block6(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 block6 transpose)");
            check_cuda_block6(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime(M5 block6 transpose)");
            transpose_samples.push_back(static_cast<double>(ms));
        }

        GpuM5Block6TransferResult result;
        result.block_rows = block_rows;
        result.block_cols = block_cols;
        result.block_nnz = block_nnz;
        result.forward_y_padded.resize(block_rows * 6U, 0.0f);
        result.transpose_y_padded.resize(block_cols * 6U, 0.0f);
        check_cuda_block6(cudaMemcpy(result.forward_y_padded.data(), d_fy, forward_y_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(M5 block6 forward y D2H)");
        check_cuda_block6(cudaMemcpy(result.transpose_y_padded.data(), d_ty, transpose_y_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(M5 block6 transpose y D2H)");
        result.forward_timing.median_ms = median_block6(forward_samples);
        result.forward_timing.best_ms = *std::min_element(forward_samples.begin(), forward_samples.end());
        result.transpose_timing.median_ms = median_block6(transpose_samples);
        result.transpose_timing.best_ms = *std::min_element(transpose_samples.begin(), transpose_samples.end());
        result.device_bytes = frow_bytes + fcol_bytes + value_bytes +
                              toff_bytes + trow_bytes + tid_bytes +
                              forward_x_bytes + forward_y_bytes +
                              transpose_x_bytes + transpose_y_bytes;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_frows);
        cudaFree(d_fcols);
        cudaFree(d_toffs);
        cudaFree(d_trows);
        cudaFree(d_tids);
        cudaFree(d_values);
        cudaFree(d_fx);
        cudaFree(d_fy);
        cudaFree(d_tx);
        cudaFree(d_ty);
        return result;
    } catch (...) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (d_frows) cudaFree(d_frows);
        if (d_fcols) cudaFree(d_fcols);
        if (d_toffs) cudaFree(d_toffs);
        if (d_trows) cudaFree(d_trows);
        if (d_tids) cudaFree(d_tids);
        if (d_values) cudaFree(d_values);
        if (d_fx) cudaFree(d_fx);
        if (d_fy) cudaFree(d_fy);
        if (d_tx) cudaFree(d_tx);
        if (d_ty) cudaFree(d_ty);
        throw;
    }
}

}  // namespace gfss
