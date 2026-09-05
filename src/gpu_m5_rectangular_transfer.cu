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

// One warp owns one algebraic block row. Only lanes 0..5 are active, each
// producing one scalar component of the padded six-entry output node. This
// deliberately trades lane utilization for very simple, deterministic access
// and no atomics; each active lane performs all 6x6 block FMAs for its row.
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

// Transpose uses the same block-value payload. A separate column-oriented
// index maps each P^T block-row entry to the original P block id and P block
// row. Again one warp owns one output algebraic node and lanes 0..5 produce its
// six scalar components, avoiding atomics and duplicated 6x6 values.
__global__ void m5_block6_transpose_kernel(
    std::uint32_t block_cols,
    const std::uint32_t* __restrict__ column_offsets,
    const std::uint32_t* __restrict__ row_indices,
    const std::uint32_t* __restrict__ block_ids,
    const float* __restrict__ block_values,
    const float* __restrict__ x,
    float* __restrict__ y) {
    const std::uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t col = global_thread >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    if (col >= block_cols || lane >= 6U) return;

    float sum = 0.0f;
    const std::uint32_t first = column_offsets[col];
    const std::uint32_t last = column_offsets[col + 1U];
    for (std::uint32_t p = first; p < last; ++p) {
        const std::uint32_t row = row_indices[p];
        const std::uint32_t block_id = block_ids[p];
        const float* block = block_values + static_cast<std::size_t>(block_id) * 36U;
        const float* xv = x + static_cast<std::size_t>(row) * 6U;
#pragma unroll
        for (std::uint32_t r = 0U; r < 6U; ++r) {
            sum = fmaf(block[r * 6U + lane], xv[r], sum);
        }
    }
    y[static_cast<std::size_t>(col) * 6U + lane] = sum;
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
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_frows), frow_bytes),
                            "cudaMalloc(M5 block6 forward rows)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_fcols), fcol_bytes),
                            "cudaMalloc(M5 block6 forward columns)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_toffs), toff_bytes),
                            "cudaMalloc(M5 block6 transpose offsets)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_trows), trow_bytes),
                            "cudaMalloc(M5 block6 transpose rows)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_tids), tid_bytes),
                            "cudaMalloc(M5 block6 transpose ids)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_values), value_bytes),
                            "cudaMalloc(M5 block6 values)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_fx), forward_x_bytes),
                            "cudaMalloc(M5 block6 forward x)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_fy), forward_y_bytes),
                            "cudaMalloc(M5 block6 forward y)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_tx), transpose_x_bytes),
                            "cudaMalloc(M5 block6 transpose x)");
        check_cuda_transfer(cudaMalloc(reinterpret_cast<void**>(&d_ty), transpose_y_bytes),
                            "cudaMalloc(M5 block6 transpose y)");

        check_cuda_transfer(cudaMemcpy(d_frows, forward_row_offsets.data(), frow_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 block6 forward rows H2D)");
        check_cuda_transfer(cudaMemcpy(d_fcols, forward_column_indices.data(), fcol_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 block6 forward columns H2D)");
        check_cuda_transfer(cudaMemcpy(d_toffs, transpose_column_offsets.data(), toff_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 block6 transpose offsets H2D)");
        check_cuda_transfer(cudaMemcpy(d_trows, transpose_row_indices.data(), trow_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 block6 transpose rows H2D)");
        check_cuda_transfer(cudaMemcpy(d_tids, transpose_block_ids.data(), tid_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 block6 transpose ids H2D)");
        check_cuda_transfer(cudaMemcpy(d_values, block_values_row_major_6x6.data(), value_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 block6 values H2D)");
        check_cuda_transfer(cudaMemcpy(d_fx, forward_x_padded.data(), forward_x_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 block6 forward x H2D)");
        check_cuda_transfer(cudaMemcpy(d_tx, transpose_x_padded.data(), transpose_x_bytes,
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy(M5 block6 transpose x H2D)");

        constexpr unsigned int threads = 256U;
        const unsigned int warps_per_block = threads / 32U;
        const unsigned int forward_blocks = static_cast<unsigned int>(
            (block_rows + warps_per_block - 1U) / warps_per_block);
        const unsigned int transpose_blocks = static_cast<unsigned int>(
            (block_cols + warps_per_block - 1U) / warps_per_block);

        m5_block6_forward_kernel<<<forward_blocks, threads>>>(
            static_cast<std::uint32_t>(block_rows), d_frows, d_fcols,
            d_values, d_fx, d_fy);
        check_cuda_transfer(cudaGetLastError(), "M5 block6 forward warmup launch");
        m5_block6_transpose_kernel<<<transpose_blocks, threads>>>(
            static_cast<std::uint32_t>(block_cols), d_toffs, d_trows, d_tids,
            d_values, d_tx, d_ty);
        check_cuda_transfer(cudaGetLastError(), "M5 block6 transpose warmup launch");
        check_cuda_transfer(cudaDeviceSynchronize(), "M5 block6 warmup sync");

        check_cuda_transfer(cudaEventCreate(&start), "cudaEventCreate(M5 block6 start)");
        check_cuda_transfer(cudaEventCreate(&stop), "cudaEventCreate(M5 block6 stop)");

        std::vector<double> forward_samples;
        std::vector<double> transpose_samples;
        forward_samples.reserve(static_cast<std::size_t>(repeats));
        transpose_samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_transfer(cudaEventRecord(start), "cudaEventRecord(M5 block6 forward start)");
            m5_block6_forward_kernel<<<forward_blocks, threads>>>(
                static_cast<std::uint32_t>(block_rows), d_frows, d_fcols,
                d_values, d_fx, d_fy);
            check_cuda_transfer(cudaGetLastError(), "M5 block6 forward launch");
            check_cuda_transfer(cudaEventRecord(stop), "cudaEventRecord(M5 block6 forward stop)");
            check_cuda_transfer(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 block6 forward)");
            float ms = 0.0f;
            check_cuda_transfer(cudaEventElapsedTime(&ms, start, stop),
                                "cudaEventElapsedTime(M5 block6 forward)");
            forward_samples.push_back(static_cast<double>(ms));

            check_cuda_transfer(cudaEventRecord(start), "cudaEventRecord(M5 block6 transpose start)");
            m5_block6_transpose_kernel<<<transpose_blocks, threads>>>(
                static_cast<std::uint32_t>(block_cols), d_toffs, d_trows, d_tids,
                d_values, d_tx, d_ty);
            check_cuda_transfer(cudaGetLastError(), "M5 block6 transpose launch");
            check_cuda_transfer(cudaEventRecord(stop), "cudaEventRecord(M5 block6 transpose stop)");
            check_cuda_transfer(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 block6 transpose)");
            check_cuda_transfer(cudaEventElapsedTime(&ms, start, stop),
                                "cudaEventElapsedTime(M5 block6 transpose)");
            transpose_samples.push_back(static_cast<double>(ms));
        }

        GpuM5Block6TransferResult result;
        result.block_rows = block_rows;
        result.block_cols = block_cols;
        result.block_nnz = block_nnz;
        result.forward_y_padded.resize(block_rows * 6U, 0.0f);
        result.transpose_y_padded.resize(block_cols * 6U, 0.0f);
        check_cuda_transfer(cudaMemcpy(result.forward_y_padded.data(), d_fy, forward_y_bytes,
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy(M5 block6 forward y D2H)");
        check_cuda_transfer(cudaMemcpy(result.transpose_y_padded.data(), d_ty, transpose_y_bytes,
                                       cudaMemcpyDeviceToHost),
                            "cudaMemcpy(M5 block6 transpose y D2H)");
        result.forward_timing.median_ms = median(forward_samples);
        result.forward_timing.best_ms = *std::min_element(forward_samples.begin(), forward_samples.end());
        result.transpose_timing.median_ms = median(transpose_samples);
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
