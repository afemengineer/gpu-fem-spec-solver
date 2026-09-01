#include "gfss/gpu_m5_rectangular_transfer.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

void check_cuda_block6_soa(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

// Six warps own each L2 algebraic node, one warp per output component q.
// For a column with n incident block entries the value payload is packed as
// [q][r][entry]. Consecutive lanes therefore read consecutive coefficients for
// a fixed r, unlike the shared-forward-value representation where block ids
// cause 144-byte-strided/gathered coefficient accesses.
__global__ void m5_block6_transpose_soa_kernel(
    std::uint32_t block_cols,
    const std::uint32_t* __restrict__ column_offsets,
    const std::uint32_t* __restrict__ row_indices,
    const float* __restrict__ values_q_r_entry,
    const float* __restrict__ x,
    float* __restrict__ y) {
    const std::uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t warp_global = global_thread >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t col = warp_global / 6U;
    const std::uint32_t q = warp_global - col * 6U;
    if (col >= block_cols) return;

    const std::uint32_t first = column_offsets[col];
    const std::uint32_t last = column_offsets[col + 1U];
    const std::uint32_t count = last - first;
    const std::size_t value_base = static_cast<std::size_t>(first) * 36U;

    float sum = 0.0f;
    for (std::uint32_t local = lane; local < count; local += 32U) {
        const std::uint32_t row = row_indices[first + local];
        const float* xv = x + static_cast<std::size_t>(row) * 6U;
#pragma unroll
        for (std::uint32_t r = 0U; r < 6U; ++r) {
            const std::size_t coeff = value_base +
                (static_cast<std::size_t>(q) * 6U + r) * count + local;
            sum = fmaf(values_q_r_entry[coeff], xv[r], sum);
        }
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, delta);
    }
    if (lane == 0U) {
        y[static_cast<std::size_t>(col) * 6U + q] = sum;
    }
}

double median_block6_soa(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

}  // namespace

GpuM5Block6TransposeSoAResult benchmark_m5_block6_transpose_soa(
    std::size_t block_rows,
    std::size_t block_cols,
    const std::vector<std::uint32_t>& transpose_column_offsets,
    const std::vector<std::uint32_t>& transpose_row_indices,
    const std::vector<float>& transpose_values_q_r_entry,
    const std::vector<float>& x_padded,
    int repeats) {
    if (block_rows == 0U || block_cols == 0U ||
        block_rows > std::numeric_limits<std::uint32_t>::max() ||
        block_cols > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("M5 block6 transpose SoA unsupported dimensions");
    }
    if (transpose_column_offsets.size() != block_cols + 1U ||
        x_padded.size() != block_rows * 6U) {
        throw std::invalid_argument("M5 block6 transpose SoA shape mismatch");
    }
    const std::size_t block_nnz = transpose_row_indices.size();
    if (transpose_column_offsets.back() != block_nnz ||
        transpose_values_q_r_entry.size() != block_nnz * 36U) {
        throw std::invalid_argument("M5 block6 transpose SoA payload mismatch");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("M5 block6 transpose SoA repeats must be positive");
    }
    for (std::size_t c = 0; c < block_cols; ++c) {
        if (transpose_column_offsets[c] > transpose_column_offsets[c + 1U]) {
            throw std::invalid_argument("M5 block6 transpose SoA offsets not monotone");
        }
    }
    for (const auto row : transpose_row_indices) {
        if (row >= block_rows) {
            throw std::invalid_argument("M5 block6 transpose SoA row out of range");
        }
    }

    const std::size_t offset_bytes = transpose_column_offsets.size() * sizeof(std::uint32_t);
    const std::size_t row_bytes = transpose_row_indices.size() * sizeof(std::uint32_t);
    const std::size_t value_bytes = transpose_values_q_r_entry.size() * sizeof(float);
    const std::size_t x_bytes = block_rows * 6U * sizeof(float);
    const std::size_t y_bytes = block_cols * 6U * sizeof(float);

    std::uint32_t* d_offsets = nullptr;
    std::uint32_t* d_rows = nullptr;
    float* d_values = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    try {
        check_cuda_block6_soa(cudaMalloc(reinterpret_cast<void**>(&d_offsets), offset_bytes),
                              "cudaMalloc(M5 block6 SoA offsets)");
        check_cuda_block6_soa(cudaMalloc(reinterpret_cast<void**>(&d_rows), row_bytes),
                              "cudaMalloc(M5 block6 SoA rows)");
        check_cuda_block6_soa(cudaMalloc(reinterpret_cast<void**>(&d_values), value_bytes),
                              "cudaMalloc(M5 block6 SoA values)");
        check_cuda_block6_soa(cudaMalloc(reinterpret_cast<void**>(&d_x), x_bytes),
                              "cudaMalloc(M5 block6 SoA x)");
        check_cuda_block6_soa(cudaMalloc(reinterpret_cast<void**>(&d_y), y_bytes),
                              "cudaMalloc(M5 block6 SoA y)");

        check_cuda_block6_soa(cudaMemcpy(d_offsets, transpose_column_offsets.data(), offset_bytes,
                                         cudaMemcpyHostToDevice),
                              "cudaMemcpy(M5 block6 SoA offsets H2D)");
        check_cuda_block6_soa(cudaMemcpy(d_rows, transpose_row_indices.data(), row_bytes,
                                         cudaMemcpyHostToDevice),
                              "cudaMemcpy(M5 block6 SoA rows H2D)");
        check_cuda_block6_soa(cudaMemcpy(d_values, transpose_values_q_r_entry.data(), value_bytes,
                                         cudaMemcpyHostToDevice),
                              "cudaMemcpy(M5 block6 SoA values H2D)");
        check_cuda_block6_soa(cudaMemcpy(d_x, x_padded.data(), x_bytes,
                                         cudaMemcpyHostToDevice),
                              "cudaMemcpy(M5 block6 SoA x H2D)");

        constexpr unsigned int threads = 256U;
        const std::size_t total_warps = block_cols * 6U;
        const unsigned int warps_per_block = threads / 32U;
        const unsigned int blocks = static_cast<unsigned int>(
            (total_warps + warps_per_block - 1U) / warps_per_block);

        m5_block6_transpose_soa_kernel<<<blocks, threads>>>(
            static_cast<std::uint32_t>(block_cols), d_offsets, d_rows,
            d_values, d_x, d_y);
        check_cuda_block6_soa(cudaGetLastError(), "M5 block6 SoA warmup launch");
        check_cuda_block6_soa(cudaDeviceSynchronize(), "M5 block6 SoA warmup sync");

        check_cuda_block6_soa(cudaEventCreate(&start), "cudaEventCreate(M5 block6 SoA start)");
        check_cuda_block6_soa(cudaEventCreate(&stop), "cudaEventCreate(M5 block6 SoA stop)");
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_block6_soa(cudaEventRecord(start),
                                  "cudaEventRecord(M5 block6 SoA start)");
            m5_block6_transpose_soa_kernel<<<blocks, threads>>>(
                static_cast<std::uint32_t>(block_cols), d_offsets, d_rows,
                d_values, d_x, d_y);
            check_cuda_block6_soa(cudaGetLastError(), "M5 block6 SoA launch");
            check_cuda_block6_soa(cudaEventRecord(stop),
                                  "cudaEventRecord(M5 block6 SoA stop)");
            check_cuda_block6_soa(cudaEventSynchronize(stop),
                                  "cudaEventSynchronize(M5 block6 SoA)");
            float ms = 0.0f;
            check_cuda_block6_soa(cudaEventElapsedTime(&ms, start, stop),
                                  "cudaEventElapsedTime(M5 block6 SoA)");
            samples.push_back(static_cast<double>(ms));
        }

        GpuM5Block6TransposeSoAResult result;
        result.block_rows = block_rows;
        result.block_cols = block_cols;
        result.block_nnz = block_nnz;
        result.y_padded.resize(block_cols * 6U, 0.0f);
        check_cuda_block6_soa(cudaMemcpy(result.y_padded.data(), d_y, y_bytes,
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy(M5 block6 SoA y D2H)");
        result.timing.median_ms = median_block6_soa(samples);
        result.timing.best_ms = *std::min_element(samples.begin(), samples.end());
        result.device_bytes = offset_bytes + row_bytes + value_bytes + x_bytes + y_bytes;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_offsets);
        cudaFree(d_rows);
        cudaFree(d_values);
        cudaFree(d_x);
        cudaFree(d_y);
        return result;
    } catch (...) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (d_offsets) cudaFree(d_offsets);
        if (d_rows) cudaFree(d_rows);
        if (d_values) cudaFree(d_values);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        throw;
    }
}

}  // namespace gfss
