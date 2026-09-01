// Standalone staging implementation of the complete L1 V-cycle shell. This TU
// includes the validated factorized A1 implementation directly so the shell can
// keep its packed L1 vectors in the context's existing device buffers. Do not
// link gfss_cuda_operator into the corresponding executable.
#include "../src/gpu_smoothed_aggregation.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gfss {
namespace {

__global__ void m5_l1_pre_block_update_kernel(
    std::uint32_t aggregate_count,
    const DeviceAggregateSa* __restrict__ aggregates,
    const float* __restrict__ inverse_blocks,
    float weight,
    const float* __restrict__ rhs,
    float* __restrict__ x) {
    const std::uint32_t a = blockIdx.x;
    if (a >= aggregate_count) return;
    const DeviceAggregateSa* aggregate = aggregates + a;
    const std::uint32_t q = threadIdx.x;
    if (q >= aggregate->rank) return;
    const std::uint32_t offset = aggregate->coarse_offset;
    const float* inverse = inverse_blocks + 36U * a;
    float value = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0U; j < 6U; ++j) {
        if (j < aggregate->rank) {
            value = fmaf(inverse[6U * q + j], rhs[offset + j], value);
        }
    }
    x[offset + q] = weight * value;
}

__global__ void m5_l1_post_block_update_kernel(
    std::uint32_t aggregate_count,
    const DeviceAggregateSa* __restrict__ aggregates,
    const float* __restrict__ inverse_blocks,
    float weight,
    const float* __restrict__ rhs,
    const float* __restrict__ ax,
    float* __restrict__ x) {
    const std::uint32_t a = blockIdx.x;
    if (a >= aggregate_count) return;
    const DeviceAggregateSa* aggregate = aggregates + a;
    const std::uint32_t q = threadIdx.x;
    if (q >= aggregate->rank) return;
    const std::uint32_t offset = aggregate->coarse_offset;
    const float* inverse = inverse_blocks + 36U * a;
    float value = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0U; j < 6U; ++j) {
        if (j < aggregate->rank) {
            value = fmaf(inverse[6U * q + j],
                         rhs[offset + j] - ax[offset + j], value);
        }
    }
    x[offset + q] = fmaf(weight, value, x[offset + q]);
}

__global__ void m5_l1_residual_kernel(
    std::size_t n,
    const float* __restrict__ rhs,
    const float* __restrict__ ax,
    float* __restrict__ residual) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) residual[i] = rhs[i] - ax[i];
}

__global__ void m5_l1_pack6_kernel(
    std::uint32_t aggregate_count,
    const DeviceAggregateSa* __restrict__ aggregates,
    const float* __restrict__ packed,
    float* __restrict__ padded) {
    const std::uint32_t a = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (a >= aggregate_count || q >= 6U) return;
    const DeviceAggregateSa* aggregate = aggregates + a;
    padded[static_cast<std::size_t>(a) * 6U + q] =
        q < aggregate->rank ? packed[aggregate->coarse_offset + q] : 0.0f;
}

__global__ void m5_l1_add_padded6_kernel(
    std::uint32_t aggregate_count,
    const DeviceAggregateSa* __restrict__ aggregates,
    const float* __restrict__ padded,
    float* __restrict__ packed) {
    const std::uint32_t a = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (a >= aggregate_count) return;
    const DeviceAggregateSa* aggregate = aggregates + a;
    if (q < aggregate->rank) {
        packed[aggregate->coarse_offset + q] +=
            padded[static_cast<std::size_t>(a) * 6U + q];
    }
}

// Forward explicit P1. One warp owns one L1 algebraic node; lanes 0..5 produce
// the six padded components. The forward value payload is in block-row order.
__global__ void m5_l1_p1_forward_block6_kernel(
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

// Exact transpose P1^T using the frozen transpose-ordered [q][r][entry] block
// payload. Six warps own each L2 node, one warp per output component q.
__global__ void m5_l1_p1t_block6_soa_kernel(
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

void launch_m5_l1_pre(
    std::uint32_t aggregate_count,
    const DeviceAggregateSa* aggregates,
    const float* inverse,
    float weight,
    const float* rhs,
    float* x) {
    m5_l1_pre_block_update_kernel<<<aggregate_count, 32U>>>(
        aggregate_count, aggregates, inverse, weight, rhs, x);
    check_cuda_pcg(cudaGetLastError(), "M5 L1 shell pre update launch");
}

void launch_m5_l1_post(
    std::uint32_t aggregate_count,
    const DeviceAggregateSa* aggregates,
    const float* inverse,
    float weight,
    const float* rhs,
    const float* ax,
    float* x) {
    m5_l1_post_block_update_kernel<<<aggregate_count, 32U>>>(
        aggregate_count, aggregates, inverse, weight, rhs, ax, x);
    check_cuda_pcg(cudaGetLastError(), "M5 L1 shell post update launch");
}

double m5_l1_shell_median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

double m5_l1_shell_elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    check_cuda_pcg(cudaEventElapsedTime(&ms, a, b),
                   "cudaEventElapsedTime(M5 L1 shell)");
    return static_cast<double>(ms);
}

}  // namespace

GpuSmoothedAggregationL1ShellResult GpuSmoothedAggregationContext::l1_full_shell(
    const std::vector<float>& rhs,
    const std::vector<float>& inverse_blocks_6x6,
    double lambda_max,
    std::size_t a1_transfer_smoothing_steps,
    std::size_t l2_nodes,
    const std::vector<std::uint32_t>& p1_forward_row_offsets,
    const std::vector<std::uint32_t>& p1_forward_column_indices,
    const std::vector<float>& p1_forward_values_6x6,
    const std::vector<std::uint32_t>& p1_transpose_column_offsets,
    const std::vector<std::uint32_t>& p1_transpose_row_indices,
    const std::vector<float>& p1_transpose_values_q_r_entry,
    const std::vector<float>& external_l2_correction_padded,
    int repeats) {
    if (!impl_) throw std::runtime_error("M5 L1 shell context is empty");
    if (rhs.size() != impl_->coarse_dof_count) {
        throw std::invalid_argument("M5 L1 shell rhs size mismatch");
    }
    if (inverse_blocks_6x6.size() !=
        static_cast<std::size_t>(impl_->aggregate_count) * 36U) {
        throw std::invalid_argument("M5 L1 shell inverse block size mismatch");
    }
    if (!(lambda_max > 0.0) || !std::isfinite(lambda_max) || repeats <= 0 ||
        a1_transfer_smoothing_steps > 8U || l2_nodes == 0U ||
        l2_nodes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("M5 L1 shell invalid scalar options");
    }
    const std::size_t block_rows = impl_->aggregate_count;
    if (p1_forward_row_offsets.size() != block_rows + 1U ||
        p1_transpose_column_offsets.size() != l2_nodes + 1U ||
        external_l2_correction_padded.size() != l2_nodes * 6U) {
        throw std::invalid_argument("M5 L1 shell transfer shape mismatch");
    }
    const std::size_t block_nnz = p1_forward_column_indices.size();
    if (p1_forward_row_offsets.back() != block_nnz ||
        p1_forward_values_6x6.size() != block_nnz * 36U ||
        p1_transpose_column_offsets.back() != block_nnz ||
        p1_transpose_row_indices.size() != block_nnz ||
        p1_transpose_values_q_r_entry.size() != block_nnz * 36U) {
        throw std::invalid_argument("M5 L1 shell transfer payload mismatch");
    }
    for (const auto c : p1_forward_column_indices) {
        if (c >= l2_nodes) throw std::invalid_argument("M5 L1 shell P1 column out of range");
    }
    for (const auto r : p1_transpose_row_indices) {
        if (r >= block_rows) throw std::invalid_argument("M5 L1 shell P1T row out of range");
    }

    const float weight = static_cast<float>(1.0 / (0.55 * lambda_max));
    const std::size_t l1_bytes = impl_->coarse_dof_count * sizeof(float);
    const std::size_t l1_padded_bytes = block_rows * 6U * sizeof(float);
    const std::size_t l2_padded_bytes = l2_nodes * 6U * sizeof(float);
    const std::size_t inverse_bytes = inverse_blocks_6x6.size() * sizeof(float);
    const std::size_t frow_bytes = p1_forward_row_offsets.size() * sizeof(std::uint32_t);
    const std::size_t fcol_bytes = p1_forward_column_indices.size() * sizeof(std::uint32_t);
    const std::size_t fval_bytes = p1_forward_values_6x6.size() * sizeof(float);
    const std::size_t toff_bytes = p1_transpose_column_offsets.size() * sizeof(std::uint32_t);
    const std::size_t trow_bytes = p1_transpose_row_indices.size() * sizeof(std::uint32_t);
    const std::size_t tval_bytes = p1_transpose_values_q_r_entry.size() * sizeof(float);

    float* d_rhs = nullptr;
    float* d_inverse = nullptr;
    float* d_residual = nullptr;
    float* d_l1_padded = nullptr;
    float* d_l2_residual = nullptr;
    float* d_l2_external = nullptr;
    std::uint32_t* d_frows = nullptr;
    std::uint32_t* d_fcols = nullptr;
    float* d_fvals = nullptr;
    std::uint32_t* d_toffs = nullptr;
    std::uint32_t* d_trows = nullptr;
    float* d_tvals = nullptr;
    cudaEvent_t events[8]{};

    auto cleanup = [&]() noexcept {
        for (auto& e : events) {
            if (e) cudaEventDestroy(e);
            e = nullptr;
        }
        if (d_rhs) cudaFree(d_rhs);
        if (d_inverse) cudaFree(d_inverse);
        if (d_residual) cudaFree(d_residual);
        if (d_l1_padded) cudaFree(d_l1_padded);
        if (d_l2_residual) cudaFree(d_l2_residual);
        if (d_l2_external) cudaFree(d_l2_external);
        if (d_frows) cudaFree(d_frows);
        if (d_fcols) cudaFree(d_fcols);
        if (d_fvals) cudaFree(d_fvals);
        if (d_toffs) cudaFree(d_toffs);
        if (d_trows) cudaFree(d_trows);
        if (d_tvals) cudaFree(d_tvals);
    };

    try {
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_rhs), l1_bytes),
                       "cudaMalloc(M5 L1 shell rhs)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_inverse), inverse_bytes),
                       "cudaMalloc(M5 L1 shell inverse)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_residual), l1_bytes),
                       "cudaMalloc(M5 L1 shell residual)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_l1_padded), l1_padded_bytes),
                       "cudaMalloc(M5 L1 shell padded L1)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_l2_residual), l2_padded_bytes),
                       "cudaMalloc(M5 L1 shell L2 residual)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_l2_external), l2_padded_bytes),
                       "cudaMalloc(M5 L1 shell L2 external)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_frows), frow_bytes),
                       "cudaMalloc(M5 L1 shell P1 rows)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_fcols), fcol_bytes),
                       "cudaMalloc(M5 L1 shell P1 columns)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_fvals), fval_bytes),
                       "cudaMalloc(M5 L1 shell P1 values)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_toffs), toff_bytes),
                       "cudaMalloc(M5 L1 shell P1T offsets)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_trows), trow_bytes),
                       "cudaMalloc(M5 L1 shell P1T rows)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_tvals), tval_bytes),
                       "cudaMalloc(M5 L1 shell P1T values)");

        check_cuda_pcg(cudaMemcpy(d_rhs, rhs.data(), l1_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 shell rhs H2D)");
        check_cuda_pcg(cudaMemcpy(d_inverse, inverse_blocks_6x6.data(), inverse_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 shell inverse H2D)");
        check_cuda_pcg(cudaMemcpy(d_l2_external, external_l2_correction_padded.data(),
                                  l2_padded_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 shell external L2 H2D)");
        check_cuda_pcg(cudaMemcpy(d_frows, p1_forward_row_offsets.data(), frow_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 shell P1 rows H2D)");
        check_cuda_pcg(cudaMemcpy(d_fcols, p1_forward_column_indices.data(), fcol_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 shell P1 columns H2D)");
        check_cuda_pcg(cudaMemcpy(d_fvals, p1_forward_values_6x6.data(), fval_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 shell P1 values H2D)");
        check_cuda_pcg(cudaMemcpy(d_toffs, p1_transpose_column_offsets.data(), toff_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 shell P1T offsets H2D)");
        check_cuda_pcg(cudaMemcpy(d_trows, p1_transpose_row_indices.data(), trow_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 shell P1T rows H2D)");
        check_cuda_pcg(cudaMemcpy(d_tvals, p1_transpose_values_q_r_entry.data(), tval_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 L1 shell P1T values H2D)");

        constexpr unsigned int vector_threads = 256U;
        const unsigned int residual_blocks = static_cast<unsigned int>(
            (impl_->coarse_dof_count + vector_threads - 1U) / vector_threads);
        constexpr unsigned int transfer_threads = 256U;
        const unsigned int warps_per_block = transfer_threads / 32U;
        const unsigned int p1_blocks = static_cast<unsigned int>(
            (block_rows + warps_per_block - 1U) / warps_per_block);
        const std::size_t transpose_warps = l2_nodes * 6U;
        const unsigned int p1t_blocks = static_cast<unsigned int>(
            (transpose_warps + warps_per_block - 1U) / warps_per_block);

        auto run_shell = [&]() {
            // Degree-1 pre-smoothing from x=0: x=w*B^-1*b exactly. A1*0 is
            // mathematically zero, so the otherwise required A1 action is elided.
            launch_m5_l1_pre(impl_->aggregate_count, impl_->d_aggregates,
                              d_inverse, weight, d_rhs, impl_->d_coarse_x);

            // True L1 residual using the validated factorized A1 action.
            impl_->run_pipeline(a1_transfer_smoothing_steps, nullptr, nullptr);
            m5_l1_residual_kernel<<<residual_blocks, vector_threads>>>(
                impl_->coarse_dof_count, d_rhs, impl_->d_coarse_y, d_residual);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell residual launch");

            m5_l1_pack6_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates, d_residual, d_l1_padded);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell pack launch");

            m5_l1_p1t_block6_soa_kernel<<<p1t_blocks, transfer_threads>>>(
                static_cast<std::uint32_t>(l2_nodes), d_toffs, d_trows, d_tvals,
                d_l1_padded, d_l2_residual);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell P1T launch");

            m5_l1_p1_forward_block6_kernel<<<p1_blocks, transfer_threads>>>(
                static_cast<std::uint32_t>(block_rows), d_frows, d_fcols, d_fvals,
                d_l2_external, d_l1_padded);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell P1 launch");

            m5_l1_add_padded6_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates,
                d_l1_padded, impl_->d_coarse_x);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell correction launch");

            impl_->run_pipeline(a1_transfer_smoothing_steps, nullptr, nullptr);
            launch_m5_l1_post(impl_->aggregate_count, impl_->d_aggregates,
                               d_inverse, weight, d_rhs,
                               impl_->d_coarse_y, impl_->d_coarse_x);
        };

        run_shell();
        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(M5 L1 shell warmup)");
        for (auto& e : events) {
            check_cuda_pcg(cudaEventCreate(&e), "cudaEventCreate(M5 L1 shell)");
        }

        std::vector<double> pre_samples, residual_samples, pack_samples,
            p1t_samples, p1_samples, correction_samples, post_samples, total_samples;
        const std::size_t nr = static_cast<std::size_t>(repeats);
        pre_samples.reserve(nr); residual_samples.reserve(nr); pack_samples.reserve(nr);
        p1t_samples.reserve(nr); p1_samples.reserve(nr); correction_samples.reserve(nr);
        post_samples.reserve(nr); total_samples.reserve(nr);

        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_pcg(cudaEventRecord(events[0]), "cudaEventRecord(M5 L1 shell start)");
            launch_m5_l1_pre(impl_->aggregate_count, impl_->d_aggregates,
                              d_inverse, weight, d_rhs, impl_->d_coarse_x);
            check_cuda_pcg(cudaEventRecord(events[1]), "cudaEventRecord(M5 L1 shell pre)");

            impl_->run_pipeline(a1_transfer_smoothing_steps, nullptr, nullptr);
            m5_l1_residual_kernel<<<residual_blocks, vector_threads>>>(
                impl_->coarse_dof_count, d_rhs, impl_->d_coarse_y, d_residual);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell residual timed launch");
            check_cuda_pcg(cudaEventRecord(events[2]), "cudaEventRecord(M5 L1 shell residual)");

            m5_l1_pack6_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates, d_residual, d_l1_padded);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell pack timed launch");
            check_cuda_pcg(cudaEventRecord(events[3]), "cudaEventRecord(M5 L1 shell pack)");

            m5_l1_p1t_block6_soa_kernel<<<p1t_blocks, transfer_threads>>>(
                static_cast<std::uint32_t>(l2_nodes), d_toffs, d_trows, d_tvals,
                d_l1_padded, d_l2_residual);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell P1T timed launch");
            check_cuda_pcg(cudaEventRecord(events[4]), "cudaEventRecord(M5 L1 shell P1T)");

            m5_l1_p1_forward_block6_kernel<<<p1_blocks, transfer_threads>>>(
                static_cast<std::uint32_t>(block_rows), d_frows, d_fcols, d_fvals,
                d_l2_external, d_l1_padded);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell P1 timed launch");
            check_cuda_pcg(cudaEventRecord(events[5]), "cudaEventRecord(M5 L1 shell P1)");

            m5_l1_add_padded6_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates,
                d_l1_padded, impl_->d_coarse_x);
            check_cuda_pcg(cudaGetLastError(), "M5 L1 shell correction timed launch");
            check_cuda_pcg(cudaEventRecord(events[6]), "cudaEventRecord(M5 L1 shell correction)");

            impl_->run_pipeline(a1_transfer_smoothing_steps, nullptr, nullptr);
            launch_m5_l1_post(impl_->aggregate_count, impl_->d_aggregates,
                               d_inverse, weight, d_rhs,
                               impl_->d_coarse_y, impl_->d_coarse_x);
            check_cuda_pcg(cudaEventRecord(events[7]), "cudaEventRecord(M5 L1 shell post)");
            check_cuda_pcg(cudaEventSynchronize(events[7]),
                           "cudaEventSynchronize(M5 L1 shell)");

            pre_samples.push_back(m5_l1_shell_elapsed(events[0], events[1]));
            residual_samples.push_back(m5_l1_shell_elapsed(events[1], events[2]));
            pack_samples.push_back(m5_l1_shell_elapsed(events[2], events[3]));
            p1t_samples.push_back(m5_l1_shell_elapsed(events[3], events[4]));
            p1_samples.push_back(m5_l1_shell_elapsed(events[4], events[5]));
            correction_samples.push_back(m5_l1_shell_elapsed(events[5], events[6]));
            post_samples.push_back(m5_l1_shell_elapsed(events[6], events[7]));
            total_samples.push_back(m5_l1_shell_elapsed(events[0], events[7]));
        }

        GpuSmoothedAggregationL1ShellResult result;
        result.final_x.resize(impl_->coarse_dof_count, 0.0f);
        result.l2_residual_padded.resize(l2_nodes * 6U, 0.0f);
        check_cuda_pcg(cudaMemcpy(result.final_x.data(), impl_->d_coarse_x, l1_bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 L1 shell x D2H)");
        check_cuda_pcg(cudaMemcpy(result.l2_residual_padded.data(), d_l2_residual,
                                  l2_padded_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 L1 shell L2 residual D2H)");

        result.median_pre_smooth_ms = m5_l1_shell_median(pre_samples);
        result.median_residual_ms = m5_l1_shell_median(residual_samples);
        result.median_pack_ms = m5_l1_shell_median(pack_samples);
        result.median_p1t_ms = m5_l1_shell_median(p1t_samples);
        result.median_p1_ms = m5_l1_shell_median(p1_samples);
        result.median_correction_ms = m5_l1_shell_median(correction_samples);
        result.median_post_smooth_ms = m5_l1_shell_median(post_samples);
        result.median_total_ms = m5_l1_shell_median(total_samples);
        result.best_total_ms = *std::min_element(total_samples.begin(), total_samples.end());
        result.mathematical_a1_applies = 3U;
        result.executed_a1_applies = 2U;
        result.zero_start_pre_a1_elided = true;
        result.persistent_l1_shell_bytes =
            impl_->fine_workspace_bytes + impl_->coarse_workspace_bytes +
            impl_->aggregation_metadata_bytes + impl_->model_coordinate_bytes +
            l1_bytes + inverse_bytes + l1_bytes + l1_padded_bytes +
            2U * l2_padded_bytes + frow_bytes + fcol_bytes + fval_bytes +
            toff_bytes + trow_bytes + tval_bytes;

        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

}  // namespace gfss
