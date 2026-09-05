// M5 GPU productionization stage 7: complete device-resident 4-level V-cycle.
// Include the validated fine-level implementation directly so this staging TU
// can reuse its persistent A0/P0 metadata and work vectors without exposing raw
// device pointers in the production API. Do not link gfss_cuda_operator into
// the corresponding executable.
#include "../src/gpu_m5_fine_level.cu"

#include <cublas_v2.h>

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

void m5_cv_check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

__global__ void m5_cv_l1_zero_start_block_kernel(
    std::uint32_t aggregate_count,
    const DeviceAggregateM5* __restrict__ aggregates,
    const float* __restrict__ inverse_blocks,
    float weight,
    const float* __restrict__ rhs,
    float* __restrict__ x) {
    const std::uint32_t a = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (a >= aggregate_count) return;
    const DeviceAggregateM5* aggregate = aggregates + a;
    if (q >= aggregate->rank) return;
    const float* inverse = inverse_blocks + static_cast<std::size_t>(a) * 36U;
    const std::uint32_t offset = aggregate->coarse_offset;
    float value = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0U; j < 6U; ++j) {
        if (j < aggregate->rank) {
            value = fmaf(inverse[6U * q + j], rhs[offset + j], value);
        }
    }
    x[offset + q] = weight * value;
}

__global__ void m5_cv_l1_post_block_kernel(
    std::uint32_t aggregate_count,
    const DeviceAggregateM5* __restrict__ aggregates,
    const float* __restrict__ inverse_blocks,
    float weight,
    const float* __restrict__ rhs,
    const float* __restrict__ ax,
    float* __restrict__ x) {
    const std::uint32_t a = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (a >= aggregate_count) return;
    const DeviceAggregateM5* aggregate = aggregates + a;
    if (q >= aggregate->rank) return;
    const float* inverse = inverse_blocks + static_cast<std::size_t>(a) * 36U;
    const std::uint32_t offset = aggregate->coarse_offset;
    float value = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0U; j < 6U; ++j) {
        if (j < aggregate->rank) {
            value = fmaf(inverse[6U * q + j], rhs[offset + j] - ax[offset + j], value);
        }
    }
    x[offset + q] = fmaf(weight, value, x[offset + q]);
}

__global__ void m5_cv_vector_residual_kernel(
    std::size_t n,
    const float* __restrict__ rhs,
    const float* __restrict__ ax,
    float* __restrict__ residual) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) residual[i] = rhs[i] - ax[i];
}

__global__ void m5_cv_vector_add_kernel(
    std::size_t n,
    const float* __restrict__ correction,
    float* __restrict__ x) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] += correction[i];
}

__global__ void m5_cv_pack_l1_kernel(
    std::uint32_t aggregate_count,
    const DeviceAggregateM5* __restrict__ aggregates,
    const float* __restrict__ packed,
    float* __restrict__ padded) {
    const std::uint32_t a = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (a >= aggregate_count || q >= 6U) return;
    const DeviceAggregateM5* aggregate = aggregates + a;
    padded[static_cast<std::size_t>(a) * 6U + q] =
        q < aggregate->rank ? packed[aggregate->coarse_offset + q] : 0.0f;
}

__global__ void m5_cv_add_l1_padded_kernel(
    std::uint32_t aggregate_count,
    const DeviceAggregateM5* __restrict__ aggregates,
    const float* __restrict__ padded,
    float* __restrict__ packed) {
    const std::uint32_t a = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (a >= aggregate_count) return;
    const DeviceAggregateM5* aggregate = aggregates + a;
    if (q < aggregate->rank) {
        packed[aggregate->coarse_offset + q] +=
            padded[static_cast<std::size_t>(a) * 6U + q];
    }
}

__global__ void m5_cv_p1_forward_kernel(
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
        const float* block = block_values + static_cast<std::size_t>(p) * 36U;
        const float* xv = x + static_cast<std::size_t>(column_indices[p]) * 6U;
#pragma unroll
        for (std::uint32_t q = 0U; q < 6U; ++q) {
            sum = fmaf(block[lane * 6U + q], xv[q], sum);
        }
    }
    y[static_cast<std::size_t>(row) * 6U + lane] = sum;
}

__global__ void m5_cv_p1t_kernel(
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
    const std::size_t base = static_cast<std::size_t>(first) * 36U;
    float sum = 0.0f;
    for (std::uint32_t local = lane; local < count; local += 32U) {
        const float* xv = x + static_cast<std::size_t>(row_indices[first + local]) * 6U;
#pragma unroll
        for (std::uint32_t r = 0U; r < 6U; ++r) {
            const std::size_t coeff = base +
                (static_cast<std::size_t>(q) * 6U + r) * count + local;
            sum = fmaf(values_q_r_entry[coeff], xv[r], sum);
        }
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, delta);
    }
    if (lane == 0U) y[static_cast<std::size_t>(col) * 6U + q] = sum;
}

__global__ void m5_cv_l2_zero_start_block_kernel(
    std::uint32_t blocks,
    const float* __restrict__ inverse_blocks,
    float weight,
    const float* __restrict__ rhs,
    float* __restrict__ x) {
    const std::uint32_t block_id = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (block_id >= blocks || q >= 6U) return;
    const float* inverse = inverse_blocks + static_cast<std::size_t>(block_id) * 36U;
    const float* b = rhs + static_cast<std::size_t>(block_id) * 6U;
    float value = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0U; j < 6U; ++j) value = fmaf(inverse[6U * q + j], b[j], value);
    x[static_cast<std::size_t>(block_id) * 6U + q] = weight * value;
}

__global__ void m5_cv_l2_post_block_kernel(
    std::uint32_t blocks,
    const float* __restrict__ inverse_blocks,
    float weight,
    const float* __restrict__ rhs,
    const float* __restrict__ ax,
    float* __restrict__ x) {
    const std::uint32_t block_id = blockIdx.x;
    const std::uint32_t q = threadIdx.x;
    if (block_id >= blocks || q >= 6U) return;
    const float* inverse = inverse_blocks + static_cast<std::size_t>(block_id) * 36U;
    const float* b = rhs + static_cast<std::size_t>(block_id) * 6U;
    const float* a = ax + static_cast<std::size_t>(block_id) * 6U;
    float value = 0.0f;
#pragma unroll
    for (std::uint32_t j = 0U; j < 6U; ++j) {
        value = fmaf(inverse[6U * q + j], b[j] - a[j], value);
    }
    x[static_cast<std::size_t>(block_id) * 6U + q] =
        fmaf(weight, value, x[static_cast<std::size_t>(block_id) * 6U + q]);
}

__device__ __forceinline__ float m5_cv_warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, delta);
    }
    return value;
}

__global__ void m5_cv_bottom_solve_kernel(
    std::uint32_t n,
    const float* __restrict__ lower,
    const float* __restrict__ rhs,
    float* __restrict__ x) {
    extern __shared__ float solution[];
    const std::uint32_t lane = threadIdx.x & 31U;
    for (std::uint32_t i = 0U; i < n; ++i) {
        float partial = 0.0f;
        const float* row = lower + static_cast<std::size_t>(i) * n;
        for (std::uint32_t k = lane; k < i; k += 32U) {
            partial = fmaf(row[k], solution[k], partial);
        }
        partial = m5_cv_warp_sum(partial);
        if (lane == 0U) solution[i] = (rhs[i] - partial) / row[i];
        __syncwarp();
    }
    for (std::uint32_t ii = n; ii-- > 0U;) {
        float partial = 0.0f;
        for (std::uint32_t k = ii + 1U + lane; k < n; k += 32U) {
            partial = fmaf(lower[static_cast<std::size_t>(k) * n + ii], solution[k], partial);
        }
        partial = m5_cv_warp_sum(partial);
        if (lane == 0U) {
            const float diagonal = lower[static_cast<std::size_t>(ii) * n + ii];
            solution[ii] = (solution[ii] - partial) / diagonal;
        }
        __syncwarp();
    }
    for (std::uint32_t i = lane; i < n; i += 32U) x[i] = solution[i];
}

GpuM5CompleteVcycleTiming m5_cv_median(const std::vector<GpuM5CompleteVcycleTiming>& s) {
    auto field = [&](auto member) {
        std::vector<double> v;
        v.reserve(s.size());
        for (const auto& x : s) v.push_back(x.*member);
        std::sort(v.begin(), v.end());
        const std::size_t n = v.size();
        return (n & 1U) ? v[n / 2U] : 0.5 * (v[n / 2U - 1U] + v[n / 2U]);
    };
    GpuM5CompleteVcycleTiming out;
    out.l0_down_ms = field(&GpuM5CompleteVcycleTiming::l0_down_ms);
    out.l1_down_ms = field(&GpuM5CompleteVcycleTiming::l1_down_ms);
    out.l2_down_ms = field(&GpuM5CompleteVcycleTiming::l2_down_ms);
    out.l3_solve_ms = field(&GpuM5CompleteVcycleTiming::l3_solve_ms);
    out.l2_up_ms = field(&GpuM5CompleteVcycleTiming::l2_up_ms);
    out.l1_up_ms = field(&GpuM5CompleteVcycleTiming::l1_up_ms);
    out.l0_up_ms = field(&GpuM5CompleteVcycleTiming::l0_up_ms);
    out.total_ms = field(&GpuM5CompleteVcycleTiming::total_ms);
    return out;
}

GpuM5CompleteVcycleTiming m5_cv_best(const std::vector<GpuM5CompleteVcycleTiming>& s) {
    GpuM5CompleteVcycleTiming out;
    out.l0_down_ms = out.l1_down_ms = out.l2_down_ms = out.l3_solve_ms =
        out.l2_up_ms = out.l1_up_ms = out.l0_up_ms = out.total_ms =
            std::numeric_limits<double>::infinity();
    for (const auto& x : s) {
        out.l0_down_ms = std::min(out.l0_down_ms, x.l0_down_ms);
        out.l1_down_ms = std::min(out.l1_down_ms, x.l1_down_ms);
        out.l2_down_ms = std::min(out.l2_down_ms, x.l2_down_ms);
        out.l3_solve_ms = std::min(out.l3_solve_ms, x.l3_solve_ms);
        out.l2_up_ms = std::min(out.l2_up_ms, x.l2_up_ms);
        out.l1_up_ms = std::min(out.l1_up_ms, x.l1_up_ms);
        out.l0_up_ms = std::min(out.l0_up_ms, x.l0_up_ms);
        out.total_ms = std::min(out.total_ms, x.total_ms);
    }
    return out;
}

double m5_cv_elapsed(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    check_cuda_pcg(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime(M5 complete V-cycle)");
    return static_cast<double>(ms);
}

}  // namespace

GpuM5CompleteVcycleResult GpuM5FineLevelContext::complete_vcycle_5x1x1(
    const std::vector<float>& rhs_aos,
    std::size_t l0_smoother_degree,
    std::size_t m0,
    const std::vector<float>& l1_inverse_blocks_6x6,
    double lambda1,
    std::size_t l2_nodes,
    const std::vector<std::uint32_t>& p1_forward_row_offsets,
    const std::vector<std::uint32_t>& p1_forward_column_indices,
    const std::vector<float>& p1_forward_values_6x6,
    const std::vector<std::uint32_t>& p1_transpose_column_offsets,
    const std::vector<std::uint32_t>& p1_transpose_row_indices,
    const std::vector<float>& p1_transpose_values_q_r_entry,
    const std::vector<float>& a2_dense_row_major,
    const std::vector<float>& l2_inverse_blocks_6x6,
    double lambda2,
    const std::vector<float>& p2_dense_row_major,
    std::size_t l3_dofs,
    const std::vector<float>& l3_cholesky_lower_row_major,
    int repeats) {
    if (!impl_) throw std::runtime_error("M5 complete V-cycle context empty");
    if (rhs_aos.size() != impl_->ndof || l0_smoother_degree == 0U ||
        l0_smoother_degree > 32U || m0 == 0U || m0 > 8U || repeats <= 0 ||
        !(lambda1 > 0.0) || !std::isfinite(lambda1) ||
        !(lambda2 > 0.0) || !std::isfinite(lambda2)) {
        throw std::invalid_argument("M5 complete V-cycle scalar/vector options invalid");
    }
    const std::size_t l1_nodes = impl_->aggregate_count;
    const std::size_t l1_dofs = impl_->coarse_dof_count;
    if (l1_inverse_blocks_6x6.size() != l1_nodes * 36U || l2_nodes == 0U) {
        throw std::invalid_argument("M5 complete V-cycle L1 metric shape mismatch");
    }
    const std::size_t l2_dofs = l2_nodes * 6U;
    if (p1_forward_row_offsets.size() != l1_nodes + 1U ||
        p1_transpose_column_offsets.size() != l2_nodes + 1U) {
        throw std::invalid_argument("M5 complete V-cycle P1 index shape mismatch");
    }
    const std::size_t p1_nnz = p1_forward_column_indices.size();
    if (p1_forward_row_offsets.back() != p1_nnz ||
        p1_transpose_column_offsets.back() != p1_nnz ||
        p1_transpose_row_indices.size() != p1_nnz ||
        p1_forward_values_6x6.size() != p1_nnz * 36U ||
        p1_transpose_values_q_r_entry.size() != p1_nnz * 36U) {
        throw std::invalid_argument("M5 complete V-cycle P1 payload mismatch");
    }
    if (a2_dense_row_major.size() != l2_dofs * l2_dofs ||
        l2_inverse_blocks_6x6.size() != l2_nodes * 36U ||
        l3_dofs == 0U || l3_dofs > 256U ||
        p2_dense_row_major.size() != l2_dofs * l3_dofs ||
        l3_cholesky_lower_row_major.size() != l3_dofs * l3_dofs) {
        throw std::invalid_argument("M5 complete V-cycle deep payload mismatch");
    }
    if (l2_dofs > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        l3_dofs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("M5 complete V-cycle cuBLAS dimensions unsupported");
    }
    for (std::size_t i = 0; i < l3_dofs; ++i) {
        if (!(l3_cholesky_lower_row_major[i * l3_dofs + i] > 0.0f)) {
            throw std::invalid_argument("M5 complete V-cycle bottom diagonal invalid");
        }
    }

    std::vector<float> rhs_soa(impl_->ndof, 0.0f);
    for (std::size_t node = 0; node < impl_->nodes; ++node) {
        rhs_soa[node] = rhs_aos[3U * node + 0U];
        rhs_soa[impl_->nodes + node] = rhs_aos[3U * node + 1U];
        rhs_soa[2U * impl_->nodes + node] = rhs_aos[3U * node + 2U];
    }
    const auto l0_weights = impl_->chebyshev_weights(l0_smoother_degree);
    const float omega0 = static_cast<float>(impl_->transfer_omega_value);
    const float weight1 = static_cast<float>(1.0 / (0.55 * lambda1));
    const float weight2 = static_cast<float>(1.0 / (0.55 * lambda2));

    const std::size_t l1_bytes = l1_dofs * sizeof(float);
    const std::size_t l1_padded_bytes = l1_nodes * 6U * sizeof(float);
    const std::size_t l2_bytes = l2_dofs * sizeof(float);
    const std::size_t l3_bytes = l3_dofs * sizeof(float);
    const std::size_t l1_inv_bytes = l1_inverse_blocks_6x6.size() * sizeof(float);
    const std::size_t p1_frow_bytes = p1_forward_row_offsets.size() * sizeof(std::uint32_t);
    const std::size_t p1_fcol_bytes = p1_forward_column_indices.size() * sizeof(std::uint32_t);
    const std::size_t p1_fval_bytes = p1_forward_values_6x6.size() * sizeof(float);
    const std::size_t p1_toff_bytes = p1_transpose_column_offsets.size() * sizeof(std::uint32_t);
    const std::size_t p1_trow_bytes = p1_transpose_row_indices.size() * sizeof(std::uint32_t);
    const std::size_t p1_tval_bytes = p1_transpose_values_q_r_entry.size() * sizeof(float);
    const std::size_t a2_bytes = a2_dense_row_major.size() * sizeof(float);
    const std::size_t l2_inv_bytes = l2_inverse_blocks_6x6.size() * sizeof(float);
    const std::size_t p2_bytes = p2_dense_row_major.size() * sizeof(float);
    const std::size_t bottom_bytes = l3_cholesky_lower_row_major.size() * sizeof(float);

    float* d_l1_inv = nullptr;
    float* d_l1_ax = nullptr;
    float* d_l1_residual = nullptr;
    float* d_l1_padded = nullptr;
    std::uint32_t* d_p1_frows = nullptr;
    std::uint32_t* d_p1_fcols = nullptr;
    float* d_p1_fvals = nullptr;
    std::uint32_t* d_p1_toffs = nullptr;
    std::uint32_t* d_p1_trows = nullptr;
    float* d_p1_tvals = nullptr;
    float* d_a2 = nullptr;
    float* d_l2_inv = nullptr;
    float* d_p2 = nullptr;
    float* d_l2_rhs = nullptr;
    float* d_l2_x = nullptr;
    float* d_l2_ax = nullptr;
    float* d_l2_residual = nullptr;
    float* d_l2_correction = nullptr;
    float* d_l3_rhs = nullptr;
    float* d_l3_x = nullptr;
    float* d_l3_lower = nullptr;
    cublasHandle_t handle{};
    cudaEvent_t events[8]{};

    auto cleanup = [&]() noexcept {
        for (auto& e : events) { if (e) cudaEventDestroy(e); e = nullptr; }
        if (handle) cublasDestroy(handle);
        if (d_l1_inv) cudaFree(d_l1_inv);
        if (d_l1_ax) cudaFree(d_l1_ax);
        if (d_l1_residual) cudaFree(d_l1_residual);
        if (d_l1_padded) cudaFree(d_l1_padded);
        if (d_p1_frows) cudaFree(d_p1_frows);
        if (d_p1_fcols) cudaFree(d_p1_fcols);
        if (d_p1_fvals) cudaFree(d_p1_fvals);
        if (d_p1_toffs) cudaFree(d_p1_toffs);
        if (d_p1_trows) cudaFree(d_p1_trows);
        if (d_p1_tvals) cudaFree(d_p1_tvals);
        if (d_a2) cudaFree(d_a2);
        if (d_l2_inv) cudaFree(d_l2_inv);
        if (d_p2) cudaFree(d_p2);
        if (d_l2_rhs) cudaFree(d_l2_rhs);
        if (d_l2_x) cudaFree(d_l2_x);
        if (d_l2_ax) cudaFree(d_l2_ax);
        if (d_l2_residual) cudaFree(d_l2_residual);
        if (d_l2_correction) cudaFree(d_l2_correction);
        if (d_l3_rhs) cudaFree(d_l3_rhs);
        if (d_l3_x) cudaFree(d_l3_x);
        if (d_l3_lower) cudaFree(d_l3_lower);
    };

    try {
#define M5_CV_MALLOC(ptr, bytes, label) \
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)), (label))
        M5_CV_MALLOC(d_l1_inv, l1_inv_bytes, "cudaMalloc(M5 CV L1 inverse)");
        M5_CV_MALLOC(d_l1_ax, l1_bytes, "cudaMalloc(M5 CV L1 ax)");
        M5_CV_MALLOC(d_l1_residual, l1_bytes, "cudaMalloc(M5 CV L1 residual)");
        M5_CV_MALLOC(d_l1_padded, l1_padded_bytes, "cudaMalloc(M5 CV L1 padded)");
        M5_CV_MALLOC(d_p1_frows, p1_frow_bytes, "cudaMalloc(M5 CV P1 rows)");
        M5_CV_MALLOC(d_p1_fcols, p1_fcol_bytes, "cudaMalloc(M5 CV P1 columns)");
        M5_CV_MALLOC(d_p1_fvals, p1_fval_bytes, "cudaMalloc(M5 CV P1 values)");
        M5_CV_MALLOC(d_p1_toffs, p1_toff_bytes, "cudaMalloc(M5 CV P1T offsets)");
        M5_CV_MALLOC(d_p1_trows, p1_trow_bytes, "cudaMalloc(M5 CV P1T rows)");
        M5_CV_MALLOC(d_p1_tvals, p1_tval_bytes, "cudaMalloc(M5 CV P1T values)");
        M5_CV_MALLOC(d_a2, a2_bytes, "cudaMalloc(M5 CV A2)");
        M5_CV_MALLOC(d_l2_inv, l2_inv_bytes, "cudaMalloc(M5 CV L2 inverse)");
        M5_CV_MALLOC(d_p2, p2_bytes, "cudaMalloc(M5 CV P2)");
        M5_CV_MALLOC(d_l2_rhs, l2_bytes, "cudaMalloc(M5 CV L2 rhs)");
        M5_CV_MALLOC(d_l2_x, l2_bytes, "cudaMalloc(M5 CV L2 x)");
        M5_CV_MALLOC(d_l2_ax, l2_bytes, "cudaMalloc(M5 CV L2 ax)");
        M5_CV_MALLOC(d_l2_residual, l2_bytes, "cudaMalloc(M5 CV L2 residual)");
        M5_CV_MALLOC(d_l2_correction, l2_bytes, "cudaMalloc(M5 CV L2 correction)");
        M5_CV_MALLOC(d_l3_rhs, l3_bytes, "cudaMalloc(M5 CV L3 rhs)");
        M5_CV_MALLOC(d_l3_x, l3_bytes, "cudaMalloc(M5 CV L3 x)");
        M5_CV_MALLOC(d_l3_lower, bottom_bytes, "cudaMalloc(M5 CV L3 factor)");
#undef M5_CV_MALLOC

        check_cuda_pcg(cudaMemcpy(impl_->d_rhs, rhs_soa.data(), impl_->ndof * sizeof(float),
                                  cudaMemcpyHostToDevice), "cudaMemcpy(M5 CV rhs H2D)");
#define M5_CV_COPY(dst, src, bytes, label) \
        check_cuda_pcg(cudaMemcpy((dst), (src).data(), (bytes), cudaMemcpyHostToDevice), (label))
        M5_CV_COPY(d_l1_inv, l1_inverse_blocks_6x6, l1_inv_bytes, "cudaMemcpy(M5 CV L1 inverse H2D)");
        M5_CV_COPY(d_p1_frows, p1_forward_row_offsets, p1_frow_bytes, "cudaMemcpy(M5 CV P1 rows H2D)");
        M5_CV_COPY(d_p1_fcols, p1_forward_column_indices, p1_fcol_bytes, "cudaMemcpy(M5 CV P1 cols H2D)");
        M5_CV_COPY(d_p1_fvals, p1_forward_values_6x6, p1_fval_bytes, "cudaMemcpy(M5 CV P1 vals H2D)");
        M5_CV_COPY(d_p1_toffs, p1_transpose_column_offsets, p1_toff_bytes, "cudaMemcpy(M5 CV P1T offs H2D)");
        M5_CV_COPY(d_p1_trows, p1_transpose_row_indices, p1_trow_bytes, "cudaMemcpy(M5 CV P1T rows H2D)");
        M5_CV_COPY(d_p1_tvals, p1_transpose_values_q_r_entry, p1_tval_bytes, "cudaMemcpy(M5 CV P1T vals H2D)");
        M5_CV_COPY(d_a2, a2_dense_row_major, a2_bytes, "cudaMemcpy(M5 CV A2 H2D)");
        M5_CV_COPY(d_l2_inv, l2_inverse_blocks_6x6, l2_inv_bytes, "cudaMemcpy(M5 CV L2 inverse H2D)");
        M5_CV_COPY(d_p2, p2_dense_row_major, p2_bytes, "cudaMemcpy(M5 CV P2 H2D)");
        M5_CV_COPY(d_l3_lower, l3_cholesky_lower_row_major, bottom_bytes, "cudaMemcpy(M5 CV L3 factor H2D)");
#undef M5_CV_COPY
        m5_cv_check_cublas(cublasCreate(&handle), "cublasCreate(M5 complete V-cycle)");

        constexpr unsigned int vec_threads = 256U;
        const unsigned int l1_vec_blocks = static_cast<unsigned int>(
            (l1_dofs + vec_threads - 1U) / vec_threads);
        const unsigned int l2_vec_blocks = static_cast<unsigned int>(
            (l2_dofs + vec_threads - 1U) / vec_threads);
        constexpr unsigned int transfer_threads = 256U;
        const unsigned int warps_per_block = transfer_threads / 32U;
        const unsigned int p1_blocks = static_cast<unsigned int>(
            (l1_nodes + warps_per_block - 1U) / warps_per_block);
        const unsigned int p1t_blocks = static_cast<unsigned int>(
            (l2_nodes * 6U + warps_per_block - 1U) / warps_per_block);
        const int n2 = static_cast<int>(l2_dofs);
        const int n3 = static_cast<int>(l3_dofs);
        const float alpha = 1.0f;
        const float beta = 0.0f;

        auto launch_a1 = [&](const float* x1, float* y1) {
            m5_launch_p0(impl_->mesh, impl_->nodes, impl_->aggregate_count,
                         impl_->d_aggregate_offsets, impl_->d_aggregate_nodes,
                         impl_->d_aggregates, impl_->d_coordinates, x1, impl_->d_work0);
            for (std::size_t step = 0; step < m0; ++step) {
                launch_pcg_matvec(impl_->mesh, impl_->block_y, impl_->nodes,
                                  impl_->d_work0, impl_->d_work1);
                m5_launch_forward_transfer_update(impl_->mesh, impl_->block_y, impl_->nodes,
                                                   omega0, impl_->d_work1, impl_->d_work0);
            }
            launch_pcg_matvec(impl_->mesh, impl_->block_y, impl_->nodes,
                              impl_->d_work0, impl_->d_work1);
            for (std::size_t step = 0; step < m0; ++step) {
                m5_launch_inverse_scale(impl_->mesh, impl_->block_y, impl_->nodes,
                                        impl_->d_work1, impl_->d_work2);
                launch_pcg_matvec(impl_->mesh, impl_->block_y, impl_->nodes,
                                  impl_->d_work2, impl_->d_work0);
                m5_launch_transpose_update(impl_->mesh, impl_->block_y, impl_->nodes,
                                           omega0, impl_->d_work0, impl_->d_work1);
            }
            m5_launch_p0t(impl_->nodes, impl_->aggregate_count,
                          impl_->d_aggregate_offsets, impl_->d_aggregate_nodes,
                          impl_->d_aggregates, impl_->d_coordinates, impl_->d_work1, y1);
        };
        auto launch_a2 = [&](const float* x2, float* y2) {
            // Row-major symmetric A2 is column-major A2^T == A2.
            m5_cv_check_cublas(cublasSgemv(handle, CUBLAS_OP_N, n2, n2,
                                           &alpha, d_a2, n2, x2, 1, &beta, y2, 1),
                               "cublasSgemv(M5 CV A2)");
        };
        auto launch_p2t = [&](const float* x2, float* y3) {
            // Row-major P2(n2 x n3) is column-major P2^T(n3 x n2).
            m5_cv_check_cublas(cublasSgemv(handle, CUBLAS_OP_N, n3, n2,
                                           &alpha, d_p2, n3, x2, 1, &beta, y3, 1),
                               "cublasSgemv(M5 CV P2T)");
        };
        auto launch_p2 = [&](const float* x3, float* y2) {
            m5_cv_check_cublas(cublasSgemv(handle, CUBLAS_OP_T, n3, n2,
                                           &alpha, d_p2, n3, x3, 1, &beta, y2, 1),
                               "cublasSgemv(M5 CV P2)");
        };

        auto run_once = [&](bool mark) {
            if (mark) check_cuda_pcg(cudaEventRecord(events[0]), "M5 CV event 0");

            // L0 down: degree-5 smoother, true residual, exact P0^T.
            check_cuda_pcg(cudaMemsetAsync(impl_->d_x, 0, impl_->ndof * sizeof(float)),
                           "cudaMemsetAsync(M5 CV x0)");
            for (const float w : l0_weights) {
                launch_pcg_matvec(impl_->mesh, impl_->block_y, impl_->nodes,
                                  impl_->d_x, impl_->d_work0);
                m5_launch_chebyshev_update(impl_->mesh, impl_->block_y, impl_->nodes,
                                           w, impl_->d_rhs, impl_->d_work0, impl_->d_x);
            }
            launch_pcg_matvec(impl_->mesh, impl_->block_y, impl_->nodes,
                              impl_->d_x, impl_->d_work0);
            m5_launch_residual(impl_->mesh, impl_->block_y, impl_->nodes,
                               impl_->d_rhs, impl_->d_work0);
            for (std::size_t step = 0; step < m0; ++step) {
                m5_launch_inverse_scale(impl_->mesh, impl_->block_y, impl_->nodes,
                                        impl_->d_work0, impl_->d_work1);
                launch_pcg_matvec(impl_->mesh, impl_->block_y, impl_->nodes,
                                  impl_->d_work1, impl_->d_work2);
                m5_launch_transpose_update(impl_->mesh, impl_->block_y, impl_->nodes,
                                           omega0, impl_->d_work2, impl_->d_work0);
            }
            m5_launch_p0t(impl_->nodes, impl_->aggregate_count,
                          impl_->d_aggregate_offsets, impl_->d_aggregate_nodes,
                          impl_->d_aggregates, impl_->d_coordinates,
                          impl_->d_work0, impl_->d_coarse);
            if (mark) check_cuda_pcg(cudaEventRecord(events[1]), "M5 CV event 1");

            // L1 down: exact zero-start block step, A1 residual, explicit P1^T.
            m5_cv_l1_zero_start_block_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates, d_l1_inv,
                weight1, impl_->d_coarse, impl_->d_coarse_correction);
            check_cuda_pcg(cudaGetLastError(), "M5 CV L1 pre launch");
            launch_a1(impl_->d_coarse_correction, d_l1_ax);
            m5_cv_vector_residual_kernel<<<l1_vec_blocks, vec_threads>>>(
                l1_dofs, impl_->d_coarse, d_l1_ax, d_l1_residual);
            check_cuda_pcg(cudaGetLastError(), "M5 CV L1 residual launch");
            m5_cv_pack_l1_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates, d_l1_residual, d_l1_padded);
            check_cuda_pcg(cudaGetLastError(), "M5 CV L1 pack launch");
            m5_cv_p1t_kernel<<<p1t_blocks, transfer_threads>>>(
                static_cast<std::uint32_t>(l2_nodes), d_p1_toffs, d_p1_trows,
                d_p1_tvals, d_l1_padded, d_l2_rhs);
            check_cuda_pcg(cudaGetLastError(), "M5 CV P1T launch");
            if (mark) check_cuda_pcg(cudaEventRecord(events[2]), "M5 CV event 2");

            // L2 down: exact zero-start block step, dense A2 residual, dense P2^T.
            m5_cv_l2_zero_start_block_kernel<<<static_cast<unsigned int>(l2_nodes), 32U>>>(
                static_cast<std::uint32_t>(l2_nodes), d_l2_inv, weight2, d_l2_rhs, d_l2_x);
            check_cuda_pcg(cudaGetLastError(), "M5 CV L2 pre launch");
            launch_a2(d_l2_x, d_l2_ax);
            m5_cv_vector_residual_kernel<<<l2_vec_blocks, vec_threads>>>(
                l2_dofs, d_l2_rhs, d_l2_ax, d_l2_residual);
            check_cuda_pcg(cudaGetLastError(), "M5 CV L2 residual launch");
            launch_p2t(d_l2_residual, d_l3_rhs);
            if (mark) check_cuda_pcg(cudaEventRecord(events[3]), "M5 CV event 3");

            // Exact bottom solve.
            m5_cv_bottom_solve_kernel<<<1U, 32U, l3_bytes>>>(
                static_cast<std::uint32_t>(l3_dofs), d_l3_lower, d_l3_rhs, d_l3_x);
            check_cuda_pcg(cudaGetLastError(), "M5 CV bottom solve launch");
            if (mark) check_cuda_pcg(cudaEventRecord(events[4]), "M5 CV event 4");

            // L2 up: dense P2 correction and degree-1 actual-block post step.
            launch_p2(d_l3_x, d_l2_correction);
            m5_cv_vector_add_kernel<<<l2_vec_blocks, vec_threads>>>(
                l2_dofs, d_l2_correction, d_l2_x);
            check_cuda_pcg(cudaGetLastError(), "M5 CV L2 correction launch");
            launch_a2(d_l2_x, d_l2_ax);
            m5_cv_l2_post_block_kernel<<<static_cast<unsigned int>(l2_nodes), 32U>>>(
                static_cast<std::uint32_t>(l2_nodes), d_l2_inv, weight2,
                d_l2_rhs, d_l2_ax, d_l2_x);
            check_cuda_pcg(cudaGetLastError(), "M5 CV L2 post launch");
            if (mark) check_cuda_pcg(cudaEventRecord(events[5]), "M5 CV event 5");

            // L1 up: explicit P1 correction and degree-1 actual-block post step.
            m5_cv_p1_forward_kernel<<<p1_blocks, transfer_threads>>>(
                static_cast<std::uint32_t>(l1_nodes), d_p1_frows, d_p1_fcols,
                d_p1_fvals, d_l2_x, d_l1_padded);
            check_cuda_pcg(cudaGetLastError(), "M5 CV P1 launch");
            m5_cv_add_l1_padded_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates,
                d_l1_padded, impl_->d_coarse_correction);
            check_cuda_pcg(cudaGetLastError(), "M5 CV L1 correction launch");
            launch_a1(impl_->d_coarse_correction, d_l1_ax);
            m5_cv_l1_post_block_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates, d_l1_inv,
                weight1, impl_->d_coarse, d_l1_ax, impl_->d_coarse_correction);
            check_cuda_pcg(cudaGetLastError(), "M5 CV L1 post launch");
            if (mark) check_cuda_pcg(cudaEventRecord(events[6]), "M5 CV event 6");

            // L0 up: exact P0 correction and degree-5 post smoother.
            m5_launch_p0(impl_->mesh, impl_->nodes, impl_->aggregate_count,
                         impl_->d_aggregate_offsets, impl_->d_aggregate_nodes,
                         impl_->d_aggregates, impl_->d_coordinates,
                         impl_->d_coarse_correction, impl_->d_work0);
            for (std::size_t step = 0; step < m0; ++step) {
                launch_pcg_matvec(impl_->mesh, impl_->block_y, impl_->nodes,
                                  impl_->d_work0, impl_->d_work1);
                m5_launch_forward_transfer_update(impl_->mesh, impl_->block_y, impl_->nodes,
                                                   omega0, impl_->d_work1, impl_->d_work0);
            }
            m5_launch_add_correction(impl_->mesh, impl_->block_y, impl_->nodes,
                                     impl_->d_work0, impl_->d_x);
            for (const float w : l0_weights) {
                launch_pcg_matvec(impl_->mesh, impl_->block_y, impl_->nodes,
                                  impl_->d_x, impl_->d_work0);
                m5_launch_chebyshev_update(impl_->mesh, impl_->block_y, impl_->nodes,
                                           w, impl_->d_rhs, impl_->d_work0, impl_->d_x);
            }
            if (mark) check_cuda_pcg(cudaEventRecord(events[7]), "M5 CV event 7");
        };

        run_once(false);
        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(M5 CV warmup)");
        for (auto& event : events) {
            check_cuda_pcg(cudaEventCreate(&event), "cudaEventCreate(M5 complete V-cycle)");
        }
        std::vector<GpuM5CompleteVcycleTiming> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            run_once(true);
            check_cuda_pcg(cudaEventSynchronize(events[7]), "cudaEventSynchronize(M5 CV)");
            GpuM5CompleteVcycleTiming t;
            t.l0_down_ms = m5_cv_elapsed(events[0], events[1]);
            t.l1_down_ms = m5_cv_elapsed(events[1], events[2]);
            t.l2_down_ms = m5_cv_elapsed(events[2], events[3]);
            t.l3_solve_ms = m5_cv_elapsed(events[3], events[4]);
            t.l2_up_ms = m5_cv_elapsed(events[4], events[5]);
            t.l1_up_ms = m5_cv_elapsed(events[5], events[6]);
            t.l0_up_ms = m5_cv_elapsed(events[6], events[7]);
            t.total_ms = m5_cv_elapsed(events[0], events[7]);
            samples.push_back(t);
        }

        GpuM5CompleteVcycleResult result;
        result.fine_correction_aos.assign(impl_->ndof, 0.0f);
        std::vector<float> x_soa(impl_->ndof, 0.0f);
        check_cuda_pcg(cudaMemcpy(x_soa.data(), impl_->d_x, impl_->ndof * sizeof(float),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy(M5 CV x D2H)");
        for (std::size_t node = 0; node < impl_->nodes; ++node) {
            result.fine_correction_aos[3U * node + 0U] = x_soa[node];
            result.fine_correction_aos[3U * node + 1U] = x_soa[impl_->nodes + node];
            result.fine_correction_aos[3U * node + 2U] = x_soa[2U * impl_->nodes + node];
        }
        result.l3_rhs.resize(l3_dofs, 0.0f);
        check_cuda_pcg(cudaMemcpy(result.l3_rhs.data(), d_l3_rhs, l3_bytes,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy(M5 CV L3 rhs D2H)");
        result.median_timing = m5_cv_median(samples);
        result.best_timing = m5_cv_best(samples);
        const std::size_t base_context_bytes = impl_->fine_vector_bytes + impl_->coarse_vector_bytes +
            impl_->aggregation_metadata_bytes + impl_->model_coordinate_bytes;
        const std::size_t extra_bytes = l1_inv_bytes + 2U * l1_bytes + l1_padded_bytes +
            p1_frow_bytes + p1_fcol_bytes + p1_fval_bytes +
            p1_toff_bytes + p1_trow_bytes + p1_tval_bytes +
            a2_bytes + l2_inv_bytes + p2_bytes + 5U * l2_bytes +
            2U * l3_bytes + bottom_bytes;
        result.device_bytes_total = base_context_bytes + extra_bytes;
        result.l1_operator_applies = 2U;
        result.l2_operator_applies = 2U;
        result.l0_operator_applies = 2U * l0_smoother_degree + 1U + 2U * m0 +
                                     result.l1_operator_applies * (2U * m0 + 1U);
        result.bottom_dofs = l3_dofs;
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

}  // namespace gfss
