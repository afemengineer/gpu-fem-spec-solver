#include "gfss/gpu_m5_recursive_tail.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gfss {
namespace {

void ctx_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void ctx_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

__global__ void ctx_csr_warp_kernel(
    std::uint32_t rows,
    const std::uint32_t* __restrict__ row_offsets,
    const std::uint32_t* __restrict__ column_indices,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y) {
    const std::uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t row = global_thread >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    if (row >= rows) return;

    float sum = 0.0f;
    for (std::uint32_t p = row_offsets[row] + lane; p < row_offsets[row + 1U]; p += 32U) {
        sum = fmaf(values[p], x[column_indices[p]], sum);
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        sum += __shfl_down_sync(0xffffffffU, sum, delta);
    }
    if (lane == 0U) y[row] = sum;
}

__global__ void ctx_block_zero_start_kernel(
    std::uint32_t nodes,
    const std::uint32_t* __restrict__ dof_offsets,
    const float* __restrict__ inverse_padded,
    float weight,
    const float* __restrict__ b,
    float* __restrict__ x) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes) return;
    const std::uint32_t begin = dof_offsets[node];
    const std::uint32_t end = dof_offsets[node + 1U];
    const std::uint32_t rank = end - begin;
    const float* inv = inverse_padded + static_cast<std::size_t>(node) * 36U;
    for (std::uint32_t r = 0U; r < rank; ++r) {
        float value = 0.0f;
        for (std::uint32_t c = 0U; c < rank; ++c) {
            value = fmaf(inv[6U * r + c], b[begin + c], value);
        }
        x[begin + r] = weight * value;
    }
}

__global__ void ctx_block_post_kernel(
    std::uint32_t nodes,
    const std::uint32_t* __restrict__ dof_offsets,
    const float* __restrict__ inverse_padded,
    float weight,
    const float* __restrict__ b,
    const float* __restrict__ ax,
    float* __restrict__ x) {
    const std::uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodes) return;
    const std::uint32_t begin = dof_offsets[node];
    const std::uint32_t end = dof_offsets[node + 1U];
    const std::uint32_t rank = end - begin;
    const float* inv = inverse_padded + static_cast<std::size_t>(node) * 36U;
    for (std::uint32_t r = 0U; r < rank; ++r) {
        float value = 0.0f;
        for (std::uint32_t c = 0U; c < rank; ++c) {
            value = fmaf(inv[6U * r + c], b[begin + c] - ax[begin + c], value);
        }
        x[begin + r] += weight * value;
    }
}

__global__ void ctx_residual_kernel(
    std::uint32_t n,
    const float* __restrict__ b,
    const float* __restrict__ ax,
    float* __restrict__ r) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - ax[i];
}

__global__ void ctx_add_kernel(
    std::uint32_t n,
    const float* __restrict__ correction,
    float* __restrict__ x) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += correction[i];
}

struct ContextDeviceLevel {
    std::size_t dofs{0U};
    std::size_t nodes{0U};
    std::size_t next_dofs{0U};
    M5RecursiveTailOperatorKind kind{M5RecursiveTailOperatorKind::dense_fp32};
    float weight{0.0f};

    std::uint32_t* d_block_offsets{nullptr};
    float* d_inverse{nullptr};
    float* d_dense{nullptr};
    std::uint32_t* d_csr_rows{nullptr};
    std::uint32_t* d_csr_cols{nullptr};
    float* d_csr_values{nullptr};
    std::uint32_t* d_p_rows{nullptr};
    std::uint32_t* d_p_cols{nullptr};
    float* d_p_values{nullptr};
    std::uint32_t* d_pt_rows{nullptr};
    std::uint32_t* d_pt_cols{nullptr};
    float* d_pt_values{nullptr};

    float* d_b{nullptr};
    float* d_x{nullptr};
    float* d_ax{nullptr};
    float* d_r{nullptr};
    float* d_correction{nullptr};
    std::size_t bytes{0U};
};

}  // namespace

struct M5RecursiveTailGpuContext::Impl {
    std::vector<ContextDeviceLevel> levels;
    cublasHandle_t handle{};
    float* d_bottom_inverse{nullptr};
    std::size_t bottom_inverse_bytes{0U};

    Impl(const std::vector<M5RecursiveTailLevelPayload>& host,
         const std::vector<float>& bottom_inverse)
        : levels(host.size()) {
        if (host.empty()) throw std::invalid_argument("M5 recursive tail context requires levels");
        try {
            ctx_cublas(cublasCreate(&handle), "cublasCreate(M5 recursive tail context)");
            for (std::size_t i = 0U; i < host.size(); ++i) upload_level(i, host[i]);
            for (std::size_t i = 0U; i + 1U < host.size(); ++i) {
                if (host[i].next_dofs != host[i + 1U].dofs) {
                    throw std::invalid_argument("M5 recursive tail context chain mismatch");
                }
            }
            const std::size_t bottom_n = host.back().dofs;
            if (bottom_inverse.size() != bottom_n * bottom_n) {
                throw std::invalid_argument("M5 recursive tail context bottom inverse shape mismatch");
            }
            bottom_inverse_bytes = bottom_inverse.size() * sizeof(float);
            ctx_cuda(cudaMalloc(reinterpret_cast<void**>(&d_bottom_inverse), bottom_inverse_bytes),
                     "cudaMalloc(M5 recursive tail context bottom inverse)");
            ctx_cuda(cudaMemcpy(d_bottom_inverse, bottom_inverse.data(), bottom_inverse_bytes,
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy(M5 recursive tail context bottom inverse)");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Impl() { cleanup(); }

    template <class T>
    static void upload_array(T*& dst,
                             const std::vector<T>& src,
                             const char* label,
                             std::size_t& bytes) {
        if (src.empty()) return;
        const std::size_t nbytes = src.size() * sizeof(T);
        ctx_cuda(cudaMalloc(reinterpret_cast<void**>(&dst), nbytes), label);
        ctx_cuda(cudaMemcpy(dst, src.data(), nbytes, cudaMemcpyHostToDevice), label);
        bytes += nbytes;
    }

    static void alloc_float(float*& dst,
                            std::size_t count,
                            const char* label,
                            std::size_t& bytes) {
        if (count == 0U) return;
        const std::size_t nbytes = count * sizeof(float);
        ctx_cuda(cudaMalloc(reinterpret_cast<void**>(&dst), nbytes), label);
        bytes += nbytes;
    }

    void upload_level(std::size_t index, const M5RecursiveTailLevelPayload& h) {
        if (h.dofs == 0U || h.nodes == 0U ||
            h.dofs > std::numeric_limits<std::uint32_t>::max() ||
            h.nodes > std::numeric_limits<std::uint32_t>::max() ||
            h.block_dof_offsets.size() != h.nodes + 1U ||
            h.block_dof_offsets.back() != h.dofs ||
            !(h.lambda > 0.0) || !std::isfinite(h.lambda)) {
            throw std::invalid_argument("M5 recursive tail context level metadata invalid");
        }

        auto& d = levels[index];
        d.dofs = h.dofs;
        d.nodes = h.nodes;
        d.next_dofs = h.next_dofs;
        d.kind = h.operator_kind;
        d.weight = static_cast<float>(1.0 / (0.55 * h.lambda));

        upload_array(d.d_block_offsets, h.block_dof_offsets,
                     "cudaMalloc/copy(M5 recursive tail context block offsets)", d.bytes);
        alloc_float(d.d_b, h.dofs, "cudaMalloc(M5 recursive tail context b)", d.bytes);
        alloc_float(d.d_x, h.dofs, "cudaMalloc(M5 recursive tail context x)", d.bytes);

        const bool bottom = index + 1U == levels.size();
        if (bottom) {
            if (h.next_dofs != 0U) {
                throw std::invalid_argument("M5 recursive tail context bottom has next level");
            }
            return;
        }

        if (h.next_dofs == 0U || h.inverse_blocks_padded_6x6.size() != h.nodes * 36U ||
            h.p_row_offsets.size() != h.dofs + 1U ||
            h.pt_row_offsets.size() != h.next_dofs + 1U ||
            h.p_column_indices.size() != h.p_values.size() ||
            h.pt_column_indices.size() != h.pt_values.size() ||
            h.p_row_offsets.back() != h.p_values.size() ||
            h.pt_row_offsets.back() != h.pt_values.size()) {
            throw std::invalid_argument("M5 recursive tail context non-bottom payload invalid");
        }

        upload_array(d.d_inverse, h.inverse_blocks_padded_6x6,
                     "cudaMalloc/copy(M5 recursive tail context inverse blocks)", d.bytes);
        if (h.operator_kind == M5RecursiveTailOperatorKind::dense_fp32) {
            if (h.dense_row_major.size() != h.dofs * h.dofs) {
                throw std::invalid_argument("M5 recursive tail context dense operator mismatch");
            }
            upload_array(d.d_dense, h.dense_row_major,
                         "cudaMalloc/copy(M5 recursive tail context dense operator)", d.bytes);
        } else {
            if (h.csr_row_offsets.size() != h.dofs + 1U ||
                h.csr_column_indices.size() != h.csr_values.size() ||
                h.csr_row_offsets.back() != h.csr_values.size()) {
                throw std::invalid_argument("M5 recursive tail context CSR operator mismatch");
            }
            upload_array(d.d_csr_rows, h.csr_row_offsets,
                         "cudaMalloc/copy(M5 recursive tail context CSR rows)", d.bytes);
            upload_array(d.d_csr_cols, h.csr_column_indices,
                         "cudaMalloc/copy(M5 recursive tail context CSR cols)", d.bytes);
            upload_array(d.d_csr_values, h.csr_values,
                         "cudaMalloc/copy(M5 recursive tail context CSR values)", d.bytes);
        }
        upload_array(d.d_p_rows, h.p_row_offsets,
                     "cudaMalloc/copy(M5 recursive tail context P rows)", d.bytes);
        upload_array(d.d_p_cols, h.p_column_indices,
                     "cudaMalloc/copy(M5 recursive tail context P cols)", d.bytes);
        upload_array(d.d_p_values, h.p_values,
                     "cudaMalloc/copy(M5 recursive tail context P values)", d.bytes);
        upload_array(d.d_pt_rows, h.pt_row_offsets,
                     "cudaMalloc/copy(M5 recursive tail context PT rows)", d.bytes);
        upload_array(d.d_pt_cols, h.pt_column_indices,
                     "cudaMalloc/copy(M5 recursive tail context PT cols)", d.bytes);
        upload_array(d.d_pt_values, h.pt_values,
                     "cudaMalloc/copy(M5 recursive tail context PT values)", d.bytes);

        alloc_float(d.d_ax, h.dofs, "cudaMalloc(M5 recursive tail context ax)", d.bytes);
        alloc_float(d.d_r, h.dofs, "cudaMalloc(M5 recursive tail context residual)", d.bytes);
        alloc_float(d.d_correction, h.dofs,
                    "cudaMalloc(M5 recursive tail context correction)", d.bytes);
    }

    static unsigned int vector_blocks(std::size_t n) {
        constexpr unsigned int threads = 256U;
        return static_cast<unsigned int>((n + threads - 1U) / threads);
    }

    static unsigned int csr_blocks(std::size_t rows) {
        constexpr unsigned int threads = 256U;
        constexpr unsigned int warps = threads / 32U;
        return static_cast<unsigned int>((rows + warps - 1U) / warps);
    }

    void launch_csr(std::size_t rows,
                    const std::uint32_t* row_offsets,
                    const std::uint32_t* columns,
                    const float* values,
                    const float* x,
                    float* y,
                    const char* label) {
        constexpr unsigned int threads = 256U;
        ctx_csr_warp_kernel<<<csr_blocks(rows), threads>>>(
            static_cast<std::uint32_t>(rows), row_offsets, columns, values, x, y);
        ctx_cuda(cudaGetLastError(), label);
    }

    void apply_operator(ContextDeviceLevel& level, const float* x, float* y) {
        if (level.kind == M5RecursiveTailOperatorKind::dense_fp32) {
            if (level.dofs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("M5 recursive tail context dense dimension exceeds int");
            }
            const int n = static_cast<int>(level.dofs);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            // Host dense payloads are symmetric row-major. cuBLAS interprets
            // them as column-major A^T, which is identical for audited Galerkin A.
            ctx_cublas(cublasSgemv(handle, CUBLAS_OP_N, n, n,
                                   &alpha, level.d_dense, n,
                                   x, 1, &beta, y, 1),
                       "cublasSgemv(M5 recursive tail context dense operator)");
        } else {
            launch_csr(level.dofs, level.d_csr_rows, level.d_csr_cols,
                       level.d_csr_values, x, y,
                       "M5 recursive tail context CSR operator launch");
        }
    }

    void cycle_level(std::size_t index) {
        auto& level = levels[index];
        if (index + 1U == levels.size()) {
            if (level.dofs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("M5 recursive tail context bottom exceeds int");
            }
            const int n = static_cast<int>(level.dofs);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            ctx_cublas(cublasSgemv(handle, CUBLAS_OP_N, n, n,
                                   &alpha, d_bottom_inverse, n,
                                   level.d_b, 1, &beta, level.d_x, 1),
                       "cublasSgemv(M5 recursive tail context bottom inverse)");
            return;
        }

        constexpr unsigned int threads = 256U;
        const unsigned int node_blocks = static_cast<unsigned int>(
            (level.nodes + threads - 1U) / threads);
        ctx_block_zero_start_kernel<<<node_blocks, threads>>>(
            static_cast<std::uint32_t>(level.nodes), level.d_block_offsets,
            level.d_inverse, level.weight, level.d_b, level.d_x);
        ctx_cuda(cudaGetLastError(), "M5 recursive tail context pre-smoother launch");

        apply_operator(level, level.d_x, level.d_ax);
        ctx_residual_kernel<<<vector_blocks(level.dofs), threads>>>(
            static_cast<std::uint32_t>(level.dofs), level.d_b, level.d_ax, level.d_r);
        ctx_cuda(cudaGetLastError(), "M5 recursive tail context residual launch");

        auto& next = levels[index + 1U];
        launch_csr(level.next_dofs, level.d_pt_rows, level.d_pt_cols, level.d_pt_values,
                   level.d_r, next.d_b,
                   "M5 recursive tail context restriction launch");
        cycle_level(index + 1U);

        launch_csr(level.dofs, level.d_p_rows, level.d_p_cols, level.d_p_values,
                   next.d_x, level.d_correction,
                   "M5 recursive tail context prolongation launch");
        ctx_add_kernel<<<vector_blocks(level.dofs), threads>>>(
            static_cast<std::uint32_t>(level.dofs), level.d_correction, level.d_x);
        ctx_cuda(cudaGetLastError(), "M5 recursive tail context correction launch");

        apply_operator(level, level.d_x, level.d_ax);
        ctx_block_post_kernel<<<node_blocks, threads>>>(
            static_cast<std::uint32_t>(level.nodes), level.d_block_offsets,
            level.d_inverse, level.weight, level.d_b, level.d_ax, level.d_x);
        ctx_cuda(cudaGetLastError(), "M5 recursive tail context post-smoother launch");
    }

    void apply_device(const float* d_rhs, float* d_out) {
        if (d_rhs == nullptr || d_out == nullptr) {
            throw std::invalid_argument("M5 recursive tail context null device pointer");
        }
        const std::size_t bytes = levels.front().dofs * sizeof(float);
        ctx_cuda(cudaMemcpyAsync(levels.front().d_b, d_rhs, bytes, cudaMemcpyDeviceToDevice),
                 "cudaMemcpyAsync(M5 recursive tail context RHS D2D)");
        cycle_level(0U);
        ctx_cuda(cudaMemcpyAsync(d_out, levels.front().d_x, bytes, cudaMemcpyDeviceToDevice),
                 "cudaMemcpyAsync(M5 recursive tail context result D2D)");
    }

    std::size_t device_bytes() const noexcept {
        std::size_t total = bottom_inverse_bytes;
        for (const auto& level : levels) total += level.bytes;
        return total;
    }

    void cleanup() noexcept {
        if (handle) cublasDestroy(handle);
        handle = nullptr;
        if (d_bottom_inverse) cudaFree(d_bottom_inverse);
        d_bottom_inverse = nullptr;
        for (auto& d : levels) {
#define CTX_FREE(ptr) do { if (ptr) cudaFree(ptr); ptr = nullptr; } while (0)
            CTX_FREE(d.d_block_offsets); CTX_FREE(d.d_inverse); CTX_FREE(d.d_dense);
            CTX_FREE(d.d_csr_rows); CTX_FREE(d.d_csr_cols); CTX_FREE(d.d_csr_values);
            CTX_FREE(d.d_p_rows); CTX_FREE(d.d_p_cols); CTX_FREE(d.d_p_values);
            CTX_FREE(d.d_pt_rows); CTX_FREE(d.d_pt_cols); CTX_FREE(d.d_pt_values);
            CTX_FREE(d.d_b); CTX_FREE(d.d_x); CTX_FREE(d.d_ax); CTX_FREE(d.d_r);
            CTX_FREE(d.d_correction);
#undef CTX_FREE
        }
    }
};

M5RecursiveTailGpuContext::M5RecursiveTailGpuContext(
    const std::vector<M5RecursiveTailLevelPayload>& levels,
    const std::vector<float>& bottom_inverse_col_major)
    : impl_(std::make_unique<Impl>(levels, bottom_inverse_col_major)) {}

M5RecursiveTailGpuContext::~M5RecursiveTailGpuContext() = default;
M5RecursiveTailGpuContext::M5RecursiveTailGpuContext(M5RecursiveTailGpuContext&&) noexcept = default;
M5RecursiveTailGpuContext& M5RecursiveTailGpuContext::operator=(
    M5RecursiveTailGpuContext&&) noexcept = default;

void M5RecursiveTailGpuContext::apply_device(const float* d_rhs, float* d_x) {
    if (!impl_) throw std::runtime_error("M5 recursive tail context is empty");
    impl_->apply_device(d_rhs, d_x);
}

std::size_t M5RecursiveTailGpuContext::top_dofs() const noexcept {
    return impl_ && !impl_->levels.empty() ? impl_->levels.front().dofs : 0U;
}

std::size_t M5RecursiveTailGpuContext::device_bytes() const noexcept {
    return impl_ ? impl_->device_bytes() : 0U;
}

}  // namespace gfss
