#include "gfss/gpu_m5_recursive_tail.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
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

void tail_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void tail_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

__global__ void tail_csr_warp_kernel(
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

__global__ void tail_block_zero_start_kernel(
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

__global__ void tail_block_post_kernel(
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

__global__ void tail_residual_kernel(
    std::uint32_t n,
    const float* __restrict__ b,
    const float* __restrict__ ax,
    float* __restrict__ r) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - ax[i];
}

__global__ void tail_add_kernel(
    std::uint32_t n,
    const float* __restrict__ correction,
    float* __restrict__ x) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += correction[i];
}

double tail_median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return (n & 1U) != 0U ? values[n / 2U]
                           : 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

struct DeviceLevel {
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

class TailRuntime {
public:
    TailRuntime(const std::vector<M5RecursiveTailLevelPayload>& host,
                const std::vector<float>& bottom_inverse)
        : levels_(host.size()) {
        if (host.empty()) throw std::invalid_argument("M5 recursive tail requires levels");
        try {
            tail_cublas(cublasCreate(&handle_), "cublasCreate(M5 recursive tail)");
            for (std::size_t i = 0U; i < host.size(); ++i) upload_level(i, host[i]);
            const std::size_t bottom_n = host.back().dofs;
            if (bottom_inverse.size() != bottom_n * bottom_n) {
                throw std::invalid_argument("M5 recursive tail bottom inverse shape mismatch");
            }
            bottom_inverse_bytes_ = bottom_inverse.size() * sizeof(float);
            tail_cuda(cudaMalloc(reinterpret_cast<void**>(&d_bottom_inverse_), bottom_inverse_bytes_),
                      "cudaMalloc(M5 recursive tail bottom inverse)");
            tail_cuda(cudaMemcpy(d_bottom_inverse_, bottom_inverse.data(), bottom_inverse_bytes_,
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy(M5 recursive tail bottom inverse)");
            tail_cuda(cudaEventCreate(&start_), "cudaEventCreate(M5 recursive tail start)");
            tail_cuda(cudaEventCreate(&stop_), "cudaEventCreate(M5 recursive tail stop)");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~TailRuntime() { cleanup(); }

    TailRuntime(const TailRuntime&) = delete;
    TailRuntime& operator=(const TailRuntime&) = delete;

    std::size_t device_bytes() const noexcept {
        std::size_t total = bottom_inverse_bytes_;
        for (const auto& level : levels_) total += level.bytes;
        return total;
    }

    void set_rhs(const std::vector<float>& rhs) {
        if (rhs.size() != levels_.front().dofs) {
            throw std::invalid_argument("M5 recursive tail RHS size mismatch");
        }
        tail_cuda(cudaMemcpy(levels_.front().d_b, rhs.data(), rhs.size() * sizeof(float),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(M5 recursive tail RHS)");
    }

    void cycle() { cycle_level(0U); }

    std::vector<float> result() const {
        std::vector<float> x(levels_.front().dofs, 0.0f);
        tail_cuda(cudaMemcpy(x.data(), levels_.front().d_x, x.size() * sizeof(float),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(M5 recursive tail result)");
        return x;
    }

    double timed_cycle() {
        tail_cuda(cudaEventRecord(start_), "cudaEventRecord(M5 recursive tail start)");
        cycle();
        tail_cuda(cudaEventRecord(stop_), "cudaEventRecord(M5 recursive tail stop)");
        tail_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize(M5 recursive tail)");
        float ms = 0.0f;
        tail_cuda(cudaEventElapsedTime(&ms, start_, stop_),
                  "cudaEventElapsedTime(M5 recursive tail)");
        return static_cast<double>(ms);
    }

private:
    template <class T>
    static void upload_array(T*& dst, const std::vector<T>& src, const char* label,
                             std::size_t& bytes) {
        if (src.empty()) return;
        const std::size_t nbytes = src.size() * sizeof(T);
        tail_cuda(cudaMalloc(reinterpret_cast<void**>(&dst), nbytes), label);
        tail_cuda(cudaMemcpy(dst, src.data(), nbytes, cudaMemcpyHostToDevice), label);
        bytes += nbytes;
    }

    static void alloc_float(float*& dst, std::size_t count, const char* label,
                            std::size_t& bytes) {
        if (count == 0U) return;
        const std::size_t nbytes = count * sizeof(float);
        tail_cuda(cudaMalloc(reinterpret_cast<void**>(&dst), nbytes), label);
        bytes += nbytes;
    }

    void upload_level(std::size_t index, const M5RecursiveTailLevelPayload& h) {
        if (h.dofs == 0U || h.nodes == 0U || h.dofs > std::numeric_limits<std::uint32_t>::max() ||
            h.nodes > std::numeric_limits<std::uint32_t>::max() ||
            h.block_dof_offsets.size() != h.nodes + 1U ||
            h.block_dof_offsets.back() != h.dofs || !(h.lambda > 0.0) || !std::isfinite(h.lambda)) {
            throw std::invalid_argument("M5 recursive tail level metadata invalid");
        }
        DeviceLevel& d = levels_[index];
        d.dofs = h.dofs;
        d.nodes = h.nodes;
        d.next_dofs = h.next_dofs;
        d.kind = h.operator_kind;
        d.weight = static_cast<float>(1.0 / (0.55 * h.lambda));

        if (index + 1U < levels_.size()) {
            if (h.next_dofs == 0U || h.p_row_offsets.size() != h.dofs + 1U ||
                h.pt_row_offsets.size() != h.next_dofs + 1U ||
                h.p_column_indices.size() != h.p_values.size() ||
                h.pt_column_indices.size() != h.pt_values.size() ||
                h.p_row_offsets.back() != h.p_values.size() ||
                h.pt_row_offsets.back() != h.pt_values.size()) {
                throw std::invalid_argument("M5 recursive tail transfer payload invalid");
            }
        } else if (h.next_dofs != 0U) {
            throw std::invalid_argument("M5 recursive tail bottom cannot have next level");
        }

        upload_array(d.d_block_offsets, h.block_dof_offsets,
                     "cudaMalloc/copy(M5 recursive tail block offsets)", d.bytes);
        if (index + 1U < levels_.size()) {
            if (h.inverse_blocks_padded_6x6.size() != h.nodes * 36U) {
                throw std::invalid_argument("M5 recursive tail inverse-block payload invalid");
            }
            upload_array(d.d_inverse, h.inverse_blocks_padded_6x6,
                         "cudaMalloc/copy(M5 recursive tail inverse blocks)", d.bytes);
            if (h.operator_kind == M5RecursiveTailOperatorKind::dense_fp32) {
                if (h.dense_row_major.size() != h.dofs * h.dofs) {
                    throw std::invalid_argument("M5 recursive tail dense operator shape mismatch");
                }
                upload_array(d.d_dense, h.dense_row_major,
                             "cudaMalloc/copy(M5 recursive tail dense operator)", d.bytes);
            } else {
                if (h.csr_row_offsets.size() != h.dofs + 1U ||
                    h.csr_column_indices.size() != h.csr_values.size() ||
                    h.csr_row_offsets.back() != h.csr_values.size()) {
                    throw std::invalid_argument("M5 recursive tail CSR operator shape mismatch");
                }
                upload_array(d.d_csr_rows, h.csr_row_offsets,
                             "cudaMalloc/copy(M5 recursive tail CSR rows)", d.bytes);
                upload_array(d.d_csr_cols, h.csr_column_indices,
                             "cudaMalloc/copy(M5 recursive tail CSR cols)", d.bytes);
                upload_array(d.d_csr_values, h.csr_values,
                             "cudaMalloc/copy(M5 recursive tail CSR values)", d.bytes);
            }
            upload_array(d.d_p_rows, h.p_row_offsets,
                         "cudaMalloc/copy(M5 recursive tail P rows)", d.bytes);
            upload_array(d.d_p_cols, h.p_column_indices,
                         "cudaMalloc/copy(M5 recursive tail P cols)", d.bytes);
            upload_array(d.d_p_values, h.p_values,
                         "cudaMalloc/copy(M5 recursive tail P values)", d.bytes);
            upload_array(d.d_pt_rows, h.pt_row_offsets,
                         "cudaMalloc/copy(M5 recursive tail PT rows)", d.bytes);
            upload_array(d.d_pt_cols, h.pt_column_indices,
                         "cudaMalloc/copy(M5 recursive tail PT cols)", d.bytes);
            upload_array(d.d_pt_values, h.pt_values,
                         "cudaMalloc/copy(M5 recursive tail PT values)", d.bytes);
        }

        alloc_float(d.d_b, h.dofs, "cudaMalloc(M5 recursive tail b)", d.bytes);
        alloc_float(d.d_x, h.dofs, "cudaMalloc(M5 recursive tail x)", d.bytes);
        if (index + 1U < levels_.size()) {
            alloc_float(d.d_ax, h.dofs, "cudaMalloc(M5 recursive tail ax)", d.bytes);
            alloc_float(d.d_r, h.dofs, "cudaMalloc(M5 recursive tail residual)", d.bytes);
            alloc_float(d.d_correction, h.dofs, "cudaMalloc(M5 recursive tail correction)", d.bytes);
        }
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
        tail_csr_warp_kernel<<<csr_blocks(rows), threads>>>(
            static_cast<std::uint32_t>(rows), row_offsets, columns, values, x, y);
        tail_cuda(cudaGetLastError(), label);
    }

    void apply_operator(DeviceLevel& level, const float* x, float* y) {
        if (level.kind == M5RecursiveTailOperatorKind::dense_fp32) {
            const int n = static_cast<int>(level.dofs);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            tail_cublas(cublasSgemv(handle_, CUBLAS_OP_N, n, n,
                                    &alpha, level.d_dense, n, x, 1, &beta, y, 1),
                        "cublasSgemv(M5 recursive tail dense operator)");
        } else {
            launch_csr(level.dofs, level.d_csr_rows, level.d_csr_cols,
                       level.d_csr_values, x, y,
                       "M5 recursive tail CSR operator launch");
        }
    }

    void cycle_level(std::size_t index) {
        DeviceLevel& level = levels_[index];
        if (index + 1U == levels_.size()) {
            const int n = static_cast<int>(level.dofs);
            const float alpha = 1.0f;
            const float beta = 0.0f;
            tail_cublas(cublasSgemv(handle_, CUBLAS_OP_N, n, n,
                                    &alpha, d_bottom_inverse_, n,
                                    level.d_b, 1, &beta, level.d_x, 1),
                        "cublasSgemv(M5 recursive tail bottom inverse)");
            return;
        }

        constexpr unsigned int threads = 256U;
        const unsigned int node_blocks = static_cast<unsigned int>((level.nodes + threads - 1U) / threads);
        tail_block_zero_start_kernel<<<node_blocks, threads>>>(
            static_cast<std::uint32_t>(level.nodes), level.d_block_offsets,
            level.d_inverse, level.weight, level.d_b, level.d_x);
        tail_cuda(cudaGetLastError(), "M5 recursive tail pre-smoother launch");

        apply_operator(level, level.d_x, level.d_ax);
        tail_residual_kernel<<<vector_blocks(level.dofs), threads>>>(
            static_cast<std::uint32_t>(level.dofs), level.d_b, level.d_ax, level.d_r);
        tail_cuda(cudaGetLastError(), "M5 recursive tail residual launch");

        DeviceLevel& next = levels_[index + 1U];
        launch_csr(level.next_dofs, level.d_pt_rows, level.d_pt_cols, level.d_pt_values,
                   level.d_r, next.d_b, "M5 recursive tail restriction launch");
        cycle_level(index + 1U);

        launch_csr(level.dofs, level.d_p_rows, level.d_p_cols, level.d_p_values,
                   next.d_x, level.d_correction, "M5 recursive tail prolongation launch");
        tail_add_kernel<<<vector_blocks(level.dofs), threads>>>(
            static_cast<std::uint32_t>(level.dofs), level.d_correction, level.d_x);
        tail_cuda(cudaGetLastError(), "M5 recursive tail correction launch");

        apply_operator(level, level.d_x, level.d_ax);
        tail_block_post_kernel<<<node_blocks, threads>>>(
            static_cast<std::uint32_t>(level.nodes), level.d_block_offsets,
            level.d_inverse, level.weight, level.d_b, level.d_ax, level.d_x);
        tail_cuda(cudaGetLastError(), "M5 recursive tail post-smoother launch");
    }

    void cleanup() noexcept {
        if (start_) cudaEventDestroy(start_);
        if (stop_) cudaEventDestroy(stop_);
        if (handle_) cublasDestroy(handle_);
        if (d_bottom_inverse_) cudaFree(d_bottom_inverse_);
        start_ = stop_ = nullptr;
        handle_ = nullptr;
        d_bottom_inverse_ = nullptr;
        for (auto& d : levels_) {
#define TAIL_FREE(ptr) do { if (ptr) cudaFree(ptr); ptr = nullptr; } while (0)
            TAIL_FREE(d.d_block_offsets); TAIL_FREE(d.d_inverse); TAIL_FREE(d.d_dense);
            TAIL_FREE(d.d_csr_rows); TAIL_FREE(d.d_csr_cols); TAIL_FREE(d.d_csr_values);
            TAIL_FREE(d.d_p_rows); TAIL_FREE(d.d_p_cols); TAIL_FREE(d.d_p_values);
            TAIL_FREE(d.d_pt_rows); TAIL_FREE(d.d_pt_cols); TAIL_FREE(d.d_pt_values);
            TAIL_FREE(d.d_b); TAIL_FREE(d.d_x); TAIL_FREE(d.d_ax); TAIL_FREE(d.d_r);
            TAIL_FREE(d.d_correction);
#undef TAIL_FREE
        }
    }

    std::vector<DeviceLevel> levels_;
    cublasHandle_t handle_{};
    float* d_bottom_inverse_{nullptr};
    std::size_t bottom_inverse_bytes_{0U};
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

}  // namespace

M5RecursiveTailGpuResult benchmark_m5_recursive_tail_vcycle(
    const std::vector<M5RecursiveTailLevelPayload>& levels,
    const std::vector<float>& bottom_inverse_col_major,
    const std::vector<float>& rhs,
    int repeats) {
    if (repeats <= 0) throw std::invalid_argument("M5 recursive tail repeats must be positive");
    if (levels.empty() || rhs.size() != levels.front().dofs) {
        throw std::invalid_argument("M5 recursive tail benchmark input mismatch");
    }
    for (std::size_t i = 0U; i + 1U < levels.size(); ++i) {
        if (levels[i].next_dofs != levels[i + 1U].dofs) {
            throw std::invalid_argument("M5 recursive tail level chain mismatch");
        }
    }

    TailRuntime runtime(levels, bottom_inverse_col_major);
    runtime.set_rhs(rhs);
    runtime.cycle();
    tail_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(M5 recursive tail warmup)");

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) samples.push_back(runtime.timed_cycle());

    M5RecursiveTailGpuResult out;
    out.x = runtime.result();
    out.median_ms = tail_median(samples);
    out.best_ms = *std::min_element(samples.begin(), samples.end());
    out.device_bytes = runtime.device_bytes();
    out.runtime_representations.reserve(levels.size());
    for (std::size_t i = 0U; i < levels.size(); ++i) {
        if (i + 1U == levels.size()) {
            out.runtime_representations.emplace_back("dense_inverse_fp32");
        } else if (levels[i].operator_kind == M5RecursiveTailOperatorKind::dense_fp32) {
            out.runtime_representations.emplace_back("dense_fp32");
        } else {
            out.runtime_representations.emplace_back("structural_scalar_CSR_fp32");
        }
    }
    return out;
}

}  // namespace gfss
