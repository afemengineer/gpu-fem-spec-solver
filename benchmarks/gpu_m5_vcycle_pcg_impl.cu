// M5 GPU productionization stage 8: fixed-iteration fully device-resident PCG
// using the validated complete 5x1x1 V-cycle as M^-1. Include the complete-cycle
// staging TU so this solver reuses exactly the same kernels and hierarchy
// representations. No host scalar reductions or H2D/D2H occur inside a timed
// solve; the final solution is copied back only for the independent FP64 oracle.
#include "gpu_m5_complete_vcycle_impl.cu"

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

__global__ void m5_vpcg_update_x_r_kernel(
    std::size_t n,
    const float* __restrict__ rz,
    const float* __restrict__ pap,
    const float* __restrict__ p,
    const float* __restrict__ ap,
    float* __restrict__ x,
    float* __restrict__ r,
    int* __restrict__ breakdown) {
    const float rzv = *rz;
    const float papv = *pap;
    if (!(rzv > 0.0f) || !(papv > 0.0f) || !isfinite(rzv) || !isfinite(papv)) {
        if (blockIdx.x == 0U && threadIdx.x == 0U) atomicExch(breakdown, 1);
        return;
    }
    const float alpha = rzv / papv;
    if (!isfinite(alpha)) {
        if (blockIdx.x == 0U && threadIdx.x == 0U) atomicExch(breakdown, 1);
        return;
    }
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = fmaf(alpha, p[i], x[i]);
        r[i] = fmaf(-alpha, ap[i], r[i]);
    }
}

__global__ void m5_vpcg_update_p_kernel(
    std::size_t n,
    const float* __restrict__ rz_new,
    const float* __restrict__ rz_old,
    const float* __restrict__ z,
    float* __restrict__ p,
    int* __restrict__ breakdown) {
    const float newv = *rz_new;
    const float oldv = *rz_old;
    if (!(newv > 0.0f) || !(oldv > 0.0f) || !isfinite(newv) || !isfinite(oldv)) {
        if (blockIdx.x == 0U && threadIdx.x == 0U) atomicExch(breakdown, 1);
        return;
    }
    const float beta = newv / oldv;
    if (!(beta >= 0.0f) || !isfinite(beta)) {
        if (blockIdx.x == 0U && threadIdx.x == 0U) atomicExch(breakdown, 1);
        return;
    }
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = fmaf(beta, p[i], z[i]);
}

double m5_vpcg_median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    return (n & 1U) ? values[n / 2U] : 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

}  // namespace

GpuM5VcyclePcgResult GpuM5FineLevelContext::solve_pcg_vcycle_5x1x1_fixed(
    const std::vector<float>& rhs_aos,
    std::size_t iterations,
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
    if (!impl_) throw std::runtime_error("M5 V-cycle PCG context empty");
    if (rhs_aos.size() != impl_->ndof || iterations == 0U || iterations > 256U ||
        l0_smoother_degree == 0U || l0_smoother_degree > 32U ||
        m0 == 0U || m0 > 8U || repeats <= 0 ||
        !(lambda1 > 0.0) || !std::isfinite(lambda1) ||
        !(lambda2 > 0.0) || !std::isfinite(lambda2)) {
        throw std::invalid_argument("M5 V-cycle PCG scalar/vector options invalid");
    }

    const std::size_t l1_nodes = impl_->aggregate_count;
    const std::size_t l1_dofs = impl_->coarse_dof_count;
    if (l1_inverse_blocks_6x6.size() != l1_nodes * 36U || l2_nodes == 0U) {
        throw std::invalid_argument("M5 V-cycle PCG L1 metric shape mismatch");
    }
    const std::size_t l2_dofs = l2_nodes * 6U;
    if (p1_forward_row_offsets.size() != l1_nodes + 1U ||
        p1_transpose_column_offsets.size() != l2_nodes + 1U) {
        throw std::invalid_argument("M5 V-cycle PCG P1 index shape mismatch");
    }
    const std::size_t p1_nnz = p1_forward_column_indices.size();
    if (p1_forward_row_offsets.back() != p1_nnz ||
        p1_transpose_column_offsets.back() != p1_nnz ||
        p1_transpose_row_indices.size() != p1_nnz ||
        p1_forward_values_6x6.size() != p1_nnz * 36U ||
        p1_transpose_values_q_r_entry.size() != p1_nnz * 36U) {
        throw std::invalid_argument("M5 V-cycle PCG P1 payload mismatch");
    }
    if (a2_dense_row_major.size() != l2_dofs * l2_dofs ||
        l2_inverse_blocks_6x6.size() != l2_nodes * 36U ||
        l3_dofs == 0U || l3_dofs > 256U ||
        p2_dense_row_major.size() != l2_dofs * l3_dofs ||
        l3_cholesky_lower_row_major.size() != l3_dofs * l3_dofs) {
        throw std::invalid_argument("M5 V-cycle PCG deep payload mismatch");
    }
    if (l2_dofs > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        l3_dofs > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        impl_->ndof > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("M5 V-cycle PCG cuBLAS dimensions unsupported");
    }
    for (std::size_t i = 0; i < l3_dofs; ++i) {
        if (!(l3_cholesky_lower_row_major[i * l3_dofs + i] > 0.0f)) {
            throw std::invalid_argument("M5 V-cycle PCG bottom diagonal invalid");
        }
    }

    std::vector<float> rhs_soa(impl_->ndof, 0.0f);
    double rhs_norm2 = 0.0;
    for (std::size_t node = 0; node < impl_->nodes; ++node) {
        for (std::size_t c = 0; c < 3U; ++c) {
            const float value = rhs_aos[3U * node + c];
            rhs_soa[c * impl_->nodes + node] = value;
            rhs_norm2 += static_cast<double>(value) * static_cast<double>(value);
        }
    }
    const double rhs_norm = std::sqrt(rhs_norm2);
    if (!(rhs_norm > 0.0)) throw std::invalid_argument("M5 V-cycle PCG RHS is zero");

    const auto l0_weights = impl_->chebyshev_weights(l0_smoother_degree);
    const float omega0 = static_cast<float>(impl_->transfer_omega_value);
    const float weight1 = static_cast<float>(1.0 / (0.55 * lambda1));
    const float weight2 = static_cast<float>(1.0 / (0.55 * lambda2));

    const std::size_t fine_bytes = impl_->ndof * sizeof(float);
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

    float* d_b = nullptr;
    float* d_solution = nullptr;
    float* d_r = nullptr;
    float* d_z = nullptr;
    float* d_p = nullptr;
    float* d_ap = nullptr;
    float* d_rz0 = nullptr;
    float* d_rz1 = nullptr;
    float* d_pap = nullptr;
    float* d_recursive_norm = nullptr;
    int* d_breakdown = nullptr;

    cublasHandle_t prec_handle{};
    cublasHandle_t dot_handle{};
    cudaEvent_t start{};
    cudaEvent_t stop{};

    auto cleanup = [&]() noexcept {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (prec_handle) cublasDestroy(prec_handle);
        if (dot_handle) cublasDestroy(dot_handle);
#define M5_VPCG_FREE(ptr) do { if (ptr) cudaFree(ptr); ptr = nullptr; } while (0)
        M5_VPCG_FREE(d_l1_inv); M5_VPCG_FREE(d_l1_ax); M5_VPCG_FREE(d_l1_residual);
        M5_VPCG_FREE(d_l1_padded); M5_VPCG_FREE(d_p1_frows); M5_VPCG_FREE(d_p1_fcols);
        M5_VPCG_FREE(d_p1_fvals); M5_VPCG_FREE(d_p1_toffs); M5_VPCG_FREE(d_p1_trows);
        M5_VPCG_FREE(d_p1_tvals); M5_VPCG_FREE(d_a2); M5_VPCG_FREE(d_l2_inv);
        M5_VPCG_FREE(d_p2); M5_VPCG_FREE(d_l2_rhs); M5_VPCG_FREE(d_l2_x);
        M5_VPCG_FREE(d_l2_ax); M5_VPCG_FREE(d_l2_residual); M5_VPCG_FREE(d_l2_correction);
        M5_VPCG_FREE(d_l3_rhs); M5_VPCG_FREE(d_l3_x); M5_VPCG_FREE(d_l3_lower);
        M5_VPCG_FREE(d_b); M5_VPCG_FREE(d_solution); M5_VPCG_FREE(d_r); M5_VPCG_FREE(d_z);
        M5_VPCG_FREE(d_p); M5_VPCG_FREE(d_ap); M5_VPCG_FREE(d_rz0); M5_VPCG_FREE(d_rz1);
        M5_VPCG_FREE(d_pap); M5_VPCG_FREE(d_recursive_norm); M5_VPCG_FREE(d_breakdown);
#undef M5_VPCG_FREE
    };

    try {
#define M5_VPCG_MALLOC(ptr, bytes, label) \
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)), (label))
        M5_VPCG_MALLOC(d_l1_inv, l1_inv_bytes, "cudaMalloc(M5 VPCG L1 inverse)");
        M5_VPCG_MALLOC(d_l1_ax, l1_bytes, "cudaMalloc(M5 VPCG L1 ax)");
        M5_VPCG_MALLOC(d_l1_residual, l1_bytes, "cudaMalloc(M5 VPCG L1 residual)");
        M5_VPCG_MALLOC(d_l1_padded, l1_padded_bytes, "cudaMalloc(M5 VPCG L1 padded)");
        M5_VPCG_MALLOC(d_p1_frows, p1_frow_bytes, "cudaMalloc(M5 VPCG P1 rows)");
        M5_VPCG_MALLOC(d_p1_fcols, p1_fcol_bytes, "cudaMalloc(M5 VPCG P1 cols)");
        M5_VPCG_MALLOC(d_p1_fvals, p1_fval_bytes, "cudaMalloc(M5 VPCG P1 vals)");
        M5_VPCG_MALLOC(d_p1_toffs, p1_toff_bytes, "cudaMalloc(M5 VPCG P1T offs)");
        M5_VPCG_MALLOC(d_p1_trows, p1_trow_bytes, "cudaMalloc(M5 VPCG P1T rows)");
        M5_VPCG_MALLOC(d_p1_tvals, p1_tval_bytes, "cudaMalloc(M5 VPCG P1T vals)");
        M5_VPCG_MALLOC(d_a2, a2_bytes, "cudaMalloc(M5 VPCG A2)");
        M5_VPCG_MALLOC(d_l2_inv, l2_inv_bytes, "cudaMalloc(M5 VPCG L2 inverse)");
        M5_VPCG_MALLOC(d_p2, p2_bytes, "cudaMalloc(M5 VPCG P2)");
        M5_VPCG_MALLOC(d_l2_rhs, l2_bytes, "cudaMalloc(M5 VPCG L2 rhs)");
        M5_VPCG_MALLOC(d_l2_x, l2_bytes, "cudaMalloc(M5 VPCG L2 x)");
        M5_VPCG_MALLOC(d_l2_ax, l2_bytes, "cudaMalloc(M5 VPCG L2 ax)");
        M5_VPCG_MALLOC(d_l2_residual, l2_bytes, "cudaMalloc(M5 VPCG L2 residual)");
        M5_VPCG_MALLOC(d_l2_correction, l2_bytes, "cudaMalloc(M5 VPCG L2 correction)");
        M5_VPCG_MALLOC(d_l3_rhs, l3_bytes, "cudaMalloc(M5 VPCG L3 rhs)");
        M5_VPCG_MALLOC(d_l3_x, l3_bytes, "cudaMalloc(M5 VPCG L3 x)");
        M5_VPCG_MALLOC(d_l3_lower, bottom_bytes, "cudaMalloc(M5 VPCG L3 factor)");
        M5_VPCG_MALLOC(d_b, fine_bytes, "cudaMalloc(M5 VPCG b)");
        M5_VPCG_MALLOC(d_solution, fine_bytes, "cudaMalloc(M5 VPCG solution)");
        M5_VPCG_MALLOC(d_r, fine_bytes, "cudaMalloc(M5 VPCG residual)");
        M5_VPCG_MALLOC(d_z, fine_bytes, "cudaMalloc(M5 VPCG z)");
        M5_VPCG_MALLOC(d_p, fine_bytes, "cudaMalloc(M5 VPCG p)");
        M5_VPCG_MALLOC(d_ap, fine_bytes, "cudaMalloc(M5 VPCG Ap)");
        M5_VPCG_MALLOC(d_rz0, sizeof(float), "cudaMalloc(M5 VPCG rz0)");
        M5_VPCG_MALLOC(d_rz1, sizeof(float), "cudaMalloc(M5 VPCG rz1)");
        M5_VPCG_MALLOC(d_pap, sizeof(float), "cudaMalloc(M5 VPCG pAp)");
        M5_VPCG_MALLOC(d_recursive_norm, sizeof(float), "cudaMalloc(M5 VPCG residual norm)");
        M5_VPCG_MALLOC(d_breakdown, sizeof(int), "cudaMalloc(M5 VPCG breakdown)");
#undef M5_VPCG_MALLOC

        check_cuda_pcg(cudaMemcpy(d_b, rhs_soa.data(), fine_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 VPCG b H2D)");
#define M5_VPCG_COPY(dst, src, bytes, label) \
        check_cuda_pcg(cudaMemcpy((dst), (src).data(), (bytes), cudaMemcpyHostToDevice), (label))
        M5_VPCG_COPY(d_l1_inv, l1_inverse_blocks_6x6, l1_inv_bytes, "cudaMemcpy(M5 VPCG L1 inverse)");
        M5_VPCG_COPY(d_p1_frows, p1_forward_row_offsets, p1_frow_bytes, "cudaMemcpy(M5 VPCG P1 rows)");
        M5_VPCG_COPY(d_p1_fcols, p1_forward_column_indices, p1_fcol_bytes, "cudaMemcpy(M5 VPCG P1 cols)");
        M5_VPCG_COPY(d_p1_fvals, p1_forward_values_6x6, p1_fval_bytes, "cudaMemcpy(M5 VPCG P1 vals)");
        M5_VPCG_COPY(d_p1_toffs, p1_transpose_column_offsets, p1_toff_bytes, "cudaMemcpy(M5 VPCG P1T offs)");
        M5_VPCG_COPY(d_p1_trows, p1_transpose_row_indices, p1_trow_bytes, "cudaMemcpy(M5 VPCG P1T rows)");
        M5_VPCG_COPY(d_p1_tvals, p1_transpose_values_q_r_entry, p1_tval_bytes, "cudaMemcpy(M5 VPCG P1T vals)");
        M5_VPCG_COPY(d_a2, a2_dense_row_major, a2_bytes, "cudaMemcpy(M5 VPCG A2)");
        M5_VPCG_COPY(d_l2_inv, l2_inverse_blocks_6x6, l2_inv_bytes, "cudaMemcpy(M5 VPCG L2 inverse)");
        M5_VPCG_COPY(d_p2, p2_dense_row_major, p2_bytes, "cudaMemcpy(M5 VPCG P2)");
        M5_VPCG_COPY(d_l3_lower, l3_cholesky_lower_row_major, bottom_bytes, "cudaMemcpy(M5 VPCG L3 factor)");
#undef M5_VPCG_COPY

        m5_cv_check_cublas(cublasCreate(&prec_handle), "cublasCreate(M5 VPCG preconditioner)");
        m5_cv_check_cublas(cublasCreate(&dot_handle), "cublasCreate(M5 VPCG dot)");
        m5_cv_check_cublas(cublasSetPointerMode(dot_handle, CUBLAS_POINTER_MODE_DEVICE),
                           "cublasSetPointerMode(M5 VPCG device)");

        constexpr unsigned int vec_threads = 256U;
        const unsigned int fine_vec_blocks = static_cast<unsigned int>(
            (impl_->ndof + vec_threads - 1U) / vec_threads);
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
        const int n0 = static_cast<int>(impl_->ndof);
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
            m5_cv_check_cublas(cublasSgemv(prec_handle, CUBLAS_OP_N, n2, n2,
                                           &alpha, d_a2, n2, x2, 1, &beta, y2, 1),
                               "cublasSgemv(M5 VPCG A2)");
        };
        auto launch_p2t = [&](const float* x2, float* y3) {
            m5_cv_check_cublas(cublasSgemv(prec_handle, CUBLAS_OP_N, n3, n2,
                                           &alpha, d_p2, n3, x2, 1, &beta, y3, 1),
                               "cublasSgemv(M5 VPCG P2T)");
        };
        auto launch_p2 = [&](const float* x3, float* y2) {
            m5_cv_check_cublas(cublasSgemv(prec_handle, CUBLAS_OP_T, n3, n2,
                                           &alpha, d_p2, n3, x3, 1, &beta, y2, 1),
                               "cublasSgemv(M5 VPCG P2)");
        };

        auto apply_preconditioner = [&](const float* residual, float* z) {
            check_cuda_pcg(cudaMemcpyAsync(impl_->d_rhs, residual, fine_bytes,
                                           cudaMemcpyDeviceToDevice),
                           "cudaMemcpyAsync(M5 VPCG residual to V-cycle rhs)");

            // L0 down.
            check_cuda_pcg(cudaMemsetAsync(impl_->d_x, 0, fine_bytes),
                           "cudaMemsetAsync(M5 VPCG preconditioner x0)");
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

            // L1 down.
            m5_cv_l1_zero_start_block_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates, d_l1_inv,
                weight1, impl_->d_coarse, impl_->d_coarse_correction);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG L1 pre launch");
            launch_a1(impl_->d_coarse_correction, d_l1_ax);
            m5_cv_vector_residual_kernel<<<l1_vec_blocks, vec_threads>>>(
                l1_dofs, impl_->d_coarse, d_l1_ax, d_l1_residual);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG L1 residual launch");
            m5_cv_pack_l1_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates, d_l1_residual, d_l1_padded);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG L1 pack launch");
            m5_cv_p1t_kernel<<<p1t_blocks, transfer_threads>>>(
                static_cast<std::uint32_t>(l2_nodes), d_p1_toffs, d_p1_trows,
                d_p1_tvals, d_l1_padded, d_l2_rhs);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG P1T launch");

            // L2 down.
            m5_cv_l2_zero_start_block_kernel<<<static_cast<unsigned int>(l2_nodes), 32U>>>(
                static_cast<std::uint32_t>(l2_nodes), d_l2_inv, weight2, d_l2_rhs, d_l2_x);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG L2 pre launch");
            launch_a2(d_l2_x, d_l2_ax);
            m5_cv_vector_residual_kernel<<<l2_vec_blocks, vec_threads>>>(
                l2_dofs, d_l2_rhs, d_l2_ax, d_l2_residual);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG L2 residual launch");
            launch_p2t(d_l2_residual, d_l3_rhs);

            // L3 exact direct solve.
            m5_cv_bottom_solve_kernel<<<1U, 32U, l3_bytes>>>(
                static_cast<std::uint32_t>(l3_dofs), d_l3_lower, d_l3_rhs, d_l3_x);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG bottom launch");

            // L2 up.
            launch_p2(d_l3_x, d_l2_correction);
            m5_cv_vector_add_kernel<<<l2_vec_blocks, vec_threads>>>(
                l2_dofs, d_l2_correction, d_l2_x);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG L2 correction launch");
            launch_a2(d_l2_x, d_l2_ax);
            m5_cv_l2_post_block_kernel<<<static_cast<unsigned int>(l2_nodes), 32U>>>(
                static_cast<std::uint32_t>(l2_nodes), d_l2_inv, weight2,
                d_l2_rhs, d_l2_ax, d_l2_x);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG L2 post launch");

            // L1 up.
            m5_cv_p1_forward_kernel<<<p1_blocks, transfer_threads>>>(
                static_cast<std::uint32_t>(l1_nodes), d_p1_frows, d_p1_fcols,
                d_p1_fvals, d_l2_x, d_l1_padded);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG P1 launch");
            m5_cv_add_l1_padded_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates,
                d_l1_padded, impl_->d_coarse_correction);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG L1 correction launch");
            launch_a1(impl_->d_coarse_correction, d_l1_ax);
            m5_cv_l1_post_block_kernel<<<impl_->aggregate_count, 32U>>>(
                impl_->aggregate_count, impl_->d_aggregates, d_l1_inv,
                weight1, impl_->d_coarse, d_l1_ax, impl_->d_coarse_correction);
            check_cuda_pcg(cudaGetLastError(), "M5 VPCG L1 post launch");

            // L0 up.
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
            check_cuda_pcg(cudaMemcpyAsync(z, impl_->d_x, fine_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpyAsync(M5 VPCG V-cycle output)");
        };

        float* rz_old = d_rz0;
        float* rz_new = d_rz1;
        auto run_solve = [&]() {
            check_cuda_pcg(cudaMemsetAsync(d_solution, 0, fine_bytes),
                           "cudaMemsetAsync(M5 VPCG solution)");
            check_cuda_pcg(cudaMemcpyAsync(d_r, d_b, fine_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpyAsync(M5 VPCG initial residual)");
            check_cuda_pcg(cudaMemsetAsync(d_breakdown, 0, sizeof(int)),
                           "cudaMemsetAsync(M5 VPCG breakdown)");

            apply_preconditioner(d_r, d_z);
            m5_cv_check_cublas(cublasSdot(dot_handle, n0, d_r, 1, d_z, 1, rz_old),
                               "cublasSdot(M5 VPCG initial rz)");
            check_cuda_pcg(cudaMemcpyAsync(d_p, d_z, fine_bytes, cudaMemcpyDeviceToDevice),
                           "cudaMemcpyAsync(M5 VPCG initial p)");

            for (std::size_t it = 0; it < iterations; ++it) {
                launch_pcg_matvec(impl_->mesh, impl_->block_y, impl_->nodes, d_p, d_ap);
                m5_cv_check_cublas(cublasSdot(dot_handle, n0, d_p, 1, d_ap, 1, d_pap),
                                   "cublasSdot(M5 VPCG pAp)");
                m5_vpcg_update_x_r_kernel<<<fine_vec_blocks, vec_threads>>>(
                    impl_->ndof, rz_old, d_pap, d_p, d_ap, d_solution, d_r, d_breakdown);
                check_cuda_pcg(cudaGetLastError(), "M5 VPCG x/r update launch");
                if (it + 1U == iterations) break;

                apply_preconditioner(d_r, d_z);
                m5_cv_check_cublas(cublasSdot(dot_handle, n0, d_r, 1, d_z, 1, rz_new),
                                   "cublasSdot(M5 VPCG rz new)");
                m5_vpcg_update_p_kernel<<<fine_vec_blocks, vec_threads>>>(
                    impl_->ndof, rz_new, rz_old, d_z, d_p, d_breakdown);
                check_cuda_pcg(cudaGetLastError(), "M5 VPCG p update launch");
                std::swap(rz_old, rz_new);
            }
        };

        // Warm up every path before measuring complete solves.
        run_solve();
        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(M5 VPCG warmup)");
        check_cuda_pcg(cudaEventCreate(&start), "cudaEventCreate(M5 VPCG start)");
        check_cuda_pcg(cudaEventCreate(&stop), "cudaEventCreate(M5 VPCG stop)");

        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            check_cuda_pcg(cudaEventRecord(start), "cudaEventRecord(M5 VPCG start)");
            run_solve();
            check_cuda_pcg(cudaEventRecord(stop), "cudaEventRecord(M5 VPCG stop)");
            check_cuda_pcg(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 VPCG)");
            float ms = 0.0f;
            check_cuda_pcg(cudaEventElapsedTime(&ms, start, stop),
                           "cudaEventElapsedTime(M5 VPCG)");
            samples.push_back(static_cast<double>(ms));
        }

        m5_cv_check_cublas(cublasSnrm2(dot_handle, n0, d_r, 1, d_recursive_norm),
                           "cublasSnrm2(M5 VPCG final residual)");
        float recursive_norm = 0.0f;
        int breakdown = 0;
        check_cuda_pcg(cudaMemcpy(&recursive_norm, d_recursive_norm, sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 VPCG residual norm D2H)");
        check_cuda_pcg(cudaMemcpy(&breakdown, d_breakdown, sizeof(int), cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 VPCG breakdown D2H)");

        std::vector<float> solution_soa(impl_->ndof, 0.0f);
        check_cuda_pcg(cudaMemcpy(solution_soa.data(), d_solution, fine_bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 VPCG solution D2H)");

        GpuM5VcyclePcgResult result;
        result.solution_aos.assign(impl_->ndof, 0.0f);
        for (std::size_t node = 0; node < impl_->nodes; ++node) {
            result.solution_aos[3U * node + 0U] = solution_soa[node];
            result.solution_aos[3U * node + 1U] = solution_soa[impl_->nodes + node];
            result.solution_aos[3U * node + 2U] = solution_soa[2U * impl_->nodes + node];
        }
        result.median_solve_ms = m5_vpcg_median(samples);
        result.best_solve_ms = *std::min_element(samples.begin(), samples.end());
        result.recursive_relative_residual = static_cast<double>(recursive_norm) / rhs_norm;
        result.iterations = iterations;
        result.preconditioner_applications = iterations;
        result.pcg_operator_applications = iterations;
        result.vcycle_l0_operator_applies =
            iterations * (2U * l0_smoother_degree + 1U + 2U * m0 + 2U * (2U * m0 + 1U));
        result.total_l0_operator_applies = result.vcycle_l0_operator_applies + iterations;
        const std::size_t base_context_bytes = impl_->fine_vector_bytes + impl_->coarse_vector_bytes +
            impl_->aggregation_metadata_bytes + impl_->model_coordinate_bytes;
        const std::size_t deep_bytes = l1_inv_bytes + 2U * l1_bytes + l1_padded_bytes +
            p1_frow_bytes + p1_fcol_bytes + p1_fval_bytes +
            p1_toff_bytes + p1_trow_bytes + p1_tval_bytes +
            a2_bytes + l2_inv_bytes + p2_bytes + 5U * l2_bytes +
            2U * l3_bytes + bottom_bytes;
        const std::size_t pcg_bytes = 6U * fine_bytes + 4U * sizeof(float) + sizeof(int);
        result.device_bytes_total = base_context_bytes + deep_bytes + pcg_bytes;
        result.breakdown = breakdown != 0;

        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

}  // namespace gfss
