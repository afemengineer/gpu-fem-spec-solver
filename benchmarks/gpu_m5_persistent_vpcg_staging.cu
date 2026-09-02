// M5 stage 13: persistent deep-hierarchy MG-PCG correction context with the
// validated explicit symmetric FP32 A3^-1 bottom representation.
//
// This TU deliberately includes the validated complete V-cycle implementation
// so the exact same A0/P0, block6 P1/P1T and dense A2/P2 kernels are used.
// All deep hierarchy payloads, cuBLAS handles, PCG vectors and timing events are
// allocated/uploaded once. L3 is applied by cuBLAS SGEMV rather than the serial
// one-warp Cholesky forward/backward substitution.
#include "m5_persistent_vpcg_staging.hpp"
#include "gpu_m5_complete_vcycle_impl.cu"

#include <cublas_v2.h>

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

void persist_check_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS status=" +
                                 std::to_string(static_cast<int>(status)));
    }
}

__global__ void persist_update_x_r_kernel(
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

__global__ void persist_update_p_kernel(
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

std::vector<float> persist_chebyshev_weights(double lambda_max, std::size_t degree) {
    if (!(lambda_max > 0.0) || degree == 0U || degree > 32U) {
        throw std::invalid_argument("persistent M5 Chebyshev options invalid");
    }
    std::vector<float> weights(degree, 0.0f);
    const double lambda_low = kM5ChebyshevLowerFraction * lambda_max;
    const double theta = 0.5 * (lambda_max + lambda_low);
    const double delta = 0.5 * (lambda_max - lambda_low);
    for (std::size_t k = 0U; k < degree; ++k) {
        const double angle = kM5Pi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        const double root = theta + delta * std::cos(angle);
        if (!(root > 0.0) || !std::isfinite(root)) {
            throw std::runtime_error("persistent M5 Chebyshev root invalid");
        }
        weights[k] = static_cast<float>(1.0 / root);
    }
    return weights;
}

}  // namespace

struct M5PersistentPcgStaging::Impl {
    StructuredHexMesh mesh{};
    int block_y{4};
    std::size_t nodes{0U};
    std::size_t ndof{0U};
    std::size_t l1_nodes{0U};
    std::size_t l1_dofs{0U};
    std::size_t l2_nodes{0U};
    std::size_t l2_dofs{0U};
    std::size_t l3_dofs{0U};
    std::uint32_t aggregate_count{0U};
    std::size_t m0{1U};
    float omega0{0.0f};
    float weight1{0.0f};
    float weight2{0.0f};
    std::vector<float> l0_weights;

    std::uint32_t* d_aggregate_offsets{nullptr};
    std::uint32_t* d_aggregate_nodes{nullptr};
    DeviceAggregateM5* d_aggregates{nullptr};
    float* d_coordinates{nullptr};
    float* d_base_rhs{nullptr};
    float* d_base_x{nullptr};
    float* d_work0{nullptr};
    float* d_work1{nullptr};
    float* d_work2{nullptr};
    float* d_coarse{nullptr};
    float* d_coarse_correction{nullptr};

    float* d_l1_inv{nullptr};
    float* d_l1_ax{nullptr};
    float* d_l1_residual{nullptr};
    float* d_l1_padded{nullptr};
    std::uint32_t* d_p1_frows{nullptr};
    std::uint32_t* d_p1_fcols{nullptr};
    float* d_p1_fvals{nullptr};
    std::uint32_t* d_p1_toffs{nullptr};
    std::uint32_t* d_p1_trows{nullptr};
    float* d_p1_tvals{nullptr};
    float* d_a2{nullptr};
    float* d_l2_inv{nullptr};
    float* d_p2{nullptr};
    float* d_l2_rhs{nullptr};
    float* d_l2_x{nullptr};
    float* d_l2_ax{nullptr};
    float* d_l2_residual{nullptr};
    float* d_l2_correction{nullptr};
    float* d_l3_rhs{nullptr};
    float* d_l3_x{nullptr};
    float* d_l3_inverse{nullptr};

    float* d_b{nullptr};
    float* d_solution{nullptr};
    float* d_r{nullptr};
    float* d_z{nullptr};
    float* d_p{nullptr};
    float* d_ap{nullptr};
    float* d_rz0{nullptr};
    float* d_rz1{nullptr};
    float* d_pap{nullptr};
    float* d_recursive_norm{nullptr};
    int* d_breakdown{nullptr};

    cublasHandle_t prec_handle{};
    cublasHandle_t dot_handle{};
    cudaEvent_t start{};
    cudaEvent_t stop{};

    std::size_t fine_bytes{0U};
    std::size_t l1_bytes{0U};
    std::size_t l1_padded_bytes{0U};
    std::size_t l2_bytes{0U};
    std::size_t l3_bytes{0U};
    std::size_t device_bytes_total_value{0U};

    unsigned int fine_vec_blocks{0U};
    unsigned int l1_vec_blocks{0U};
    unsigned int l2_vec_blocks{0U};
    unsigned int p1_blocks{0U};
    unsigned int p1t_blocks{0U};
    int n0{0};
    int n2{0};
    int n3{0};

    Impl(
        const StructuredHexMesh& mesh_in,
        int block_y_in,
        std::size_t nodes_in,
        std::size_t ndof_in,
        std::uint32_t aggregate_count_in,
        std::size_t l1_dofs_in,
        std::uint32_t* aggregate_offsets,
        std::uint32_t* aggregate_nodes,
        DeviceAggregateM5* aggregates,
        float* coordinates,
        float* base_rhs,
        float* base_x,
        float* work0,
        float* work1,
        float* work2,
        float* coarse,
        float* coarse_correction,
        std::size_t base_context_bytes,
        double transfer_omega,
        double smoother_lambda,
        std::size_t l0_smoother_degree,
        std::size_t m0_in,
        const std::vector<float>& l1_inverse_blocks_6x6,
        double lambda1,
        std::size_t l2_nodes_in,
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
        std::size_t l3_dofs_in,
        const std::vector<float>& l3_inverse_col_major)
        : mesh(mesh_in), block_y(block_y_in), nodes(nodes_in), ndof(ndof_in),
          l1_nodes(aggregate_count_in), l1_dofs(l1_dofs_in),
          l2_nodes(l2_nodes_in), l2_dofs(l2_nodes_in * 6U), l3_dofs(l3_dofs_in),
          aggregate_count(aggregate_count_in), m0(m0_in),
          omega0(static_cast<float>(transfer_omega)),
          weight1(static_cast<float>(1.0 / (0.55 * lambda1))),
          weight2(static_cast<float>(1.0 / (0.55 * lambda2))),
          l0_weights(persist_chebyshev_weights(smoother_lambda, l0_smoother_degree)),
          d_aggregate_offsets(aggregate_offsets), d_aggregate_nodes(aggregate_nodes),
          d_aggregates(aggregates), d_coordinates(coordinates), d_base_rhs(base_rhs),
          d_base_x(base_x), d_work0(work0), d_work1(work1), d_work2(work2),
          d_coarse(coarse), d_coarse_correction(coarse_correction) {
        if (m0 == 0U || m0 > 8U || l2_nodes == 0U || l3_dofs == 0U || l3_dofs > 256U ||
            !(lambda1 > 0.0) || !(lambda2 > 0.0) || !std::isfinite(lambda1) ||
            !std::isfinite(lambda2)) {
            throw std::invalid_argument("persistent M5 scalar hierarchy options invalid");
        }
        if (l1_inverse_blocks_6x6.size() != l1_nodes * 36U ||
            p1_forward_row_offsets.size() != l1_nodes + 1U ||
            p1_transpose_column_offsets.size() != l2_nodes + 1U) {
            throw std::invalid_argument("persistent M5 L1/P1 shape mismatch");
        }
        const std::size_t p1_nnz = p1_forward_column_indices.size();
        if (p1_forward_row_offsets.back() != p1_nnz ||
            p1_transpose_column_offsets.back() != p1_nnz ||
            p1_transpose_row_indices.size() != p1_nnz ||
            p1_forward_values_6x6.size() != p1_nnz * 36U ||
            p1_transpose_values_q_r_entry.size() != p1_nnz * 36U) {
            throw std::invalid_argument("persistent M5 P1 payload mismatch");
        }
        if (a2_dense_row_major.size() != l2_dofs * l2_dofs ||
            l2_inverse_blocks_6x6.size() != l2_nodes * 36U ||
            p2_dense_row_major.size() != l2_dofs * l3_dofs ||
            l3_inverse_col_major.size() != l3_dofs * l3_dofs) {
            throw std::invalid_argument("persistent M5 deep payload mismatch");
        }
        if (ndof > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            l2_dofs > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            l3_dofs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("persistent M5 cuBLAS dimensions unsupported");
        }
        for (const float v : l3_inverse_col_major) {
            if (!std::isfinite(v)) {
                throw std::invalid_argument("persistent M5 bottom inverse contains non-finite value");
            }
        }

        fine_bytes = ndof * sizeof(float);
        l1_bytes = l1_dofs * sizeof(float);
        l1_padded_bytes = l1_nodes * 6U * sizeof(float);
        l2_bytes = l2_dofs * sizeof(float);
        l3_bytes = l3_dofs * sizeof(float);
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
        const std::size_t bottom_bytes = l3_inverse_col_major.size() * sizeof(float);

        try {
#define PERSIST_MALLOC(ptr, bytes, label) \
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&(ptr)), (bytes)), (label))
            PERSIST_MALLOC(d_l1_inv, l1_inv_bytes, "cudaMalloc(M5 persistent L1 inverse)");
            PERSIST_MALLOC(d_l1_ax, l1_bytes, "cudaMalloc(M5 persistent L1 ax)");
            PERSIST_MALLOC(d_l1_residual, l1_bytes, "cudaMalloc(M5 persistent L1 residual)");
            PERSIST_MALLOC(d_l1_padded, l1_padded_bytes, "cudaMalloc(M5 persistent L1 padded)");
            PERSIST_MALLOC(d_p1_frows, p1_frow_bytes, "cudaMalloc(M5 persistent P1 rows)");
            PERSIST_MALLOC(d_p1_fcols, p1_fcol_bytes, "cudaMalloc(M5 persistent P1 cols)");
            PERSIST_MALLOC(d_p1_fvals, p1_fval_bytes, "cudaMalloc(M5 persistent P1 values)");
            PERSIST_MALLOC(d_p1_toffs, p1_toff_bytes, "cudaMalloc(M5 persistent P1T offsets)");
            PERSIST_MALLOC(d_p1_trows, p1_trow_bytes, "cudaMalloc(M5 persistent P1T rows)");
            PERSIST_MALLOC(d_p1_tvals, p1_tval_bytes, "cudaMalloc(M5 persistent P1T values)");
            PERSIST_MALLOC(d_a2, a2_bytes, "cudaMalloc(M5 persistent A2)");
            PERSIST_MALLOC(d_l2_inv, l2_inv_bytes, "cudaMalloc(M5 persistent L2 inverse)");
            PERSIST_MALLOC(d_p2, p2_bytes, "cudaMalloc(M5 persistent P2)");
            PERSIST_MALLOC(d_l2_rhs, l2_bytes, "cudaMalloc(M5 persistent L2 rhs)");
            PERSIST_MALLOC(d_l2_x, l2_bytes, "cudaMalloc(M5 persistent L2 x)");
            PERSIST_MALLOC(d_l2_ax, l2_bytes, "cudaMalloc(M5 persistent L2 ax)");
            PERSIST_MALLOC(d_l2_residual, l2_bytes, "cudaMalloc(M5 persistent L2 residual)");
            PERSIST_MALLOC(d_l2_correction, l2_bytes, "cudaMalloc(M5 persistent L2 correction)");
            PERSIST_MALLOC(d_l3_rhs, l3_bytes, "cudaMalloc(M5 persistent L3 rhs)");
            PERSIST_MALLOC(d_l3_x, l3_bytes, "cudaMalloc(M5 persistent L3 x)");
            PERSIST_MALLOC(d_l3_inverse, bottom_bytes, "cudaMalloc(M5 persistent L3 inverse)");
            PERSIST_MALLOC(d_b, fine_bytes, "cudaMalloc(M5 persistent b)");
            PERSIST_MALLOC(d_solution, fine_bytes, "cudaMalloc(M5 persistent solution)");
            PERSIST_MALLOC(d_r, fine_bytes, "cudaMalloc(M5 persistent r)");
            PERSIST_MALLOC(d_z, fine_bytes, "cudaMalloc(M5 persistent z)");
            PERSIST_MALLOC(d_p, fine_bytes, "cudaMalloc(M5 persistent p)");
            PERSIST_MALLOC(d_ap, fine_bytes, "cudaMalloc(M5 persistent Ap)");
            PERSIST_MALLOC(d_rz0, sizeof(float), "cudaMalloc(M5 persistent rz0)");
            PERSIST_MALLOC(d_rz1, sizeof(float), "cudaMalloc(M5 persistent rz1)");
            PERSIST_MALLOC(d_pap, sizeof(float), "cudaMalloc(M5 persistent pAp)");
            PERSIST_MALLOC(d_recursive_norm, sizeof(float), "cudaMalloc(M5 persistent norm)");
            PERSIST_MALLOC(d_breakdown, sizeof(int), "cudaMalloc(M5 persistent breakdown)");
#undef PERSIST_MALLOC

#define PERSIST_COPY(dst, src, bytes, label) \
            check_cuda_pcg(cudaMemcpy((dst), (src).data(), (bytes), cudaMemcpyHostToDevice), (label))
            PERSIST_COPY(d_l1_inv, l1_inverse_blocks_6x6, l1_inv_bytes, "cudaMemcpy(M5 persistent L1 inverse)");
            PERSIST_COPY(d_p1_frows, p1_forward_row_offsets, p1_frow_bytes, "cudaMemcpy(M5 persistent P1 rows)");
            PERSIST_COPY(d_p1_fcols, p1_forward_column_indices, p1_fcol_bytes, "cudaMemcpy(M5 persistent P1 cols)");
            PERSIST_COPY(d_p1_fvals, p1_forward_values_6x6, p1_fval_bytes, "cudaMemcpy(M5 persistent P1 values)");
            PERSIST_COPY(d_p1_toffs, p1_transpose_column_offsets, p1_toff_bytes, "cudaMemcpy(M5 persistent P1T offsets)");
            PERSIST_COPY(d_p1_trows, p1_transpose_row_indices, p1_trow_bytes, "cudaMemcpy(M5 persistent P1T rows)");
            PERSIST_COPY(d_p1_tvals, p1_transpose_values_q_r_entry, p1_tval_bytes, "cudaMemcpy(M5 persistent P1T values)");
            PERSIST_COPY(d_a2, a2_dense_row_major, a2_bytes, "cudaMemcpy(M5 persistent A2)");
            PERSIST_COPY(d_l2_inv, l2_inverse_blocks_6x6, l2_inv_bytes, "cudaMemcpy(M5 persistent L2 inverse)");
            PERSIST_COPY(d_p2, p2_dense_row_major, p2_bytes, "cudaMemcpy(M5 persistent P2)");
            PERSIST_COPY(d_l3_inverse, l3_inverse_col_major, bottom_bytes, "cudaMemcpy(M5 persistent L3 inverse)");
#undef PERSIST_COPY

            persist_check_cublas(cublasCreate(&prec_handle), "cublasCreate(M5 persistent preconditioner)");
            persist_check_cublas(cublasCreate(&dot_handle), "cublasCreate(M5 persistent dot)");
            persist_check_cublas(cublasSetPointerMode(dot_handle, CUBLAS_POINTER_MODE_DEVICE),
                                 "cublasSetPointerMode(M5 persistent device)");
            check_cuda_pcg(cudaEventCreate(&start), "cudaEventCreate(M5 persistent start)");
            check_cuda_pcg(cudaEventCreate(&stop), "cudaEventCreate(M5 persistent stop)");
        } catch (...) {
            cleanup();
            throw;
        }

        constexpr unsigned int vec_threads = 256U;
        fine_vec_blocks = static_cast<unsigned int>((ndof + vec_threads - 1U) / vec_threads);
        l1_vec_blocks = static_cast<unsigned int>((l1_dofs + vec_threads - 1U) / vec_threads);
        l2_vec_blocks = static_cast<unsigned int>((l2_dofs + vec_threads - 1U) / vec_threads);
        constexpr unsigned int transfer_threads = 256U;
        constexpr unsigned int warps_per_block = transfer_threads / 32U;
        p1_blocks = static_cast<unsigned int>((l1_nodes + warps_per_block - 1U) / warps_per_block);
        p1t_blocks = static_cast<unsigned int>((l2_nodes * 6U + warps_per_block - 1U) / warps_per_block);
        n0 = static_cast<int>(ndof);
        n2 = static_cast<int>(l2_dofs);
        n3 = static_cast<int>(l3_dofs);

        const std::size_t deep_bytes = l1_inv_bytes + 2U * l1_bytes + l1_padded_bytes +
            p1_frow_bytes + p1_fcol_bytes + p1_fval_bytes +
            p1_toff_bytes + p1_trow_bytes + p1_tval_bytes +
            a2_bytes + l2_inv_bytes + p2_bytes + 5U * l2_bytes +
            2U * l3_bytes + bottom_bytes;
        const std::size_t pcg_bytes = 6U * fine_bytes + 4U * sizeof(float) + sizeof(int);
        device_bytes_total_value = base_context_bytes + deep_bytes + pcg_bytes;
    }

    ~Impl() { cleanup(); }

    void cleanup() noexcept {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (prec_handle) cublasDestroy(prec_handle);
        if (dot_handle) cublasDestroy(dot_handle);
        start = stop = nullptr;
        prec_handle = dot_handle = nullptr;
#define PERSIST_FREE(ptr) do { if (ptr) cudaFree(ptr); ptr = nullptr; } while (0)
        PERSIST_FREE(d_l1_inv); PERSIST_FREE(d_l1_ax); PERSIST_FREE(d_l1_residual);
        PERSIST_FREE(d_l1_padded); PERSIST_FREE(d_p1_frows); PERSIST_FREE(d_p1_fcols);
        PERSIST_FREE(d_p1_fvals); PERSIST_FREE(d_p1_toffs); PERSIST_FREE(d_p1_trows);
        PERSIST_FREE(d_p1_tvals); PERSIST_FREE(d_a2); PERSIST_FREE(d_l2_inv);
        PERSIST_FREE(d_p2); PERSIST_FREE(d_l2_rhs); PERSIST_FREE(d_l2_x);
        PERSIST_FREE(d_l2_ax); PERSIST_FREE(d_l2_residual); PERSIST_FREE(d_l2_correction);
        PERSIST_FREE(d_l3_rhs); PERSIST_FREE(d_l3_x); PERSIST_FREE(d_l3_inverse);
        PERSIST_FREE(d_b); PERSIST_FREE(d_solution); PERSIST_FREE(d_r); PERSIST_FREE(d_z);
        PERSIST_FREE(d_p); PERSIST_FREE(d_ap); PERSIST_FREE(d_rz0); PERSIST_FREE(d_rz1);
        PERSIST_FREE(d_pap); PERSIST_FREE(d_recursive_norm); PERSIST_FREE(d_breakdown);
#undef PERSIST_FREE
    }

    void upload_rhs(const std::vector<float>& rhs_aos, double& rhs_norm) {
        if (rhs_aos.size() != ndof) {
            throw std::invalid_argument("persistent M5 RHS size mismatch");
        }
        std::vector<float> rhs_soa(ndof, 0.0f);
        double norm2 = 0.0;
        for (std::size_t node = 0U; node < nodes; ++node) {
            for (std::size_t c = 0U; c < 3U; ++c) {
                const float value = rhs_aos[3U * node + c];
                rhs_soa[c * nodes + node] = value;
                norm2 += static_cast<double>(value) * static_cast<double>(value);
            }
        }
        rhs_norm = std::sqrt(norm2);
        if (!(rhs_norm > 0.0) || !std::isfinite(rhs_norm)) {
            throw std::invalid_argument("persistent M5 RHS norm invalid");
        }
        check_cuda_pcg(cudaMemcpy(d_b, rhs_soa.data(), fine_bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(M5 persistent RHS)");
    }

    void launch_a1(const float* x1, float* y1) {
        m5_launch_p0(mesh, nodes, aggregate_count,
                     d_aggregate_offsets, d_aggregate_nodes, d_aggregates,
                     d_coordinates, x1, d_work0);
        for (std::size_t step = 0U; step < m0; ++step) {
            launch_pcg_matvec(mesh, block_y, nodes, d_work0, d_work1);
            m5_launch_forward_transfer_update(mesh, block_y, nodes, omega0, d_work1, d_work0);
        }
        launch_pcg_matvec(mesh, block_y, nodes, d_work0, d_work1);
        for (std::size_t step = 0U; step < m0; ++step) {
            m5_launch_inverse_scale(mesh, block_y, nodes, d_work1, d_work2);
            launch_pcg_matvec(mesh, block_y, nodes, d_work2, d_work0);
            m5_launch_transpose_update(mesh, block_y, nodes, omega0, d_work0, d_work1);
        }
        m5_launch_p0t(nodes, aggregate_count,
                      d_aggregate_offsets, d_aggregate_nodes, d_aggregates,
                      d_coordinates, d_work1, y1);
    }

    void launch_a2(const float* x2, float* y2) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        persist_check_cublas(cublasSgemv(prec_handle, CUBLAS_OP_N, n2, n2,
                                         &alpha, d_a2, n2, x2, 1, &beta, y2, 1),
                             "cublasSgemv(M5 persistent A2)");
    }

    void launch_p2t(const float* x2, float* y3) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        persist_check_cublas(cublasSgemv(prec_handle, CUBLAS_OP_N, n3, n2,
                                         &alpha, d_p2, n3, x2, 1, &beta, y3, 1),
                             "cublasSgemv(M5 persistent P2T)");
    }

    void launch_p2(const float* x3, float* y2) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        persist_check_cublas(cublasSgemv(prec_handle, CUBLAS_OP_T, n3, n2,
                                         &alpha, d_p2, n3, x3, 1, &beta, y2, 1),
                             "cublasSgemv(M5 persistent P2)");
    }

    void launch_l3_inverse(const float* rhs3, float* x3) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        persist_check_cublas(cublasSgemv(prec_handle, CUBLAS_OP_N, n3, n3,
                                         &alpha, d_l3_inverse, n3, rhs3, 1,
                                         &beta, x3, 1),
                             "cublasSgemv(M5 persistent L3 inverse)");
    }

    void apply_preconditioner(const float* residual, float* z) {
        constexpr unsigned int vec_threads = 256U;
        constexpr unsigned int transfer_threads = 256U;
        check_cuda_pcg(cudaMemcpyAsync(d_base_rhs, residual, fine_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(M5 persistent residual to V-cycle rhs)");
        check_cuda_pcg(cudaMemsetAsync(d_base_x, 0, fine_bytes),
                       "cudaMemsetAsync(M5 persistent V-cycle x0)");

        for (const float w : l0_weights) {
            launch_pcg_matvec(mesh, block_y, nodes, d_base_x, d_work0);
            m5_launch_chebyshev_update(mesh, block_y, nodes, w, d_base_rhs, d_work0, d_base_x);
        }
        launch_pcg_matvec(mesh, block_y, nodes, d_base_x, d_work0);
        m5_launch_residual(mesh, block_y, nodes, d_base_rhs, d_work0);
        for (std::size_t step = 0U; step < m0; ++step) {
            m5_launch_inverse_scale(mesh, block_y, nodes, d_work0, d_work1);
            launch_pcg_matvec(mesh, block_y, nodes, d_work1, d_work2);
            m5_launch_transpose_update(mesh, block_y, nodes, omega0, d_work2, d_work0);
        }
        m5_launch_p0t(nodes, aggregate_count,
                      d_aggregate_offsets, d_aggregate_nodes, d_aggregates,
                      d_coordinates, d_work0, d_coarse);

        m5_cv_l1_zero_start_block_kernel<<<aggregate_count, 32U>>>(
            aggregate_count, d_aggregates, d_l1_inv, weight1, d_coarse, d_coarse_correction);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent L1 pre launch");
        launch_a1(d_coarse_correction, d_l1_ax);
        m5_cv_vector_residual_kernel<<<l1_vec_blocks, vec_threads>>>(
            l1_dofs, d_coarse, d_l1_ax, d_l1_residual);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent L1 residual launch");
        m5_cv_pack_l1_kernel<<<aggregate_count, 32U>>>(
            aggregate_count, d_aggregates, d_l1_residual, d_l1_padded);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent L1 pack launch");
        m5_cv_p1t_kernel<<<p1t_blocks, transfer_threads>>>(
            static_cast<std::uint32_t>(l2_nodes), d_p1_toffs, d_p1_trows,
            d_p1_tvals, d_l1_padded, d_l2_rhs);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent P1T launch");

        m5_cv_l2_zero_start_block_kernel<<<static_cast<unsigned int>(l2_nodes), 32U>>>(
            static_cast<std::uint32_t>(l2_nodes), d_l2_inv, weight2, d_l2_rhs, d_l2_x);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent L2 pre launch");
        launch_a2(d_l2_x, d_l2_ax);
        m5_cv_vector_residual_kernel<<<l2_vec_blocks, vec_threads>>>(
            l2_dofs, d_l2_rhs, d_l2_ax, d_l2_residual);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent L2 residual launch");
        launch_p2t(d_l2_residual, d_l3_rhs);

        launch_l3_inverse(d_l3_rhs, d_l3_x);

        launch_p2(d_l3_x, d_l2_correction);
        m5_cv_vector_add_kernel<<<l2_vec_blocks, vec_threads>>>(l2_dofs, d_l2_correction, d_l2_x);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent L2 correction launch");
        launch_a2(d_l2_x, d_l2_ax);
        m5_cv_l2_post_block_kernel<<<static_cast<unsigned int>(l2_nodes), 32U>>>(
            static_cast<std::uint32_t>(l2_nodes), d_l2_inv, weight2,
            d_l2_rhs, d_l2_ax, d_l2_x);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent L2 post launch");

        m5_cv_p1_forward_kernel<<<p1_blocks, transfer_threads>>>(
            static_cast<std::uint32_t>(l1_nodes), d_p1_frows, d_p1_fcols,
            d_p1_fvals, d_l2_x, d_l1_padded);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent P1 launch");
        m5_cv_add_l1_padded_kernel<<<aggregate_count, 32U>>>(
            aggregate_count, d_aggregates, d_l1_padded, d_coarse_correction);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent L1 correction launch");
        launch_a1(d_coarse_correction, d_l1_ax);
        m5_cv_l1_post_block_kernel<<<aggregate_count, 32U>>>(
            aggregate_count, d_aggregates, d_l1_inv, weight1,
            d_coarse, d_l1_ax, d_coarse_correction);
        check_cuda_pcg(cudaGetLastError(), "M5 persistent L1 post launch");

        m5_launch_p0(mesh, nodes, aggregate_count,
                     d_aggregate_offsets, d_aggregate_nodes, d_aggregates,
                     d_coordinates, d_coarse_correction, d_work0);
        for (std::size_t step = 0U; step < m0; ++step) {
            launch_pcg_matvec(mesh, block_y, nodes, d_work0, d_work1);
            m5_launch_forward_transfer_update(mesh, block_y, nodes, omega0, d_work1, d_work0);
        }
        m5_launch_add_correction(mesh, block_y, nodes, d_work0, d_base_x);
        for (const float w : l0_weights) {
            launch_pcg_matvec(mesh, block_y, nodes, d_base_x, d_work0);
            m5_launch_chebyshev_update(mesh, block_y, nodes, w, d_base_rhs, d_work0, d_base_x);
        }
        check_cuda_pcg(cudaMemcpyAsync(z, d_base_x, fine_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(M5 persistent V-cycle output)");
    }

    void run_solve(std::size_t iterations) {
        if (iterations == 0U || iterations > 256U) {
            throw std::invalid_argument("persistent M5 iteration count invalid");
        }
        constexpr unsigned int vec_threads = 256U;
        float* rz_old = d_rz0;
        float* rz_new = d_rz1;

        check_cuda_pcg(cudaMemsetAsync(d_solution, 0, fine_bytes),
                       "cudaMemsetAsync(M5 persistent solution)");
        check_cuda_pcg(cudaMemcpyAsync(d_r, d_b, fine_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(M5 persistent initial residual)");
        check_cuda_pcg(cudaMemsetAsync(d_breakdown, 0, sizeof(int)),
                       "cudaMemsetAsync(M5 persistent breakdown)");

        apply_preconditioner(d_r, d_z);
        persist_check_cublas(cublasSdot(dot_handle, n0, d_r, 1, d_z, 1, rz_old),
                             "cublasSdot(M5 persistent initial rz)");
        check_cuda_pcg(cudaMemcpyAsync(d_p, d_z, fine_bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(M5 persistent initial p)");

        for (std::size_t it = 0U; it < iterations; ++it) {
            launch_pcg_matvec(mesh, block_y, nodes, d_p, d_ap);
            persist_check_cublas(cublasSdot(dot_handle, n0, d_p, 1, d_ap, 1, d_pap),
                                 "cublasSdot(M5 persistent pAp)");
            persist_update_x_r_kernel<<<fine_vec_blocks, vec_threads>>>(
                ndof, rz_old, d_pap, d_p, d_ap, d_solution, d_r, d_breakdown);
            check_cuda_pcg(cudaGetLastError(), "M5 persistent x/r update launch");
            if (it + 1U == iterations) break;

            apply_preconditioner(d_r, d_z);
            persist_check_cublas(cublasSdot(dot_handle, n0, d_r, 1, d_z, 1, rz_new),
                                 "cublasSdot(M5 persistent rz new)");
            persist_update_p_kernel<<<fine_vec_blocks, vec_threads>>>(
                ndof, rz_new, rz_old, d_z, d_p, d_breakdown);
            check_cuda_pcg(cudaGetLastError(), "M5 persistent p update launch");
            std::swap(rz_old, rz_new);
        }
    }

    void warmup(const std::vector<float>& rhs_aos, std::size_t iterations) {
        double ignored_norm = 0.0;
        upload_rhs(rhs_aos, ignored_norm);
        run_solve(iterations);
        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(M5 persistent warmup)");
    }

    M5PersistentPcgSolveResult solve(
        const std::vector<float>& rhs_aos,
        std::size_t iterations) {
        double rhs_norm = 0.0;
        upload_rhs(rhs_aos, rhs_norm);

        check_cuda_pcg(cudaEventRecord(start), "cudaEventRecord(M5 persistent start)");
        run_solve(iterations);
        check_cuda_pcg(cudaEventRecord(stop), "cudaEventRecord(M5 persistent stop)");
        check_cuda_pcg(cudaEventSynchronize(stop), "cudaEventSynchronize(M5 persistent solve)");
        float elapsed = 0.0f;
        check_cuda_pcg(cudaEventElapsedTime(&elapsed, start, stop),
                       "cudaEventElapsedTime(M5 persistent solve)");

        persist_check_cublas(cublasSnrm2(dot_handle, n0, d_r, 1, d_recursive_norm),
                             "cublasSnrm2(M5 persistent residual)");
        float recursive_norm = 0.0f;
        int breakdown = 0;
        check_cuda_pcg(cudaMemcpy(&recursive_norm, d_recursive_norm, sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 persistent residual norm)");
        check_cuda_pcg(cudaMemcpy(&breakdown, d_breakdown, sizeof(int), cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 persistent breakdown)");

        std::vector<float> solution_soa(ndof, 0.0f);
        check_cuda_pcg(cudaMemcpy(solution_soa.data(), d_solution, fine_bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(M5 persistent solution)");

        M5PersistentPcgSolveResult out;
        out.solution_aos.assign(ndof, 0.0f);
        for (std::size_t node = 0U; node < nodes; ++node) {
            out.solution_aos[3U * node + 0U] = solution_soa[node];
            out.solution_aos[3U * node + 1U] = solution_soa[nodes + node];
            out.solution_aos[3U * node + 2U] = solution_soa[2U * nodes + node];
        }
        out.solve_ms = static_cast<double>(elapsed);
        out.recursive_relative_residual = static_cast<double>(recursive_norm) / rhs_norm;
        out.iterations = iterations;
        out.total_l0_operator_applies = iterations *
            (2U * l0_weights.size() + 1U + 2U * m0 + 2U * (2U * m0 + 1U) + 1U);
        out.device_bytes_total = device_bytes_total_value;
        out.breakdown = breakdown != 0;
        return out;
    }
};

M5PersistentPcgStaging::M5PersistentPcgStaging(
    GpuM5FineLevelContext& fine,
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
    const std::vector<float>& l3_inverse_col_major) {
    if (!fine.impl_) throw std::runtime_error("persistent M5 fine context empty");
    auto& b = *fine.impl_;
    const std::size_t base_context_bytes = b.fine_vector_bytes + b.coarse_vector_bytes +
        b.aggregation_metadata_bytes + b.model_coordinate_bytes;
    impl_ = std::make_unique<Impl>(
        b.mesh, b.block_y, b.nodes, b.ndof, b.aggregate_count, b.coarse_dof_count,
        b.d_aggregate_offsets, b.d_aggregate_nodes, b.d_aggregates, b.d_coordinates,
        b.d_rhs, b.d_x, b.d_work0, b.d_work1, b.d_work2, b.d_coarse,
        b.d_coarse_correction, base_context_bytes,
        b.transfer_omega_value, b.smoother_lambda_max_value,
        l0_smoother_degree, m0,
        l1_inverse_blocks_6x6, lambda1, l2_nodes,
        p1_forward_row_offsets, p1_forward_column_indices, p1_forward_values_6x6,
        p1_transpose_column_offsets, p1_transpose_row_indices,
        p1_transpose_values_q_r_entry, a2_dense_row_major,
        l2_inverse_blocks_6x6, lambda2, p2_dense_row_major, l3_dofs,
        l3_inverse_col_major);
}

M5PersistentPcgStaging::~M5PersistentPcgStaging() = default;
M5PersistentPcgStaging::M5PersistentPcgStaging(M5PersistentPcgStaging&&) noexcept = default;
M5PersistentPcgStaging& M5PersistentPcgStaging::operator=(M5PersistentPcgStaging&&) noexcept = default;

void M5PersistentPcgStaging::warmup(
    const std::vector<float>& rhs_aos,
    std::size_t iterations) {
    if (!impl_) throw std::runtime_error("persistent M5 context empty");
    impl_->warmup(rhs_aos, iterations);
}

M5PersistentPcgSolveResult M5PersistentPcgStaging::solve(
    const std::vector<float>& rhs_aos,
    std::size_t iterations) {
    if (!impl_) throw std::runtime_error("persistent M5 context empty");
    return impl_->solve(rhs_aos, iterations);
}

std::size_t M5PersistentPcgStaging::device_bytes_total() const noexcept {
    return impl_ ? impl_->device_bytes_total_value : 0U;
}

}  // namespace gfss
