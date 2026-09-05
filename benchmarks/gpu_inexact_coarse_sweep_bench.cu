// Standalone experimental TU: include the persistent smoothed-aggregation CUDA
// implementation so this benchmark can extend the context with a device-resident
// fixed-budget coarse PCG without exposing internal device pointers as public API.
// This target deliberately does NOT link gfss_cuda_operator, avoiding duplicate
// symbols from the included implementation.
#include "../src/gpu_smoothed_aggregation.cu"

#include "gfss/aggregation_two_grid_reference.hpp"
#include "gfss/cpu_elasticity.hpp"
#include "gfss/hex8.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gfss {
namespace {

__global__ void coarse_precondition_sa_kernel(
    int n,
    const float* __restrict__ inverse_diagonal,
    const float* __restrict__ r,
    float* __restrict__ z) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) z[i] = inverse_diagonal[i] * r[i];
}

void launch_coarse_precondition_sa(int n,
                                   const float* inverse_diagonal,
                                   const float* r,
                                   float* z) {
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    coarse_precondition_sa_kernel<<<blocks, threads>>>(n, inverse_diagonal, r, z);
    check_cuda_pcg(cudaGetLastError(), "SA coarse PCG precondition launch");
}

}  // namespace

GpuSmoothedAggregationCoarsePcgResult
GpuSmoothedAggregationContext::solve_coarse_pcg_fixed_iterations(
    const std::vector<float>& rhs,
    const std::vector<float>& inverse_preconditioner,
    std::size_t transfer_smoothing_steps,
    std::size_t max_iterations) {
    if (!impl_) throw std::runtime_error("GPU smoothed aggregation context is empty");
    if (rhs.size() != impl_->coarse_dof_count ||
        inverse_preconditioner.size() != impl_->coarse_dof_count) {
        throw std::invalid_argument("GPU coarse PCG vector size mismatch");
    }
    if (max_iterations == 0U) {
        throw std::invalid_argument("GPU coarse PCG fixed iteration budget must be positive");
    }
    if (transfer_smoothing_steps > 8U) {
        throw std::invalid_argument("GPU coarse PCG reference limits transfer smoothing m to 8");
    }
    if (impl_->coarse_dof_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("GPU coarse PCG currently requires coarse_dofs <= INT_MAX");
    }

    double bnorm2_host = 0.0;
    for (float value : rhs) bnorm2_host += static_cast<double>(value) * value;
    if (!(bnorm2_host > 0.0) || !std::isfinite(bnorm2_host)) {
        throw std::invalid_argument("GPU coarse PCG requires finite non-zero RHS");
    }
    for (float value : inverse_preconditioner) {
        if (!(value > 0.0f) || !std::isfinite(value)) {
            throw std::invalid_argument("GPU coarse PCG inverse preconditioner must be finite and positive");
        }
    }

    const int n = static_cast<int>(impl_->coarse_dof_count);
    const std::size_t bytes = impl_->coarse_dof_count * sizeof(float);
    float* d_x = nullptr;
    float* d_r = nullptr;
    float* d_z = nullptr;
    float* d_p = nullptr;
    float* d_inv = nullptr;
    cublasHandle_t handle = nullptr;
    cudaEvent_t start{};
    cudaEvent_t stop{};

    try {
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_x), bytes),
                       "cudaMalloc(SA coarse PCG x)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_r), bytes),
                       "cudaMalloc(SA coarse PCG r)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_z), bytes),
                       "cudaMalloc(SA coarse PCG z)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_p), bytes),
                       "cudaMalloc(SA coarse PCG p)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_inv), bytes),
                       "cudaMalloc(SA coarse PCG inverse diagonal)");
        check_cuda_pcg(cudaMemcpy(d_r, rhs.data(), bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy(SA coarse PCG rhs H2D)");
        check_cuda_pcg(cudaMemcpy(d_inv, inverse_preconditioner.data(), bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(SA coarse PCG inverse diagonal H2D)");
        check_cuda_pcg(cudaMemset(d_x, 0, bytes), "cudaMemset(SA coarse PCG x)");

        check_cublas_pcg(cublasCreate(&handle), "cublasCreate(SA coarse PCG)");
        check_cublas_pcg(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                         "cublasSetPointerMode(SA coarse PCG)");

        launch_coarse_precondition_sa(n, d_inv, d_r, d_z);
        check_cuda_pcg(cudaMemcpyAsync(d_p, d_z, bytes, cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(SA coarse PCG p=z)");
        float rho = dot_pcg(handle, n, d_r, d_z);
        if (!(rho > 0.0f) || !std::isfinite(rho)) {
            throw std::runtime_error("GPU coarse PCG initial rho invalid");
        }

        // Warm one coarse matvec outside the timed iteration loop. This also
        // warms the fused transfer kernels and GoldSparse center action.
        check_cuda_pcg(cudaMemcpyAsync(impl_->d_coarse_x, d_p, bytes,
                                       cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(SA coarse PCG warm input)");
        impl_->run_pipeline(transfer_smoothing_steps, nullptr, nullptr);
        check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(SA coarse PCG warmup)");

        check_cuda_pcg(cudaEventCreate(&start), "cudaEventCreate(SA coarse PCG start)");
        check_cuda_pcg(cudaEventCreate(&stop), "cudaEventCreate(SA coarse PCG stop)");
        check_cuda_pcg(cudaEventRecord(start), "cudaEventRecord(SA coarse PCG start)");

        std::size_t completed = 0U;
        for (std::size_t it = 0; it < max_iterations; ++it) {
            check_cuda_pcg(cudaMemcpyAsync(impl_->d_coarse_x, d_p, bytes,
                                           cudaMemcpyDeviceToDevice),
                           "cudaMemcpyAsync(SA coarse PCG matvec input)");
            impl_->run_pipeline(transfer_smoothing_steps, nullptr, nullptr);

            const float pq = dot_pcg(handle, n, d_p, impl_->d_coarse_y);
            if (!(pq > 0.0f) || !std::isfinite(pq)) {
                throw std::runtime_error("GPU coarse PCG lost positive curvature");
            }
            const float alpha = rho / pq;
            constexpr int threads = 256;
            const int blocks = (n + threads - 1) / threads;
            update_x_r_pcg_kernel<<<blocks, threads>>>(
                n, alpha, d_p, impl_->d_coarse_y, d_x, d_r);
            check_cuda_pcg(cudaGetLastError(), "SA coarse PCG x/r update launch");

            launch_coarse_precondition_sa(n, d_inv, d_r, d_z);
            const float rho_new = dot_pcg(handle, n, d_r, d_z);
            if (!(rho_new > 0.0f) || !std::isfinite(rho_new)) {
                throw std::runtime_error("GPU coarse PCG rho became invalid");
            }
            const float beta = rho_new / rho;
            update_p_pcg_kernel<<<blocks, threads>>>(n, beta, d_z, d_p);
            check_cuda_pcg(cudaGetLastError(), "SA coarse PCG p update launch");
            rho = rho_new;
            completed = it + 1U;
        }

        const float rr = dot_pcg(handle, n, d_r, d_r);
        check_cuda_pcg(cudaEventRecord(stop), "cudaEventRecord(SA coarse PCG stop)");
        check_cuda_pcg(cudaEventSynchronize(stop), "cudaEventSynchronize(SA coarse PCG stop)");
        float solve_ms = 0.0f;
        check_cuda_pcg(cudaEventElapsedTime(&solve_ms, start, stop),
                       "cudaEventElapsedTime(SA coarse PCG)");

        GpuSmoothedAggregationCoarsePcgResult result;
        result.x.resize(impl_->coarse_dof_count);
        check_cuda_pcg(cudaMemcpy(result.x.data(), d_x, bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(SA coarse PCG x D2H)");
        result.iterations = completed;
        result.relative_residual =
            std::sqrt(std::max(0.0, static_cast<double>(rr)) / bnorm2_host);
        result.solve_ms = static_cast<double>(solve_ms);
        // Five benchmark-local vectors: x,r,z,p,M^-1. The context's existing
        // d_coarse_x/d_coarse_y are reused as matvec input/output.
        result.persistent_coarse_pcg_bytes = 5U * bytes;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cublasDestroy(handle);
        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_z);
        cudaFree(d_p);
        cudaFree(d_inv);
        return result;
    } catch (...) {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (handle) cublasDestroy(handle);
        if (d_x) cudaFree(d_x);
        if (d_r) cudaFree(d_r);
        if (d_z) cudaFree(d_z);
        if (d_p) cudaFree(d_p);
        if (d_inv) cudaFree(d_inv);
        throw;
    }
}

}  // namespace gfss

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kChebyshevLowerFraction = 0.10;
constexpr double kLambdaSafety = 1.25;
constexpr double kSaDampingNumerator = 4.0 / 3.0;

struct FactorizedTransferCpu {
    const gfss::StructuredHexMesh& mesh;
    const gfss::Material& material;
    const gfss::ElasticityAggregationCoarseSpace& space;
    const std::vector<double>& inverse_diagonal;
    double omega{0.0};
    std::size_t steps{0};

    std::vector<double> prolong(const std::vector<double>& coarse) const;
    std::vector<double> restrict_transpose(const std::vector<double>& fine) const;
};

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("inexact sweep dot size mismatch");
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) value += a[i] * b[i];
    return value;
}

double norm(const std::vector<double>& v) {
    return std::sqrt(std::max(0.0, dot(v, v)));
}

void clamp_x0(const gfss::StructuredHexMesh& mesh, std::vector<double>& v) {
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = static_cast<std::size_t>(mesh.node_index(0U, j, k));
            v[3U * node + 0U] = 0.0;
            v[3U * node + 1U] = 0.0;
            v[3U * node + 2U] = 0.0;
        }
    }
}

std::vector<double> apply_clamped_openmp(const gfss::StructuredHexMesh& mesh,
                                          const gfss::Material& material,
                                          const std::vector<double>& x) {
    auto free_x = x;
    clamp_x0(mesh, free_x);
    auto y = gfss::apply_matrix_free_openmp(mesh, material, free_x);
    clamp_x0(mesh, y);
    return y;
}

std::vector<double> build_inverse_diagonal(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space) {
    std::vector<double> diagonal(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                for (std::size_t a = 0; a < 8U; ++a) {
                    const auto node = static_cast<std::size_t>(nodes[a]);
                    if (space.graph.constrained[node] != 0U) continue;
                    for (std::size_t c = 0; c < 3U; ++c) {
                        const std::size_t local = 3U * a + c;
                        diagonal[3U * node + c] += ke[local][local];
                    }
                }
            }
        }
    }
    std::vector<double> inverse(diagonal.size(), 0.0);
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        if (space.graph.constrained[node] != 0U) continue;
        for (std::size_t c = 0; c < 3U; ++c) {
            const std::size_t dof = 3U * node + c;
            if (!(diagonal[dof] > 0.0) || !std::isfinite(diagonal[dof])) {
                throw std::runtime_error("inexact sweep fine Jacobi diagonal invalid");
            }
            inverse[dof] = 1.0 / diagonal[dof];
        }
    }
    return inverse;
}

double estimate_lambda_max(const gfss::StructuredHexMesh& mesh,
                           const gfss::Material& material,
                           const std::vector<double>& inverse_diagonal,
                           std::size_t iterations) {
    std::vector<double> q(inverse_diagonal.size(), 0.0);
    std::vector<double> scaled(q.size(), 0.0);
    for (std::size_t i = 0; i < q.size(); ++i) {
        if (inverse_diagonal[i] > 0.0) {
            const double t = static_cast<double>((i % 251U) + 1U);
            q[i] = std::sin(0.173 * t) + 0.37 * std::cos(0.071 * t);
        }
    }
    double qnorm = norm(q);
    if (!(qnorm > 0.0)) throw std::runtime_error("inexact sweep lambda probe zero");
    for (double& v : q) v /= qnorm;

    double lambda = 0.0;
    for (std::size_t it = 0; it < iterations; ++it) {
        for (std::size_t i = 0; i < q.size(); ++i) {
            scaled[i] = inverse_diagonal[i] > 0.0
                ? std::sqrt(inverse_diagonal[i]) * q[i] : 0.0;
        }
        auto y = apply_clamped_openmp(mesh, material, scaled);
        for (std::size_t i = 0; i < y.size(); ++i) {
            y[i] = inverse_diagonal[i] > 0.0
                ? std::sqrt(inverse_diagonal[i]) * y[i] : 0.0;
        }
        const double rq = dot(q, y);
        const double ynorm = norm(y);
        if (!(rq > 0.0) || !(ynorm > 0.0)) {
            throw std::runtime_error("inexact sweep lambda estimate invalid");
        }
        lambda = std::max(lambda, rq);
        q = std::move(y);
        for (double& v : q) v /= ynorm;
    }
    return kLambdaSafety * lambda;
}

void chebyshev_jacobi_smooth(const gfss::StructuredHexMesh& mesh,
                             const gfss::Material& material,
                             const std::vector<double>& inverse_diagonal,
                             double lambda_max,
                             const std::vector<double>& b,
                             std::vector<double>& x,
                             std::size_t degree) {
    const double lambda_low = kChebyshevLowerFraction * lambda_max;
    const double theta = 0.5 * (lambda_max + lambda_low);
    const double delta = 0.5 * (lambda_max - lambda_low);
    for (std::size_t k = 0; k < degree; ++k) {
        const double angle = kPi * (2.0 * static_cast<double>(k) + 1.0) /
                             (2.0 * static_cast<double>(degree));
        const double root = theta + delta * std::cos(angle);
        auto ax = apply_clamped_openmp(mesh, material, x);
        const double weight = 1.0 / root;
        for (std::size_t i = 0; i < x.size(); ++i) {
            if (inverse_diagonal[i] > 0.0) {
                x[i] += weight * inverse_diagonal[i] * (b[i] - ax[i]);
            }
        }
        clamp_x0(mesh, x);
    }
}

std::vector<double> FactorizedTransferCpu::prolong(
    const std::vector<double>& coarse) const {
    auto fine = gfss::apply_elasticity_tentative_prolongation(space, coarse);
    for (std::size_t step = 0; step < steps; ++step) {
        const auto af = apply_clamped_openmp(mesh, material, fine);
        for (std::size_t i = 0; i < fine.size(); ++i) {
            if (inverse_diagonal[i] > 0.0) {
                fine[i] -= omega * inverse_diagonal[i] * af[i];
            }
        }
        clamp_x0(mesh, fine);
    }
    return fine;
}

std::vector<double> FactorizedTransferCpu::restrict_transpose(
    const std::vector<double>& fine) const {
    auto work = fine;
    clamp_x0(mesh, work);
    std::vector<double> scaled(work.size(), 0.0);
    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t i = 0; i < work.size(); ++i) {
            scaled[i] = inverse_diagonal[i] * work[i];
        }
        const auto a_scaled = apply_clamped_openmp(mesh, material, scaled);
        for (std::size_t i = 0; i < work.size(); ++i) {
            work[i] -= omega * a_scaled[i];
        }
        clamp_x0(mesh, work);
    }
    return gfss::apply_elasticity_tentative_restriction(space, work);
}

std::vector<double> make_rhs(const gfss::StructuredHexMesh& mesh) {
    std::vector<double> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const double count = static_cast<double>(mesh.ny + 1U) *
                         static_cast<double>(mesh.nz + 1U);
    const double magnitude = 1.0 / count;
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = -magnitude;
        }
    }
    return rhs;
}

std::vector<std::size_t> parse_budgets(const std::string& text) {
    std::vector<std::size_t> budgets;
    std::size_t start = 0U;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token = text.substr(
            start, comma == std::string::npos ? std::string::npos : comma - start);
        const auto value = static_cast<std::size_t>(std::stoull(token));
        if (value == 0U) throw std::invalid_argument("coarse budgets must be positive");
        budgets.push_back(value);
        if (comma == std::string::npos) break;
        start = comma + 1U;
    }
    if (budgets.empty()) throw std::invalid_argument("coarse budget list is empty");
    return budgets;
}

void run_budget(std::size_t m,
                std::size_t budget,
                std::size_t cycles,
                const gfss::StructuredHexMesh& mesh,
                const gfss::Material& material,
                const gfss::ElasticityAggregationCoarseSpace& space,
                const std::vector<double>& rhs,
                const std::vector<double>& fine_inverse_diagonal,
                const std::vector<float>& coarse_inverse_diagonal,
                double lambda_max,
                double omega,
                gfss::GpuSmoothedAggregationContext& gpu) {
    FactorizedTransferCpu transfer{
        mesh, material, space, fine_inverse_diagonal, omega, m};
    const double bnorm = norm(rhs);
    std::vector<double> x(rhs.size(), 0.0);
    std::vector<double> residual(rhs.size(), 0.0);
    std::vector<double> residual_history;
    residual_history.reserve(cycles + 1U);
    residual_history.push_back(1.0);

    double total_gpu_coarse_ms = 0.0;
    double last_coarse_rel = 0.0;
    std::size_t pcg_bytes = 0U;

    for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
        chebyshev_jacobi_smooth(mesh, material, fine_inverse_diagonal,
                                lambda_max, rhs, x, 3U);
        auto ax = apply_clamped_openmp(mesh, material, x);
        for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
        const auto coarse_rhs_d = transfer.restrict_transpose(residual);
        std::vector<float> coarse_rhs(coarse_rhs_d.size(), 0.0f);
        for (std::size_t i = 0; i < coarse_rhs.size(); ++i) {
            coarse_rhs[i] = static_cast<float>(coarse_rhs_d[i]);
        }

        const auto coarse = gpu.solve_coarse_pcg_fixed_iterations(
            coarse_rhs, coarse_inverse_diagonal, m, budget);
        total_gpu_coarse_ms += coarse.solve_ms;
        last_coarse_rel = coarse.relative_residual;
        pcg_bytes = coarse.persistent_coarse_pcg_bytes;

        std::vector<double> coarse_x(coarse.x.size(), 0.0);
        for (std::size_t i = 0; i < coarse_x.size(); ++i) {
            coarse_x[i] = static_cast<double>(coarse.x[i]);
        }
        const auto correction = transfer.prolong(coarse_x);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += correction[i];
        clamp_x0(mesh, x);

        chebyshev_jacobi_smooth(mesh, material, fine_inverse_diagonal,
                                lambda_max, rhs, x, 3U);
        ax = apply_clamped_openmp(mesh, material, x);
        for (std::size_t i = 0; i < rhs.size(); ++i) residual[i] = rhs[i] - ax[i];
        residual_history.push_back(norm(residual) / bnorm);
    }

    std::cout << "budget=" << budget
              << " gpu_coarse_total_ms=" << std::fixed << std::setprecision(6)
              << total_gpu_coarse_ms
              << " gpu_coarse_ms_per_cycle=" << total_gpu_coarse_ms / static_cast<double>(cycles)
              << " coarse_pcg_workspace_bytes=" << pcg_bytes
              << " coarse_pcg_workspace_bytes_per_fine_free_dof="
              << static_cast<double>(pcg_bytes) / static_cast<double>(space.fine_free_dofs)
              << std::scientific << std::setprecision(9)
              << " last_coarse_relative_residual=" << last_coarse_rel;
    for (std::size_t i = 1; i < residual_history.size(); ++i) {
        std::cout << " r" << i << '=' << residual_history[i]
                  << " q" << i << '=' << residual_history[i] / residual_history[i - 1U];
    }
    if (residual_history.size() > 2U) {
        double log_sum = 0.0;
        std::size_t count = 0U;
        for (std::size_t i = 2U; i < residual_history.size(); ++i) {
            const double q = residual_history[i] / residual_history[i - 1U];
            if (q > 0.0 && std::isfinite(q)) {
                log_sum += std::log(q);
                ++count;
            }
        }
        if (count > 0U) {
            std::cout << " post_transient_geomean_q=" <<
                std::exp(log_sum / static_cast<double>(count));
        }
    }
    std::cout << '\n';
}

void run_m(std::size_t m,
           const std::vector<std::size_t>& budgets,
           std::size_t cycles,
           int block_y,
           std::size_t target_nodes,
           std::size_t min_nodes) {
    const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
    const gfss::Material material{210.0e9, 0.30};
    auto graph = gfss::build_structured_hex_nodal_graph_x0(mesh);
    const auto space = gfss::build_elasticity_aggregation_coarse_space(
        std::move(graph), {target_nodes, min_nodes, 1.0e-10});
    const auto tentative = gfss::assemble_structured_hex_aggregation_galerkin(
        mesh, material, space);
    const auto fine_inverse = build_inverse_diagonal(mesh, material, space);
    const double lambda_max = estimate_lambda_max(mesh, material, fine_inverse, 8U);
    const double omega = kSaDampingNumerator / lambda_max;
    const auto rhs = make_rhs(mesh);

    std::vector<float> coarse_inverse(tentative.inverse_diagonal.size(), 0.0f);
    for (std::size_t i = 0; i < coarse_inverse.size(); ++i) {
        coarse_inverse[i] = static_cast<float>(tentative.inverse_diagonal[i]);
    }

    gfss::GpuSmoothedAggregationContext gpu(mesh, material, space, omega, block_y);

    std::cout << "\n========================================\n"
              << "transfer_smoothing_steps=" << m << '\n'
              << "fine_lambda_max_est=" << std::fixed << std::setprecision(6) << lambda_max
              << " transfer_omega=" << omega << '\n'
              << "fine_free_dofs=" << space.fine_free_dofs
              << " coarse_dofs=" << space.coarse_dofs
              << " aggregates=" << space.aggregates.size()
              << " fine_to_coarse_dof_ratio="
              << static_cast<double>(space.fine_free_dofs) /
                     static_cast<double>(space.coarse_dofs) << '\n';

    for (std::size_t budget : budgets) {
        run_budget(m, budget, cycles, mesh, material, space, rhs,
                   fine_inverse, coarse_inverse, lambda_max, omega, gpu);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "all";
        const std::vector<std::size_t> budgets = parse_budgets(
            argc > 2 ? argv[2] : "5,10,20,40,80,160");
        const std::size_t cycles = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 4U;
        const int block_y = argc > 4 ? std::stoi(argv[4]) : 4;
        const std::size_t target_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 12U;
        const std::size_t min_nodes = argc > 6
            ? static_cast<std::size_t>(std::stoull(argv[6])) : 4U;
        if (cycles < 2U || block_y <= 0 || target_nodes == 0U || min_nodes == 0U) {
            throw std::invalid_argument("invalid inexact coarse sweep options");
        }

        std::cout << "GFSS M5 GPU inexact coarse-solve sweep\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "purpose=measure_two_grid_q_vs_fixed_GPU_coarse_PCG_budget\n"
                  << "fine_shell=cpu_fp64_reference_pre_post_smoothing_and_true_residual\n"
                  << "coarse_solver=gpu_fp32_fixed_budget_PCG\n"
                  << "coarse_operator=P_m^T_A_f_P_m_persistent_GPU\n"
                  << "coarse_preconditioner=tentative_Ac_diagonal_reference\n"
                  << "coarse_action_H2D_D2H_per_iteration=false\n"
                  << "coarse_PCG_alloc_H2D_D2H_excluded_from_solve_ms=true\n"
                  << "pre_smooth_degree=3 post_smooth_degree=3 power_iterations=8\n"
                  << "cycles_per_budget=" << cycles
                  << " block_y=" << block_y
                  << " target_nodes=" << target_nodes
                  << " min_nodes=" << min_nodes << '\n';

        if (selector == "all" || selector == "1") {
            run_m(1U, budgets, cycles, block_y, target_nodes, min_nodes);
        }
        if (selector == "all" || selector == "2") {
            run_m(2U, budgets, cycles, block_y, target_nodes, min_nodes);
        }
        if (selector != "all" && selector != "1" && selector != "2") {
            throw std::invalid_argument("selector must be 1, 2, or all");
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_inexact_coarse_sweep_bench "
                  << "[1|2|all [budgets_csv [cycles [block_y [target_nodes [min_nodes]]]]]]\n";
        return 1;
    }
}
