#include "gpu_pcg_audit.cu"

#include <limits>
#include <memory>
#include <utility>

namespace gfss {

struct GpuPcgContext::Impl {
    StructuredHexMesh mesh;
    Material material;
    int block_y{4};
    std::size_t nodes{0};
    std::size_t ndof{0};
    std::size_t vector_bytes{0};
    int n{0};

    float* d_b{nullptr};
    float* d_x{nullptr};
    float* d_r{nullptr};
    float* d_z{nullptr};
    float* d_p{nullptr};
    float* d_q{nullptr};
    cublasHandle_t handle{nullptr};
    std::vector<float> host_b;
    std::vector<float> host_x;

    Impl(const StructuredHexMesh& mesh_in,
         const Material& material_in,
         int block_y_in)
        : mesh(mesh_in), material(material_in), block_y(block_y_in) {
        if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
            throw std::invalid_argument(
                "GPU PCG context requires non-empty mesh dimensions");
        }
        if (block_y <= 0 || block_y > 32 || 32 * block_y > 1024) {
            throw std::invalid_argument(
                "GPU PCG context block_y must produce a valid 32 x block_y block");
        }
        if (mesh.node_count() >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "GPU PCG context currently requires node_count <= INT_MAX");
        }
        if (mesh.dof_count() >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument(
                "GPU PCG context currently requires dof_count <= INT_MAX");
        }

        nodes = static_cast<std::size_t>(mesh.node_count());
        ndof = static_cast<std::size_t>(mesh.dof_count());
        vector_bytes = ndof * sizeof(float);
        n = static_cast<int>(ndof);
        host_b.assign(ndof, 0.0f);
        host_x.resize(ndof);

        try {
            upload_pcg_stencil(mesh, material);

            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_b), vector_bytes),
                           "cudaMalloc(PCG context b)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_x), vector_bytes),
                           "cudaMalloc(PCG context x)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_r), vector_bytes),
                           "cudaMalloc(PCG context r)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_z), vector_bytes),
                           "cudaMalloc(PCG context z)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_p), vector_bytes),
                           "cudaMalloc(PCG context p)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_q), vector_bytes),
                           "cudaMalloc(PCG context q)");

            check_cublas_pcg(cublasCreate(&handle),
                             "cublasCreate(PCG context)");
            check_cublas_pcg(
                cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                "cublasSetPointerMode(PCG context)");

            check_cuda_pcg(cudaMemset(d_p, 0, vector_bytes),
                           "cudaMemset(PCG context warm p)");
            launch_pcg_matvec(mesh, block_y, nodes, d_p, d_q);
            check_cuda_pcg(cudaDeviceSynchronize(),
                           "cudaDeviceSynchronize(PCG context warmup)");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Impl() { cleanup(); }

    void cleanup() noexcept {
        if (handle) {
            cublasDestroy(handle);
            handle = nullptr;
        }
        if (d_b) cudaFree(d_b);
        if (d_x) cudaFree(d_x);
        if (d_r) cudaFree(d_r);
        if (d_z) cudaFree(d_z);
        if (d_p) cudaFree(d_p);
        if (d_q) cudaFree(d_q);
        d_b = d_x = d_r = d_z = d_p = d_q = nullptr;
    }

    GpuPcgResult solve(const std::vector<float>& rhs,
                       double relative_tolerance,
                       std::size_t max_iterations) {
        if (rhs.size() != ndof) {
            throw std::invalid_argument(
                "GPU PCG context RHS size does not match mesh DOF count");
        }
        if (!(relative_tolerance > 0.0) || relative_tolerance >= 1.0) {
            throw std::invalid_argument(
                "GPU PCG context relative tolerance must be in (0, 1)");
        }
        if (max_iterations == 0U) {
            throw std::invalid_argument(
                "GPU PCG context max_iterations must be positive");
        }

        for (std::size_t node = 0; node < nodes; ++node) {
            host_b[node] = rhs[3U * node + 0U];
            host_b[nodes + node] = rhs[3U * node + 1U];
            host_b[2U * nodes + node] = rhs[3U * node + 2U];
        }
        for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
            for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
                const std::size_t node = static_cast<std::size_t>(
                    mesh.node_index(0U, j, k));
                host_b[node] = 0.0f;
                host_b[nodes + node] = 0.0f;
                host_b[2U * nodes + node] = 0.0f;
            }
        }

        check_cuda_pcg(cudaMemcpy(d_b, host_b.data(), vector_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(PCG context b H2D)");

        const auto wall_start = Clock::now();

        check_cuda_pcg(cudaMemsetAsync(d_x, 0, vector_bytes),
                       "cudaMemsetAsync(PCG context x)");
        check_cuda_pcg(cudaMemcpyAsync(d_r, d_b, vector_bytes,
                                       cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(PCG context r=b)");
        launch_pcg_jacobi(mesh, block_y, nodes, d_r, d_z);
        check_cuda_pcg(cudaMemcpyAsync(d_p, d_z, vector_bytes,
                                       cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(PCG context p=z)");

        const float bnorm2 = dot_pcg(handle, n, d_b, d_b);
        float rho = dot_pcg(handle, n, d_r, d_z);
        float rr = dot_pcg(handle, n, d_r, d_r);

        GpuPcgResult result;
        result.explicit_device_bytes = 6U * vector_bytes;
        result.requested_relative_residual = relative_tolerance;
        result.reported_relative_residual =
            bnorm2 > 0.0f
                ? std::sqrt(static_cast<double>(rr) /
                            static_cast<double>(bnorm2))
                : 0.0;
        result.audited_relative_residual = result.reported_relative_residual;
        result.best_audited_relative_residual =
            result.reported_relative_residual;

        if (!(bnorm2 >= 0.0f) || !std::isfinite(bnorm2)) {
            throw std::runtime_error(
                "GPU PCG context RHS norm became invalid");
        }

        if (bnorm2 == 0.0f) {
            result.converged = true;
        } else {
            constexpr int vector_threads = 256;
            const int vector_blocks =
                (n + vector_threads - 1) / vector_threads;
            constexpr double milestone_factor = 0.31622776601683794;
            constexpr std::size_t target_audit_period = 32U;
            constexpr std::size_t stagnation_audits = 6U;
            constexpr double meaningful_improvement = 0.005;

            double next_milestone = 1.0e-1;
            std::size_t last_audit_iteration = 0U;
            std::size_t no_progress_audits = 0U;
            double last_target_audited =
                std::numeric_limits<double>::infinity();

            auto audit_current = [&](std::size_t iteration) {
                launch_pcg_matvec(mesh, block_y, nodes, d_x, d_q);
                ++result.matvecs;
                ++result.residual_audits;
                residual_from_b_ax_pcg_kernel<<<vector_blocks,
                                               vector_threads>>>(
                    n, d_b, d_q, d_z);
                check_cuda_pcg(cudaGetLastError(),
                               "GPU PCG context residual audit launch");
                const float rr_audit = dot_pcg(handle, n, d_z, d_z);
                const double audited =
                    std::sqrt(static_cast<double>(rr_audit) /
                              static_cast<double>(bnorm2));
                if (!std::isfinite(rr_audit) || !std::isfinite(audited)) {
                    throw std::runtime_error(
                        "GPU PCG context audited residual became non-finite");
                }
                result.audited_relative_residual = audited;
                if (audited < result.best_audited_relative_residual) {
                    result.best_audited_relative_residual = audited;
                    result.best_audited_iteration = iteration;
                }
                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - wall_start).count();
                result.audit_samples.push_back(GpuPcgAuditSample{
                    iteration,
                    result.reported_relative_residual,
                    audited,
                    elapsed_ms});
                last_audit_iteration = iteration;
                return audited;
            };

            for (std::size_t iteration = 0;
                 iteration < max_iterations;
                 ++iteration) {
                launch_pcg_matvec(mesh, block_y, nodes, d_p, d_q);
                ++result.matvecs;

                const float pq = dot_pcg(handle, n, d_p, d_q);
                if (!(pq > 0.0f) || !std::isfinite(pq) ||
                    !(rho > 0.0f) || !std::isfinite(rho)) {
                    throw std::runtime_error(
                        "GPU PCG context breakdown: non-positive or non-finite scalar");
                }

                const float alpha = rho / pq;
                update_x_r_pcg_kernel<<<vector_blocks, vector_threads>>>(
                    n, alpha, d_p, d_q, d_x, d_r);
                check_cuda_pcg(cudaGetLastError(),
                               "GPU PCG context x/r update launch");

                rr = dot_pcg(handle, n, d_r, d_r);
                result.iterations = iteration + 1U;
                result.reported_relative_residual =
                    std::sqrt(static_cast<double>(rr) /
                              static_cast<double>(bnorm2));

                if (!std::isfinite(rr) ||
                    !std::isfinite(result.reported_relative_residual)) {
                    throw std::runtime_error(
                        "GPU PCG context residual became non-finite");
                }

                const bool milestone_due =
                    result.reported_relative_residual <= next_milestone;
                const bool target_region =
                    result.reported_relative_residual <= relative_tolerance;
                const bool periodic_target_due =
                    target_region &&
                    (last_audit_iteration == 0U ||
                     result.iterations - last_audit_iteration >=
                         target_audit_period);

                if (milestone_due || periodic_target_due) {
                    const double audited = audit_current(result.iterations);

                    while (result.reported_relative_residual <= next_milestone &&
                           next_milestone > 1.0e-12) {
                        next_milestone *= milestone_factor;
                    }

                    if (audited <= relative_tolerance) {
                        result.converged = true;
                        break;
                    }

                    if (target_region) {
                        if (std::isfinite(last_target_audited)) {
                            const double required =
                                last_target_audited *
                                (1.0 - meaningful_improvement);
                            if (audited >= required) {
                                ++no_progress_audits;
                            } else {
                                no_progress_audits = 0U;
                            }
                        }
                        last_target_audited = audited;
                        if (no_progress_audits >= stagnation_audits) {
                            result.stagnated = true;
                            break;
                        }
                    }
                }

                launch_pcg_jacobi(mesh, block_y, nodes, d_r, d_z);
                const float rho_new = dot_pcg(handle, n, d_r, d_z);
                if (!(rho_new > 0.0f) || !std::isfinite(rho_new)) {
                    throw std::runtime_error(
                        "GPU PCG context breakdown: preconditioned residual became invalid");
                }
                const float beta = rho_new / rho;
                update_p_pcg_kernel<<<vector_blocks, vector_threads>>>(
                    n, beta, d_z, d_p);
                check_cuda_pcg(cudaGetLastError(),
                               "GPU PCG context p update launch");
                rho = rho_new;
            }

            if (!result.converged &&
                last_audit_iteration != result.iterations) {
                audit_current(result.iterations);
            }
        }

        check_cuda_pcg(cudaDeviceSynchronize(),
                       "cudaDeviceSynchronize(PCG context solve)");
        const auto wall_stop = Clock::now();
        result.solve_ms =
            std::chrono::duration<double, std::milli>(
                wall_stop - wall_start).count();

        check_cuda_pcg(cudaMemcpy(host_x.data(), d_x, vector_bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(PCG context x D2H)");
        result.x.resize(ndof);
        for (std::size_t node = 0; node < nodes; ++node) {
            result.x[3U * node + 0U] = host_x[node];
            result.x[3U * node + 1U] = host_x[nodes + node];
            result.x[3U * node + 2U] = host_x[2U * nodes + node];
        }
        return result;
    }
};

GpuPcgContext::GpuPcgContext(const StructuredHexMesh& mesh,
                             const Material& material,
                             int block_y)
    : impl_(std::make_unique<Impl>(mesh, material, block_y)) {}

GpuPcgContext::~GpuPcgContext() = default;
GpuPcgContext::GpuPcgContext(GpuPcgContext&&) noexcept = default;
GpuPcgContext& GpuPcgContext::operator=(GpuPcgContext&&) noexcept = default;

GpuPcgResult GpuPcgContext::solve(const std::vector<float>& rhs,
                                  double relative_tolerance,
                                  std::size_t max_iterations) {
    if (!impl_) {
        throw std::runtime_error("GPU PCG context is empty");
    }
    return impl_->solve(rhs, relative_tolerance, max_iterations);
}

std::size_t GpuPcgContext::explicit_device_bytes() const noexcept {
    return impl_ ? 6U * impl_->vector_bytes : 0U;
}

}  // namespace gfss
