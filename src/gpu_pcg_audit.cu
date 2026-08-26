#define solve_pcg_cuda_gold_sparse_x0 solve_pcg_cuda_gold_sparse_x0_restart_baseline
#include "gpu_pcg.cu"
#undef solve_pcg_cuda_gold_sparse_x0

namespace gfss {

GpuPcgResult solve_pcg_cuda_gold_sparse_x0(
    const StructuredHexMesh& mesh,
    const Material& material,
    const std::vector<float>& rhs,
    double relative_tolerance,
    std::size_t max_iterations,
    int block_y) {
    if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
        throw std::invalid_argument("GPU PCG requires non-empty mesh dimensions");
    }
    if (rhs.size() != static_cast<std::size_t>(mesh.dof_count())) {
        throw std::invalid_argument("GPU PCG RHS size does not match mesh DOF count");
    }
    if (!(relative_tolerance > 0.0)) {
        throw std::invalid_argument("GPU PCG relative tolerance must be positive");
    }
    if (max_iterations == 0U) {
        throw std::invalid_argument("GPU PCG max_iterations must be positive");
    }
    if (block_y <= 0 || block_y > 32 || 32 * block_y > 1024) {
        throw std::invalid_argument("GPU PCG block_y must produce a valid 32 x block_y block");
    }
    if (mesh.node_count() > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("GPU PCG currently requires node_count <= INT_MAX");
    }
    if (mesh.dof_count() > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("GPU PCG currently requires dof_count <= INT_MAX for cuBLAS reductions");
    }

    upload_pcg_stencil(mesh, material);

    const std::size_t nodes = static_cast<std::size_t>(mesh.node_count());
    const std::size_t ndof = static_cast<std::size_t>(mesh.dof_count());
    const std::size_t vector_bytes = ndof * sizeof(float);
    const int n = static_cast<int>(ndof);

    std::vector<float> host_b(ndof, 0.0f);
    for (std::size_t node = 0; node < nodes; ++node) {
        host_b[node] = rhs[3U * node + 0U];
        host_b[nodes + node] = rhs[3U * node + 1U];
        host_b[2U * nodes + node] = rhs[3U * node + 2U];
    }
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const std::size_t node =
                static_cast<std::size_t>(mesh.node_index(0U, j, k));
            host_b[node] = 0.0f;
            host_b[nodes + node] = 0.0f;
            host_b[2U * nodes + node] = 0.0f;
        }
    }

    float* d_b = nullptr;
    float* d_x = nullptr;
    float* d_r = nullptr;
    float* d_z = nullptr;
    float* d_p = nullptr;
    float* d_q = nullptr;
    cublasHandle_t handle = nullptr;

    try {
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_b), vector_bytes),
                       "cudaMalloc(PCG b)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_x), vector_bytes),
                       "cudaMalloc(PCG x)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_r), vector_bytes),
                       "cudaMalloc(PCG r)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_z), vector_bytes),
                       "cudaMalloc(PCG z)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_p), vector_bytes),
                       "cudaMalloc(PCG p)");
        check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_q), vector_bytes),
                       "cudaMalloc(PCG q)");
        check_cuda_pcg(cudaMemcpy(d_b, host_b.data(), vector_bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(PCG b H2D)");

        check_cublas_pcg(cublasCreate(&handle), "cublasCreate(PCG)");
        check_cublas_pcg(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST),
                         "cublasSetPointerMode(PCG)");

        // Warm the exact matvec path without changing timed PCG state.
        check_cuda_pcg(cudaMemset(d_p, 0, vector_bytes),
                       "cudaMemset(PCG warm p)");
        launch_pcg_matvec(mesh, block_y, nodes, d_p, d_q);
        check_cuda_pcg(cudaDeviceSynchronize(),
                       "cudaDeviceSynchronize(PCG warmup)");

        const auto wall_start = Clock::now();

        check_cuda_pcg(cudaMemsetAsync(d_x, 0, vector_bytes),
                       "cudaMemsetAsync(PCG x)");
        check_cuda_pcg(cudaMemcpyAsync(d_r, d_b, vector_bytes,
                                       cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(PCG r=b)");
        launch_pcg_jacobi(mesh, block_y, nodes, d_r, d_z);
        check_cuda_pcg(cudaMemcpyAsync(d_p, d_z, vector_bytes,
                                       cudaMemcpyDeviceToDevice),
                       "cudaMemcpyAsync(PCG p=z)");

        const float bnorm2 = dot_pcg(handle, n, d_b, d_b);
        float rho = dot_pcg(handle, n, d_r, d_z);
        float rr = dot_pcg(handle, n, d_r, d_r);

        GpuPcgResult result;
        result.explicit_device_bytes = 6U * vector_bytes;
        result.reported_relative_residual =
            bnorm2 > 0.0f
                ? std::sqrt(static_cast<double>(rr) /
                            static_cast<double>(bnorm2))
                : 0.0;
        result.audited_relative_residual = result.reported_relative_residual;

        if (!(bnorm2 >= 0.0f) || !std::isfinite(bnorm2)) {
            throw std::runtime_error("GPU PCG RHS norm became invalid");
        }

        if (bnorm2 == 0.0f) {
            result.converged = true;
        } else {
            constexpr int vector_threads = 256;
            const int vector_blocks =
                (n + vector_threads - 1) / vector_threads;

            for (std::size_t iteration = 0;
                 iteration < max_iterations;
                 ++iteration) {
                launch_pcg_matvec(mesh, block_y, nodes, d_p, d_q);
                ++result.matvecs;

                const float pq = dot_pcg(handle, n, d_p, d_q);
                if (!(pq > 0.0f) || !std::isfinite(pq) ||
                    !(rho > 0.0f) || !std::isfinite(rho)) {
                    throw std::runtime_error(
                        "GPU PCG breakdown: non-positive or non-finite scalar");
                }

                const float alpha = rho / pq;
                update_x_r_pcg_kernel<<<vector_blocks, vector_threads>>>(
                    n, alpha, d_p, d_q, d_x, d_r);
                check_cuda_pcg(cudaGetLastError(),
                               "GPU PCG x/r update launch");

                rr = dot_pcg(handle, n, d_r, d_r);
                result.iterations = iteration + 1U;
                result.reported_relative_residual =
                    std::sqrt(static_cast<double>(rr) /
                              static_cast<double>(bnorm2));

                if (!std::isfinite(rr) ||
                    !std::isfinite(result.reported_relative_residual)) {
                    throw std::runtime_error(
                        "GPU PCG residual became non-finite");
                }

                const bool recursive_candidate =
                    result.reported_relative_residual <= relative_tolerance;

                if (recursive_candidate) {
                    // Audit the actual current solution without changing the
                    // recursive residual or the Krylov search direction.
                    // d_z is safe scratch here and is rebuilt by Jacobi below.
                    launch_pcg_matvec(mesh, block_y, nodes, d_x, d_q);
                    ++result.matvecs;
                    ++result.residual_audits;

                    residual_from_b_ax_pcg_kernel<<<vector_blocks,
                                                   vector_threads>>>(
                        n, d_b, d_q, d_z);
                    check_cuda_pcg(cudaGetLastError(),
                                   "GPU PCG residual audit launch");

                    const float rr_audit = dot_pcg(handle, n, d_z, d_z);
                    result.audited_relative_residual =
                        std::sqrt(static_cast<double>(rr_audit) /
                                  static_cast<double>(bnorm2));

                    if (!std::isfinite(rr_audit) ||
                        !std::isfinite(result.audited_relative_residual)) {
                        throw std::runtime_error(
                            "GPU PCG audited residual became non-finite");
                    }
                    if (result.audited_relative_residual <=
                        relative_tolerance) {
                        result.converged = true;
                        break;
                    }
                }

                // Rebuild z from the untouched recursive residual. An audit
                // therefore observes the solver but never mutates its state.
                launch_pcg_jacobi(mesh, block_y, nodes, d_r, d_z);
                const float rho_new = dot_pcg(handle, n, d_r, d_z);
                if (!(rho_new > 0.0f) || !std::isfinite(rho_new)) {
                    throw std::runtime_error(
                        "GPU PCG breakdown: preconditioned residual became invalid");
                }
                const float beta = rho_new / rho;
                update_p_pcg_kernel<<<vector_blocks, vector_threads>>>(
                    n, beta, d_z, d_p);
                check_cuda_pcg(cudaGetLastError(),
                               "GPU PCG p update launch");
                rho = rho_new;
            }

            if (!result.converged) {
                // Always audit the final x so the returned audited residual is
                // meaningful even if the recursive criterion never succeeds.
                launch_pcg_matvec(mesh, block_y, nodes, d_x, d_q);
                ++result.matvecs;
                ++result.residual_audits;
                residual_from_b_ax_pcg_kernel<<<vector_blocks,
                                               vector_threads>>>(
                    n, d_b, d_q, d_z);
                check_cuda_pcg(cudaGetLastError(),
                               "GPU PCG final residual audit launch");
                const float rr_audit = dot_pcg(handle, n, d_z, d_z);
                result.audited_relative_residual =
                    std::sqrt(static_cast<double>(rr_audit) /
                              static_cast<double>(bnorm2));
                if (!std::isfinite(result.audited_relative_residual)) {
                    throw std::runtime_error(
                        "GPU PCG final audited residual became non-finite");
                }
            }
        }

        check_cuda_pcg(cudaDeviceSynchronize(),
                       "cudaDeviceSynchronize(PCG solve)");
        const auto wall_stop = Clock::now();
        result.solve_ms =
            std::chrono::duration<double, std::milli>(
                wall_stop - wall_start).count();

        std::vector<float> host_x(ndof);
        check_cuda_pcg(cudaMemcpy(host_x.data(), d_x, vector_bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(PCG x D2H)");
        result.x.resize(ndof);
        for (std::size_t node = 0; node < nodes; ++node) {
            result.x[3U * node + 0U] = host_x[node];
            result.x[3U * node + 1U] = host_x[nodes + node];
            result.x[3U * node + 2U] = host_x[2U * nodes + node];
        }

        cublasDestroy(handle);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_r);
        cudaFree(d_z);
        cudaFree(d_p);
        cudaFree(d_q);
        return result;
    } catch (...) {
        if (handle) cublasDestroy(handle);
        if (d_b) cudaFree(d_b);
        if (d_x) cudaFree(d_x);
        if (d_r) cudaFree(d_r);
        if (d_z) cudaFree(d_z);
        if (d_p) cudaFree(d_p);
        if (d_q) cudaFree(d_q);
        throw;
    }
}

}  // namespace gfss
