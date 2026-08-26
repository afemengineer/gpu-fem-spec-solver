#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_solver.hpp"
#include "gfss/gpu_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

std::vector<float> make_rhs(const gfss::StructuredHexMesh& mesh) {
    std::vector<float> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0f);
    const float load = -1.0f /
        static_cast<float>((mesh.ny + 1U) * (mesh.nz + 1U));
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = load;
        }
    }
    return rhs;
}

double relative_l2(const std::vector<double>& reference,
                   const std::vector<float>& candidate) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const double d = reference[i] - static_cast<double>(candidate[i]);
        num += d * d;
        den += reference[i] * reference[i];
    }
    return std::sqrt(num / std::max(den, 1.0e-300));
}

double true_relative_residual(const gfss::StructuredHexMesh& mesh,
                              const gfss::Material& material,
                              const std::vector<float>& rhs,
                              const std::vector<float>& x) {
    std::vector<double> xd(x.begin(), x.end());
    const auto ax = gfss::apply_clamped_x0_matrix_free(mesh, material, xd);
    double rr = 0.0;
    double bb = 0.0;
    for (std::size_t i = 0; i < rhs.size(); ++i) {
        const double b = static_cast<double>(rhs[i]);
        const double r = b - ax[i];
        rr += r * r;
        bb += b * b;
    }
    return std::sqrt(rr / bb);
}

}  // namespace

int main() {
    const gfss::StructuredHexMesh mesh{6, 4, 3, 1.0, 0.8, 0.6};
    const gfss::Material material{73000.0, 0.29};
    const auto rhs = make_rhs(mesh);

    // This is a correctness smoke test, not the FP32 precision-limit experiment.
    // Keep it in a regime where single-precision PCG is expected to converge
    // reliably; the benchmark exercises the harder 1e-5 target separately.
    constexpr double gpu_tolerance = 1.0e-4;

    gfss::GpuPcgResult gpu;
    try {
        gpu = gfss::solve_pcg_cuda_gold_sparse_x0(
            mesh, material, rhs, gpu_tolerance, 1500U, 4);
    } catch (const std::exception& e) {
        std::cerr << "FAILED: GPU PCG threw during smoke test: " << e.what() << '\n';
        return 1;
    }

    std::vector<double> rhs_double(rhs.begin(), rhs.end());
    const auto cpu = gfss::conjugate_gradient(
        [&](const std::vector<double>& x) {
            return gfss::apply_clamped_x0_matrix_free(mesh, material, x);
        },
        rhs_double,
        1.0e-10,
        5000U);

    require(cpu.converged, "trusted CPU CG must converge");
    require(gpu.converged, "GPU Jacobi-PCG must converge in smoke-test regime");
    require(gpu.iterations > 0U, "GPU PCG must perform iterations");
    require(gpu.residual_audits > 0U,
            "GPU PCG must verify convergence with at least one residual audit");
    require(gpu.matvecs == gpu.iterations + gpu.residual_audits,
            "GPU PCG matvec accounting must include residual audits");
    require(gpu.solve_ms > 0.0, "GPU PCG solve timing must be positive");
    require(gpu.explicit_device_bytes == 6U * rhs.size() * sizeof(float),
            "GPU PCG explicit vector memory accounting mismatch");

    const double true_rel = true_relative_residual(mesh, material, rhs, gpu.x);
    const double solution_rel = relative_l2(cpu.x, gpu.x);

    require(gpu.reported_relative_residual <= gpu_tolerance,
            "GPU PCG recursive residual must meet smoke-test tolerance");
    require(gpu.audited_relative_residual <= gpu_tolerance,
            "GPU PCG audited residual must meet smoke-test tolerance");
    require(true_rel < 2.0e-4,
            "GPU PCG true residual must remain close to audited FP32 residual");
    require(solution_rel < 1.0e-3,
            "GPU PCG solution must match trusted CPU CG solution");

    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(0U, j, k);
            for (int c = 0; c < 3; ++c) {
                require(gpu.x[static_cast<std::size_t>(3ULL * node + c)] == 0.0f,
                        "GPU PCG constrained x=0 DOFs must remain exactly zero");
            }
        }
    }

    std::cout << "GPU PCG checks passed; iterations=" << gpu.iterations
              << " audits=" << gpu.residual_audits
              << " recursive_rel=" << gpu.reported_relative_residual
              << " audited_rel=" << gpu.audited_relative_residual
              << " true_rel=" << true_rel
              << " solution_rel_l2=" << solution_rel
              << " solve_ms=" << gpu.solve_ms << '\n';
    return 0;
}
