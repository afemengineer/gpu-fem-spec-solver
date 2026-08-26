#include "gfss/cpu_elasticity.hpp"
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

double relative_l2(const std::vector<float>& a,
                   const std::vector<float>& b) {
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double av = static_cast<double>(a[i]);
        const double d = av - static_cast<double>(b[i]);
        num += d * d;
        den += av * av;
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
    const gfss::StructuredHexMesh mesh{6U, 4U, 4U, 1.0, 0.8, 0.8};
    const gfss::Material material{73000.0, 0.29};
    const auto rhs = make_rhs(mesh);
    const std::vector<float> zero_seed(rhs.size(), 0.0f);
    constexpr double tolerance = 1.0e-4;

    gfss::GpuPcgResult baseline;
    gfss::GpuPcgResult seeded;
    try {
        baseline = gfss::solve_pcg_cuda_gold_sparse_x0(
            mesh, material, rhs, tolerance, 1500U, 4);
        seeded = gfss::solve_pcg_cuda_gold_sparse_seeded_x0(
            mesh, material, rhs, zero_seed, tolerance, 1500U, 4);
    } catch (const std::exception& e) {
        std::cerr << "FAILED: seeded PCG regression threw: " << e.what() << '\n';
        return 1;
    }

    require(baseline.converged, "baseline PCG must converge");
    require(seeded.converged, "zero-seeded PCG must converge");
    require(seeded.audited_relative_residual <= tolerance,
            "zero-seeded PCG must satisfy audited tolerance");
    require(seeded.explicit_device_bytes == 6U * rhs.size() * sizeof(float),
            "seeded PCG must retain six-vector device memory model");
    require(seeded.matvecs >= seeded.iterations + 1U,
            "seeded PCG matvec count must include initial A*x0 evaluation");

    const double seeded_true =
        true_relative_residual(mesh, material, rhs, seeded.x);
    const double solution_delta = relative_l2(baseline.x, seeded.x);
    require(seeded_true < 2.0e-4,
            "zero-seeded PCG true residual must remain close to audit");
    require(solution_delta < 1.0e-5,
            "zero-seeded PCG solution must match baseline PCG");

    std::cout << "Seeded GPU PCG checks passed; baseline_iterations="
              << baseline.iterations
              << " seeded_iterations=" << seeded.iterations
              << " seeded_true_rel=" << seeded_true
              << " solution_delta=" << solution_delta
              << " seeded_solve_ms=" << seeded.solve_ms << '\n';
    return 0;
}
