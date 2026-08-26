#include "gfss/cpu_elasticity.hpp"
#include "gfss/gpu_solver.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t parse_u32(const char* text) {
    const auto value = std::stoul(text);
    if (value == 0 || value > 1000000UL) {
        throw std::invalid_argument("mesh dimensions must be in [1, 1000000]");
    }
    return static_cast<std::uint32_t>(value);
}

std::size_t parse_size(const char* text) {
    const auto value = std::stoull(text);
    if (value == 0ULL) {
        throw std::invalid_argument("max iterations must be positive");
    }
    return static_cast<std::size_t>(value);
}

double parse_positive_double(const char* text) {
    const double value = std::stod(text);
    if (!(value > 0.0)) {
        throw std::invalid_argument("tolerance must be positive");
    }
    return value;
}

double true_relative_residual(const gfss::StructuredHexMesh& mesh,
                              const gfss::Material& material,
                              const std::vector<float>& rhs,
                              const std::vector<float>& x) {
    std::vector<double> xd(x.begin(), x.end());
    std::vector<double> bd(rhs.begin(), rhs.end());
    const auto ax = gfss::apply_clamped_x0_matrix_free(mesh, material, xd);

    double rr = 0.0;
    double bb = 0.0;
    for (std::size_t i = 0; i < bd.size(); ++i) {
        const double ri = bd[i] - ax[i];
        rr += ri * ri;
        bb += bd[i] * bd[i];
    }
    return bb > 0.0 ? std::sqrt(rr / bb) : 0.0;
}

std::vector<float> make_cantilever_rhs(const gfss::StructuredHexMesh& mesh) {
    std::vector<float> rhs(static_cast<std::size_t>(mesh.dof_count()), 0.0f);
    const double face_nodes = static_cast<double>(mesh.ny + 1U) *
                              static_cast<double>(mesh.nz + 1U);
    const float nodal_load = static_cast<float>(-1.0 / face_nodes);

    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = mesh.node_index(mesh.nx, j, k);
            rhs[static_cast<std::size_t>(3ULL * node + 2ULL)] = nodal_load;
        }
    }
    return rhs;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::uint32_t nx = argc > 1 ? parse_u32(argv[1]) : 32U;
        const std::uint32_t ny = argc > 2 ? parse_u32(argv[2]) : nx;
        const std::uint32_t nz = argc > 3 ? parse_u32(argv[3]) : nx;
        const double tolerance = argc > 4 ? parse_positive_double(argv[4]) : 1.0e-5;
        const std::size_t max_iterations = argc > 5 ? parse_size(argv[5]) : 5000U;

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 1.0};
        const gfss::Material material{210.0e9, 0.30};
        const auto rhs = make_cantilever_rhs(mesh);

        const auto result = gfss::solve_pcg_cuda_gold_sparse_x0(
            mesh, material, rhs, tolerance, max_iterations, 4);

        const double true_rel = true_relative_residual(mesh, material, rhs, result.x);
        const double mib = static_cast<double>(result.explicit_device_bytes) /
                           (1024.0 * 1024.0);
        const double bytes_per_dof =
            static_cast<double>(result.explicit_device_bytes) /
            static_cast<double>(mesh.dof_count());
        const double ms_per_iteration = result.iterations > 0U
            ? result.solve_ms / static_cast<double>(result.iterations)
            : 0.0;

        std::cout << std::fixed << std::setprecision(6)
                  << "GFSS CUDA GoldSparse Jacobi-PCG benchmark\n"
                  << "mesh=" << nx << 'x' << ny << 'x' << nz
                  << " elements=" << mesh.element_count()
                  << " dofs=" << mesh.dof_count() << '\n'
                  << "constraint=x0_zero_dirichlet\n"
                  << "rhs=uniform_z_load_on_xmax_face total_load=-1\n"
                  << "tolerance=" << std::scientific << tolerance
                  << " max_iterations=" << std::fixed << max_iterations << '\n'
                  << "converged=" << (result.converged ? "true" : "false")
                  << " iterations=" << result.iterations
                  << " matvecs=" << result.matvecs << '\n'
                  << "solve_ms=" << result.solve_ms
                  << " ms_per_iteration=" << ms_per_iteration << '\n'
                  << std::scientific
                  << "reported_relative_residual=" << result.reported_relative_residual << '\n'
                  << "true_relative_residual=" << true_rel << '\n'
                  << std::fixed
                  << "explicit_device_vectors=" << mib << " MiB\n"
                  << "explicit_bytes_per_dof=" << bytes_per_dof << '\n'
                  << "timing_excludes=allocation,stencil_setup,H2D,D2H,true_residual_audit\n";

        if (!result.converged) {
            std::cerr << "WARNING: PCG did not reach the requested recursive-residual tolerance\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_pcg_bench [nx [ny nz [tolerance [max_iterations]]]]\n";
        return 1;
    }
}
