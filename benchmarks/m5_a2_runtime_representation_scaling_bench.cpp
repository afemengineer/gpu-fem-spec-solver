// M5 scaling decision probe: build the current exact fast hierarchy, then
// compare structural FP32 CSR against dense FP32 cuBLAS SGEMV for the actual A2
// produced at an arbitrary thin-plate refinement. Production remains dense.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_fast_hierarchy_bundle.hpp"
#include "gfss/gpu_m5_l2_materialized.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<double> deterministic_probe(std::size_t n) {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0U; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        x[i] = 1.0e-9 * (std::sin(0.013 * t + 0.37) +
                         0.31 * std::cos(0.041 * t - 0.19));
    }
    return x;
}

std::vector<float> to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0U; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

double relative_error(const std::vector<float>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) {
        throw std::invalid_argument("A2 runtime scaling oracle size mismatch");
    }
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0U; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    return std::sqrt(d2 / std::max(r2, 1.0e-300));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t nx = argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 64U;
        const std::size_t ny = argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 64U;
        const std::size_t nz = argc > 3 ? static_cast<std::size_t>(std::stoull(argv[3])) : 8U;
        const int repeats = argc > 4 ? std::stoi(argv[4]) : 100;
        const std::size_t target_nodes = argc > 5
            ? static_cast<std::size_t>(std::stoull(argv[5])) : 12U;
        const std::size_t min_nodes = argc > 6
            ? static_cast<std::size_t>(std::stoull(argv[6])) : 4U;
        if (nx < 2U || ny < 2U || nz < 1U || repeats <= 0 ||
            target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid A2 runtime scaling options");
        }

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto hierarchy = m5_fast_bundle::build(
            mesh, material, target_nodes, min_nodes, false, false);
        const auto& a2 = hierarchy.a2;
        if (a2.n == 0U || a2.n > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("A2 dimension unsupported for CSR probe");
        }

        std::vector<std::uint32_t> row_offsets(a2.n + 1U, 0U);
        std::vector<std::uint32_t> column_indices;
        std::vector<float> values;
        for (std::size_t row = 0U; row < a2.n; ++row) {
            row_offsets[row] = static_cast<std::uint32_t>(column_indices.size());
            const double* dense_row = a2.fp64.data() + row * a2.n;
            for (std::size_t col = 0U; col < a2.n; ++col) {
                const double v = dense_row[col];
                if (v == 0.0) continue;
                if (column_indices.size() >= std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error("A2 CSR nnz exceeds uint32 range");
                }
                column_indices.push_back(static_cast<std::uint32_t>(col));
                values.push_back(static_cast<float>(v));
            }
        }
        row_offsets[a2.n] = static_cast<std::uint32_t>(column_indices.size());

        const auto probe = deterministic_probe(a2.n);
        const auto reference = m5_l2_setup::apply_dense_a2(a2, probe);
        const auto probe_f = to_float(probe);
        const auto dense_f = m5_l2_setup::to_float(a2.fp64);

        const auto gpu_csr = gfss::benchmark_m5_l2_csr(
            row_offsets, column_indices, values, probe_f, repeats);
        const auto gpu_dense = gfss::benchmark_m5_l2_dense_symmetric(
            dense_f, a2.n, probe_f, repeats);

        const double csr_error = relative_error(gpu_csr.y, reference);
        const double dense_error = relative_error(gpu_dense.y, reference);
        const std::size_t csr_matrix_bytes = row_offsets.size() * sizeof(std::uint32_t) +
            column_indices.size() * sizeof(std::uint32_t) + values.size() * sizeof(float);
        const std::size_t dense_matrix_bytes = dense_f.size() * sizeof(float);
        const double density = static_cast<double>(values.size()) /
            static_cast<double>(a2.n * a2.n);
        const double csr_over_dense_memory = dense_matrix_bytes > 0U
            ? static_cast<double>(csr_matrix_bytes) / static_cast<double>(dense_matrix_bytes) : 0.0;
        const double dense_over_csr_speed = gpu_csr.timing.median_ms > 0.0
            ? gpu_dense.timing.median_ms / gpu_csr.timing.median_ms : 0.0;
        const bool oracle_accept = csr_error <= 1.0e-4 && dense_error <= 1.0e-4;
        const bool csr_faster = gpu_csr.timing.median_ms < gpu_dense.timing.median_ms;
        const bool csr_smaller = csr_matrix_bytes < dense_matrix_bytes;

        std::cout << "GFSS M5 A2 runtime representation scaling probe\n"
                  << "problem=thin_plate mesh=" << nx << 'x' << ny << 'x' << nz
                  << " physical=1x1x0.125\n"
                  << "production=dense_fp32_cublas_sgemv candidate=structural_scalar_CSR_fp32\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << hierarchy.space0.coarse_dofs
                  << " L2_dofs=" << a2.n
                  << " L2_nodes=" << hierarchy.transfer1.aggregates.size()
                  << " L3_dofs=" << hierarchy.bottom.factor.n << '\n'
                  << std::scientific << std::setprecision(12)
                  << "gpu_CSR_vs_FP64_dense_relative_error=" << csr_error
                  << " gpu_dense_vs_FP64_dense_relative_error=" << dense_error
                  << " oracle_accept_1e-4=" << (oracle_accept ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "A2_nnz=" << values.size()
                  << " A2_nnz_per_row=" << static_cast<double>(values.size()) / static_cast<double>(a2.n)
                  << " A2_scalar_density=" << density << '\n'
                  << "A2_CSR_matrix_bytes=" << csr_matrix_bytes
                  << " A2_dense_matrix_bytes=" << dense_matrix_bytes
                  << " CSR_over_dense_memory_ratio=" << csr_over_dense_memory << '\n'
                  << "CSR_median_ms=" << gpu_csr.timing.median_ms
                  << " CSR_best_ms=" << gpu_csr.timing.best_ms
                  << " dense_median_ms=" << gpu_dense.timing.median_ms
                  << " dense_best_ms=" << gpu_dense.timing.best_ms
                  << " dense_over_CSR_median_ratio=" << dense_over_csr_speed << '\n'
                  << "CSR_device_bytes_including_vectors=" << gpu_csr.device_bytes
                  << " dense_device_bytes_including_vectors=" << gpu_dense.device_bytes << '\n'
                  << "CSR_faster=" << (csr_faster ? "true" : "false")
                  << " CSR_smaller=" << (csr_smaller ? "true" : "false")
                  << " preferred_runtime_representation="
                  << ((csr_faster && csr_smaller) ? "structural_scalar_CSR_fp32" : "dense_fp32_cublas_sgemv")
                  << '\n'
                  << "hierarchy_setup_ms=" << hierarchy.production_setup_ms
                  << " representation_probe_excluded_from_setup_timing=true\n";
        return oracle_accept ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_a2_runtime_representation_scaling_bench "
                  << "[nx=64 [ny=64 [nz=8 [repeats=100 [target_nodes=12 [min_nodes=4]]]]]]\n";
        return 1;
    }
}
