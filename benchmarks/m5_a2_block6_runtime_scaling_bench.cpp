// M5 scaling decision probe: compare the current dense FP32 A2 runtime format
// against scalar structural CSR and the natural 6x6 block-CSR representation.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_fast_hierarchy_bundle.hpp"
#include "gfss/gpu_m5_l2_materialized.hpp"
#include "gfss/gpu_m5_l2_block6.hpp"

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

struct ScalarCsr {
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> columns;
    std::vector<float> values;
};

struct Block6Csr {
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> columns;
    std::vector<float> values;
};

std::vector<float> deterministic_probe(std::size_t n) {
    std::vector<float> x(n, 0.0f);
    for (std::size_t i = 0U; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        x[i] = static_cast<float>(1.0e-9 *
            (std::sin(0.017 * t + 0.37) + 0.31 * std::cos(0.041 * t - 0.19)));
    }
    return x;
}

std::vector<double> to_double(const std::vector<float>& x) {
    std::vector<double> out(x.size(), 0.0);
    for (std::size_t i = 0U; i < x.size(); ++i) out[i] = static_cast<double>(x[i]);
    return out;
}

double relative_error(const std::vector<float>& got,
                      const std::vector<double>& reference) {
    if (got.size() != reference.size()) throw std::invalid_argument("A2 block6 oracle size mismatch");
    double d2 = 0.0;
    double r2 = 0.0;
    for (std::size_t i = 0U; i < got.size(); ++i) {
        const double d = static_cast<double>(got[i]) - reference[i];
        d2 += d * d;
        r2 += reference[i] * reference[i];
    }
    return std::sqrt(d2 / std::max(r2, 1.0e-300));
}

ScalarCsr compress_scalar(const m5_l2_setup::DenseA2& a2) {
    if (a2.n > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("scalar CSR A2 dimension exceeds uint32");
    }
    ScalarCsr out;
    out.row_offsets.resize(a2.n + 1U, 0U);
    for (std::size_t row = 0U; row < a2.n; ++row) {
        out.row_offsets[row] = static_cast<std::uint32_t>(out.columns.size());
        for (std::size_t col = 0U; col < a2.n; ++col) {
            const double value = a2.fp64[row * a2.n + col];
            if (value == 0.0) continue;
            if (out.columns.size() >= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw std::runtime_error("scalar CSR A2 nnz exceeds uint32");
            }
            out.columns.push_back(static_cast<std::uint32_t>(col));
            out.values.push_back(static_cast<float>(value));
        }
    }
    out.row_offsets[a2.n] = static_cast<std::uint32_t>(out.columns.size());
    return out;
}

Block6Csr compress_block6(const m5_l2_setup::DenseA2& a2, std::size_t block_rows) {
    if (a2.n != block_rows * 6U ||
        block_rows > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::runtime_error("A2 is not a full-rank 6x6 block layout");
    }
    Block6Csr out;
    out.row_offsets.resize(block_rows + 1U, 0U);
    for (std::size_t br = 0U; br < block_rows; ++br) {
        out.row_offsets[br] = static_cast<std::uint32_t>(out.columns.size());
        for (std::size_t bc = 0U; bc < block_rows; ++bc) {
            bool nonzero = false;
            for (std::size_t r = 0U; r < 6U && !nonzero; ++r) {
                for (std::size_t c = 0U; c < 6U; ++c) {
                    if (a2.fp64[(br * 6U + r) * a2.n + (bc * 6U + c)] != 0.0) {
                        nonzero = true;
                        break;
                    }
                }
            }
            if (!nonzero) continue;
            if (out.columns.size() >= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
                throw std::runtime_error("block6 CSR A2 nnz exceeds uint32");
            }
            out.columns.push_back(static_cast<std::uint32_t>(bc));
            for (std::size_t r = 0U; r < 6U; ++r) {
                for (std::size_t c = 0U; c < 6U; ++c) {
                    out.values.push_back(static_cast<float>(
                        a2.fp64[(br * 6U + r) * a2.n + (bc * 6U + c)]));
                }
            }
        }
    }
    out.row_offsets[block_rows] = static_cast<std::uint32_t>(out.columns.size());
    return out;
}

std::size_t scalar_csr_bytes(const ScalarCsr& a) {
    return a.row_offsets.size() * sizeof(std::uint32_t) +
           a.columns.size() * sizeof(std::uint32_t) +
           a.values.size() * sizeof(float);
}

std::size_t block6_csr_bytes(const Block6Csr& a) {
    return a.row_offsets.size() * sizeof(std::uint32_t) +
           a.columns.size() * sizeof(std::uint32_t) +
           a.values.size() * sizeof(float);
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
        if (nx < 2U || ny < 2U || nz < 1U || repeats <= 0 || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid block6 A2 scaling options");
        }

        const gfss::StructuredHexMesh mesh{nx, ny, nz, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto hierarchy = m5_fast_bundle::build(
            mesh, material, target_nodes, min_nodes, false);
        const std::size_t block_rows = hierarchy.transfer1.aggregates.size();
        if (hierarchy.a2.n != block_rows * 6U) {
            throw std::runtime_error("current L2 hierarchy is not uniformly rank six");
        }

        const auto scalar = compress_scalar(hierarchy.a2);
        const auto block6 = compress_block6(hierarchy.a2, block_rows);
        if (scalar.values.size() != block6.columns.size() * 36U) {
            throw std::runtime_error("A2 scalar/block structural nnz mismatch");
        }

        const auto x = deterministic_probe(hierarchy.a2.n);
        const auto reference = m5_l2_setup::apply_dense_a2(hierarchy.a2, to_double(x));
        const auto dense = gfss::benchmark_m5_l2_dense_symmetric(
            hierarchy.a2_fp32, hierarchy.a2.n, x, repeats);
        const auto csr = gfss::benchmark_m5_l2_csr(
            scalar.row_offsets, scalar.columns, scalar.values, x, repeats);
        const auto bsr = gfss::benchmark_m5_l2_block6_csr(
            block6.row_offsets, block6.columns, block6.values, x, repeats);

        const double dense_error = relative_error(dense.y, reference);
        const double csr_error = relative_error(csr.y, reference);
        const double bsr_error = relative_error(bsr.y, reference);
        const bool oracle_accept = dense_error <= 1.0e-4 && csr_error <= 1.0e-4 && bsr_error <= 1.0e-4;

        const std::size_t dense_bytes = hierarchy.a2.n * hierarchy.a2.n * sizeof(float);
        const std::size_t scalar_bytes = scalar_csr_bytes(scalar);
        const std::size_t block_bytes = block6_csr_bytes(block6);
        const double scalar_density = static_cast<double>(scalar.values.size()) /
            static_cast<double>(hierarchy.a2.n * hierarchy.a2.n);
        const double block_density = static_cast<double>(block6.columns.size()) /
            static_cast<double>(block_rows * block_rows);

        const double dense_over_csr = dense.timing.median_ms > 0.0
            ? csr.timing.median_ms / dense.timing.median_ms : 0.0;
        const double dense_over_bsr = bsr.timing.median_ms > 0.0
            ? dense.timing.median_ms / bsr.timing.median_ms : 0.0;
        const double csr_over_bsr = bsr.timing.median_ms > 0.0
            ? csr.timing.median_ms / bsr.timing.median_ms : 0.0;

        const char* preferred = "dense_fp32_cublas_sgemv";
        double preferred_ms = dense.timing.median_ms;
        if (csr.timing.median_ms < preferred_ms) {
            preferred = "structural_scalar_CSR_fp32";
            preferred_ms = csr.timing.median_ms;
        }
        if (bsr.timing.median_ms < preferred_ms) {
            preferred = "structural_block6_CSR_fp32";
            preferred_ms = bsr.timing.median_ms;
        }

        std::cout << "GFSS M5 A2 dense-vs-CSR-vs-block6 scaling probe\n"
                  << "problem=thin_plate mesh=" << nx << 'x' << ny << 'x' << nz
                  << " physical=1x1x0.125\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << hierarchy.space0.coarse_dofs
                  << " L2_dofs=" << hierarchy.a2.n
                  << " L2_nodes=" << block_rows
                  << " L3_dofs=" << hierarchy.bottom.factor.n << '\n'
                  << std::scientific << std::setprecision(12)
                  << "gpu_dense_vs_FP64_relative_error=" << dense_error
                  << " gpu_scalar_CSR_vs_FP64_relative_error=" << csr_error
                  << " gpu_block6_CSR_vs_FP64_relative_error=" << bsr_error
                  << " oracle_accept_1e-4=" << (oracle_accept ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "A2_scalar_nnz=" << scalar.values.size()
                  << " A2_scalar_density=" << scalar_density
                  << " A2_block_nnz=" << block6.columns.size()
                  << " A2_block_density=" << block_density << '\n'
                  << "dense_matrix_bytes=" << dense_bytes
                  << " scalar_CSR_matrix_bytes=" << scalar_bytes
                  << " block6_CSR_matrix_bytes=" << block_bytes
                  << " scalar_CSR_over_dense_memory_ratio="
                  << static_cast<double>(scalar_bytes) / static_cast<double>(dense_bytes)
                  << " block6_CSR_over_dense_memory_ratio="
                  << static_cast<double>(block_bytes) / static_cast<double>(dense_bytes)
                  << " block6_CSR_over_scalar_CSR_memory_ratio="
                  << static_cast<double>(block_bytes) / static_cast<double>(scalar_bytes) << '\n'
                  << "dense_median_ms=" << dense.timing.median_ms
                  << " dense_best_ms=" << dense.timing.best_ms
                  << " scalar_CSR_median_ms=" << csr.timing.median_ms
                  << " scalar_CSR_best_ms=" << csr.timing.best_ms
                  << " block6_CSR_median_ms=" << bsr.timing.median_ms
                  << " block6_CSR_best_ms=" << bsr.timing.best_ms << '\n'
                  << "scalar_CSR_over_dense_time_ratio=" << dense_over_csr
                  << " dense_over_block6_CSR_speedup=" << dense_over_bsr
                  << " scalar_CSR_over_block6_CSR_speedup=" << csr_over_bsr << '\n'
                  << "dense_device_bytes_including_vectors=" << dense.device_bytes
                  << " scalar_CSR_device_bytes_including_vectors=" << csr.device_bytes
                  << " block6_CSR_device_bytes_including_vectors=" << bsr.device_bytes << '\n'
                  << "preferred_runtime_representation=" << preferred << '\n'
                  << "hierarchy_setup_ms=" << hierarchy.production_setup_ms
                  << " representation_probe_excluded_from_setup_timing=true\n";
        return oracle_accept ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_a2_block6_runtime_scaling_bench "
                  << "[nx=64 [ny=64 [nz=8 [repeats=100 [target_nodes=12 [min_nodes=4]]]]]]\n";
        return 1;
    }
}
