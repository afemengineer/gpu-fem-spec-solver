// M5 GPU productionization stage 4b: the scalar-CSR P1 representation is fast
// but duplicates ~31.5 MB of values/indices for P1 and P1^T. Rebuild the same
// exact frozen smoothed transfer in its natural padded 6x6 algebraic-block form
// and store the 6x6 value payload once. P1 uses forward block-row indices; P1^T
// uses a separate column-oriented index that references the original block ids.
// This keeps both directions atomic-free while sharing all FP32 block values.
#define main gfss_m5_scalar_transfer_reference_main
#include "gpu_m5_l1_l2_transfer_representation_bench.cpp"
#undef main

#include <array>
#include <functional>

namespace {

struct Block6Transfer {
    std::size_t block_rows{0U};
    std::size_t block_cols{0U};
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
    std::vector<double> values_fp64;
    std::vector<float> values_fp32;
    std::vector<std::uint32_t> transpose_offsets;
    std::vector<std::uint32_t> transpose_rows;
    std::vector<std::uint32_t> transpose_block_ids;
    double assembly_ms{0.0};

    std::size_t block_nnz() const noexcept { return column_indices.size(); }
    std::size_t value_bytes_fp32() const noexcept {
        return values_fp32.size() * sizeof(float);
    }
    std::size_t forward_index_bytes() const noexcept {
        return (row_offsets.size() + column_indices.size()) * sizeof(std::uint32_t);
    }
    std::size_t transpose_index_bytes() const noexcept {
        return (transpose_offsets.size() + transpose_rows.size() +
                transpose_block_ids.size()) * sizeof(std::uint32_t);
    }
    std::size_t shared_matrix_bytes_fp32() const noexcept {
        return value_bytes_fp32() + forward_index_bytes() + transpose_index_bytes();
    }
};

Block6Transfer assemble_block6_transfer(
    const CandidateTransfer& transfer1_tentative,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis) {
    const auto start = BenchClock::now();
    Block6Transfer out;
    out.block_rows = block1.nodes();
    out.block_cols = l2_basis.size();
    if (out.block_rows == 0U || out.block_cols == 0U ||
        transfer1_tentative.coarse_graph.nodes() != out.block_cols ||
        transfer1_tentative.coarse_dofs == 0U) {
        throw std::invalid_argument("M5 block6 P1 layout mismatch");
    }

    using DenseBlock = std::array<double, 36U>;
    std::vector<std::vector<std::pair<std::uint32_t, DenseBlock>>> rows(out.block_rows);

    for (std::size_t l2_node = 0; l2_node < l2_basis.size(); ++l2_node) {
        const auto& basis = l2_basis[l2_node];
        const std::size_t rank2 = transfer1_tentative.aggregates[l2_node].rank;
        if (basis.cols != rank2 || rank2 == 0U || rank2 > 6U) {
            throw std::runtime_error("M5 block6 P1 L2 rank mismatch");
        }
        for (const auto& entry : basis.values) {
            const std::size_t l1_node = entry.first;
            if (l1_node >= out.block_rows) {
                throw std::out_of_range("M5 block6 P1 L1 node out of range");
            }
            const std::size_t rank1 = block1.dof_offsets[l1_node + 1U] -
                                      block1.dof_offsets[l1_node];
            if (rank1 == 0U || rank1 > 6U) {
                throw std::runtime_error("M5 block6 P1 L1 rank mismatch");
            }
            DenseBlock block{};
            bool nonzero = false;
            for (std::size_t r = 0; r < rank1; ++r) {
                for (std::size_t q = 0; q < rank2; ++q) {
                    const double v = entry.second[r * 6U + q];
                    block[r * 6U + q] = v;
                    nonzero = nonzero || (v != 0.0);
                }
            }
            if (nonzero) {
                rows[l1_node].emplace_back(static_cast<std::uint32_t>(l2_node), block);
            }
        }
    }

    out.row_offsets.resize(out.block_rows + 1U, 0U);
    std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>>
        transpose_entries(out.block_cols);
    for (std::size_t row = 0; row < out.block_rows; ++row) {
        auto& entries = rows[row];
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        out.row_offsets[row] = static_cast<std::uint32_t>(out.column_indices.size());
        std::uint32_t previous = std::numeric_limits<std::uint32_t>::max();
        for (const auto& item : entries) {
            if (item.first == previous) {
                throw std::runtime_error("M5 block6 P1 duplicate structural block");
            }
            previous = item.first;
            if (item.first >= out.block_cols ||
                out.column_indices.size() >= std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("M5 block6 P1 indexing overflow");
            }
            const auto block_id = static_cast<std::uint32_t>(out.column_indices.size());
            out.column_indices.push_back(item.first);
            for (const double v : item.second) {
                out.values_fp64.push_back(v);
                out.values_fp32.push_back(static_cast<float>(v));
            }
            transpose_entries[item.first].emplace_back(
                static_cast<std::uint32_t>(row), block_id);
        }
    }
    out.row_offsets[out.block_rows] = static_cast<std::uint32_t>(out.column_indices.size());

    out.transpose_offsets.resize(out.block_cols + 1U, 0U);
    for (std::size_t col = 0; col < out.block_cols; ++col) {
        out.transpose_offsets[col] = static_cast<std::uint32_t>(out.transpose_rows.size());
        auto& entries = transpose_entries[col];
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (const auto& item : entries) {
            out.transpose_rows.push_back(item.first);
            out.transpose_block_ids.push_back(item.second);
        }
    }
    out.transpose_offsets[out.block_cols] =
        static_cast<std::uint32_t>(out.transpose_rows.size());

    if (out.values_fp64.size() != out.block_nnz() * 36U ||
        out.transpose_rows.size() != out.block_nnz() ||
        out.transpose_block_ids.size() != out.block_nnz()) {
        throw std::runtime_error("M5 block6 P1 final payload mismatch");
    }
    out.assembly_ms = std::chrono::duration<double, std::milli>(
        BenchClock::now() - start).count();
    return out;
}

std::vector<double> pad_l1(const L1BlockMetric& block1, const std::vector<double>& packed) {
    if (packed.size() != block1.dofs()) throw std::invalid_argument("M5 block6 L1 pad size mismatch");
    std::vector<double> padded(block1.nodes() * 6U, 0.0);
    for (std::size_t node = 0; node < block1.nodes(); ++node) {
        const std::size_t begin = block1.dof_offsets[node];
        const std::size_t rank = block1.dof_offsets[node + 1U] - begin;
        for (std::size_t r = 0; r < rank; ++r) padded[node * 6U + r] = packed[begin + r];
    }
    return padded;
}

std::vector<double> unpad_l1(const L1BlockMetric& block1, const std::vector<double>& padded) {
    if (padded.size() != block1.nodes() * 6U) throw std::invalid_argument("M5 block6 L1 unpad size mismatch");
    std::vector<double> packed(block1.dofs(), 0.0);
    for (std::size_t node = 0; node < block1.nodes(); ++node) {
        const std::size_t begin = block1.dof_offsets[node];
        const std::size_t rank = block1.dof_offsets[node + 1U] - begin;
        for (std::size_t r = 0; r < rank; ++r) packed[begin + r] = padded[node * 6U + r];
    }
    return packed;
}

std::vector<double> pad_l2(const CandidateTransfer& transfer, const std::vector<double>& packed) {
    if (packed.size() != transfer.coarse_dofs) throw std::invalid_argument("M5 block6 L2 pad size mismatch");
    std::vector<double> padded(transfer.aggregates.size() * 6U, 0.0);
    for (std::size_t node = 0; node < transfer.aggregates.size(); ++node) {
        const std::size_t begin = transfer.aggregates[node].coarse_offset;
        const std::size_t rank = transfer.aggregates[node].rank;
        for (std::size_t q = 0; q < rank; ++q) padded[node * 6U + q] = packed[begin + q];
    }
    return padded;
}

std::vector<double> unpad_l2(const CandidateTransfer& transfer, const std::vector<double>& padded) {
    if (padded.size() != transfer.aggregates.size() * 6U) {
        throw std::invalid_argument("M5 block6 L2 unpad size mismatch");
    }
    std::vector<double> packed(transfer.coarse_dofs, 0.0);
    for (std::size_t node = 0; node < transfer.aggregates.size(); ++node) {
        const std::size_t begin = transfer.aggregates[node].coarse_offset;
        const std::size_t rank = transfer.aggregates[node].rank;
        for (std::size_t q = 0; q < rank; ++q) packed[begin + q] = padded[node * 6U + q];
    }
    return packed;
}

std::vector<float> to_float_vec(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

std::vector<double> to_double_vec(const std::vector<float>& x) {
    std::vector<double> y(x.size(), 0.0);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<double>(x[i]);
    return y;
}

std::vector<double> block6_forward_cpu(const Block6Transfer& p,
                                       const std::vector<double>& x_padded) {
    if (x_padded.size() != p.block_cols * 6U) throw std::invalid_argument("M5 block6 CPU forward size mismatch");
    std::vector<double> y(p.block_rows * 6U, 0.0);
    for (std::size_t row = 0; row < p.block_rows; ++row) {
        for (std::size_t bid = p.row_offsets[row]; bid < p.row_offsets[row + 1U]; ++bid) {
            const std::size_t col = p.column_indices[bid];
            const double* block = p.values_fp64.data() + bid * 36U;
            for (std::size_t r = 0; r < 6U; ++r) {
                for (std::size_t q = 0; q < 6U; ++q) {
                    y[row * 6U + r] += block[r * 6U + q] * x_padded[col * 6U + q];
                }
            }
        }
    }
    return y;
}

std::vector<double> block6_transpose_cpu(const Block6Transfer& p,
                                         const std::vector<double>& x_padded) {
    if (x_padded.size() != p.block_rows * 6U) throw std::invalid_argument("M5 block6 CPU transpose size mismatch");
    std::vector<double> y(p.block_cols * 6U, 0.0);
    for (std::size_t col = 0; col < p.block_cols; ++col) {
        for (std::size_t k = p.transpose_offsets[col]; k < p.transpose_offsets[col + 1U]; ++k) {
            const std::size_t row = p.transpose_rows[k];
            const std::size_t bid = p.transpose_block_ids[k];
            const double* block = p.values_fp64.data() + bid * 36U;
            for (std::size_t q = 0; q < 6U; ++q) {
                for (std::size_t r = 0; r < 6U; ++r) {
                    y[col * 6U + q] += block[r * 6U + q] * x_padded[row * 6U + r];
                }
            }
        }
    }
    return y;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 100;
        const int block_y = argc > 2 ? std::stoi(argv[2]) : 4;
        const std::size_t target_nodes = argc > 3
            ? static_cast<std::size_t>(std::stoull(argv[3])) : 12U;
        const std::size_t min_nodes = argc > 4
            ? static_cast<std::size_t>(std::stoull(argv[4])) : 4U;
        if (repeats <= 0 || block_y <= 0 || target_nodes < 2U ||
            min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid M5 block6 transfer options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        const auto setup_start = BenchClock::now();

        auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
        const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
            mesh, material, space0);
        const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
        const Apply apply0 = [&](const Vec& x) { return apply_fine_clamped(mesh, material, x); };
        const double lambda0 = estimate_lambda_max(apply0, fine_inverse, 8U);
        const double omega0 = kSaDampingNumerator / lambda0;
        const FineSmoothedTransfer transfer0{
            mesh, material, space0, fine_inverse, omega0, m0};
        const Apply apply1 = [&](const Vec& x) {
            return transfer0.restrict_transpose(apply0(transfer0.prolong(x)));
        };

        const auto graph1_tentative = graph_from_variable_blocks(tentative_a1);
        const auto candidates1 = make_level1_candidates(space0);
        const auto block1 = build_exact_l1_block_metric(
            mesh, material, space0, graph1_tentative, fine_inverse, omega0);
        const double block1_oracle_error = audit_l1_block_metric(block1, apply1);
        const double block_lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
        const double block_omega1 = kSaDampingNumerator / block_lambda1;

        double p0_support_ms = 0.0;
        const auto fine_supports = build_fine_basis_support_cache(
            mesh, material, space0, fine_inverse, omega0, p0_support_ms);
        const auto element_supports = build_element_support_index(mesh, fine_supports);
        const auto actual_a1_offdiagonal = accumulate_combined_actual_a1_offdiagonal_blocks(
            mesh, material, fine_supports, element_supports);
        const auto strength1 = build_combined_strength_graph(
            graph1_tentative, block1, actual_a1_offdiagonal, strength_threshold);
        const auto transfer1_tentative = build_candidate_transfer(
            strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
        const L1BlockSmoothedTransfer transfer1{
            transfer1_tentative, apply1, block1, block_omega1, m1};

        const auto local_a1_apply = [&](const LocalColumns& x) {
            return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
        };
        const auto basis_start = BenchClock::now();
        const auto l2_basis = build_smoothed_candidate_supports(
            transfer1_tentative, strength1.graph, block1,
            block_omega1, m1, local_a1_apply);
        const double basis_setup_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - basis_start).count();

        const auto scalar_start = BenchClock::now();
        const auto scalar_p1 = assemble_explicit_p1(transfer1_tentative, block1, l2_basis);
        const double scalar_assembly_ms = std::chrono::duration<double, std::milli>(
            BenchClock::now() - scalar_start).count();
        const auto block6 = assemble_block6_transfer(transfer1_tentative, block1, l2_basis);
        const auto setup_stop = BenchClock::now();

        const auto coarse_probe = deterministic_probe(
            transfer1_tentative.coarse_dofs, 1.0e-9, 0.27);
        const auto fine_probe = deterministic_probe(space0.coarse_dofs, 1.0e-9, 0.71);
        const auto factorized_forward = transfer1.prolong(coarse_probe);
        const auto factorized_transpose = transfer1.restrict_transpose(fine_probe);

        const auto coarse_padded = pad_l2(transfer1_tentative, coarse_probe);
        const auto fine_padded = pad_l1(block1, fine_probe);
        const auto cpu_block_forward = unpad_l1(
            block1, block6_forward_cpu(block6, coarse_padded));
        const auto cpu_block_transpose = unpad_l2(
            transfer1_tentative, block6_transpose_cpu(block6, fine_padded));
        const double cpu_block_forward_error = relative_error(cpu_block_forward, factorized_forward);
        const double cpu_block_transpose_error = relative_error(cpu_block_transpose, factorized_transpose);

        const auto gpu_block = gfss::benchmark_m5_block6_transfer(
            block6.block_rows,
            block6.block_cols,
            block6.row_offsets,
            block6.column_indices,
            block6.values_fp32,
            block6.transpose_offsets,
            block6.transpose_rows,
            block6.transpose_block_ids,
            to_float_vec(coarse_padded),
            to_float_vec(fine_padded),
            repeats);
        const auto gpu_block_forward_packed = unpad_l1(
            block1, to_double_vec(gpu_block.forward_y_padded));
        const auto gpu_block_transpose_packed = unpad_l2(
            transfer1_tentative, to_double_vec(gpu_block.transpose_y_padded));
        const double gpu_block_forward_error = relative_error(
            gpu_block_forward_packed, factorized_forward);
        const double gpu_block_transpose_error = relative_error(
            gpu_block_transpose_packed, factorized_transpose);

        // Re-measure scalar CSR in the same process/GPU for a direct runtime
        // comparison against the already-validated representation.
        const auto gpu_scalar_forward = gfss::benchmark_m5_rectangular_csr(
            scalar_p1.forward.rows, scalar_p1.forward.cols,
            scalar_p1.forward.row_offsets, scalar_p1.forward.column_indices,
            scalar_p1.forward.values_fp32, to_float(coarse_probe), repeats);
        const auto gpu_scalar_transpose = gfss::benchmark_m5_rectangular_csr(
            scalar_p1.transpose.rows, scalar_p1.transpose.cols,
            scalar_p1.transpose.row_offsets, scalar_p1.transpose.column_indices,
            scalar_p1.transpose.values_fp32, to_float(fine_probe), repeats);

        const auto a1_probe = deterministic_probe(space0.coarse_dofs, 1.0e-9, 0.43);
        gfss::GpuSmoothedAggregationContext a1_gpu(
            mesh, material, space0, omega0, block_y);
        const auto a1_timing = a1_gpu.apply(to_float(a1_probe), m0, repeats);
        const double factorized_per_direction_lower_bound_ms =
            static_cast<double>(m1) * a1_timing.median_timing.total_ms;
        const double factorized_roundtrip_lower_bound_ms =
            2.0 * factorized_per_direction_lower_bound_ms;

        const double block_roundtrip_ms =
            gpu_block.forward_timing.median_ms + gpu_block.transpose_timing.median_ms;
        const double scalar_roundtrip_ms =
            gpu_scalar_forward.timing.median_ms + gpu_scalar_transpose.timing.median_ms;
        const std::size_t scalar_dual_bytes =
            scalar_p1.forward.matrix_bytes_fp32() + scalar_p1.transpose.matrix_bytes_fp32();
        const std::size_t scalar_single_bytes = scalar_p1.forward.matrix_bytes_fp32();
        const std::size_t l1_padded_dofs = block1.nodes() * 6U;
        const std::size_t l2_padded_dofs = transfer1_tentative.aggregates.size() * 6U;
        const double block_density = static_cast<double>(block6.block_nnz()) /
            static_cast<double>(block6.block_rows * block6.block_cols);

        const bool oracle_ok =
            block1_oracle_error <= 1.0e-10 &&
            cpu_block_forward_error <= 1.0e-10 &&
            cpu_block_transpose_error <= 1.0e-10 &&
            gpu_block_forward_error <= 1.0e-4 &&
            gpu_block_transpose_error <= 1.0e-4;
        const bool block_beats_factorized =
            gpu_block.forward_timing.median_ms < factorized_per_direction_lower_bound_ms &&
            gpu_block.transpose_timing.median_ms < factorized_per_direction_lower_bound_ms;
        const bool block_less_memory = block6.shared_matrix_bytes_fp32() < scalar_dual_bytes;
        const bool block_faster_than_scalar = block_roundtrip_ms < scalar_roundtrip_ms;

        const char* preferred = nullptr;
        if (oracle_ok && block_beats_factorized && block_less_memory && block_faster_than_scalar) {
            preferred = "shared_block6";
        } else if (oracle_ok && block_beats_factorized && block_less_memory) {
            preferred = "scalar_CSR_speed_vs_shared_block6_memory_tradeoff";
        } else if (oracle_ok) {
            preferred = "dual_scalar_CSR";
        } else {
            preferred = "none_oracle_failed";
        }

        std::cout << "GFSS M5 L1<->L2 block-transfer representation decision\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "hierarchy=frozen_theta_0p05_actual_L1_block_metric\n"
                  << "m0=1 m1=2\n"
                  << "block_representation=padded_6x6_BSR_values_shared_between_P1_and_P1T\n"
                  << "transpose_index=column_oriented_row_plus_forward_block_id_no_atomics\n"
                  << "scalar_baseline=separate_scalar_CSR_P1_and_P1T\n"
                  << "factorized_comparator=strict_lower_bound_2x_measured_A1_per_direction\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L1_nodes=" << block1.nodes()
                  << " L1_padded_dofs=" << l1_padded_dofs
                  << " L2_dofs=" << transfer1_tentative.coarse_dofs
                  << " L2_nodes=" << transfer1_tentative.aggregates.size()
                  << " L2_padded_dofs=" << l2_padded_dofs << '\n'
                  << std::fixed << std::setprecision(6)
                  << "lambda0=" << lambda0 << " omega0=" << omega0
                  << " block_lambda1=" << block_lambda1
                  << " block_omega1=" << block_omega1 << '\n'
                  << "P0_support_setup_ms=" << p0_support_ms
                  << " P1_smoothed_support_setup_ms=" << basis_setup_ms
                  << " scalar_CSR_assembly_ms=" << scalar_assembly_ms
                  << " block6_assembly_ms=" << block6.assembly_ms
                  << " total_cpu_setup_ms="
                  << std::chrono::duration<double, std::milli>(setup_stop - setup_start).count()
                  << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error << '\n'
                  << "cpu_block6_P1_vs_factorized_relative_error=" << cpu_block_forward_error
                  << " cpu_block6_P1T_vs_factorized_relative_error=" << cpu_block_transpose_error << '\n'
                  << "gpu_block6_P1_vs_factorized_relative_error=" << gpu_block_forward_error
                  << " gpu_block6_P1T_vs_factorized_relative_error=" << gpu_block_transpose_error << '\n'
                  << std::fixed << std::setprecision(6)
                  << "P1_block_nnz=" << block6.block_nnz()
                  << " P1_blocks_per_L1_node="
                  << static_cast<double>(block6.block_nnz()) / static_cast<double>(block6.block_rows)
                  << " P1_blocks_per_L2_node="
                  << static_cast<double>(block6.block_nnz()) / static_cast<double>(block6.block_cols)
                  << " P1_block_density=" << block_density << '\n'
                  << "block6_value_bytes_fp32=" << block6.value_bytes_fp32()
                  << " block6_forward_index_bytes=" << block6.forward_index_bytes()
                  << " block6_transpose_index_bytes=" << block6.transpose_index_bytes()
                  << " block6_shared_matrix_bytes_fp32=" << block6.shared_matrix_bytes_fp32() << '\n'
                  << "scalar_single_CSR_matrix_bytes_fp32=" << scalar_single_bytes
                  << " scalar_dual_CSR_matrix_bytes_fp32=" << scalar_dual_bytes
                  << " block6_vs_scalar_single_memory_ratio="
                  << static_cast<double>(block6.shared_matrix_bytes_fp32()) /
                     static_cast<double>(std::max<std::size_t>(scalar_single_bytes, 1U))
                  << " block6_vs_scalar_dual_memory_ratio="
                  << static_cast<double>(block6.shared_matrix_bytes_fp32()) /
                     static_cast<double>(std::max<std::size_t>(scalar_dual_bytes, 1U)) << '\n'
                  << "block6_shared_bytes_per_fine_dof="
                  << static_cast<double>(block6.shared_matrix_bytes_fp32()) /
                     static_cast<double>(mesh.dof_count()) << '\n'
                  << "A1_median_ms=" << a1_timing.median_timing.total_ms
                  << " factorized_per_direction_strict_lower_bound_ms="
                  << factorized_per_direction_lower_bound_ms << '\n'
                  << "scalar_P1_median_ms=" << gpu_scalar_forward.timing.median_ms
                  << " block6_P1_median_ms=" << gpu_block.forward_timing.median_ms
                  << " block6_P1_best_ms=" << gpu_block.forward_timing.best_ms
                  << " block6_vs_scalar_P1_speedup="
                  << gpu_scalar_forward.timing.median_ms /
                     std::max(gpu_block.forward_timing.median_ms, 1.0e-30)
                  << " block6_vs_factorized_P1_lower_bound_speedup="
                  << factorized_per_direction_lower_bound_ms /
                     std::max(gpu_block.forward_timing.median_ms, 1.0e-30) << '\n'
                  << "scalar_P1T_median_ms=" << gpu_scalar_transpose.timing.median_ms
                  << " block6_P1T_median_ms=" << gpu_block.transpose_timing.median_ms
                  << " block6_P1T_best_ms=" << gpu_block.transpose_timing.best_ms
                  << " block6_vs_scalar_P1T_speedup="
                  << gpu_scalar_transpose.timing.median_ms /
                     std::max(gpu_block.transpose_timing.median_ms, 1.0e-30)
                  << " block6_vs_factorized_P1T_lower_bound_speedup="
                  << factorized_per_direction_lower_bound_ms /
                     std::max(gpu_block.transpose_timing.median_ms, 1.0e-30) << '\n'
                  << "scalar_roundtrip_median_ms=" << scalar_roundtrip_ms
                  << " block6_roundtrip_median_ms=" << block_roundtrip_ms
                  << " factorized_roundtrip_strict_lower_bound_ms="
                  << factorized_roundtrip_lower_bound_ms
                  << " block6_vs_scalar_roundtrip_speedup="
                  << scalar_roundtrip_ms / std::max(block_roundtrip_ms, 1.0e-30)
                  << " block6_vs_factorized_roundtrip_lower_bound_speedup="
                  << factorized_roundtrip_lower_bound_ms /
                     std::max(block_roundtrip_ms, 1.0e-30) << '\n'
                  << "gpu_block6_device_bytes_including_both_direction_vectors="
                  << gpu_block.device_bytes << '\n'
                  << "oracle_accept=" << (oracle_ok ? "true" : "false")
                  << " block6_beats_factorized_lower_bound="
                  << (block_beats_factorized ? "true" : "false")
                  << " block6_uses_less_memory_than_scalar_dual="
                  << (block_less_memory ? "true" : "false")
                  << " block6_faster_roundtrip_than_scalar="
                  << (block_faster_than_scalar ? "true" : "false") << '\n'
                  << "preferred_transfer_representation=" << preferred << '\n';

        return oracle_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_l1_l2_block_transfer_bench "
                  << "[repeats=100 [block_y=4 [target_nodes=12 [min_nodes=4]]]]\n";
        return 1;
    }
}
