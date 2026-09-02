// M5 stage 14: attribute the CPU hierarchy-construction bottleneck.
// This benchmark reconstructs the frozen production hierarchy exactly once and
// reports required setup separately from benchmark-only FP64 validation oracles.
#include "recursive_sa_local_l2_helpers.inc"
#include "recursive_sa_actual_a1_strength_local_helpers.inc"
#include "m5_p1_block6_setup.hpp"
#include "m5_l2_dense_setup.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using ProfileClock = std::chrono::steady_clock;

double profile_elapsed_ms(ProfileClock::time_point a, ProfileClock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<float> symmetric_inverse_col_major(const DenseCholesky& factor) {
    const std::size_t n = factor.n;
    std::vector<double> inverse(n * n, 0.0);
    for (std::size_t j = 0U; j < n; ++j) {
        Vec e(n, 0.0);
        e[j] = 1.0;
        const auto column = factor.solve(e);
        for (std::size_t i = 0U; i < n; ++i) inverse[i * n + j] = column[i];
    }
    std::vector<float> out(n * n, 0.0f);
    for (std::size_t i = 0U; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            const float value = static_cast<float>(
                0.5 * (inverse[i * n + j] + inverse[j * n + i]));
            out[j * n + i] = value;
            out[i * n + j] = value;
        }
    }
    return out;
}

double inverse_identity_relative_error(
    const LocalBottomReference& bottom,
    const std::vector<float>& inverse_col_major) {
    const std::size_t n = bottom.factor.n;
    double d2 = 0.0;
    for (std::size_t row = 0U; row < n; ++row) {
        for (std::size_t col = 0U; col < n; ++col) {
            double value = 0.0;
            for (std::size_t k = 0U; k < n; ++k) {
                value += bottom.values[row * n + k] *
                         static_cast<double>(inverse_col_major[col * n + k]);
            }
            const double target = row == col ? 1.0 : 0.0;
            const double d = value - target;
            d2 += d * d;
        }
    }
    return std::sqrt(d2 / static_cast<double>(n));
}

struct SetupTimes {
    double graph_aggregation{0.0};
    double tentative_a1{0.0};
    double fine_inverse_lambda0{0.0};
    double l1_graph_candidates{0.0};
    double l1_block_metric{0.0};
    double l1_block_oracle{0.0};
    double lambda1_estimate{0.0};
    double p0_support_cache{0.0};
    double element_support_index{0.0};
    double actual_a1_offdiagonal{0.0};
    double strength_transfer1{0.0};
    double l2_basis{0.0};
    double l2_block_metric{0.0};
    double l2_block_oracle{0.0};
    double lambda2_estimate{0.0};
    double p1_and_block_inverses{0.0};
    double a2_dense{0.0};
    double transfer2_tentative{0.0};
    double bottom_basis{0.0};
    double p2_dense{0.0};
    double bottom_assembly{0.0};
    double bottom_oracle{0.0};
    double bottom_inverse{0.0};
    double bottom_inverse_oracle{0.0};

    double oracle_only() const {
        return l1_block_oracle + l2_block_oracle + bottom_oracle + bottom_inverse_oracle;
    }
    double all_profiled() const {
        return graph_aggregation + tentative_a1 + fine_inverse_lambda0 +
               l1_graph_candidates + l1_block_metric + l1_block_oracle +
               lambda1_estimate + p0_support_cache + element_support_index +
               actual_a1_offdiagonal + strength_transfer1 + l2_basis +
               l2_block_metric + l2_block_oracle + lambda2_estimate +
               p1_and_block_inverses + a2_dense + transfer2_tentative +
               bottom_basis + p2_dense + bottom_assembly + bottom_oracle +
               bottom_inverse + bottom_inverse_oracle;
    }
};

void print_stage(const char* name, double ms, double required_total) {
    const double pct = required_total > 0.0 ? 100.0 * ms / required_total : 0.0;
    std::cout << "setup_stage=" << name
              << " ms=" << std::fixed << std::setprecision(6) << ms
              << " pct_of_required=" << std::setprecision(3) << pct << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t target_nodes = argc > 1
            ? static_cast<std::size_t>(std::stoull(argv[1])) : 12U;
        const std::size_t min_nodes = argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : 4U;
        if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid M5 setup-profile options");
        }

        constexpr std::size_t m0 = 1U;
        constexpr std::size_t m1 = 2U;
        constexpr std::size_t m2 = 1U;
        constexpr double strength_threshold = 0.05;
        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        SetupTimes t;
        const auto total_start = ProfileClock::now();
        auto stage = ProfileClock::now();

        auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space0 = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
        t.graph_aggregation = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
            mesh, material, space0);
        t.tentative_a1 = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
        const Apply apply0 = [&](const Vec& x) { return apply_fine_clamped(mesh, material, x); };
        const double lambda0 = estimate_lambda_max(apply0, fine_inverse, 8U);
        const double omega0 = kSaDampingNumerator / lambda0;
        const FineSmoothedTransfer transfer0{mesh, material, space0, fine_inverse, omega0, m0};
        const Apply apply1 = [&](const Vec& x) {
            return transfer0.restrict_transpose(apply0(transfer0.prolong(x)));
        };
        t.fine_inverse_lambda0 = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto graph1_tentative = graph_from_variable_blocks(tentative_a1);
        const auto candidates1 = make_level1_candidates(space0);
        t.l1_graph_candidates = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto block1 = build_exact_l1_block_metric(
            mesh, material, space0, graph1_tentative, fine_inverse, omega0);
        t.l1_block_metric = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const double block1_oracle_error = audit_l1_block_metric(block1, apply1);
        t.l1_block_oracle = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const double lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
        const double omega1 = kSaDampingNumerator / lambda1;
        t.lambda1_estimate = profile_elapsed_ms(stage, ProfileClock::now());

        double internal_p0_support_ms = 0.0;
        stage = ProfileClock::now();
        const auto fine_supports = build_fine_basis_support_cache(
            mesh, material, space0, fine_inverse, omega0, internal_p0_support_ms);
        t.p0_support_cache = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto element_supports = build_element_support_index(mesh, fine_supports);
        t.element_support_index = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto actual_a1_offdiagonal = accumulate_combined_actual_a1_offdiagonal_blocks(
            mesh, material, fine_supports, element_supports);
        t.actual_a1_offdiagonal = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto strength1 = build_combined_strength_graph(
            graph1_tentative, block1, actual_a1_offdiagonal, strength_threshold);
        const auto transfer1_tentative = build_candidate_transfer(
            strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
        const L1BlockSmoothedTransfer transfer1{
            transfer1_tentative, apply1, block1, omega1, m1};
        const Apply apply2 = [&](const Vec& x) {
            return transfer1.restrict_transpose(apply1(transfer1.prolong(x)));
        };
        t.strength_transfer1 = profile_elapsed_ms(stage, ProfileClock::now());

        const auto local_a1_apply_lambda = [&](const LocalColumns& x) {
            return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
        };
        const std::function<LocalColumns(const LocalColumns&)> local_a1_apply = local_a1_apply_lambda;

        stage = ProfileClock::now();
        const auto l2_basis = build_smoothed_candidate_supports(
            transfer1_tentative, strength1.graph, block1,
            omega1, m1, local_a1_apply_lambda);
        t.l2_basis = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto block2 = build_metric_from_local_supports(
            transfer1_tentative, block1, l2_basis, local_a1_apply_lambda);
        t.l2_block_metric = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const double block2_oracle_error = audit_l1_block_metric(block2, apply2);
        t.l2_block_oracle = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const double lambda2 = estimate_lambda_max_l1_block(apply2, block2, 8U);
        const double omega2 = kSaDampingNumerator / lambda2;
        t.lambda2_estimate = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto p1 = m5_p1_setup::assemble_dual_order_block6(
            transfer1_tentative, block1, l2_basis);
        const auto inverse1 = m5_l2_setup::inverse_blocks_6x6_fp32(block1);
        const auto inverse2 = m5_l2_setup::inverse_blocks_6x6_fp32(block2);
        t.p1_and_block_inverses = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto a2 = m5_l2_setup::assemble_dense_a2(
            transfer1_tentative, block1, l2_basis, local_a1_apply);
        t.a2_dense = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto transfer2_tentative = build_candidate_transfer(
            transfer1_tentative.coarse_graph,
            transfer1_tentative.coarse_candidates,
            target_nodes, min_nodes, 1.0e-10);
        const L1BlockSmoothedTransfer transfer2{
            transfer2_tentative, apply2, block2, omega2, m2};
        t.transfer2_tentative = profile_elapsed_ms(stage, ProfileClock::now());

        const auto local_a2_apply_lambda = [&](const LocalColumns& x) {
            return apply_local_a2_columns(x, l2_basis, block1, local_a1_apply_lambda);
        };
        const std::function<LocalColumns(const LocalColumns&)> local_a2_apply = local_a2_apply_lambda;

        stage = ProfileClock::now();
        const auto bottom_basis = build_smoothed_candidate_supports(
            transfer2_tentative, transfer1_tentative.coarse_graph,
            block2, omega2, m2, local_a2_apply_lambda);
        t.bottom_basis = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto p2 = m5_l2_setup::assemble_dense_p2(
            transfer2_tentative, block2, bottom_basis);
        t.p2_dense = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto bottom = build_local_bottom(
            transfer2_tentative, block2, bottom_basis, local_a2_apply);
        t.bottom_assembly = profile_elapsed_ms(stage, ProfileClock::now());

        const Apply apply3_nested = [&](const Vec& x) {
            return transfer2.restrict_transpose(apply2(transfer2.prolong(x)));
        };
        stage = ProfileClock::now();
        const double bottom_oracle_error = bottom_local_oracle_error(bottom, apply3_nested);
        t.bottom_oracle = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const auto bottom_inverse = symmetric_inverse_col_major(bottom.factor);
        t.bottom_inverse = profile_elapsed_ms(stage, ProfileClock::now());

        stage = ProfileClock::now();
        const double bottom_inverse_identity_error =
            inverse_identity_relative_error(bottom, bottom_inverse);
        t.bottom_inverse_oracle = profile_elapsed_ms(stage, ProfileClock::now());

        const double total_ms = profile_elapsed_ms(total_start, ProfileClock::now());
        const double oracle_ms = t.oracle_only();
        const double required_ms = total_ms - oracle_ms;
        const double profiled_ms = t.all_profiled();
        const double unattributed_ms = total_ms - profiled_ms;
        const bool hierarchy_ok = block1_oracle_error <= 1.0e-10 &&
                                  block2_oracle_error <= 1.0e-10 &&
                                  bottom_oracle_error <= 1.0e-10 &&
                                  bottom_inverse_identity_error <= 1.0e-4;

        std::cout << "GFSS M5 hierarchy setup profile\n"
                  << "problem=thin_plate mesh=64x64x8\n"
                  << "hierarchy=frozen_5x1x1_dual_block6_dense_A2_P2_inverse_L3\n"
                  << "timing_semantics=required_setup_separated_from_validation_oracles\n"
                  << "fine_dofs=" << mesh.dof_count()
                  << " L1_dofs=" << space0.coarse_dofs
                  << " L2_dofs=" << block2.dofs()
                  << " L3_dofs=" << bottom.factor.n << '\n'
                  << std::scientific << std::setprecision(9)
                  << "L1_block_vs_nested_relative_error=" << block1_oracle_error
                  << " L2_block_vs_nested_relative_error=" << block2_oracle_error
                  << " bottom_local_vs_nested_relative_error=" << bottom_oracle_error
                  << " bottom_inverse_identity_relative_error=" << bottom_inverse_identity_error
                  << " hierarchy_oracle_accept=" << (hierarchy_ok ? "true" : "false") << '\n'
                  << std::fixed << std::setprecision(6)
                  << "reported_total_setup_ms=" << total_ms
                  << " benchmark_validation_oracle_ms=" << oracle_ms
                  << " production_required_setup_ms=" << required_ms
                  << " profiled_stage_sum_ms=" << profiled_ms
                  << " unattributed_ms=" << unattributed_ms << '\n'
                  << "internal_P0_support_setup_ms=" << internal_p0_support_ms
                  << " A2_internal_assembly_ms=" << a2.assembly_ms
                  << " P2_internal_assembly_ms=" << p2.assembly_ms
                  << " bottom_internal_assembly_ms=" << bottom.assembly_ms << '\n';

        print_stage("graph_aggregation", t.graph_aggregation, required_ms);
        print_stage("tentative_A1", t.tentative_a1, required_ms);
        print_stage("fine_inverse_plus_lambda0", t.fine_inverse_lambda0, required_ms);
        print_stage("L1_graph_candidates", t.l1_graph_candidates, required_ms);
        print_stage("L1_block_metric", t.l1_block_metric, required_ms);
        std::cout << "oracle_stage=L1_block_audit ms=" << t.l1_block_oracle << '\n';
        print_stage("lambda1_estimate", t.lambda1_estimate, required_ms);
        print_stage("P0_support_cache", t.p0_support_cache, required_ms);
        print_stage("element_support_index", t.element_support_index, required_ms);
        print_stage("actual_A1_offdiagonal", t.actual_a1_offdiagonal, required_ms);
        print_stage("strength_plus_transfer1", t.strength_transfer1, required_ms);
        print_stage("L2_basis", t.l2_basis, required_ms);
        print_stage("L2_block_metric", t.l2_block_metric, required_ms);
        std::cout << "oracle_stage=L2_block_audit ms=" << t.l2_block_oracle << '\n';
        print_stage("lambda2_estimate", t.lambda2_estimate, required_ms);
        print_stage("P1_plus_block_inverses", t.p1_and_block_inverses, required_ms);
        print_stage("A2_dense_assembly", t.a2_dense, required_ms);
        print_stage("transfer2_tentative", t.transfer2_tentative, required_ms);
        print_stage("bottom_basis", t.bottom_basis, required_ms);
        print_stage("P2_dense_assembly", t.p2_dense, required_ms);
        print_stage("bottom_assembly", t.bottom_assembly, required_ms);
        std::cout << "oracle_stage=bottom_nested_audit ms=" << t.bottom_oracle << '\n';
        print_stage("bottom_inverse_build", t.bottom_inverse, required_ms);
        std::cout << "oracle_stage=bottom_inverse_identity_audit ms="
                  << t.bottom_inverse_oracle << '\n';

        std::cout << "payload_P1_nnz=" << p1.forward_column_indices.size()
                  << " A2_values=" << a2.fp64.size()
                  << " P2_values=" << p2.fp64.size()
                  << " bottom_inverse_values=" << bottom_inverse.size() << '\n';
        return hierarchy_ok ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_gpu_m5_hierarchy_setup_profile_bench "
                  << "[target_nodes=12 [min_nodes=4]]\n";
        return 1;
    }
}
