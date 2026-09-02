#pragma once

// Complete exact M5 hierarchy construction using the validated fast setup path.
// Include after recursive_sa_local_l2_helpers.inc and
// recursive_sa_actual_a1_strength_local_helpers.inc.

#include "m5_p1_block6_setup.hpp"
#include "m5_l2_dense_setup.hpp"
#include "m5_fast_hierarchy_setup.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace m5_fast_bundle {

using Clock = std::chrono::steady_clock;

struct StageTimes {
    double l2_basis_ms{0.0};
    double cached_a1p1_ms{0.0};
    double l2_metric_ms{0.0};
    double p1_payload_ms{0.0};
    double a2_ms{0.0};
    double lambda2_ms{0.0};
    double p2_ms{0.0};
    double bottom_ms{0.0};
    double final_payload_ms{0.0};
};

struct OracleErrors {
    double l1_block{0.0};
    double l2_block{0.0};
    double a2_dense{0.0};
    double p2_dense{0.0};
    double bottom{0.0};
    double bottom_inverse_identity{0.0};
    bool accept{false};
};

struct FastHierarchy {
    gfss::ElasticityAggregationCoarseSpace space0;
    double lambda0{0.0};
    double omega0{0.0};
    double lambda1{0.0};
    double omega1{0.0};
    double lambda2{0.0};
    double omega2{0.0};

    L1BlockMetric block1;
    L1BlockMetric block2;
    CandidateTransfer transfer1;
    CandidateTransfer transfer2;
    m5_p1_setup::DualOrderBlock6Transfer p1;
    std::vector<float> inverse1;
    std::vector<float> inverse2;
    m5_l2_setup::DenseA2 a2;
    m5_l2_setup::DenseP2 p2;
    LocalBottomReference bottom;
    std::vector<float> a2_fp32;
    std::vector<float> p2_fp32;
    std::vector<float> bottom_inverse_fp32;

    StageTimes stages;
    OracleErrors oracle;
    double production_setup_ms{0.0};
    double validation_oracle_ms{0.0};
};

inline double relative_error(const Vec& a, const Vec& b) {
    if (a.size() != b.size()) throw std::invalid_argument("fast hierarchy oracle size mismatch");
    Vec d(a.size(), 0.0);
    for (std::size_t i = 0U; i < a.size(); ++i) d[i] = a[i] - b[i];
    return norm(d) / std::max(norm(b), 1.0e-300);
}

inline double inverse_identity_relative_error(
    const LocalBottomReference& bottom,
    const std::vector<float>& inverse_col_major) {
    const std::size_t n = bottom.factor.n;
    if (inverse_col_major.size() != n * n) {
        throw std::invalid_argument("fast hierarchy bottom inverse size mismatch");
    }
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

inline FastHierarchy build(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    std::size_t target_nodes,
    std::size_t min_nodes,
    bool run_validation_oracles = true) {
    if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
        throw std::invalid_argument("invalid fast hierarchy aggregation options");
    }
    constexpr std::size_t m0 = 1U;
    constexpr std::size_t m1 = 2U;
    constexpr std::size_t m2 = 1U;
    constexpr double strength_threshold = 0.05;

    FastHierarchy out;
    const auto production_start = Clock::now();

    auto graph0 = gfss::build_structured_hex_nodal_graph_x0(mesh);
    auto space0 = gfss::build_elasticity_aggregation_coarse_space(
        std::move(graph0), {target_nodes, min_nodes, 1.0e-10});
    const auto tentative_a1 = gfss::assemble_structured_hex_aggregation_galerkin(
        mesh, material, space0);
    const auto fine_inverse = build_fine_inverse_diagonal(mesh, material, space0);
    const Apply apply0 = [&](const Vec& x) { return apply_fine_clamped(mesh, material, x); };
    const double lambda0 = estimate_lambda_max(apply0, fine_inverse, 8U);
    const double omega0 = kSaDampingNumerator / lambda0;
    const FineSmoothedTransfer transfer0{mesh, material, space0, fine_inverse, omega0, m0};
    const Apply apply1 = [&](const Vec& x) {
        return transfer0.restrict_transpose(apply0(transfer0.prolong(x)));
    };

    const auto graph1_tentative = graph_from_variable_blocks(tentative_a1);
    const auto candidates1 = make_level1_candidates(space0);
    auto block1 = build_exact_l1_block_metric(
        mesh, material, space0, graph1_tentative, fine_inverse, omega0);
    const double lambda1 = estimate_lambda_max_l1_block(apply1, block1, 8U);
    const double omega1 = kSaDampingNumerator / lambda1;

    double p0_support_ms = 0.0;
    const auto fine_supports = build_fine_basis_support_cache(
        mesh, material, space0, fine_inverse, omega0, p0_support_ms);
    const auto element_supports = build_element_support_index(mesh, fine_supports);
    const auto actual_a1_offdiagonal = accumulate_combined_actual_a1_offdiagonal_blocks(
        mesh, material, fine_supports, element_supports);
    const auto strength1 = build_combined_strength_graph(
        graph1_tentative, block1, actual_a1_offdiagonal, strength_threshold);
    auto transfer1 = build_candidate_transfer(
        strength1.graph, candidates1, target_nodes, min_nodes, 1.0e-10);
    const L1BlockSmoothedTransfer transfer1_nested{
        transfer1, apply1, block1, omega1, m1};
    const Apply apply2_nested = [&](const Vec& x) {
        return transfer1_nested.restrict_transpose(apply1(transfer1_nested.prolong(x)));
    };
    const auto local_a1 = [&](const LocalColumns& x) {
        return apply_local_a1_columns(mesh, material, fine_supports, element_supports, x);
    };

    auto stage = Clock::now();
    const auto l2_basis = m5_fast_setup::build_smoothed_supports_parallel(
        transfer1, strength1.graph, block1, omega1, m1, local_a1);
    out.stages.l2_basis_ms = m5_fast_setup::elapsed_ms(stage, Clock::now());

    stage = Clock::now();
    const auto applied_l2_basis = m5_fast_setup::apply_supports_parallel(l2_basis, local_a1);
    out.stages.cached_a1p1_ms = m5_fast_setup::elapsed_ms(stage, Clock::now());

    stage = Clock::now();
    auto block2 = m5_fast_setup::metric_from_cached_applied(
        transfer1, block1, l2_basis, applied_l2_basis);
    out.stages.l2_metric_ms = m5_fast_setup::elapsed_ms(stage, Clock::now());

    stage = Clock::now();
    auto p1 = m5_p1_setup::assemble_dual_order_block6(transfer1, block1, l2_basis);
    auto inverse1 = m5_l2_setup::inverse_blocks_6x6_fp32(block1);
    auto inverse2 = m5_l2_setup::inverse_blocks_6x6_fp32(block2);
    out.stages.p1_payload_ms = m5_fast_setup::elapsed_ms(stage, Clock::now());

    stage = Clock::now();
    auto a2 = m5_fast_setup::dense_a2_from_cached_applied(
        transfer1, block1, l2_basis, applied_l2_basis);
    out.stages.a2_ms = m5_fast_setup::elapsed_ms(stage, Clock::now());

    const Apply apply2_dense = [&](const Vec& x) {
        return m5_l2_setup::apply_dense_a2(a2, x);
    };
    stage = Clock::now();
    const double lambda2 = estimate_lambda_max_l1_block(apply2_dense, block2, 8U);
    const double omega2 = kSaDampingNumerator / lambda2;
    out.stages.lambda2_ms = m5_fast_setup::elapsed_ms(stage, Clock::now());

    auto transfer2 = build_candidate_transfer(
        transfer1.coarse_graph, transfer1.coarse_candidates,
        target_nodes, min_nodes, 1.0e-10);

    stage = Clock::now();
    auto p2 = m5_fast_setup::dense_smoothed_p2_from_a2(
        transfer2, block2, a2, omega2);
    out.stages.p2_ms = m5_fast_setup::elapsed_ms(stage, Clock::now());

    stage = Clock::now();
    auto bottom = m5_fast_setup::dense_bottom_from_a2_p2(a2, p2);
    out.stages.bottom_ms = m5_fast_setup::elapsed_ms(stage, Clock::now());

    stage = Clock::now();
    auto bottom_inverse_fp32 = m5_fast_setup::symmetric_inverse_col_major(bottom.factor);
    auto a2_fp32 = m5_l2_setup::to_float(a2.fp64);
    auto p2_fp32 = m5_l2_setup::to_float(p2.fp64);
    out.stages.final_payload_ms = m5_fast_setup::elapsed_ms(stage, Clock::now());

    out.production_setup_ms = m5_fast_setup::elapsed_ms(production_start, Clock::now());

    if (run_validation_oracles) {
        const auto validation_start = Clock::now();
        out.oracle.l1_block = audit_l1_block_metric(block1, apply1);
        out.oracle.l2_block = audit_l1_block_metric(block2, apply2_nested);
        const auto probe2 = deterministic_actual_a2_probe(a2.n, 0.63);
        out.oracle.a2_dense = relative_error(apply2_dense(probe2), apply2_nested(probe2));
        const L1BlockSmoothedTransfer transfer2_nested{
            transfer2, apply2_nested, block2, omega2, m2};
        const auto probe3 = deterministic_actual_a2_probe(transfer2.coarse_dofs, 0.91);
        out.oracle.p2_dense = relative_error(
            m5_l2_setup::apply_p2(p2, probe3), transfer2_nested.prolong(probe3));
        const Apply apply3_nested = [&](const Vec& x) {
            return transfer2_nested.restrict_transpose(
                apply2_nested(transfer2_nested.prolong(x)));
        };
        out.oracle.bottom = bottom_local_oracle_error(bottom, apply3_nested);
        out.oracle.bottom_inverse_identity =
            inverse_identity_relative_error(bottom, bottom_inverse_fp32);
        out.validation_oracle_ms = m5_fast_setup::elapsed_ms(validation_start, Clock::now());
        out.oracle.accept = out.oracle.l1_block <= 1.0e-10 &&
                            out.oracle.l2_block <= 1.0e-10 &&
                            out.oracle.a2_dense <= 1.0e-10 &&
                            out.oracle.p2_dense <= 1.0e-10 &&
                            out.oracle.bottom <= 1.0e-10 &&
                            out.oracle.bottom_inverse_identity <= 1.0e-4;
    } else {
        out.oracle.accept = true;
    }

    out.space0 = std::move(space0);
    out.lambda0 = lambda0;
    out.omega0 = omega0;
    out.lambda1 = lambda1;
    out.omega1 = omega1;
    out.lambda2 = lambda2;
    out.omega2 = omega2;
    out.block1 = std::move(block1);
    out.block2 = std::move(block2);
    out.transfer1 = std::move(transfer1);
    out.transfer2 = std::move(transfer2);
    out.p1 = std::move(p1);
    out.inverse1 = std::move(inverse1);
    out.inverse2 = std::move(inverse2);
    out.a2 = std::move(a2);
    out.p2 = std::move(p2);
    out.bottom = std::move(bottom);
    out.a2_fp32 = std::move(a2_fp32);
    out.p2_fp32 = std::move(p2_fp32);
    out.bottom_inverse_fp32 = std::move(bottom_inverse_fp32);
    return out;
}

}  // namespace m5_fast_bundle
