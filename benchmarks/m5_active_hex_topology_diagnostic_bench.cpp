// Diagnostic companion to the frozen active-HEX topology front-end gate.
// Reuses its compact orphan-connectivity implementation, then decomposes the
// shallow two-grid residual trajectory and compares raw tentative vs one-step
// smoothed aggregation.  This is diagnostic-only: no solver parameter is tuned.
#define main gfss_m5_active_hex_topology_reference_main
#include "m5_active_hex_topology_reference_bench.cpp"
#undef main

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct CycleStages {
    double after_pre{0.0};
    double after_coarse{0.0};
    double after_post{0.0};
};

CycleStages cycle_stages(const ActiveHexDomain& domain,
                         const Apply& apply,
                         const Vec& inverse,
                         double lambda0,
                         const GenericSmoothedTransfer& transfer,
                         const DenseFactor& a1) {
    const auto b = make_rhs(domain);
    const double bnorm = norm(b);
    if (!(bnorm > 0.0)) throw std::runtime_error("topology diagnostic RHS is zero");

    Vec x(domain.dofs(), 0.0);
    Vec r(b.size(), 0.0);
    const auto weights = chebyshev_weights(lambda0, 5U);

    smooth(domain, apply, inverse, weights, b, x);
    auto ax = apply(x);
    for (std::size_t i = 0U; i < b.size(); ++i) r[i] = b[i] - ax[i];
    CycleStages out;
    out.after_pre = norm(r) / bnorm;

    const auto b1 = transfer.restrict_transpose(r);
    const auto x1 = a1.solve(b1);
    const auto corr = transfer.prolong(x1);
    for (std::size_t i = 0U; i < x.size(); ++i) x[i] += corr[i];
    ax = apply(x);
    for (std::size_t i = 0U; i < b.size(); ++i) r[i] = b[i] - ax[i];
    out.after_coarse = norm(r) / bnorm;

    smooth(domain, apply, inverse, weights, b, x);
    ax = apply(x);
    for (std::size_t i = 0U; i < b.size(); ++i) r[i] = b[i] - ax[i];
    out.after_post = norm(r) / bnorm;
    return out;
}

void print_rank_deficient_aggregates(const gfss::ElasticityAggregationCoarseSpace& space) {
    std::vector<std::vector<std::size_t>> nodes(space.aggregates.size());
    for (std::size_t node = 0U; node < space.aggregate_of_node.size(); ++node) {
        const auto aggregate = space.aggregate_of_node[node];
        if (aggregate < nodes.size()) nodes[aggregate].push_back(node);
    }

    std::size_t count = 0U;
    for (std::size_t a = 0U; a < space.aggregates.size(); ++a) {
        const auto& info = space.aggregates[a];
        if (info.rank >= 6U) continue;
        ++count;
        std::array<double, 3> lo{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
        std::array<double, 3> hi{
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()};
        double degree_sum = 0.0;
        std::size_t min_degree = std::numeric_limits<std::size_t>::max();
        std::size_t max_degree = 0U;
        for (const auto node : nodes[a]) {
            const auto& xyz = space.graph.coordinates[node];
            for (std::size_t d = 0U; d < 3U; ++d) {
                lo[d] = std::min(lo[d], xyz[d]);
                hi[d] = std::max(hi[d], xyz[d]);
            }
            const std::size_t degree =
                static_cast<std::size_t>(space.graph.row_offsets[node + 1U] -
                                         space.graph.row_offsets[node]);
            degree_sum += static_cast<double>(degree);
            min_degree = std::min(min_degree, degree);
            max_degree = std::max(max_degree, degree);
        }
        const double avg_degree = nodes[a].empty()
            ? 0.0 : degree_sum / static_cast<double>(nodes[a].size());
        std::cout << std::fixed << std::setprecision(6)
                  << "rank_deficient_aggregate=" << a
                  << " rank=" << info.rank
                  << " nodes=" << info.node_count
                  << " centroid=" << info.centroid[0] << ','
                  << info.centroid[1] << ',' << info.centroid[2]
                  << " span=" << (hi[0] - lo[0]) << ','
                  << (hi[1] - lo[1]) << ',' << (hi[2] - lo[2])
                  << " graph_degree_min=" << (nodes[a].empty() ? 0U : min_degree)
                  << " graph_degree_avg=" << avg_degree
                  << " graph_degree_max=" << max_degree << '\n';
    }
    std::cout << "rank_deficient_aggregate_count=" << count << '\n';
}

void run_diagnostic(const CaseDef& test,
                    std::size_t target_nodes,
                    std::size_t min_nodes) {
    const auto domain = make_domain(test);
    auto graph = build_graph(domain);
    const std::size_t components = graph_components(graph);
    const auto space = gfss::build_elasticity_aggregation_coarse_space(
        std::move(graph), {target_nodes, min_nodes, 1.0e-10});

    const Apply apply = [&](const Vec& x) { return apply_fine(domain, x); };
    const auto inverse = inverse_diagonal(domain);
    const double lambda0 = estimate_lambda(apply, inverse, 8U);
    const double omega0 = kSaDampingNumerator / lambda0;

    const GenericSmoothedTransfer smoothed{domain, space, apply, inverse, omega0};
    const auto smoothed_a1 = materialize_and_factor_a1(smoothed, apply);
    const auto smoothed_stages = cycle_stages(
        domain, apply, inverse, lambda0, smoothed, smoothed_a1);

    const GenericSmoothedTransfer tentative{domain, space, apply, inverse, 0.0};
    const auto tentative_a1 = materialize_and_factor_a1(tentative, apply);
    const auto tentative_stages = cycle_stages(
        domain, apply, inverse, lambda0, tentative, tentative_a1);

    const double rb_error = gfss::audit_elasticity_rigid_body_reproduction(space);
    const double smoothed_pt_error = adjoint_error(smoothed);
    const bool algebraic_oracle_accept =
        components == 1U &&
        rb_error <= 1.0e-10 &&
        smoothed_pt_error <= 1.0e-10 &&
        smoothed_a1.symmetry_relative_defect <= 1.0e-10 &&
        smoothed_a1.min_pivot > 0.0 && std::isfinite(smoothed_a1.min_pivot);

    std::cout << "case=" << test.name << '\n'
              << "description=" << test.description << '\n'
              << "active_elements=" << domain.elements.size()
              << " active_nodes=" << domain.nodes()
              << " dofs=" << domain.dofs()
              << " graph_components=" << components
              << " L1_dofs=" << space.coarse_dofs
              << " aggregates=" << space.aggregates.size() << '\n'
              << std::scientific << std::setprecision(12)
              << "rigid_body_reproduction_error=" << rb_error
              << " smoothed_transfer_adjoint_error=" << smoothed_pt_error
              << " smoothed_A1_symmetry_relative_defect="
              << smoothed_a1.symmetry_relative_defect
              << " smoothed_A1_min_cholesky_pivot=" << smoothed_a1.min_pivot
              << " algebraic_oracle_accept="
              << (algebraic_oracle_accept ? "true" : "false") << '\n'
              << std::fixed << std::setprecision(6)
              << "lambda0=" << lambda0 << " omega0=" << omega0 << '\n';

    print_rank_deficient_aggregates(space);

    const auto print_stages = [](const char* name, const CycleStages& s) {
        const double coarse_ratio = s.after_pre > 0.0
            ? s.after_coarse / s.after_pre : 0.0;
        const double post_ratio = s.after_coarse > 0.0
            ? s.after_post / s.after_coarse : 0.0;
        std::cout << std::fixed << std::setprecision(6)
                  << name
                  << "_q_after_pre=" << s.after_pre
                  << " " << name << "_q_after_coarse=" << s.after_coarse
                  << " " << name << "_coarse_stage_ratio=" << coarse_ratio
                  << " " << name << "_q_after_post=" << s.after_post
                  << " " << name << "_post_stage_ratio=" << post_ratio << '\n';
    };
    print_stages("smoothed", smoothed_stages);
    print_stages("tentative", tentative_stages);
    std::cout << std::fixed << std::setprecision(6)
              << "smoothed_vs_tentative_final_ratio="
              << smoothed_stages.after_post /
                 std::max(tentative_stages.after_post, 1.0e-300)
              << " smoothed_two_grid_contracts="
              << (smoothed_stages.after_post < 1.0 ? "true" : "false")
              << " tentative_two_grid_contracts="
              << (tentative_stages.after_post < 1.0 ? "true" : "false") << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string selector = argc > 1 ? argv[1] : "notched_beam";
        const std::size_t target_nodes = argc > 2 ? std::stoull(argv[2]) : 12U;
        const std::size_t min_nodes = argc > 3 ? std::stoull(argv[3]) : 4U;
        if (target_nodes < 2U || min_nodes == 0U || min_nodes > target_nodes) {
            throw std::invalid_argument("invalid topology diagnostic aggregation options");
        }

        std::cout << "GFSS M5 active-HEX topology diagnostic\n"
                  << "purpose=decompose_shallow_two_grid_topology_sensitivity\n"
                  << "no_parameter_tuning=true\n"
                  << "target_nodes=" << target_nodes
                  << " min_nodes=" << min_nodes
                  << " selector=" << selector << '\n';

        std::size_t selected = 0U;
        for (const auto& test : cases()) {
            if (selector != "all" && selector != test.name) continue;
            ++selected;
            std::cout << "\n========================================\n";
            run_diagnostic(test, target_nodes, min_nodes);
        }
        if (selected == 0U) throw std::invalid_argument("unknown topology diagnostic case");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n'
                  << "usage: gfss_m5_active_hex_topology_diagnostic_bench "
                  << "[all|box_control|l_solid|through_hole|notched_beam "
                  << "[target_nodes=12 [min_nodes=4]]]\n";
        return 1;
    }
}
