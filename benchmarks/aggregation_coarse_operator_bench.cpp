#include "gfss/aggregation_coarse_operator.hpp"
#include "gfss/cpu_elasticity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    return (n & 1U) ? v[n / 2U] : 0.5 * (v[n / 2U - 1U] + v[n / 2U]);
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

std::vector<double> probe(std::size_t n, double phase) {
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        x[i] = std::sin(0.013 * t + phase) + 0.31 * std::cos(0.037 * t);
    }
    const double nrm = std::sqrt(dot(x, x));
    for (double& v : x) v /= nrm;
    return x;
}
}

int main(int argc, char** argv) {
    try {
        const std::size_t target = argc > 1 ? std::stoull(argv[1]) : 12U;
        const std::size_t minimum = argc > 2 ? std::stoull(argv[2]) : 4U;
        const std::size_t repeats = argc > 3 ? std::stoull(argv[3]) : 10U;
        if (target == 0U || minimum == 0U || repeats == 0U) {
            throw std::invalid_argument("arguments must be positive");
        }

        const gfss::StructuredHexMesh mesh{64U, 64U, 8U, 1.0, 1.0, 0.125};
        const gfss::Material material{210.0e9, 0.30};
        auto graph = gfss::build_structured_hex_nodal_graph_x0(mesh);
        const auto space = gfss::build_elasticity_aggregation_coarse_space(
            std::move(graph), {target, minimum, 1.0e-10});

        const auto apply_fine = [&](const std::vector<double>& x) {
            return gfss::apply_matrix_free_openmp(mesh, material, x);
        };

        const auto u = probe(space.coarse_dofs, 0.17);
        const auto v = probe(space.coarse_dofs, 0.73);
        const auto pu = gfss::apply_elasticity_tentative_prolongation(space, u);
        const auto apu = apply_fine(pu);
        const auto acu = gfss::apply_elasticity_tentative_restriction(space, apu);
        const auto pv = gfss::apply_elasticity_tentative_prolongation(space, v);
        const auto apv = apply_fine(pv);
        const auto acv = gfss::apply_elasticity_tentative_restriction(space, apv);

        const double ec = dot(u, acu);
        const double ef = dot(pu, apu);
        const double energy_defect = std::abs(ec - ef) / std::max(std::abs(ef), 1.0e-300);
        const double uv = dot(u, acv);
        const double vu = dot(v, acu);
        const double symmetry_defect = std::abs(uv - vu) /
            std::max({std::abs(uv), std::abs(vu), 1.0});

        std::vector<double> tp, ta, tr, tt;
        tp.reserve(repeats); ta.reserve(repeats); tr.reserve(repeats); tt.reserve(repeats);
        for (std::size_t rep = 0; rep < repeats; ++rep) {
            const auto t0 = Clock::now();
            auto f = gfss::apply_elasticity_tentative_prolongation(space, u);
            const auto t1 = Clock::now();
            auto af = apply_fine(f);
            const auto t2 = Clock::now();
            auto c = gfss::apply_elasticity_tentative_restriction(space, af);
            const auto t3 = Clock::now();
            if (c.size() != space.coarse_dofs) throw std::runtime_error("coarse size mismatch");
            tp.push_back(ms(t0, t1)); ta.push_back(ms(t1, t2));
            tr.push_back(ms(t2, t3)); tt.push_back(ms(t0, t3));
        }

        const double p = median(tp), a = median(ta), r = median(tr), total = median(tt);
        std::cout << "GFSS M5 matrix-light aggregation coarse action\n"
                  << "coarse_action=P^T_Af_P\n"
                  << "fine_stiffness_matrix=not_assembled\n"
                  << "coarse_mesh_required=false\n"
                  << "fine_mesh=64x64x8 fine_free_dofs=" << space.fine_free_dofs
                  << " coarse_dofs=" << space.coarse_dofs << '\n'
                  << std::fixed << std::setprecision(6)
                  << "fine_to_coarse_dof_ratio="
                  << static_cast<double>(space.fine_free_dofs) / space.coarse_dofs << '\n'
                  << "P_median_ms=" << p << " Af_median_ms=" << a
                  << " PT_median_ms=" << r << " Ac_median_ms=" << total << '\n'
                  << "transfer_overhead_fraction=" << (p + r) / total
                  << " fine_operator_fraction=" << a / total << '\n'
                  << "coarse_action_fine_matvecs_per_apply=1\n"
                  << "matrix_free_transfer_payload_bytes_per_fine_free_dof="
                  << static_cast<double>(space.estimated_matrix_free_transfer_payload_bytes) /
                         space.fine_free_dofs << '\n'
                  << "one_fp32_coarse_vector_bytes_per_fine_free_dof="
                  << 4.0 * static_cast<double>(space.coarse_dofs) / space.fine_free_dofs << '\n'
                  << std::scientific << std::setprecision(9)
                  << "galerkin_energy_identity_relative_error=" << energy_defect << '\n'
                  << "coarse_symmetry_relative_defect=" << symmetry_defect << '\n'
                  << "coarse_operator_spd_probe=" << (ec > 0.0 ? "true" : "false") << '\n'
                  << "coarse_matrix_materialized=false\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
