// Reference-only recursive gamma-cycle diagnostics.
// Requires recursive_sa_full_vcycle_spectral_helpers.inc and its dependencies.

namespace {

struct MgGammaVisitTrace {
    std::size_t level{0U};
    std::string label;
    double initial_residual_norm{0.0};
    double after_pre_residual_norm{0.0};
    double coarse_rhs_norm{0.0};
    double coarse_correction_norm{0.0};
    double coarse_correction_energy{0.0};
    double after_coarse_residual_norm{0.0};
    double after_post_residual_norm{0.0};
};

struct MgGammaTopEnergyTrace {
    double initial_energy{0.0};
    double after_pre_energy{0.0};
    double after_coarse_energy{0.0};
    double after_post_energy{0.0};
    double initial_residual_norm{0.0};
    double after_pre_residual_norm{0.0};
    double after_coarse_residual_norm{0.0};
    double after_post_residual_norm{0.0};
};

struct MgGammaWorkCounters {
    std::vector<std::size_t> level_visits;
};

double mg_vector_energy_norm(const gfss::ReferenceMultilevelLevel& level,
                             const Vec& x) {
    const auto ax = level.apply(x);
    if (ax.size() != x.size()) {
        throw std::runtime_error("gamma-cycle energy apply size mismatch");
    }
    const double e2 = dot(x, ax);
    if (!(e2 >= -1.0e-18) || !std::isfinite(e2)) {
        throw std::runtime_error("gamma-cycle energy norm invalid");
    }
    return std::sqrt(std::max(0.0, e2));
}

void mg_gamma_cycle(std::size_t level_index,
                    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
                    const Vec& b,
                    Vec& x,
                    std::size_t pre_smooth_steps,
                    std::size_t post_smooth_steps,
                    std::size_t gamma,
                    MgGammaWorkCounters* counters,
                    std::vector<MgGammaVisitTrace>* visit_trace,
                    MgGammaTopEnergyTrace* top_trace) {
    if (gamma == 0U || level_index >= levels.size()) {
        throw std::invalid_argument("gamma-cycle recursion options invalid");
    }
    const auto& level = levels[level_index];
    if (b.size() != level.dofs || x.size() != level.dofs) {
        throw std::runtime_error("gamma-cycle level vector size mismatch");
    }

    if (counters != nullptr) {
        if (counters->level_visits.size() != levels.size()) {
            counters->level_visits.assign(levels.size(), 0U);
        }
        ++counters->level_visits[level_index];
    }

    MgGammaVisitTrace visit;
    visit.level = level_index;
    visit.label = level.label;
    visit.initial_residual_norm = mg_level_residual_norm(level, b, x);

    if (level_index == 0U && top_trace != nullptr) {
        top_trace->initial_energy = mg_vector_energy_norm(level, x);
        top_trace->initial_residual_norm = visit.initial_residual_norm;
    }

    if (level_index + 1U == levels.size()) {
        x = level.bottom_solve(b);
        if (x.size() != level.dofs) {
            throw std::runtime_error("gamma-cycle bottom solve size mismatch");
        }
        visit.after_pre_residual_norm = visit.initial_residual_norm;
        visit.after_coarse_residual_norm = visit.initial_residual_norm;
        visit.after_post_residual_norm = mg_level_residual_norm(level, b, x);
        if (visit_trace != nullptr) visit_trace->push_back(std::move(visit));
        return;
    }

    level.smooth(b, x, pre_smooth_steps);
    visit.after_pre_residual_norm = mg_level_residual_norm(level, b, x);
    if (level_index == 0U && top_trace != nullptr) {
        top_trace->after_pre_energy = mg_vector_energy_norm(level, x);
        top_trace->after_pre_residual_norm = visit.after_pre_residual_norm;
    }

    const auto ax = level.apply(x);
    Vec residual(level.dofs, 0.0);
    for (std::size_t i = 0; i < level.dofs; ++i) residual[i] = b[i] - ax[i];
    auto coarse_b = level.restrict_to_coarse(residual);
    const auto& coarse = levels[level_index + 1U];
    if (coarse_b.size() != coarse.dofs) {
        throw std::runtime_error("gamma-cycle restriction size mismatch");
    }
    visit.coarse_rhs_norm = norm(coarse_b);

    Vec coarse_e(coarse.dofs, 0.0);
    for (std::size_t rep = 0; rep < gamma; ++rep) {
        mg_gamma_cycle(level_index + 1U,
                       levels,
                       coarse_b,
                       coarse_e,
                       pre_smooth_steps,
                       post_smooth_steps,
                       gamma,
                       counters,
                       visit_trace,
                       nullptr);
    }
    visit.coarse_correction_norm = norm(coarse_e);
    visit.coarse_correction_energy = mg_vector_energy_norm(coarse, coarse_e);

    const auto correction = level.prolong_from_coarse(coarse_e);
    if (correction.size() != level.dofs) {
        throw std::runtime_error("gamma-cycle prolongation size mismatch");
    }
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += correction[i];
    visit.after_coarse_residual_norm = mg_level_residual_norm(level, b, x);
    if (level_index == 0U && top_trace != nullptr) {
        top_trace->after_coarse_energy = mg_vector_energy_norm(level, x);
        top_trace->after_coarse_residual_norm = visit.after_coarse_residual_norm;
    }

    level.smooth(b, x, post_smooth_steps);
    visit.after_post_residual_norm = mg_level_residual_norm(level, b, x);
    if (level_index == 0U && top_trace != nullptr) {
        top_trace->after_post_energy = mg_vector_energy_norm(level, x);
        top_trace->after_post_residual_norm = visit.after_post_residual_norm;
    }

    if (visit_trace != nullptr) visit_trace->push_back(std::move(visit));
}

Vec mg_apply_gamma_error_cycle(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& e,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    std::size_t gamma) {
    if (levels.empty() || e.size() != levels.front().dofs) {
        throw std::invalid_argument("gamma-cycle error input mismatch");
    }
    Vec b(e.size(), 0.0);
    Vec x = e;
    mg_gamma_cycle(0U, levels, b, x,
                   pre_smooth_steps, post_smooth_steps, gamma,
                   nullptr, nullptr, nullptr);
    return x;
}

struct MgGammaTraceResult {
    MgGammaTopEnergyTrace top;
    std::vector<MgGammaVisitTrace> visits;
    MgGammaWorkCounters work;
};

MgGammaTraceResult mg_trace_gamma_error_cycle(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& e,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    std::size_t gamma) {
    Vec b(e.size(), 0.0);
    Vec x = e;
    MgGammaTraceResult result;
    result.work.level_visits.assign(levels.size(), 0U);
    mg_gamma_cycle(0U, levels, b, x,
                   pre_smooth_steps, post_smooth_steps, gamma,
                   &result.work, &result.visits, &result.top);
    return result;
}

FullMgSpectrum run_gamma_mg_spectrum(
    const std::vector<Vec>& seeds,
    const Apply& apply,
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    std::size_t iterations,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    std::size_t gamma) {
    if (seeds.empty() || iterations == 0U || gamma == 0U) {
        throw std::invalid_argument("gamma-cycle block spectrum options invalid");
    }

    auto block = a_orthonormalize_block(seeds, apply);
    FullMgSpectrum result;
    result.subspace_leakage.reserve(iterations);

    for (std::size_t it = 0; it < iterations; ++it) {
        std::vector<Vec> next;
        next.reserve(block.q.size());
        for (const auto& v : block.q) {
            next.push_back(mg_apply_gamma_error_cycle(
                levels, v, pre_smooth_steps, post_smooth_steps, gamma));
        }
        auto next_block = a_orthonormalize_block(std::move(next), apply);
        result.subspace_leakage.push_back(block_subspace_leakage(block, next_block));
        block = std::move(next_block);
    }
    result.final_a_orthogonality_error = block.orthogonality_error;

    const std::size_t m = block.q.size();
    std::vector<Vec> ev(m);
    std::vector<Vec> aev(m);
    std::vector<double> h(m * m, 0.0);
    double h_norm2 = 0.0;
    double h_asym2 = 0.0;

    for (std::size_t j = 0; j < m; ++j) {
        ev[j] = mg_apply_gamma_error_cycle(
            levels, block.q[j], pre_smooth_steps, post_smooth_steps, gamma);
        aev[j] = apply(ev[j]);
    }
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            h[i * m + j] = dot(block.q[i], aev[j]);
            h_norm2 += h[i * m + j] * h[i * m + j];
        }
    }
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i + 1U; j < m; ++j) {
            const double d = h[i * m + j] - h[j * m + i];
            h_asym2 += 2.0 * d * d;
            const double sym = 0.5 * (h[i * m + j] + h[j * m + i]);
            h[i * m + j] = sym;
            h[j * m + i] = sym;
        }
    }
    result.projected_operator_symmetry_defect =
        h_norm2 > 0.0 ? std::sqrt(h_asym2 / h_norm2) : 0.0;

    const auto eig = jacobi_symmetric_eigen(h, m);
    std::vector<std::size_t> order(m, 0U);
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return std::abs(eig.values[a]) > std::abs(eig.values[b]);
    });

    result.modes.reserve(m);
    result.eigenvalue.reserve(m);
    result.one_cycle_energy_q.reserve(m);
    result.one_cycle_residual_q.reserve(m);
    for (const auto idx : order) {
        Vec mode(block.q.front().size(), 0.0);
        for (std::size_t j = 0; j < m; ++j) {
            const double coeff = eig.vectors[j * m + idx];
            for (std::size_t d = 0; d < mode.size(); ++d) {
                mode[d] += coeff * block.q[j][d];
            }
        }
        oracle_normalize_energy(apply, mode);
        const auto amode = apply(mode);
        const auto emode = mg_apply_gamma_error_cycle(
            levels, mode, pre_smooth_steps, post_smooth_steps, gamma);
        const auto aemode = apply(emode);
        result.modes.push_back(std::move(mode));
        result.eigenvalue.push_back(dot(result.modes.back(), aemode));
        result.one_cycle_energy_q.push_back(oracle_energy_norm(apply, emode));
        result.one_cycle_residual_q.push_back(
            norm(aemode) / std::max(norm(amode), 1.0e-300));
    }
    return result;
}

struct MgGammaSolveResult {
    gfss::ReferenceMultilevelResult solve;
    MgGammaWorkCounters work;
};

MgGammaSolveResult mg_solve_gamma_cycles(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& rhs,
    double relative_tolerance,
    std::size_t max_cycles,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    std::size_t gamma) {
    if (levels.empty() || rhs.size() != levels.front().dofs ||
        !(relative_tolerance > 0.0) || max_cycles == 0U || gamma == 0U) {
        throw std::invalid_argument("gamma-cycle solve options invalid");
    }
    MgGammaSolveResult out;
    out.work.level_visits.assign(levels.size(), 0U);
    out.solve.x.assign(rhs.size(), 0.0);
    const double bnorm = norm(rhs);
    if (!(bnorm > 0.0)) throw std::invalid_argument("gamma-cycle solve RHS is zero");
    out.solve.relative_residuals.push_back(1.0);

    const auto start = Clock::now();
    for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
        mg_gamma_cycle(0U, levels, rhs, out.solve.x,
                       pre_smooth_steps, post_smooth_steps, gamma,
                       &out.work, nullptr, nullptr);
        out.solve.cycles = cycle + 1U;
        const auto r = mg_true_residual(levels.front(), rhs, out.solve.x);
        const double rel = norm(r) / bnorm;
        if (!std::isfinite(rel)) {
            throw std::runtime_error("gamma-cycle true residual became non-finite");
        }
        out.solve.relative_residuals.push_back(rel);
        if (rel <= relative_tolerance) {
            out.solve.converged = true;
            break;
        }
    }
    out.solve.solve_ms = elapsed_ms(start, Clock::now());
    return out;
}

struct MgGammaVisitSummary {
    std::size_t level{0U};
    std::string label;
    std::size_t visits{0U};
    double visit_local_pre_geomean{std::numeric_limits<double>::quiet_NaN()};
    double visit_local_post_geomean{std::numeric_limits<double>::quiet_NaN()};
    double max_after_coarse_over_initial{0.0};
    double last_after_post_over_initial{std::numeric_limits<double>::quiet_NaN()};
    double last_correction_energy{0.0};
};

MgGammaVisitSummary mg_summarize_gamma_visits(
    const std::vector<MgGammaVisitTrace>& visits,
    std::size_t level_index) {
    MgGammaVisitSummary out;
    out.level = level_index;
    double pre_log = 0.0;
    double post_log = 0.0;
    std::size_t pre_count = 0U;
    std::size_t post_count = 0U;
    for (const auto& v : visits) {
        if (v.level != level_index) continue;
        out.label = v.label;
        ++out.visits;
        const double denom = std::max(v.initial_residual_norm, 1.0e-300);
        const double qpre = v.after_pre_residual_norm / denom;
        const double qpost = v.after_post_residual_norm / denom;
        const double qcoarse = v.after_coarse_residual_norm / denom;
        if (qpre > 0.0 && std::isfinite(qpre)) {
            pre_log += std::log(qpre);
            ++pre_count;
        }
        if (qpost > 0.0 && std::isfinite(qpost)) {
            post_log += std::log(qpost);
            ++post_count;
        }
        out.max_after_coarse_over_initial =
            std::max(out.max_after_coarse_over_initial, qcoarse);
        out.last_after_post_over_initial = qpost;
        out.last_correction_energy = v.coarse_correction_energy;
    }
    if (pre_count > 0U) {
        out.visit_local_pre_geomean =
            std::exp(pre_log / static_cast<double>(pre_count));
    }
    if (post_count > 0U) {
        out.visit_local_post_geomean =
            std::exp(post_log / static_cast<double>(post_count));
    }
    return out;
}

}  // namespace
