// Reference-only level-selective recursive-cycle diagnostics.
// Requires recursive_sa_gamma_cycle_helpers.inc and its dependencies.

namespace {

using MgSelectiveSchedule = std::array<std::size_t, 3>;

void mg_validate_selective_schedule(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const MgSelectiveSchedule& repetitions) {
    if (levels.size() != 4U) {
        throw std::invalid_argument("selective gamma reference expects four levels");
    }
    if (repetitions[0] == 0U || repetitions[1] == 0U || repetitions[2] != 1U) {
        throw std::invalid_argument(
            "selective gamma schedule requires positive L0/L1 repeats and exact-bottom repeat=1");
    }
}

void mg_selective_cycle(
    std::size_t level_index,
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& b,
    Vec& x,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    const MgSelectiveSchedule& repetitions,
    MgGammaWorkCounters* counters,
    std::vector<MgGammaVisitTrace>* visit_trace,
    MgGammaTopEnergyTrace* top_trace) {
    mg_validate_selective_schedule(levels, repetitions);
    if (level_index >= levels.size()) {
        throw std::invalid_argument("selective gamma level index invalid");
    }
    const auto& level = levels[level_index];
    if (b.size() != level.dofs || x.size() != level.dofs) {
        throw std::runtime_error("selective gamma level vector size mismatch");
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
            throw std::runtime_error("selective gamma bottom solve size mismatch");
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
        throw std::runtime_error("selective gamma restriction size mismatch");
    }
    visit.coarse_rhs_norm = norm(coarse_b);

    Vec coarse_e(coarse.dofs, 0.0);
    const std::size_t recursive_repetitions = repetitions[level_index];
    for (std::size_t rep = 0; rep < recursive_repetitions; ++rep) {
        mg_selective_cycle(level_index + 1U,
                           levels,
                           coarse_b,
                           coarse_e,
                           pre_smooth_steps,
                           post_smooth_steps,
                           repetitions,
                           counters,
                           visit_trace,
                           nullptr);
    }
    visit.coarse_correction_norm = norm(coarse_e);
    visit.coarse_correction_energy = mg_vector_energy_norm(coarse, coarse_e);

    const auto correction = level.prolong_from_coarse(coarse_e);
    if (correction.size() != level.dofs) {
        throw std::runtime_error("selective gamma prolongation size mismatch");
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

Vec mg_apply_selective_error_cycle(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& e,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    const MgSelectiveSchedule& repetitions) {
    if (levels.empty() || e.size() != levels.front().dofs) {
        throw std::invalid_argument("selective gamma error input mismatch");
    }
    mg_validate_selective_schedule(levels, repetitions);
    Vec b(e.size(), 0.0);
    Vec x = e;
    mg_selective_cycle(0U, levels, b, x,
                       pre_smooth_steps, post_smooth_steps, repetitions,
                       nullptr, nullptr, nullptr);
    return x;
}

MgGammaTraceResult mg_trace_selective_error_cycle(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& e,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    const MgSelectiveSchedule& repetitions) {
    if (levels.empty() || e.size() != levels.front().dofs) {
        throw std::invalid_argument("selective gamma trace input mismatch");
    }
    mg_validate_selective_schedule(levels, repetitions);
    Vec b(e.size(), 0.0);
    Vec x = e;
    MgGammaTraceResult result;
    result.work.level_visits.assign(levels.size(), 0U);
    mg_selective_cycle(0U, levels, b, x,
                       pre_smooth_steps, post_smooth_steps, repetitions,
                       &result.work, &result.visits, &result.top);
    return result;
}

FullMgSpectrum run_selective_mg_spectrum(
    const std::vector<Vec>& seeds,
    const Apply& apply,
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    std::size_t iterations,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    const MgSelectiveSchedule& repetitions) {
    if (seeds.empty() || iterations == 0U) {
        throw std::invalid_argument("selective gamma block spectrum options invalid");
    }
    mg_validate_selective_schedule(levels, repetitions);

    auto block = a_orthonormalize_block(seeds, apply);
    FullMgSpectrum result;
    result.subspace_leakage.reserve(iterations);

    for (std::size_t it = 0; it < iterations; ++it) {
        std::vector<Vec> next;
        next.reserve(block.q.size());
        for (const auto& v : block.q) {
            next.push_back(mg_apply_selective_error_cycle(
                levels, v, pre_smooth_steps, post_smooth_steps, repetitions));
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
        ev[j] = mg_apply_selective_error_cycle(
            levels, block.q[j], pre_smooth_steps, post_smooth_steps, repetitions);
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
        const auto emode = mg_apply_selective_error_cycle(
            levels, mode, pre_smooth_steps, post_smooth_steps, repetitions);
        const auto aemode = apply(emode);
        result.modes.push_back(std::move(mode));
        result.eigenvalue.push_back(dot(result.modes.back(), aemode));
        result.one_cycle_energy_q.push_back(oracle_energy_norm(apply, emode));
        result.one_cycle_residual_q.push_back(
            norm(aemode) / std::max(norm(amode), 1.0e-300));
    }
    return result;
}

MgGammaSolveResult mg_solve_selective_cycles(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& rhs,
    double relative_tolerance,
    std::size_t max_cycles,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    const MgSelectiveSchedule& repetitions) {
    if (levels.empty() || rhs.size() != levels.front().dofs ||
        !(relative_tolerance > 0.0) || max_cycles == 0U) {
        throw std::invalid_argument("selective gamma solve options invalid");
    }
    mg_validate_selective_schedule(levels, repetitions);

    MgGammaSolveResult out;
    out.work.level_visits.assign(levels.size(), 0U);
    out.solve.x.assign(rhs.size(), 0.0);
    const double bnorm = norm(rhs);
    if (!(bnorm > 0.0)) throw std::invalid_argument("selective gamma solve RHS is zero");
    out.solve.relative_residuals.push_back(1.0);

    const auto start = Clock::now();
    for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
        mg_selective_cycle(0U, levels, rhs, out.solve.x,
                           pre_smooth_steps, post_smooth_steps, repetitions,
                           &out.work, nullptr, nullptr);
        out.solve.cycles = cycle + 1U;
        const auto r = mg_true_residual(levels.front(), rhs, out.solve.x);
        const double rel = norm(r) / bnorm;
        if (!std::isfinite(rel)) {
            throw std::runtime_error("selective gamma true residual became non-finite");
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

}  // namespace
