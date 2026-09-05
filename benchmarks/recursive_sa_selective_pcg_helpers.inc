// Reference-only selective-recursion multigrid preconditioner + FP64 PCG helpers.
// Requires recursive_sa_selective_gamma_helpers.inc and its dependencies.

namespace {

struct MgPcgProfile {
    std::array<std::size_t, 4> level_visits{};
    double pre_smooth_ms{0.0};
    double residual_apply_ms{0.0};
    double restrict_ms{0.0};
    double prolong_ms{0.0};
    double post_smooth_ms{0.0};
    double bottom_solve_ms{0.0};

    double accounted_ms() const {
        return pre_smooth_ms + residual_apply_ms + restrict_ms + prolong_ms +
               post_smooth_ms + bottom_solve_ms;
    }

    void add(const MgPcgProfile& other) {
        for (std::size_t i = 0; i < level_visits.size(); ++i) {
            level_visits[i] += other.level_visits[i];
        }
        pre_smooth_ms += other.pre_smooth_ms;
        residual_apply_ms += other.residual_apply_ms;
        restrict_ms += other.restrict_ms;
        prolong_ms += other.prolong_ms;
        post_smooth_ms += other.post_smooth_ms;
        bottom_solve_ms += other.bottom_solve_ms;
    }
};

void mg_selective_preconditioner_cycle_profiled(
    std::size_t level_index,
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& b,
    Vec& x,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    const MgSelectiveSchedule& repetitions,
    MgPcgProfile* profile) {
    mg_validate_selective_schedule(levels, repetitions);
    if (level_index >= levels.size()) {
        throw std::invalid_argument("selective PCG level index invalid");
    }
    const auto& level = levels[level_index];
    if (b.size() != level.dofs || x.size() != level.dofs) {
        throw std::runtime_error("selective PCG level vector size mismatch");
    }
    if (profile != nullptr) ++profile->level_visits[level_index];

    if (level_index + 1U == levels.size()) {
        const auto start = Clock::now();
        x = level.bottom_solve(b);
        if (profile != nullptr) profile->bottom_solve_ms += elapsed_ms(start, Clock::now());
        if (x.size() != level.dofs) {
            throw std::runtime_error("selective PCG bottom solve size mismatch");
        }
        return;
    }

    {
        const auto start = Clock::now();
        level.smooth(b, x, pre_smooth_steps);
        if (profile != nullptr) profile->pre_smooth_ms += elapsed_ms(start, Clock::now());
    }

    Vec residual(level.dofs, 0.0);
    {
        const auto start = Clock::now();
        const auto ax = level.apply(x);
        if (profile != nullptr) profile->residual_apply_ms += elapsed_ms(start, Clock::now());
        if (ax.size() != level.dofs) {
            throw std::runtime_error("selective PCG residual apply size mismatch");
        }
        for (std::size_t i = 0; i < level.dofs; ++i) residual[i] = b[i] - ax[i];
    }

    Vec coarse_b;
    {
        const auto start = Clock::now();
        coarse_b = level.restrict_to_coarse(residual);
        if (profile != nullptr) profile->restrict_ms += elapsed_ms(start, Clock::now());
    }
    const auto& coarse = levels[level_index + 1U];
    if (coarse_b.size() != coarse.dofs) {
        throw std::runtime_error("selective PCG restriction size mismatch");
    }

    Vec coarse_e(coarse.dofs, 0.0);
    const std::size_t recursive_repetitions = repetitions[level_index];
    for (std::size_t rep = 0; rep < recursive_repetitions; ++rep) {
        mg_selective_preconditioner_cycle_profiled(
            level_index + 1U,
            levels,
            coarse_b,
            coarse_e,
            pre_smooth_steps,
            post_smooth_steps,
            repetitions,
            profile);
    }

    Vec correction;
    {
        const auto start = Clock::now();
        correction = level.prolong_from_coarse(coarse_e);
        if (profile != nullptr) profile->prolong_ms += elapsed_ms(start, Clock::now());
    }
    if (correction.size() != level.dofs) {
        throw std::runtime_error("selective PCG prolongation size mismatch");
    }
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += correction[i];

    {
        const auto start = Clock::now();
        level.smooth(b, x, post_smooth_steps);
        if (profile != nullptr) profile->post_smooth_ms += elapsed_ms(start, Clock::now());
    }
}

Vec mg_apply_selective_preconditioner(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& rhs,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    const MgSelectiveSchedule& repetitions,
    MgPcgProfile* profile = nullptr) {
    if (levels.empty() || rhs.size() != levels.front().dofs) {
        throw std::invalid_argument("selective PCG preconditioner RHS mismatch");
    }
    Vec z(rhs.size(), 0.0);
    mg_selective_preconditioner_cycle_profiled(
        0U, levels, rhs, z,
        pre_smooth_steps, post_smooth_steps, repetitions, profile);
    return z;
}

struct MgPreconditionerAudit {
    double max_symmetry_relative_defect{0.0};
    double max_linearity_relative_error{0.0};
    double min_normalized_quadratic{std::numeric_limits<double>::infinity()};
    bool positive{true};
};

Vec mg_pcg_deterministic_probe(std::size_t n, double phase) {
    Vec v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = std::sin(0.013 * t + phase)
             + 0.37 * std::cos(0.047 * t - 0.31 * phase)
             + 0.19 * std::sin(0.089 * t + 0.23 * phase);
    }
    return v;
}

MgPreconditionerAudit mg_audit_selective_preconditioner(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    const MgSelectiveSchedule& repetitions,
    const std::function<void(Vec&)>& sanitize_probe) {
    MgPreconditionerAudit out;
    const std::array<double, 4> phases{0.17, 0.83, 1.71, 2.63};
    std::vector<Vec> probes;
    std::vector<Vec> images;
    probes.reserve(phases.size());
    images.reserve(phases.size());

    for (const double phase : phases) {
        auto v = mg_pcg_deterministic_probe(levels.front().dofs, phase);
        sanitize_probe(v);
        const double vn = norm(v);
        if (!(vn > 0.0)) throw std::runtime_error("selective PCG audit probe is zero");
        probes.push_back(v);
        images.push_back(mg_apply_selective_preconditioner(
            levels, probes.back(), pre_smooth_steps, post_smooth_steps,
            repetitions, nullptr));
        const double q = dot(probes.back(), images.back());
        const double normalized = q / std::max(dot(probes.back(), probes.back()), 1.0e-300);
        out.min_normalized_quadratic = std::min(out.min_normalized_quadratic, normalized);
        if (!(q > 0.0) || !std::isfinite(q)) out.positive = false;
    }

    for (std::size_t i = 0; i < probes.size(); ++i) {
        for (std::size_t j = i + 1U; j < probes.size(); ++j) {
            const double a = dot(probes[i], images[j]);
            const double b = dot(probes[j], images[i]);
            const double scale = std::max({std::abs(a), std::abs(b), 1.0e-300});
            out.max_symmetry_relative_defect = std::max(
                out.max_symmetry_relative_defect, std::abs(a - b) / scale);
        }
    }

    for (std::size_t i = 0; i + 1U < probes.size(); ++i) {
        Vec sum(probes[i].size(), 0.0);
        for (std::size_t d = 0; d < sum.size(); ++d) {
            sum[d] = probes[i][d] + probes[i + 1U][d];
        }
        const auto bsum = mg_apply_selective_preconditioner(
            levels, sum, pre_smooth_steps, post_smooth_steps, repetitions, nullptr);
        Vec reference(sum.size(), 0.0);
        for (std::size_t d = 0; d < sum.size(); ++d) {
            reference[d] = images[i][d] + images[i + 1U][d];
        }
        Vec diff(sum.size(), 0.0);
        for (std::size_t d = 0; d < sum.size(); ++d) diff[d] = bsum[d] - reference[d];
        out.max_linearity_relative_error = std::max(
            out.max_linearity_relative_error,
            norm(diff) / std::max(norm(reference), 1.0e-300));
    }
    return out;
}

struct MgPcgResult {
    Vec x;
    std::vector<double> true_relative_residuals;
    std::size_t iterations{0U};
    std::size_t preconditioner_applications{0U};
    std::size_t pcg_operator_applications{0U};
    std::size_t verification_operator_applications{0U};
    bool converged{false};
    bool breakdown{false};
    std::string breakdown_reason;
    double solve_ms{0.0};
    double preconditioner_ms{0.0};
    double verification_ms{0.0};
    MgPcgProfile preconditioner_profile;
};

MgPcgResult mg_solve_pcg_selective(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& rhs,
    double true_relative_tolerance,
    std::size_t max_iterations,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps,
    const MgSelectiveSchedule& repetitions) {
    if (levels.empty() || rhs.size() != levels.front().dofs ||
        !(true_relative_tolerance > 0.0) || true_relative_tolerance >= 1.0 ||
        max_iterations == 0U) {
        throw std::invalid_argument("selective PCG solve options invalid");
    }
    mg_validate_selective_schedule(levels, repetitions);

    MgPcgResult out;
    out.x.assign(rhs.size(), 0.0);
    const double bnorm = norm(rhs);
    if (!(bnorm > 0.0)) throw std::invalid_argument("selective PCG RHS is zero");

    Vec r = rhs;
    out.true_relative_residuals.push_back(1.0);

    auto apply_preconditioner = [&](const Vec& residual) {
        MgPcgProfile one;
        const auto start = Clock::now();
        auto z = mg_apply_selective_preconditioner(
            levels, residual, pre_smooth_steps, post_smooth_steps,
            repetitions, &one);
        out.preconditioner_ms += elapsed_ms(start, Clock::now());
        out.preconditioner_profile.add(one);
        ++out.preconditioner_applications;
        return z;
    };

    const auto solve_start = Clock::now();
    Vec z = apply_preconditioner(r);
    double rz = dot(r, z);
    if (!(rz > 0.0) || !std::isfinite(rz)) {
        out.breakdown = true;
        out.breakdown_reason = "initial_rz_nonpositive";
        out.solve_ms = elapsed_ms(solve_start, Clock::now());
        return out;
    }
    Vec p = z;

    for (std::size_t it = 0; it < max_iterations; ++it) {
        const auto ap = levels.front().apply(p);
        ++out.pcg_operator_applications;
        const double pap = dot(p, ap);
        if (!(pap > 0.0) || !std::isfinite(pap)) {
            out.breakdown = true;
            out.breakdown_reason = "pAp_nonpositive";
            break;
        }
        const double alpha = rz / pap;
        if (!std::isfinite(alpha)) {
            out.breakdown = true;
            out.breakdown_reason = "alpha_nonfinite";
            break;
        }
        for (std::size_t i = 0; i < out.x.size(); ++i) {
            out.x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        out.iterations = it + 1U;

        // Independent FP64 true residual is the correctness authority.
        const auto verify_start = Clock::now();
        const auto ax_true = levels.front().apply(out.x);
        ++out.verification_operator_applications;
        Vec r_true(rhs.size(), 0.0);
        for (std::size_t i = 0; i < rhs.size(); ++i) r_true[i] = rhs[i] - ax_true[i];
        const double rel_true = norm(r_true) / bnorm;
        out.verification_ms += elapsed_ms(verify_start, Clock::now());
        if (!std::isfinite(rel_true)) {
            out.breakdown = true;
            out.breakdown_reason = "true_residual_nonfinite";
            break;
        }
        out.true_relative_residuals.push_back(rel_true);
        if (rel_true <= true_relative_tolerance) {
            out.converged = true;
            break;
        }

        // Refresh the algorithmic residual from the exact true residual. In FP64
        // this deliberately trades one extra axpy-equivalent copy for zero drift.
        r = std::move(r_true);
        z = apply_preconditioner(r);
        const double rz_new = dot(r, z);
        if (!(rz_new > 0.0) || !std::isfinite(rz_new)) {
            out.breakdown = true;
            out.breakdown_reason = "rz_nonpositive";
            break;
        }
        const double beta = rz_new / rz;
        if (!(beta >= 0.0) || !std::isfinite(beta)) {
            out.breakdown = true;
            out.breakdown_reason = "beta_invalid";
            break;
        }
        for (std::size_t i = 0; i < p.size(); ++i) p[i] = z[i] + beta * p[i];
        rz = rz_new;
    }

    out.solve_ms = elapsed_ms(solve_start, Clock::now());
    return out;
}

}  // namespace
