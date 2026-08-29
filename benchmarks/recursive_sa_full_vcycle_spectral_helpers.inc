// Reference-only full-V-cycle spectral/error-propagation helpers.
// Requires recursive_sa_local_l2_helpers.inc, recursive_sa_cr_oracle_helpers.inc,
// and recursive_sa_block_cr_spectral_helpers.inc.

namespace {

struct MgLevelTrace {
    std::string label;
    double initial_residual_norm{0.0};
    double after_pre_residual_norm{0.0};
    double coarse_rhs_norm{0.0};
    double coarse_correction_norm{0.0};
    double after_coarse_residual_norm{0.0};
    double after_post_residual_norm{0.0};
};

double mg_level_residual_norm(const gfss::ReferenceMultilevelLevel& level,
                              const Vec& b,
                              const Vec& x) {
    const auto ax = level.apply(x);
    if (ax.size() != b.size()) {
        throw std::runtime_error("full-MG trace apply size mismatch");
    }
    double rr = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        const double r = b[i] - ax[i];
        rr += r * r;
    }
    return std::sqrt(std::max(0.0, rr));
}

void mg_diagnostic_vcycle(std::size_t level_index,
                          const std::vector<gfss::ReferenceMultilevelLevel>& levels,
                          const Vec& b,
                          Vec& x,
                          std::size_t pre_smooth_steps,
                          std::size_t post_smooth_steps,
                          std::vector<MgLevelTrace>* trace) {
    const auto& level = levels[level_index];
    if (b.size() != level.dofs || x.size() != level.dofs) {
        throw std::runtime_error("full-MG diagnostic level size mismatch");
    }

    MgLevelTrace* t = nullptr;
    if (trace != nullptr) {
        if (trace->size() != levels.size()) trace->resize(levels.size());
        t = &(*trace)[level_index];
        t->label = level.label;
        t->initial_residual_norm = mg_level_residual_norm(level, b, x);
    }

    if (level_index + 1U == levels.size()) {
        x = level.bottom_solve(b);
        if (x.size() != level.dofs) {
            throw std::runtime_error("full-MG diagnostic bottom solve size mismatch");
        }
        if (t != nullptr) {
            t->after_pre_residual_norm = t->initial_residual_norm;
            t->after_coarse_residual_norm = t->initial_residual_norm;
            t->after_post_residual_norm = mg_level_residual_norm(level, b, x);
        }
        return;
    }

    level.smooth(b, x, pre_smooth_steps);
    if (t != nullptr) {
        t->after_pre_residual_norm = mg_level_residual_norm(level, b, x);
    }

    const auto ax = level.apply(x);
    Vec residual(level.dofs, 0.0);
    for (std::size_t i = 0; i < level.dofs; ++i) residual[i] = b[i] - ax[i];
    auto coarse_b = level.restrict_to_coarse(residual);
    const auto& coarse = levels[level_index + 1U];
    if (coarse_b.size() != coarse.dofs) {
        throw std::runtime_error("full-MG diagnostic restriction size mismatch");
    }
    if (t != nullptr) t->coarse_rhs_norm = norm(coarse_b);

    Vec coarse_e(coarse.dofs, 0.0);
    mg_diagnostic_vcycle(level_index + 1U,
                         levels,
                         coarse_b,
                         coarse_e,
                         pre_smooth_steps,
                         post_smooth_steps,
                         trace);
    if (t != nullptr) t->coarse_correction_norm = norm(coarse_e);

    const auto correction = level.prolong_from_coarse(coarse_e);
    if (correction.size() != level.dofs) {
        throw std::runtime_error("full-MG diagnostic prolongation size mismatch");
    }
    for (std::size_t i = 0; i < x.size(); ++i) x[i] += correction[i];
    if (t != nullptr) {
        t->after_coarse_residual_norm = mg_level_residual_norm(level, b, x);
    }

    level.smooth(b, x, post_smooth_steps);
    if (t != nullptr) {
        t->after_post_residual_norm = mg_level_residual_norm(level, b, x);
    }
}

Vec mg_apply_error_cycle(const std::vector<gfss::ReferenceMultilevelLevel>& levels,
                         const Vec& e,
                         std::size_t pre_smooth_steps,
                         std::size_t post_smooth_steps) {
    if (levels.empty() || e.size() != levels.front().dofs) {
        throw std::invalid_argument("full-MG error-cycle input mismatch");
    }
    Vec b(e.size(), 0.0);
    Vec x = e;
    mg_diagnostic_vcycle(0U, levels, b, x,
                         pre_smooth_steps, post_smooth_steps, nullptr);
    return x;
}

std::vector<MgLevelTrace> mg_trace_error_cycle(
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    const Vec& e,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps) {
    Vec b(e.size(), 0.0);
    Vec x = e;
    std::vector<MgLevelTrace> trace(levels.size());
    mg_diagnostic_vcycle(0U, levels, b, x,
                         pre_smooth_steps, post_smooth_steps, &trace);
    return trace;
}

struct FullMgSpectrum {
    std::vector<Vec> modes;
    std::vector<double> eigenvalue;
    std::vector<double> one_cycle_energy_q;
    std::vector<double> one_cycle_residual_q;
    std::vector<double> subspace_leakage;
    double final_a_orthogonality_error{0.0};
    double projected_operator_symmetry_defect{0.0};
};

FullMgSpectrum run_full_mg_spectrum(
    const std::vector<Vec>& seeds,
    const Apply& apply,
    const std::vector<gfss::ReferenceMultilevelLevel>& levels,
    std::size_t iterations,
    std::size_t pre_smooth_steps,
    std::size_t post_smooth_steps) {
    if (seeds.empty() || iterations == 0U) {
        throw std::invalid_argument("full-MG block spectrum requires seeds and iterations");
    }

    auto block = a_orthonormalize_block(seeds, apply);
    FullMgSpectrum result;
    result.subspace_leakage.reserve(iterations);

    for (std::size_t it = 0; it < iterations; ++it) {
        std::vector<Vec> next;
        next.reserve(block.q.size());
        for (const auto& v : block.q) {
            next.push_back(mg_apply_error_cycle(
                levels, v, pre_smooth_steps, post_smooth_steps));
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
        ev[j] = mg_apply_error_cycle(
            levels, block.q[j], pre_smooth_steps, post_smooth_steps);
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
        const auto emode = mg_apply_error_cycle(
            levels, mode, pre_smooth_steps, post_smooth_steps);
        const auto aemode = apply(emode);
        const double q_energy = oracle_energy_norm(apply, emode);
        const double q_residual = norm(aemode) / std::max(norm(amode), 1.0e-300);
        const double rayleigh = dot(mode, aemode);
        result.modes.push_back(std::move(mode));
        result.eigenvalue.push_back(rayleigh);
        result.one_cycle_energy_q.push_back(q_energy);
        result.one_cycle_residual_q.push_back(q_residual);
    }
    return result;
}

double mg_tail_geomean_q(const std::vector<double>& residuals,
                         std::size_t tail_count) {
    if (residuals.size() < 2U || tail_count == 0U) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t ratios = residuals.size() - 1U;
    const std::size_t count = std::min(tail_count, ratios);
    const std::size_t first = residuals.size() - count;
    double log_sum = 0.0;
    std::size_t used = 0U;
    for (std::size_t i = first; i < residuals.size(); ++i) {
        const double q = residuals[i] / residuals[i - 1U];
        if (q > 0.0 && std::isfinite(q)) {
            log_sum += std::log(q);
            ++used;
        }
    }
    return used > 0U
        ? std::exp(log_sum / static_cast<double>(used))
        : std::numeric_limits<double>::quiet_NaN();
}

double mg_abs_cosine(const Vec& a, const Vec& b) {
    const double denom = norm(a) * norm(b);
    return denom > 0.0 ? std::abs(dot(a, b)) / denom : 0.0;
}

struct MgPhysicalSummary {
    std::array<double, 3> component_fraction{};
    std::array<double, 3> weighted_centroid{};
    double bending_abs_cosine{0.0};
};

MgPhysicalSummary mg_summarize_fine_mode(
    const Vec& fine,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const Vec& bending_shape) {
    if (fine.size() != 3U * space.graph.coordinates.size()) {
        throw std::invalid_argument("full-MG physical mode size mismatch");
    }
    MgPhysicalSummary out;
    double total = 0.0;
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        const double ux = fine[3U * node + 0U];
        const double uy = fine[3U * node + 1U];
        const double uz = fine[3U * node + 2U];
        const std::array<double, 3> c2{ux * ux, uy * uy, uz * uz};
        const double amp2 = c2[0] + c2[1] + c2[2];
        total += amp2;
        for (std::size_t c = 0; c < 3U; ++c) {
            out.component_fraction[c] += c2[c];
            out.weighted_centroid[c] += amp2 * space.graph.coordinates[node][c];
        }
    }
    if (total > 0.0) {
        for (double& v : out.component_fraction) v /= total;
        for (double& v : out.weighted_centroid) v /= total;
    }
    out.bending_abs_cosine = mg_abs_cosine(fine, bending_shape);
    return out;
}

Vec mg_true_residual(const gfss::ReferenceMultilevelLevel& level,
                     const Vec& rhs,
                     const Vec& x) {
    const auto ax = level.apply(x);
    if (ax.size() != rhs.size()) {
        throw std::runtime_error("full-MG true residual size mismatch");
    }
    Vec r(rhs.size(), 0.0);
    for (std::size_t i = 0; i < rhs.size(); ++i) r[i] = rhs[i] - ax[i];
    return r;
}

}  // namespace
