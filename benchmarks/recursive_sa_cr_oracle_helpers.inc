// Reference-only compatible-relaxation helpers for oracle enrichment tests.
// Requires recursive_sa_local_l2_helpers.inc and the recursive-SA common types.

namespace {

struct OracleCrProbeResult {
    std::string label;
    bool valid{false};
    std::vector<double> cycle_q;
    double post_transient_geomean_q{std::numeric_limits<double>::quiet_NaN()};
    double projection_coarse_residual_ratio{std::numeric_limits<double>::quiet_NaN()};
    double projection_idempotence_error{std::numeric_limits<double>::quiet_NaN()};
    Vec final_mode;
};

Vec oracle_deterministic_cr_seed(std::size_t n, double phase) {
    Vec v(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i + 1U);
        v[i] = std::sin(0.017 * t + phase)
             + 0.41 * std::cos(0.053 * t - 0.37 * phase)
             + 0.13 * std::sin(0.101 * t + 0.19 * phase);
    }
    return v;
}

Vec oracle_make_plate_bending_shape(
    const gfss::ElasticityAggregationCoarseSpace& space) {
    Vec fine(3U * space.graph.coordinates.size(), 0.0);
    constexpr double z_mid = 0.0625;
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        if (space.graph.constrained[node] != 0U) continue;
        const auto& p = space.graph.coordinates[node];
        const double x = p[0];
        const double z = p[2] - z_mid;
        fine[3U * node + 0U] = -2.0 * x * z;
        fine[3U * node + 2U] = x * x;
    }
    return fine;
}

double oracle_energy_norm(const Apply& apply, const Vec& x) {
    const auto ax = apply(x);
    const double e2 = dot(x, ax);
    if (!(e2 >= 0.0) || !std::isfinite(e2)) {
        throw std::runtime_error("oracle CR energy norm invalid");
    }
    return std::sqrt(std::max(0.0, e2));
}

void oracle_normalize_energy(const Apply& apply, Vec& x) {
    const double en = oracle_energy_norm(apply, x);
    if (!(en > 0.0) || !std::isfinite(en)) {
        throw std::runtime_error("oracle CR projected mode has zero/invalid energy");
    }
    for (double& v : x) v /= en;
}

template <class Transfer>
Vec oracle_project_a_complement(const Vec& x,
                                const Apply& apply,
                                const Transfer& transfer,
                                const DenseCholesky& coarse_factor) {
    const auto ax = apply(x);
    const auto coarse_residual = transfer.restrict_transpose(ax);
    const auto coarse_correction = coarse_factor.solve(coarse_residual);
    const auto represented = transfer.prolong(coarse_correction);
    if (represented.size() != x.size()) {
        throw std::runtime_error("oracle CR projection size mismatch");
    }
    Vec q = x;
    for (std::size_t i = 0; i < q.size(); ++i) q[i] -= represented[i];
    return q;
}

template <class Transfer>
std::pair<double, double> oracle_audit_projection(
    const Vec& seed,
    const Apply& apply,
    const Transfer& transfer,
    const DenseCholesky& coarse_factor) {
    const auto before = transfer.restrict_transpose(apply(seed));
    const auto q = oracle_project_a_complement(seed, apply, transfer, coarse_factor);
    const auto after = transfer.restrict_transpose(apply(q));
    const double residual_ratio = norm(after) / std::max(norm(before), 1.0e-300);
    const auto qq = oracle_project_a_complement(q, apply, transfer, coarse_factor);
    Vec diff(q.size(), 0.0);
    for (std::size_t i = 0; i < q.size(); ++i) diff[i] = qq[i] - q[i];
    const double idempotence = norm(diff) / std::max(norm(q), 1.0e-300);
    return {residual_ratio, idempotence};
}

template <class Transfer, class SmoothError>
OracleCrProbeResult oracle_run_cr_probe(
    const std::string& label,
    const Vec& seed,
    const Apply& apply,
    const Transfer& transfer,
    const DenseCholesky& coarse_factor,
    const SmoothError& smooth_error,
    std::size_t iterations) {
    OracleCrProbeResult result;
    result.label = label;
    if (iterations == 0U) return result;

    const auto audit = oracle_audit_projection(seed, apply, transfer, coarse_factor);
    result.projection_coarse_residual_ratio = audit.first;
    result.projection_idempotence_error = audit.second;

    auto e = oracle_project_a_complement(seed, apply, transfer, coarse_factor);
    const double initial_energy = oracle_energy_norm(apply, e);
    if (!(initial_energy > 1.0e-24) || !std::isfinite(initial_energy)) return result;
    oracle_normalize_energy(apply, e);
    result.valid = true;
    result.cycle_q.reserve(iterations);

    for (std::size_t it = 0; it < iterations; ++it) {
        auto next = smooth_error(e);
        next = oracle_project_a_complement(next, apply, transfer, coarse_factor);
        const double q = oracle_energy_norm(apply, next);
        if (!(q > 0.0) || !std::isfinite(q)) {
            throw std::runtime_error("oracle CR contraction factor invalid");
        }
        result.cycle_q.push_back(q);
        for (double& v : next) v /= q;
        e = std::move(next);
    }

    if (result.cycle_q.size() > 2U) {
        double log_sum = 0.0;
        std::size_t count = 0U;
        for (std::size_t i = 2U; i < result.cycle_q.size(); ++i) {
            const double q = result.cycle_q[i];
            if (q > 0.0 && std::isfinite(q)) {
                log_sum += std::log(q);
                ++count;
            }
        }
        if (count > 0U) {
            result.post_transient_geomean_q =
                std::exp(log_sum / static_cast<double>(count));
        }
    }
    result.final_mode = std::move(e);
    return result;
}

const OracleCrProbeResult* oracle_worst_valid_probe(
    const std::vector<OracleCrProbeResult>& probes) {
    const OracleCrProbeResult* worst = nullptr;
    for (const auto& probe : probes) {
        if (!probe.valid || !std::isfinite(probe.post_transient_geomean_q)) continue;
        if (worst == nullptr ||
            probe.post_transient_geomean_q > worst->post_transient_geomean_q) {
            worst = &probe;
        }
    }
    return worst;
}

}  // namespace
