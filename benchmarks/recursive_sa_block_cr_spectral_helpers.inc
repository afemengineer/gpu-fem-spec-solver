// Reference-only block compatible-relaxation / spectral-oracle helpers.
// Requires recursive_sa_local_l2_helpers.inc and recursive_sa_cr_oracle_helpers.inc.

namespace {

struct AOrthonormalBlock {
    std::vector<Vec> q;
    std::vector<Vec> aq;
    double orthogonality_error{0.0};
};

AOrthonormalBlock a_orthonormalize_block(std::vector<Vec> vectors,
                                         const Apply& apply) {
    AOrthonormalBlock result;
    result.q.reserve(vectors.size());
    result.aq.reserve(vectors.size());
    for (auto& v : vectors) {
        auto av = apply(v);
        for (std::size_t pass = 0; pass < 2U; ++pass) {
            for (std::size_t i = 0; i < result.q.size(); ++i) {
                const double coeff = dot(result.q[i], av);
                for (std::size_t d = 0; d < v.size(); ++d) {
                    v[d] -= coeff * result.q[i][d];
                    av[d] -= coeff * result.aq[i][d];
                }
            }
        }
        const double n2 = dot(v, av);
        if (!(n2 > 1.0e-24) || !std::isfinite(n2)) {
            throw std::runtime_error("block CR A-orthonormalization lost rank");
        }
        const double inv = 1.0 / std::sqrt(n2);
        for (double& x : v) x *= inv;
        for (double& x : av) x *= inv;
        result.q.push_back(std::move(v));
        result.aq.push_back(std::move(av));
    }

    double err2 = 0.0;
    for (std::size_t i = 0; i < result.q.size(); ++i) {
        for (std::size_t j = 0; j < result.q.size(); ++j) {
            const double gij = dot(result.q[i], result.aq[j]);
            const double target = i == j ? 1.0 : 0.0;
            const double e = gij - target;
            err2 += e * e;
        }
    }
    result.orthogonality_error = std::sqrt(err2);
    return result;
}

double block_subspace_leakage(const AOrthonormalBlock& old_block,
                              const AOrthonormalBlock& new_block) {
    if (old_block.q.size() != new_block.q.size()) {
        throw std::invalid_argument("block CR subspace size mismatch");
    }
    const std::size_t n = old_block.q.size();
    double overlap2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double c = dot(old_block.q[i], new_block.aq[j]);
            overlap2 += c * c;
        }
    }
    const double missing = std::max(0.0, static_cast<double>(n) - overlap2);
    return std::sqrt(missing / std::max(1.0, static_cast<double>(n)));
}

struct SymmetricEigenResult {
    std::vector<double> values;
    std::vector<double> vectors; // row-major; eigenvectors are columns
};

SymmetricEigenResult jacobi_symmetric_eigen(std::vector<double> a,
                                            std::size_t n) {
    if (a.size() != n * n || n == 0U) {
        throw std::invalid_argument("small symmetric eigensystem size mismatch");
    }
    SymmetricEigenResult result;
    result.vectors.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) result.vectors[i * n + i] = 1.0;

    const std::size_t max_sweeps = 100U * n * n;
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        std::size_t p = 0U;
        std::size_t q = 0U;
        double max_off = 0.0;
        double diag_scale = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            diag_scale = std::max(diag_scale, std::abs(a[i * n + i]));
            for (std::size_t j = i + 1U; j < n; ++j) {
                const double off = std::abs(a[i * n + j]);
                if (off > max_off) {
                    max_off = off;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off <= 1.0e-13 * std::max(1.0, diag_scale)) break;

        const double app = a[p * n + p];
        const double aqq = a[q * n + q];
        const double apq = a[p * n + q];
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi);
        const double s = std::sin(phi);

        for (std::size_t k = 0; k < n; ++k) {
            if (k == p || k == q) continue;
            const double akp = a[k * n + p];
            const double akq = a[k * n + q];
            const double new_kp = c * akp - s * akq;
            const double new_kq = s * akp + c * akq;
            a[k * n + p] = new_kp;
            a[p * n + k] = new_kp;
            a[k * n + q] = new_kq;
            a[q * n + k] = new_kq;
        }
        a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        for (std::size_t k = 0; k < n; ++k) {
            const double vkp = result.vectors[k * n + p];
            const double vkq = result.vectors[k * n + q];
            result.vectors[k * n + p] = c * vkp - s * vkq;
            result.vectors[k * n + q] = s * vkp + c * vkq;
        }
    }

    result.values.resize(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) result.values[i] = a[i * n + i];
    return result;
}

struct BlockCrSpectrum {
    std::vector<Vec> modes;
    std::vector<double> one_step_q;
    std::vector<double> projected_rayleigh_abs;
    std::vector<double> subspace_leakage;
    double final_a_orthogonality_error{0.0};
    double projected_operator_symmetry_defect{0.0};
};

template <class Transfer, class SmoothError>
BlockCrSpectrum run_block_cr_spectrum(const std::vector<Vec>& seeds,
                                      const Apply& apply,
                                      const Transfer& coarse_transfer,
                                      const DenseCholesky& coarse_factor,
                                      const SmoothError& smooth_error,
                                      std::size_t iterations) {
    if (seeds.empty() || iterations == 0U) {
        throw std::invalid_argument("block CR requires nonempty seeds and iterations");
    }
    std::vector<Vec> projected;
    projected.reserve(seeds.size());
    for (const auto& seed : seeds) {
        projected.push_back(oracle_project_a_complement(
            seed, apply, coarse_transfer, coarse_factor));
    }
    auto block = a_orthonormalize_block(std::move(projected), apply);

    BlockCrSpectrum result;
    result.subspace_leakage.reserve(iterations);
    for (std::size_t it = 0; it < iterations; ++it) {
        std::vector<Vec> next;
        next.reserve(block.q.size());
        for (const auto& v : block.q) {
            auto w = smooth_error(v);
            w = oracle_project_a_complement(
                w, apply, coarse_transfer, coarse_factor);
            next.push_back(std::move(w));
        }
        auto next_block = a_orthonormalize_block(std::move(next), apply);
        result.subspace_leakage.push_back(block_subspace_leakage(block, next_block));
        block = std::move(next_block);
    }
    result.final_a_orthogonality_error = block.orthogonality_error;

    const std::size_t m = block.q.size();
    std::vector<Vec> ev(m);
    std::vector<Vec> aev(m);
    for (std::size_t j = 0; j < m; ++j) {
        ev[j] = smooth_error(block.q[j]);
        ev[j] = oracle_project_a_complement(
            ev[j], apply, coarse_transfer, coarse_factor);
        aev[j] = apply(ev[j]);
    }

    std::vector<double> h(m * m, 0.0);
    std::vector<double> g(m * m, 0.0);
    double h_norm2 = 0.0;
    double h_asym2 = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            h[i * m + j] = dot(block.q[i], aev[j]);
            g[i * m + j] = dot(ev[i], aev[j]);
            h_norm2 += h[i * m + j] * h[i * m + j];
        }
    }
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = i + 1U; j < m; ++j) {
            const double d = h[i * m + j] - h[j * m + i];
            h_asym2 += 2.0 * d * d;
            const double sym = 0.5 * (g[i * m + j] + g[j * m + i]);
            g[i * m + j] = sym;
            g[j * m + i] = sym;
        }
    }
    result.projected_operator_symmetry_defect =
        h_norm2 > 0.0 ? std::sqrt(h_asym2 / h_norm2) : 0.0;

    const auto eig = jacobi_symmetric_eigen(g, m);
    std::vector<std::size_t> order(m, 0U);
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return eig.values[a] > eig.values[b];
    });

    result.modes.reserve(m);
    result.one_step_q.reserve(m);
    result.projected_rayleigh_abs.reserve(m);
    for (const auto idx : order) {
        Vec mode(block.q.front().size(), 0.0);
        for (std::size_t j = 0; j < m; ++j) {
            const double coeff = eig.vectors[j * m + idx];
            for (std::size_t d = 0; d < mode.size(); ++d) {
                mode[d] += coeff * block.q[j][d];
            }
        }
        oracle_normalize_energy(apply, mode);
        const auto emode0 = smooth_error(mode);
        const auto emode = oracle_project_a_complement(
            emode0, apply, coarse_transfer, coarse_factor);
        const auto aemode = apply(emode);
        const double q2 = std::max(0.0, dot(emode, aemode));
        const double rayleigh = std::abs(dot(mode, aemode));
        result.modes.push_back(std::move(mode));
        result.one_step_q.push_back(std::sqrt(q2));
        result.projected_rayleigh_abs.push_back(rayleigh);
    }
    return result;
}

struct MultiVectorEnrichedTransfer {
    const L1BlockSmoothedTransfer& base;
    const std::vector<Vec>& modes;
    std::size_t active_modes{0U};

    std::size_t coarse_dofs() const {
        return base.tentative.coarse_dofs + active_modes;
    }

    Vec prolong(const Vec& coarse) const {
        const std::size_t base_n = base.tentative.coarse_dofs;
        if (active_modes > modes.size() || coarse.size() != base_n + active_modes) {
            throw std::invalid_argument("multi-mode oracle prolongation size mismatch");
        }
        Vec base_coarse(base_n, 0.0);
        std::copy(coarse.begin(), coarse.begin() + static_cast<std::ptrdiff_t>(base_n),
                  base_coarse.begin());
        auto fine = base.prolong(base_coarse);
        for (std::size_t k = 0; k < active_modes; ++k) {
            if (modes[k].size() != fine.size()) {
                throw std::invalid_argument("multi-mode oracle basis size mismatch");
            }
            const double alpha = coarse[base_n + k];
            for (std::size_t i = 0; i < fine.size(); ++i) fine[i] += alpha * modes[k][i];
        }
        return fine;
    }

    Vec restrict_transpose(const Vec& fine) const {
        if (active_modes > modes.size()) {
            throw std::invalid_argument("multi-mode oracle active mode count invalid");
        }
        auto coarse = base.restrict_transpose(fine);
        for (std::size_t k = 0; k < active_modes; ++k) {
            if (modes[k].size() != fine.size()) {
                throw std::invalid_argument("multi-mode oracle restriction size mismatch");
            }
            coarse.push_back(dot(modes[k], fine));
        }
        return coarse;
    }
};

void spectral_factor_dense_bottom(LocalBottomReference& bottom) {
    const std::size_t n = bottom.factor.n;
    if (n == 0U || bottom.values.size() != n * n) {
        throw std::invalid_argument("spectral oracle bottom size mismatch");
    }
    bottom.factor.lower.assign(n * n, 0.0);
    bottom.factor.min_pivot = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = bottom.values[i * n + j];
            for (std::size_t k = 0; k < j; ++k) {
                value -= bottom.factor.lower[i * n + k] *
                         bottom.factor.lower[j * n + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("spectral oracle bottom Cholesky lost SPD");
                }
                bottom.factor.min_pivot = std::min(bottom.factor.min_pivot, value);
                bottom.factor.lower[i * n + j] = std::sqrt(value);
            } else {
                bottom.factor.lower[i * n + j] =
                    value / bottom.factor.lower[j * n + j];
            }
        }
    }
}

struct MultiAugmentedBottomResult {
    LocalBottomReference bottom;
    double max_base_span_energy_fraction{0.0};
    double max_mode_energy_diag_defect{0.0};
    double max_mode_energy_offdiag{0.0};
};

MultiAugmentedBottomResult build_multi_vector_augmented_bottom(
    const LocalBottomReference& base_bottom,
    const L1BlockSmoothedTransfer& base_transfer,
    const Apply& apply,
    const std::vector<Vec>& modes,
    std::size_t active_modes) {
    if (active_modes > modes.size()) {
        throw std::invalid_argument("spectral oracle active mode count exceeds basis");
    }
    const auto start = Clock::now();
    const std::size_t n = base_transfer.tentative.coarse_dofs;
    if (base_bottom.factor.n != n || base_bottom.values.size() != n * n) {
        throw std::invalid_argument("spectral oracle base bottom mismatch");
    }
    const std::size_t na = n + active_modes;
    MultiAugmentedBottomResult result;
    result.bottom.factor.n = na;
    result.bottom.values.assign(na * na, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result.bottom.values[i * na + j] = base_bottom.values[i * n + j];
        }
    }

    std::vector<Vec> amodes(active_modes);
    std::vector<std::vector<double>> cross(active_modes);
    for (std::size_t k = 0; k < active_modes; ++k) {
        amodes[k] = apply(modes[k]);
        cross[k] = base_transfer.restrict_transpose(amodes[k]);
        const double alpha = dot(modes[k], amodes[k]);
        if (!(alpha > 0.0) || !std::isfinite(alpha)) {
            throw std::runtime_error("spectral oracle mode energy invalid");
        }
        const auto solved = base_bottom.factor.solve(cross[k]);
        const double captured = dot(cross[k], solved);
        result.max_base_span_energy_fraction = std::max(
            result.max_base_span_energy_fraction, captured / alpha);
        result.max_mode_energy_diag_defect = std::max(
            result.max_mode_energy_diag_defect, std::abs(alpha - 1.0));
        for (std::size_t i = 0; i < n; ++i) {
            result.bottom.values[i * na + (n + k)] = cross[k][i];
            result.bottom.values[(n + k) * na + i] = cross[k][i];
        }
    }
    for (std::size_t i = 0; i < active_modes; ++i) {
        for (std::size_t j = 0; j < active_modes; ++j) {
            const double gij = dot(modes[i], amodes[j]);
            result.bottom.values[(n + i) * na + (n + j)] = gij;
            if (i != j) {
                result.max_mode_energy_offdiag = std::max(
                    result.max_mode_energy_offdiag, std::abs(gij));
            }
        }
    }

    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0; i < na; ++i) {
        norm2 += result.bottom.values[i * na + i] * result.bottom.values[i * na + i];
        for (std::size_t j = i + 1U; j < na; ++j) {
            const double aij = result.bottom.values[i * na + j];
            const double aji = result.bottom.values[j * na + i];
            const double d = aij - aji;
            asym2 += 2.0 * d * d;
            norm2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            result.bottom.values[i * na + j] = sym;
            result.bottom.values[j * na + i] = sym;
        }
    }
    result.bottom.factor.symmetry_relative_defect =
        norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;
    spectral_factor_dense_bottom(result.bottom);
    result.bottom.assembly_ms = elapsed_ms(start, Clock::now());
    return result;
}

template <class Transfer>
double spectral_transfer_adjoint_error(const Transfer& transfer,
                                       std::size_t fine_dofs,
                                       std::size_t coarse_dofs) {
    const auto fine = oracle_deterministic_cr_seed(fine_dofs, 0.73);
    const auto coarse = oracle_deterministic_cr_seed(coarse_dofs, 0.29);
    const auto pc = transfer.prolong(coarse);
    const auto ptf = transfer.restrict_transpose(fine);
    const double lhs = dot(pc, fine);
    const double rhs = dot(coarse, ptf);
    return std::abs(lhs - rhs) / std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

}  // namespace
