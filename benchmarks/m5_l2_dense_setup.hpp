#pragma once

// Reusable CPU/FP64 staging helpers for the frozen deep hierarchy. Include only
// after recursive_sa_local_l2_helpers.inc so CandidateTransfer, L1BlockMetric,
// LocalColumns and local_cross_gram are visible.

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace m5_l2_setup {

using Clock = std::chrono::steady_clock;

struct DenseA2 {
    std::size_t n{0U};
    std::vector<double> fp64;
    double symmetry_relative_defect{0.0};
    double assembly_ms{0.0};
};

inline DenseA2 assemble_dense_a2(
    const CandidateTransfer& transfer1,
    const L1BlockMetric& block1,
    const std::vector<LocalColumns>& l2_basis,
    const std::function<LocalColumns(const LocalColumns&)>& local_a1_apply) {
    const auto start = Clock::now();
    DenseA2 out;
    out.n = transfer1.coarse_dofs;
    if (out.n == 0U || l2_basis.size() != transfer1.aggregates.size()) {
        throw std::invalid_argument("M5 dense A2 support/layout mismatch");
    }
    out.fp64.assign(out.n * out.n, 0.0);
    for (std::size_t jnode = 0; jnode < l2_basis.size(); ++jnode) {
        const auto applied = local_a1_apply(l2_basis[jnode]);
        const std::size_t joff = transfer1.aggregates[jnode].coarse_offset;
        const std::size_t jrank = transfer1.aggregates[jnode].rank;
        for (std::size_t inode = 0; inode < l2_basis.size(); ++inode) {
            const auto block = local_cross_gram(l2_basis[inode], applied, block1);
            const std::size_t ioff = transfer1.aggregates[inode].coarse_offset;
            const std::size_t irank = transfer1.aggregates[inode].rank;
            for (std::size_t i = 0; i < irank; ++i) {
                for (std::size_t j = 0; j < jrank; ++j) {
                    out.fp64[(ioff + i) * out.n + (joff + j)] =
                        block[i * kCandidates + j];
                }
            }
        }
    }

    double asym2 = 0.0;
    double norm2 = 0.0;
    for (std::size_t i = 0; i < out.n; ++i) {
        const double d = out.fp64[i * out.n + i];
        if (!(d > 0.0) || !std::isfinite(d)) {
            throw std::runtime_error("M5 dense A2 diagonal invalid");
        }
        norm2 += d * d;
        for (std::size_t j = i + 1U; j < out.n; ++j) {
            const double aij = out.fp64[i * out.n + j];
            const double aji = out.fp64[j * out.n + i];
            const double diff = aij - aji;
            asym2 += 2.0 * diff * diff;
            norm2 += aij * aij + aji * aji;
            const double sym = 0.5 * (aij + aji);
            out.fp64[i * out.n + j] = sym;
            out.fp64[j * out.n + i] = sym;
        }
    }
    out.symmetry_relative_defect = norm2 > 0.0 ? std::sqrt(asym2 / norm2) : 0.0;
    out.assembly_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    return out;
}

inline std::vector<double> apply_dense_a2(const DenseA2& a,
                                          const std::vector<double>& x) {
    if (x.size() != a.n) throw std::invalid_argument("M5 dense A2 apply size mismatch");
    std::vector<double> y(a.n, 0.0);
    for (std::size_t i = 0; i < a.n; ++i) {
        const double* row = a.fp64.data() + i * a.n;
        for (std::size_t j = 0; j < a.n; ++j) y[i] += row[j] * x[j];
    }
    return y;
}

inline std::vector<float> inverse_blocks_6x6_fp32(const L1BlockMetric& metric) {
    if (metric.nodes() == 0U || metric.dofs() == 0U) {
        throw std::runtime_error("M5 inverse-block metric empty");
    }
    std::vector<float> inverse(metric.nodes() * 36U, 0.0f);
    for (std::size_t node = 0; node < metric.nodes(); ++node) {
        const std::size_t begin = metric.dof_offsets[node];
        const std::size_t rank = metric.dof_offsets[node + 1U] - begin;
        if (rank == 0U || rank > 6U) {
            throw std::runtime_error("M5 inverse-block rank invalid");
        }
        const double* l = metric.lower.data() + metric.value_offsets[node];
        float* dst = inverse.data() + node * 36U;
        for (std::size_t col = 0; col < rank; ++col) {
            std::array<double, 6U> y{};
            std::array<double, 6U> x{};
            for (std::size_t i = 0; i < rank; ++i) {
                double value = i == col ? 1.0 : 0.0;
                for (std::size_t j = 0; j < i; ++j) {
                    value -= l[i * rank + j] * y[j];
                }
                y[i] = value / l[i * rank + i];
            }
            for (std::size_t ii = rank; ii-- > 0U;) {
                double value = y[ii];
                for (std::size_t j = ii + 1U; j < rank; ++j) {
                    value -= l[j * rank + ii] * x[j];
                }
                x[ii] = value / l[ii * rank + ii];
            }
            for (std::size_t row = 0; row < rank; ++row) {
                dst[row * 6U + col] = static_cast<float>(x[row]);
            }
        }
    }
    return inverse;
}

struct DenseP2 {
    std::size_t rows{0U};
    std::size_t cols{0U};
    std::vector<double> fp64;
    double assembly_ms{0.0};
};

inline DenseP2 assemble_dense_p2(
    const CandidateTransfer& transfer2,
    const L1BlockMetric& block2,
    const std::vector<LocalColumns>& bottom_basis) {
    const auto start = Clock::now();
    DenseP2 out;
    out.rows = block2.dofs();
    out.cols = transfer2.coarse_dofs;
    if (out.rows == 0U || out.cols == 0U ||
        bottom_basis.size() != transfer2.aggregates.size()) {
        throw std::invalid_argument("M5 dense P2 support/layout mismatch");
    }
    out.fp64.assign(out.rows * out.cols, 0.0);
    for (std::size_t cnode = 0; cnode < bottom_basis.size(); ++cnode) {
        const auto& basis = bottom_basis[cnode];
        const std::size_t coff = transfer2.aggregates[cnode].coarse_offset;
        const std::size_t crank = transfer2.aggregates[cnode].rank;
        if (basis.cols != crank || crank == 0U || crank > 6U) {
            throw std::runtime_error("M5 dense P2 coarse rank mismatch");
        }
        for (const auto& entry : basis.values) {
            const std::size_t rnode = entry.first;
            if (rnode >= block2.nodes()) throw std::out_of_range("M5 dense P2 row node");
            const std::size_t roff = block2.dof_offsets[rnode];
            const std::size_t rrank = block2.dof_offsets[rnode + 1U] - roff;
            for (std::size_t r = 0; r < rrank; ++r) {
                for (std::size_t q = 0; q < crank; ++q) {
                    out.fp64[(roff + r) * out.cols + (coff + q)] =
                        entry.second[r * 6U + q];
                }
            }
        }
    }
    out.assembly_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
    return out;
}

inline std::vector<double> apply_p2(const DenseP2& p,
                                    const std::vector<double>& x) {
    if (x.size() != p.cols) throw std::invalid_argument("M5 dense P2 apply size mismatch");
    std::vector<double> y(p.rows, 0.0);
    for (std::size_t i = 0; i < p.rows; ++i) {
        for (std::size_t j = 0; j < p.cols; ++j) {
            y[i] += p.fp64[i * p.cols + j] * x[j];
        }
    }
    return y;
}

inline std::vector<double> apply_p2t(const DenseP2& p,
                                     const std::vector<double>& x) {
    if (x.size() != p.rows) throw std::invalid_argument("M5 dense P2T apply size mismatch");
    std::vector<double> y(p.cols, 0.0);
    for (std::size_t i = 0; i < p.rows; ++i) {
        for (std::size_t j = 0; j < p.cols; ++j) {
            y[j] += p.fp64[i * p.cols + j] * x[i];
        }
    }
    return y;
}

inline std::vector<float> to_float(const std::vector<double>& x) {
    std::vector<float> y(x.size(), 0.0f);
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i]);
    return y;
}

}  // namespace m5_l2_setup
