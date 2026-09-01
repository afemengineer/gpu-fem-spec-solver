#pragma once

#include "gfss/aggregation_coarse_space.hpp"
#include "gfss/cpu_elasticity.hpp"
#include "gfss/hex8.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace m5_l1_setup {

constexpr std::size_t kCandidates = 6U;
using Vec = std::vector<double>;
using NodeColumns = std::array<double, 18>;

struct ExactL1Blocks {
    std::vector<std::uint32_t> ranks;
    std::vector<double> blocks_6x6;
    std::vector<float> inverse_blocks_6x6_fp32;
    double min_cholesky_pivot{std::numeric_limits<double>::infinity()};

    std::size_t count() const noexcept { return ranks.size(); }
    std::size_t storage_bytes_fp32() const noexcept {
        return inverse_blocks_6x6_fp32.size() * sizeof(float);
    }
};

inline void clamp_x0(const gfss::StructuredHexMesh& mesh, Vec& v) {
    for (std::uint32_t k = 0; k <= mesh.nz; ++k) {
        for (std::uint32_t j = 0; j <= mesh.ny; ++j) {
            const auto node = static_cast<std::size_t>(mesh.node_index(0U, j, k));
            v[3U * node + 0U] = 0.0;
            v[3U * node + 1U] = 0.0;
            v[3U * node + 2U] = 0.0;
        }
    }
}

inline Vec apply_clamped(const gfss::StructuredHexMesh& mesh,
                         const gfss::Material& material,
                         const Vec& x) {
    auto free_x = x;
    clamp_x0(mesh, free_x);
    auto y = gfss::apply_matrix_free_openmp(mesh, material, free_x);
    clamp_x0(mesh, y);
    return y;
}

inline Vec build_fine_inverse_diagonal(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space) {
    Vec diagonal(static_cast<std::size_t>(mesh.dof_count()), 0.0);
    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    for (std::uint32_t ez = 0; ez < mesh.nz; ++ez) {
        for (std::uint32_t ey = 0; ey < mesh.ny; ++ey) {
            for (std::uint32_t ex = 0; ex < mesh.nx; ++ex) {
                const auto nodes = mesh.element_nodes(ex, ey, ez);
                for (std::size_t a = 0; a < 8U; ++a) {
                    const auto node = static_cast<std::size_t>(nodes[a]);
                    if (space.graph.constrained[node] != 0U) continue;
                    for (std::size_t c = 0; c < 3U; ++c) {
                        const std::size_t local = 3U * a + c;
                        diagonal[3U * node + c] += ke[local][local];
                    }
                }
            }
        }
    }
    Vec inverse(diagonal.size(), 0.0);
    for (std::size_t node = 0; node < space.graph.coordinates.size(); ++node) {
        if (space.graph.constrained[node] != 0U) continue;
        for (std::size_t c = 0; c < 3U; ++c) {
            const std::size_t dof = 3U * node + c;
            if (!(diagonal[dof] > 0.0) || !std::isfinite(diagonal[dof])) {
                throw std::runtime_error("M5 L1 setup fine diagonal invalid");
            }
            inverse[dof] = 1.0 / diagonal[dof];
        }
    }
    return inverse;
}

inline std::array<double, 18> local_rigid_rows(
    const std::array<double, 3>& xyz,
    const gfss::ElasticityAggregateInfo& aggregate) {
    const double scale = aggregate.coordinate_scale > 0.0
        ? aggregate.coordinate_scale : 1.0;
    const double x = (xyz[0] - aggregate.centroid[0]) / scale;
    const double y = (xyz[1] - aggregate.centroid[1]) / scale;
    const double z = (xyz[2] - aggregate.centroid[2]) / scale;
    return {
        1.0, 0.0, 0.0,  0.0,  z,   -y,
        0.0, 1.0, 0.0, -z,    0.0,  x,
        0.0, 0.0, 1.0,  y,   -x,    0.0,
    };
}

inline double local_tentative_basis_value(
    const gfss::ElasticityAggregationCoarseSpace& space,
    std::size_t aggregate_id,
    std::size_t node,
    std::size_t component,
    std::size_t q) {
    if (space.aggregate_of_node[node] != aggregate_id) return 0.0;
    const auto& aggregate = space.aggregates[aggregate_id];
    const auto rows = local_rigid_rows(space.graph.coordinates[node], aggregate);
    const double* row = rows.data() + 6U * component;
    const double* transform = aggregate.rigid_transform.data() + 6U * q;
    double value = 0.0;
    for (std::size_t j = 0; j < 6U; ++j) value += row[j] * transform[j];
    return value;
}

inline void append_node_touching_elements(
    const gfss::StructuredHexMesh& mesh,
    std::size_t node,
    std::vector<std::size_t>& elements) {
    const std::size_t sx = static_cast<std::size_t>(mesh.nx) + 1U;
    const std::size_t sy = static_cast<std::size_t>(mesh.ny) + 1U;
    const std::size_t i = node % sx;
    const std::size_t t = node / sx;
    const std::size_t j = t % sy;
    const std::size_t k = t / sy;
    const std::size_t ex0 = i == 0U ? 0U : i - 1U;
    const std::size_t ex1 = std::min<std::size_t>(i, mesh.nx - 1U);
    const std::size_t ey0 = j == 0U ? 0U : j - 1U;
    const std::size_t ey1 = std::min<std::size_t>(j, mesh.ny - 1U);
    const std::size_t ez0 = k == 0U ? 0U : k - 1U;
    const std::size_t ez1 = std::min<std::size_t>(k, mesh.nz - 1U);
    for (std::size_t ez = ez0; ez <= ez1; ++ez) {
        for (std::size_t ey = ey0; ey <= ey1; ++ey) {
            for (std::size_t ex = ex0; ex <= ex1; ++ex) {
                elements.push_back(ex + static_cast<std::size_t>(mesh.nx) *
                    (ey + static_cast<std::size_t>(mesh.ny) * ez));
            }
        }
    }
}

inline void sort_unique(std::vector<std::size_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

inline std::array<std::uint32_t, 3> decode_element(
    const gfss::StructuredHexMesh& mesh,
    std::size_t element) {
    const std::size_t ex = element % mesh.nx;
    const std::size_t t = element / mesh.nx;
    const std::size_t ey = t % mesh.ny;
    const std::size_t ez = t / mesh.ny;
    return {static_cast<std::uint32_t>(ex),
            static_cast<std::uint32_t>(ey),
            static_cast<std::uint32_t>(ez)};
}

inline std::array<double, 36> exact_smoothed_aggregate_block(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const std::vector<std::vector<std::uint32_t>>& aggregate_nodes,
    const Vec& fine_inverse,
    double omega0,
    std::size_t aggregate_id) {
    const auto& aggregate = space.aggregates[aggregate_id];
    const std::size_t rank = aggregate.rank;
    if (rank == 0U || rank > kCandidates) {
        throw std::runtime_error("M5 L1 setup aggregate rank invalid");
    }

    const auto ke = gfss::hex8_stiffness(mesh.element_coordinates(0U, 0U, 0U), material);
    std::vector<std::size_t> first_ring_elements;
    for (const auto node : aggregate_nodes[aggregate_id]) {
        append_node_touching_elements(mesh, node, first_ring_elements);
    }
    sort_unique(first_ring_elements);

    std::unordered_map<std::size_t, NodeColumns> aphi;
    aphi.reserve(first_ring_elements.size() * 4U + 16U);
    for (const auto element : first_ring_elements) {
        const auto ijk = decode_element(mesh, element);
        const auto nodes = mesh.element_nodes(ijk[0], ijk[1], ijk[2]);
        std::array<double, 24U * kCandidates> phi{};
        for (std::size_t a = 0; a < 8U; ++a) {
            const std::size_t node = static_cast<std::size_t>(nodes[a]);
            if (space.aggregate_of_node[node] != aggregate_id) continue;
            for (std::size_t c = 0; c < 3U; ++c) {
                const std::size_t ldof = 3U * a + c;
                for (std::size_t q = 0; q < rank; ++q) {
                    phi[ldof * kCandidates + q] =
                        local_tentative_basis_value(space, aggregate_id, node, c, q);
                }
            }
        }
        for (std::size_t i = 0; i < 24U; ++i) {
            const std::size_t node = static_cast<std::size_t>(nodes[i / 3U]);
            if (space.graph.constrained[node] != 0U) continue;
            auto& out = aphi[node];
            const std::size_t c = i % 3U;
            for (std::size_t q = 0; q < rank; ++q) {
                double value = 0.0;
                for (std::size_t j = 0; j < 24U; ++j) {
                    value += ke[i][j] * phi[j * kCandidates + q];
                }
                out[c * kCandidates + q] += value;
            }
        }
    }

    std::unordered_map<std::size_t, NodeColumns> psi;
    psi.reserve(aphi.size() + aggregate_nodes[aggregate_id].size() + 16U);
    for (const auto& entry : aphi) {
        auto& values = psi[entry.first];
        for (std::size_t c = 0; c < 3U; ++c) {
            const double inv = fine_inverse[3U * entry.first + c];
            for (std::size_t q = 0; q < rank; ++q) {
                values[c * kCandidates + q] =
                    -omega0 * inv * entry.second[c * kCandidates + q];
            }
        }
    }
    for (const auto node_u32 : aggregate_nodes[aggregate_id]) {
        const std::size_t node = node_u32;
        auto& values = psi[node];
        for (std::size_t c = 0; c < 3U; ++c) {
            for (std::size_t q = 0; q < rank; ++q) {
                values[c * kCandidates + q] +=
                    local_tentative_basis_value(space, aggregate_id, node, c, q);
            }
        }
    }

    std::vector<std::size_t> energy_elements;
    energy_elements.reserve(psi.size() * 8U);
    for (const auto& entry : psi) {
        append_node_touching_elements(mesh, entry.first, energy_elements);
    }
    sort_unique(energy_elements);

    std::array<double, 36> block{};
    for (const auto element : energy_elements) {
        const auto ijk = decode_element(mesh, element);
        const auto nodes = mesh.element_nodes(ijk[0], ijk[1], ijk[2]);
        std::array<double, 24U * kCandidates> local{};
        for (std::size_t a = 0; a < 8U; ++a) {
            const auto it = psi.find(static_cast<std::size_t>(nodes[a]));
            if (it == psi.end()) continue;
            for (std::size_t c = 0; c < 3U; ++c) {
                const std::size_t ldof = 3U * a + c;
                for (std::size_t q = 0; q < rank; ++q) {
                    local[ldof * kCandidates + q] = it->second[c * kCandidates + q];
                }
            }
        }
        std::array<double, 24U * kCandidates> klocal{};
        for (std::size_t i = 0; i < 24U; ++i) {
            for (std::size_t q = 0; q < rank; ++q) {
                double value = 0.0;
                for (std::size_t j = 0; j < 24U; ++j) {
                    value += ke[i][j] * local[j * kCandidates + q];
                }
                klocal[i * kCandidates + q] = value;
            }
        }
        for (std::size_t q = 0; q < rank; ++q) {
            for (std::size_t r = 0; r < rank; ++r) {
                double value = 0.0;
                for (std::size_t i = 0; i < 24U; ++i) {
                    value += local[i * kCandidates + q] * klocal[i * kCandidates + r];
                }
                block[6U * q + r] += value;
            }
        }
    }
    return block;
}

inline void invert_spd_block(const std::array<double, 36>& block,
                             std::size_t rank,
                             float* inverse_out,
                             double& min_pivot) {
    std::array<double, 36> lower{};
    for (std::size_t i = 0; i < rank; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = 0.5 * (block[6U * i + j] + block[6U * j + i]);
            for (std::size_t k = 0; k < j; ++k) {
                value -= lower[6U * i + k] * lower[6U * j + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("M5 L1 exact block is not SPD");
                }
                lower[6U * i + j] = std::sqrt(value);
                min_pivot = std::min(min_pivot, lower[6U * i + j]);
            } else {
                lower[6U * i + j] = value / lower[6U * j + j];
            }
        }
    }

    for (std::size_t col = 0; col < rank; ++col) {
        std::array<double, 6> y{};
        std::array<double, 6> x{};
        for (std::size_t i = 0; i < rank; ++i) {
            double value = i == col ? 1.0 : 0.0;
            for (std::size_t j = 0; j < i; ++j) {
                value -= lower[6U * i + j] * y[j];
            }
            y[i] = value / lower[6U * i + i];
        }
        for (std::size_t ii = rank; ii-- > 0U;) {
            double value = y[ii];
            for (std::size_t j = ii + 1U; j < rank; ++j) {
                value -= lower[6U * j + ii] * x[j];
            }
            x[ii] = value / lower[6U * ii + ii];
        }
        for (std::size_t row = 0; row < rank; ++row) {
            inverse_out[6U * row + col] = static_cast<float>(x[row]);
        }
    }
}

inline ExactL1Blocks build_exact_l1_blocks(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const Vec& fine_inverse,
    double omega0) {
    ExactL1Blocks result;
    result.ranks.resize(space.aggregates.size(), 0U);
    result.blocks_6x6.assign(space.aggregates.size() * 36U, 0.0);
    result.inverse_blocks_6x6_fp32.assign(space.aggregates.size() * 36U, 0.0f);

    std::vector<std::vector<std::uint32_t>> aggregate_nodes(space.aggregates.size());
    for (std::size_t node = 0; node < space.aggregate_of_node.size(); ++node) {
        const auto a = space.aggregate_of_node[node];
        if (a < space.aggregates.size()) {
            aggregate_nodes[a].push_back(static_cast<std::uint32_t>(node));
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (std::int64_t a64 = 0;
         a64 < static_cast<std::int64_t>(space.aggregates.size()); ++a64) {
        const std::size_t a = static_cast<std::size_t>(a64);
        const auto block = exact_smoothed_aggregate_block(
            mesh, material, space, aggregate_nodes, fine_inverse, omega0, a);
        const std::size_t rank = space.aggregates[a].rank;
        result.ranks[a] = static_cast<std::uint32_t>(rank);
        for (std::size_t q = 0; q < 36U; ++q) {
            result.blocks_6x6[36U * a + q] = block[q];
        }
        double local_min = std::numeric_limits<double>::infinity();
        invert_spd_block(block, rank,
                         result.inverse_blocks_6x6_fp32.data() + 36U * a,
                         local_min);
#ifdef _OPENMP
#pragma omp critical
#endif
        result.min_cholesky_pivot = std::min(result.min_cholesky_pivot, local_min);
    }
    return result;
}

inline Vec apply_factorized_a1(
    const gfss::StructuredHexMesh& mesh,
    const gfss::Material& material,
    const gfss::ElasticityAggregationCoarseSpace& space,
    const Vec& fine_inverse,
    double omega0,
    std::size_t transfer_steps,
    const Vec& coarse) {
    auto fine = gfss::apply_elasticity_tentative_prolongation(space, coarse);
    for (std::size_t step = 0; step < transfer_steps; ++step) {
        const auto af = apply_clamped(mesh, material, fine);
        for (std::size_t i = 0; i < fine.size(); ++i) {
            fine[i] -= omega0 * fine_inverse[i] * af[i];
        }
        clamp_x0(mesh, fine);
    }

    auto work = apply_clamped(mesh, material, fine);
    Vec scaled(work.size(), 0.0);
    for (std::size_t step = 0; step < transfer_steps; ++step) {
        for (std::size_t i = 0; i < work.size(); ++i) {
            scaled[i] = fine_inverse[i] * work[i];
        }
        const auto a_scaled = apply_clamped(mesh, material, scaled);
        for (std::size_t i = 0; i < work.size(); ++i) {
            work[i] -= omega0 * a_scaled[i];
        }
        clamp_x0(mesh, work);
    }
    return gfss::apply_elasticity_tentative_restriction(space, work);
}

inline Vec apply_block_matrix(const gfss::ElasticityAggregationCoarseSpace& space,
                              const ExactL1Blocks& blocks,
                              const Vec& x) {
    if (x.size() != space.coarse_dofs) {
        throw std::invalid_argument("M5 L1 block matrix vector size mismatch");
    }
    Vec y(x.size(), 0.0);
    for (std::size_t a = 0; a < space.aggregates.size(); ++a) {
        const std::size_t rank = space.aggregates[a].rank;
        const std::size_t offset = space.aggregates[a].coarse_offset;
        const double* block = blocks.blocks_6x6.data() + 36U * a;
        for (std::size_t i = 0; i < rank; ++i) {
            double value = 0.0;
            for (std::size_t j = 0; j < rank; ++j) {
                value += block[6U * i + j] * x[offset + j];
            }
            y[offset + i] = value;
        }
    }
    return y;
}

}  // namespace m5_l1_setup
