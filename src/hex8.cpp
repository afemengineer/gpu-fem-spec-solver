#include "gfss/hex8.hpp"

#include <cmath>
#include <stdexcept>

namespace gfss {
namespace {

constexpr std::array<std::array<double, 3>, 8> kNodeSigns{{
    {{-1.0, -1.0, -1.0}},
    {{+1.0, -1.0, -1.0}},
    {{+1.0, +1.0, -1.0}},
    {{-1.0, +1.0, -1.0}},
    {{-1.0, -1.0, +1.0}},
    {{+1.0, -1.0, +1.0}},
    {{+1.0, +1.0, +1.0}},
    {{-1.0, +1.0, +1.0}},
}};

Mat3 inverse_3x3(const Mat3& a, double& determinant) {
    determinant =
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
        a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
        a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if (!(determinant > 0.0)) {
        throw std::runtime_error("HEX8 element has non-positive Jacobian determinant");
    }

    const double inv_det = 1.0 / determinant;
    Mat3 inv{};
    inv[0][0] = +(a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_det;
    inv[0][1] = -(a[0][1] * a[2][2] - a[0][2] * a[2][1]) * inv_det;
    inv[0][2] = +(a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_det;
    inv[1][0] = -(a[1][0] * a[2][2] - a[1][2] * a[2][0]) * inv_det;
    inv[1][1] = +(a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_det;
    inv[1][2] = -(a[0][0] * a[1][2] - a[0][2] * a[1][0]) * inv_det;
    inv[2][0] = +(a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_det;
    inv[2][1] = -(a[0][0] * a[2][1] - a[0][1] * a[2][0]) * inv_det;
    inv[2][2] = +(a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_det;
    return inv;
}

using BMatrix = std::array<std::array<double, 24>, 6>;

BMatrix make_b_matrix(const std::array<Vec3, 8>& dshape_dx) {
    BMatrix b{};
    for (std::size_t node = 0; node < 8; ++node) {
        const std::size_t col = 3 * node;
        const double dx = dshape_dx[node][0];
        const double dy = dshape_dx[node][1];
        const double dz = dshape_dx[node][2];

        b[0][col + 0] = dx;
        b[1][col + 1] = dy;
        b[2][col + 2] = dz;

        b[3][col + 0] = dy;
        b[3][col + 1] = dx;

        b[4][col + 1] = dz;
        b[4][col + 2] = dy;

        b[5][col + 0] = dz;
        b[5][col + 2] = dx;
    }
    return b;
}

}  // namespace

Mat6 isotropic_elasticity_matrix(const Material& material) {
    const double e = material.young_modulus;
    const double nu = material.poisson_ratio;
    if (!(e > 0.0)) {
        throw std::invalid_argument("Young's modulus must be positive");
    }
    if (!(nu > -1.0 && nu < 0.5)) {
        throw std::invalid_argument("Poisson ratio must satisfy -1 < nu < 0.5");
    }

    const double lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu = e / (2.0 * (1.0 + nu));

    Mat6 d{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            d[i][j] = lambda;
        }
        d[i][i] += 2.0 * mu;
    }
    d[3][3] = mu;
    d[4][4] = mu;
    d[5][5] = mu;
    return d;
}

std::array<Hex8QuadraturePoint, 8> hex8_gauss_points_2x2x2() {
    const double a = 1.0 / std::sqrt(3.0);
    return {{
        {-a, -a, -a, 1.0},
        {+a, -a, -a, 1.0},
        {+a, +a, -a, 1.0},
        {-a, +a, -a, 1.0},
        {-a, -a, +a, 1.0},
        {+a, -a, +a, 1.0},
        {+a, +a, +a, 1.0},
        {-a, +a, +a, 1.0},
    }};
}

Hex8Kinematics hex8_kinematics(const Hex8Coordinates& coordinates,
                               double xi,
                               double eta,
                               double zeta) {
    Hex8Kinematics result{};

    for (std::size_t i = 0; i < 8; ++i) {
        const double sx = kNodeSigns[i][0];
        const double sy = kNodeSigns[i][1];
        const double sz = kNodeSigns[i][2];

        result.shape[i] = 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta);
        result.dshape_dnatural[i][0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);
        result.dshape_dnatural[i][1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta);
        result.dshape_dnatural[i][2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta);
    }

    for (std::size_t i = 0; i < 8; ++i) {
        for (int physical = 0; physical < 3; ++physical) {
            for (int natural = 0; natural < 3; ++natural) {
                result.jacobian[physical][natural] +=
                    coordinates[i][physical] * result.dshape_dnatural[i][natural];
            }
        }
    }

    double determinant = 0.0;
    const Mat3 inv_j = inverse_3x3(result.jacobian, determinant);
    result.det_jacobian = determinant;

    for (std::size_t i = 0; i < 8; ++i) {
        for (int physical = 0; physical < 3; ++physical) {
            for (int natural = 0; natural < 3; ++natural) {
                // grad_x N = J^{-T} grad_xi N.
                result.dshape_dx[i][physical] +=
                    inv_j[natural][physical] * result.dshape_dnatural[i][natural];
            }
        }
    }

    return result;
}

Hex8Matrix hex8_stiffness(const Hex8Coordinates& coordinates,
                          const Material& material) {
    const Mat6 d = isotropic_elasticity_matrix(material);
    Hex8Matrix k{};

    for (const auto& qp : hex8_gauss_points_2x2x2()) {
        const auto kin = hex8_kinematics(coordinates, qp.xi, qp.eta, qp.zeta);
        const BMatrix b = make_b_matrix(kin.dshape_dx);
        const double scale = kin.det_jacobian * qp.weight;

        std::array<std::array<double, 24>, 6> db{};
        for (int i = 0; i < 6; ++i) {
            for (int col = 0; col < 24; ++col) {
                for (int j = 0; j < 6; ++j) {
                    db[i][col] += d[i][j] * b[j][col];
                }
            }
        }

        for (int row = 0; row < 24; ++row) {
            for (int col = 0; col < 24; ++col) {
                for (int i = 0; i < 6; ++i) {
                    k[row][col] += b[i][row] * db[i][col] * scale;
                }
            }
        }
    }

    return k;
}

Hex8Vector hex8_apply(const Hex8Coordinates& coordinates,
                      const Material& material,
                      const Hex8Vector& x) {
    const Hex8Matrix k = hex8_stiffness(coordinates, material);
    Hex8Vector y{};
    for (int row = 0; row < 24; ++row) {
        for (int col = 0; col < 24; ++col) {
            y[row] += k[row][col] * x[col];
        }
    }
    return y;
}

double hex8_strain_energy(const Hex8Matrix& stiffness,
                          const Hex8Vector& x) {
    double value = 0.0;
    for (int row = 0; row < 24; ++row) {
        for (int col = 0; col < 24; ++col) {
            value += x[row] * stiffness[row][col] * x[col];
        }
    }
    return 0.5 * value;
}

}  // namespace gfss
