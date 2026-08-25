#include "gfss/hex8.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

bool close(double a, double b, double atol = 1.0e-10, double rtol = 1.0e-10) {
    return std::abs(a - b) <= atol + rtol * std::max(std::abs(a), std::abs(b));
}

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

gfss::Hex8Coordinates unit_cube() {
    return {{
        {{0.0, 0.0, 0.0}},
        {{1.0, 0.0, 0.0}},
        {{1.0, 1.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}},
        {{1.0, 0.0, 1.0}},
        {{1.0, 1.0, 1.0}},
        {{0.0, 1.0, 1.0}},
    }};
}

std::array<double, 6> strain_from_kinematics(const gfss::Hex8Kinematics& kin,
                                              const gfss::Hex8Vector& u) {
    std::array<double, 6> strain{};
    for (std::size_t node = 0; node < 8; ++node) {
        const auto& g = kin.dshape_dx[node];
        const double ux = u[3 * node + 0];
        const double uy = u[3 * node + 1];
        const double uz = u[3 * node + 2];
        strain[0] += g[0] * ux;
        strain[1] += g[1] * uy;
        strain[2] += g[2] * uz;
        strain[3] += g[1] * ux + g[0] * uy;
        strain[4] += g[2] * uy + g[1] * uz;
        strain[5] += g[2] * ux + g[0] * uz;
    }
    return strain;
}

}  // namespace

int main() {
    const gfss::Material material{1000.0, 0.25};
    const auto coords = unit_cube();

    // Shape functions must form a partition of unity and reproduce a regular-cube Jacobian.
    const auto center = gfss::hex8_kinematics(coords, 0.0, 0.0, 0.0);
    double sum_n = 0.0;
    gfss::Vec3 sum_grad{{0.0, 0.0, 0.0}};
    for (std::size_t i = 0; i < 8; ++i) {
        sum_n += center.shape[i];
        for (int d = 0; d < 3; ++d) {
            sum_grad[d] += center.dshape_dx[i][d];
        }
    }
    require(close(sum_n, 1.0), "HEX8 shape functions must sum to one");
    require(close(center.det_jacobian, 0.125), "unit cube det(J) must be 1/8");
    require(close(sum_grad[0], 0.0) && close(sum_grad[1], 0.0) && close(sum_grad[2], 0.0),
            "shape gradients must sum to zero");

    const auto k = gfss::hex8_stiffness(coords, material);

    double max_asymmetry = 0.0;
    double max_entry = 0.0;
    for (int i = 0; i < 24; ++i) {
        for (int j = 0; j < 24; ++j) {
            max_asymmetry = std::max(max_asymmetry, std::abs(k[i][j] - k[j][i]));
            max_entry = std::max(max_entry, std::abs(k[i][j]));
        }
    }
    require(max_asymmetry <= 1.0e-12 * std::max(1.0, max_entry),
            "element stiffness must be symmetric");

    // A rigid translation must lie in the element nullspace.
    gfss::Hex8Vector rigid{};
    for (int node = 0; node < 8; ++node) {
        rigid[3 * node + 0] = 1.25;
        rigid[3 * node + 1] = -0.75;
        rigid[3 * node + 2] = 0.50;
    }
    require(std::abs(gfss::hex8_strain_energy(k, rigid)) < 1.0e-9,
            "rigid translation must have approximately zero strain energy");

    // A non-rigid displacement field must have positive energy.
    gfss::Hex8Vector deformation{};
    for (std::size_t node = 0; node < 8; ++node) {
        deformation[3 * node + 0] = 0.01 * coords[node][0];
        deformation[3 * node + 1] = -0.02 * coords[node][1];
        deformation[3 * node + 2] = 0.03 * coords[node][2];
    }
    require(gfss::hex8_strain_energy(k, deformation) > 0.0,
            "non-rigid deformation must have positive strain energy");

    // Constant-strain patch test: an affine displacement field must yield the exact constant strain
    // at every Gauss point of an undistorted trilinear HEX8 element.
    const std::array<std::array<double, 3>, 3> a{{
        {{0.011, -0.004, 0.006}},
        {{0.003, -0.008, 0.002}},
        {{-0.005, 0.007, 0.013}},
    }};
    const gfss::Vec3 c{{0.2, -0.1, 0.05}};

    gfss::Hex8Vector affine{};
    for (std::size_t node = 0; node < 8; ++node) {
        for (int component = 0; component < 3; ++component) {
            double value = c[component];
            for (int d = 0; d < 3; ++d) {
                value += a[component][d] * coords[node][d];
            }
            affine[3 * node + component] = value;
        }
    }

    const std::array<double, 6> expected{{
        a[0][0],
        a[1][1],
        a[2][2],
        a[0][1] + a[1][0],
        a[1][2] + a[2][1],
        a[0][2] + a[2][0],
    }};

    for (const auto& qp : gfss::hex8_gauss_points_2x2x2()) {
        const auto kin = gfss::hex8_kinematics(coords, qp.xi, qp.eta, qp.zeta);
        const auto strain = strain_from_kinematics(kin, affine);
        for (int component = 0; component < 6; ++component) {
            require(close(strain[component], expected[component], 1.0e-12, 1.0e-11),
                    "constant-strain patch test failed");
        }
    }

    // The convenience apply path must agree exactly with explicit K*x to roundoff.
    gfss::Hex8Vector x{};
    for (int i = 0; i < 24; ++i) {
        x[i] = 0.1 + 0.017 * static_cast<double>(i);
    }
    const auto y = gfss::hex8_apply(coords, material, x);
    for (int row = 0; row < 24; ++row) {
        double reference = 0.0;
        for (int col = 0; col < 24; ++col) {
            reference += k[row][col] * x[col];
        }
        require(close(y[row], reference, 1.0e-10, 1.0e-12), "hex8_apply must match K*x");
    }

    std::cout << "HEX8 reference checks passed\n";
    return 0;
}
