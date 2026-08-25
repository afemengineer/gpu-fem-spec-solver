#pragma once

#include <array>
#include <cstddef>

namespace gfss {

struct Material {
    double young_modulus{210.0e9};
    double poisson_ratio{0.30};
};

using Vec3 = std::array<double, 3>;
using Mat3 = std::array<std::array<double, 3>, 3>;
using Mat6 = std::array<std::array<double, 6>, 6>;
using Hex8Coordinates = std::array<Vec3, 8>;
using Hex8Vector = std::array<double, 24>;
using Hex8Matrix = std::array<std::array<double, 24>, 24>;

struct Hex8QuadraturePoint {
    double xi;
    double eta;
    double zeta;
    double weight;
};

struct Hex8Kinematics {
    std::array<double, 8> shape{};
    std::array<Vec3, 8> dshape_dnatural{};
    std::array<Vec3, 8> dshape_dx{};
    Mat3 jacobian{};
    double det_jacobian{0.0};
};

Mat6 isotropic_elasticity_matrix(const Material& material);
std::array<Hex8QuadraturePoint, 8> hex8_gauss_points_2x2x2();
Hex8Kinematics hex8_kinematics(const Hex8Coordinates& coordinates,
                               double xi,
                               double eta,
                               double zeta);
Hex8Matrix hex8_stiffness(const Hex8Coordinates& coordinates,
                          const Material& material);
Hex8Vector hex8_apply(const Hex8Coordinates& coordinates,
                      const Material& material,
                      const Hex8Vector& x);
double hex8_strain_energy(const Hex8Matrix& stiffness,
                          const Hex8Vector& x);

}  // namespace gfss
