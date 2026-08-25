#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_stencil.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

template <typename A, typename B>
double relative_max_difference(const A& a, const B& b) {
    require(a.size() == b.size(), "comparison vectors must have equal size");
    double max_abs = 0.0;
    double scale = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double av = static_cast<double>(a[i]);
        const double bv = static_cast<double>(b[i]);
        max_abs = std::max(max_abs, std::abs(av - bv));
        scale = std::max(scale, std::abs(av));
    }
    return max_abs / std::max(1.0, scale);
}

}  // namespace

int main() {
    const gfss::StructuredHexMesh mesh{4, 3, 2, 2.0, 1.3, 0.7};
    const gfss::Material material{73000.0, 0.29};

    std::vector<double> xd(static_cast<std::size_t>(mesh.dof_count()));
    std::vector<float> xf(xd.size());
    for (std::size_t i = 0; i < xd.size(); ++i) {
        const double value = 0.17 * std::sin(0.13 * static_cast<double>(i + 1)) +
                             0.04 * std::cos(0.07 * static_cast<double>(i + 5));
        xf[i] = static_cast<float>(value);
        xd[i] = static_cast<double>(xf[i]);
    }

    const auto oracle = gfss::apply_matrix_free(mesh, material, xd);

    const auto stencil64 = gfss::build_regular_node_stencil_fp64(mesh, material);
    std::vector<double> y64(xd.size(), 0.0);
    gfss::apply_node_stencil_openmp(mesh, stencil64, xd.data(), y64.data());
    const double fp64_rel = relative_max_difference(oracle, y64);
    require(fp64_rel < 2.0e-12,
            "FP64 node stencil must match trusted element-by-element operator");

    const auto stencil32 = gfss::build_regular_node_stencil_fp32(mesh, material);
    std::vector<float> y32(xf.size(), 0.0f);
    gfss::apply_node_stencil_openmp(mesh, stencil32, xf.data(), y32.data());
    const double fp32_rel = relative_max_difference(oracle, y32);
    require(fp32_rel < 2.0e-5,
            "FP32 node stencil must match FP64 trusted operator within tolerance");

    const auto first = y32;
    gfss::apply_node_stencil_openmp(mesh, stencil32, xf.data(), y32.data());
    require(std::memcmp(first.data(), y32.data(), y32.size() * sizeof(float)) == 0,
            "node-owned FP32 stencil must be bitwise repeatable");

    std::cout << "CPU node-stencil checks passed: fp64_rel=" << fp64_rel
              << " fp32_rel=" << fp32_rel << '\n';
    return 0;
}
