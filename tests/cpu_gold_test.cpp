#include "gfss/cpu_elasticity.hpp"
#include "gfss/cpu_gold.hpp"
#include "gfss/cpu_gold_padded.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

class AlignedFloatBuffer {
public:
    explicit AlignedFloatBuffer(std::size_t count)
        : storage_(count + 16U, 0.0f) {
        const auto raw = reinterpret_cast<std::uintptr_t>(storage_.data());
        const auto aligned =
            (raw + static_cast<std::uintptr_t>(63U)) &
            ~static_cast<std::uintptr_t>(63U);
        data_ = reinterpret_cast<float*>(aligned);
    }

    float* data() noexcept { return data_; }
    const float* data() const noexcept { return data_; }

private:
    std::vector<float> storage_;
    float* data_{nullptr};
};

template <typename A, typename B>
double relative_max_difference(const A& a, const B& b) {
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
    const gfss::StructuredHexMesh mesh{17, 10, 9, 1.7, 1.0, 0.9};
    const gfss::Material material{210.0e9, 0.30};
    const std::size_t nodes = static_cast<std::size_t>(mesh.node_count());
    const std::size_t ndof = static_cast<std::size_t>(mesh.dof_count());

    std::vector<float> x(ndof);
    std::vector<double> xd(ndof);
    for (std::size_t i = 0; i < ndof; ++i) {
        const float value = static_cast<float>(
            0.07 * std::sin(0.013 * static_cast<double>(i + 1)) +
            0.02 * std::cos(0.021 * static_cast<double>(i + 7)));
        x[i] = value;
        xd[i] = static_cast<double>(value);
    }

    const auto gold_stencil = gfss::build_cpu_gold_stencil_fp32(mesh, material);
    std::vector<float> ux(nodes), uy(nodes), uz(nodes);
    std::vector<float> yx(nodes), yy(nodes), yz(nodes);
    std::vector<float> y_gold(ndof);
    gfss::aos_to_soa_fp32(x.data(), nodes, ux.data(), uy.data(), uz.data());

    gfss::apply_cpu_gold_soa_fp32(mesh, gold_stencil,
                                  ux.data(), uy.data(), uz.data(),
                                  yx.data(), yy.data(), yz.data());
    gfss::soa_to_aos_fp32(yx.data(), yy.data(), yz.data(), nodes, y_gold.data());

    const auto padded_layout = gfss::make_cpu_gold_padded_layout_fp32(mesh);
    const auto padded_stencil =
        gfss::build_cpu_gold_padded_stencil_fp32(mesh, material, padded_layout);
    AlignedFloatBuffer pux(padded_layout.storage_nodes);
    AlignedFloatBuffer puy(padded_layout.storage_nodes);
    AlignedFloatBuffer puz(padded_layout.storage_nodes);
    AlignedFloatBuffer pyx(padded_layout.storage_nodes);
    AlignedFloatBuffer pyy(padded_layout.storage_nodes);
    AlignedFloatBuffer pyz(padded_layout.storage_nodes);
    std::vector<float> y_padded(ndof);
    gfss::aos_to_padded_soa_fp32(
        mesh, padded_layout, x.data(), pux.data(), puy.data(), puz.data());
    gfss::apply_cpu_gold_padded_soa_fp32(
        mesh, padded_stencil,
        pux.data(), puy.data(), puz.data(),
        pyx.data(), pyy.data(), pyz.data());
    gfss::padded_soa_to_aos_fp32(
        mesh, padded_layout,
        pyx.data(), pyy.data(), pyz.data(), y_padded.data());

    const auto oracle = gfss::apply_matrix_free_openmp(mesh, material, xd);
    const double rel = relative_max_difference(oracle, y_gold);
    const double padded_rel = relative_max_difference(oracle, y_padded);
    if (rel >= 2.0e-5 || padded_rel >= 2.0e-5) {
        std::cerr << "CPU Gold differs from FP64 oracle: rel_max=" << rel
                  << " padded_rel_max=" << padded_rel << '\n';
        return 1;
    }

    const auto first_yx = yx;
    const auto first_yy = yy;
    const auto first_yz = yz;
    gfss::apply_cpu_gold_soa_fp32(mesh, gold_stencil,
                                  ux.data(), uy.data(), uz.data(),
                                  yx.data(), yy.data(), yz.data());
    if (yx != first_yx || yy != first_yy || yz != first_yz) {
        std::cerr << "CPU Gold result is not bitwise repeatable\n";
        return 2;
    }

    std::vector<float> first_padded(ndof);
    gfss::padded_soa_to_aos_fp32(
        mesh, padded_layout,
        pyx.data(), pyy.data(), pyz.data(), first_padded.data());
    gfss::apply_cpu_gold_padded_soa_fp32(
        mesh, padded_stencil,
        pux.data(), puy.data(), puz.data(),
        pyx.data(), pyy.data(), pyz.data());
    gfss::padded_soa_to_aos_fp32(
        mesh, padded_layout,
        pyx.data(), pyy.data(), pyz.data(), y_padded.data());
    if (y_padded != first_padded) {
        std::cerr << "padded CPU Gold result is not bitwise repeatable\n";
        return 3;
    }

    std::cout << "CPU Gold SoA operator check passed; avx2="
              << (gfss::cpu_gold_avx2_enabled() ? "on" : "off")
              << " rel_max=" << rel
              << " padded_rel_max=" << padded_rel << '\n';
    return 0;
}
