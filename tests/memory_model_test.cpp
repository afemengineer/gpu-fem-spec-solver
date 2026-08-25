#include <cassert>
#include <cmath>
#include <iostream>

#include "gfss/memory_model.hpp"

int main() {
    gfss::StructuredHexProblem p;
    p.nx = p.ny = p.nz = 10;

    const auto estimate = gfss::estimate_structured_matrix_free_memory(p);
    assert(p.element_count() == 1000);
    assert(p.node_count() == 1331);
    assert(p.dof_count() == 3993);

    // Six FP32 work vectors + one FP32 diagonal + one byte mask.
    assert(std::abs(estimate.bytes_per_dof() - 29.0) < 1.0e-12);
    assert(estimate.variable_bytes == p.dof_count() * 29ULL);
    assert(estimate.total_bytes() > estimate.variable_bytes);

    const auto n4 = gfss::max_cubic_mesh_for_budget(4ULL * 1024ULL * 1024ULL * 1024ULL);
    const auto n8 = gfss::max_cubic_mesh_for_budget(8ULL * 1024ULL * 1024ULL * 1024ULL);
    assert(n8 > n4);

    std::cout << "memory model tests passed\n";
    return 0;
}
