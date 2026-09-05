#pragma once

#include "gfss/aggregation_coarse_space.hpp"

#include <functional>
#include <vector>

namespace gfss {

using AggregationFineOperator =
    std::function<std::vector<double>(const std::vector<double>&)>;

// Matrix-light Galerkin coarse action. The fine operator remains entirely
// matrix-free: A_c v = P^T [ A_f (P v) ]. No fine stiffness matrix, coarse CAD,
// or coarse finite-element mesh is required.
std::vector<double> apply_elasticity_aggregation_coarse_operator(
    const ElasticityAggregationCoarseSpace& space,
    const AggregationFineOperator& apply_fine,
    const std::vector<double>& coarse);

}  // namespace gfss
