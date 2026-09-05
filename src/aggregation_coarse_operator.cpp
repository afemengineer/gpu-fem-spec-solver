#include "gfss/aggregation_coarse_operator.hpp"

#include <stdexcept>

namespace gfss {

std::vector<double> apply_elasticity_aggregation_coarse_operator(
    const ElasticityAggregationCoarseSpace& space,
    const AggregationFineOperator& apply_fine,
    const std::vector<double>& coarse) {
    if (!apply_fine) {
        throw std::invalid_argument("aggregation coarse operator missing fine apply callback");
    }
    if (coarse.size() != space.coarse_dofs) {
        throw std::invalid_argument("aggregation coarse operator coarse vector size mismatch");
    }

    const auto fine = apply_elasticity_tentative_prolongation(space, coarse);
    auto fine_ax = apply_fine(fine);
    if (fine_ax.size() != fine.size()) {
        throw std::runtime_error("aggregation fine operator returned wrong vector size");
    }
    return apply_elasticity_tentative_restriction(space, fine_ax);
}

}  // namespace gfss
