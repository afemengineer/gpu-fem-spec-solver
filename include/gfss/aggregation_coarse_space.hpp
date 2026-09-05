#pragma once

#include "gfss/structured_hex_mesh.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfss {

// Minimal mesh-independent nodal graph needed by the aggregation prototype.
// Production importers can build this directly from arbitrary element
// connectivity; no CAD or valid coarse finite-element mesh is required.
struct NodalGraph3D {
    std::vector<std::array<double, 3>> coordinates;
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
    std::vector<std::uint8_t> constrained;
};

struct ElasticityAggregationOptions {
    std::size_t target_nodes_per_aggregate{12};
    std::size_t min_nodes_per_aggregate{4};
    double rank_tolerance{1.0e-10};
};

struct ElasticityAggregateInfo {
    std::size_t node_count{0};
    std::size_t coarse_offset{0};
    std::size_t rank{0};
    std::array<double, 3> centroid{};
    double coordinate_scale{1.0};

    // Column-major 6x6 transform. The first rank columns map the six raw
    // rigid-body modes [Tx,Ty,Tz,Rx,Ry,Rz] to an orthonormal local tentative
    // prolongation basis. Unused columns are zero.
    std::array<double, 36> rigid_transform{};
};

struct ElasticityAggregationCoarseSpace {
    NodalGraph3D graph;
    std::vector<std::uint32_t> aggregate_of_node;
    std::vector<ElasticityAggregateInfo> aggregates;

    std::size_t free_nodes{0};
    std::size_t fine_free_dofs{0};
    std::size_t coarse_dofs{0};
    std::size_t tentative_p_nnz{0};

    // Logical production payload estimate for a matrix-free tentative P:
    // one aggregate id per node plus per-aggregate centroid/scale/transform/
    // rank/offset metadata in FP32. Existing mesh coordinates/connectivity are
    // deliberately excluded because the FEM model already owns them.
    std::size_t estimated_matrix_free_transfer_payload_bytes{0};
};

// Convenience adapter for current structured experiments. It builds the same
// nodal coupling graph that an orphan-mesh importer could provide: each HEX8
// node is adjacent to all nodes sharing an element (the 26-neighbor stencil).
NodalGraph3D build_structured_hex_nodal_graph_x0(
    const StructuredHexMesh& mesh);

// Greedy graph aggregation followed by a rigid-body tentative coarse basis.
// Each aggregate contributes up to six coarse DOFs. The local rigid-body
// columns are orthonormalized using only a 6x6 Gram matrix, so P need not be
// stored explicitly.
//
// Scope boundary: this constructs the tentative coarse space only. It does not
// yet build or apply A_c = P^T A P, smooth the tentative interpolation, recurse
// aggregation to additional levels, or choose strength-of-connection weights.
ElasticityAggregationCoarseSpace build_elasticity_aggregation_coarse_space(
    NodalGraph3D graph,
    const ElasticityAggregationOptions& options = {});

// Matrix-free tentative transfer operators. Fine vectors are 3 DOFs/node in
// AoS order. R is exactly P^T for this reference representation.
std::vector<double> apply_elasticity_tentative_prolongation(
    const ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& coarse);

std::vector<double> apply_elasticity_tentative_restriction(
    const ElasticityAggregationCoarseSpace& space,
    const std::vector<double>& fine);

// Projects the six global 3D rigid-body modes through P P^T and returns the
// worst relative reproduction error over non-zero modes. This is a direct
// audit that the tentative coarse space contains the elasticity near-nullspace
// without requiring a coarse CAD mesh.
double audit_elasticity_rigid_body_reproduction(
    const ElasticityAggregationCoarseSpace& space);

}  // namespace gfss
