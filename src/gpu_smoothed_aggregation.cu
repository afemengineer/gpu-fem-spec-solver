// Reuse the exact persistent-PCG GoldSparse and Jacobi primitives in this TU.
// The public PCG entry point is renamed so this source contributes only the
// smoothed-aggregation API while retaining access to the internal launchers.
#define solve_pcg_cuda_gold_sparse_x0 solve_pcg_cuda_gold_sparse_x0_sa_internal_unused
#include "gpu_pcg.cu"
#undef solve_pcg_cuda_gold_sparse_x0

#include "gfss/gpu_smoothed_aggregation.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gfss {
namespace {

struct DeviceAggregateSa {
    std::uint32_t coarse_offset{0};
    std::uint32_t rank{0};
    float centroid[3]{};
    float inv_scale{1.0f};
    float transform[36]{};
};

static_assert(sizeof(DeviceAggregateSa) == 168,
              "unexpected smoothed-aggregation aggregate layout");

enum class TimedStage {
    P0,
    FineA,
    Jacobi,
    Update,
    P0T,
};

__device__ __forceinline__ void aggregate_basis_values(
    const DeviceAggregateSa* aggregate,
    float x,
    float y,
    float z,
    std::uint32_t q,
    float& bx,
    float& by,
    float& bz) {
    const float* t = aggregate->transform + 6U * q;
    bx = t[0] + z * t[4] - y * t[5];
    by = t[1] - z * t[3] + x * t[5];
    bz = t[2] + y * t[3] - x * t[4];
}

// One warp owns one aggregate. The aggregate metadata and coarse coefficients
// are shared by all of its nodes, and every fine node has exactly one owner.
// This turns tentative prolongation from node->aggregate random lookup into a
// compact aggregate->nodes traversal with direct, non-atomic fine writes.
__global__ void tentative_prolongation_aggregate_sa_kernel(
    std::size_t nodes,
    const std::uint32_t* __restrict__ aggregate_offsets,
    const std::uint32_t* __restrict__ aggregate_nodes,
    const DeviceAggregateSa* __restrict__ aggregates,
    const float* __restrict__ coordinates,
    const float* __restrict__ coarse,
    float* __restrict__ fine) {
    const std::uint32_t aggregate_id = blockIdx.x;
    const DeviceAggregateSa* aggregate = aggregates + aggregate_id;
    const std::uint32_t first = aggregate_offsets[aggregate_id];
    const std::uint32_t last = aggregate_offsets[aggregate_id + 1U];

    __shared__ float coarse_local[6];
    if (threadIdx.x < aggregate->rank) {
        coarse_local[threadIdx.x] = coarse[aggregate->coarse_offset + threadIdx.x];
    }
    __syncwarp();

    for (std::uint32_t p = first + threadIdx.x; p < last; p += blockDim.x) {
        const std::size_t node = aggregate_nodes[p];
        const float x = (coordinates[node] - aggregate->centroid[0]) * aggregate->inv_scale;
        const float y = (coordinates[nodes + node] - aggregate->centroid[1]) * aggregate->inv_scale;
        const float z = (coordinates[2U * nodes + node] - aggregate->centroid[2]) * aggregate->inv_scale;

        float ux = 0.0f;
        float uy = 0.0f;
        float uz = 0.0f;
        for (std::uint32_t q = 0; q < aggregate->rank; ++q) {
            float bx = 0.0f;
            float by = 0.0f;
            float bz = 0.0f;
            aggregate_basis_values(aggregate, x, y, z, q, bx, by, bz);
            const float c = coarse_local[q];
            ux = fmaf(bx, c, ux);
            uy = fmaf(by, c, uy);
            uz = fmaf(bz, c, uz);
        }
        fine[node] = ux;
        fine[nodes + node] = uy;
        fine[2U * nodes + node] = uz;
    }
}

// Exact transpose of the aggregate-owned P0 kernel. Every block owns every
// coarse DOF of one aggregate, so node contributions can be reduced inside a
// warp and written once. No global atomicAdd and no coarse-vector memset are
// required.
__global__ void tentative_restriction_aggregate_sa_kernel(
    std::size_t nodes,
    const std::uint32_t* __restrict__ aggregate_offsets,
    const std::uint32_t* __restrict__ aggregate_nodes,
    const DeviceAggregateSa* __restrict__ aggregates,
    const float* __restrict__ coordinates,
    const float* __restrict__ fine,
    float* __restrict__ coarse) {
    const std::uint32_t aggregate_id = blockIdx.x;
    const DeviceAggregateSa* aggregate = aggregates + aggregate_id;
    const std::uint32_t first = aggregate_offsets[aggregate_id];
    const std::uint32_t last = aggregate_offsets[aggregate_id + 1U];

    float sums[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (std::uint32_t p = first + threadIdx.x; p < last; p += blockDim.x) {
        const std::size_t node = aggregate_nodes[p];
        const float x = (coordinates[node] - aggregate->centroid[0]) * aggregate->inv_scale;
        const float y = (coordinates[nodes + node] - aggregate->centroid[1]) * aggregate->inv_scale;
        const float z = (coordinates[2U * nodes + node] - aggregate->centroid[2]) * aggregate->inv_scale;
        const float fx = fine[node];
        const float fy = fine[nodes + node];
        const float fz = fine[2U * nodes + node];

        for (std::uint32_t q = 0; q < aggregate->rank; ++q) {
            float bx = 0.0f;
            float by = 0.0f;
            float bz = 0.0f;
            aggregate_basis_values(aggregate, x, y, z, q, bx, by, bz);
            sums[q] = fmaf(bx, fx, sums[q]);
            sums[q] = fmaf(by, fy, sums[q]);
            sums[q] = fmaf(bz, fz, sums[q]);
        }
    }

    constexpr unsigned int mask = 0xffffffffU;
    for (int delta = 16; delta > 0; delta >>= 1) {
#pragma unroll
        for (int q = 0; q < 6; ++q) {
            sums[q] += __shfl_down_sync(mask, sums[q], delta);
        }
    }

    if (threadIdx.x == 0U) {
        for (std::uint32_t q = 0; q < aggregate->rank; ++q) {
            coarse[aggregate->coarse_offset + q] = sums[q];
        }
    }
}

__global__ void axpy_negative_sa_kernel(
    std::size_t n,
    float omega,
    const float* __restrict__ correction,
    float* __restrict__ state) {
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        state[i] = fmaf(-omega, correction[i], state[i]);
    }
}

void launch_p0(const StructuredHexMesh& mesh,
               std::size_t nodes,
               std::uint32_t aggregate_count,
               const std::uint32_t* aggregate_offsets,
               const std::uint32_t* aggregate_nodes,
               const DeviceAggregateSa* aggregates,
               const float* coordinates,
               const float* coarse,
               float* fine) {
    constexpr unsigned int threads = 32U;
    tentative_prolongation_aggregate_sa_kernel<<<aggregate_count, threads>>>(
        nodes, aggregate_offsets, aggregate_nodes, aggregates,
        coordinates, coarse, fine);
    check_cuda_pcg(cudaGetLastError(), "smoothed aggregation aggregate P0 launch");

    // Aggregates contain only free nodes. Preserve exact x=0 Dirichlet values
    // without a full fine-vector memset.
    const std::uint32_t face_nodes = (mesh.ny + 1U) * (mesh.nz + 1U);
    constexpr unsigned int clamp_threads = 256U;
    const unsigned int clamp_blocks = (face_nodes + clamp_threads - 1U) / clamp_threads;
    zero_x0_face_pcg_kernel<<<clamp_blocks, clamp_threads>>>(
        mesh.nx, mesh.ny, mesh.nz, nodes, fine);
    check_cuda_pcg(cudaGetLastError(), "smoothed aggregation P0 clamp launch");
}

void launch_p0t(std::size_t nodes,
                std::uint32_t aggregate_count,
                const std::uint32_t* aggregate_offsets,
                const std::uint32_t* aggregate_nodes,
                const DeviceAggregateSa* aggregates,
                const float* coordinates,
                const float* fine,
                float* coarse) {
    constexpr unsigned int threads = 32U;
    tentative_restriction_aggregate_sa_kernel<<<aggregate_count, threads>>>(
        nodes, aggregate_offsets, aggregate_nodes, aggregates,
        coordinates, fine, coarse);
    check_cuda_pcg(cudaGetLastError(), "smoothed aggregation aggregate P0T launch");
}

void launch_negative_axpy(std::size_t n,
                          float omega,
                          const float* correction,
                          float* state) {
    constexpr unsigned int threads = 256U;
    const auto blocks = static_cast<unsigned int>((n + threads - 1U) / threads);
    axpy_negative_sa_kernel<<<blocks, threads>>>(n, omega, correction, state);
    check_cuda_pcg(cudaGetLastError(), "smoothed aggregation update launch");
}

GpuSmoothedAggregationTiming summarize_fieldwise_min(
    const std::vector<GpuSmoothedAggregationTiming>& samples) {
    GpuSmoothedAggregationTiming out;
    out.p0_ms = std::numeric_limits<double>::infinity();
    out.fine_operator_ms = std::numeric_limits<double>::infinity();
    out.jacobi_ms = std::numeric_limits<double>::infinity();
    out.vector_update_ms = std::numeric_limits<double>::infinity();
    out.p0t_ms = std::numeric_limits<double>::infinity();
    out.total_ms = std::numeric_limits<double>::infinity();
    for (const auto& s : samples) {
        out.p0_ms = std::min(out.p0_ms, s.p0_ms);
        out.fine_operator_ms = std::min(out.fine_operator_ms, s.fine_operator_ms);
        out.jacobi_ms = std::min(out.jacobi_ms, s.jacobi_ms);
        out.vector_update_ms = std::min(out.vector_update_ms, s.vector_update_ms);
        out.p0t_ms = std::min(out.p0t_ms, s.p0t_ms);
        out.total_ms = std::min(out.total_ms, s.total_ms);
    }
    return out;
}

double median_scalar(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

GpuSmoothedAggregationTiming summarize_fieldwise_median(
    const std::vector<GpuSmoothedAggregationTiming>& samples) {
    std::vector<double> p0, a, jacobi, update, p0t, total;
    p0.reserve(samples.size());
    a.reserve(samples.size());
    jacobi.reserve(samples.size());
    update.reserve(samples.size());
    p0t.reserve(samples.size());
    total.reserve(samples.size());
    for (const auto& s : samples) {
        p0.push_back(s.p0_ms);
        a.push_back(s.fine_operator_ms);
        jacobi.push_back(s.jacobi_ms);
        update.push_back(s.vector_update_ms);
        p0t.push_back(s.p0t_ms);
        total.push_back(s.total_ms);
    }
    return {median_scalar(std::move(p0)),
            median_scalar(std::move(a)),
            median_scalar(std::move(jacobi)),
            median_scalar(std::move(update)),
            median_scalar(std::move(p0t)),
            median_scalar(std::move(total))};
}

float elapsed_event(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.0f;
    check_cuda_pcg(cudaEventElapsedTime(&ms, a, b),
                   "cudaEventElapsedTime(smoothed aggregation)");
    return ms;
}

}  // namespace

struct GpuSmoothedAggregationContext::Impl {
    StructuredHexMesh mesh;
    Material material;
    double omega_value{0.0};
    int block_y{4};
    std::size_t nodes{0};
    std::size_t ndof{0};
    std::size_t coarse_dof_count{0};
    std::uint32_t aggregate_count{0};

    std::uint32_t* d_aggregate_offsets{nullptr};
    std::uint32_t* d_aggregate_nodes{nullptr};
    DeviceAggregateSa* d_aggregates{nullptr};
    float* d_coordinates{nullptr};
    float* d_coarse_x{nullptr};
    float* d_coarse_y{nullptr};
    float* d_f0{nullptr};
    float* d_f1{nullptr};
    float* d_f2{nullptr};

    std::size_t aggregation_metadata_bytes{0};
    std::size_t model_coordinate_bytes{0};
    std::size_t fine_workspace_bytes{0};
    std::size_t coarse_workspace_bytes{0};

    Impl(const StructuredHexMesh& mesh_in,
         const Material& material_in,
         const ElasticityAggregationCoarseSpace& space,
         double omega_in,
         int block_y_in)
        : mesh(mesh_in), material(material_in), omega_value(omega_in), block_y(block_y_in) {
        if (mesh.nx == 0U || mesh.ny == 0U || mesh.nz == 0U) {
            throw std::invalid_argument("GPU smoothed aggregation requires non-empty mesh");
        }
        if (!(omega_value > 0.0) || !std::isfinite(omega_value)) {
            throw std::invalid_argument("GPU smoothed aggregation omega must be finite and positive");
        }
        if (block_y <= 0 || block_y > 32 || 32 * block_y > 1024) {
            throw std::invalid_argument("GPU smoothed aggregation block_y is invalid");
        }
        nodes = static_cast<std::size_t>(mesh.node_count());
        ndof = static_cast<std::size_t>(mesh.dof_count());
        coarse_dof_count = space.coarse_dofs;
        if (space.graph.coordinates.size() != nodes ||
            space.aggregate_of_node.size() != nodes ||
            coarse_dof_count == 0U || space.aggregates.empty()) {
            throw std::invalid_argument("GPU smoothed aggregation coarse space/mesh mismatch");
        }
        if (space.aggregates.size() > std::numeric_limits<std::uint32_t>::max() ||
            coarse_dof_count > std::numeric_limits<std::uint32_t>::max() ||
            nodes > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("GPU smoothed aggregation currently requires 32-bit indexing");
        }
        aggregate_count = static_cast<std::uint32_t>(space.aggregates.size());

        std::vector<DeviceAggregateSa> host_aggregates(space.aggregates.size());
        for (std::size_t a = 0; a < space.aggregates.size(); ++a) {
            const auto& src = space.aggregates[a];
            if (src.rank == 0U || src.rank > 6U ||
                src.coarse_offset + src.rank > coarse_dof_count ||
                src.coarse_offset > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("GPU smoothed aggregation aggregate metadata is invalid");
            }
            auto& dst = host_aggregates[a];
            dst.coarse_offset = static_cast<std::uint32_t>(src.coarse_offset);
            dst.rank = static_cast<std::uint32_t>(src.rank);
            dst.centroid[0] = static_cast<float>(src.centroid[0]);
            dst.centroid[1] = static_cast<float>(src.centroid[1]);
            dst.centroid[2] = static_cast<float>(src.centroid[2]);
            dst.inv_scale = static_cast<float>(1.0 / src.coordinate_scale);
            for (std::size_t q = 0; q < 36U; ++q) {
                dst.transform[q] = static_cast<float>(src.rigid_transform[q]);
            }
        }

        // Invert node->aggregate ownership once during setup. Constrained nodes
        // have no aggregate and are intentionally absent from the compact list.
        std::vector<std::uint32_t> host_aggregate_offsets(
            static_cast<std::size_t>(aggregate_count) + 1U, 0U);
        std::size_t owned_nodes = 0U;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto a = space.aggregate_of_node[node];
            if (a >= aggregate_count) continue;
            ++host_aggregate_offsets[static_cast<std::size_t>(a) + 1U];
            ++owned_nodes;
        }
        for (std::size_t a = 0; a < aggregate_count; ++a) {
            host_aggregate_offsets[a + 1U] += host_aggregate_offsets[a];
        }
        if (owned_nodes != space.free_nodes ||
            host_aggregate_offsets.back() != owned_nodes) {
            throw std::runtime_error("GPU smoothed aggregation aggregate ownership count mismatch");
        }

        std::vector<std::uint32_t> host_aggregate_nodes(owned_nodes, 0U);
        auto cursor = host_aggregate_offsets;
        for (std::size_t node = 0; node < nodes; ++node) {
            const auto a = space.aggregate_of_node[node];
            if (a >= aggregate_count) continue;
            host_aggregate_nodes[cursor[a]++] = static_cast<std::uint32_t>(node);
        }
        for (std::size_t a = 0; a < aggregate_count; ++a) {
            const std::size_t count = host_aggregate_offsets[a + 1U] - host_aggregate_offsets[a];
            if (count == 0U) {
                throw std::runtime_error("GPU smoothed aggregation found empty aggregate");
            }
        }

        std::vector<float> host_coordinates(3U * nodes, 0.0f);
        for (std::size_t node = 0; node < nodes; ++node) {
            host_coordinates[node] = static_cast<float>(space.graph.coordinates[node][0]);
            host_coordinates[nodes + node] = static_cast<float>(space.graph.coordinates[node][1]);
            host_coordinates[2U * nodes + node] = static_cast<float>(space.graph.coordinates[node][2]);
        }

        const std::size_t aggregate_offset_bytes = host_aggregate_offsets.size() * sizeof(std::uint32_t);
        const std::size_t aggregate_node_bytes = host_aggregate_nodes.size() * sizeof(std::uint32_t);
        const std::size_t aggregate_bytes = host_aggregates.size() * sizeof(DeviceAggregateSa);
        aggregation_metadata_bytes = aggregate_offset_bytes + aggregate_node_bytes + aggregate_bytes;
        model_coordinate_bytes = host_coordinates.size() * sizeof(float);
        fine_workspace_bytes = 3U * ndof * sizeof(float);
        coarse_workspace_bytes = 2U * coarse_dof_count * sizeof(float);

        try {
            upload_pcg_stencil(mesh, material);
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_aggregate_offsets), aggregate_offset_bytes),
                           "cudaMalloc(SA aggregate offsets)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_aggregate_nodes), aggregate_node_bytes),
                           "cudaMalloc(SA aggregate nodes)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_aggregates), aggregate_bytes),
                           "cudaMalloc(SA aggregate metadata)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_coordinates), model_coordinate_bytes),
                           "cudaMalloc(SA coordinates)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_coarse_x), coarse_dof_count * sizeof(float)),
                           "cudaMalloc(SA coarse x)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_coarse_y), coarse_dof_count * sizeof(float)),
                           "cudaMalloc(SA coarse y)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_f0), ndof * sizeof(float)),
                           "cudaMalloc(SA fine f0)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_f1), ndof * sizeof(float)),
                           "cudaMalloc(SA fine f1)");
            check_cuda_pcg(cudaMalloc(reinterpret_cast<void**>(&d_f2), ndof * sizeof(float)),
                           "cudaMalloc(SA fine f2)");

            check_cuda_pcg(cudaMemcpy(d_aggregate_offsets,
                                      host_aggregate_offsets.data(),
                                      aggregate_offset_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(SA aggregate offsets H2D)");
            check_cuda_pcg(cudaMemcpy(d_aggregate_nodes,
                                      host_aggregate_nodes.data(),
                                      aggregate_node_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(SA aggregate nodes H2D)");
            check_cuda_pcg(cudaMemcpy(d_aggregates,
                                      host_aggregates.data(),
                                      aggregate_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(SA aggregate metadata H2D)");
            check_cuda_pcg(cudaMemcpy(d_coordinates,
                                      host_coordinates.data(),
                                      model_coordinate_bytes,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpy(SA coordinates H2D)");
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~Impl() { cleanup(); }

    void cleanup() noexcept {
        if (d_aggregate_offsets) cudaFree(d_aggregate_offsets);
        if (d_aggregate_nodes) cudaFree(d_aggregate_nodes);
        if (d_aggregates) cudaFree(d_aggregates);
        if (d_coordinates) cudaFree(d_coordinates);
        if (d_coarse_x) cudaFree(d_coarse_x);
        if (d_coarse_y) cudaFree(d_coarse_y);
        if (d_f0) cudaFree(d_f0);
        if (d_f1) cudaFree(d_f1);
        if (d_f2) cudaFree(d_f2);
        d_aggregate_offsets = nullptr;
        d_aggregate_nodes = nullptr;
        d_aggregates = nullptr;
        d_coordinates = nullptr;
        d_coarse_x = nullptr;
        d_coarse_y = nullptr;
        d_f0 = d_f1 = d_f2 = nullptr;
    }

    void run_pipeline(std::size_t m,
                      std::vector<cudaEvent_t>* markers,
                      std::vector<TimedStage>* stages) {
        std::size_t marker_index = 0U;
        auto mark = [&](TimedStage stage) {
            if (markers && stages) {
                check_cuda_pcg(cudaEventRecord((*markers)[++marker_index]),
                               "cudaEventRecord(SA marker)");
                stages->push_back(stage);
            }
        };
        if (markers && stages) {
            stages->clear();
            marker_index = 0U;
            check_cuda_pcg(cudaEventRecord((*markers)[0]),
                           "cudaEventRecord(SA start)");
        }

        launch_p0(mesh, nodes, aggregate_count, d_aggregate_offsets,
                  d_aggregate_nodes, d_aggregates,
                  d_coordinates, d_coarse_x, d_f0);
        mark(TimedStage::P0);

        for (std::size_t step = 0; step < m; ++step) {
            launch_pcg_matvec(mesh, block_y, nodes, d_f0, d_f1);
            mark(TimedStage::FineA);
            launch_pcg_jacobi(mesh, block_y, nodes, d_f1, d_f2);
            mark(TimedStage::Jacobi);
            launch_negative_axpy(ndof, static_cast<float>(omega_value), d_f2, d_f0);
            mark(TimedStage::Update);
        }

        launch_pcg_matvec(mesh, block_y, nodes, d_f0, d_f1);
        mark(TimedStage::FineA);

        for (std::size_t step = 0; step < m; ++step) {
            launch_pcg_jacobi(mesh, block_y, nodes, d_f1, d_f2);
            mark(TimedStage::Jacobi);
            launch_pcg_matvec(mesh, block_y, nodes, d_f2, d_f0);
            mark(TimedStage::FineA);
            launch_negative_axpy(ndof, static_cast<float>(omega_value), d_f0, d_f1);
            mark(TimedStage::Update);
        }

        launch_p0t(nodes, aggregate_count, d_aggregate_offsets,
                   d_aggregate_nodes, d_aggregates,
                   d_coordinates, d_f1, d_coarse_y);
        mark(TimedStage::P0T);
    }
};

GpuSmoothedAggregationContext::GpuSmoothedAggregationContext(
    const StructuredHexMesh& mesh,
    const Material& material,
    const ElasticityAggregationCoarseSpace& space,
    double omega,
    int block_y)
    : impl_(std::make_unique<Impl>(mesh, material, space, omega, block_y)) {}

GpuSmoothedAggregationContext::~GpuSmoothedAggregationContext() = default;
GpuSmoothedAggregationContext::GpuSmoothedAggregationContext(
    GpuSmoothedAggregationContext&&) noexcept = default;
GpuSmoothedAggregationContext& GpuSmoothedAggregationContext::operator=(
    GpuSmoothedAggregationContext&&) noexcept = default;

GpuSmoothedAggregationApplyResult GpuSmoothedAggregationContext::apply(
    const std::vector<float>& coarse_x,
    std::size_t transfer_smoothing_steps,
    int repeats) {
    if (!impl_) throw std::runtime_error("GPU smoothed aggregation context is empty");
    if (coarse_x.size() != impl_->coarse_dof_count) {
        throw std::invalid_argument("GPU smoothed aggregation coarse input size mismatch");
    }
    if (repeats <= 0) {
        throw std::invalid_argument("GPU smoothed aggregation repeats must be positive");
    }
    if (transfer_smoothing_steps > 8U) {
        throw std::invalid_argument("GPU smoothed aggregation reference limits m to 8");
    }

    check_cuda_pcg(cudaMemcpy(impl_->d_coarse_x,
                              coarse_x.data(),
                              coarse_x.size() * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy(SA coarse x H2D)");

    impl_->run_pipeline(transfer_smoothing_steps, nullptr, nullptr);
    check_cuda_pcg(cudaDeviceSynchronize(), "cudaDeviceSynchronize(SA warmup)");

    const std::size_t intervals = 6U * transfer_smoothing_steps + 3U;
    std::vector<cudaEvent_t> markers(intervals + 1U, nullptr);
    try {
        for (auto& event : markers) {
            check_cuda_pcg(cudaEventCreate(&event), "cudaEventCreate(SA marker)");
        }

        std::vector<GpuSmoothedAggregationTiming> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        std::vector<TimedStage> stages;
        stages.reserve(intervals);

        for (int repeat = 0; repeat < repeats; ++repeat) {
            impl_->run_pipeline(transfer_smoothing_steps, &markers, &stages);
            check_cuda_pcg(cudaEventSynchronize(markers.back()),
                           "cudaEventSynchronize(SA final marker)");
            if (stages.size() != intervals) {
                throw std::runtime_error("GPU smoothed aggregation timing marker mismatch");
            }

            GpuSmoothedAggregationTiming sample;
            for (std::size_t i = 0; i < intervals; ++i) {
                const double ms = static_cast<double>(elapsed_event(markers[i], markers[i + 1U]));
                switch (stages[i]) {
                    case TimedStage::P0: sample.p0_ms += ms; break;
                    case TimedStage::FineA: sample.fine_operator_ms += ms; break;
                    case TimedStage::Jacobi: sample.jacobi_ms += ms; break;
                    case TimedStage::Update: sample.vector_update_ms += ms; break;
                    case TimedStage::P0T: sample.p0t_ms += ms; break;
                }
            }
            sample.total_ms = static_cast<double>(elapsed_event(markers.front(), markers.back()));
            samples.push_back(sample);
        }

        GpuSmoothedAggregationApplyResult result;
        result.coarse_y.resize(impl_->coarse_dof_count);
        check_cuda_pcg(cudaMemcpy(result.coarse_y.data(),
                                  impl_->d_coarse_y,
                                  result.coarse_y.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(SA coarse y D2H)");
        result.median_timing = summarize_fieldwise_median(samples);
        result.best_timing = summarize_fieldwise_min(samples);
        result.transfer_smoothing_steps = transfer_smoothing_steps;
        result.fine_operator_applies = 2U * transfer_smoothing_steps + 1U;
        result.fine_workspace_bytes = impl_->fine_workspace_bytes;
        result.coarse_workspace_bytes = impl_->coarse_workspace_bytes;
        result.aggregation_metadata_bytes = impl_->aggregation_metadata_bytes;
        result.model_coordinate_bytes = impl_->model_coordinate_bytes;
        result.device_bytes_total = impl_->fine_workspace_bytes +
                                    impl_->coarse_workspace_bytes +
                                    impl_->aggregation_metadata_bytes +
                                    impl_->model_coordinate_bytes;

        for (auto event : markers) cudaEventDestroy(event);
        return result;
    } catch (...) {
        for (auto event : markers) {
            if (event) cudaEventDestroy(event);
        }
        throw;
    }
}

std::size_t GpuSmoothedAggregationContext::fine_dofs() const noexcept {
    return impl_ ? impl_->ndof : 0U;
}

std::size_t GpuSmoothedAggregationContext::coarse_dofs() const noexcept {
    return impl_ ? impl_->coarse_dof_count : 0U;
}

double GpuSmoothedAggregationContext::omega() const noexcept {
    return impl_ ? impl_->omega_value : 0.0;
}

}  // namespace gfss
