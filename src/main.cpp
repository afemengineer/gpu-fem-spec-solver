#include <exception>
#include <iostream>
#include <string>

#include "gfss/gpu_info.hpp"
#include "gfss/memory_model.hpp"
#include "gfss/problem.hpp"

namespace {
constexpr std::uint64_t gib(std::uint64_t n) {
    return n * 1024ULL * 1024ULL * 1024ULL;
}

void print_usage() {
    std::cout
        << "gpu-fem-spec-solver scaffold\n\n"
        << "Usage:\n"
        << "  gfss info\n"
        << "  gfss capacity [budget_gib]\n"
        << "  gfss estimate <nx> <ny> <nz>\n";
}

std::uint64_t parse_u64(const char* text) {
    return static_cast<std::uint64_t>(std::stoull(text));
}
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage();
            return 0;
        }

        const std::string command = argv[1];
        if (command == "info") {
            const auto gpu = gfss::query_gpu_info();
            if (!gpu.available) {
                std::cout << "CUDA device: unavailable (CPU-only build or CUDA runtime error)\n";
                return 0;
            }
            std::cout << "CUDA device: " << gpu.name << '\n'
                      << "compute capability: " << gpu.compute_major << '.' << gpu.compute_minor << '\n'
                      << "SM count: " << gpu.multiprocessor_count << '\n'
                      << "VRAM total: " << gfss::format_bytes(gpu.total_global_memory) << '\n'
                      << "VRAM free now: " << gfss::format_bytes(gpu.free_global_memory) << '\n';
            return 0;
        }

        if (command == "capacity") {
            const std::uint64_t budget_gib = argc >= 3 ? parse_u64(argv[2]) : 8ULL;
            const auto n = gfss::max_cubic_mesh_for_budget(gib(budget_gib));
            gfss::StructuredHexProblem problem;
            problem.nx = problem.ny = problem.nz = n;
            const auto estimate = gfss::estimate_structured_matrix_free_memory(problem);
            std::cout << "Analytical scaffold estimate only; not a measured allocation.\n"
                      << "budget: " << budget_gib << " GiB\n"
                      << "largest cubic mesh: " << n << " x " << n << " x " << n << " elements\n"
                      << "elements: " << problem.element_count() << '\n'
                      << "nodes: " << problem.node_count() << '\n'
                      << "DOFs: " << problem.dof_count() << '\n'
                      << "variable bytes/DOF: " << estimate.bytes_per_dof() << '\n'
                      << "estimated total: " << gfss::format_bytes(estimate.total_bytes()) << '\n';
            return 0;
        }

        if (command == "estimate") {
            if (argc != 5) {
                print_usage();
                return 2;
            }
            gfss::StructuredHexProblem problem;
            problem.nx = parse_u64(argv[2]);
            problem.ny = parse_u64(argv[3]);
            problem.nz = parse_u64(argv[4]);
            const auto estimate = gfss::estimate_structured_matrix_free_memory(problem);
            std::cout << "elements: " << problem.element_count() << '\n'
                      << "nodes: " << problem.node_count() << '\n'
                      << "DOFs: " << problem.dof_count() << '\n'
                      << "variable bytes/DOF: " << estimate.bytes_per_dof() << '\n';
            for (const auto& component : estimate.components) {
                std::cout << "  " << component.name << ": " << gfss::format_bytes(component.bytes) << '\n';
            }
            std::cout << "fixed reserve: " << gfss::format_bytes(estimate.fixed_reserve_bytes) << '\n'
                      << "estimated total: " << gfss::format_bytes(estimate.total_bytes()) << '\n';
            return 0;
        }

        print_usage();
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
