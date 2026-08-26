#include "gfss/solver_record.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        gfss::SolverRecord record;
        record.run_id = "test-run";
        record.benchmark = "solver_record_test";
        record.solver_family = "test_family";
        record.solver_variant = "test_variant";
        record.problem_name = "test_problem";
        record.fine_dofs = 1000U;
        record.converged = true;
        record.target_relative_residual = 1.0e-6;
        record.initial_true_relative_residual = 1.0;
        record.true_relative_residual = 1.0e-7;
        record.setup_ms = 2.0;
        record.solve_ms = 8.0;
        record.total_ms = 10.0;
        record.peak_solver_bytes = 24000.0;
        record.peak_bytes_per_dof = 24.0;
        record.fine_operator_applies = 7U;
        record.fine_matvec_equivalents = 7.0;
        record.useful_flops = 306000.0;
        record.estimated_bytes_moved = 153000.0;
        record.gpu = "test \"gpu\"";
        record.extra_strings["note"] = "line1\nline2";

        const std::string json = gfss::solver_record_json(record);
        const auto require = [&](const std::string& token) {
            if (json.find(token) == std::string::npos) {
                throw std::runtime_error("missing JSON token: " + token);
            }
        };
        require("\"schema\":\"gfss.solver_record\"");
        require("\"capacity_mdof_per_gib\":");
        require("\"residual_digits_removed\":7");
        require("\"time_memory_burden_ms_bpd\":240");
        require("\"arithmetic_intensity_flop_per_byte\":2");
        require("test \\\"gpu\\\"");
        require("line1\\nline2");

        gfss::emit_solver_record(std::cout, record);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "solver_record_test failed: " << e.what() << '\n';
        return 1;
    }
}
