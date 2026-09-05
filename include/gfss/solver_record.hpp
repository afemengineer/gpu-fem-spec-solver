#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <string>

namespace gfss {

struct SolverRecord {
    std::string schema{"gfss.solver_record"};
    std::string schema_version{"1.0"};
    std::string run_id;
    std::string timestamp_utc;
    std::string benchmark;

    std::string git_commit;
    std::string git_branch;

    std::string solver_family;
    std::string solver_variant;

    std::string problem_name;
    std::uint64_t fine_dofs{0};
    std::string discretization;
    std::string material;

    bool converged{false};
    double target_relative_residual{0.0};
    double initial_true_relative_residual{1.0};
    double true_relative_residual{std::numeric_limits<double>::quiet_NaN()};
    std::string breakdown;

    double setup_ms{0.0};
    double solve_ms{0.0};
    double total_ms{0.0};
    double truth_ms{std::numeric_limits<double>::quiet_NaN()};

    std::string memory_kind{"modeled"};
    double peak_solver_bytes{std::numeric_limits<double>::quiet_NaN()};
    double peak_bytes_per_dof{std::numeric_limits<double>::quiet_NaN()};

    std::uint64_t fine_operator_applies{0};
    std::uint64_t coarse_operator_applies{0};
    std::uint64_t prolongation_applies{0};
    std::uint64_t restriction_applies{0};
    std::uint64_t dot_products{0};
    std::uint64_t axpy_like_ops{0};
    std::uint64_t true_residual_audits{0};
    double fine_matvec_equivalents{std::numeric_limits<double>::quiet_NaN()};
    double useful_flops{std::numeric_limits<double>::quiet_NaN()};
    double estimated_bytes_moved{std::numeric_limits<double>::quiet_NaN()};

    std::string cpu;
    std::string gpu;
    double gpu_vram_bytes{std::numeric_limits<double>::quiet_NaN()};
    std::uint64_t openmp_threads{0};
    std::string precision_policy;

    // Stable escape hatch for benchmark-specific scalar/string telemetry.
    // Prefer adding common metrics to the schema instead of proliferating extras.
    std::map<std::string, double> extra_numbers;
    std::map<std::string, std::string> extra_strings;
};

namespace solver_record_detail {

inline std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20U) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(c) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

inline void quoted(std::ostream& out, const std::string& value) {
    out << '"' << json_escape(value) << '"';
}

inline void number_or_null(std::ostream& out, double value) {
    if (std::isfinite(value)) {
        out << std::setprecision(17) << value;
    } else {
        out << "null";
    }
}

inline double capacity_mdof_per_gib(double bytes_per_dof) {
    if (!(bytes_per_dof > 0.0) || !std::isfinite(bytes_per_dof)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return static_cast<double>(1ULL << 30U) / bytes_per_dof / 1.0e6;
}

inline double residual_digits_removed(double r0, double r) {
    if (!(r0 > 0.0) || !(r > 0.0) || !std::isfinite(r0) || !std::isfinite(r)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::max(0.0, -std::log10(r / r0));
}

}  // namespace solver_record_detail

inline std::string solver_record_json(const SolverRecord& record) {
    using namespace solver_record_detail;
    std::ostringstream out;
    out << '{';

    out << "\"schema\":"; quoted(out, record.schema);
    out << ",\"schema_version\":"; quoted(out, record.schema_version);
    out << ",\"run_id\":"; quoted(out, record.run_id);
    if (!record.timestamp_utc.empty()) {
        out << ",\"timestamp_utc\":"; quoted(out, record.timestamp_utc);
    }
    out << ",\"benchmark\":"; quoted(out, record.benchmark);

    out << ",\"git\":{";
    out << "\"commit\":"; if (record.git_commit.empty()) out << "null"; else quoted(out, record.git_commit);
    out << ",\"branch\":"; if (record.git_branch.empty()) out << "null"; else quoted(out, record.git_branch);
    out << '}';

    out << ",\"solver\":{";
    out << "\"family\":"; quoted(out, record.solver_family);
    out << ",\"variant\":"; quoted(out, record.solver_variant);
    out << '}';

    out << ",\"problem\":{";
    out << "\"name\":"; quoted(out, record.problem_name);
    out << ",\"fine_dofs\":" << record.fine_dofs;
    if (!record.discretization.empty()) {
        out << ",\"discretization\":"; quoted(out, record.discretization);
    }
    if (!record.material.empty()) {
        out << ",\"material\":"; quoted(out, record.material);
    }
    out << '}';

    out << ",\"correctness\":{";
    out << "\"converged\":" << (record.converged ? "true" : "false");
    out << ",\"target_relative_residual\":"; number_or_null(out, record.target_relative_residual);
    out << ",\"initial_true_relative_residual\":"; number_or_null(out, record.initial_true_relative_residual);
    out << ",\"true_relative_residual\":"; number_or_null(out, record.true_relative_residual);
    if (!record.breakdown.empty()) {
        out << ",\"breakdown\":"; quoted(out, record.breakdown);
    }
    out << '}';

    out << ",\"timing\":{";
    out << "\"setup_ms\":"; number_or_null(out, record.setup_ms);
    out << ",\"solve_ms\":"; number_or_null(out, record.solve_ms);
    out << ",\"total_ms\":"; number_or_null(out, record.total_ms);
    if (std::isfinite(record.truth_ms)) {
        out << ",\"truth_ms\":"; number_or_null(out, record.truth_ms);
    }
    out << '}';

    const double capacity = capacity_mdof_per_gib(record.peak_bytes_per_dof);
    out << ",\"memory\":{";
    out << "\"kind\":"; quoted(out, record.memory_kind);
    if (std::isfinite(record.peak_solver_bytes)) {
        out << ",\"peak_solver_bytes\":"; number_or_null(out, record.peak_solver_bytes);
    }
    out << ",\"peak_bytes_per_dof\":"; number_or_null(out, record.peak_bytes_per_dof);
    out << ",\"capacity_mdof_per_gib\":"; number_or_null(out, capacity);
    out << '}';

    out << ",\"work\":{";
    out << "\"fine_operator_applies\":" << record.fine_operator_applies;
    out << ",\"coarse_operator_applies\":" << record.coarse_operator_applies;
    out << ",\"prolongation_applies\":" << record.prolongation_applies;
    out << ",\"restriction_applies\":" << record.restriction_applies;
    out << ",\"dot_products\":" << record.dot_products;
    out << ",\"axpy_like_ops\":" << record.axpy_like_ops;
    out << ",\"true_residual_audits\":" << record.true_residual_audits;
    if (std::isfinite(record.fine_matvec_equivalents)) {
        out << ",\"fine_matvec_equivalents\":"; number_or_null(out, record.fine_matvec_equivalents);
    }
    if (std::isfinite(record.useful_flops)) {
        out << ",\"useful_flops\":"; number_or_null(out, record.useful_flops);
    }
    if (std::isfinite(record.estimated_bytes_moved)) {
        out << ",\"estimated_bytes_moved\":"; number_or_null(out, record.estimated_bytes_moved);
    }
    if (std::isfinite(record.useful_flops) && record.useful_flops >= 0.0 &&
        std::isfinite(record.estimated_bytes_moved) && record.estimated_bytes_moved > 0.0) {
        out << ",\"arithmetic_intensity_flop_per_byte\":";
        number_or_null(out, record.useful_flops / record.estimated_bytes_moved);
    }
    out << '}';

    const double digits = residual_digits_removed(
        record.initial_true_relative_residual, record.true_relative_residual);
    out << ",\"derived\":{";
    out << "\"residual_digits_removed\":"; number_or_null(out, digits);
    if (std::isfinite(digits) && record.solve_ms > 0.0) {
        out << ",\"digits_per_second\":"; number_or_null(out, digits * 1000.0 / record.solve_ms);
    }
    if (std::isfinite(record.peak_bytes_per_dof) && record.peak_bytes_per_dof > 0.0 &&
        std::isfinite(record.total_ms)) {
        out << ",\"time_memory_burden_ms_bpd\":";
        number_or_null(out, record.total_ms * record.peak_bytes_per_dof);
    }
    if (std::isfinite(digits) && record.solve_ms > 0.0 &&
        std::isfinite(record.peak_bytes_per_dof) && record.peak_bytes_per_dof > 0.0) {
        out << ",\"digits_per_second_per_bpd\":";
        number_or_null(out, digits * 1000.0 /
            (record.solve_ms * record.peak_bytes_per_dof));
    }
    if (std::isfinite(digits) && std::isfinite(record.fine_matvec_equivalents) &&
        record.fine_matvec_equivalents > 0.0) {
        out << ",\"digits_per_fine_matvec_equivalent\":";
        number_or_null(out, digits / record.fine_matvec_equivalents);
    }
    out << '}';

    out << ",\"hardware\":{";
    bool first_hardware = true;
    const auto hardware_string = [&](const char* key, const std::string& value) {
        if (value.empty()) return;
        if (!first_hardware) out << ',';
        first_hardware = false;
        quoted(out, key); out << ':'; quoted(out, value);
    };
    hardware_string("cpu", record.cpu);
    hardware_string("gpu", record.gpu);
    if (std::isfinite(record.gpu_vram_bytes)) {
        if (!first_hardware) out << ',';
        first_hardware = false;
        out << "\"gpu_vram_bytes\":"; number_or_null(out, record.gpu_vram_bytes);
    }
    if (record.openmp_threads > 0U) {
        if (!first_hardware) out << ',';
        first_hardware = false;
        out << "\"openmp_threads\":" << record.openmp_threads;
    }
    hardware_string("precision_policy", record.precision_policy);
    out << '}';

    if (!record.extra_numbers.empty() || !record.extra_strings.empty()) {
        out << ",\"extra\":{";
        bool first = true;
        for (const auto& [key, value] : record.extra_numbers) {
            if (!first) out << ',';
            first = false;
            quoted(out, key); out << ':'; number_or_null(out, value);
        }
        for (const auto& [key, value] : record.extra_strings) {
            if (!first) out << ',';
            first = false;
            quoted(out, key); out << ':'; quoted(out, value);
        }
        out << '}';
    }

    out << '}';
    return out.str();
}

inline void emit_solver_record(std::ostream& out, const SolverRecord& record) {
    out << "GFSS_RECORD_JSON=" << solver_record_json(record) << '\n';
}

}  // namespace gfss
