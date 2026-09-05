# GFSS Solver Record quickstart

Canonical local dataset:

```text
results/solver_records.jsonl
```

`results/` is git-ignored. Each line is one complete solver run and can be uploaded directly for comparison. The full format contract is in `docs/GFSS_SOLVER_RECORD_SPEC.md`.

## Capture a real solver run

The adaptive problem suite is the first integrated producer. Build it:

```powershell
cmake --build build-cuda-cpuopt --target gfss_gpu_adaptive_problem_suite_bench
```

Then capture a thin-plate solve:

```powershell
python .\scripts\capture_solver_records.py `
    --out .\results\solver_records.jsonl `
    -- `
    .\build-cuda-cpuopt\gfss_gpu_adaptive_problem_suite_bench.exe `
    thin_plate `
    1e-6 `
    12 `
    5000
```

The benchmark keeps its normal output and also prints `GFSS_RECORD_JSON=<json>`. The capture tool extracts those lines, validates required v1 fields, fills missing timestamp/run-id/git provenance, and appends them to the JSONL file.

## Plot the upper-left Pareto frontier

Install Matplotlib once if needed:

```powershell
python -m pip install matplotlib
```

Plot and export a flat CSV:

```powershell
python .\scripts\plot_solver_pareto.py `
    .\results\solver_records.jsonl `
    --problem thin_plate `
    --tolerance 1e-6 `
    --annotate `
    --csv-out .\results\exports\thin_plate_points.csv `
    --output .\results\exports\thin_plate_pareto.png
```

Axes:

- x = time to verified tolerance in ms; lower is better.
- y = solver-state capacity in MDOF/GiB; higher is better.
- frontier = upper-left non-dominated envelope.

Only runs passing the true-residual correctness gate contribute to the frontier. The script refuses to mix different problems or target tolerances unless filtered explicitly.

## Setup reuse

For `R` right-hand sides the plot x-coordinate is `setup_ms / R + solve_ms`. Example:

```powershell
python .\scripts\plot_solver_pareto.py `
    .\results\solver_records.jsonl `
    --problem thin_plate `
    --tolerance 1e-6 `
    --reuse 10 `
    --output .\results\exports\thin_plate_pareto_R10.png
```

## Export/inspect without plotting

```powershell
python .\scripts\plot_solver_pareto.py `
    .\results\solver_records.jsonl `
    --problem thin_plate `
    --tolerance 1e-6 `
    --no-plot `
    --csv-out .\results\exports\thin_plate_points.csv
```

## New C++ solver benchmark integration

Include:

```cpp
#include "gfss/solver_record.hpp"
```

Populate `gfss::SolverRecord` after the complete verified solve, then call:

```cpp
gfss::emit_solver_record(std::cout, record);
```

At minimum populate solver family/variant, problem/fine DOFs, convergence and true residuals, setup/solve/total time, peak B/DOF, and any known work counters. The helper derives MDOF/GiB, residual digits, time-memory burden, and other secondary metrics.

Use `extra_strings["memory_scope"]` to state the capacity domain, e.g. `accelerator_persistent_solver_state`.

Do **not** emit Solver Records for isolated operator benchmarks, transfer timings, storage audits, or incomplete coarse-space experiments. Those are supporting telemetry, not time-to-verified-solution Pareto points.
