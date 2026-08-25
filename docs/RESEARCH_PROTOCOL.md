# Research and Benchmark Protocol

## 1. Purpose

The easiest way to fool ourselves in this project is to report a faster or larger solve while changing the convergence criterion, excluding setup/memory, or comparing unlike CPU/GPU formulations. This protocol fixes the rules before optimization begins.

## 2. Reference problem family

Initial family:

- 3D small-strain isotropic linear elasticity;
- HEX8/Q1 elements;
- full 2×2×2 Gauss integration;
- cantilever-style essential boundary at `x=0`;
- deterministic end traction/load;
- structured mesh with increasing resolution.

Validation family must additionally contain:

- constant-strain patch test;
- small random-vector matvec comparisons;
- at least one manufactured/analytically checkable displacement field if practical;
- a load case with meaningful reaction-force equilibrium.

## 3. Accuracy contract

Default solver convergence:

\[
\frac{\|b-Ax\|_2}{\|b\|_2} \le 10^{-6}
\]

The reported residual is the **true residual evaluated by the accurate operator**, not only the recursively updated Krylov residual.

Initial cross-implementation targets, to be tightened after M1 if appropriate:

- relative displacement L2 error vs trusted reference: `<= 1e-5` for small verification cases;
- strain-energy relative error: `<= 1e-4`;
- reaction-force imbalance: `<= 1e-4` relative to applied resultant;
- stress comparisons exclude known singular points/edges and define the sampling/recovery procedure.

Reduced-precision runs are judged by the same final contract.

## 4. Timing contract

Every benchmark records:

- `setup_s`;
- `solve_s`;
- `end_to_end_s`;
- `operator_kernel_s` where applicable;
- warmup count;
- measured repetition count.

Default procedure:

1. one untimed warmup;
2. at least five measured runs for stable cases;
3. report median and min/max or interquartile range;
4. synchronize the GPU at timing boundaries;
5. exclude compilation time unless runtime compilation becomes part of normal use.

## 5. Memory contract

Report both:

1. analytical component ledger;
2. measured device peak/free-memory delta.

Named components include at minimum:

- solution vector;
- RHS if stored;
- Krylov work vectors;
- preconditioner state;
- multigrid hierarchy;
- mesh coordinates;
- connectivity;
- BC/load metadata;
- temporary reduction buffers;
- CUDA/library workspaces;
- allocator reserve/pool;
- fixed runtime margin.

For each run record

\[
B_{DOF} = \frac{\text{measured peak device bytes attributable to run}}{N_{DOF}}.
\]

An OOM result records the largest previous successful case and the failed requested case.

## 6. CPU comparison contract

The CPU baseline uses the same mathematical problem. Report:

- CPU model;
- physical cores/threads used;
- thread count;
- compiler/build type;
- whether the CPU operator is assembled or matrix-free;
- setup and solve times separately;
- peak host RAM when practical.

Eventually keep two CPU references:

- **trusted reference** — prioritizes correctness and inspectability;
- **performance reference** — optimized/OpenMP and intended for the GPU crossover comparison.

Do not compare an optimized GPU solver only against a deliberately slow serial assembled baseline and call that CPU competitiveness.

## 7. Hardware/environment metadata

Each published result records:

- GPU name and VRAM;
- compute capability;
- NVIDIA driver;
- CUDA toolkit/runtime;
- GPU clocks/power limit if modified;
- CPU model;
- RAM capacity/speed if known;
- OS;
- compiler and version;
- git commit SHA;
- build type and relevant CMake options.

## 8. Planned run schema

Each run should eventually emit JSON/JSONL similar to:

```json
{
  "commit": "...",
  "hardware": {"gpu": "RTX 2080 SUPER", "vram_bytes": 8589934592},
  "mesh": {"nx": 128, "ny": 128, "nz": 128, "dofs": 6440067},
  "operator": {"backend": "cuda", "matrix_free": true, "precision": "fp32"},
  "solver": {"name": "pcg", "preconditioner": "jacobi", "iterations": 312},
  "accuracy": {"relative_true_residual": 8.2e-7},
  "timing": {"setup_s": 0.12, "solve_s": 1.84, "end_to_end_s": 1.96},
  "memory": {"peak_device_bytes": 1234567890, "bytes_per_dof": 191.7}
}
```

The exact schema should stabilize before M3 results are published.

## 9. Profiling protocol

Use Nsight Systems first for launch/synchronization bottlenecks and Nsight Compute for kernel-level analysis.

For the matrix-free operator record at least:

- achieved DRAM bandwidth;
- arithmetic throughput;
- occupancy;
- register count;
- atomic throughput/contention for scatter kernels;
- L1/L2 hit behavior.

Optimization must be driven by a measured bottleneck.

## 10. Required plots

At each major milestone regenerate:

1. solve time vs DOF;
2. GPU/CPU speedup vs DOF;
3. measured peak VRAM vs DOF;
4. bytes/DOF vs DOF;
5. iterations vs DOF;
6. true residual history for representative cases;
7. capacity/performance Pareto frontier.

For reduced-precision/speculative stages also plot:

- memory saved vs extra iterations;
- time-to-solution vs preconditioner precision;
- true residual vs recursive residual gap;
- convergence gained per preconditioner byte.
