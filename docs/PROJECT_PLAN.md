# Project Plan — GPU FEM Speculative Solver

## 1. Project thesis

This project asks a systems-oriented FEM question:

> **What is the largest 3D finite-element problem that can be solved accurately on an 8 GiB RTX 2080 SUPER while matching or beating a CPU reference in wall-clock solve time?**

The initial project is **not** a claim that matrix-free FEM, mixed precision, multigrid, or iterative refinement are novel. Those are established techniques. The research opportunity is to optimize the complete solver around a hard VRAM capacity constraint and then test whether an adaptive, aggressively compressed preconditioner can improve the capacity/performance frontier.

The project should remain useful even if the speculative idea fails. A high-quality matrix-free CUDA FEM solver, with rigorous CPU validation and capacity/performance measurements, is already a strong HPC portfolio artifact.

## 2. Primary objective and metric

For a prescribed numerical/engineering accuracy contract, maximize

\[
N_{\mathrm{competitive}} = \max\{N:\ M_{GPU}(N) \le M_{budget},\ T_{GPU}(N) \le T_{CPU}(N),\ E(N) \le E_{max}\}.
\]

Reference machine and initial problem:

- GPU: NVIDIA RTX 2080 SUPER, 8 GiB VRAM, Turing (`sm_75`).
- Primary precision: FP32.
- Reduced-precision experiments: FP16 first; do not assume BF16 as a baseline on Turing.
- Physics: 3D small-strain isotropic linear elasticity.
- Element: first-order 8-node hexahedron (HEX8/Q1).
- Topology: structured Cartesian mesh with implicit connectivity.

Secondary metrics:

- measured peak GPU bytes / DOF;
- setup, solve, and end-to-end time separately;
- operator throughput (MDOF/s and elements/s);
- achieved memory bandwidth;
- Krylov iterations and accurate operator applications;
- host↔device bytes transferred during solve;
- relative true residual;
- displacement, strain-energy, reaction-force, and stress errors versus reference.

The project must publish failed configurations as well as successful ones. OOM, divergence, and unacceptable error are data points.

## 3. Non-negotiable research rules

1. **No hidden memory.** Peak VRAM must include solver vectors, preconditioner state, geometry/connectivity, masks, temporary buffers, CUDA/library workspaces, and allocator reserve where relevant.
2. **No hidden CPU solve.** Host-side preprocessing or tiny coarse-grid work must be reported explicitly. A GPU result cannot silently offload the expensive part to CPU RAM.
3. **True residual is authoritative.** Reduced-precision/speculative experiments periodically compute `r=b-Ax` with the accurate FP32 matrix-free operator.
4. **Accuracy before speed.** Performance numbers are invalid if the solution violates the reference accuracy contract.
5. **Separate setup and solve.** Report `setup_s`, `solve_s`, and `end_to_end_s`.
6. **Never compare different physics.** CPU and GPU use the same mesh, element formulation, integration rule, BCs, material law, and convergence criterion.
7. **No novelty claim before the baseline wall is measured.** The speculative stage starts only after the conventional matrix-free + multigrid Pareto frontier is characterized.

## 4. Milestones

### M0 — Reproducible foundation

**Goal:** project structure, hardware reporting, memory accounting, benchmark schema, tests, and build system.

Deliverables:

- CMake project for Windows/MSVC and CPU-only CI;
- CUDA device information command;
- analytical memory ledger with named components;
- benchmark configuration example;
- CI smoke/unit test;
- research and validation protocol.

Exit gate: clean CPU-only build/test, CUDA-enabled configure on the development machine, and every subsequent allocation has a named memory-ledger category.

### M1 — Trusted CPU reference

Implement:

- structured HEX8 mesh indexing;
- isotropic constitutive matrix `D(E,nu)`;
- 2×2×2 Gauss integration;
- element stiffness/reference operator;
- assembled CPU reference for small meshes;
- CPU matrix-free operator;
- deterministic BC/load handling;
- serial CG, then OpenMP matrix-free CPU performance baseline.

Validation:

- constant-strain patch test;
- symmetry and positive-energy checks after essential constraints;
- matrix-free `A*x` versus assembled `K*x` on random vectors;
- iterative solution versus assembled/direct small-case reference;
- force/reaction equilibrium.

Exit gate: matrix-free/assembled matvec and solution agreement within defined FP32 tolerances.

### M2 — CUDA matrix-free elasticity operator

Implement and compare:

1. element-centric kernel with atomic accumulation;
2. node-centric/gather or coloring formulation that avoids/reduces atomics.

Start with regular Cartesian HEX8, where geometry and connectivity can be procedural. Distorted/general geometry is a later benchmark.

Instrumentation:

- CUDA-event kernel time;
- elements/s and MDOF/s;
- achieved bandwidth;
- occupancy/register pressure;
- atomic contention for scatter kernels;
- roofline classification.

Exit gate: GPU `A*x` matches CPU reference, no global stiffness matrix is allocated, and a repeatable throughput benchmark exists.

### M3 — GPU PCG baseline

Implement:

- device-resident vectors;
- reductions/dot products;
- PCG;
- Jacobi preconditioner;
- true-residual recomputation;
- convergence/breakdown checks;
- deterministic timing harness.

Only after correctness, test fused updates, reduced synchronizations, buffer reuse, procedural RHS/BC handling, and CUDA Graphs if profiling justifies them.

Exit gate: valid GPU solution, measured peak VRAM, and at least one sufficiently large problem where GPU solve time beats the CPU performance baseline.

### M4 — Capacity frontier

Optimize **bytes per converged DOF**, not merely kernel throughput.

Candidates:

- implicit connectivity/coordinates for the structured benchmark;
- 32-bit indices where safe;
- bit-packed or procedural BC masks;
- procedural loads;
- work-vector lifetime analysis and safe aliasing;
- on-the-fly diagonal for uniform cases if advantageous;
- preallocated memory pool to eliminate fragmentation noise.

Required plots:

- max DOF vs VRAM;
- solve time vs DOF;
- measured bytes/DOF vs DOF;
- GPU/CPU speedup vs DOF.

Initial stretch target, not a promised result: **20 million displacement DOFs in 8 GiB with a valid solution and GPU solve time no worse than CPU.**

Exit gate: capacity wall is measured rather than estimated and the dominant memory consumers are known.

### M5 — Matrix-free geometric multigrid

Implement:

- 2:1 structured hierarchy;
- matrix-free operator on every level;
- procedural restriction/prolongation;
- Jacobi/Chebyshev smoothing;
- V-cycle;
- GPU-resident coarse-level handling.

Measure hierarchy bytes/DOF, setup cost, V-cycle cost, convergence factor, total time to solution, and crossover versus PCG+Jacobi.

Exit gate: GMG improves time-to-solution at useful sizes without recreating an AMG-sized memory footprint.

### M6 — Reduced-precision preconditioning

Keep the accurate fine-grid operator and outer solution in FP32. Reduce precision inside the preconditioner in this order:

1. FP16 preconditioner work vectors;
2. FP16 coarse-level state;
3. level-dependent precision;
4. quantized diagonal/smoother parameters;
5. optional INT8/block-scaled representations where mathematically sensible.

If the preconditioner changes between outer iterations, standard PCG assumptions no longer hold. Use a mathematically appropriate flexible method (e.g. flexible CG for SPD problems) or another suitable flexible Krylov method.

Exit gate: a reduced-precision configuration shifts the capacity/performance Pareto frontier, or the direction is explicitly rejected with data.

### M7 — Speculative/adaptive solver experiment

Candidate flow:

1. accurate FP32 matrix-free operator defines the system;
2. a cheap reduced-fidelity multigrid/preconditioner proposes a correction direction;
3. the outer flexible solver evaluates actual progress with the accurate operator;
4. a controller increases/decreases draft fidelity when convergence behavior changes.

Possible adaptation knobs:

- V-cycle count;
- smoother iterations;
- hierarchy precision by level;
- temporary FP16→FP32 promotion;
- true-residual recomputation frequency;
- fallback to a safer preconditioner state.

A speculative configuration is interesting only if it dominates a fixed strategy under the same accuracy criterion. Merely converging is not enough.

**Novelty gate:** conduct a targeted literature review only after the exact winning mechanism is known. Do not build a novelty claim around a vague architecture.

### M8 — Distorted/unstructured generality test

Add progressively:

- distorted structured HEX8;
- explicit coordinates;
- explicit connectivity;
- coloring/partition metadata if needed;
- heterogeneous materials;
- representative unstructured mesh import.

Clearly distinguish the `structured-capacity record` from `general-mesh capacity`.

### M9 — Optional research branches

Only after the linear-elasticity solver is mature:

- nonlinear elasticity / Newton-Krylov with stale or approximate tangent preconditioning;
- matrix-free Jacobian-vector products;
- CFD/saddle-point systems;
- multi-GPU decomposition;
- out-of-core experiments.

## 5. Repository architecture

```text
gpu-fem-spec-solver/
├─ CMakeLists.txt
├─ README.md
├─ configs/
│  └─ benchmark_2080super.json
├─ docs/
│  ├─ PROJECT_PLAN.md
│  ├─ ARCHITECTURE.md
│  └─ RESEARCH_PROTOCOL.md
├─ include/gfss/
│  ├─ problem.hpp
│  ├─ memory_model.hpp
│  ├─ gpu_info.hpp
│  └─ solver_contract.hpp
├─ src/
│  ├─ main.cpp
│  ├─ memory_model.cpp
│  ├─ gpu_info.cu
│  └─ gpu_info_stub.cpp
├─ tests/
│  └─ memory_model_test.cpp
└─ scripts/
   └─ build/configuration helpers
```

As the numerical implementation grows, split by domain (`fem/`, `mesh/`, `operators/`, `solvers/`, `preconditioners/`, `backends/`, `benchmark/`) rather than accumulating miscellaneous CPU/GPU files.

## 6. Success definitions

### Portfolio success

The repo demonstrates correct FEM formulation, CUDA kernel design/profiling, numerical linear algebra, explicit memory/performance tradeoffs, a real GPU/CPU crossover, a large 8 GiB capacity result, and disciplined validation.

### Research success

The speculative branch is successful only if it produces a reproducible improvement to the best conventional capacity/performance/accuracy frontier and survives comparison with existing matrix-free and mixed-precision methods.
