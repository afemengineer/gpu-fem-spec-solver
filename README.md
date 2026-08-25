# gpu-fem-spec-solver

**Memory-bounded GPU finite-element solver research project.**

The project asks one concrete question:

> **What is the largest 3D finite-element problem that can be solved accurately on an 8 GiB RTX 2080 SUPER while matching or beating a CPU reference in wall-clock solve time?**

The first target is 3D small-strain linear elasticity with structured HEX8/Q1 elements. The solver will move from a trusted CPU reference to a matrix-free CUDA operator, GPU PCG, matrix-free geometric multigrid, reduced-precision preconditioning, and finally an adaptive “speculative” preconditioner experiment.

The project does **not** assume that matrix-free FEM or mixed-precision iterative solving is novel. The speculative research branch only earns a novelty claim if it measurably shifts the best conventional capacity/performance/accuracy Pareto frontier.

## Status

Current development branch: `scaffold/v0-foundation`.

Implemented in the scaffold:

- CMake build with optional CUDA support;
- CPU-only CI path;
- RTX/CUDA device information command;
- structured HEX problem metadata;
- explicit analytical memory ledger;
- capacity estimator clearly separated from measured VRAM;
- benchmark configuration example;
- unit test for the memory accounting model;
- full staged project and research protocol.

Not implemented yet:

- finite-element stiffness/operator evaluation;
- CPU reference solver;
- CUDA matrix-free elasticity kernel;
- GPU Krylov solver;
- multigrid or speculative preconditioning.

Those begin at M1.

## Build — Windows + CUDA

The development target is Windows with MSVC/Visual Studio 2026 Build Tools and CUDA.

PowerShell:

```powershell
.\scripts\configure_windows_cuda.ps1
```

Or manually:

```powershell
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DGFSS_ENABLE_CUDA=ON -DGFSS_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\gfss.exe info
```

CPU-only configuration:

```powershell
cmake -S . -B build-cpu -DGFSS_ENABLE_CUDA=OFF -DGFSS_BUILD_TESTS=ON
cmake --build build-cpu --config Release
ctest --test-dir build-cpu -C Release --output-on-failure
```

## Current scaffold CLI

Show CUDA device/VRAM information:

```text
gfss info
```

Estimate the currently modeled memory components for a structured mesh:

```text
gfss estimate 128 64 64
```

Compute the largest *analytical* cubic mesh under a nominal memory budget:

```text
gfss capacity 8
```

`capacity` is deliberately labeled as an analytical scaffold estimate. It is **not** a solver capacity claim. Published capacity results will use measured allocations/peak VRAM after the CUDA solver exists.

## Research metric

The primary metric is

\[
N_{competitive}=\max\{N:\ M_{GPU}(N)\le M_{budget},\ T_{GPU}(N)\le T_{CPU}(N),\ E(N)\le E_{max}\}.
\]

In plain terms: the largest problem that fits, is accurate, and is no slower than the CPU reference.

Supporting metrics include measured bytes/DOF, setup time, solve time, MDOF/s, Krylov iterations, accurate matrix-free operator evaluations, true residual, displacement error, strain-energy error, reaction equilibrium, and stress error away from singularities.

## Roadmap

- **M0** — reproducible scaffold, memory accounting, benchmark contract.
- **M1** — trusted CPU HEX8 reference and patch/energy/equilibrium tests.
- **M2** — CUDA FP32 matrix-free elasticity operator.
- **M3** — GPU PCG + Jacobi baseline and first CPU/GPU crossover.
- **M4** — capacity optimization and measured 8 GiB frontier.
- **M5** — GPU matrix-free geometric multigrid.
- **M6** — FP16/reduced-precision preconditioner experiments.
- **M7** — flexible/adaptive “speculative” solver experiment.
- **M8** — distorted/unstructured mesh generality tests.
- **M9** — optional nonlinear/CFD branches.

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the complete plan and [`docs/RESEARCH_PROTOCOL.md`](docs/RESEARCH_PROTOCOL.md) for the benchmark/validation rules.

## Important design rule

The accurate operator and the preconditioner are separate concepts:

```text
accurate physical system:   FP32 matrix-free A*x
outer iterative solver:     PCG initially; flexible method when required
preconditioner/draft solve: conventional → reduced precision → adaptive
```

A later approximate preconditioner may be intentionally crude. It does not get to redefine the physical system or the final convergence criterion.

## Initial stretch target

A useful early target—not a promised result—is:

> **20 million displacement DOFs in 8 GiB, valid engineering solution, and GPU solve time no worse than the host CPU baseline.**

The real target is whatever the measured Pareto frontier supports.
