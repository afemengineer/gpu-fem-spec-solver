# Architecture Notes

## 1. Layering

Keep four concepts separate:

1. **Physics/problem definition** — mesh, material, BCs, loads.
2. **Accurate operator** — defines the system `A x = b`; initially FP32 matrix-free elasticity.
3. **Outer solver** — PCG initially; flexible Krylov method when variable preconditioning requires it.
4. **Preconditioner/draft solver** — allowed to become increasingly approximate in later milestones.

This separation lets us experiment with aggressive preconditioners without accidentally changing the physical problem.

## 2. Accurate operator

For linear elasticity,

\[
K_e = \int_{\Omega_e} B^T D B\,d\Omega,
\]

but the production GPU path should not assemble global `K`.

The first regular structured HEX8 implementation can exploit:

- implicit element-to-node mapping;
- constant/reference geometry factors for a regular Cartesian mesh;
- procedural coordinates;
- deterministic material parameters;
- matrix-free element action.

A general distorted/unstructured path comes later and must pay its real coordinate/connectivity memory cost.

## 3. Scatter/gather design question

A matrix-free FEM operator still combines element contributions at shared nodes. Early CUDA experiments should compare:

- element-centric scatter with atomics;
- node-centric gather;
- graph coloring/staged accumulation if useful.

Do not assume atomics are bad or gather is better. Profile on Turing.

## 4. Solver state

A straightforward PCG implementation may hold:

- `x` solution;
- `b` RHS;
- `r` residual;
- `z` preconditioned residual;
- `p` search direction;
- `Ap` operator product;
- optional stored preconditioner diagonal/state.

M4 should perform lifetime analysis. If `b`, masks, loads, or diagonals are procedural for the structured benchmark, eliminate them only when doing so does not increase time-to-solution disproportionately.

Every eliminated FP32 vector is about 4 bytes/DOF — potentially a material capacity gain.

## 5. Multigrid architecture

Prefer geometric multigrid before algebraic multigrid for the structured capacity benchmark because the hierarchy can be represented procedurally.

Target shape:

```text
fine grid: accurate FP32 matrix-free operator
   ↓ restriction
coarse grid: matrix-free operator
   ↓
coarser grid
   ↓
small GPU-resident coarse solve
   ↑
prolongation + smoothing
```

Restriction/prolongation should be generated from grid relationships rather than stored as large sparse matrices when possible.

## 6. Reduced precision

Do not globally change the physical operator to FP16 and call the result equivalent.

Preferred progression:

```text
outer solution / true operator: FP32
preconditioner internals:       FP32 → FP16 → more aggressive experiments
```

On the RTX 2080 SUPER/Turing target, FP16 is the primary reduced-precision path to investigate.

## 7. Speculative stage

The word *speculative* describes control flow, not numerical justification.

```text
r = b - A_fp32(x)
        │
        ▼
cheap draft preconditioner M_draft^-1(r)
        │
        ▼
proposed direction/correction z
        │
        ▼
accurate FP32 operator evaluates actual progress
        │
   ┌────┴────┐
 good      poor/stagnating
   │           │
continue    increase draft fidelity / fallback
```

If the draft preconditioner changes with iteration, use a flexible outer method whose theory permits variable preconditioning.

## 8. What not to optimize prematurely

Until profiling says otherwise, avoid spending time on:

- tensor-core formulations just because tensor cores exist;
- exotic quantization formats;
- CUDA Graphs;
- custom allocators;
- hand-written PTX;
- nonlinear physics;
- unstructured mesh compression;
- multi-GPU support.

The first bottleneck may simply be a poor matrix-free kernel or excessive Krylov iterations.
