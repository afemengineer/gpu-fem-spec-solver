# Development branch policy

The repository uses milestone branches for forward development and short-lived benchmark branches for isolated experiments.

## Canonical line

- `main` — stable project root
- `scaffold/v0-foundation` — V0 foundation
- `m1/cpu-hex8-reference` — trusted CPU physics/reference layer
- `m2/cuda-matrix-free-operator` — validated optimized CUDA structured-Q1 operator
- `m3/gpu-pcg-jacobi` — current full-solver work

New milestone work should branch from the latest validated milestone, not from a `bench/*` branch.

## Benchmark branches

`bench/*` branches exist only to isolate a performance hypothesis. Each experiment must end in one of three states:

1. **Promoted** — validated code is fast-forwarded/incorporated into the owning milestone; the benchmark branch can then be deleted.
2. **Rejected** — the branch is retained temporarily only when its code is useful for reproducing a negative result; the result must be summarized in the owning issue.
3. **Null** — same treatment as rejected unless there is a strong reason to keep the source implementation.

Do not stack new development on rejected/null experiment branches.

## Current operator experiment record

The M2 canonical operator is GoldSparse with full fixed-loop unrolling and a `32 x 4 x 1` launch on the RTX 2080 SUPER. Operator experiments and Nsight conclusions are tracked in issue #2.

Branches already subsumed by M2 should be pruned after local worktrees are clean. Negative-result branches may remain temporarily until their findings are sufficiently documented.

## Pull requests

Milestone PRs are draft while the milestone is under active validation. Benchmark PRs should not remain open once their work is subsumed by a milestone branch.
