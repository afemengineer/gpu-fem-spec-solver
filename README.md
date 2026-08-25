# gpu-fem-spec-solver

Memory-bounded GPU finite-element solver research project.

The initial objective is to maximize the size of a 3D linear-elasticity problem that can be solved entirely on an 8 GB RTX 2080 SUPER while remaining competitive with a CPU reference in wall-clock time and meeting explicit numerical/engineering accuracy criteria.

The project begins with a reproducible matrix-free CUDA baseline and only introduces reduced-precision or speculative preconditioning after the baseline memory/performance ceiling is measured.

See `docs/PROJECT_PLAN.md` on the development branch for the staged research plan.
