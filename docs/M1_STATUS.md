# M1 Status — CPU HEX8 Reference

Current branch: `m1/cpu-hex8-reference`

## Implemented

- 3D isotropic linear-elastic constitutive matrix.
- Trilinear HEX8 shape functions.
- 2x2x2 Gauss integration.
- General element Jacobian and inverse-Jacobian path.
- Element stiffness matrix and element operator application.
- Structured HEX8 mesh indexing and coordinates.
- Small dense assembled global stiffness reference.
- Matrix-free CPU element-scatter operator.
- Symmetric homogeneous Dirichlet handling for an x=0 clamp.
- Serial conjugate-gradient solver.
- Small dense Gaussian direct solver for validation only.

## Current tests

- Shape-function partition of unity.
- Shape-gradient sum-to-zero.
- Unit-cube Jacobian determinant.
- Element stiffness symmetry.
- Rigid translation near-zero strain energy.
- Positive energy under non-rigid deformation.
- Affine constant-strain patch test.
- Assembled `K*x` vs matrix-free `A*x`.
- Assembled vs matrix-free symmetric Dirichlet operator.
- Matrix-free CG vs dense direct solution.
- Recomputed true residual.
- Global reaction-force equilibrium.

## Still required before M1 closes

- Run the complete test suite on the target Windows/MSVC toolchain.
- Add a modest CPU performance benchmark and OpenMP reference path.
- Record deterministic baseline timings.
- Add at least one distorted-element patch/kinematics test.
- Decide whether to add a six-rigid-body-mode nullspace test or keep the current translation check plus patch test.

M2 CUDA operator development should not begin until the numerical checks above pass on the target machine.
