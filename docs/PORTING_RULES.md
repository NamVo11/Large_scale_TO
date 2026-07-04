# Porting Rules

Scope for the current C++/PETSc port:

- Prioritize large-scale execution before convenience.
- Keep the MATLAB data split, function flow, and variable names as close as practical.
- Port only the code paths used by `examples/dynamic_rom_example_1.m` unless a later stage explicitly expands scope.
- Keep objective, constraint, sensitivity, `AVM`, and `Debug` as separate flow-level functions, matching MATLAB structure.
- Solver choice must stay behind the PETSc/FOM/ROM interface. Optimization and sensitivity code should call the solver object, not hard-code LU, PCG, GAMG, MKL, or MUMPS logic.
- Prefer PETSc data structures for large vectors/matrices. Gather to rank 0 only for small tests, debugging, or explicit output.
- Keep the original MATLAB repository read-only for this port. Copy or reimplement needed logic inside `Large_scale_TO`.
- Add new build/runtime variants in separate build directories or tasks so existing MPICH/PETSc workflows remain usable.
