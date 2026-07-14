# Stage 2 Adjustments

This file records implementation adjustments that we have agreed to keep.
The goal is to keep decisions short, clear, and easy to review later for
conflicts.

## Agreed Adjustments

### 0. Stage 1 will be removed

Status: agreed, pending final cleanup

Reason:
- The active development path is now `stage2`.
- We do not want to keep maintaining a parallel legacy structure if it is no
  longer used.

Adjustment:
- Do not spend effort improving or preserving `stage1`.
- After the current cleanup and standardization pass is complete, remove
  `stage1`.

### 1. Filtering should follow MATLAB sparse form

Status: agreed, not implemented yet

Reason:
- We already agreed that the C++ structure should match MATLAB as closely as
  possible.
- The current `neighbors + weights + loops` form is harder to read and less
  aligned with the MATLAB source.

Adjustment:
- Replace the current filter storage based on `neighbors`, `weights`, and `Hs`
  as the main representation.
- Store the density filter in sparse form, equivalent to MATLAB `H` and `Hs`.
- Rewrite:
  - `filtering_calculation(...)` to follow `xTilde = (H * x) ./ Hs`
  - `filtering_sensitivity(...)` to follow `df_dx = H^T * (...)`

Expected benefit:
- Closer to MATLAB
- Cleaner code
- Better chance of higher performance on large models
- Easier to verify against MATLAB

### 2. Dirichlet BC must not be treated as zero by default

Status: agreed, not implemented yet

Reason:
- In MATLAB, Dirichlet values come from `D.fun_value(istep)` and may be
  nonzero or time-dependent.
- The current C++ forward flow treats prescribed Dirichlet dofs as zero, which
  changes the physical model.

Adjustment:
- Do not assume Dirichlet values are zero.
- In forward solve, prescribed dofs must be assigned from `bc.fun_value`
  exactly like MATLAB `set_Dirich(...)`.
- RHS construction must include the correction term using the actual
  prescribed displacement values, for both the initial solve and each time
  step.

Expected benefit:
- Correct match with MATLAB boundary conditions
- Correct handling of nonzero and time-dependent prescribed displacement
- Avoid hidden model error in forward and sensitivity
- Refer to the method that Neum is implementing in C++

### 3. Objective and constraint must not be fixed to one hard-coded function

Status: agreed, not implemented yet

Reason:
- In MATLAB, objective and constraint functions are selected from input
  declarations, not hard-coded in one fixed path.
- The current C++ version is still fixed to specific functions such as
  L2_compliance_elastic and volume_fraction.

Adjustment:
- Do not hard-code objective and constraint to one function each.
- The selected function should come from the declared input data, following the
  MATLAB flow.
- Keep the structure open so that later we can use derived structs/classes to
  reuse shared fields and override function-specific behavior.

Expected benefit:
- Closer to MATLAB
- Easier to extend to new objective and constraint types
- Cleaner path for inheritance-based design later

### 4. Use struct-based OOP when extending the design

Status: agreed, not implemented yet

Reason:
- We want the C++ data structure to stay as close to MATLAB as possible.
- For this project, strict private data control is not a priority.
- Using struct-based OOP keeps the code easier to read and less heavy for this workflow.

Adjustment:
- Future extension can use inheritance, virtual functions, and polymorphism with struct.
- Do not force a class-heavy design only for private access control.
- Prefer struct when it keeps the code closer to the MATLAB organization.

Expected benefit:
- Closer to MATLAB
- Easier extension toward OOP later
- Lower code complexity for this project style

### 5. Single-rank element loops should be able to use OpenMP

Status: agreed, not implemented yet

Reason:
- When running with MPI rank = 1, all element work stays inside one process.
- If element loops still run with one thread, CPU resources are wasted on large models.

Adjustment:
- In single-rank mode, heavy element-wise loops should be designed so they can be parallelized with OpenMP.
- Prioritize this for loops such as assembly, explicit gradient, filtering, and other element-wise calculations.
- Control reduction and accumulation carefully to avoid bottlenecks or race conditions when combining thread-local results.
- Avoid uncontrolled oversubscription between OpenMP loops and threaded solvers.

Expected benefit:
- Better use of one-machine multi-core CPU
- Better performance for large single-rank runs
- Clearer separation between MPI parallelism and thread parallelism

### 6. Interpolation derivatives must not be fixed to RAMP

Status: agreed, not implemented yet

Reason:
- In the current C++ code, some derivative paths are still hard-coded to use
  ramp_derivative(...).
- This makes the implementation less flexible than MATLAB and locks the code to
  one interpolation type.

Adjustment:
- Do not hard-code interpolation derivatives to RAMP in places such as
  explicit_gradient_dynamic(...).
- Derivative evaluation should follow the interpolation declared in the input
  data, consistent with the MATLAB flow.
- Keep the interpolation structure open so other interpolation laws can be used
  later without rewriting sensitivity code.

Expected benefit:
- Closer to MATLAB
- More flexible interpolation handling
- Cleaner extension path for future material models

### 7. PETSc and plain C++ data conversion must be centralized in a helper library

Status: agreed, not implemented yet

Reason:
- The project already has many small helper functions that move data between
  PETSc objects and plain C++ containers, such as gather_vec_to_all_local,
  set_vec_from_column_root, and similar patterns.
- The underlying ideas are simple, but the helpers are currently scattered and
  not easy to reuse consistently.

Adjustment:
- Create one dedicated helper module for PETSc and plain C++ data conversion.
- Cover both vector and matrix related operations.
- Use this helper layer as shared project infrastructure instead of creating
  ad hoc conversion helpers inside solver files.
- Add clear usage guidance so these helpers can be used directly later without
  rethinking the conversion logic each time.

Expected benefit:
- Cleaner code
- Less duplicated PETSc/C++ glue logic
- Easier maintenance and extension later
- More reliable and consistent data exchange across the project

### 8. Live plotting will use per-iteration CSV output

Status: agreed, not implemented yet

Reason:
- We want a simple and reliable live plotting path during optimization.
- Direct plotting from in-memory C++ data is possible, but it adds more runtime
  coupling and complexity.

Adjustment:
- During optimization, write plotting data to CSV after each iteration.
- A separate reader/watcher will read the CSV and update plots.
- The exact list of plots to display will be decided later.

Expected benefit:
- Simple live plotting workflow
- Clear separation between solver and plotting
- Easier debugging and post-processing reuse

## 9. Move local ROM algebra from PETSc Vec/Mat to Eigen

Status: agreed, not implemented.

For the current single-rank/multi-thread ROM path, keep PETSc as the FOM sparse/distributed solver backend, but store ROM basis, snapshots, reduced matrices, reduced RHS, and reduced solution coefficients as Eigen dense objects. Convert PETSc Vec/Mat to Eigen only at clear boundary helpers. Use Eigen SVD/QR/dense solvers for ROM operations. This should make ROM code closer to MATLAB and easier to optimize with OpenMP. Distributed ROM can be revisited later.
