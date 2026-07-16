# Factorization And Solve Timing Note

Purpose: record benchmark notes for direct solver settings that affect factorization and linear solve time.

## Model

- Case type: 2D plane stress dynamic elastic
- Mesh: `1200 x 400`
- Elements: `480000`
- DOFs: about `963202`
- Test mode: `FOM`
- Time steps: `20`
- Optimization iterations: `1`
- Solver path: PETSc direct solve through `KSP_TYPE=preonly`

## Quick Summary Tables

### FOM Direct Solver, Same Model `1200 x 400`

| Case | NP | `KSP_TYPE` | `PC_TYPE` | Factor Solver | Threads `OMP/MKL` | Forward `K_eff` Factor | Obj Sens `Keff` Factor | Forward Solve Mean | Obj Sens Solve Mean | Main Meaning |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|
| FOM-LU-Pardiso-1T | 1 | `preonly` | `lu` | `mkl_pardiso` | `1/1` | `~9.03 s` | `~9.41 s` | `~0.268 s` | `~0.270 s` | Single-thread baseline |
| FOM-LU-Pardiso-8T | 1 | `preonly` | `lu` | `mkl_pardiso` | `8/8` | `~4.37 s` | `~4.72 s` | `~0.084 s` | `~0.087 s` | Threading strongly helps FOM |
| FOM-Chol-Pardiso-8T | 1 | `preonly` | `cholesky` | `mkl_pardiso` | `8/8` | `~3.51 s` | `~3.58 s` | `~0.076 s` | `~0.087 s` | Best tested production direct setting |
| FOM-Chol-Default-1T | 1 | `preonly` | `cholesky` | PETSc default | `1/1` | `~6.41 s` | `~6.58 s` | `~0.255 s` | `~0.259 s` | Similar to Pardiso-1T without explicit factor solver |
| FOM-Chol-MUMPS-1T | 1 | `preonly` | `cholesky` | `mumps` | `1/1` | `~9.38 s` | `~9.33 s` | `~0.514 s` | `~0.502 s` | Slower single-rank direct baseline |
| FOM-Chol-MUMPS-8T | 1 | `preonly` | `cholesky` | `mumps` | `8/8` | `~13.41 s` | `~14.80 s` | `~1.035 s` | `~0.897 s` | Worse on this single-rank case; avoid for 1-rank |

### Factorization-Only Solver/Thread Comparison

| Case | NP | `PC_TYPE` | Factor Solver | Threads `OMP/MKL` | Forward `K_eff` Factor | Note |
|---|---:|---|---|---:|---:|---|
| LU-Pardiso-1T | 1 | `lu` | `mkl_pardiso` | `1/1` | `~9.03 s` | Baseline |
| LU-Pardiso-4T | 1 | `lu` | `mkl_pardiso` | `4/4` | `~4.84 s` | Threads help strongly |
| LU-Pardiso-8T | 1 | `lu` | `mkl_pardiso` | `8/8` | `~4.37 s` | Slightly better than 4T |
| Chol-Pardiso-1T | 1 | `cholesky` | `mkl_pardiso` | `1/1` | `~6.21 s` | Better than LU if SPD |
| Chol-Pardiso-4T | 1 | `cholesky` | `mkl_pardiso` | `4/4` | `~3.84 s` | Good production option |
| Chol-Pardiso-8T | 1 | `cholesky` | `mkl_pardiso` | `8/8` | `~3.51 s` | Best tested factorization |
| Chol-Default-1T | 1 | `cholesky` | PETSc default | `1/1` | `~6.41 s` | Similar to Cholesky Pardiso 1T |
| Chol-MUMPS-1T | 1 | `cholesky` | `mumps` | `1/1` | `~9.38 s` | Slower than default/Pardiso on one rank |
| Chol-MUMPS-8T | 1 | `cholesky` | `mumps` | `8/8` | `~13.41 s` | Slower with threads in this single-rank test |

### ROM Basis/Update Benchmark, Same DOF

Benchmark:

```text
basis_update_benchmark --ndof 963202 --nsnapshot 100 --max_rank 20 --online_updates 20
```

| Case | `POD_BACKEND` | `ONLINE_BACKEND` | Threads `OMP/MKL` | Offline POD | Online Update, 20 Updates | Subspace Error | Main Meaning |
|---|---|---|---:|---:|---:|---:|---|
| ROM-Eigen-1T | `snapshot_eigen` | `incremental_eigen` | `1/1` | `~4.55 s` | `~4.41 s` | reference | Eigen baseline |
| ROM-MKL-1T | `snapshot_mkl` | `incremental_mkl` | `1/1` | `~1.43 s` | `~3.90 s` | `~1e-7` | Good method-comparison setting |
| ROM-BDCSVD-1T | `bdcsvd` | `incremental_eigen` | `1/1` | `~23.05 s` | not main | `~1e-7` | Old slow backend |
| ROM-Eigen-4T | `snapshot_eigen` | `incremental_eigen` | `4/4` | `~3.81 s` | `~4.07 s` | reference | Threads help little |
| ROM-MKL-4T | `snapshot_mkl` | `incremental_mkl` | `4/4` | `~1.09 s` | `~3.35 s` | `~1e-7` | Good compromise |
| ROM-BDCSVD-4T | `bdcsvd` | `incremental_eigen` | `4/4` | `~13.72 s` | not main | `~1e-7` | Still slow |
| ROM-Eigen-8T | `snapshot_eigen` | `incremental_eigen` | `8/8` | `~3.88 s` | `~4.54 s` | reference | No clear thread benefit |
| ROM-MKL-8T | `snapshot_mkl` | `incremental_mkl` | `8/8` | `~1.02 s` | `~3.27 s` | `~1e-7` | Best tested ROM update time |
| ROM-BDCSVD-8T | `bdcsvd` | `incremental_eigen` | `8/8` | `~12.79 s` | not main | `~1e-7` | Old backend, keep for debug only |

### Existing Test2 ROM Component Timing

Source: `case_study/test2/result`.

These values are useful for overhead scale. If the run predates `snapshot_mkl/incremental_mkl`, rerun `test2` for final updated full-optimization numbers.

| Role | Component | Mean Time | Meaning |
|---|---|---:|---|
| Forward | `petsc_to_eigen_mat` | `~0.55 s` | PETSc matrix to Eigen sparse conversion |
| Forward | `projection V'AV` | `~0.32 s` | Build reduced matrix |
| Forward | `reduced_rhs V'rhs` | `~0.010 s` | Reduced RHS |
| Forward | `reduced_solve` | `~0.00006 s` | Tiny reduced system solve |
| Forward | `reconstruct V*q` | `~0.014 s` | Full vector reconstruction |
| Forward | `residual` | `~0.020 s` | ROM acceptance residual |
| Forward | `FOM fallback solve` | `~0.31 s` | Full solve after factorization |
| Forward | `timing_total, accepted ROM, rank=20, cached path` | mean `~0.055 s`, p90 `~0.058 s` | One accepted ROM step at full online basis rank, excluding projection/cache rebuild outliers |
| Forward | `timing_total, rejected ROM, rank=20` | mean `~1.75 s`, max `~2.97 s` | One rejected ROM step at full online basis rank; excludes offline/end update |
| Forward | `projection/cache rebuild overhead` | mean `~0.54 s`, max `~0.64 s` | Account this with reject/update penalty, not with normal accepted ROM cost |
| Obj Sens | `petsc_to_eigen_mat` | `~0.67 s` | PETSc matrix to Eigen sparse conversion |
| Obj Sens | `projection V'AV` | `~0.32 s` | Build reduced matrix |
| Obj Sens | `reduced_rhs V'rhs` | `~0.010 s` | Reduced RHS |
| Obj Sens | `reduced_solve` | `~0.00003 s` | Tiny reduced system solve |
| Obj Sens | `reconstruct V*q` | `~0.013 s` | Full vector reconstruction |
| Obj Sens | `residual` | `~0.021 s` | ROM acceptance residual |
| Obj Sens | `FOM fallback solve` | `~0.32 s` | Full solve after factorization |
| Obj Sens | `timing_total, accepted ROM, rank=20, cached path` | mean `~0.059 s`, p90 `~0.062 s` | One accepted ROM step at full online basis rank, excluding projection/cache rebuild outliers |
| Obj Sens | `timing_total, rejected ROM, rank=20` | mean `~2.08 s`, max `~3.41 s` | One rejected ROM step at full online basis rank; excludes offline/end update |
| Obj Sens | `projection/cache rebuild overhead` | mean `~0.59 s`, max `~0.69 s` | Account this with reject/update penalty, not with normal accepted ROM cost |

`timing_total, rejected ROM, rank=20` is the practical penalty of a failed ROM trial after the online basis is already full. It includes projection, reduced solve, reconstruction, residual check, FOM fallback solve, PETSc-to-Eigen vector copy, and online basis update. It does not include the offline/end-of-iteration POD update.

`timing_total, accepted ROM, rank=20, cached path` is the practical benefit case after the online basis is already full and the reduced matrix/cache does not need rebuilding. In this common path it is about `0.055-0.059 s`, which is lower than a one-thread FOM solve (`~0.25-0.26 s`) and also lower than the tested eight-thread FOM solve (`~0.08 s`).

Projection/cache rebuild can appear in the raw CSV on a step that is finally accepted because the code must build `A_r = V'AV` before it knows whether the ROM solution will pass the residual check. For break-even accounting, do not mix this rebuild cost into the normal accepted-ROM mean. Treat it as cache/update overhead and assign it to the reject/update penalty side. Otherwise the same type of preparation cost is effectively counted against both accepted and rejected ROM paths.

### Direct Takeaways

| Observation | Evidence | Practical Meaning |
|---|---|---|
| FOM benefits strongly from threads | Solve mean `~0.268 s` at 1T to `~0.084 s` at 8T | Multi-thread FOM can reduce apparent ROM speedup |
| ROM basis benefits less from threads | `online_mkl ~3.90 s` at 1T to `~3.27 s` at 8T | Single-thread comparison is fairer for ROM novelty |
| `snapshot_mkl` is the best POD backend | `~1.43 s` at 1T vs `~4.55 s` Eigen and `~23.05 s` BDCSVD | Use `snapshot_mkl` by default |
| Cholesky is best for current SPD matrices | `K_eff ~3.51 s` vs LU `~4.37 s` at 8T | Use `cholesky` unless it fails |
| Reduced solve is negligible | `~0.00003-0.00006 s` | Optimize projection, conversion, residual, and basis update instead |

## Settings Tested

### 1. LU + MKL Pardiso + 1 Thread

```text
KSP_TYPE=preonly
PC_TYPE=lu
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Timing:

```text
factorization:
  forward K_eff  ~ 9.03 s
  obj_sens Keff  ~ 9.41 s

solve after factorization:
  forward mean   ~ 0.268 s / solve
  obj_sens mean  ~ 0.270 s / solve
```

### 2. LU + MKL Pardiso + 8 Threads

```text
KSP_TYPE=preonly
PC_TYPE=lu
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

Timing:

```text
factorization:
  forward K_eff  ~ 4.37 s
  obj_sens Keff  ~ 4.72 s

solve after factorization:
  forward mean   ~ 0.084 s / solve
  obj_sens mean  ~ 0.087 s / solve
```

### 3. Cholesky + MKL Pardiso + 8 Threads

```text
KSP_TYPE=preonly
PC_TYPE=cholesky
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

Timing:

```text
factorization:
  forward K_eff  ~ 3.51 s
  obj_sens Keff  ~ 3.58 s

solve after factorization:
  forward mean   ~ 0.076 s / solve
  obj_sens mean  ~ 0.087 s / solve
```

## Main Conclusion

For the current 2D linear elastic dynamic model, the best tested single-node direct setting is:

```text
KSP_TYPE=preonly
PC_TYPE=cholesky
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

Compared with `LU + MKL Pardiso + 1 thread`:

```text
factorization: ~9.0 s  -> ~3.5 s
solve:         ~0.27 s -> ~0.08 s
```

So the improved setting reduces both factorization time and solve-after-factorization time.

## Effect Of Each Setting

### `PC_TYPE`

```text
cholesky
```

- Best for symmetric positive definite matrices.
- Current dynamic elastic matrices appear suitable.
- Faster than LU in this benchmark.
- Can reduce both factorization time and memory use.

```text
lu
```

- More general.
- Use when the matrix is not SPD or Cholesky fails.
- Slower for the current model.

### `PC_FACTOR_MAT_SOLVER_TYPE`

```text
mkl_pardiso
```

- Best current choice for single-node, single-rank direct solve.
- Uses MKL shared-memory parallelism.
- Strongly affected by `MKL_NUM_THREADS`.

```text
mumps
```

- Better choice for MPI/multinode direct solve.
- Slower than MKL Pardiso for the current single-node test.

### `OMP_NUM_THREADS` And `MKL_NUM_THREADS`

- Directly affect MKL Pardiso factorization and triangular solve/back-substitution.
- For single-rank Pardiso, `8` threads was much faster than `1`.
- Need benchmark per machine/model. Good values to test: `4`, `8`, `16`.

### `KSP_TYPE`

```text
preonly
```

- Direct-solver mode.
- Factorization is done in `set_factorization`.
- Repeated RHS solves use the factorized matrix.

```text
cg
```

- Iterative mode.
- Does not use direct LU/Cholesky factorization.
- Usually paired with `PC_TYPE=gamg`.

### `POD_BACKEND` And `ONLINE_BACKEND`

```text
snapshot_mkl
snapshot_eigen
bdcsvd
```

- Affects ROM basis update/SVD only.
- Does not affect PETSc factorization or FOM linear solve.

```text
incremental_mkl
incremental_eigen
```

- Affects online IncrementalSVD update only.
- Does not affect PETSc factorization or FOM linear solve.

## Thread Effect And ROM/FOM Fairness

Important point for ROM evaluation:

- Thread count is hardware-dependent.
- Different machines have different numbers of physical cores.
- A single-thread benchmark is often a fairer baseline when the goal is to isolate the ROM method itself.
- Multi-threading is still very useful for production runs, but it is a solver/backend optimization, not the ROM novelty.

In the current code, FOM with `mkl_pardiso` benefits very strongly from threads, while ROM basis kernels benefit more mildly.

This means:

```text
multi-thread FOM becomes extremely fast,
so the measured ROM/FOM speedup ratio becomes smaller,
even though total runtime improves.
```

For method comparison, keep a separate single-thread table.
For production performance, use the best multi-thread setting.

## Model Used For Scaling Notes

Unless stated otherwise:

```text
mesh       = 1200 x 400
elements   = 480000
ndof       = about 963202
NP         = 1
fem_type   = plane_stress
```

## FOM Direct Solver Timing

### Single-Thread Baseline

```text
KSP_TYPE=preonly
PC_TYPE=lu
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

Measured with `nstep=20`, `max_iter=1`:

```text
factorization:
  forward K_eff  ~ 9.03 s
  obj_sens Keff  ~ 9.41 s

solve after factorization:
  forward mean   ~ 0.268 s / solve
  obj_sens mean  ~ 0.270 s / solve
```

### Multi-Thread Direct Solver

```text
KSP_TYPE=preonly
PC_TYPE=lu
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

```text
factorization:
  forward K_eff  ~ 4.37 s
  obj_sens Keff  ~ 4.72 s

solve after factorization:
  forward mean   ~ 0.084 s / solve
  obj_sens mean  ~ 0.087 s / solve
```

```text
KSP_TYPE=preonly
PC_TYPE=cholesky
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

```text
factorization:
  forward K_eff  ~ 3.51 s
  obj_sens Keff  ~ 3.58 s

solve after factorization:
  forward mean   ~ 0.076 s / solve
  obj_sens mean  ~ 0.087 s / solve
```

### Factorization-Only Thread/Solver Notes

The following factorization-only checks were also observed for the same `1200 x 400` model:

| Solver | Threads | Forward `K_eff` Factorization |
|---|---:|---:|
| `lu + mkl_pardiso` | 1 | `~9.03 s` |
| `lu + mkl_pardiso` | 4 | `~4.84 s` |
| `lu + mkl_pardiso` | 8 | `~4.37 s` |
| `cholesky + mkl_pardiso` | 1 | `~6.21 s` |
| `cholesky + mkl_pardiso` | 4 | `~3.84 s` |
| `cholesky + mkl_pardiso` | 8 | `~3.51 s` |

Takeaway:

```text
FOM direct solve benefits strongly from both:
  1. cholesky instead of lu
  2. MKL/Pardiso threading
```

This is good for production runtime but can reduce the apparent ROM speedup ratio.

## ROM Basis Update Timing

Benchmark command pattern:

```text
basis_update_benchmark --ndof 963202 --nsnapshot 100 --max_rank 20 --online_updates 20
```

### 1 Thread

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

| Component | Time |
|---|---:|
| `offline_snapshot_eigen` | `~4.55 s` |
| `offline_snapshot_mkl` | `~1.43 s` |
| `offline_bdcsvd` | `~23.05 s` |
| `online_eigen` | `~4.41 s / 20 updates` |
| `online_mkl` | `~3.90 s / 20 updates` |

### 4 Threads

```text
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

| Component | Time |
|---|---:|
| `offline_snapshot_eigen` | `~3.81 s` |
| `offline_snapshot_mkl` | `~1.09 s` |
| `offline_bdcsvd` | `~13.72 s` |
| `online_eigen` | `~4.07 s / 20 updates` |
| `online_mkl` | `~3.35 s / 20 updates` |

### 8 Threads

```text
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

| Component | Time |
|---|---:|
| `offline_snapshot_eigen` | `~3.88 s` |
| `offline_snapshot_mkl` | `~1.02 s` |
| `offline_bdcsvd` | `~12.79 s` |
| `online_eigen` | `~4.54 s / 20 updates` |
| `online_mkl` | `~3.27 s / 20 updates` |

Subspace differences were small:

```text
offline_mkl_subspace_error ~ 1e-7
online_mkl_subspace_error  ~ 1e-7
```

Takeaway:

```text
offline_snapshot_mkl is much faster than snapshot_eigen and bdcsvd.
online_mkl is moderately faster than online_eigen.
threading helps ROM basis update, but much less dramatically than it helps FOM Pardiso.
```

This is a positive result for single-thread fairness:

```text
FOM loses much more speed when going from 8 threads to 1 thread.
ROM basis update loses less speed.
```

Therefore, for method-focused ROM comparison, `1 thread` is a defensible and clearer baseline.

## ROM Component Timing From Existing Test2 CSV

The following values came from `case_study/test2/result`.
They are useful to understand the scale of ROM overhead, but if the case was run before the latest `snapshot_mkl/incremental_mkl` changes, the basis update numbers should be treated as historical.

Forward ROM components:

| Component | Mean Time |
|---|---:|
| `petsc_to_eigen_mat` | `~0.55 s` |
| `projection V'AV` | `~0.32 s` |
| `reduced_rhs V'rhs` | `~0.010 s` |
| `reduced_solve` | `~0.00006 s` |
| `reconstruct V*q` | `~0.014 s` |
| `residual` | `~0.020 s` |
| `FOM fallback solve` | `~0.31 s` |

Objective-sensitivity ROM components:

| Component | Mean Time |
|---|---:|
| `petsc_to_eigen_mat` | `~0.67 s` |
| `projection V'AV` | `~0.32 s` |
| `reduced_rhs V'rhs` | `~0.010 s` |
| `reduced_solve` | `~0.00003 s` |
| `reconstruct V*q` | `~0.013 s` |
| `residual` | `~0.021 s` |
| `FOM fallback solve` | `~0.32 s` |

Current likely ROM optimization priorities after MKL POD/online update:

```text
1. projection V'AV
2. PETSc Mat -> Eigen conversion
3. residual check
4. online IncrementalSVD update
```

The reduced solve itself is negligible.

## Recommended Benchmark Split

For paper/method evaluation:

```text
NP=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
KSP_TYPE=preonly
PC_TYPE=cholesky or lu, but keep same for FOM and ROM
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
POD_BACKEND=snapshot_mkl
ONLINE_BACKEND=incremental_mkl
```

Reason:

```text
single-thread comparison reduces hardware dependence
and avoids making FOM artificially dominant through thread-level solver optimization.
```

For production runtime:

```text
NP=1
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
KSP_TYPE=preonly
PC_TYPE=cholesky
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
POD_BACKEND=snapshot_mkl
ONLINE_BACKEND=incremental_mkl
```

Reason:

```text
best total runtime observed so far on this machine/model.
```

## Practical Rule

Use this for current single-node direct testing:

```text
KSP_TYPE=preonly
PC_TYPE=cholesky
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

Use this when Cholesky fails:

```text
KSP_TYPE=preonly
PC_TYPE=lu
PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
```

Use this for multinode direct solve:

```text
KSP_TYPE=preonly
PC_TYPE=lu
PC_FACTOR_MAT_SOLVER_TYPE=mumps
```

Use this for large distributed iterative runs:

```text
KSP_TYPE=cg
PC_TYPE=gamg
PC_FACTOR_MAT_SOLVER_TYPE=
```
