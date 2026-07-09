# PETSc + MKL + MUMPS Setup

Purpose: build one PETSc stack that keeps the current PETSc iterative solvers,
adds MKL/PARDISO for fast one-node direct solves, and adds MUMPS for distributed
MPI direct solves when we need multi-node LU.

Use the same versions and paths on master and worker. Do not mix system MPI,
apt PETSc, OpenMPI, or different MPICH/PETSc prefixes.

## Target Paths

```text
MPICH      /home/votrongnam/opt/mpich-4.0-ch3-cxx
MKL        /opt/intel/oneapi/mkl/2024.2
PETSc src  /home/votrongnam/src/petsc-3.19.6
PETSc new  /home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl-mumps
Project    /home/votrongnam/PROJECT/Large_scale_TO
```

## 1. Check Prerequisites On Each Machine

```bash
whoami
/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpicxx -show
ls /opt/intel/oneapi/mkl/2024.2/lib
ls /home/votrongnam/src/petsc-3.19.6
```

If one machine differs, fix it before building. Multi-node PETSc is very
sensitive to MPI/library mismatch.

## 2. Build PETSc + MKL + MUMPS

Run on each machine, or run on master and copy the final installed prefix to the
worker only if OS, CPU architecture, MPICH, and MKL path match.

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO
bash scripts/build_petsc_mkl_mumps.sh
```

Useful overrides:

```bash
PETSC_PREFIX=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl-mumps \
BUILD_JOBS=8 \
bash scripts/build_petsc_mkl_mumps.sh
```

## 3. Configure Project Against The New PETSc

```bash
cd /home/votrongnam/PROJECT/Large_scale_TO

PKG_CONFIG_PATH=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl-mumps/lib/pkgconfig \
cmake -S . -B build-mpich-mkl-mumps-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DCMAKE_CXX_COMPILER=/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpicxx \
  -DPETSC_PKG=PETSc

cmake --build build-mpich-mkl-mumps-release --target stage2_dynamic_forward -j
```

## 4. Solver Modes To Test

One node, direct MKL/PARDISO, one MPI rank with OpenMP threads:

```bash
NP=1 \
BUILD_DIR=build-mpich-mkl-mumps-release \
KSP_TYPE=preonly PC_TYPE=lu PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
bash scripts/run_stage2_case.sh
```

One or many nodes, iterative PETSc solve:

```bash
NP=8 HOSTS=localhost,192.168.11.21 \
BUILD_DIR=build-mpich-mkl-mumps-release \
KSP_TYPE=cg PC_TYPE=gamg PC_FACTOR_MAT_SOLVER_TYPE= \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
bash scripts/run_stage2_case.sh
```

Many nodes, distributed direct MUMPS:

```bash
NP=8 HOSTS=localhost,192.168.11.21 \
BUILD_DIR=build-mpich-mkl-mumps-release \
KSP_TYPE=preonly PC_TYPE=lu PC_FACTOR_MAT_SOLVER_TYPE=mumps \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
bash scripts/run_stage2_case.sh
```

The current launcher scripts may still need standardization before the `HOSTS`
example above works through `run_stage2_case.sh`; this document defines the
intended interface.

## Common Errors

- `Unknown MatSolverType mumps`: project is still linked to PETSc without MUMPS,
  or `PKG_CONFIG_PATH` points to the old PETSc prefix.
- `Unknown MatSolverType mkl_pardiso`: PETSc was built without MKL/PARDISO, or
  `LD_LIBRARY_PATH` misses the MKL runtime path.
- MPI hangs or Hydra proxy mismatch: MPICH differs between nodes. Sync
  `/home/votrongnam/opt/mpich-4.0-ch3-cxx` and rebuild/reconfigure PETSc.
- Runtime library not found on worker: worker lacks the same PETSc/MKL path or
  `LD_LIBRARY_PATH` is not propagated by the launcher.
- Direct solve with `mkl_pardiso` on `NP>1`: this is not the intended distributed
  direct mode. Use `NP=1` for MKL/PARDISO, or use `mumps` for distributed LU.
- Slow multinode iterative run: set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`
  to avoid oversubscribing CPU cores.
