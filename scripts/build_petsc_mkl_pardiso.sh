#!/usr/bin/env bash
set -euo pipefail

PETSC_SRC="${PETSC_SRC:-/home/votrongnam/src/petsc-3.19.6}"
PETSC_ARCH_NAME="${PETSC_ARCH_NAME:-arch-mpich-ch3-cxx-mkl-opt}"
PETSC_PREFIX="${PETSC_PREFIX:-/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl}"
MPICH_DIR="${MPICH_DIR:-/home/votrongnam/opt/mpich-4.0-ch3-cxx}"
MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/2024.2}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/votrongnam/PROJECT/Large_scale_TO}"

export MKLROOT
export PATH="${MPICH_DIR}/bin:${PATH}"

cd "${PETSC_SRC}"

./configure \
  "PETSC_ARCH=${PETSC_ARCH_NAME}" \
  "--prefix=${PETSC_PREFIX}" \
  "--with-mpi-dir=${MPICH_DIR}" \
  "--with-mpiexec=${PROJECT_ROOT}/scripts/mpiexec_mpich_ch3_cxx.sh" \
  --with-debugging=0 \
  --with-shared-libraries=1 \
  "--with-blaslapack-dir=${MKLROOT}" \
  --with-mkl_pardiso=1 \
  --with-openmp=1 \
  --with-x=0 \
  COPTFLAGS="-O3 -march=native" \
  CXXOPTFLAGS="-O3 -march=native"

make "PETSC_ARCH=${PETSC_ARCH_NAME}" all
make "PETSC_ARCH=${PETSC_ARCH_NAME}" install

echo "PETSc MKL/PARDISO installed at ${PETSC_PREFIX}"
