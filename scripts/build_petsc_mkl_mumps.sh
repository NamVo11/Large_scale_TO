#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Build PETSc against the project MPICH stack with MKL/PARDISO and MUMPS enabled.
# Usage:
#   Run manually on each Ubuntu machine:
#   bash scripts/build_petsc_mkl_mumps.sh

# Build PETSc with the existing MPICH stack, Intel MKL/PARDISO, and MUMPS.
# This script is intentionally a template: run it manually on each machine,
# or run once on master and rsync the installed prefix to worker after success.

PETSC_VERSION="${PETSC_VERSION:-3.19.6}"
PETSC_SRC="${PETSC_SRC:-/home/votrongnam/src/petsc-${PETSC_VERSION}}"
PETSC_ARCH_NAME="${PETSC_ARCH_NAME:-arch-mpich-ch3-cxx-mkl-mumps-opt}"
PETSC_PREFIX="${PETSC_PREFIX:-/home/votrongnam/opt/petsc-${PETSC_VERSION}-mpich-ch3-cxx-mkl-mumps}"
MPICH_DIR="${MPICH_DIR:-/home/votrongnam/opt/mpich-4.0-ch3-cxx}"
MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/2024.2}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/votrongnam/PROJECT/Large_scale_TO}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

export MKLROOT
export PATH="${MPICH_DIR}/bin:${PATH}"

if [[ ! -d "${PETSC_SRC}" ]]; then
  echo "Missing PETSc source: ${PETSC_SRC}" >&2
  echo "Download/extract petsc-${PETSC_VERSION} under /home/votrongnam/src first." >&2
  exit 1
fi
if [[ ! -x "${MPICH_DIR}/bin/mpicxx" ]]; then
  echo "Missing MPICH mpicxx: ${MPICH_DIR}/bin/mpicxx" >&2
  exit 1
fi
if [[ ! -d "${MKLROOT}" ]]; then
  echo "Missing MKLROOT: ${MKLROOT}" >&2
  echo "Install Intel oneAPI MKL or set MKLROOT to the matching path." >&2
  exit 1
fi

unset PETSC_DIR PETSC_ARCH
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
  --download-mumps=1 \
  --download-scalapack=1 \
  --download-metis=1 \
  --download-parmetis=1 \
  --with-x=0 \
  --with-batch=1 \
  COPTFLAGS="-O3 -march=native" \
  CXXOPTFLAGS="-O3 -march=native"

make "PETSC_ARCH=${PETSC_ARCH_NAME}" all -j"${BUILD_JOBS}"
make "PETSC_ARCH=${PETSC_ARCH_NAME}" install

cat <<MSG
PETSc MKL/MUMPS installed at:
  ${PETSC_PREFIX}

Use for CMake:
  PKG_CONFIG_PATH=${PETSC_PREFIX}/lib/pkgconfig

Use at runtime:
  LD_LIBRARY_PATH=${PETSC_PREFIX}/lib:${MPICH_DIR}/lib:${MKLROOT}/lib:\${LD_LIBRARY_PATH:-}
MSG
