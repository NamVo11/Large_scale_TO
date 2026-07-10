#!/usr/bin/env bash

# Shared runtime environment for stage2 on both master and worker.
# Source this file from scripts, tasks, or a clean shell before running.

stage2_unset_runtime_env() {
    unset PETSC_DIR PETSC_ARCH
    unset PETSC_PREFIX MPICH_HOME MKLROOT
    unset MPIEXEC_PORT_RANGE MPICH_PORT_RANGE HWLOC_COMPONENTS
    unset OMP_NUM_THREADS MKL_NUM_THREADS OMP_PROC_BIND OMP_PLACES
    unset LD_LIBRARY_PATH PKG_CONFIG_PATH
}

stage2_setup_runtime_env() {
    export PETSC_PREFIX="${PETSC_PREFIX:-/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl2025-mumps}"
    export MPICH_HOME="${MPICH_HOME:-/home/votrongnam/opt/mpich-4.0-ch3-cxx}"
    export MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/2025.0}"

    export MPIEXEC_PORT_RANGE="${MPIEXEC_PORT_RANGE:-40000:41000}"
    export MPICH_PORT_RANGE="${MPICH_PORT_RANGE:-40000:41000}"
    export HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl}"

    export PKG_CONFIG_PATH="${PETSC_PREFIX}/lib/pkgconfig"
    export LD_LIBRARY_PATH="${PETSC_PREFIX}/lib:${MPICH_HOME}/lib:${MKLROOT}/lib"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    stage2_unset_runtime_env
    stage2_setup_runtime_env
    env | grep -E '^(PETSC_PREFIX|MPICH_HOME|MKLROOT|MPIEXEC_PORT_RANGE|MPICH_PORT_RANGE|HWLOC_COMPONENTS|PKG_CONFIG_PATH|LD_LIBRARY_PATH)=' | sort
fi
