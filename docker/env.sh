#!/usr/bin/env bash

# Shared Docker runtime for Large_scale_TO.
# Source this file inside the container before configuring, building, or running.

large_scale_to_use_toolchain() {
    local toolchain="${1:-${TO_TOOLCHAIN:-gcc13}}"

    case "$toolchain" in
        gcc11)
            export TO_TOOLCHAIN="gcc11"
            export CC="${CC:-gcc}"
            export CXX="${CXX:-g++}"
            export FC="${FC:-gfortran}"
            export MPICH_HOME="${MPICH_HOME:-/home/votrongnam/opt/mpich-4.0-ch3-cxx-gcc11}"
            export PETSC_PREFIX="${PETSC_PREFIX:-/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl2025-mumps-gcc11}"
            export BUILD_DIR="${BUILD_DIR:-build-docker-release-gcc11}"
            ;;
        gcc13)
            export TO_TOOLCHAIN="gcc13"
            export CC="${CC:-gcc-13}"
            export CXX="${CXX:-g++-13}"
            export FC="${FC:-gfortran-13}"
            export MPICH_HOME="${MPICH_HOME:-/home/votrongnam/opt/mpich-4.0-ch3-cxx-gcc13}"
            export PETSC_PREFIX="${PETSC_PREFIX:-/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl2025-mumps-gcc13}"
            export BUILD_DIR="${BUILD_DIR:-build-docker-release-gcc13}"
            ;;
        *)
            echo "Unknown TO_TOOLCHAIN: $toolchain" >&2
            echo "Expected: gcc11 or gcc13" >&2
            return 2
            ;;
    esac

    export MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/2025.0}"
    export PATH="${MPICH_HOME}/bin:${PATH}"
    export PKG_CONFIG_PATH="${PETSC_PREFIX}/lib/pkgconfig"
    export LD_LIBRARY_PATH="${PETSC_PREFIX}/lib:${MPICH_HOME}/lib:${MKLROOT}/lib:${MKLROOT}/lib/intel64:${LD_LIBRARY_PATH:-}"

    export MPIEXEC_PORT_RANGE="${MPIEXEC_PORT_RANGE:-40000:41000}"
    export MPICH_PORT_RANGE="${MPICH_PORT_RANGE:-40000:41000}"
    export HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
    export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
    export OMP_PLACES="${OMP_PLACES:-cores}"
}

large_scale_to_use_toolchain "${TO_TOOLCHAIN:-gcc13}"
