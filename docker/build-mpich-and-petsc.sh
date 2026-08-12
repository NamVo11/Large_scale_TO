#!/usr/bin/env bash
set -euo pipefail

MPICH_VERSION="${MPICH_VERSION:-4.0}"
PETSC_VERSION="${PETSC_VERSION:-3.19.6}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
SRC_DIR="${SRC_DIR:-/home/votrongnam/src}"
OPT_DIR="${OPT_DIR:-/home/votrongnam/opt}"
MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/2025.0}"

mkdir -p "$SRC_DIR" "$OPT_DIR"

download_source() {
    local url="$1"
    local archive="$2"

    if [[ ! -f "$SRC_DIR/$archive" ]]; then
        wget -O "$SRC_DIR/$archive" "$url"
    fi
}

prepare_sources() {
    download_source \
        "https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz" \
        "mpich-${MPICH_VERSION}.tar.gz"
    download_source \
        "https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz" \
        "petsc-${PETSC_VERSION}.tar.gz"

    if [[ ! -d "$SRC_DIR/mpich-${MPICH_VERSION}" ]]; then
        tar -C "$SRC_DIR" -xf "$SRC_DIR/mpich-${MPICH_VERSION}.tar.gz"
    fi
    if [[ ! -d "$SRC_DIR/petsc-${PETSC_VERSION}" ]]; then
        tar -C "$SRC_DIR" -xf "$SRC_DIR/petsc-${PETSC_VERSION}.tar.gz"
    fi
}

build_mpich() {
    local toolchain="$1"
    local cc="$2"
    local cxx="$3"
    local fc="$4"
    local device="$5"
    local prefix="$6"
    local build_dir="$SRC_DIR/mpich-${MPICH_VERSION}-${toolchain}-build"

    if [[ -x "$prefix/bin/mpicxx" ]]; then
        echo "MPICH already installed for ${toolchain}: $prefix"
        return
    fi

    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    cd "$build_dir"

    "$SRC_DIR/mpich-${MPICH_VERSION}/configure" \
        "--prefix=${prefix}" \
        "CC=${cc}" \
        "CXX=${cxx}" \
        "FC=${fc}" \
        "F77=${fc}" \
        "FCFLAGS=-fallow-argument-mismatch" \
        "FFLAGS=-fallow-argument-mismatch" \
        --enable-cxx \
        --enable-shared \
        "--with-device=${device}"

    make -j"$BUILD_JOBS"
    make install
}

build_petsc() {
    local toolchain="$1"
    local cc="$2"
    local cxx="$3"
    local fc="$4"
    local mpich_prefix="$5"
    local petsc_prefix="$6"
    local arch="arch-mpich-ch3-cxx-mkl2025-mumps-${toolchain}-opt"

    if [[ -f "$petsc_prefix/lib/pkgconfig/PETSc.pc" ]]; then
        echo "PETSc already installed for ${toolchain}: $petsc_prefix"
        return
    fi

    cd "$SRC_DIR/petsc-${PETSC_VERSION}"
    unset PETSC_DIR PETSC_ARCH
    export PATH="${mpich_prefix}/bin:${PATH}"
    export CC CXX FC

    ./configure \
        "PETSC_ARCH=${arch}" \
        "--prefix=${petsc_prefix}" \
        "--with-mpi-dir=${mpich_prefix}" \
        "--with-mpiexec=${mpich_prefix}/bin/mpiexec" \
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
        COPTFLAGS="-O3 -mtune=generic" \
        CXXOPTFLAGS="-O3 -mtune=generic" \
        FOPTFLAGS="-O3 -mtune=generic"

    make "PETSC_ARCH=${arch}" all -j"$BUILD_JOBS"
    make "PETSC_ARCH=${arch}" install
}

main() {
    if [[ ! -d "$MKLROOT" ]]; then
        echo "Missing MKLROOT: $MKLROOT" >&2
        exit 1
    fi

    prepare_sources

    build_mpich \
        gcc11 gcc g++ gfortran ch3:nemesis \
        "$OPT_DIR/mpich-4.0-ch3-cxx-gcc11"

    ln -sfn "$OPT_DIR/mpich-4.0-ch3-cxx-gcc11" "$OPT_DIR/mpich-4.0-ch3-cxx"

    build_mpich \
        gcc13 gcc-13 g++-13 gfortran-13 ch3:sock \
        "$OPT_DIR/mpich-4.0-ch3-cxx-gcc13"

    build_petsc \
        gcc11 gcc g++ gfortran \
        "$OPT_DIR/mpich-4.0-ch3-cxx-gcc11" \
        "$OPT_DIR/petsc-3.19.6-mpich-ch3-cxx-mkl2025-mumps-gcc11"

    build_petsc \
        gcc13 gcc-13 g++-13 gfortran-13 \
        "$OPT_DIR/mpich-4.0-ch3-cxx-gcc13" \
        "$OPT_DIR/petsc-3.19.6-mpich-ch3-cxx-mkl2025-mumps-gcc13"
}

main "$@"
