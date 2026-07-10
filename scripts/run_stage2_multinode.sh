#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT/scripts/env_petsc_mpi.sh"
stage2_setup_runtime_env

PORT_RANGE="$MPIEXEC_PORT_RANGE"

HOSTS="${1:-localhost,192.168.11.21}"
NP="${2:-2}"

BUILD_DIR="${BUILD_DIR:-build-mpich-mkl2025-mumps-release}"

if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="$ROOT/$BUILD_DIR"
fi

BINARY="${BINARY:-$BUILD_DIR/stage2_dynamic_forward}"

if [[ $# -ge 1 ]]; then
    shift
fi

if [[ $# -ge 1 ]]; then
    shift
fi

APP_ARGS=("$@")

cd "$ROOT"

if [[ ! -x "$BINARY" ]]; then
    echo "Error: missing executable:" >&2
    echo "  $BINARY" >&2
    exit 1
fi

# ============================================================
# Synchronize executable with remote hosts
# ============================================================

IFS=',' read -ra HOST_ARRAY <<< "$HOSTS"

for host in "${HOST_ARRAY[@]}"; do
    host="${host//[[:space:]]/}"

    if [[ -z "$host" ||
          "$host" == "localhost" ||
          "$host" == "127.0.0.1" ]]; then
        continue
    fi

    echo "Preparing remote host: $host"

    "$ROOT/scripts/ssh_worker_2222.sh" \
        "$host" \
        "mkdir -p '$ROOT/$BUILD_DIR'"

    echo "Copying executable to: $host:$BINARY"

    rsync -az \
        -e "$ROOT/scripts/ssh_worker_2222.sh" \
        "$BINARY" \
        "$host:$BINARY"
done

# ============================================================
# MPI and runtime environment
# ============================================================

export MPIEXEC_PORT_RANGE="$PORT_RANGE"
export MPICH_PORT_RANGE="$PORT_RANGE"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo
echo "Launching multi-node MPI job"
echo "  hosts                 = $HOSTS"
echo "  total MPI processes   = $NP"
echo "  binary                = $BINARY"
echo "  working directory     = $ROOT"
echo "  OMP threads/process   = $OMP_NUM_THREADS"
echo "  MKL threads/process   = $MKL_NUM_THREADS"
echo

exec "$MPICH_HOME/bin/mpiexec" \
    -iface eth1 \
    -launcher ssh \
    -launcher-exec "$ROOT/scripts/ssh_worker_2222.sh" \
    -wdir "$ROOT" \
    -n "$NP" \
    -hosts "$HOSTS" \
    -genv LD_LIBRARY_PATH "$LD_LIBRARY_PATH" \
    -genv OMP_NUM_THREADS "$OMP_NUM_THREADS" \
    -genv MKL_NUM_THREADS "$MKL_NUM_THREADS" \
    -genv OMP_PROC_BIND "$OMP_PROC_BIND" \
    -genv OMP_PLACES "$OMP_PLACES" \
    -genv HWLOC_COMPONENTS "$HWLOC_COMPONENTS" \
    "$BINARY" \
    "${APP_ARGS[@]}"
