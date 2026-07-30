#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Launch stage2_dynamic_forward across multiple Ubuntu nodes with MPICH.
# Usage:
#   bash scripts/run_stage2_multinode.sh localhost,192.168.11.21 8 [stage2 args...]

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

LOCAL_MPI_IP="${MPI_MASTER_IP:-$(ip -4 -o addr show eth1 | awk '{split($4,a,"/"); print a[1]; exit}')}"
if [[ -z "$LOCAL_MPI_IP" ]]; then
    echo "Error: cannot determine local MPI IP on eth1. Set MPI_MASTER_IP explicitly." >&2
    exit 1
fi

RANK_IFACE_ARGS=()
HOST_IPS=()
for host in "${HOST_ARRAY[@]}"; do
    host="${host//[[:space:]]/}"
    if [[ -z "$host" ]]; then
        continue
    fi

    if [[ "$host" == "localhost" || "$host" == "127.0.0.1" ]]; then
        HOST_IPS+=("$LOCAL_MPI_IP")
    else
        HOST_IPS+=("$host")
    fi
done

if [[ "${#HOST_IPS[@]}" -eq 0 ]]; then
    echo "Error: HOSTS is empty after parsing." >&2
    exit 1
fi

for ((rank = 0; rank < NP; ++rank)); do
    host_index=$((rank % ${#HOST_IPS[@]}))
    RANK_IFACE_ARGS+=(
        -genv "MPICH_INTERFACE_HOSTNAME_R${rank}" "${HOST_IPS[$host_index]}"
    )
done

echo
echo "Launching multi-node MPI job"
echo "  hosts                 = $HOSTS"
echo "  total MPI processes   = $NP"
echo "  binary                = $BINARY"
echo "  working directory     = $ROOT"
echo "  OMP threads/process   = $OMP_NUM_THREADS"
echo "  MKL threads/process   = $MKL_NUM_THREADS"
echo "  MPI rank IP mapping   = ${HOST_IPS[*]}"
echo

exec "$MPICH_HOME/bin/mpiexec" \
    -iface eth1 \
    -launcher ssh \
    -launcher-exec "$ROOT/scripts/ssh_worker_2222.sh" \
    -wdir "$ROOT" \
    -n "$NP" \
    -hosts "$HOSTS" \
    -genv LD_LIBRARY_PATH "$LD_LIBRARY_PATH" \
    -genv MPIEXEC_PORT_RANGE "$MPIEXEC_PORT_RANGE" \
    -genv MPICH_PORT_RANGE "$MPICH_PORT_RANGE" \
    -genv OMP_NUM_THREADS "$OMP_NUM_THREADS" \
    -genv MKL_NUM_THREADS "$MKL_NUM_THREADS" \
    -genv OMP_PROC_BIND "$OMP_PROC_BIND" \
    -genv OMP_PLACES "$OMP_PLACES" \
    -genv HWLOC_COMPONENTS "$HWLOC_COMPONENTS" \
    "${RANK_IFACE_ARGS[@]}" \
    "$BINARY" \
    "${APP_ARGS[@]}"
