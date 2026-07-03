#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPICH_HOME="${MPICH_HOME:-/home/votrongnam/opt/mpich-4.0-ch3-cxx}"
PORT_RANGE="${MPIEXEC_PORT_RANGE:-45010:46000}"
HOSTS="${1:-localhost,192.168.11.21}"
NP="${2:-2}"

cd "$ROOT"

export MPIEXEC_PORT_RANGE="$PORT_RANGE"
export MPICH_PORT_RANGE="$PORT_RANGE"
export HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl}"

"$MPICH_HOME/bin/mpiexec" \
  -iface eth1 \
  -launcher ssh \
  -launcher-exec "$ROOT/scripts/ssh_worker_2222.sh" \
  -wdir "$ROOT" \
  -n "$NP" \
  -hosts "$HOSTS" \
  /bin/hostname
