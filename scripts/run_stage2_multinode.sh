#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPICH_HOME="${MPICH_HOME:-/home/votrongnam/opt/mpich-4.0-ch3-cxx}"
PORT_RANGE="${MPIEXEC_PORT_RANGE:-45010:46000}"
HOSTS="${1:-localhost,192.168.11.21}"
NP="${2:-2}"
BINARY="${BINARY:-$ROOT/build-mpich-release/stage2_dynamic_forward}"

if [[ $# -ge 1 ]]; then shift; fi
if [[ $# -ge 1 ]]; then shift; fi
APP_ARGS=("$@")

cd "$ROOT"

if [[ ! -x "$BINARY" ]]; then
  echo "Missing executable: $BINARY" >&2
  exit 1
fi

IFS=',' read -ra HOST_ARRAY <<< "$HOSTS"
for host in "${HOST_ARRAY[@]}"; do
  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || -z "$host" ]]; then
    continue
  fi
  "$ROOT/scripts/ssh_worker_2222.sh" "$host" "mkdir -p '$ROOT/build-mpich-release'"
  rsync -az -e "$ROOT/scripts/ssh_worker_2222.sh" "$BINARY" "$host:$BINARY"
done

export MPIEXEC_PORT_RANGE="$PORT_RANGE"
export MPICH_PORT_RANGE="$PORT_RANGE"
export HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl}"
export LD_LIBRARY_PATH="/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx/lib:/home/votrongnam/opt/mpich-4.0-ch3-cxx/lib:${LD_LIBRARY_PATH:-}"

"$MPICH_HOME/bin/mpiexec" \
  -iface eth1 \
  -launcher ssh \
  -launcher-exec "$ROOT/scripts/ssh_worker_2222.sh" \
  -wdir "$ROOT" \
  -n "$NP" \
  -hosts "$HOSTS" \
  "$BINARY" \
  "${APP_ARGS[@]}"
