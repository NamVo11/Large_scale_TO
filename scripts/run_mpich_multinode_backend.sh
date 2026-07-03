#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MPICH_HOME="${MPICH_HOME:-/home/votrongnam/opt/mpich-4.0-ch3-cxx}"
PORT_RANGE="${MPIEXEC_PORT_RANGE:-45010:46000}"
HOSTS="${1:-localhost,192.168.11.21}"
NP="${2:-2}"
BINARY="${BINARY:-$ROOT/build-mpich-release/stage1_petsc_backend}"

cd "$ROOT"

if [[ ! -x "$BINARY" ]]; then
  echo "Missing executable: $BINARY" >&2
  exit 1
fi

if [[ ! -f "$ROOT/debug_manifest.txt" ]]; then
  echo "Missing manifest: $ROOT/debug_manifest.txt" >&2
  exit 1
fi

IFS=',' read -ra HOST_ARRAY <<< "$HOSTS"
for host in "${HOST_ARRAY[@]}"; do
  if [[ "$host" == "localhost" || "$host" == "127.0.0.1" || -z "$host" ]]; then
    continue
  fi
  rsync -az -e "$ROOT/scripts/ssh_worker_2222.sh" "$ROOT/debug_manifest.txt" "$host:$ROOT/debug_manifest.txt"
done

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
  "$BINARY"
