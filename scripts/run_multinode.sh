#!/usr/bin/env bash
set -euo pipefail

HOSTFILE="${1:-scripts/hosts.txt}"
NP="${2:-2}"
CASE_NAME="${3:-simple_3d_cantilever}"
OUT_NAME="${4:-multinode_smoke_n${NP}}"
KSP_TYPE="${5:-cg}"
PC_TYPE="${6:-gamg}"
RTOL="${7:-1e-10}"
NO_HISTORY="${8:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash "$ROOT/scripts/make_manifest.sh" "$CASE_NAME" "$OUT_NAME" "$KSP_TYPE" "$PC_TYPE" "$RTOL" "$NO_HISTORY"

echo "Launching np=$NP with hostfile=$HOSTFILE"
/usr/bin/time -f 'wall_time=%e s' \
  mpirun --hostfile "$HOSTFILE" -np "$NP" "$ROOT/build-release/stage1_petsc_backend"

echo
cat "$ROOT/results/$OUT_NAME/solver_info.txt"
