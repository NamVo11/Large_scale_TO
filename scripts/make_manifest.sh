#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CASE_NAME="${1:-simple_3d_cantilever}"
OUT_NAME="${2:-local_smoke}"
KSP_TYPE="${3:-cg}"
PC_TYPE="${4:-gamg}"
RTOL="${5:-1e-10}"
NO_HISTORY="${6:-0}"
MAX_IT="${7:-20000}"

cat > "$ROOT/debug_manifest.txt" <<EOF
case=$ROOT/cases/$CASE_NAME
out=$ROOT/results/$OUT_NAME
no_history=$NO_HISTORY
ksp_type=$KSP_TYPE
pc_type=$PC_TYPE
ksp_rtol=$RTOL
ksp_max_it=$MAX_IT
EOF

echo "Wrote $ROOT/debug_manifest.txt"
cat "$ROOT/debug_manifest.txt"
