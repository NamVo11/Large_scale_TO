#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NP="${1:-2}"

bash "$ROOT/scripts/make_manifest.sh" simple_3d_cantilever "local_smoke_n${NP}" cg gamg 1e-10 0

cd "$ROOT"
mpirun -n "$NP" "$ROOT/build-release/stage1_petsc_backend"

echo
cat "$ROOT/results/local_smoke_n${NP}/solver_info.txt"
