#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   VS Code/task launcher for JSON-driven stage2 runs.
#   Model/solver/output parameters are read by C++ from INPUT_JSON.
#   The shell keeps only toolchain choices needed before the executable starts.
#   MPI process count and hosts are read from Run.mpi in INPUT_JSON.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/env_petsc_mpi.sh"
stage2_setup_runtime_env

if [[ -n "${INPUT_JSONS:-}" ]]; then
    echo
    echo "============================================================"
    echo "Running JSON batch"
    echo "============================================================"
    echo "$INPUT_JSONS" | tr ' ' '\n' | sed '/^$/d' | nl -w2 -s'. '
    echo "============================================================"
    echo

    case_id=0
    for cfg in $INPUT_JSONS; do
        case_id=$((case_id + 1))
        echo
        echo "============================================================"
        echo "Batch case $case_id"
        echo "INPUT_JSON=$cfg"
        echo "============================================================"
        echo
        INPUT_JSON="$cfg" INPUT_JSONS= bash "$ROOT/scripts/run_stage2_case.sh"
    done

    echo
    echo "============================================================"
    echo "JSON batch finished"
    echo "============================================================"
    exit 0
fi

INPUT_JSON="${INPUT_JSON:?INPUT_JSON is not set}"
if [[ "$INPUT_JSON" != /* ]]; then
    INPUT_JSON="$ROOT/$INPUT_JSON"
fi
if [[ ! -f "$INPUT_JSON" ]]; then
    echo "Error: input JSON file does not exist:" >&2
    echo "  $INPUT_JSON" >&2
    exit 1
fi

BUILD_DIR="${BUILD_DIR:-build-mpich-release-gcc13}"
if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="$ROOT/$BUILD_DIR"
fi
BINARY="${BINARY:-$BUILD_DIR/stage2_dynamic_forward}"
JSON_MPI="$(python3 - "$INPUT_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)

mpi = cfg.get("Run", {}).get("mpi", {})
print(str(mpi.get("np", "")))
print(str(mpi.get("hosts", "")))
PY
)"
JSON_NP="$(printf '%s\n' "$JSON_MPI" | sed -n '1p')"
JSON_HOSTS="$(printf '%s\n' "$JSON_MPI" | sed -n '2p')"
NP="${JSON_NP:-${NP:-1}}"
HOSTS="${JSON_HOSTS:-${HOSTS:-localhost}}"

if [[ ! -x "$BINARY" ]]; then
    echo "Executable not found. Building:"
    echo "  $BINARY"
    cmake --build "$BUILD_DIR" --target stage2_dynamic_forward -j "${BUILD_JOBS:-8}"
fi

if [[ ! -x "$BINARY" ]]; then
    echo "Error: executable is missing:" >&2
    echo "  $BINARY" >&2
    exit 1
fi

APP_ARGS=(-input_json "$INPUT_JSON")

echo
echo "============================================================"
echo "Running stage2 JSON case"
echo "============================================================"
echo "  input_json                = $INPUT_JSON"
echo "  binary                    = $BINARY"
echo "  hosts                     = $HOSTS"
echo "  MPI processes             = $NP"
echo "============================================================"
echo

NORMALIZED_HOSTS="${HOSTS//[[:space:]]/}"

case "$NORMALIZED_HOSTS" in
    "" | "localhost" | "127.0.0.1")
        echo "Execution mode: single node"
        exec "$ROOT/scripts/mpiexec_mpich_ch3_cxx.sh" \
            -n "$NP" \
            "$BINARY" \
            "${APP_ARGS[@]}"
        ;;
    *)
        echo "Execution mode: multi-node"
        exec "$ROOT/scripts/run_stage2_multinode.sh" \
            "$NORMALIZED_HOSTS" \
            "$NP" \
            "${APP_ARGS[@]}"
        ;;
esac
