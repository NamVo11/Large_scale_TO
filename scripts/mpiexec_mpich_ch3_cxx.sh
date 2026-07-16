#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run MPICH mpiexec with the stage2 runtime environment already loaded.
# Usage:
#   scripts/mpiexec_mpich_ch3_cxx.sh -n 1 build-mpich-release/stage2_dynamic_forward ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env_petsc_mpi.sh"
stage2_setup_runtime_env

exec "$MPICH_HOME/bin/mpiexec" \
    "$@"
