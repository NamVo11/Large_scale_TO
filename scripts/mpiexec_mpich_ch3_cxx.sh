#!/usr/bin/env bash
set -euo pipefail

MPICH_HOME="${MPICH_HOME:-/home/votrongnam/opt/mpich-4.0-ch3-cxx}"

export MPIEXEC_PORT_RANGE="${MPIEXEC_PORT_RANGE:-40000:41000}"
export MPICH_PORT_RANGE="${MPICH_PORT_RANGE:-40000:41000}"
export HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl}"

exec "$MPICH_HOME/bin/mpiexec" \
    -iface eth1 \
    "$@"
