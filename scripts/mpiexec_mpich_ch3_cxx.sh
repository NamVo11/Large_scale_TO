#!/usr/bin/env bash
export MPIEXEC_PORT_RANGE="${MPIEXEC_PORT_RANGE:-45010:46000}"
export MPICH_PORT_RANGE="${MPICH_PORT_RANGE:-45010:46000}"
export HWLOC_COMPONENTS="${HWLOC_COMPONENTS:--gl}"
exec /home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpiexec -iface eth1 "$@"
