#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LABEL="${LABEL:-Mystrategy_rank1_0_max_rank_10}"
SOLVE_MODE="${SOLVE_MODE:-ROM}"
METHOD_NAME="${METHOD_NAME:-MyStrategy}"
FOM_SOLVER_TYPE="${FOM_SOLVER_TYPE:-direct}"
ROM_TOL="${ROM_TOL:-0.02}"
RANK1="${RANK1:-0}"
MAX_RANK="${MAX_RANK:-10}"
ONLINE="${ONLINE:-1}"
TMAX="${TMAX:-0.008}"
NP="${NP:-1}"

NELX="${NELX:-120}"
NELY="${NELY:-40}"
NELZ="${NELZ:-1}"
LX="${LX:-1200}"
LY="${LY:-400}"
LZ="${LZ:-120}"
NSTEP="${NSTEP:-100}"
MAX_ITER="${MAX_ITER:-50}"
MOVE="${MOVE:-0.1}"
VOLUME_LIMIT="${VOLUME_LIMIT:-0.4}"
SENS_TYPE="${SENS_TYPE:-AVM}"
X0="${X0:-0.5}"
FORCE_AMP="${FORCE_AMP:-3333.0}"
P_RHO="${P_RHO:-0.0}"

KSP_TYPE="${KSP_TYPE:-preonly}"
PC_TYPE="${PC_TYPE:-lu}"
PC_FACTOR_MAT_SOLVER_TYPE="${PC_FACTOR_MAT_SOLVER_TYPE:-mkl_pardiso}"

DENSITY_INTERVAL="${DENSITY_INTERVAL:-1}"
SAVE_DENSITY="${SAVE_DENSITY:-1}"

BUILD_DIR="${BUILD_DIR:-build-mpich-mkl-release}"
BINARY="${BINARY:-$ROOT/$BUILD_DIR/stage2_dynamic_forward}"
OUT="${OUT:-results/$LABEL}"

export LD_LIBRARY_PATH="/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl/lib:/home/votrongnam/opt/mpich-4.0-ch3-cxx/lib:/opt/intel/oneapi/mkl/2024.2/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

if [[ ! -x "$BINARY" ]]; then
  cmake --build "$BUILD_DIR" --target stage2_dynamic_forward -j "${BUILD_JOBS:-8}"
fi

mkdir -p "$OUT"

echo "Running stage2 case: $LABEL"
echo "  solve_mode=$SOLVE_MODE method=$METHOD_NAME fom_solver=$FOM_SOLVER_TYPE"
echo "  rom_tol=$ROM_TOL rank1=$RANK1 max_rank=$MAX_RANK online=$ONLINE tmax=$TMAX"
echo "  mesh=${NELX}x${NELY}x${NELZ} size=${LX}x${LY}x${LZ} np=$NP out=$OUT"

scripts/mpiexec_mpich_ch3_cxx.sh -n "$NP" "$BINARY" \
  -run_optimization 1 \
  -solve_mode "$SOLVE_MODE" \
  -method_name "$METHOD_NAME" \
  -fom_solver_type "$FOM_SOLVER_TYPE" \
  -rom_tol "$ROM_TOL" \
  -rank1 "$RANK1" \
  -max_rank "$MAX_RANK" \
  -online "$ONLINE" \
  -tmax "$TMAX" \
  -Lx "$LX" \
  -Ly "$LY" \
  -Lz "$LZ" \
  -nelx "$NELX" \
  -nely "$NELY" \
  -nelz "$NELZ" \
  -nstep "$NSTEP" \
  -max_iter "$MAX_ITER" \
  -move "$MOVE" \
  -volume_limit "$VOLUME_LIMIT" \
  -sens_type "$SENS_TYPE" \
  -x0 "$X0" \
  -force_amp "$FORCE_AMP" \
  -p_rho "$P_RHO" \
  -stage2_out "$OUT" \
  -save_density_each_iter "$SAVE_DENSITY" \
  -density_plot_interval "$DENSITY_INTERVAL" \
  -ksp_type "$KSP_TYPE" \
  -pc_type "$PC_TYPE" \
  -pc_factor_mat_solver_type "$PC_FACTOR_MAT_SOLVER_TYPE"
