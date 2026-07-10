#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CASE_CONFIG="${CASE_CONFIG:?CASE_CONFIG is not set}"

NP="${NP:?NP is not set}"
HOSTS="${HOSTS:?HOSTS is not set}"

KSP_TYPE="${KSP_TYPE:?KSP_TYPE is not set}"
PC_TYPE="${PC_TYPE:?PC_TYPE is not set}"
PC_FACTOR_MAT_SOLVER_TYPE="${PC_FACTOR_MAT_SOLVER_TYPE:-}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:?OMP_NUM_THREADS is not set}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:?MKL_NUM_THREADS is not set}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

PETSC_PREFIX="${PETSC_PREFIX:-/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl2025-mumps}"
MPICH_HOME="${MPICH_HOME:-/home/votrongnam/opt/mpich-4.0-ch3-cxx}"
MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/2025.0}"
BUILD_DIR="${BUILD_DIR:?BUILD_DIR is not set}"

if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="$ROOT/$BUILD_DIR"
fi

BINARY="${BINARY:-$BUILD_DIR/stage2_dynamic_forward}"

# ============================================================
# Load case configuration
# ============================================================

if [[ "$CASE_CONFIG" != /* ]]; then
    CASE_CONFIG="$ROOT/$CASE_CONFIG"
fi

if [[ ! -f "$CASE_CONFIG" ]]; then
    echo "Error: case configuration file does not exist:" >&2
    echo "  $CASE_CONFIG" >&2
    exit 1
fi

echo "Loading case configuration:"
echo "  $CASE_CONFIG"

set -a
# shellcheck disable=SC1090
source "$CASE_CONFIG"
set +a

# ============================================================
# Variables below must be defined in the case configuration.
# With set -u, the script stops immediately if any are missing.
# ============================================================

CASE_DIR="$(dirname "$CASE_CONFIG")"

LABEL="$(basename "$CASE_DIR")"

OUT="${OUT:-$CASE_DIR/result}"


# ============================================================
# Runtime libraries
# ============================================================

export LD_LIBRARY_PATH="${PETSC_PREFIX}/lib:${MPICH_HOME}/lib:${MKLROOT}/lib:${LD_LIBRARY_PATH:-}"

# ============================================================
# Build executable if needed
# ============================================================

if [[ ! -x "$BINARY" ]]; then
    echo "Executable not found. Building:"
    echo "  $BINARY"

    cmake --build "$BUILD_DIR" \
        --target stage2_dynamic_forward \
        -j "${BUILD_JOBS:-8}"
fi

if [[ ! -x "$BINARY" ]]; then
    echo "Error: executable is missing:" >&2
    echo "  $BINARY" >&2
    exit 1
fi

mkdir -p "$OUT"

# ============================================================
# Application arguments
# All model parameters come directly from CASE_CONFIG.
# ============================================================

APP_ARGS=(
    -run_optimization 1

    -solve_mode "$SOLVE_MODE"
    -method_name "$METHOD_NAME"
    -fom_solver_type "$FOM_SOLVER_TYPE"

    -rom_tol "$ROM_TOL"
    -rank1 "$RANK1"
    -max_rank "$MAX_RANK"
    -online "$ONLINE"

    -tmax "$TMAX"

    -Lx "$LX"
    -Ly "$LY"
    -Lz "$LZ"

    -nelx "$NELX"
    -nely "$NELY"
    -nelz "$NELZ"

    -fem_type "$fem_type"

    -nstep "$NSTEP"

    -max_iter "$MAX_ITER"
    -move "$MOVE"
    -volume_limit "$VOLUME_LIMIT"

    -sens_type "$SENS_TYPE"
    -x0 "$X0"
    -force_amp "$FORCE_AMP"
    -p_rho "$P_RHO"

    -stage2_out "$OUT"

    -save_density_each_iter "$SAVE_DENSITY"
    -density_plot_interval "$DENSITY_INTERVAL"

    -ksp_type "$KSP_TYPE"
    -pc_type "$PC_TYPE"
)

if [[ -n "$PC_FACTOR_MAT_SOLVER_TYPE" ]]; then
    APP_ARGS+=(
        -pc_factor_mat_solver_type "$PC_FACTOR_MAT_SOLVER_TYPE"
    )
fi

# ============================================================
# Print configuration
# ============================================================

echo
echo "============================================================"
echo "Running stage2 case"
echo "============================================================"
echo "  case_config               = $CASE_CONFIG"
echo "  label                     = $CASE_DIR"
echo "  binary                    = $BINARY"
echo "  output                    = $OUT"
echo
echo "  solve_mode                = $SOLVE_MODE"
echo "  method_name               = $METHOD_NAME"
echo "  fom_solver_type           = $FOM_SOLVER_TYPE"
echo
echo "  geometry                  = ${LX} x ${LY} x ${LZ}"
echo "  mesh                      = ${NELX} x ${NELY} x ${NELZ}"
echo "  tmax                      = $TMAX"
echo "  nstep                     = $NSTEP"
echo
echo "  hosts                     = $HOSTS"
echo "  MPI processes             = $NP"
echo "  OMP threads/process       = $OMP_NUM_THREADS"
echo "  MKL threads/process       = $MKL_NUM_THREADS"
echo
echo "  ksp_type                  = $KSP_TYPE"
echo "  pc_type                   = $PC_TYPE"
echo "  factor solver             = ${PC_FACTOR_MAT_SOLVER_TYPE:-<not set>}"
echo "============================================================"
echo

# ============================================================
# Select single-node or multi-node launcher
# ============================================================

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