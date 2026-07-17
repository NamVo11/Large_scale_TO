#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Main launcher used by VS Code tasks for stage2 single-node and multi-node runs.
# Usage:
#   Set BUILD_DIR, CASE_CONFIG, NP, HOSTS, solver env vars, then run:
#   bash scripts/run_stage2_case.sh
#
# Main runtime choices used by VS Code tasks:
#   KSP_TYPE:
#     preonly      direct factorization solve; use with PC_TYPE=lu/cholesky
#     cg           iterative SPD solve; use with PC_TYPE=gamg
#   PC_TYPE:
#     cholesky     fastest direct option for SPD elastic matrices
#     lu           general direct option when SPD is not guaranteed
#     gamg         iterative AMG preconditioner, usually with KSP_TYPE=cg
#   PC_FACTOR_MAT_SOLVER_TYPE:
#     mkl_pardiso  best for single-node/single-rank direct solves
#     mumps        distributed direct solver for MPI/multinode
#     empty        use PETSc/default, or leave empty for iterative runs
#   POD_BACKEND:
#     snapshot_mkl    fast MATLAB-equivalent snapshot POD using MKL BLAS/LAPACK
#     snapshot_eigen  fast MATLAB-equivalent snapshot POD
#     bdcsvd          old Eigen BDCSVD backend for comparison/debug
#   ONLINE_BACKEND:
#     incremental_mkl    IncrementalSVD with MKL BLAS kernels
#     incremental_eigen  IncrementalSVD with Eigen kernels
#   METHOD_NAME:
#     MyStrategy         MATLAB-like strategy with IncrementalSVD online basis
#     MyStrategy2        experimental strategy with MyIncrementalQR1 online basis
#   Threads:
#     mkl_pardiso: try OMP_NUM_THREADS=MKL_NUM_THREADS=4,8,16
#     MPI/GAMG/MUMPS many-rank runs: usually start with 1 thread/rank

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "$ROOT/scripts/env_petsc_mpi.sh"
stage2_setup_runtime_env

if [[ -n "${CASE_CONFIGS:-}" ]]; then
    echo
    echo "============================================================"
    echo "Running stage2 batch"
    echo "============================================================"
    echo "$CASE_CONFIGS" | tr ' ' '\n' | sed '/^$/d' | nl -w2 -s'. '
    echo "============================================================"
    echo

    case_id=0
    for cfg in $CASE_CONFIGS; do
        case_id=$((case_id + 1))
        echo
        echo "============================================================"
        echo "Batch case $case_id"
        echo "CASE_CONFIG=$cfg"
        echo "============================================================"
        echo

        CASE_CONFIG="$cfg" CASE_CONFIGS= bash "$ROOT/scripts/run_stage2_case.sh"
    done

    echo
    echo "============================================================"
    echo "Stage2 batch finished"
    echo "============================================================"
    exit 0
fi

CASE_CONFIG="${CASE_CONFIG:?CASE_CONFIG is not set}"

NP="${NP:?NP is not set}"
HOSTS="${HOSTS:?HOSTS is not set}"

KSP_TYPE="${KSP_TYPE:?KSP_TYPE is not set}"
PC_TYPE="${PC_TYPE:?PC_TYPE is not set}"
PC_FACTOR_MAT_SOLVER_TYPE="${PC_FACTOR_MAT_SOLVER_TYPE:-}"
POD_BACKEND="${POD_BACKEND:-snapshot_mkl}"
ONLINE_BACKEND="${ONLINE_BACKEND:-incremental_mkl}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:?OMP_NUM_THREADS is not set}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:?MKL_NUM_THREADS is not set}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

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
DENSITY_OUT="${DENSITY_OUT:-$CASE_DIR/density}"


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

mkdir -p "$OUT" "$DENSITY_OUT"

# ============================================================
# Application arguments
# All model parameters come directly from CASE_CONFIG.
# ============================================================

APP_ARGS=(
    -run_optimization 1

    -solve_mode "$SOLVE_MODE"
    -method_name "$METHOD_NAME"
    -fom_solver_type "$FOM_SOLVER_TYPE"
    -pod_backend "$POD_BACKEND"
    -online_backend "$ONLINE_BACKEND"

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
    -density_out "$DENSITY_OUT"

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
echo "  density                   = $DENSITY_OUT"
echo
echo "  solve_mode                = $SOLVE_MODE"
echo "  method_name               = $METHOD_NAME"
echo "  fom_solver_type           = $FOM_SOLVER_TYPE"
echo "  pod_backend               = $POD_BACKEND"
echo "  online_backend            = $ONLINE_BACKEND"
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
