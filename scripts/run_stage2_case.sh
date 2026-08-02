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

BUILD_DIR="${BUILD_DIR:-build-mpich-release-gcc13}"
if [[ "$BUILD_DIR" != /* ]]; then
    BUILD_DIR="$ROOT/$BUILD_DIR"
fi
BINARY="${BINARY:-$BUILD_DIR/stage2_dynamic_forward}"

NP="${NP:?NP is not set}"
HOSTS="${HOSTS:?HOSTS is not set}"

KSP_TYPE="${KSP_TYPE:?KSP_TYPE is not set}"
PC_TYPE="${PC_TYPE:?PC_TYPE is not set}"
PC_FACTOR_MAT_SOLVER_TYPE="${PC_FACTOR_MAT_SOLVER_TYPE:-}"
POD_BACKEND="${POD_BACKEND:-snapshot_mkl}"
ONLINE_BACKEND="${ONLINE_BACKEND:-incremental_mkl}"
LINEAR_BACKEND="${LINEAR_BACKEND:-petsc}"
SOLVE_MODE="${SOLVE_MODE:?SOLVE_MODE is not set}"
METHOD_NAME="${METHOD_NAME:-MyStrategy}"
RANK1="${RANK1:-10}"
MAX_RANK="${MAX_RANK:-25}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:?OMP_NUM_THREADS is not set}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:?MKL_NUM_THREADS is not set}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

RUN_NAME="${RUN_NAME:-}"
if [[ -z "$RUN_NAME" ]]; then
    if [[ "${SOLVE_MODE,,}" == "fom" ]]; then
        RUN_NAME="FOM"
    else
        RUN_NAME="${METHOD_NAME}_rank1_${RANK1}_max_rank_${MAX_RANK}"
    fi
fi
RUN_DIR="${RUN_DIR:-$CASE_DIR/$RUN_NAME}"
OUT="${OUT:-$RUN_DIR/result}"
DENSITY_OUT="${DENSITY_OUT:-$RUN_DIR/density}"
FEM_UNIFORM="${FEM_UNIFORM:-1}"
FEM_TYPE="${FEM_TYPE:-${fem_type:-plane_stress}}"
MESH_SOURCE="${MESH_SOURCE:-structured}"
FEM_FILE="${FEM_FILE:-}"
REGION_SOURCE="${REGION_SOURCE:-range}"
BC_SOURCE="${BC_SOURCE:-range}"
DESIGN_HMSET="${DESIGN_HMSET:-design_val}"
NONDESIGN_HMSET="${NONDESIGN_HMSET:-nondesign}"
SPC_HMSET="${SPC_HMSET:-SPC}"
LOAD_HMSET="${LOAD_HMSET:-load}"
SAVE_VTK_RESPONSE_INTERVAL="${SAVE_VTK_RESPONSE_INTERVAL:-1}"
SAVE_VTK_OPT_INTERVAL="${SAVE_VTK_OPT_INTERVAL:-1}"
SAVE_DENSITY="${SAVE_DENSITY:-1}"
DENSITY_INTERVAL="${DENSITY_INTERVAL:-1}"
RUN_OPTIMIZATION="${RUN_OPTIMIZATION:-1}"
FILTER_RMIN="${FILTER_RMIN:-3.0}"
FILTER_BETA="${FILTER_BETA:-1.0}"
FILTER_BETA_LOOP="${FILTER_BETA_LOOP:-50}"
FILTER_BETA_MAX="${FILTER_BETA_MAX:-8.0}"
FILTER_T="${FILTER_T:-0.5}"
VOID_E="${VOID_E:-1e-3}"
VOID_RHO="${VOID_RHO:-1e-12}"
SOLID_E="${SOLID_E:-210000}"
SOLID_RHO="${SOLID_RHO:-6e-9}"
NU="${NU:-0.3}"
THICKNESS="${THICKNESS:-1.0}"
P_E_ELASTIC="${P_E_ELASTIC:-8.0}"
DIRICH_FLAG_X="${DIRICH_FLAG_X:-1}"
DIRICH_FLAG_Y="${DIRICH_FLAG_Y:-1}"
DIRICH_FLAG_Z="${DIRICH_FLAG_Z:-1}"
NEUM_FLAG_X="${NEUM_FLAG_X:-0}"
NEUM_FLAG_Y="${NEUM_FLAG_Y:-1}"
NEUM_FLAG_Z="${NEUM_FLAG_Z:-0}"
DIRICH_X_MIN="${DIRICH_X_MIN:-0}"
DIRICH_X_MAX="${DIRICH_X_MAX:-2}"
DIRICH_Y_MIN="${DIRICH_Y_MIN:-0}"
DIRICH_Y_MAX="${DIRICH_Y_MAX:-$LY}"
DIRICH_Z_MIN="${DIRICH_Z_MIN:--1000}"
DIRICH_Z_MAX="${DIRICH_Z_MAX:-1000}"
DIRICH_AMP="${DIRICH_AMP:-0}"
DIRICH_TIME_TYPE="${DIRICH_TIME_TYPE:-constant}"
LOAD_X_MIN_OFFSET="${LOAD_X_MIN_OFFSET:-5}"
LOAD_X_MIN="${LOAD_X_MIN:-}"
LOAD_X_MAX="${LOAD_X_MAX:-}"
LOAD_Y_MIN="${LOAD_Y_MIN:-190}"
LOAD_Y_MAX="${LOAD_Y_MAX:-210}"
LOAD_Z_MIN="${LOAD_Z_MIN:--1000}"
LOAD_Z_MAX="${LOAD_Z_MAX:-1000}"
LOAD_TIME_TYPE="${LOAD_TIME_TYPE:-sin}"
LOAD_NCYCLE="${LOAD_NCYCLE:-1}"
FACTOR_NUM_THREADS="${FACTOR_NUM_THREADS:-${MKL_NUM_THREADS:-1}}"
SENS_ELEMENT_THREADS="${SENS_ELEMENT_THREADS:-1}"
DEBUG_FD_EPSILON="${DEBUG_FD_EPSILON:-1e-6}"
DEBUG_FD_MAX_CHECKS="${DEBUG_FD_MAX_CHECKS:-20}"
export FACTOR_NUM_THREADS
export SENS_ELEMENT_THREADS


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
    -run_optimization "$RUN_OPTIMIZATION"

    -linear_backend "$LINEAR_BACKEND"

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

    -fem_type "$FEM_TYPE"
    -fem_uniform "$FEM_UNIFORM"
    -mesh_source "$MESH_SOURCE"
    -fem_file "$FEM_FILE"
    -region_source "$REGION_SOURCE"
    -bc_source "$BC_SOURCE"
    -design_hmset "$DESIGN_HMSET"
    -nondesign_hmset "$NONDESIGN_HMSET"
    -spc_hmset "$SPC_HMSET"
    -load_hmset "$LOAD_HMSET"
    -factor_num_threads "$FACTOR_NUM_THREADS"
    -sens_element_threads "$SENS_ELEMENT_THREADS"

    -nstep "$NSTEP"

    -max_iter "$MAX_ITER"
    -move "$MOVE"
    -volume_limit "$VOLUME_LIMIT"
    -debug_fd_epsilon "$DEBUG_FD_EPSILON"
    -debug_fd_max_checks "$DEBUG_FD_MAX_CHECKS"

    -filter_rmin "$FILTER_RMIN"
    -filter_beta "$FILTER_BETA"
    -filter_beta_loop "$FILTER_BETA_LOOP"
    -filter_beta_max "$FILTER_BETA_MAX"
    -filter_T "$FILTER_T"

    -sens_type "$SENS_TYPE"
    -x0 "$X0"
    -force_amp "$FORCE_AMP"
    -p_rho "$P_RHO"
    -p_E_elastic "$P_E_ELASTIC"

    -void_E "$VOID_E"
    -void_rho "$VOID_RHO"
    -solid_E "$SOLID_E"
    -solid_rho "$SOLID_RHO"
    -nu "$NU"
    -thickness "$THICKNESS"

    -dirich_flag_x "$DIRICH_FLAG_X"
    -dirich_flag_y "$DIRICH_FLAG_Y"
    -dirich_flag_z "$DIRICH_FLAG_Z"
    -neum_flag_x "$NEUM_FLAG_X"
    -neum_flag_y "$NEUM_FLAG_Y"
    -neum_flag_z "$NEUM_FLAG_Z"
    -dirich_x_min "$DIRICH_X_MIN"
    -dirich_x_max "$DIRICH_X_MAX"
    -dirich_y_min "$DIRICH_Y_MIN"
    -dirich_y_max "$DIRICH_Y_MAX"
    -dirich_z_min "$DIRICH_Z_MIN"
    -dirich_z_max "$DIRICH_Z_MAX"
    -dirich_amp "$DIRICH_AMP"
    -dirich_time_type "$DIRICH_TIME_TYPE"
    -load_x_min_offset "$LOAD_X_MIN_OFFSET"
    -load_y_min "$LOAD_Y_MIN"
    -load_y_max "$LOAD_Y_MAX"
    -load_z_min "$LOAD_Z_MIN"
    -load_z_max "$LOAD_Z_MAX"
    -load_time_type "$LOAD_TIME_TYPE"
    -load_nCycle "$LOAD_NCYCLE"

    -stage2_out "$OUT"
    -density_out "$DENSITY_OUT"

    -save_density_each_iter "$SAVE_DENSITY"
    -density_plot_interval "$DENSITY_INTERVAL"
    -save_vtk_response_interval "$SAVE_VTK_RESPONSE_INTERVAL"
    -save_vtk_opt_interval "$SAVE_VTK_OPT_INTERVAL"

    -ksp_type "$KSP_TYPE"
    -pc_type "$PC_TYPE"
)

if [[ -n "$LOAD_X_MIN" ]]; then
    APP_ARGS+=(-load_x_min "$LOAD_X_MIN")
fi
if [[ -n "$LOAD_X_MAX" ]]; then
    APP_ARGS+=(-load_x_max "$LOAD_X_MAX")
fi

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
echo "  linear_backend            = $LINEAR_BACKEND"
echo "  method_name               = $METHOD_NAME"
echo "  fom_solver_type           = $FOM_SOLVER_TYPE"
echo "  pod_backend               = $POD_BACKEND"
echo "  online_backend            = $ONLINE_BACKEND"
echo
echo "  geometry                  = ${LX} x ${LY} x ${LZ}"
echo "  mesh                      = ${NELX} x ${NELY} x ${NELZ}"
echo "  mesh_source               = $MESH_SOURCE"
echo "  fem_file                  = ${FEM_FILE:-<not set>}"
echo "  region_source             = $REGION_SOURCE"
echo "  bc_source                 = $BC_SOURCE"
echo "  spc/load hmset            = $SPC_HMSET / $LOAD_HMSET"
echo "  fem_uniform               = $FEM_UNIFORM"
echo "  tmax                      = $TMAX"
echo "  nstep                     = $NSTEP"
echo
echo "  hosts                     = $HOSTS"
echo "  MPI processes             = $NP"
echo "  OMP threads/process       = $OMP_NUM_THREADS"
echo "  MKL threads/process       = $MKL_NUM_THREADS"
echo "  factorization threads     = $FACTOR_NUM_THREADS"
echo "  sensitivity elem threads  = $SENS_ELEMENT_THREADS"
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
