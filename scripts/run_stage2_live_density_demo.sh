#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT="${1:-results/vscode_stage2_optimization_tiny_live}"
PORT="${2:-8765}"

export LD_LIBRARY_PATH="/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl/lib:/home/votrongnam/opt/mpich-4.0-ch3-cxx/lib:/opt/intel/oneapi/mkl/2024.2/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

python3 scripts/watch_density_live.py "$OUT" --scale 24 --refresh 0.5 --port "$PORT" &
WATCH_PID=$!
trap 'kill "$WATCH_PID" 2>/dev/null || true' EXIT

sleep 1
echo "Density viewer: http://localhost:${PORT}/index.html"

scripts/mpiexec_mpich_ch3_cxx.sh -n 1 build-mpich-mkl-release/stage2_dynamic_forward \
  -run_optimization 1 \
  -max_iter 2 \
  -density_plot_interval 1 \
  -nelx 4 \
  -nely 2 \
  -nstep 2 \
  -stage2_out "$OUT" \
  -fom_solver_type direct \
  -ksp_type preonly \
  -pc_type lu \
  -pc_factor_mat_solver_type mkl_pardiso

python3 scripts/watch_density_live.py "$OUT" --scale 24 --once
