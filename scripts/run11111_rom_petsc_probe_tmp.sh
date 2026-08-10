#!/usr/bin/env bash
set -u
cd /home/votrongnam/PROJECT/Large_scale_TO
mkdir -p debug_logs
LOG="debug_logs/run11111_rom_petsc_probe_$(date +%Y%m%d_%H%M%S).log"
echo "LOG=$LOG"
cmake --build build-mpich-release-gcc13 --target stage2_dynamic_forward -j 4 >/tmp/run11111_build.log 2>&1 || { cat /tmp/run11111_build.log; exit 1; }
export BUILD_DIR="build-mpich-release-gcc13"
export PETSC_PREFIX="/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl2025-mumps-gcc13"
export MPICH_HOME="/home/votrongnam/opt/mpich-4.0-ch3-cxx-gcc13"
export INPUT_JSONS="case_study/Engine_bracket/mesh_size1.5/strategy_10_20/input.json"
export KSP_TYPE="preonly"
export PC_TYPE="cholesky"
export PC_FACTOR_MAT_SOLVER_TYPE="mkl_pardiso"
export LINEAR_BACKEND="petsc"
export STAGE2_EIGEN_CHECK_FOM_RESIDUAL="0"
export STAGE2_MEMORY_DEBUG="0"
export POD_BACKEND="snapshot_mkl"
export ONLINE_BACKEND="incremental_mkl"
export STAGE2_MYSTRATEGY2_FULL_PROJECTION="0"
export OMP_NUM_THREADS="8"
export MKL_NUM_THREADS="8"
export FACTOR_NUM_THREADS="8"
export SENS_ELEMENT_THREADS="8"
export OMP_PROC_BIND="spread"
export OMP_PLACES="cores"
export MKLROOT="/opt/intel/oneapi/mkl/2025.0"
export LD_LIBRARY_PATH="$PETSC_PREFIX/lib:$MPICH_HOME/lib:$MKLROOT/lib:$MKLROOT/lib/intel64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PETSC_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
# Start clean for this case's result files, but keep input.json.
rm -rf case_study/Engine_bracket/mesh_size1.5/strategy_10_20/result case_study/Engine_bracket/mesh_size1.5/strategy_10_20/density case_study/Engine_bracket/mesh_size1.5/strategy_10_20/vtk case_study/Engine_bracket/mesh_size1.5/strategy_10_20/figure
stdbuf -oL -eL timeout 720 bash scripts/run_stage2_case.sh >"$LOG" 2>&1 || true
pkill -TERM -f stage2_dynamic_forward || true
pkill -TERM -f mpiexec || true
echo "===== LOG TAIL ====="
tail -160 "$LOG" || true
echo "===== RESULT FILES ====="
find case_study/Engine_bracket/mesh_size1.5/strategy_10_20/result -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort || true
