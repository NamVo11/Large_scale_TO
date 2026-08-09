#!/usr/bin/env bash
set -u

cd /home/votrongnam/PROJECT/Large_scale_TO
mkdir -p debug_logs

LOG="debug_logs/engine_bracket_eigen_memory_full_$(date +%Y%m%d_%H%M%S).log"
MON="${LOG%.log}_monitor.log"

echo "LOG=$LOG"
echo "MONITOR=$MON"

cmake --build build-mpich-release-gcc13 --target stage2_dynamic_forward -j 4 \
  >/tmp/stage2_build_memory_probe_eigen.log 2>&1 || {
  cat /tmp/stage2_build_memory_probe_eigen.log
  exit 1
}

export INPUT_JSONS="debug_logs/engine_bracket_eigen_input_tmp.json"
export BUILD_DIR="build-mpich-release-gcc13"
export PETSC_PREFIX="/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl2025-mumps-gcc13"
export MPICH_HOME="/home/votrongnam/opt/mpich-4.0-ch3-cxx-gcc13"
export MKLROOT="/opt/intel/oneapi/mkl/2025.0"
export LD_LIBRARY_PATH="$PETSC_PREFIX/lib:$MPICH_HOME/lib:$MKLROOT/lib:$MKLROOT/lib/intel64:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$PETSC_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export STAGE2_MEMORY_DEBUG=1
export LINEAR_BACKEND="eigen"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export FACTOR_NUM_THREADS=8
export SENS_ELEMENT_THREADS=8

stdbuf -oL -eL bash scripts/run_stage2_case.sh >"$LOG" 2>&1 &
run_pid=$!
start=$(date +%s)
low_count=0

while kill -0 "$run_pid" 2>/dev/null; do
  elapsed=$(($(date +%s) - start))
  mem_avail_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  swap_free_kb=$(awk '/SwapFree:/ {print $2}' /proc/meminfo)
  log_size=$(stat -c%s "$LOG" 2>/dev/null || echo 0)
  awk -v e="$elapsed" -v m="$mem_avail_kb" -v s="$swap_free_kb" -v z="$log_size" \
    'BEGIN{printf "monitor elapsed_s=%s mem_available_GB=%.3f swap_free_GB=%.3f log_bytes=%s\n", e, m/1048576, s/1048576, z}' |
    tee -a "$MON"
  grep 'memory\[' "$LOG" | tail -5 | sed 's/^/  last: /' >>"$MON" || true

  if [ "$mem_avail_kb" -lt $((6 * 1024 * 1024)) ] || [ "$swap_free_kb" -lt $((8 * 1024 * 1024)) ]; then
    low_count=$((low_count + 1))
  else
    low_count=0
  fi

  if [ "$low_count" -ge 2 ]; then
    echo "monitor: low memory threshold reached, stopping run_pid=$run_pid" | tee -a "$MON"
    kill -INT "$run_pid" 2>/dev/null || true
    sleep 8
    kill -TERM "$run_pid" 2>/dev/null || true
    sleep 3
    pkill -TERM -f stage2_dynamic_forward || true
    pkill -TERM -f mpiexec || true
    break
  fi

  if [ "$elapsed" -ge 1200 ]; then
    echo "monitor: time limit reached, stopping run_pid=$run_pid" | tee -a "$MON"
    kill -INT "$run_pid" 2>/dev/null || true
    sleep 8
    kill -TERM "$run_pid" 2>/dev/null || true
    sleep 3
    pkill -TERM -f stage2_dynamic_forward || true
    pkill -TERM -f mpiexec || true
    break
  fi

  sleep 10
done

wait "$run_pid" || true

MEM="${LOG%.log}_memory_only.log"
grep 'memory\[' "$LOG" >"$MEM" || true

echo "===== LOG PATHS ====="
echo "$LOG"
echo "$MEM"
echo "$MON"
echo "===== MEMORY ONLY TAIL ====="
tail -100 "$MEM" || true
echo "===== RUN LOG TAIL ====="
tail -120 "$LOG" || true
