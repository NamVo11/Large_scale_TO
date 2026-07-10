#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Sync the Large_scale_TO project to worker machines at the same path.
# Usage:
#   bash scripts/sync_to_hosts.sh
#   bash scripts/sync_to_hosts.sh scripts/hosts.example

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTFILE="${1:-$ROOT/scripts/hosts.example}"
DEST="/home/votrongnam/PROJECT/Large_scale_TO"

cd "$ROOT"
hosts=$(awk 'NF && $1 !~ /^#/ {print $1}' "$HOSTFILE" | sed 's/:.*//' | sort -u)

for host in $hosts; do
  if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
    continue
  fi
  echo "Syncing to $host:$DEST"
  "$ROOT/scripts/ssh_worker_2222.sh" "$host" "mkdir -p '$DEST'"
  rsync -az --delete \
    -e "$ROOT/scripts/ssh_worker_2222.sh" \
    --exclude '.git/' \
    --exclude 'build-mpich-debug/' \
    --exclude 'build-mpich-release/' \
    --exclude 'results/' \
    "$ROOT/" "$host:$DEST/"
  "$ROOT/scripts/ssh_worker_2222.sh" \
    "$host" \
    "cd '$DEST' && export PKG_CONFIG_PATH='/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl2025-mumps/lib/pkgconfig' && cmake -S . -B build-mpich-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=/home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpicxx -DPETSC_PKG=PETSc && cmake --build build-mpich-release --target stage2_dynamic_forward -j \$(nproc)"
done

echo "Sync complete."
