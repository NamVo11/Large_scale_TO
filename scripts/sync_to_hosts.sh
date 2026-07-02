#!/usr/bin/env bash
set -euo pipefail

HOSTFILE="${1:-scripts/hosts.txt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="/home/votrongnam/PROJECT/Large_scale_TO"

cd "$ROOT"
hosts=$(awk 'NF && $1 !~ /^#/ {print $1}' "$HOSTFILE" | sed 's/:.*//' | sort -u)

for host in $hosts; do
  if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
    continue
  fi
  echo "Syncing to $host:$DEST"
  ssh "$host" "mkdir -p '$DEST'"
  rsync -az --delete \
    --exclude '.git/' \
    --exclude 'build/' \
    --exclude 'build-release/' \
    --exclude 'results/' \
    "$ROOT/" "$host:$DEST/"
  ssh "$host" "cd '$DEST' && cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release && cmake --build build-release -j \$(nproc)"
done

echo "Sync complete."
