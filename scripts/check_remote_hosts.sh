#!/usr/bin/env bash
set -euo pipefail

HOSTFILE="${1:-scripts/hosts.txt}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"
if [[ ! -f "$HOSTFILE" ]]; then
  echo "Hostfile not found: $HOSTFILE"
  exit 1
fi

hosts=$(awk 'NF && $1 !~ /^#/ {print $1}' "$HOSTFILE" | sed 's/:.*//' | sort -u)

for host in $hosts; do
  if [[ "$host" == "localhost" || "$host" == "$(hostname)" ]]; then
    continue
  fi
  echo "== $host =="
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" \
    'hostname; hostname -I; nproc; free -h; command -v mpirun; mpirun --version | head -2; pkg-config --modversion PETSc'
done

echo "Remote host check complete."
