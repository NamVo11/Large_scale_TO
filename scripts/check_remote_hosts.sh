#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Verify SSH access and basic runtime tools on worker machines.
# Usage:
#   bash scripts/check_remote_hosts.sh
#   bash scripts/check_remote_hosts.sh scripts/hosts.example

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOSTFILE="${1:-$ROOT/scripts/hosts.example}"

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
  "$ROOT/scripts/ssh_worker_2222.sh" \
    -o BatchMode=yes -o ConnectTimeout=5 "$host" \
    'hostname; hostname -I; nproc; free -h; command -v mpiexec; mpiexec -version | head -2; pkg-config --modversion PETSc'
done

echo "Remote host check complete."
