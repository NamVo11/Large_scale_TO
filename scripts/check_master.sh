#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Quick sanity check for the master machine before build/run/debug.
# Usage:
#   bash scripts/check_master.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# shellcheck disable=SC1091
source "$ROOT/scripts/env_petsc_mpi.sh"
stage2_setup_runtime_env

echo "== Master node =="
whoami
hostname
hostname -I || true
nproc
free -h

echo
echo "== Tools =="
command -v ssh
command -v ssh-keygen
command -v rsync
command -v "$MPICH_HOME/bin/mpiexec"
command -v cmake
command -v mpicxx
"$MPICH_HOME/bin/mpiexec" -version | head -2
pkg-config --modversion PETSc

echo
echo "== SSH key =="
if [[ -f "$HOME/.ssh/id_ed25519.pub" ]]; then
  echo "Using $HOME/.ssh/id_ed25519.pub"
  cat "$HOME/.ssh/id_ed25519.pub"
elif [[ -f "$HOME/.ssh/id_rsa.pub" ]]; then
  echo "Using $HOME/.ssh/id_rsa.pub"
  cat "$HOME/.ssh/id_rsa.pub"
else
  echo "No SSH public key found. Creating ed25519 key."
  ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519"
  cat "$HOME/.ssh/id_ed25519.pub"
fi

echo
echo "== Project =="
echo "$ROOT"
test -x "$ROOT/build-mpich-release/stage2_dynamic_forward"
ldd "$ROOT/build-mpich-release/stage2_dynamic_forward" | grep -E 'petsc|mpi|mkl' || true

echo
echo "Master check complete."
