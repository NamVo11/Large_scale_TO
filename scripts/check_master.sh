#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
command -v mpirun
command -v cmake
command -v mpicxx
mpirun --version | head -2
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
test -x "$ROOT/build-release/stage1_petsc_backend"
ldd "$ROOT/build-release/stage1_petsc_backend" | grep -E 'petsc|mpi' || true

echo
echo "Master check complete."
