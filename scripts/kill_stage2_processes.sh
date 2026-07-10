#!/usr/bin/env bash
set -euo pipefail

pkill -f 'stage2_dynamic_forward|mpiexec|hydra_pmi_proxy|hydra_nameserver' || true
sleep 2
pkill -9 -f 'stage2_dynamic_forward|mpiexec|hydra_pmi_proxy|hydra_nameserver' || true
pgrep -af 'stage2_dynamic_forward|mpiexec|hydra_pmi_proxy|hydra_nameserver' || true
