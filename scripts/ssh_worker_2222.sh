#!/usr/bin/env bash

# Purpose:
#   Central SSH wrapper for the worker node on port 2222, shared by rsync and mpiexec.
# Usage:
#   scripts/ssh_worker_2222.sh 192.168.11.21 "hostname"

exec /usr/bin/ssh -p 2222 -i /home/votrongnam/.ssh/id_rsa -o StrictHostKeyChecking=accept-new "$@"
