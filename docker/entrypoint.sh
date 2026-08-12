#!/usr/bin/env bash
set -e

source /opt/large-scale-to/env.sh

if [[ $# -eq 0 ]]; then
    exec bash
fi

exec "$@"
