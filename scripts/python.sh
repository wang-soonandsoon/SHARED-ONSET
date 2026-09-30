#!/usr/bin/env bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec "${PYTHON:-python3}" "$@"
