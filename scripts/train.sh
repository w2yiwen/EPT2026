#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
config="${1:-configs/main.yaml}"
shift || true
PYTHONPATH=src python -u -m eptnet.training.trainer --config "$config" "$@"
