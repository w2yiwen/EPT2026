#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
SOURCE_ROOT="${1:?source BCI bundle path is required}"
PYTHONPATH=src python3 -m eptnet.data.prepare_bci \
  --source "$SOURCE_ROOT" \
  --data-root data \
  --window-size 64 \
  --stride 32 \
  --min-window-size 16
