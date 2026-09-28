#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
CONFIG_PATH="${1:-configs/default.yaml}"
mkdir -p logs
STAMP="$(date +'%y-%m-%d_%H-%M-%S')"
LOG_PATH="logs/train_${STAMP}.log"
echo "Starting foreground training: log=$LOG_PATH config=$CONFIG_PATH"
env PYTHONPATH=src python3 -u -m eptnet.train --config "$CONFIG_PATH" --progress 2>&1 | tee "$LOG_PATH"
echo "Training completed successfully: log=$LOG_PATH config=$CONFIG_PATH"
