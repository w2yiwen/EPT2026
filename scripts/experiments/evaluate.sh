#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
CONFIG_PATH="${1:-configs/default.yaml}"
CHECKPOINT_PATH="${2:?checkpoint path is required}"
mkdir -p logs results
STAMP="$(date +'%y-%m-%d_%H-%M-%S')"
LOG_PATH="logs/evaluate_${STAMP}.log"
OUTPUT_PATH="results/evaluation_${STAMP}.json"
echo "Starting foreground evaluation: log=$LOG_PATH output=$OUTPUT_PATH"
env PYTHONPATH=src python3 -u -m eptnet.evaluate \
  --config "$CONFIG_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output "$OUTPUT_PATH" 2>&1 | tee "$LOG_PATH"
echo "Evaluation completed successfully: log=$LOG_PATH output=$OUTPUT_PATH"
