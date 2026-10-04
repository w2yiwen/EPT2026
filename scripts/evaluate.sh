#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
config="${1:?usage: scripts/evaluate.sh CONFIG CHECKPOINT OUTPUT_DIR}"
checkpoint="${2:?usage: scripts/evaluate.sh CONFIG CHECKPOINT OUTPUT_DIR}"
output_dir="${3:?usage: scripts/evaluate.sh CONFIG CHECKPOINT OUTPUT_DIR}"
mkdir -p "$output_dir"

PYTHONPATH=src python -u -m eptnet.evaluation.evaluator \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --output "$output_dir/test_metrics.json" \
  --predictions-output "$output_dir/test_predictions.jsonl" \
  --events-output "$output_dir/test_events.json"
