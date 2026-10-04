#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src
device="${DEVICE:-auto}"
seed="${SEED:-42}"

experiments=(main fixed_reader no_persistent video_only lstr gatehub testra mult)

for name in "${experiments[@]}"; do
  config="configs/${name}.yaml"
  output_name="$name"
  [[ "$name" == "main" ]] && output_name="eptnet"
  result="results/final/${output_name}/seed_${seed}"

  if [[ ! -s "$result/best.pt" || ! -s "$result/last.pt" ]]; then
    python -u -m eptnet.training.trainer \
      --config "$config" --seed "$seed" --device "$device"
  fi

  if [[ ! -s "$result/test_metrics.json" \
     || ! -s "$result/test_predictions.jsonl" \
     || ! -s "$result/test_events.json" ]]; then
    python -u -m eptnet.evaluation.evaluator \
      --config "$config" \
      --checkpoint "$result/best.pt" \
      --output "$result/test_metrics.json" \
      --predictions-output "$result/test_predictions.jsonl" \
      --events-output "$result/test_events.json" \
      --device "$device"
  fi

  if [[ ! -s "results/final/${output_name}/aggregate.json" \
     || ! -s "results/final/${output_name}/aggregate.csv" ]]; then
    python -u -m eptnet.evaluation.aggregate "$result/test_metrics.json" \
      --output "results/final/${output_name}/aggregate.json" \
      --csv "results/final/${output_name}/aggregate.csv"
  fi
done

python figures/generate.py
