#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src"

python_bin="${PYTHON_BIN:-python3}"
device="${DEVICE:-cuda:0}"
seeds=(13 42 73)
primary=(
  "configs/default.yaml:eptnet_continuous_no_text"
  "configs/baseline_early_fusion_gru.yaml:baseline_early_fusion_gru_continuous_no_text"
  "configs/baseline_fusion_transformer.yaml:baseline_fusion_transformer_continuous_no_text"
  "configs/ablation_no_persistent.yaml:ablation_no_persistent_continuous_no_text"
)
diagnostics=(
  "configs/ablation_fixed_reader.yaml:ablation_fixed_reader_continuous_no_text"
  "configs/ablation_no_behavior.yaml:ablation_no_behavior_continuous_no_text"
  "configs/ablation_single_scale.yaml:ablation_single_scale_continuous_no_text"
)

run_one() {
  local config="$1"
  local experiment="$2"
  local seed="$3"
  local run_dir="results/${experiment}/seed_${seed}"
  if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse existing formal run directory: $run_dir" >&2
    exit 1
  fi
  "$python_bin" -u -m eptnet.train \
    --config "$config" --seed "$seed" --device "$device"
  "$python_bin" -u -m eptnet.evaluate \
    --config "$config" --checkpoint "$run_dir/best.pt" \
    --output "$run_dir/test_metrics.json" --device "$device"
}

for entry in "${primary[@]}"; do
  IFS=: read -r config experiment <<<"$entry"
  metric_files=()
  for seed in "${seeds[@]}"; do
    run_one "$config" "$experiment" "$seed"
    metric_files+=("results/${experiment}/seed_${seed}/test_metrics.json")
  done
  "$python_bin" -u -m eptnet.aggregate "${metric_files[@]}" \
    --output "results/${experiment}/aggregate.json" \
    --csv "results/${experiment}/aggregate.csv"
done

if [[ "${SKIP_DIAGNOSTICS:-0}" != "1" ]]; then
  for entry in "${diagnostics[@]}"; do
    IFS=: read -r config experiment <<<"$entry"
    run_one "$config" "$experiment" 42
  done
fi

echo "Formal experiment matrix completed successfully."
