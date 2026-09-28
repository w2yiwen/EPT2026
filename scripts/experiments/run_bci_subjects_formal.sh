#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src"

python_bin="${PYTHON_BIN:-python3}"
device="${DEVICE:-cuda:0}"
seeds=(13 42 73)
matrix=(
  "configs/bci_subjects.yaml:eptnet_bci_subjects_no_text"
  "configs/bci_subjects_baseline_early_fusion_gru.yaml:baseline_early_fusion_gru_bci_subjects_no_text"
  "configs/bci_subjects_baseline_fusion_transformer.yaml:baseline_fusion_transformer_bci_subjects_no_text"
  "configs/bci_subjects_ablation_no_recurrent_fusion.yaml:ablation_no_recurrent_fusion_bci_subjects_no_text"
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

"$python_bin" -u scripts/audit/audit_bci_subjects.py \
  --config configs/bci_subjects.yaml \
  --json-output results/bci_subjects_data_audit.json \
  --markdown-output results/bci_subjects_data_audit.md
"$python_bin" -u scripts/audit/audit_parameter_fairness.py \
  configs/bci_subjects.yaml \
  configs/bci_subjects_baseline_early_fusion_gru.yaml \
  configs/bci_subjects_baseline_fusion_transformer.yaml \
  configs/bci_subjects_ablation_no_recurrent_fusion.yaml \
  --device cpu --output results/parameter_fairness_bci_subjects.json

for entry in "${matrix[@]}"; do
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

echo "BCI-subject formal experiment matrix completed successfully."
