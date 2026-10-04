#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
device="${DEVICE:-cuda}"
seed=42

if [[ ! -x "$python_bin" ]]; then
  echo "Python is not executable: $python_bin" >&2
  exit 1
fi

matrix=(
  "configs/eptnet_marlin11_eeg_ppg_video.yaml|eptnet_marlin11_eeg_ppg_video_no_text|EPT-Net"
  "configs/lstr_marlin11_eeg_ppg_video.yaml|lstr_marlin11_eeg_ppg_video_no_text|LSTR"
  "configs/gatehub_marlin11_eeg_ppg_video.yaml|gatehub_marlin11_eeg_ppg_video_no_text|GateHUB"
  "configs/testra_marlin11_eeg_ppg_video.yaml|testra_marlin11_eeg_ppg_video_no_text|TeSTra"
)

aggregate_specs=()
for entry in "${matrix[@]}"; do
  IFS='|' read -r config experiment label <<<"$entry"
  run_dir="results/$experiment/seed_$seed"
  metrics="$run_dir/test_metrics.json"
  predictions="$run_dir/test_predictions.jsonl"
  events="$run_dir/test_events.json"
  best="$run_dir/best.pt"
  last="$run_dir/last.pt"

  if [[ -f "$metrics" && -f "$predictions" && -f "$events" ]]; then
    echo "[REUSE] $label seed $seed is already complete"
  else
    if [[ -f "$last" && -f "$best" ]]; then
      echo "[RESUME] $label seed $seed"
      "$python_bin" -u -m eptnet.train \
        --config "$config" --seed "$seed" --device "$device" --resume "$last"
    elif [[ -e "$run_dir" ]]; then
      echo "Incomplete run requires both last.pt and best.pt: $run_dir" >&2
      exit 1
    else
      echo "[TRAIN] $label seed $seed"
      "$python_bin" -u -m eptnet.train \
        --config "$config" --seed "$seed" --device "$device"
    fi
    echo "[EVALUATE] $label seed $seed"
    "$python_bin" -u -m eptnet.evaluate \
      --config "$config" \
      --checkpoint "$best" \
      --output "$metrics" \
      --predictions-output "$predictions" \
      --events-output "$events" \
      --device "$device"
  fi

  aggregate_json="results/$experiment/aggregate_seed_42.json"
  aggregate_csv="results/$experiment/aggregate_seed_42.csv"
  if [[ -f "$aggregate_json" && -f "$aggregate_csv" ]]; then
    echo "[REUSE] $label seed-42 aggregate"
  elif [[ -e "$aggregate_json" || -e "$aggregate_csv" ]]; then
    echo "Incomplete aggregate pair: $aggregate_json / $aggregate_csv" >&2
    exit 1
  else
    "$python_bin" -u -m eptnet.aggregate "$metrics" \
      --output "$aggregate_json" \
      --csv "$aggregate_csv" \
      --no-overwrite
  fi
  aggregate_specs+=("--aggregate" "$label=$aggregate_json")
done

echo "[FIGURE] EPT-Net versus LSTR, GateHUB, and TeSTra"
"$python_bin" fig/fig03_main_comparison/plot_fig03_main_comparison.py \
  "${aggregate_specs[@]}"

echo "[DONE] Seed-42 main comparison completed."
echo "[OUTPUT] results/*/seed_42 and results/*/aggregate_seed_42.{json,csv}"
echo "[OUTPUT] fig/fig03_main_comparison/fig03_main_comparison.{pdf,svg,png}"
