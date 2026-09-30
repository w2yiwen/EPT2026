#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

default_python="$project_root/.venv/bin/python"
if [[ ! -x "$default_python" ]]; then
  default_python="python3"
fi
python_bin="${PYTHON_BIN:-$default_python}"
device="${DEVICE:-cuda:0}"
suite="all"
primary_seed_string="13 42 73"
eptnet_seed_string=""
analysis_seed_string="42"
dry_run=0
preflight_only=0
skip_existing=0
resume_partial=0

usage() {
  cat <<'EOF'
Usage: scripts/experiments/run_marlin11_shortpaper.sh [options]

Read-only experiment orchestration over the frozen aligned11 artifact. By
default, main comparisons use seeds 13/42/73; the video-only and mechanism
diagnostics use seed 42. EPT-Net main seeds can be overridden independently.

Options:
  --suite NAME             all, main, modalities, or diagnostics (default: all)
  --seeds "13 42 73"       Main-comparison seeds
  --eptnet-seeds "13"      EPT-Net main seeds (default: same as --seeds)
  --analysis-seeds "42"    Modality/diagnostic seeds
  --device DEVICE          Training device (default: cuda:0)
  --python PATH            Python executable (default: $PYTHON_BIN or .venv/bin/python)
  --dry-run                Validate configs without data/GPU and print commands
  --preflight-only         Verify and fingerprint frozen inputs, then exit
  --skip-existing          Reuse only runs with metrics, predictions, and events
  --resume-partial         Resume an incomplete run with its trusted last.pt and best.pt
  -h, --help               Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      suite="$2"
      shift 2
      ;;
    --seeds)
      primary_seed_string="$2"
      shift 2
      ;;
    --eptnet-seeds)
      eptnet_seed_string="$2"
      shift 2
      ;;
    --analysis-seeds)
      analysis_seed_string="$2"
      shift 2
      ;;
    --device)
      device="$2"
      shift 2
      ;;
    --python)
      python_bin="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --preflight-only)
      preflight_only=1
      shift
      ;;
    --skip-existing)
      skip_existing=1
      shift
      ;;
    --resume-partial)
      resume_partial=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$suite" in
  all|main|modalities|diagnostics) ;;
  *)
    echo "--suite must be all, main, modalities, or diagnostics" >&2
    exit 2
    ;;
esac

read -r -a primary_seeds <<<"$primary_seed_string"
if [[ -z "$eptnet_seed_string" ]]; then
  eptnet_seed_string="$primary_seed_string"
fi
read -r -a eptnet_seeds <<<"$eptnet_seed_string"
read -r -a analysis_seeds <<<"$analysis_seed_string"
if [[ ${#primary_seeds[@]} -eq 0 \
      || ${#eptnet_seeds[@]} -eq 0 \
      || ${#analysis_seeds[@]} -eq 0 ]]; then
  echo "Seed lists must not be empty" >&2
  exit 2
fi
for seed in "${primary_seeds[@]}" "${eptnet_seeds[@]}" "${analysis_seeds[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "Seeds must be non-negative integers; got: $seed" >&2
    exit 2
  fi
done

main_matrix=(
  "configs/eptnet_marlin11_eeg_ppg_video.yaml|eptnet_marlin11_eeg_ppg_video_no_text|main"
  "configs/gru_marlin11_eeg_ppg_video.yaml|gru_marlin11_eeg_ppg_video_no_text|main"
  "configs/transformer_marlin11_eeg_ppg_video.yaml|transformer_marlin11_eeg_ppg_video_no_text|main"
)
modality_matrix=(
  "configs/eptnet_marlin11_video_only.yaml|eptnet_marlin11_video_only_no_text|analysis"
)
diagnostic_matrix=(
  "configs/eptnet_marlin11_fixed_reader.yaml|eptnet_marlin11_fixed_reader_no_text|analysis"
  "configs/eptnet_marlin11_no_persistent.yaml|eptnet_marlin11_no_persistent_no_text|analysis"
)

selected=()
if [[ "$suite" == "all" || "$suite" == "main" ]]; then
  selected+=("${main_matrix[@]}")
fi
if [[ "$suite" == "all" || "$suite" == "modalities" ]]; then
  selected+=("${modality_matrix[@]}")
fi
if [[ "$suite" == "all" || "$suite" == "diagnostics" ]]; then
  selected+=("${diagnostic_matrix[@]}")
fi

selected_configs=()
for entry in "${selected[@]}"; do
  IFS='|' read -r config _ _ <<<"$entry"
  selected_configs+=("$config")
done

print_command() {
  printf 'DRY-RUN:'
  printf ' %q' "$@"
  printf '\n'
}

run_command() {
  if [[ "$dry_run" -eq 1 ]]; then
    print_command "$@"
  else
    "$@"
  fi
}

preflight=(
  "$python_bin" -u scripts/experiments/preflight_marlin11_shortpaper.py
  "${selected_configs[@]}"
  --device "$device"
)
preflight_output="results/marlin11_shortpaper_preflight.json"
if [[ "$dry_run" -eq 1 ]]; then
  "${preflight[@]}" --allow-missing-data --skip-device-check
else
  if [[ -e "$preflight_output" ]]; then
    if [[ "$skip_existing" -eq 1 || "$resume_partial" -eq 1 ]]; then
      "${preflight[@]}" --reference "$preflight_output"
      echo "Preserving existing preflight record: $preflight_output"
    else
      echo "Refusing to overwrite preflight record: $preflight_output" >&2
      exit 1
    fi
  else
    "${preflight[@]}" --output "$preflight_output"
  fi
fi

if [[ "$suite" == "all" || "$suite" == "main" ]]; then
  fairness_output="results/marlin11_shortpaper_parameter_fairness.json"
  fairness_configs=(
    configs/eptnet_marlin11_eeg_ppg_video.yaml
    configs/gru_marlin11_eeg_ppg_video.yaml
    configs/transformer_marlin11_eeg_ppg_video.yaml
  )
  if [[ -e "$fairness_output" && "$dry_run" -eq 0 ]]; then
    if [[ "$skip_existing" -eq 1 || "$resume_partial" -eq 1 ]]; then
      "$python_bin" -u scripts/audit/audit_parameter_fairness.py \
        "${fairness_configs[@]}" \
        --check-reference "$fairness_output"
      echo "Preserving existing parameter-fairness audit: $fairness_output"
    else
      echo "Refusing to overwrite parameter-fairness audit: $fairness_output" >&2
      exit 1
    fi
  else
    run_command "$python_bin" -u scripts/audit/audit_parameter_fairness.py \
      "${fairness_configs[@]}" \
      --device cpu \
      --output "$fairness_output"
  fi
fi

if [[ "$preflight_only" -eq 1 ]]; then
  echo "Frozen aligned11 preflight completed successfully."
  exit 0
fi

run_one() {
  local config="$1"
  local experiment="$2"
  local seed="$3"
  local run_dir="results/${experiment}/seed_${seed}"
  local metrics="$run_dir/test_metrics.json"
  local predictions="$run_dir/test_predictions.jsonl"
  local events="$run_dir/test_events.json"
  local resume_checkpoint="$run_dir/last.pt"
  local best_checkpoint="$run_dir/best.pt"
  local resume_from=""

  if [[ -e "$run_dir" && "$dry_run" -eq 0 ]]; then
    if [[ "$skip_existing" -eq 1 \
          && -f "$metrics" \
          && -f "$predictions" \
          && -f "$events" ]]; then
      "$python_bin" -u scripts/experiments/verify_marlin11_shortpaper_results.py run \
        --config "$config" \
        --seed "$seed" \
        --metrics "$metrics" \
        --predictions "$predictions" \
        --events "$events" \
        --preflight "$preflight_output"
      echo "Reusing completed run: $run_dir"
      return
    fi
    if [[ "$resume_partial" -eq 1 \
          && ! -f "$metrics" \
          && -f "$resume_checkpoint" \
          && -f "$best_checkpoint" ]]; then
      echo "Resuming incomplete run from: $resume_checkpoint"
      resume_from="$resume_checkpoint"
    else
      echo "Refusing to overwrite existing or partial run directory: $run_dir" >&2
      echo "A completed run requires metrics, predictions, and events; otherwise inspect it manually." >&2
      echo "Use --skip-existing only for complete runs or --resume-partial with last.pt and best.pt." >&2
      exit 1
    fi
  fi

  if [[ -n "$resume_from" ]]; then
    run_command "$python_bin" -u -m eptnet.train \
      --config "$config" \
      --seed "$seed" \
      --device "$device" \
      --resume "$resume_from"
  else
    run_command "$python_bin" -u -m eptnet.train \
      --config "$config" \
      --seed "$seed" \
      --device "$device"
  fi
  run_command "$python_bin" -u -m eptnet.evaluate \
    --config "$config" \
    --checkpoint "$run_dir/best.pt" \
    --output "$metrics" \
    --predictions-output "$predictions" \
    --events-output "$events" \
    --device "$device"
}

for entry in "${selected[@]}"; do
  IFS='|' read -r config experiment group <<<"$entry"
  if [[ "$group" == "main" ]]; then
    if [[ "$experiment" == "eptnet_marlin11_eeg_ppg_video_no_text" ]]; then
      seeds=("${eptnet_seeds[@]}")
    else
      seeds=("${primary_seeds[@]}")
    fi
  else
    seeds=("${analysis_seeds[@]}")
  fi
  metric_files=()
  for seed in "${seeds[@]}"; do
    run_one "$config" "$experiment" "$seed"
    metric_files+=("results/${experiment}/seed_${seed}/test_metrics.json")
  done
  aggregate_json="results/${experiment}/aggregate.json"
  aggregate_csv="results/${experiment}/aggregate.csv"
  if [[ ( -e "$aggregate_json" || -e "$aggregate_csv" ) && "$dry_run" -eq 0 ]]; then
    if [[ ( "$skip_existing" -eq 1 || "$resume_partial" -eq 1 ) \
          && -f "$aggregate_json" \
          && -f "$aggregate_csv" ]]; then
      if "$python_bin" -u scripts/experiments/verify_marlin11_shortpaper_results.py aggregate \
          "${metric_files[@]}" \
          --aggregate-json "$aggregate_json" \
          --aggregate-csv "$aggregate_csv" \
          --preflight "$preflight_output"; then
        echo "Preserving identity-verified aggregate: $experiment"
        continue
      elif [[ "$resume_partial" -ne 1 ]]; then
        echo "Existing aggregate failed read-only identity verification: $experiment" >&2
        exit 1
      fi
    elif [[ "$resume_partial" -ne 1 ]]; then
      echo "Refusing to reuse incomplete or unverified aggregate: $experiment" >&2
      exit 1
    fi

    if [[ "$resume_partial" -eq 1 ]]; then
      seed_tag="${seeds[*]}"
      seed_tag="${seed_tag// /-}"
      aggregate_json="results/${experiment}/aggregate_recovered_seeds_${seed_tag}.json"
      aggregate_csv="results/${experiment}/aggregate_recovered_seeds_${seed_tag}.csv"
      if [[ -e "$aggregate_json" || -e "$aggregate_csv" ]]; then
        echo "Refusing to overwrite recovered aggregate target: $aggregate_json / $aggregate_csv" >&2
        exit 1
      fi
      echo "Writing recovered aggregate to unused paths: $aggregate_json / $aggregate_csv"
    else
      echo "Refusing to overwrite existing aggregate: $experiment" >&2
      exit 1
    fi
  fi
  run_command "$python_bin" -u -m eptnet.aggregate "${metric_files[@]}" \
    --output "$aggregate_json" \
    --csv "$aggregate_csv" \
    --no-overwrite
done

echo "Frozen aligned11 paper experiment matrix completed successfully."
