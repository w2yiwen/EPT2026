#!/usr/bin/env bash
set -uo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
campaign_root="${CAMPAIGN_ROOT:-results/dual4090_seed42_3way}"
log_root="${LOG_ROOT:-logs/dual4090_seed42_3way}"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$campaign_root" "$log_root"

if [[ ! -x "$python_bin" ]]; then
  echo "[ERROR] Python is not executable: $python_bin" >&2
  exit 1
fi

run_one() {
  local gpu="$1"
  local config="$2"
  local experiment="$3"
  local label="$4"
  local experiment_root="${campaign_root}/${experiment}"
  local run_dir="${experiment_root}/seed_42"
  local metrics="${run_dir}/test_metrics.json"
  local predictions="${run_dir}/test_predictions.jsonl"
  local events="${run_dir}/test_events.json"
  local best="${run_dir}/best.pt"
  local last="${run_dir}/last.pt"
  local aggregate_json="${experiment_root}/aggregate_seed_42.json"
  local aggregate_csv="${experiment_root}/aggregate_seed_42.csv"

  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[START] $(date -u +%FT%TZ) label=${label} physical_gpu=${gpu} config=${config}"

  if [[ -f "$metrics" && -f "$predictions" && -f "$events" ]]; then
    echo "[REUSE] complete evaluation artifacts: $run_dir"
  elif [[ -f "$last" && -f "$best" ]]; then
    echo "[RESUME] $last"
    "$python_bin" -u -m eptnet.train \
      --config "$config" --seed 42 --device cuda \
      --output-dir "$experiment_root" --resume "$last" || return 1
    "$python_bin" -u -m eptnet.evaluate \
      --config "$config" --checkpoint "$best" \
      --output "$metrics" --predictions-output "$predictions" \
      --events-output "$events" --device cuda || return 1
  elif [[ -e "$run_dir" ]]; then
    echo "[ERROR] partial run lacks both resumable checkpoints: $run_dir" >&2
    return 1
  else
    "$python_bin" -u -m eptnet.train \
      --config "$config" --seed 42 --device cuda \
      --output-dir "$experiment_root" || return 1
    "$python_bin" -u -m eptnet.evaluate \
      --config "$config" --checkpoint "$best" \
      --output "$metrics" --predictions-output "$predictions" \
      --events-output "$events" --device cuda || return 1
  fi

  if [[ -f "$aggregate_json" && -f "$aggregate_csv" ]]; then
    echo "[REUSE] aggregate pair: $experiment_root"
  elif [[ -e "$aggregate_json" || -e "$aggregate_csv" ]]; then
    echo "[ERROR] incomplete aggregate pair: $experiment_root" >&2
    return 1
  else
    "$python_bin" -u -m eptnet.aggregate "$metrics" \
      --output "$aggregate_json" --csv "$aggregate_csv" --no-overwrite || return 1
  fi
  echo "[DONE] $(date -u +%FT%TZ) label=${label} physical_gpu=${gpu}"
}

slot_gpu0_a() {
  run_one 0 configs/eptnet_marlin11_eeg_ppg_video.yaml \
    eptnet_marlin11_eeg_ppg_video_no_text EPT-Net &&
  run_one 0 configs/eptnet_marlin11_video_only.yaml \
    eptnet_marlin11_video_only_no_text Video-only
}

slot_gpu0_b() {
  run_one 0 configs/eptnet_marlin11_fixed_reader.yaml \
    eptnet_marlin11_fixed_reader_no_text Fixed-reader
}

slot_gpu0_c() {
  run_one 0 configs/eptnet_marlin11_no_persistent.yaml \
    eptnet_marlin11_no_persistent_no_text No-persistent
}

slot_gpu1_a() {
  run_one 1 configs/lstr_marlin11_eeg_ppg_video.yaml \
    lstr_marlin11_eeg_ppg_video_no_text LSTR
}

slot_gpu1_b() {
  run_one 1 configs/gatehub_marlin11_eeg_ppg_video.yaml \
    gatehub_marlin11_eeg_ppg_video_no_text GateHUB
}

slot_gpu1_c() {
  run_one 1 configs/testra_marlin11_eeg_ppg_video.yaml \
    testra_marlin11_eeg_ppg_video_no_text TeSTra
}

declare -a pids=()
declare -a slots=()

launch_slot() {
  local slot="$1"
  local function_name="$2"
  "$function_name" > "${log_root}/${slot}.log" 2>&1 &
  pids+=("$!")
  slots+=("$slot")
  echo "[LAUNCH] slot=$slot pid=$!"
}

launch_slot gpu0_a slot_gpu0_a
launch_slot gpu0_b slot_gpu0_b
launch_slot gpu0_c slot_gpu0_c
launch_slot gpu1_a slot_gpu1_a
launch_slot gpu1_b slot_gpu1_b
launch_slot gpu1_c slot_gpu1_c

terminate_children() {
  local pid
  for pid in "${pids[@]}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[SLOT_DONE] ${slots[$index]}"
  else
    code=$?
    echo "[SLOT_FAILED] ${slots[$index]} exit=$code" >&2
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "[ERROR] at least one training slot failed; figures were not generated" >&2
  exit 1
fi

echo "[FIGURE] formal paper suite"
"$python_bin" fig/generate_all.py --suite paper \
  --main-aggregate "EPT-Net=${campaign_root}/eptnet_marlin11_eeg_ppg_video_no_text/aggregate_seed_42.json" \
  --main-aggregate "LSTR=${campaign_root}/lstr_marlin11_eeg_ppg_video_no_text/aggregate_seed_42.json" \
  --main-aggregate "GateHUB=${campaign_root}/gatehub_marlin11_eeg_ppg_video_no_text/aggregate_seed_42.json" \
  --main-aggregate "TeSTra=${campaign_root}/testra_marlin11_eeg_ppg_video_no_text/aggregate_seed_42.json" \
  --predictions "${campaign_root}/eptnet_marlin11_eeg_ppg_video_no_text/seed_42/test_predictions.jsonl" \
  --events "${campaign_root}/eptnet_marlin11_eeg_ppg_video_no_text/seed_42/test_events.json" \
  --metrics "${campaign_root}/eptnet_marlin11_eeg_ppg_video_no_text/seed_42/test_metrics.json" \
  --sample-id session_008 \
  --modality-aggregate "Full=${campaign_root}/eptnet_marlin11_eeg_ppg_video_no_text/aggregate_seed_42.json" \
  --modality-aggregate "Video=${campaign_root}/eptnet_marlin11_video_only_no_text/aggregate_seed_42.json"

experiments=(
  eptnet_marlin11_eeg_ppg_video_no_text
  eptnet_marlin11_fixed_reader_no_text
  eptnet_marlin11_no_persistent_no_text
  lstr_marlin11_eeg_ppg_video_no_text
  gatehub_marlin11_eeg_ppg_video_no_text
  testra_marlin11_eeg_ppg_video_no_text
  eptnet_marlin11_video_only_no_text
)
for experiment in "${experiments[@]}"; do
  run_dir="${campaign_root}/${experiment}/seed_42"
  for path in \
    resolved_config.yaml run_metadata.json history.json best.pt last.pt \
    test_metrics.json test_predictions.jsonl test_events.json \
    figures/training_history.csv figures/fig_training_dynamics.pdf \
    figures/fig_training_dynamics.png figures/training_figure_manifest.json; do
    test -s "${run_dir}/${path}" || {
      echo "[ERROR] missing artifact: ${run_dir}/${path}" >&2
      exit 1
    }
  done
  test -s "${campaign_root}/${experiment}/aggregate_seed_42.json" || exit 1
  test -s "${campaign_root}/${experiment}/aggregate_seed_42.csv" || exit 1
done

for figure in fig03_main_comparison fig04_dynamic_tracking fig05_modality_evidence; do
  for extension in pdf svg png; do
    test -s "fig/${figure}/${figure}.${extension}" || exit 1
  done
  test -s "fig/${figure}/${figure}.qa-report.json" || exit 1
done

find "$campaign_root" \
  fig/fig03_main_comparison fig/fig04_dynamic_tracking fig/fig05_modality_evidence \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${log_root}/artifact_manifest.sha256"

echo "[DONE] complete seven-run matrix, evaluation, aggregation, training plots, paper figures, and artifact validation"
