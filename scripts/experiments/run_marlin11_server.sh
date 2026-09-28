#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
ACTION="${1:-check}"

MAIN_CONFIG="configs/eptnet_v6_marlin11_4060_windowed_seed42.yaml"
MAIN_RUN="results/eptnet_v6_marlin11_4060_windowed_seed42/seed_${SEED}"
GRU_CONFIG="configs/baseline_early_fusion_gru_marlin11_4060_windowed_seed42.yaml"
GRU_RUN="results/baseline_early_fusion_gru_marlin11_4060_windowed_seed42/seed_${SEED}"
CLASSICAL="results/baselines/marlin11_classical_seed42.json"

export PYTHONPATH="${PWD}/src"
mkdir -p logs results/baselines

verify() {
  PYTHON_BIN="$PYTHON_BIN" DEVICE="$DEVICE" ./scripts/server/verify_marlin11_gpu.sh
}

train_new() {
  local config="$1"
  "$PYTHON_BIN" -u -m eptnet.train \
    --config "$config" --seed "$SEED" --device "$DEVICE" --progress
}

train_resume() {
  local config="$1" run="$2"
  [[ -f "$run/last.pt" ]] || { echo "Missing resume checkpoint: $run/last.pt" >&2; exit 2; }
  "$PYTHON_BIN" -u -m eptnet.train \
    --config "$config" --seed "$SEED" --device "$DEVICE" \
    --resume "$run/last.pt" --progress
}

evaluate_run() {
  local config="$1" run="$2"
  [[ -f "$run/best.pt" ]] || { echo "Missing checkpoint: $run/best.pt" >&2; exit 2; }
  "$PYTHON_BIN" -u -m eptnet.evaluate \
    --config "$config" --checkpoint "$run/best.pt" \
    --output "$run/test_metrics.json" --device "$DEVICE"
}

complete_run() {
  local config="$1" run="$2"
  if [[ -f "$run/test_metrics.json" ]]; then
    echo "SKIP completed run: $run"
    return
  fi
  if [[ -f "$run/last.pt" ]]; then
    train_resume "$config" "$run"
  else
    train_new "$config"
  fi
  evaluate_run "$config" "$run"
}

case "$ACTION" in
  check)
    verify
    ;;
  smoke-main)
    verify
    "$PYTHON_BIN" -u -m eptnet.train --config "$MAIN_CONFIG" --seed "$SEED" \
      --device "$DEVICE" --smoke --progress
    ;;
  train-main)
    verify
    train_new "$MAIN_CONFIG"
    ;;
  resume-main)
    verify
    train_resume "$MAIN_CONFIG" "$MAIN_RUN"
    ;;
  eval-main)
    verify
    evaluate_run "$MAIN_CONFIG" "$MAIN_RUN"
    ;;
  classical)
    verify
    "$PYTHON_BIN" scripts/experiments/run_classical_baselines.py \
      --config "$MAIN_CONFIG" --output "$CLASSICAL" --device "$DEVICE"
    ;;
  smoke-gru)
    verify
    "$PYTHON_BIN" -u -m eptnet.train --config "$GRU_CONFIG" --seed "$SEED" \
      --device "$DEVICE" --smoke --progress
    ;;
  train-gru)
    verify
    train_new "$GRU_CONFIG"
    ;;
  resume-gru)
    verify
    train_resume "$GRU_CONFIG" "$GRU_RUN"
    ;;
  eval-gru)
    verify
    evaluate_run "$GRU_CONFIG" "$GRU_RUN"
    ;;
  figures)
    "$PYTHON_BIN" fig/generate_all.py
    ;;
  full)
    verify
    "$PYTHON_BIN" scripts/experiments/run_classical_baselines.py \
      --config "$MAIN_CONFIG" --output "$CLASSICAL" --device "$DEVICE"
    complete_run "$MAIN_CONFIG" "$MAIN_RUN"
    complete_run "$GRU_CONFIG" "$GRU_RUN"
    "$PYTHON_BIN" fig/generate_all.py
    ;;
  *)
    echo "Usage: $0 {check|smoke-main|train-main|resume-main|eval-main|classical|smoke-gru|train-gru|resume-gru|eval-gru|figures|full}" >&2
    exit 2
    ;;
esac
