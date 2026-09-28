#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL="${1:-main}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-2}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
SEED="${SEED:-42}"

case "$MODEL" in
  main)
    CONFIG="configs/eptnet_marlin11_aligned_windowed_seed42.yaml"
    RUN_DIR="results/eptnet_marlin11_aligned_windowed_seed42/seed_${SEED}"
    ;;
  gru)
    CONFIG="configs/gru_marlin11_aligned_windowed_seed42.yaml"
    RUN_DIR="results/gru_marlin11_aligned_windowed_seed42/seed_${SEED}"
    ;;
  *)
    echo "Usage: $0 {main|gru}" >&2
    exit 2
    ;;
esac

while true; do
  printf '\033[2J\033[H'
  printf 'EPT-Net training monitor | model=%s | %s\n\n' "$MODEL" "$(date '+%F %T')"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits | \
    awk -F',' '{printf "GPU: %s | util=%s%% | memory=%s/%s MiB | temp=%s C | power=%s W\n", $1,$2,$3,$4,$5,$6}'

  PYTHONPATH=src "$PYTHON_BIN" - "$CONFIG" "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

from eptnet.config import load_config

config_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
config = load_config(config_path)
maximum = int(config["training"]["epochs"])
history_path = run_dir / "history.json"

if history_path.is_file():
    history = json.loads(history_path.read_text(encoding="utf-8"))
else:
    history = []

completed = len(history)
percent = 100.0 * completed / maximum
bar_width = 40
filled = min(bar_width, round(bar_width * completed / maximum))
bar = "#" * filled + "-" * (bar_width - filled)
print(f"Epochs: [{bar}] {completed}/{maximum} ({percent:.1f}%)")
print(f"Run:    {run_dir}")

if history:
    last = history[-1]
    print(
        "Latest: epoch={epoch} train={train:.4f} val={val:.4f} "
        "best_val={best:.4f} epoch_time={elapsed:.1f}s".format(
            epoch=int(last["epoch"]),
            train=float(last["train"]["total"]),
            val=float(last["val"]["total"]),
            best=float(last["best_val_loss"]),
            elapsed=float(last["elapsed_seconds"]),
        )
    )

if (run_dir / "test_metrics.json").is_file():
    state = "complete and evaluated"
elif (run_dir / "last.pt").is_file():
    state = "training/checkpoint available"
else:
    state = "waiting for first checkpoint"
print(f"State:  {state}")
PY

  printf '\nProcesses:\n'
  pgrep -af 'eptnet\.(train|evaluate)' || printf 'No active EPT-Net train/evaluate process.\n'
  printf '\nRefresh: every %ss; Ctrl+C exits monitor only.\n' "$INTERVAL_SECONDS"
  sleep "$INTERVAL_SECONDS"
done
