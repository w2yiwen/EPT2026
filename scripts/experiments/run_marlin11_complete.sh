#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"

mode="${1:-formal}"
python_bin="${PYTHON_BIN:-$project_root/.venv/bin/python}"
device="${DEVICE:-cuda:0}"
eptnet_seeds="${EPTNET_SEEDS:-13}"
comparison_seeds="${COMPARISON_SEEDS:-42}"
modality_seeds="${MODALITY_SEEDS:-42}"
diagnostic_seeds="${DIAGNOSTIC_SEEDS:-42}"
smoke_seed="${SMOKE_SEED:-42}"
trace_seed="${TRACE_SEED:-${eptnet_seeds%% *}}"

usage() {
  cat <<'EOF'
Usage: scripts/experiments/run_marlin11_complete.sh [validate|formal]

validate  Run tests, strict frozen-data preflight, one-session CUDA smoke for
          every declared configuration, and DEMO-only figure export. It never
          reads the test split and never starts a full experiment.

formal    Run the complete declared experiment matrix, evaluation, aggregation,
          and formal paper-figure export. This is the default.

Environment overrides:
  PYTHON_BIN       Python executable (default: .venv/bin/python)
  DEVICE           Training device (default: cuda:0)
  EPTNET_SEEDS     Full EPT-Net seeds (default: "13")
  COMPARISON_SEEDS GRU/Transformer seeds (default: "42")
  MODALITY_SEEDS   Video-only diagnostic seeds (default: "42")
  DIAGNOSTIC_SEEDS Mechanism-diagnostic seeds (default: "42")
  TRACE_SEED       Full EPT-Net seed used for trace figures (default: first
                   value in EPTNET_SEEDS)
  SMOKE_SEED       Validation-only smoke seed (default: "42")
EOF
}

case "$mode" in
  validate|formal) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ ! -x "$python_bin" ]]; then
  echo "[ERROR] Python is not executable: $python_bin" >&2
  echo "Create the project environment as documented in README.md." >&2
  exit 1
fi
if [[ " $eptnet_seeds " != *" $trace_seed "* ]]; then
  echo "[ERROR] TRACE_SEED must be present in EPTNET_SEEDS." >&2
  exit 1
fi

python_dir="$(cd "$(dirname "$python_bin")" && pwd)"
export PATH="$python_dir:$PATH"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs results

echo "[START] mode=$mode device=$device python=$python_bin"
"$python_bin" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
print(
    "[ENV]",
    f"torch={torch.__version__}",
    f"cuda={torch.version.cuda}",
    f"gpu={torch.cuda.get_device_name(0)}",
)
PY

echo "[CHECK] Python test suite"
"$python_bin" -m pytest -q

echo "[CHECK] command/config dry run"
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --python "$python_bin" \
  --device "$device" \
  --dry-run

echo "[CHECK] frozen aligned11 data and configuration preflight"
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite all \
  --python "$python_bin" \
  --device "$device" \
  --preflight-only \
  --skip-existing

if [[ "$mode" == "validate" ]]; then
  validation_tag="$(date -u +'%Y%m%dT%H%M%SZ')"
  validation_root="results/validation/$validation_tag"
  configs=(
    configs/eptnet_marlin11_eeg_ppg_video.yaml
    configs/gru_marlin11_eeg_ppg_video.yaml
    configs/transformer_marlin11_eeg_ppg_video.yaml
    configs/eptnet_marlin11_video_only.yaml
    configs/eptnet_marlin11_fixed_reader.yaml
    configs/eptnet_marlin11_no_persistent.yaml
  )
  for config in "${configs[@]}"; do
    name="$(basename "$config" .yaml)"
    echo "[SMOKE] config=$config"
    "$python_bin" -u -m eptnet.train \
      --config "$config" \
      --seed "$smoke_seed" \
      --device "$device" \
      --output-dir "$validation_root/$name" \
      --smoke \
      --no-progress
  done
  echo "[FIGURE] synthetic layout/export QA only"
  "$python_bin" fig/generate_all.py --suite paper --demo
  echo "[OUTPUT] validation_root=$validation_root"
  echo "[DONE] Validation passed; no full experiment was started."
  exit 0
fi

test_manifest="data/processed/bci_subjects_ept_v6_marlin4060_aligned11/manifests/sessions_test.jsonl"
selection_file="results/marlin11_shortpaper_figure_sample_id.txt"
sample_id="$("$python_bin" -c 'import json, sys; row=json.loads(next(line for line in open(sys.argv[1], encoding="utf-8") if line.strip())); print(row.get("metadata", {}).get("session_id") or row["sample_id"])' "$test_manifest")"
if [[ -f "$selection_file" ]]; then
  recorded_sample_id="$(tr -d '\r\n' < "$selection_file")"
  if [[ "$recorded_sample_id" != "$sample_id" ]]; then
    echo "[ERROR] Predeclared figure sample differs from the frozen test manifest." >&2
    echo "recorded=$recorded_sample_id current=$sample_id" >&2
    exit 1
  fi
else
  printf '%s\n' "$sample_id" > "$selection_file"
fi
echo "[CHECK] predeclared_figure_sample=$sample_id"

echo "[RUN] main comparison eptnet_seeds=$eptnet_seeds comparison_seeds=$comparison_seeds"
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite main \
  --eptnet-seeds "$eptnet_seeds" \
  --seeds "$comparison_seeds" \
  --python "$python_bin" \
  --device "$device" \
  --skip-existing \
  --resume-partial

echo "[RUN] video-only diagnostic seeds=$modality_seeds"
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite modalities \
  --analysis-seeds "$modality_seeds" \
  --python "$python_bin" \
  --device "$device" \
  --skip-existing \
  --resume-partial

echo "[RUN] mechanism diagnostics seeds=$diagnostic_seeds"
bash scripts/experiments/run_marlin11_shortpaper.sh \
  --suite diagnostics \
  --analysis-seeds "$diagnostic_seeds" \
  --python "$python_bin" \
  --device "$device" \
  --skip-existing \
  --resume-partial

full_run="results/eptnet_marlin11_eeg_ppg_video_no_text/seed_${trace_seed}"
echo "[FIGURE] formal dynamic-tracking and two-group evidence figures sample=$sample_id"
"$python_bin" fig/generate_all.py --suite paper \
  --predictions "$full_run/test_predictions.jsonl" \
  --events "$full_run/test_events.json" \
  --metrics "$full_run/test_metrics.json" \
  --sample-id "$sample_id" \
  --modality-aggregate 'Full=results/eptnet_marlin11_eeg_ppg_video_no_text/aggregate.json' \
  --modality-aggregate 'Video=results/eptnet_marlin11_video_only_no_text/aggregate.json'

echo "[OUTPUT] metrics_and_checkpoints=$project_root/results"
echo "[OUTPUT] figures=$project_root/fig/fig04_dynamic_tracking,$project_root/fig/fig05_modality_evidence"
echo "[DONE] Complete experiment matrix, evaluation, aggregation, and figures finished."
