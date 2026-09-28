#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
DATASET="data/processed/bci_subjects_ept_v6_marlin4060_aligned11"
CONFIG="configs/eptnet_v6_marlin11_4060_windowed_seed42.yaml"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -L data ]]; then
  echo "Expected data to be a symlink; refusing to duplicate the server dataset." >&2
  exit 2
fi
if [[ ! -f "$DATASET/_SUCCESS.json" ]]; then
  echo "Prepared aligned11 dataset is incomplete: $DATASET" >&2
  exit 2
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

PYTHONPATH=src "$PYTHON_BIN" - "$DEVICE" "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

import torch

from eptnet.config import load_config
from eptnet.train import resolve_device, validate_training_cohort

requested_device, config_path = sys.argv[1:]
device = resolve_device(requested_device)
if device.type != "cuda":
    raise RuntimeError(f"GPU-only server route resolved to {device}")

config = load_config(config_path)
required = (
    "use_video",
    "use_audio",
    "use_text",
    "use_eeg_time",
    "use_eeg_spec",
    "use_hr",
)
disabled = [name for name in required if not bool(config["model"].get(name))]
if disabled:
    raise RuntimeError(f"Required modalities are disabled: {disabled}")
if len(config["data"].get("expected_session_ids", [])) != 11:
    raise RuntimeError("The server route must declare exactly 11 aligned sessions")

summary_path = Path("data/processed/bci_subjects_ept_v6_marlin4060_aligned11/dataset_summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if int(summary["num_subjects"]) != 11:
    raise RuntimeError(f"Expected 11 subjects, got {summary['num_subjects']}")

cohort = validate_training_cohort(config)
print(json.dumps({
    "status": "PASS",
    "device": str(device),
    "gpu": torch.cuda.get_device_name(device),
    "cuda": torch.version.cuda,
    "subjects": 11,
    "modalities": list(required),
    "cohort": cohort,
}, ensure_ascii=False, default=str))
PY
