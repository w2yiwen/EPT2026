#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
SOURCE_ROOT="${1:-data/raw/bci_subjects_ept_v1}"
OUTPUT_ROOT="${2:-data/processed/bci_subjects_ept_v6_session_registered_no_facial}"
DATASET_NAME="${3:-bci_subjects_ept_v6_session_registered_no_facial}"

PYTHONPATH=src python3 -m eptnet.data.prepare_bci_subjects \
  --source "$SOURCE_ROOT" \
  --raw-output "$SOURCE_ROOT" \
  --output "$OUTPUT_ROOT" \
  --dataset-name "$DATASET_NAME" \
  --reuse-staged-raw \
  --required-profile model_contract \
  --video-backend legacy \
  --audio-backend none \
  --text-backend hash \
  --alignment-mode session_registered_time \
  --no-include-facial-csv \
  --no-write-compatibility-windows \
  --window-size 64 \
  --stride 32 \
  --min-window-size 16
