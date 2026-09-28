#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
SOURCE_ROOT="${1:-../BCI}"
OUTPUT_ROOT="${2:-data/processed/bci_subjects_ept_v2_macbert}"
RAW_ROOT="${3:-data/raw/bci_subjects_ept_v2_macbert}"
MODEL_CACHE="${4:-../third_party/research_candidates/hf_cache}"
USE_TF=0 TRANSFORMERS_NO_TF=1 PYTHONPATH=src python3 -m eptnet.data.prepare_bci_subjects \
  --source "$SOURCE_ROOT" \
  --output "$OUTPUT_ROOT" \
  --raw-output "$RAW_ROOT" \
  --dataset-name bci_subjects_ept_v2_macbert \
  --video-backend legacy \
  --audio-backend none \
  --text-backend macbert \
  --behavior-device cuda \
  --behavior-batch-size 32 \
  --model-cache "$MODEL_CACHE" \
  --local-files-only \
  --window-size 64 \
  --stride 32 \
  --min-window-size 16
