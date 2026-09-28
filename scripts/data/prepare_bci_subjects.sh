#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
SOURCE_ROOT="${1:-../BCI}"
OUTPUT_ROOT="${2:-data/processed/bci_subjects_ept_v1}"
RAW_ROOT="${3:-data/raw/bci_subjects_ept_v1}"
REQUIRED_PROFILE="${4:-model_contract}"
VIDEO_BACKEND="${5:-legacy}"
AUDIO_BACKEND="${6:-none}"
TEXT_BACKEND="${7:-hash}"
BEHAVIOR_DEVICE="${8:-auto}"
PYTHONPATH=src python3 -m eptnet.data.prepare_bci_subjects \
  --source "$SOURCE_ROOT" \
  --output "$OUTPUT_ROOT" \
  --raw-output "$RAW_ROOT" \
  --required-profile "$REQUIRED_PROFILE" \
  --video-backend "$VIDEO_BACKEND" \
  --audio-backend "$AUDIO_BACKEND" \
  --text-backend "$TEXT_BACKEND" \
  --behavior-device "$BEHAVIOR_DEVICE" \
  --window-size 64 \
  --stride 32 \
  --min-window-size 16
