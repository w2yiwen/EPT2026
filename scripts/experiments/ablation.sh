#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
for config in configs/ablation_*.yaml; do
  bash scripts/experiments/train.sh "$config"
done
