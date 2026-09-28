#!/usr/bin/env python3
"""Regenerate publication-quality optimization diagnostics for one training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eptnet.training_artifacts import generate_training_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing history.json",
    )
    args = parser.parse_args()
    manifest = generate_training_artifacts(args.run_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
