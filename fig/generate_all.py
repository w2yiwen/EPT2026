#!/usr/bin/env python3
from __future__ import annotations

import json
import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    scripts = [
        root / "fig01-training-dynamics" / "plot_fig01_training_dynamics.py",
        root / "fig02-model-comparison" / "plot_fig02_model_comparison.py",
        root / "fig03-heldout-uncertainty" / "plot_fig03_heldout_uncertainty.py",
    ]
    reports = []
    for script in scripts:
        namespace = runpy.run_path(str(script))
        reports.append(namespace["generate"]())
    print(json.dumps({"status": "PASS", "figures": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
