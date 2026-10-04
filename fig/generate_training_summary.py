#!/usr/bin/env python3
"""Generate the available training figure and Table 2 after a campaign finishes."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path
from typing import Any

FIG_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FIG_ROOT.parent
DEFAULT_CAMPAIGN = Path("results/dual4090_seed42_3way")
MAIN_MODELS = {
    "LSTR": "lstr_marlin11_eeg_ppg_video_no_text",
    "GateHUB": "gatehub_marlin11_eeg_ppg_video_no_text",
    "TeSTra": "testra_marlin11_eeg_ppg_video_no_text",
    "EPT-Net (ours)": "eptnet_marlin11_eeg_ppg_video_no_text",
}


def _function(path: Path, name: str):
    namespace = runpy.run_path(str(path))
    return namespace[name]


def _validation_curve_missing(campaign: Path) -> list[str]:
    missing: list[str] = []
    for experiment in MAIN_MODELS.values():
        run_dirs = sorted((campaign / experiment).glob("seed_*"))
        if not run_dirs:
            missing.append(f"{campaign / experiment}/seed_*")
            continue
        for run_dir in run_dirs:
            path = run_dir / "validation_epoch_metrics.jsonl"
            if not path.is_file():
                missing.append(str(path))
    return missing


def generate(campaign_root: str | Path, *, require_validation_curve: bool = False) -> dict[str, Any]:
    campaign = Path(campaign_root)
    campaign = campaign if campaign.is_absolute() else PROJECT_ROOT / campaign
    campaign = campaign.resolve()

    loss_generate = _function(
        FIG_ROOT / "fig01_training_loss" / "plot_fig01_training_loss.py", "generate"
    )
    table_generate = _function(
        FIG_ROOT
        / "table02_overall_performance"
        / "generate_table02_overall_performance.py",
        "generate",
    )
    report: dict[str, Any] = {
        "loss_figure": loss_generate(
            run_specs=[str(campaign / MAIN_MODELS["EPT-Net (ours)"])]
        ),
        "table02": table_generate(campaign_root=campaign),
    }

    missing = _validation_curve_missing(campaign)
    if missing:
        report["validation_performance_figure"] = {
            "status": "SKIPPED",
            "reason": (
                "Measured per-epoch validation metrics are absent; test metrics and "
                "validation loss are not substituted."
            ),
            "missing": missing,
        }
        if require_validation_curve:
            raise FileNotFoundError(json.dumps(report["validation_performance_figure"], indent=2))
    else:
        convergence_generate = _function(
            FIG_ROOT
            / "fig01_training_convergence"
            / "plot_fig01_training_convergence.py",
            "generate",
        )
        specs = [
            f"{label}={campaign / experiment}"
            for label, experiment in MAIN_MODELS.items()
        ]
        report["validation_performance_figure"] = convergence_generate(run_specs=specs)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument(
        "--require-validation-curve",
        action="store_true",
        help="Fail instead of skipping the AP panel when per-epoch validation metrics are missing",
    )
    args = parser.parse_args()
    report = generate(
        args.campaign_root,
        require_validation_curve=args.require_validation_curve,
    )
    print(json.dumps({"status": "PASS", **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
