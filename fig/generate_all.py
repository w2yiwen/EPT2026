#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path


def _load_generate(script: Path):
    namespace = runpy.run_path(str(script))
    return namespace["generate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible EPT paper figures")
    parser.add_argument(
        "--suite",
        choices=("legacy", "paper", "all"),
        default="legacy",
        help="legacy diagnostics, paper evidence figures, or both",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Render only the paper suite with clearly labelled synthetic layout-QA inputs",
    )
    parser.add_argument("--predictions", help="evaluate.py *_predictions.jsonl")
    parser.add_argument("--events", help="evaluate.py *_events.json")
    parser.add_argument("--metrics", help="evaluate.py test_metrics.json")
    parser.add_argument("--sample-id", help="Predeclared held-out sample for the trace figure")
    parser.add_argument("--frame-threshold", type=float)
    parser.add_argument("--modality-evidence", help="Combined modality evidence JSON/CSV")
    parser.add_argument(
        "--modality-aggregate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="aggregate.py output; repeat for Full, EEG+PPG, Video",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    legacy_scripts = [
        root / "fig01_training_dynamics" / "plot_fig01_training_dynamics.py",
        root / "fig02_model_comparison" / "plot_fig02_model_comparison.py",
        root / "fig03_heldout_uncertainty" / "plot_fig03_heldout_uncertainty.py",
    ]
    if args.demo and args.suite in {"legacy", "all"}:
        args.suite = "paper"

    reports: list[dict] = []
    if args.suite in {"legacy", "all"}:
        for script in legacy_scripts:
            reports.append(_load_generate(script)())

    if args.suite in {"paper", "all"}:
        dynamic = _load_generate(
            root / "fig04_dynamic_tracking" / "plot_fig04_dynamic_tracking.py"
        )
        modality = _load_generate(
            root / "fig05_modality_evidence" / "plot_fig05_modality_evidence.py"
        )
        reports.append(
            dynamic(
                predictions_path=args.predictions,
                events_path=args.events,
                metrics_path=args.metrics,
                sample_id=args.sample_id,
                frame_threshold=args.frame_threshold,
                demo=args.demo,
            )
        )
        reports.append(
            modality(
                evidence_path=args.modality_evidence,
                aggregate_specs=args.modality_aggregate,
                demo=args.demo,
            )
        )
    print(json.dumps({"status": "PASS", "figures": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
