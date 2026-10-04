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
        choices=("paper", "training", "all"),
        default="paper",
        help="generate paper evidence, training convergence, or both suites",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Render selected suites with clearly labelled synthetic layout-QA inputs",
    )
    parser.add_argument("--predictions", help="evaluate.py *_predictions.jsonl")
    parser.add_argument("--events", help="evaluate.py *_events.json")
    parser.add_argument("--metrics", help="evaluate.py test_metrics.json")
    parser.add_argument(
        "--sample-id", help="Predeclared fixed-evaluation sample for the trace figure"
    )
    parser.add_argument("--frame-threshold", type=float)
    parser.add_argument(
        "--training-run",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help=(
            "training-convergence experiment root or seed run; repeat for "
            "LSTR, GateHUB, TeSTra, and EPT-Net (ours)"
        ),
    )
    parser.add_argument(
        "--main-aggregate",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="aggregate.py output; repeat for EPT-Net, LSTR, GateHUB, and TeSTra",
    )
    parser.add_argument(
        "--modality-aggregate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="aggregate.py output; repeat for Full and Video; seeds may differ",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    reports: list[dict] = []
    if args.suite in {"training", "all"}:
        training = _load_generate(
            root
            / "fig01_training_convergence"
            / "plot_fig01_training_convergence.py"
        )
        reports.append(training(run_specs=args.training_run, demo=args.demo))
    if args.suite in {"paper", "all"}:
        main_comparison = _load_generate(
            root / "fig03_main_comparison" / "plot_fig03_main_comparison.py"
        )
        dynamic = _load_generate(
            root / "fig04_dynamic_tracking" / "plot_fig04_dynamic_tracking.py"
        )
        modality = _load_generate(
            root / "fig05_modality_evidence" / "plot_fig05_modality_evidence.py"
        )
        reports.append(main_comparison(aggregate_specs=args.main_aggregate, demo=args.demo))
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
        reports.append(modality(aggregate_specs=args.modality_aggregate, demo=args.demo))
    print(json.dumps({"status": "PASS", "figures": reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
