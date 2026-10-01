"""Single, discoverable CLI for training, evaluation, and environment checks."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML experiment configuration")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ept",
        description="Reproducible EPT-Net training and evaluation workflows.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check the runtime and optionally a config")
    doctor.add_argument("--config", help="Validate one YAML configuration")
    doctor.add_argument("--data-root", type=Path, help="Check that a data root exists")

    train = subparsers.add_parser("train", help="Run the canonical trainer")
    train.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the trainer")

    evaluate = subparsers.add_parser("evaluate", help="Run the canonical evaluator")
    evaluate.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to evaluation")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate evaluation artifacts")
    aggregate.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to aggregation")

    resolve = subparsers.add_parser("resolve-config", help="Validate and materialize a config")
    _add_config_argument(resolve)
    resolve.add_argument("--output", type=Path, required=True, help="Resolved YAML output path")

    figures = subparsers.add_parser("figures", help="Run the paper figure generator")
    figures.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to fig/generate_all.py")
    return parser


def _run_module(module: str, args: Sequence[str]) -> int:
    command = [sys.executable, "-m", module, *args]
    completed = subprocess.run(command, cwd=_project_root(), check=False)
    return int(completed.returncode)


def _doctor(config_path: str | None, data_root: Path | None) -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "project_root": str(_project_root()),
        "python_executable": str(Path(sys.executable).resolve()),
    }
    try:
        import numpy as np
        import torch

        report["numpy"] = np.__version__
        report["torch"] = torch.__version__
        report["default_device"] = str(torch.get_default_device())
    except ImportError as exc:
        report["dependency_error"] = str(exc)
    if config_path:
        from eptnet.config import load_config

        config = load_config(config_path)
        report["config"] = {
            "path": str(Path(config_path).resolve()),
            "experiment": config["experiment"]["name"],
            "cohort_policy": config["data"].get("cohort_policy", "subject_disjoint"),
            "train_manifest": config["data"]["train_manifest"],
        }
    if data_root:
        report["data_root"] = {"path": str(data_root.resolve()), "exists": data_root.is_dir()}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "doctor":
        return _doctor(args.config, args.data_root)
    if args.command == "resolve-config":
        from eptnet.config import load_config, save_resolved_config

        save_resolved_config(load_config(args.config), str(args.output))
        print(args.output)
        return 0
    if args.command == "train":
        return _run_module("eptnet.training.trainer", args.args)
    if args.command == "evaluate":
        return _run_module("eptnet.evaluation.evaluator", args.args)
    if args.command == "aggregate":
        return _run_module("eptnet.evaluation.aggregate", args.args)
    if args.command == "figures":
        command = [sys.executable, "fig/generate_all.py", *args.args]
        return int(subprocess.run(command, cwd=_project_root(), check=False).returncode)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
