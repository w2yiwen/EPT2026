"""Command-line interface for training, evaluation, and aggregation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="ept", description="EPT-Net research code")
    subcommands = command.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "aggregate"):
        child = subcommands.add_parser(name)
        child.add_argument("args", nargs=argparse.REMAINDER)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    modules = {
        "train": "eptnet.training.trainer",
        "evaluate": "eptnet.evaluation.evaluator",
        "aggregate": "eptnet.evaluation.aggregate",
    }
    process = subprocess.run(
        [sys.executable, "-m", modules[args.command], *args.args],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
    )
    return int(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
