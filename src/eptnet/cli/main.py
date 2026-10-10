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
    for name in ("train", "evaluate", "aggregate", "prepare", "normalize", "infer"):
        child = subcommands.add_parser(name)
        child.add_argument("args", nargs=argparse.REMAINDER)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(arguments[:1])
    forwarded = arguments[1:]
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    modules = {
        "train": "eptnet.training.trainer",
        "evaluate": "eptnet.evaluation.evaluator",
        "aggregate": "eptnet.evaluation.aggregate",
    }
    if args.command in ("prepare", "normalize", "infer"):
        modules[args.command] = "eptnet.cli.streaming"
        forwarded = [args.command, *forwarded]
    process = subprocess.run(
        [sys.executable, "-m", modules[args.command], *forwarded],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
    )
    return int(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
