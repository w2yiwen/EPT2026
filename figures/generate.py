#!/usr/bin/env python3
"""Generate the public PNG figures from completed experiment results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "final"
OUTPUT = ROOT / "figures"
MODELS = (
    ("EPT-Net", "eptnet"),
    ("LSTR", "lstr"),
    ("GateHUB", "gatehub"),
    ("TeSTra", "testra"),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT / name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def training_curve() -> None:
    history = read_json(RESULTS / "eptnet" / "seed_42" / "history.json")
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    axis.plot(epochs, [row["train"]["total"] for row in history], label="Train")
    axis.plot(epochs, [row["val"]["total"] for row in history], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Total loss")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "training_curve.png")


def model_comparison() -> None:
    labels, frame_f1, event_f1 = [], [], []
    for label, directory in MODELS:
        aggregate = read_json(RESULTS / directory / "aggregate.json")["aggregate"]
        labels.append(label)
        frame_f1.append(aggregate["frame.macro_f1"]["mean"])
        event_f1.append(aggregate["event.event_f1_iou_0.5"]["mean"])

    positions = range(len(labels))
    figure, axis = plt.subplots(figsize=(6.2, 3.5))
    width = 0.36
    axis.bar([x - width / 2 for x in positions], frame_f1, width, label="Frame macro-F1")
    axis.bar([x + width / 2 for x in positions], event_f1, width, label="Event F1@0.5")
    axis.set_xticks(list(positions), labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "model_comparison.png")


def event_timeline() -> None:
    path = RESULTS / "eptnet" / "seed_42" / "test_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    sample = rows[0]["sample_id"]
    rows = [row for row in rows if row["sample_id"] == sample]
    time = [row["start_time_seconds"] for row in rows]
    score = [row["positive_probability"] for row in rows]
    target = [int(row["label"] == row["positive_class"]) for row in rows]

    figure, axis = plt.subplots(figsize=(7.0, 3.2))
    axis.plot(time, score, label="Prediction", linewidth=1.4)
    axis.step(time, target, where="post", label="Target", linewidth=1.0)
    axis.set(xlabel="Time (s)", ylabel="Event probability", ylim=(-0.03, 1.03))
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    save(figure, "event_timeline.png")


def main() -> None:
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.2})
    training_curve()
    model_comparison()
    event_timeline()


if __name__ == "__main__":
    main()
