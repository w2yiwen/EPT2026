from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_training_history(path: str | Path) -> list[dict[str, Any]]:
    history_path = Path(path)
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Training history must be a non-empty list: {history_path}")
    return payload


def generate_training_artifacts(run_directory: str | Path) -> dict[str, Any]:
    """Generate a single PNG training curve from history.json."""

    run_dir = Path(run_directory)
    history_path = run_dir / "history.json"
    history = load_training_history(history_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(record["epoch"]) for record in history]
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    axis.plot(epochs, [float(record["train"]["total"]) for record in history], label="Train")
    axis.plot(epochs, [float(record["val"]["total"]) for record in history], label="Validation")
    axis.set(xlabel="Epoch", ylabel="Total loss")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)

    output = run_dir / "figures" / "training_curve.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {"training_curve": output.relative_to(run_dir).as_posix()}
