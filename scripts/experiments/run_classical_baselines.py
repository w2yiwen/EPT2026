"""Run leakage-safe majority and per-step logistic baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eptnet.config import load_config
from eptnet.data import StitchedManifestDataset


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _binary_metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probability >= threshold
    positive = target.astype(bool)
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    positive_f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    negative_f1 = _safe_divide(2 * tn, 2 * tn + fp + fn)
    tpr = _safe_divide(tp, tp + fn)
    tnr = _safe_divide(tn, tn + fp)

    order = np.argsort(-probability, kind="stable")
    ranked_target = positive[order]
    positives = int(positive.sum())
    negatives = int((~positive).sum())
    if positives and negatives:
        cumulative_tp = np.cumsum(ranked_target)
        cumulative_fp = np.cumsum(~ranked_target)
        tpr_curve = np.concatenate(([0.0], cumulative_tp / positives, [1.0]))
        fpr_curve = np.concatenate(([0.0], cumulative_fp / negatives, [1.0]))
        auroc = float(np.trapz(tpr_curve, fpr_curve))
        precision = cumulative_tp / np.arange(1, len(target) + 1)
        average_precision = float(precision[ranked_target].sum() / positives)
    else:
        auroc = 0.0
        average_precision = 0.0
    return {
        "num_steps": int(len(target)),
        "positive_steps": positives,
        "predicted_positive_steps": int(predicted.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "macro_f1": (positive_f1 + negative_f1) / 2,
        "balanced_accuracy": (tpr + tnr) / 2,
        "auroc": auroc,
        "average_precision": average_precision,
        "threshold": float(threshold),
    }


def _features(sample: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    model = config["model"]
    pieces: list[torch.Tensor] = []
    physiology = (
        ("eeg_time", 0, "use_eeg_time"),
        ("eeg_spectral", 1, "use_eeg_spec"),
        ("physiology", 2, "use_hr"),
    )
    physiology_mask = sample["physiology_mask"].bool()
    for name, index, switch in physiology:
        if bool(model[switch]):
            pieces.append(sample[name].float() * physiology_mask[:, index : index + 1])
            pieces.append(physiology_mask[:, index : index + 1].float())
    if bool(model["use_behavior_context"]):
        modality_mask = sample["modality_mask"].bool()
        for index, (name, switch) in enumerate(
            (("video", "use_video"), ("audio", "use_audio"), ("text", "use_text"))
        ):
            if bool(model[switch]):
                pieces.append(sample[name].float() * modality_mask[:, index : index + 1])
                pieces.append(modality_mask[:, index : index + 1].float())
    if not pieces:
        raise ValueError("No enabled features for the logistic baseline")
    return torch.cat(pieces, dim=1).numpy().astype(np.float32, copy=False)


def _load_split(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    dataset = StitchedManifestDataset(config["data"][f"{split}_manifest"])
    return [dataset[index] for index in range(len(dataset))]


def _flatten(
    sessions: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    session_ids: list[str] = []
    positive_class = int(config["evaluation"]["positive_class"])
    for sample in sessions:
        mask = sample["target_mask"].bool().numpy()
        features.append(_features(sample, config)[mask])
        targets.append((sample["labels"].numpy()[mask] == positive_class).astype(np.float32))
        session_ids.extend([str(sample["metadata"]["session_id"])] * int(mask.sum()))
    return np.concatenate(features), np.concatenate(targets), session_ids


def _fit_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    seed: int,
    learning_rate: float,
    epochs: int,
    weight_decay: float,
    device: torch.device,
) -> tuple[torch.nn.Linear, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    x = torch.from_numpy((train_x - mean) / scale).to(device)
    y = torch.from_numpy(train_y).to(device)
    model = torch.nn.Linear(x.shape[1], 1).to(device)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
    return model, mean, scale


@torch.no_grad()
def _predict(
    model: torch.nn.Linear,
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    standardized = torch.from_numpy((values - mean) / scale).to(device)
    return torch.sigmoid(model(standardized).squeeze(1)).cpu().numpy()


def _select_threshold(target: np.ndarray, probability: np.ndarray, config: dict[str, Any]) -> float:
    evaluation = config["evaluation"]
    thresholds = np.linspace(
        float(evaluation["calibration_min_threshold"]),
        float(evaluation["calibration_max_threshold"]),
        int(evaluation["calibration_steps"]),
    )
    candidates = [
        (_binary_metrics(target, probability, float(value))["macro_f1"], float(value))
        for value in thresholds
    ]
    return max(candidates, key=lambda item: (item[0], -abs(item[1] - 0.5), -item[1]))[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    splits = {name: _load_split(config, name) for name in ("train", "val", "test")}
    flattened = {name: _flatten(items, config) for name, items in splits.items()}
    train_x, train_y, _ = flattened["train"]
    val_x, val_y, _ = flattened["val"]
    test_x, test_y, test_sessions = flattened["test"]

    majority_positive = float(train_y.mean()) >= 0.5
    majority_probability = np.full_like(
        test_y, 1.0 if majority_positive else 0.0, dtype=np.float32
    )
    majority = _binary_metrics(test_y, majority_probability, 0.5)
    majority["training_majority_label"] = "deception" if majority_positive else "truth"

    logistic, mean, scale = _fit_logistic(
        train_x,
        train_y,
        seed=seed,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        device=device,
    )
    val_probability = _predict(logistic, val_x, mean, scale, device)
    threshold = _select_threshold(val_y, val_probability, config)
    test_probability = _predict(logistic, test_x, mean, scale, device)
    logistic_metrics = _binary_metrics(test_y, test_probability, threshold)
    per_session: dict[str, Any] = {}
    test_sessions_array = np.asarray(test_sessions)
    for session_id in sorted(set(test_sessions)):
        selected = test_sessions_array == session_id
        per_session[session_id] = _binary_metrics(
            test_y[selected], test_probability[selected], threshold
        )
    logistic_metrics["per_session"] = per_session
    logistic_metrics["validation"] = _binary_metrics(val_y, val_probability, threshold)

    payload = {
        "schema_version": 1,
        "protocol": {
            "subject_disjoint": True,
            "fit_split": "train",
            "threshold_selection_split": "val",
            "evaluation_split": "test",
            "positive_class": int(config["evaluation"]["positive_class"]),
            "normalization": "per-feature train-only mean/std",
            "logistic_optimizer": "full-batch AdamW",
            "logistic_epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": str(device),
        },
        "feature_dimension": int(train_x.shape[1]),
        "split_target_steps": {
            name: int(values[1].shape[0]) for name, values in flattened.items()
        },
        "majority": majority,
        "logistic_regression": logistic_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(args.output), "results": {
        "majority_macro_f1": majority["macro_f1"],
        "logistic_macro_f1": logistic_metrics["macro_f1"],
        "logistic_auroc": logistic_metrics["auroc"],
    }}))


if __name__ == "__main__":
    main()
