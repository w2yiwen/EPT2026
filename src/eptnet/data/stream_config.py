"""Configuration contract for the manuscript's native-time implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping


def validate_stream_config(config: Mapping) -> None:
    for name in ("experiment", "data", "model", "loss", "training", "evaluation", "encoders"):
        if not isinstance(config.get(name), Mapping):
            raise ValueError(f"Missing {name} configuration")
    data, model, training = config["data"], config["model"], config["training"]
    if model.get("name") != "eptnet":
        raise ValueError("native_streams uses model.name=eptnet")
    if data.get("cohort_policy") != "subject_disjoint":
        raise ValueError("The native-stream manuscript protocol uses participant-disjoint splits")
    if float(data.get("decision_seconds", 0.5)) != 0.5:
        raise ValueError("The manuscript decision grid is 0.5 seconds")
    if int(model.get("hidden_dim", 64)) != 64 or int(model.get("num_heads", 4)) != 4:
        raise ValueError("The manuscript uses D=64 and four attention heads")
    for name in ("time", "spec", "hr"):
        horizon = float(model.get("retention_seconds", {}).get(name, 0))
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError(f"model.retention_seconds.{name} must be a positive horizon in seconds")
    if training.get("sequence_protocol") != "chronological_chunks":
        raise ValueError("native_streams training carries state through chronological chunks")
    if int(training.get("batch_size", 1)) != 1 or int(training.get("chunk_steps", 0)) <= 0:
        raise ValueError("Use one recording at a time and a positive chunk_steps")
    if int(data.get("num_classes", 2)) != 2 or int(config["evaluation"].get("positive_class", 0)) != 0:
        raise ValueError("Keep the repository's class encoding: 0=deception, 1=truth")
