from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_SECTION_KEYS = {
    "experiment": frozenset({"name", "seed", "output_dir"}),
    "data": frozenset(
        {
            "dataset_name",
            "input_mode",
            "train_manifest",
            "val_manifest",
            "test_manifest",
            "eeg_time_dim",
            "eeg_spectral_dim",
            "physiology_dim",
            "eeg_channels",
            "eeg_samples_per_step",
            "hr_channels",
            "hr_samples_per_step",
            "video_dim",
            "audio_dim",
            "text_dim",
            "num_classes",
            "expected_session_ids",
            "excluded_session_ids",
            "require_aligned_behavior_modalities",
            "cohort_policy",
        }
    ),
    "model": frozenset(
        {
            "name",
            "hidden_dim",
            "num_heads",
            "dropout",
            "cache_lengths",
            "spectral_n_ffts",
            "num_glimpses",
            "min_read_width",
            "max_read_width",
            "fixed_read_width",
            "adaptive_reader",
            "shared_read_policy",
            "persistent_state",
            "state_update",
            "write_gate_bias",
            "retention_gate_bias",
            "latest_state_bypass",
            "use_behavior_context",
            "use_video",
            "use_audio",
            "use_text",
            "use_eeg_time",
            "use_eeg_spec",
            "use_hr",
        }
    ),
    "loss": frozenset(
        {
            "cls_weight",
            "boundary_weight",
            "offset_weight",
            "smooth_weight",
            "label_smoothing",
            "class_weights",
            "boundary_pos_weight",
        }
    ),
    "training": frozenset(
        {
            "sequence_protocol",
            "window_size",
            "window_stride",
            "window_warmup_steps",
            "positive_window_oversample",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "grad_clip_norm",
            "patience",
            "num_workers",
            "device",
            "amp",
            "deterministic",
            "deterministic_warn_only",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "allow_tf32",
            "scheduler",
        }
    ),
    "evaluation": frozenset(
        {
            "positive_class",
            "frame_threshold",
            "boundary_threshold",
            "boundary_tolerance_steps",
            "event_iou_thresholds",
            "early_detection_delays",
            "calibrate_frame_threshold",
            "calibration_min_threshold",
            "calibration_max_threshold",
            "calibration_steps",
        }
    ),
}
_TOP_LEVEL_KEYS = frozenset(_SECTION_KEYS)
_OPTIONAL_SECTION_KEYS = {
    "data": frozenset(
        {
            "expected_session_ids",
            "excluded_session_ids",
            "require_aligned_behavior_modalities",
            "cohort_policy",
        }
    ),
    "training": frozenset(
        {
            "window_size",
            "window_stride",
            "window_warmup_steps",
            "positive_window_oversample",
        }
    ),
}
_CACHE_KEYS = frozenset({"time", "spec", "hr"})
_SCHEDULER_KEYS = frozenset(
    {
        "name",
        "mode",
        "factor",
        "patience",
        "threshold",
        "threshold_mode",
        "cooldown",
        "min_lr",
        "eps",
    }
)


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _validate_keys(
    mapping: Mapping[str, Any],
    allowed: frozenset[str],
    location: str,
    *,
    optional: frozenset[str] = frozenset(),
) -> None:
    unknown = set(mapping).difference(allowed)
    if unknown:
        rendered = ", ".join(sorted(repr(key) for key in unknown))
        raise ValueError(f"Unknown configuration key(s) in {location}: {rendered}")
    missing = allowed.difference(optional).difference(mapping)
    if missing:
        rendered = ", ".join(sorted(missing))
        raise ValueError(f"Missing required configuration key(s) in {location}: {rendered}")


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, location: str) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{key} must be a non-empty string")
    return value


def _require_bool(mapping: Mapping[str, Any], key: str, location: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise ValueError(f"{location}.{key} must be a boolean")
    return value


def _require_int(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{location}.{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{location}.{key} must be <= {maximum}")
    return value


def _require_real(
    mapping: Mapping[str, Any],
    key: str,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{location}.{key} must be a finite number")
    if minimum is not None:
        below = number < minimum if minimum_inclusive else number <= minimum
        if below:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{location}.{key} must be {operator} {minimum}")
    if maximum is not None:
        above = number > maximum if maximum_inclusive else number >= maximum
        if above:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{location}.{key} must be {operator} {maximum}")
    return number


def _validate_weight_specification(
    value: Any, *, location: str, expected_length: int, allow_scalar: bool = False
) -> None:
    if value is None or value == "auto":
        return
    if allow_scalar and not isinstance(value, bool) and isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{location} must be positive and finite")
        return
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{location} must be 'auto' or a list of length {expected_length}")
    for index, item in enumerate(value):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0
        ):
            raise ValueError(f"{location}[{index}] must be positive and finite")


def _validate_config(config: Mapping[str, Any]) -> None:
    _validate_keys(config, _TOP_LEVEL_KEYS, "top level")
    sections: dict[str, Mapping[str, Any]] = {}
    for name, allowed in _SECTION_KEYS.items():
        section = _require_mapping(config[name], name)
        _validate_keys(
            section,
            allowed,
            name,
            optional=_OPTIONAL_SECTION_KEYS.get(name, frozenset()),
        )
        sections[name] = section

    experiment = sections["experiment"]
    _require_string(experiment, "name", "experiment")
    _require_string(experiment, "output_dir", "experiment")
    _require_int(experiment, "seed", "experiment", minimum=0)

    data = sections["data"]
    for key in ("dataset_name", "train_manifest", "val_manifest", "test_manifest"):
        _require_string(data, key, "data")
    input_mode = _require_string(data, "input_mode", "data")
    if input_mode not in {"raw_windows", "precomputed_features"}:
        raise ValueError("data.input_mode must be 'raw_windows' or 'precomputed_features'")
    for key in (
        "eeg_time_dim",
        "eeg_spectral_dim",
        "physiology_dim",
        "eeg_channels",
        "eeg_samples_per_step",
        "hr_channels",
        "hr_samples_per_step",
        "video_dim",
        "audio_dim",
        "text_dim",
    ):
        _require_int(data, key, "data", minimum=1)
    num_classes = _require_int(data, "num_classes", "data", minimum=2)
    for key in ("expected_session_ids", "excluded_session_ids"):
        if key in data:
            values = data[key]
            if (
                not isinstance(values, list)
                or not all(isinstance(value, str) and value.strip() for value in values)
                or len(values) != len(set(values))
            ):
                raise ValueError(f"data.{key} must be a unique list of non-empty strings")
    if "require_aligned_behavior_modalities" in data:
        _require_bool(data, "require_aligned_behavior_modalities", "data")
    if "cohort_policy" in data:
        cohort_policy = _require_string(data, "cohort_policy", "data")
        if cohort_policy not in {"subject_disjoint", "all_sessions_training"}:
            raise ValueError(
                "data.cohort_policy must be 'subject_disjoint' or "
                "'all_sessions_training'"
            )
    expected_sessions = set(data.get("expected_session_ids", []))
    excluded_sessions = set(data.get("excluded_session_ids", []))
    overlap = expected_sessions.intersection(excluded_sessions)
    if overlap:
        raise ValueError(
            "data.expected_session_ids and data.excluded_session_ids overlap: "
            f"{sorted(overlap)}"
        )

    model = sections["model"]
    model_name = _require_string(model, "name", "model")
    if model_name not in {"eptnet", "early_fusion_gru", "fusion_transformer"}:
        raise ValueError(f"Unsupported model.name: {model_name!r}")
    hidden_dim = _require_int(model, "hidden_dim", "model", minimum=1)
    num_heads = _require_int(model, "num_heads", "model", minimum=1)
    if hidden_dim % num_heads != 0:
        raise ValueError("model.hidden_dim must be divisible by model.num_heads")
    _require_real(
        model,
        "dropout",
        "model",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    cache_lengths = _require_mapping(model["cache_lengths"], "model.cache_lengths")
    _validate_keys(cache_lengths, _CACHE_KEYS, "model.cache_lengths")
    for key in _CACHE_KEYS:
        _require_int(cache_lengths, key, "model.cache_lengths", minimum=1)
    spectral_n_ffts = model["spectral_n_ffts"]
    if not isinstance(spectral_n_ffts, list) or not spectral_n_ffts:
        raise ValueError("model.spectral_n_ffts must be a non-empty list")
    for index, value in enumerate(spectral_n_ffts):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"model.spectral_n_ffts[{index}] must be an integer >= 2")
    _require_int(model, "num_glimpses", "model", minimum=1)
    min_width = _require_real(
        model,
        "min_read_width",
        "model",
        minimum=0.0,
        minimum_inclusive=False,
        maximum=1.0,
    )
    max_width = _require_real(
        model,
        "max_read_width",
        "model",
        minimum=0.0,
        minimum_inclusive=False,
        maximum=1.0,
    )
    fixed_width = _require_real(
        model,
        "fixed_read_width",
        "model",
        minimum=0.0,
        minimum_inclusive=False,
        maximum=1.0,
    )
    if min_width > max_width:
        raise ValueError("model.min_read_width must be <= model.max_read_width")
    if not min_width <= fixed_width <= max_width:
        raise ValueError(
            "model.fixed_read_width must lie between min_read_width and max_read_width"
        )
    for key in (
        "adaptive_reader",
        "shared_read_policy",
        "persistent_state",
        "latest_state_bypass",
        "use_behavior_context",
        "use_video",
        "use_audio",
        "use_text",
        "use_eeg_time",
        "use_eeg_spec",
        "use_hr",
    ):
        _require_bool(model, key, "model")
    state_update = _require_string(model, "state_update", "model").lower()
    if state_update not in {"vanilla", "gated"}:
        raise ValueError("model.state_update must be 'vanilla' or 'gated'")
    _require_real(model, "write_gate_bias", "model")
    _require_real(model, "retention_gate_bias", "model")

    loss = sections["loss"]
    for key in ("cls_weight", "boundary_weight", "offset_weight", "smooth_weight"):
        _require_real(loss, key, "loss", minimum=0.0)
    _require_real(
        loss,
        "label_smoothing",
        "loss",
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    _validate_weight_specification(
        loss["class_weights"],
        location="loss.class_weights",
        expected_length=num_classes,
    )
    _validate_weight_specification(
        loss["boundary_pos_weight"],
        location="loss.boundary_pos_weight",
        expected_length=2,
        allow_scalar=True,
    )

    training = sections["training"]
    sequence_protocol = _require_string(training, "sequence_protocol", "training")
    if sequence_protocol not in {"continuous_session", "causal_windows"}:
        raise ValueError(
            "training.sequence_protocol must be 'continuous_session' or 'causal_windows'"
        )
    _require_int(training, "epochs", "training", minimum=1)
    batch_size = _require_int(training, "batch_size", "training", minimum=1)
    if batch_size != 1:
        raise ValueError("training.batch_size must be 1 for the supported sequence protocols")
    if sequence_protocol == "causal_windows":
        window_size = _require_int(training, "window_size", "training", minimum=2)
        window_stride = _require_int(training, "window_stride", "training", minimum=1)
        window_warmup = _require_int(
            training, "window_warmup_steps", "training", minimum=0
        )
        _require_int(training, "positive_window_oversample", "training", minimum=1)
        if window_stride > window_size:
            raise ValueError("training.window_stride must not exceed training.window_size")
        if window_warmup >= window_size:
            raise ValueError(
                "training.window_warmup_steps must be smaller than training.window_size"
            )
    learning_rate = _require_real(
        training,
        "learning_rate",
        "training",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _require_real(training, "weight_decay", "training", minimum=0.0)
    _require_real(
        training,
        "grad_clip_norm",
        "training",
        minimum=0.0,
        minimum_inclusive=False,
    )
    _require_int(training, "patience", "training", minimum=1)
    _require_int(training, "num_workers", "training", minimum=0)
    _require_string(training, "device", "training")
    for key in (
        "amp",
        "deterministic",
        "deterministic_warn_only",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "allow_tf32",
    ):
        _require_bool(training, key, "training")
    scheduler = _require_mapping(training["scheduler"], "training.scheduler")
    _validate_keys(scheduler, _SCHEDULER_KEYS, "training.scheduler")
    scheduler_name = _require_string(scheduler, "name", "training.scheduler").lower()
    if scheduler_name not in {"reduce_on_plateau", "none", "off", "disabled"}:
        raise ValueError(f"Unsupported training.scheduler.name: {scheduler_name!r}")
    scheduler_mode = _require_string(scheduler, "mode", "training.scheduler").lower()
    if scheduler_mode != "min":
        raise ValueError("training.scheduler.mode must be 'min'")
    _require_real(
        scheduler,
        "factor",
        "training.scheduler",
        minimum=0.0,
        minimum_inclusive=False,
        maximum=1.0,
        maximum_inclusive=False,
    )
    _require_int(scheduler, "patience", "training.scheduler", minimum=0)
    _require_real(scheduler, "threshold", "training.scheduler", minimum=0.0)
    threshold_mode = _require_string(scheduler, "threshold_mode", "training.scheduler").lower()
    if threshold_mode not in {"rel", "abs"}:
        raise ValueError("training.scheduler.threshold_mode must be 'rel' or 'abs'")
    _require_int(scheduler, "cooldown", "training.scheduler", minimum=0)
    min_lr = _require_real(scheduler, "min_lr", "training.scheduler", minimum=0.0)
    if min_lr > learning_rate:
        raise ValueError("training.scheduler.min_lr must not exceed training.learning_rate")
    _require_real(scheduler, "eps", "training.scheduler", minimum=0.0)

    evaluation = sections["evaluation"]
    positive_class = _require_int(evaluation, "positive_class", "evaluation", minimum=0)
    if positive_class >= num_classes:
        raise ValueError("evaluation.positive_class must be smaller than data.num_classes")
    for key in ("frame_threshold", "boundary_threshold"):
        _require_real(evaluation, key, "evaluation", minimum=0.0, maximum=1.0)
    _require_int(evaluation, "boundary_tolerance_steps", "evaluation", minimum=0)
    iou_thresholds = evaluation["event_iou_thresholds"]
    if not isinstance(iou_thresholds, list) or not iou_thresholds:
        raise ValueError("evaluation.event_iou_thresholds must be a non-empty list")
    for index, value in enumerate(iou_thresholds):
        _require_real(
            {"value": value},
            "value",
            f"evaluation.event_iou_thresholds[{index}]",
            minimum=0.0,
            minimum_inclusive=False,
            maximum=1.0,
        )
    delays = evaluation["early_detection_delays"]
    if not isinstance(delays, list) or not delays:
        raise ValueError("evaluation.early_detection_delays must be a non-empty list")
    for index, value in enumerate(delays):
        _require_int(
            {"value": value},
            "value",
            f"evaluation.early_detection_delays[{index}]",
            minimum=0,
        )
    _require_bool(evaluation, "calibrate_frame_threshold", "evaluation")
    calibration_min = _require_real(
        evaluation,
        "calibration_min_threshold",
        "evaluation",
        minimum=0.0,
        maximum=1.0,
    )
    calibration_max = _require_real(
        evaluation,
        "calibration_max_threshold",
        "evaluation",
        minimum=0.0,
        maximum=1.0,
    )
    if calibration_min >= calibration_max:
        raise ValueError(
            "evaluation.calibration_min_threshold must be smaller than "
            "evaluation.calibration_max_threshold"
        )
    _require_int(evaluation, "calibration_steps", "evaluation", minimum=2)


def _load_config_file(path: Path, active: tuple[Path, ...]) -> dict[str, Any]:
    canonical = path.resolve()
    if canonical in active:
        chain = " -> ".join(str(item) for item in (*active, canonical))
        raise ValueError(f"Cyclic config inheritance detected: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Configuration at {path} must be a YAML mapping")
    config = dict(loaded)
    inherited = config.pop("inherits", None)
    if inherited is None:
        return config
    if not isinstance(inherited, str) or not inherited.strip():
        raise ValueError(f"inherits in {path} must be a non-empty path string")
    parent = path.parent / inherited
    return _deep_update(_load_config_file(parent, (*active, canonical)), config)


def load_config(path: str, base_path: str | None = None) -> dict[str, Any]:
    """Load, merge, and strictly validate a YAML experiment configuration."""
    config = _load_config_file(Path(path), ())
    if base_path is not None:
        config = _deep_update(_load_config_file(Path(base_path), ()), config)
    _validate_config(config)
    return config


def save_resolved_config(config: dict[str, Any], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
