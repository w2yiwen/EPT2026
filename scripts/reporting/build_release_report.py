#!/usr/bin/env python3
"""Build the deterministic, fail-closed EPT-Net primary release report.

The public machine-readable release contains only the 18-subject,
subject-disjoint 12/2/4 pilot matrix.  The script writes only complete
three-seed summaries, validates the independently generated aggregates, and
returns exit status 2 whenever any required artifact is missing or invalid.

Run from ``code/`` after the complete GPU matrix has finished::

    python scripts/reporting/build_release_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

EXPECTED_SEEDS = (13, 42, 73)
HASH_LENGTH = 64
PRIMARY_COHORT = "bci_subjects_fixed_split"

PRIMARY_COUNTS = {
    "num_sequences": 4,
    "num_unique_steps": 1505,
    "num_evaluated_target_steps": 1201,
    "num_target_events": 55,
}
LATENCY_DEFINITION = (
    "non-negative time-to-detection; steps=max(raw emit_step minus target "
    "start step, 0); seconds=max(word-end decision time at raw emit_step "
    "minus target event onset time, 0), including overlapping word "
    "intervals; decoded events retain the raw emit_step"
)
TARGET_MASK_DEFINITION = (
    "losses and metrics include only target-speaker intervals; the model "
    "observes the complete interaction timeline; invalid intervals close "
    "and cannot open decoded events"
)
EVENT_F1_PROTOCOL = {
    "proposal_source": "thresholded causal event decoder",
    "uses_frame_threshold": True,
    "used_for": "event precision, recall, F1, early detection, and latency",
}
EVENT_AP_PROTOCOL = {
    "proposal_source": "one dense proposal per target-valid step",
    "score": "positive-class probability at the proposal step",
    "interval": (
        "[step - predicted left offset, step + predicted right offset], clamped to "
        "the proposal step's contiguous target-valid segment"
    ),
    "score_threshold": None,
    "nms": "none",
    "ranking": ("global descending score; deterministic sequence and chronological-step tie-break"),
}
SUBJECT_UNCERTAINTY = "fixed-seed 10000-resample percentile bootstrap across subjects"
FAIRNESS_GRAPH_COUNTING_PROTOCOL = (
    "trainable parameter elements whose parameter tensor has a non-None gradient after "
    "deterministic accumulated full-multitask backward passes over the shared minimal "
    "target-valid modality-cover sample set; graph participation does not imply a "
    "non-zero update"
)
FAIRNESS_NONZERO_PARAMETER_COUNTING_PROTOCOL = (
    "trainable parameter elements belonging to a parameter tensor with at least one "
    "finite exactly non-zero accumulated gradient element on the shared audit sample set"
)
FAIRNESS_NONZERO_ELEMENT_COUNTING_PROTOCOL = (
    "individual trainable parameter elements with a finite exactly non-zero accumulated "
    "gradient on the shared audit sample set"
)
FAIRNESS_SELECTION_PROTOCOL = (
    "minimum-cardinality set of training-manifest windows covering every model-enabled "
    "modality that is observed during at least one target-valid step; ties are resolved "
    "by the lexicographically smallest ordered sample-id tuple"
)


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    label: str
    path: tuple[str, ...]
    aggregation_unit: str
    bounded: bool = True
    conditional: bool = False
    allow_none: bool = False

    def aggregate_key(self, variant: str) -> str:
        prefix = () if variant == "selected" else ("fixed_threshold_0_5_sensitivity",)
        return ".".join((*prefix, *self.path))


@dataclass(frozen=True)
class ExperimentSpec:
    cohort: str
    directory: str
    display_name: str
    role: str
    fairness_required: bool = False


SUBJECT_METRICS = (
    MetricSpec(
        "subject_frame_macro_f1",
        "Participant-macro frame Macro-F1",
        ("subject_macro", "metrics", "frame_macro_f1", "mean"),
        "participant_macro",
        allow_none=True,
    ),
    MetricSpec(
        "subject_frame_balanced_accuracy",
        "Participant-macro frame Balanced Accuracy",
        ("subject_macro", "metrics", "frame_balanced_accuracy", "mean"),
        "participant_macro",
        allow_none=True,
    ),
    MetricSpec(
        "subject_frame_auroc",
        "Participant-macro frame AUROC",
        ("subject_macro", "metrics", "frame_auroc", "mean"),
        "participant_macro",
        allow_none=True,
    ),
    MetricSpec(
        "subject_frame_average_precision",
        "Participant-macro frame AP",
        ("subject_macro", "metrics", "frame_average_precision", "mean"),
        "participant_macro",
        allow_none=True,
    ),
    MetricSpec(
        "subject_boundary_macro_f1",
        "Participant-macro boundary F1",
        ("subject_macro", "metrics", "boundary_macro_f1", "mean"),
        "participant_macro",
        allow_none=True,
    ),
    MetricSpec(
        "subject_event_f1_iou_0_5",
        "Participant-macro event F1@0.5",
        ("subject_macro", "metrics", "event_f1_iou_0.5", "mean"),
        "participant_macro",
        allow_none=True,
    ),
    MetricSpec(
        "subject_event_map",
        "Participant-macro threshold-free event mAP",
        ("subject_macro", "metrics", "event_map", "mean"),
        "participant_macro",
        allow_none=True,
    ),
)

POOLED_METRICS = (
    MetricSpec(
        "pooled_frame_macro_f1",
        "Pooled frame Macro-F1",
        ("frame", "macro_f1"),
        "pooled_test_steps",
    ),
    MetricSpec(
        "pooled_frame_balanced_accuracy",
        "Pooled frame Balanced Accuracy",
        ("frame", "balanced_accuracy"),
        "pooled_test_steps",
    ),
    MetricSpec(
        "pooled_frame_auroc",
        "Pooled frame AUROC",
        ("frame", "auroc"),
        "pooled_test_steps",
    ),
    MetricSpec(
        "pooled_frame_average_precision",
        "Pooled frame AP",
        ("frame", "average_precision"),
        "pooled_test_steps",
    ),
    MetricSpec(
        "pooled_boundary_macro_f1",
        "Pooled boundary F1",
        ("boundary", "boundary_macro_f1"),
        "pooled_test_steps",
    ),
    MetricSpec(
        "pooled_event_f1_iou_0_5",
        "Pooled event F1@0.5",
        ("event", "event_f1_iou_0.5"),
        "pooled_test_events",
    ),
    MetricSpec(
        "pooled_event_map",
        "Pooled threshold-free event mAP",
        ("event", "event_map"),
        "pooled_test_events",
    ),
    MetricSpec(
        "matched_event_delay_steps",
        "Matched-event alert TTD (steps)",
        ("event", "mean_detection_delay_steps"),
        "matched_events_only",
        bounded=False,
        conditional=True,
        allow_none=True,
    ),
    MetricSpec(
        "matched_event_delay_seconds",
        "Matched-event alert TTD (seconds)",
        ("latency", "mean_detection_delay_seconds"),
        "matched_events_only",
        bounded=False,
        conditional=True,
        allow_none=True,
    ),
)

PRIMARY_METRICS = (*SUBJECT_METRICS, *POOLED_METRICS)

PRIMARY_EXPERIMENTS = (
    ExperimentSpec(
        PRIMARY_COHORT,
        "eptnet_bci_subjects_no_text",
        "EPT-Net",
        "proposed",
        True,
    ),
    ExperimentSpec(
        PRIMARY_COHORT,
        "baseline_early_fusion_gru_bci_subjects_no_text",
        "Early-fusion GRU",
        "baseline",
        True,
    ),
    ExperimentSpec(
        PRIMARY_COHORT,
        "baseline_fusion_transformer_bci_subjects_no_text",
        "Fusion Transformer",
        "baseline",
        True,
    ),
    ExperimentSpec(
        PRIMARY_COHORT,
        "ablation_no_recurrent_fusion_bci_subjects_no_text",
        "EPT-Net w/o recurrent fusion update",
        "capacity-confounded ablation",
        True,
    ),
)
PRIMARY_EXPERIMENT_DIRECTORIES = tuple(spec.directory for spec in PRIMARY_EXPERIMENTS)

PRIMARY_FAIRNESS_EXPECTED = {
    "eptnet_bci_subjects_no_text": {
        "model": "eptnet",
        "allocated_parameters": 403852,
        "trainable_parameters": 403852,
        "graph_participating_parameters": 334796,
    },
    "baseline_early_fusion_gru_bci_subjects_no_text": {
        "model": "early_fusion_gru",
        "allocated_parameters": 436362,
        "trainable_parameters": 436362,
        "graph_participating_parameters": 334682,
    },
    "baseline_fusion_transformer_bci_subjects_no_text": {
        "model": "fusion_transformer",
        "allocated_parameters": 419066,
        "trainable_parameters": 419066,
        "graph_participating_parameters": 343626,
    },
    "ablation_no_recurrent_fusion_bci_subjects_no_text": {
        "model": "eptnet",
        "allocated_parameters": 403852,
        "trainable_parameters": 403852,
        "graph_participating_parameters": 276428,
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        detail = exc.strerror or type(exc).__name__
        raise ValueError(f"cannot read {path.name}: {detail}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _relative(path: Path, results_dir: Path) -> str:
    try:
        return path.resolve().relative_to(results_dir.resolve()).as_posix()
    except ValueError:
        return path.name


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != HASH_LENGTH:
        raise ValueError(f"{label} must be a 64-character SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} is not a lowercase hexadecimal SHA-256")
    return value


def _at(mapping: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"{label} is missing {'.'.join(path)}")
        value = value[key]
    return value


def _same_number(observed: Any, expected: float, label: str) -> None:
    value = _finite(observed, label)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}={value!r}, expected {expected!r}")


def _metric_value(
    payload: Mapping[str, Any], metric: MetricSpec, variant: str, label: str
) -> float | None:
    root: Mapping[str, Any] = payload
    if variant == "fixed_0_5":
        nested = payload.get("fixed_threshold_0_5_sensitivity")
        if not isinstance(nested, Mapping):
            raise ValueError(f"{label} is missing fixed_threshold_0_5_sensitivity")
        root = nested
    raw = _at(root, metric.path, label)
    if raw is None:
        if metric.allow_none:
            return None
        raise ValueError(f"{label}:{metric.metric_id} is unexpectedly undefined")
    value = _finite(raw, f"{label}:{metric.metric_id}")
    if metric.bounded and not 0.0 <= value <= 1.0:
        raise ValueError(f"{label}:{metric.metric_id}={value} is outside [0, 1]")
    if metric.conditional and value < 0.0:
        raise ValueError(f"{label}:{metric.metric_id} is a negative latency")
    return value


def _validate_data(payload: Mapping[str, Any], expected: Mapping[str, int], label: str) -> None:
    if dict(expected) != PRIMARY_COUNTS:
        raise ValueError(f"{label}: only the primary data signature is supported")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} has no data object")
    for key, expected_value in expected.items():
        if data.get(key) != expected_value:
            raise ValueError(f"{label}:data.{key}={data.get(key)!r}, expected {expected_value}")
    predicted = data.get("num_predicted_events")
    if not _is_integer(predicted) or predicted < 0:
        raise ValueError(f"{label}:data.num_predicted_events is invalid")
    frame = payload.get("frame")
    if not isinstance(frame, Mapping):
        raise ValueError(f"{label} has no frame metrics")
    evaluated_steps = PRIMARY_COUNTS["num_evaluated_target_steps"]
    dense_count = data.get("num_dense_event_proposals")
    if dense_count != evaluated_steps:
        raise ValueError(
            f"{label}:data.num_dense_event_proposals={dense_count!r}, expected {evaluated_steps}"
        )
    if frame.get("num_frames") != evaluated_steps:
        raise ValueError(
            f"{label}:frame.num_frames={frame.get('num_frames')!r}, expected {evaluated_steps}"
        )


def _validate_protocol(
    payload: Mapping[str, Any], *, primary: bool, fixed: bool, label: str
) -> None:
    # Keep the keyword for compatibility with focused validator tests while
    # making the public release contract explicitly primary-only.
    if not primary:
        raise ValueError(f"{label}: only the primary release protocol is supported")
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"{label} has no protocol object")
    expected: dict[str, Any] = {
        "evaluation_unit": "unique chronological rows reconstructed per session",
        "positive_class": 0,
        "boundary_threshold": 0.5,
        "boundary_tolerance_steps": 1,
        "event_iou_thresholds": [0.3, 0.5, 0.7],
        "early_detection_delays_steps": [0.0, 1.0, 2.0],
        "latency_definition": LATENCY_DEFINITION,
        "target_mask_definition": TARGET_MASK_DEFINITION,
        "event_f1_protocol": EVENT_F1_PROTOCOL,
        "event_ap_protocol": EVENT_AP_PROTOCOL,
    }
    for key, expected_value in expected.items():
        if protocol.get(key) != expected_value:
            raise ValueError(
                f"{label}:protocol.{key}={protocol.get(key)!r}, expected {expected_value!r}"
            )
    threshold = _finite(protocol.get("frame_threshold"), f"{label}:frame_threshold")
    if fixed and not math.isclose(threshold, 0.5, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}: fixed-threshold sensitivity did not use 0.5")


def _validate_subject_summary(payload: Mapping[str, Any], label: str) -> None:
    subject = payload.get("subject_macro")
    if not isinstance(subject, Mapping):
        raise ValueError(f"{label} has no subject_macro object")
    if subject.get("unit") != "held-out subject/session":
        raise ValueError(f"{label}:subject_macro.unit is invalid")
    if subject.get("num_subjects") != PRIMARY_COUNTS["num_sequences"]:
        raise ValueError(f"{label}:subject_macro.num_subjects must be 4")
    if subject.get("uncertainty") != SUBJECT_UNCERTAINTY:
        raise ValueError(f"{label}:subject_macro.uncertainty is invalid")
    metrics = subject.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{label}:subject_macro.metrics is invalid")

    names = {
        "frame_macro_f1",
        "frame_balanced_accuracy",
        "frame_auroc",
        "frame_average_precision",
        "boundary_macro_f1",
        "event_f1_iou_0.5",
        "event_map",
    }
    for name in sorted(names):
        entry = metrics.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label}:subject_macro.metrics.{name} is invalid")
        n = entry.get("n")
        missing = entry.get("missing")
        if not _is_integer(n) or not _is_integer(missing):
            raise ValueError(f"{label}:{name} n/missing must be integers")
        if n < 0 or missing < 0 or n + missing != PRIMARY_COUNTS["num_sequences"]:
            raise ValueError(f"{label}:{name} n/missing does not account for four subjects")
        fields = (
            "mean",
            "sample_std",
            "bootstrap_ci95_lower",
            "bootstrap_ci95_upper",
        )
        if n == 0:
            if any(entry.get(field) is not None for field in fields):
                raise ValueError(f"{label}:{name} has values despite n=0")
            continue
        value = _finite(entry.get("mean"), f"{label}:{name}:mean")
        lower = _finite(entry.get("bootstrap_ci95_lower"), f"{label}:{name}:ci_lower")
        upper = _finite(entry.get("bootstrap_ci95_upper"), f"{label}:{name}:ci_upper")
        if not 0.0 <= value <= 1.0 or not 0.0 <= lower <= upper <= 1.0:
            raise ValueError(f"{label}:{name} has invalid bounded summary values")
        if n == 1:
            if entry.get("sample_std") is not None:
                raise ValueError(f"{label}:{name} sample_std must be null for n=1")
            if not (
                math.isclose(lower, value, abs_tol=1e-12)
                and math.isclose(upper, value, abs_tol=1e-12)
            ):
                raise ValueError(f"{label}:{name} n=1 interval must equal the observation")
        else:
            std = _finite(entry.get("sample_std"), f"{label}:{name}:sample_std")
            if std < 0.0:
                raise ValueError(f"{label}:{name} has a negative sample_std")


def _subject_details(
    payload: Mapping[str, Any], metric: MetricSpec, variant: str
) -> dict[str, Any]:
    if metric.aggregation_unit != "participant_macro":
        return {}
    root: Mapping[str, Any] = payload
    if variant == "fixed_0_5":
        root = payload["fixed_threshold_0_5_sensitivity"]
    entry = _at(root, metric.path[:-1], f"{variant}:{metric.metric_id}")
    if not isinstance(entry, Mapping):
        raise ValueError(f"{variant}:{metric.metric_id} subject summary is invalid")
    return {
        "subject_n": entry.get("n"),
        "subject_missing": entry.get("missing"),
        "subject_sample_std": entry.get("sample_std"),
        "subject_bootstrap_ci95_lower": entry.get("bootstrap_ci95_lower"),
        "subject_bootstrap_ci95_upper": entry.get("bootstrap_ci95_upper"),
    }


def _validate_calibration(payload: Mapping[str, Any], *, label: str) -> float:
    calibration = payload.get("calibration")
    protocol = payload.get("protocol")
    if not isinstance(calibration, Mapping) or not isinstance(protocol, Mapping):
        raise ValueError(f"{label} has no calibration/protocol object")
    expected = {
        "enabled": True,
        "method": "validation_subject_macro_f1_grid",
        "selection_split": "validation",
        "calibration_min_threshold": 0.1,
        "calibration_max_threshold": 0.9,
        "calibration_steps": 81,
    }
    for key, expected_value in expected.items():
        if calibration.get(key) != expected_value:
            raise ValueError(
                f"{label}:calibration.{key}={calibration.get(key)!r}, expected {expected_value!r}"
            )
    threshold = _finite(calibration.get("frame_threshold"), f"{label}:calibration threshold")
    _same_number(protocol.get("frame_threshold"), threshold, f"{label}:protocol threshold")
    grid_index = round((threshold - 0.1) / 0.01)
    if not 0 <= grid_index <= 80 or not math.isclose(
        threshold, 0.1 + grid_index * 0.01, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(f"{label}: selected threshold is not on the frozen 81-point grid")

    if calibration.get("validation_num_subjects") != 2:
        raise ValueError(f"{label}: calibration must use exactly two validation subjects")
    values = calibration.get("validation_subject_macro_f1_values")
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{label}: calibration subject score list must have length two")
    scores = [_finite(value, f"{label}:validation subject score") for value in values]
    if any(not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError(f"{label}: calibration subject score is outside [0, 1]")
    selected_score = _finite(
        calibration.get("validation_subject_macro_f1"),
        f"{label}:validation_subject_macro_f1",
    )
    selected_std = _finite(
        calibration.get("validation_subject_macro_f1_sample_std"),
        f"{label}:validation_subject_macro_f1_sample_std",
    )
    if not math.isclose(selected_score, mean(scores), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}: validation participant mean is inconsistent")
    if not math.isclose(selected_std, stdev(scores), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label}: validation participant sample SD is inconsistent")
    if calibration.get("tie_break") != ("highest mean, then nearest 0.5, then lower threshold"):
        raise ValueError(f"{label}: calibration tie-break is not frozen")
    return threshold


def _validate_provenance(
    payload: Mapping[str, Any], checkpoint: Mapping[str, Any], label: str
) -> tuple[str, Mapping[str, Any]]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label} has no provenance object")
    identity = _hash(provenance.get("provenance_sha256"), f"{label}:provenance")
    if checkpoint.get("provenance_sha256") != identity:
        raise ValueError(f"{label}: checkpoint and evaluation provenance disagree")
    if "git" not in provenance or provenance.get("git") is not None:
        raise ValueError(
            f"{label}: public release requires git=null because the source artifact "
            "is not rooted at an independent repository"
        )
    source_tree = provenance.get("source_tree")
    if not isinstance(source_tree, Mapping):
        raise ValueError(f"{label}:provenance.source_tree is missing")
    _hash(source_tree.get("sha256"), f"{label}:source_tree")
    required = provenance.get("required_artifacts")
    manifests = provenance.get("manifests")
    if not isinstance(required, Mapping) or not isinstance(manifests, list):
        raise ValueError(f"{label}: provenance data bindings are missing")
    return identity, provenance


def _validate_primary_data_binding(
    provenance: Mapping[str, Any], audit_binding: Mapping[str, Mapping[str, str]], label: str
) -> None:
    required = provenance.get("required_artifacts")
    if not isinstance(required, Mapping):
        raise ValueError(f"{label}: required artifact provenance is invalid")
    for name, expected_hash in audit_binding["artifacts"].items():
        item = required.get(name)
        if not isinstance(item, Mapping) or item.get("sha256") != expected_hash:
            raise ValueError(f"{label}: provenance artifact {name} disagrees with data audit")
    manifests = provenance.get("manifests")
    if not isinstance(manifests, list):
        raise ValueError(f"{label}: manifest provenance is invalid")
    indexed = {
        Path(str(item.get("path", ""))).stem: item
        for item in manifests
        if isinstance(item, Mapping)
    }
    for split, expected_hash in audit_binding["manifests"].items():
        item = indexed.get(split)
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}: provenance has no {split} manifest binding")
        if item.get("manifest_and_tensors_sha256") != expected_hash:
            raise ValueError(f"{label}: {split} manifest/tensor hash disagrees with data audit")


def _validate_training_metadata(
    metadata: Mapping[str, Any],
    seed: int,
    provenance_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    if metadata.get("status") != "completed":
        raise ValueError(f"{label}: training status is not completed")
    if metadata.get("seed") != seed:
        raise ValueError(f"{label}: training seed is not {seed}")
    metadata_provenance = metadata.get("provenance")
    if not isinstance(metadata_provenance, Mapping):
        raise ValueError(f"{label}: run metadata has no provenance")
    if metadata_provenance.get("provenance_sha256") != provenance_sha256:
        raise ValueError(f"{label}: training and evaluation provenance disagree")

    device = metadata.get("device")
    if not isinstance(device, Mapping) or device.get("resolved_device") != "cuda:0":
        raise ValueError(f"{label}: run was not completed on cuda:0")
    device_name = device.get("cuda_device_name")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError(f"{label}: CUDA device name is missing")
    determinism = metadata.get("determinism")
    deterministic_expected = {
        "deterministic_algorithms": True,
        "warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    if not isinstance(determinism, Mapping) or any(
        determinism.get(key) != value for key, value in deterministic_expected.items()
    ):
        raise ValueError(f"{label}: deterministic runtime contract is invalid")

    sequence_protocol = metadata.get("sequence_protocol")
    if not isinstance(sequence_protocol, Mapping):
        raise ValueError(f"{label}: sequence_protocol is missing")
    expected_protocol: dict[str, Any] = {
        "name": "continuous_session",
        "dataset_adapter": "StitchedManifestDataset",
        "unit": "complete_session",
        "sampler": "SequentialSampler",
        "shuffle": False,
        "drop_last": False,
        "session_batch_size": 1,
        "row_accounting": "one unique (session_id, row_index) per epoch",
    }
    expected_protocol.update(
        {
            "train_sessions": 12,
            "val_sessions": 2,
            "train_unique_steps": 3617,
            "train_batches_per_epoch": 12,
            "val_batches_per_epoch": 2,
            "epoch_loss_aggregation": "unweighted mean over complete-session batches",
            "checkpoint_selection": "minimum participant-mean validation total loss",
            "target_step_weighted_losses": "reported as diagnostics only",
        }
    )
    mismatches = {
        key: {"observed": sequence_protocol.get(key), "expected": value}
        for key, value in expected_protocol.items()
        if sequence_protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label}: frozen training protocol mismatch: {mismatches}")

    total_parameters = metadata.get("total_parameters")
    if not _is_integer(total_parameters) or total_parameters <= 0:
        raise ValueError(f"{label}: total_parameters is invalid")
    if metadata.get("trainable_parameters") != total_parameters:
        raise ValueError(f"{label}: trainable and total parameter counts disagree")
    statistics = metadata.get("target_statistics")
    expected_statistics = {
        "unique_training_steps": 3617,
        "class_counts": [817, 2800],
        "boundary_positive_counts": [168, 168],
    }
    if not isinstance(statistics, Mapping) or any(
        statistics.get(key) != value for key, value in expected_statistics.items()
    ):
        raise ValueError(f"{label}: training target counts disagree with frozen audit")
    return {"device_name": device_name, "allocated_parameter_count": total_parameters}


def _validate_seed_payload(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    spec: ExperimentSpec,
    seed: int,
    audit_binding: Mapping[str, Mapping[str, str]],
    fairness: Mapping[str, Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    if spec.cohort != PRIMARY_COHORT:
        raise ValueError(f"{label}: non-primary experiment is outside the public release")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{label} has no checkpoint object")
    if checkpoint.get("experiment") != spec.directory:
        raise ValueError(f"{label}: checkpoint experiment name is invalid")
    if checkpoint.get("seed") != seed:
        raise ValueError(f"{label}: checkpoint seed is not {seed}")
    config_sha256 = _hash(
        checkpoint.get("training_config_sha256"), f"{label}:training configuration"
    )
    provenance_sha256, provenance = _validate_provenance(payload, checkpoint, label)
    _validate_primary_data_binding(provenance, audit_binding, label)

    _validate_data(payload, PRIMARY_COUNTS, label)
    _validate_protocol(payload, primary=True, fixed=False, label=label)
    threshold = _validate_calibration(payload, label=label)
    _validate_subject_summary(payload, label)
    fixed = payload.get("fixed_threshold_0_5_sensitivity")
    if not isinstance(fixed, Mapping):
        raise ValueError(f"{label}: fixed-threshold-0.5 sensitivity is missing")
    _validate_data(fixed, PRIMARY_COUNTS, f"{label}:fixed_0_5")
    _validate_protocol(fixed, primary=True, fixed=True, label=f"{label}:fixed_0_5")
    _validate_subject_summary(fixed, f"{label}:fixed_0_5")
    selected_protocol = payload["protocol"]
    fixed_protocol = fixed["protocol"]
    for key, value in selected_protocol.items():
        if key != "frame_threshold" and fixed_protocol.get(key) != value:
            raise ValueError(f"{label}: fixed-0.5 protocol differs at {key}")
    for key in PRIMARY_COUNTS:
        if fixed["data"].get(key) != payload["data"].get(key):
            raise ValueError(f"{label}: fixed-0.5 data differs at {key}")
    for key in ("event_ap_iou_0.3", "event_ap_iou_0.5", "event_ap_iou_0.7", "event_map"):
        _same_number(
            _at(fixed, ("event", key), f"{label}:fixed:{key}"),
            _finite(
                _at(payload, ("event", key), f"{label}:selected:{key}"),
                f"{label}:selected:{key}",
            ),
            f"{label}: fixed-0.5 threshold-free {key}",
        )
    selected_subject_map = _at(
        payload,
        ("subject_macro", "metrics", "event_map"),
        f"{label}:selected subject event mAP",
    )
    fixed_subject_map = _at(
        fixed,
        ("subject_macro", "metrics", "event_map"),
        f"{label}:fixed subject event mAP",
    )
    if fixed_subject_map != selected_subject_map:
        raise ValueError(f"{label}: fixed-0.5 threshold-free subject event mAP differs")

    runtime = _validate_training_metadata(
        metadata,
        seed,
        provenance_sha256,
        label=label,
    )
    efficiency = payload.get("efficiency")
    if not isinstance(efficiency, Mapping):
        raise ValueError(f"{label} has no efficiency object")
    parameter_count = efficiency.get("parameter_count")
    if parameter_count != runtime["allocated_parameter_count"]:
        raise ValueError(f"{label}: evaluation/training parameter counts disagree")
    if efficiency.get("num_steps") != PRIMARY_COUNTS["num_unique_steps"]:
        raise ValueError(f"{label}: efficiency.num_steps is invalid")
    if efficiency.get("num_target_steps") != PRIMARY_COUNTS["num_evaluated_target_steps"]:
        raise ValueError(f"{label}: efficiency.num_target_steps is invalid")
    fair = fairness.get(spec.directory)
    if fair is None:
        raise ValueError(f"{label}: parameter fairness record is missing")
    if parameter_count != fair["allocated_parameters"]:
        raise ValueError(f"{label}: fairness/evaluation parameter counts disagree")

    values: dict[tuple[str, str], float | None] = {}
    subject_details: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in ("selected", "fixed_0_5"):
        for metric in PRIMARY_METRICS:
            values[(variant, metric.metric_id)] = _metric_value(payload, metric, variant, label)
            subject_details[(variant, metric.metric_id)] = _subject_details(
                payload, metric, variant
            )
    return {
        "seed": seed,
        "values": values,
        "subject_details": subject_details,
        "allocated_parameter_count": parameter_count,
        "provenance_sha256": provenance_sha256,
        "training_config_sha256": config_sha256,
        "device_name": runtime["device_name"],
        "selected_frame_threshold": threshold,
    }


def _aggregate_metric_entry(aggregate: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    metrics = aggregate.get("aggregate")
    if not isinstance(metrics, Mapping):
        raise ValueError("aggregate.json has no aggregate metric object")
    entry = metrics.get(key)
    if entry is None:
        return None
    if not isinstance(entry, Mapping):
        raise ValueError(f"aggregate metric {key} is not an object")
    return entry


def _validate_aggregate(
    aggregate: Mapping[str, Any],
    spec: ExperimentSpec,
    runs: Sequence[Mapping[str, Any]],
    label: str,
) -> None:
    if spec.cohort != PRIMARY_COHORT:
        raise ValueError(f"{label}: non-primary experiment is outside the public release")
    if aggregate.get("experiment") != spec.directory:
        raise ValueError(f"{label}: experiment name is invalid")
    if aggregate.get("num_runs") != len(EXPECTED_SEEDS):
        raise ValueError(f"{label}: num_runs must be three")
    if aggregate.get("seeds") != list(EXPECTED_SEEDS):
        raise ValueError(f"{label}: seeds must be {list(EXPECTED_SEEDS)}")
    config_hashes = {run["training_config_sha256"] for run in runs}
    provenance_hashes = {run["provenance_sha256"] for run in runs}
    if len(config_hashes) != 1 or aggregate.get("training_config_sha256") not in config_hashes:
        raise ValueError(f"{label}: training configuration hash disagrees with seeds")
    if len(provenance_hashes) != 1 or aggregate.get("provenance_sha256") not in provenance_hashes:
        raise ValueError(f"{label}: provenance hash disagrees with seeds")

    aggregate_data = aggregate.get("data")
    if not isinstance(aggregate_data, Mapping) or any(
        aggregate_data.get(key) != value for key, value in PRIMARY_COUNTS.items()
    ):
        raise ValueError(f"{label}: aggregate data signature is invalid")
    aggregate_protocol = aggregate.get("protocol")
    expected_protocol: dict[str, Any] = {
        "evaluation_unit": "unique chronological rows reconstructed per session",
        "positive_class": 0,
        "boundary_threshold": 0.5,
        "boundary_tolerance_steps": 1,
        "event_iou_thresholds": [0.3, 0.5, 0.7],
        "early_detection_delays_steps": [0.0, 1.0, 2.0],
        "latency_definition": LATENCY_DEFINITION,
        "target_mask_definition": TARGET_MASK_DEFINITION,
        "event_f1_protocol": EVENT_F1_PROTOCOL,
        "event_ap_protocol": EVENT_AP_PROTOCOL,
    }
    if not isinstance(aggregate_protocol, Mapping) or any(
        aggregate_protocol.get(key) != value for key, value in expected_protocol.items()
    ):
        raise ValueError(f"{label}: aggregate evaluation protocol is invalid")
    if (
        aggregate_data.get("num_dense_event_proposals")
        != PRIMARY_COUNTS["num_evaluated_target_steps"]
    ):
        raise ValueError(f"{label}: aggregate dense-proposal count is invalid")
    calibration = aggregate.get("calibration_protocol")
    expected_calibration = {
        "enabled": True,
        "method": "validation_subject_macro_f1_grid",
        "selection_split": "validation",
        "calibration_min_threshold": 0.1,
        "calibration_max_threshold": 0.9,
        "calibration_steps": 81,
    }
    if not isinstance(calibration, Mapping) or any(
        calibration.get(key) != value for key, value in expected_calibration.items()
    ):
        raise ValueError(f"{label}: aggregate calibration protocol is invalid")

    aggregate_runs = aggregate.get("runs")
    if not isinstance(aggregate_runs, list) or len(aggregate_runs) != len(EXPECTED_SEEDS):
        raise ValueError(f"{label}: aggregate seed records are invalid")
    indexed_runs: dict[int, Mapping[str, Any]] = {}
    for item in aggregate_runs:
        if not isinstance(item, Mapping) or not _is_integer(item.get("seed")):
            raise ValueError(f"{label}: aggregate contains an invalid seed record")
        indexed_runs[int(item["seed"])] = item
    if sorted(indexed_runs) != list(EXPECTED_SEEDS):
        raise ValueError(f"{label}: aggregate seed records are not exactly the frozen seeds")

    incomplete = aggregate.get("incomplete_metrics")
    if not isinstance(incomplete, list) or any(not isinstance(item, str) for item in incomplete):
        raise ValueError(f"{label}: incomplete_metrics is invalid")
    incomplete_set = set(incomplete)
    by_seed = {int(run["seed"]): run for run in runs}
    for seed in EXPECTED_SEEDS:
        aggregate_run = indexed_runs[seed]
        run = by_seed[seed]
        if (
            aggregate_run.get("experiment") != spec.directory
            or aggregate_run.get("training_config_sha256") != run["training_config_sha256"]
            or aggregate_run.get("provenance_sha256") != run["provenance_sha256"]
        ):
            raise ValueError(f"{label}: aggregate identity mismatch for seed {seed}")
        flattened = aggregate_run.get("metrics")
        if not isinstance(flattened, Mapping):
            raise ValueError(f"{label}: seed {seed} has no flattened metric map")
        for variant in ("selected", "fixed_0_5"):
            for metric in PRIMARY_METRICS:
                key = metric.aggregate_key(variant)
                observed = run["values"][(variant, metric.metric_id)]
                if observed is None:
                    if key in flattened:
                        raise ValueError(f"{label}: undefined {key} was serialized as numeric")
                else:
                    flat_value = _finite(flattened.get(key), f"{label}:seed {seed}:{key}")
                    if not math.isclose(flat_value, observed, rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError(f"{label}: seed {seed} flattened {key} is inconsistent")

    for variant in ("selected", "fixed_0_5"):
        for metric in PRIMARY_METRICS:
            key = metric.aggregate_key(variant)
            values = [
                float(run["values"][(variant, metric.metric_id)])
                for run in runs
                if run["values"][(variant, metric.metric_id)] is not None
            ]
            entry = _aggregate_metric_entry(aggregate, key)
            missing = len(runs) - len(values)
            if not values:
                if entry is not None:
                    raise ValueError(f"{label}:{key} has aggregate values despite n=0")
                continue
            if entry is None:
                raise ValueError(f"{label} is missing aggregate metric {key}")
            expected = {
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
            for field, expected_value in expected.items():
                observed = _finite(entry.get(field), f"{label}:{key}:{field}")
                if not math.isclose(observed, expected_value, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"{label}:{key}:{field} disagrees with per-seed values")
            if entry.get("n") != len(values) or entry.get("missing") != missing:
                raise ValueError(f"{label}:{key} has invalid n/missing")
            if missing and key not in incomplete_set:
                raise ValueError(f"{label}:{key} is missing seeds but not declared incomplete")


def _collect_experiment(
    results_dir: Path,
    spec: ExperimentSpec,
    audit_binding: Mapping[str, Mapping[str, str]],
    fairness: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if spec.cohort != PRIMARY_COHORT or spec.directory not in PRIMARY_EXPERIMENT_DIRECTORIES:
        raise ValueError(f"{spec.directory}: experiment is outside the primary release scope")
    runs: list[dict[str, Any]] = []
    missing: list[str] = []
    errors: list[str] = []
    for seed in EXPECTED_SEEDS:
        seed_dir = results_dir / spec.directory / f"seed_{seed}"
        metrics_path = seed_dir / "test_metrics.json"
        metadata_path = seed_dir / "run_metadata.json"
        if not metrics_path.is_file():
            missing.append(_relative(metrics_path, results_dir))
            continue
        if not metadata_path.is_file():
            missing.append(_relative(metadata_path, results_dir))
            continue
        label = f"{spec.directory}/seed_{seed}"
        try:
            run = _validate_seed_payload(
                _load_json(metrics_path),
                _load_json(metadata_path),
                spec,
                seed,
                audit_binding,
                fairness,
                label,
            )
            runs.append(run)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    aggregate_path = results_dir / spec.directory / "aggregate.json"
    if not aggregate_path.is_file():
        missing.append(_relative(aggregate_path, results_dir))
    elif len(runs) == len(EXPECTED_SEEDS) and not errors:
        try:
            _validate_aggregate(
                _load_json(aggregate_path), spec, runs, f"{spec.directory}/aggregate.json"
            )
        except (OSError, ValueError) as exc:
            errors.append(str(exc))

    parameter_counts = {run["allocated_parameter_count"] for run in runs}
    provenance_hashes = {run["provenance_sha256"] for run in runs}
    config_hashes = {run["training_config_sha256"] for run in runs}
    devices = {run["device_name"] for run in runs}
    if len(parameter_counts) > 1:
        errors.append(f"{spec.directory}: parameter count changes across seeds")
    if len(provenance_hashes) > 1:
        errors.append(f"{spec.directory}: provenance changes across seeds")
    if len(config_hashes) > 1:
        errors.append(f"{spec.directory}: training configuration changes across seeds")
    if len(devices) > 1:
        errors.append(f"{spec.directory}: GPU device changes across seeds")
    complete = len(runs) == len(EXPECTED_SEEDS) and not missing and not errors
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    if complete:
        for variant in ("selected", "fixed_0_5"):
            for metric in PRIMARY_METRICS:
                values = [
                    float(run["values"][(variant, metric.metric_id)])
                    for run in runs
                    if run["values"][(variant, metric.metric_id)] is not None
                ]
                summaries[(variant, metric.metric_id)] = {
                    "mean": mean(values) if values else None,
                    "sample_std": stdev(values)
                    if len(values) > 1
                    else (0.0 if len(values) == 1 else None),
                    "n": len(values),
                    "missing": len(runs) - len(values),
                }
    return {
        "spec": spec,
        "complete": complete,
        "runs": sorted(runs, key=lambda item: item["seed"]),
        "missing": sorted(missing),
        "errors": sorted(errors),
        "summaries": summaries,
        "allocated_parameter_count": next(iter(parameter_counts), None),
        "provenance_sha256": next(iter(provenance_hashes), None),
        "training_config_sha256": next(iter(config_hashes), None),
        "devices": sorted(devices),
    }


def _validate_bci_audit(
    results_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], list[str]]:
    path = results_dir / "bci_subjects_data_audit.json"
    empty_binding = {"artifacts": {}, "manifests": {}}
    if not path.is_file():
        return {}, empty_binding, [_relative(path, results_dir)]
    try:
        payload = _load_json(path)
        if payload.get("schema_version") != 1 or payload.get("dataset") != "bci_subjects_ept_v1":
            raise ValueError("bci_subjects_data_audit.json has an invalid schema/dataset")
        privacy = payload.get("privacy_contract")
        expected_privacy = {
            "deidentified_aggregate_only": True,
            "contains_subject_ids": False,
            "contains_source_filenames": False,
            "contains_row_level_data": False,
        }
        if not isinstance(privacy, Mapping) or any(
            privacy.get(key) != value for key, value in expected_privacy.items()
        ):
            raise ValueError("BCI audit does not satisfy the deidentified release contract")
        protocol = payload.get("protocol")
        expected_protocol = {
            "subject_disjoint_fixed_split": True,
            "pilot_not_cross_validation": True,
            "split_strategy": "exhaustive_constrained_subject_stratification_v1",
            "split_frozen_before_training": True,
            "label_aware_pretraining_stratification": True,
            "split_seed": 42,
            "split_objective_value": 0.021588309116414767,
            "split_feasible_assignments": 8400,
            "positive_class": 0,
            "text_enabled": False,
            "audio_enabled": False,
            "target_mask_required": True,
        }
        if not isinstance(protocol, Mapping) or any(
            protocol.get(key) != value for key, value in expected_protocol.items()
        ):
            raise ValueError("BCI audit protocol is not the frozen pre-training split")
        _hash(protocol.get("split_inputs_sha256"), "BCI split inputs")
        _hash(protocol.get("split_assignment_sha256"), "BCI split assignment")
        expected_totals = {
            "total_subject_sessions": 18,
            "total_context_steps": 7405,
            "total_target_valid_steps": 5490,
        }
        if any(payload.get(key) != value for key, value in expected_totals.items()):
            raise ValueError("BCI audit global counts are not frozen")
        split_expected = {
            "train": {
                "num_subject_sessions": 12,
                "num_context_steps": 4982,
                "num_target_valid_steps": 3617,
                "label_counts": {"0": 817, "1": 2800},
                "num_deception_events": 168,
                "modality_session_counts": {
                    "eeg_time": 3,
                    "eeg_spectral": 3,
                    "physiology": 9,
                    "video": 11,
                    "audio": 0,
                    "text": 12,
                },
            },
            "val": {
                "num_subject_sessions": 2,
                "num_context_steps": 918,
                "num_target_valid_steps": 672,
                "label_counts": {"0": 141, "1": 531},
                "num_deception_events": 25,
                "modality_session_counts": {
                    "eeg_time": 1,
                    "eeg_spectral": 1,
                    "physiology": 1,
                    "video": 2,
                    "audio": 0,
                    "text": 2,
                },
            },
            "test": {
                "num_subject_sessions": 4,
                "num_context_steps": 1505,
                "num_target_valid_steps": 1201,
                "label_counts": {"0": 287, "1": 914},
                "num_deception_events": 55,
                "modality_session_counts": {
                    "eeg_time": 1,
                    "eeg_spectral": 1,
                    "physiology": 3,
                    "video": 4,
                    "audio": 0,
                    "text": 4,
                },
            },
        }
        splits = payload.get("splits")
        if not isinstance(splits, Mapping):
            raise ValueError("BCI audit has no split summary")
        for split, expected in split_expected.items():
            observed = splits.get(split)
            if not isinstance(observed, Mapping) or any(
                observed.get(key) != value for key, value in expected.items()
            ):
                raise ValueError(f"BCI audit {split} counts are not frozen")
            prevalence = _finite(
                observed.get("deception_prevalence"),
                f"BCI audit {split} deception prevalence",
            )
            expected_prevalence = expected["label_counts"]["0"] / expected["num_target_valid_steps"]
            if not math.isclose(prevalence, expected_prevalence, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"BCI audit {split} prevalence is inconsistent")
        coverage = payload.get("modality_coverage")
        expected_coverage = {
            "eeg_time": {"sessions_available": 5, "available_steps": 1931},
            "eeg_spectral": {"sessions_available": 5, "available_steps": 1931},
            "physiology": {"sessions_available": 13, "available_steps": 5493},
            "video": {"sessions_available": 17, "available_steps": 6999},
            "audio": {"sessions_available": 0, "available_steps": 0},
            "text": {"sessions_available": 18, "available_steps": 6954},
        }
        if not isinstance(coverage, Mapping):
            raise ValueError("BCI audit has no modality coverage")
        for modality, expected_values in expected_coverage.items():
            item = coverage.get(modality)
            if not isinstance(item, Mapping) or any(
                item.get(key) != value for key, value in expected_values.items()
            ):
                raise ValueError(f"BCI audit {modality} coverage is invalid")

        _hash(payload.get("data_fingerprint_sha256"), "BCI data fingerprint")
        fingerprint_inputs = payload.get("fingerprint_inputs")
        if not isinstance(fingerprint_inputs, Mapping):
            raise ValueError("BCI audit has no fingerprint inputs")
        artifacts = fingerprint_inputs.get("artifacts")
        manifests = fingerprint_inputs.get("manifests_and_tensors")
        if not isinstance(artifacts, Mapping) or not isinstance(manifests, Mapping):
            raise ValueError("BCI audit fingerprint inputs are invalid")
        binding = {
            "artifacts": {
                name: _hash(artifacts.get(name), f"BCI audit artifact {name}")
                for name in ("dataset_summary", "feature_schema", "normalization_stats")
            },
            "manifests": {
                split: _hash(manifests.get(split), f"BCI audit manifest {split}")
                for split in ("train", "val", "test")
            },
        }
        limitations = payload.get("limitations")
        if not isinstance(limitations, list) or len(limitations) < 5:
            raise ValueError("BCI audit limitations are missing")
        return payload, binding, []
    except (OSError, ValueError) as exc:
        return {}, empty_binding, [str(exc)]


def _load_parameter_fairness(
    results_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    path = results_dir / "parameter_fairness_bci_subjects.json"
    if not path.is_file():
        return {}, [_relative(path, results_dir)]
    try:
        payload = _load_json(path)
        if payload.get("reference_experiment") != "eptnet_bci_subjects_no_text":
            raise ValueError("parameter fairness reference experiment is invalid")
        if payload.get("device") != "cpu":
            raise ValueError("parameter fairness audit must run on CPU")
        if payload.get("deterministic_model_seed") != 0:
            raise ValueError("parameter fairness deterministic model seed is invalid")
        selection = payload.get("selection")
        if not isinstance(selection, Mapping):
            raise ValueError("parameter fairness shared sample selection is missing")
        if selection.get("selection_protocol") != FAIRNESS_SELECTION_PROTOCOL:
            raise ValueError("parameter fairness sample selection protocol is invalid")
        manifest_path = selection.get("manifest_path")
        if (
            not isinstance(manifest_path, str)
            or not manifest_path
            or Path(manifest_path).is_absolute()
        ):
            raise ValueError("parameter fairness manifest path must be non-empty and relative")
        manifest_sha256 = _hash(selection.get("manifest_sha256"), "parameter fairness manifest")
        selected_ids = selection.get("selected_sample_ids")
        if (
            not isinstance(selected_ids, list)
            or not selected_ids
            or not all(isinstance(value, str) and value for value in selected_ids)
            or selected_ids != sorted(set(selected_ids))
        ):
            raise ValueError("parameter fairness selected sample IDs are invalid")
        if selection.get("selected_sample_count") != len(selected_ids):
            raise ValueError("parameter fairness selected sample count is inconsistent")
        selected_ids_sha256 = _hash(
            selection.get("selected_sample_ids_sha256"),
            "parameter fairness selected sample IDs",
        )
        expected_modalities = ["eeg_time", "eeg_spectral", "hr", "video"]
        if selection.get("required_cover_modalities") != expected_modalities:
            raise ValueError("parameter fairness required modality cover is invalid")
        if selection.get("enabled_but_globally_unavailable_modalities") != []:
            raise ValueError("parameter fairness has an enabled but unavailable modality")
        selected_tensors = selection.get("selected_tensor_sha256")
        if not isinstance(selected_tensors, Mapping) or set(selected_tensors) != set(selected_ids):
            raise ValueError("parameter fairness selected tensor hashes are incomplete")
        for sample_id in selected_ids:
            _hash(
                selected_tensors[sample_id],
                f"parameter fairness selected tensor {sample_id}",
            )
        selected_samples = selection.get("selected_samples")
        if not isinstance(selected_samples, list) or len(selected_samples) != len(selected_ids):
            raise ValueError("parameter fairness selected sample statistics are incomplete")
        covered_modalities: set[str] = set()
        observed_ids: list[str] = []
        for sample in selected_samples:
            if not isinstance(sample, Mapping):
                raise ValueError("parameter fairness selected sample statistics are invalid")
            sample_id = sample.get("sample_id")
            if not isinstance(sample_id, str):
                raise ValueError("parameter fairness selected sample ID is invalid")
            observed_ids.append(sample_id)
            target_steps = sample.get("target_valid_steps")
            if not _is_integer(target_steps) or target_steps <= 0:
                raise ValueError("parameter fairness sample has no target-valid step")
            coverage = sample.get("covered_enabled_modalities")
            if not isinstance(coverage, list) or not all(
                isinstance(value, str) for value in coverage
            ):
                raise ValueError("parameter fairness sample modality coverage is invalid")
            covered_modalities.update(coverage)
        if observed_ids != selected_ids or not set(expected_modalities).issubset(
            covered_modalities
        ):
            raise ValueError("parameter fairness selected samples do not cover all modalities")

        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("parameter fairness records are missing")
        indexed: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("experiment"), str):
                raise ValueError("parameter fairness contains an invalid record")
            experiment = record["experiment"]
            if experiment in indexed:
                raise ValueError(f"parameter fairness duplicates {experiment}")
            indexed[experiment] = record
        if set(indexed) != set(PRIMARY_FAIRNESS_EXPECTED):
            raise ValueError("parameter fairness experiment set is not the frozen BCI matrix")
        reference_graph = PRIMARY_FAIRNESS_EXPECTED["eptnet_bci_subjects_no_text"][
            "graph_participating_parameters"
        ]
        for experiment, expected in PRIMARY_FAIRNESS_EXPECTED.items():
            record = indexed[experiment]
            for key, value in expected.items():
                if record.get(key) != value:
                    raise ValueError(f"parameter fairness {experiment}:{key} is invalid")
            if record.get("counting_protocol") != FAIRNESS_GRAPH_COUNTING_PROTOCOL:
                raise ValueError(f"parameter fairness {experiment} has an invalid graph protocol")
            if (
                record.get("nonzero_gradient_parameter_counting_protocol")
                != FAIRNESS_NONZERO_PARAMETER_COUNTING_PROTOCOL
                or record.get("nonzero_gradient_element_counting_protocol")
                != FAIRNESS_NONZERO_ELEMENT_COUNTING_PROTOCOL
            ):
                raise ValueError(
                    f"parameter fairness {experiment} has an invalid non-zero protocol"
                )
            graph = expected["graph_participating_parameters"]
            if (
                record.get("executed_parameters") != graph
                or record.get("executed_parameters_is_alias_of") != "graph_participating_parameters"
            ):
                raise ValueError(
                    f"parameter fairness {experiment} has an invalid compatibility alias"
                )
            if (
                record.get("audit_sample_ids") != selected_ids
                or record.get("audit_sample_ids_sha256") != selected_ids_sha256
                or record.get("manifest_sha256") != manifest_sha256
            ):
                raise ValueError(
                    f"parameter fairness {experiment} does not use the shared audit samples"
                )
            nonzero_parameters = record.get("nonzero_gradient_parameters")
            nonzero_elements = record.get("nonzero_gradient_elements")
            if (
                not _is_integer(nonzero_parameters)
                or not 0 < nonzero_parameters <= graph
                or not _is_integer(nonzero_elements)
                or not 0 < nonzero_elements <= nonzero_parameters
            ):
                raise ValueError(f"parameter fairness {experiment} has invalid non-zero counts")
            if record.get("zero_gradient_graph_parameters") != graph - nonzero_parameters:
                raise ValueError(
                    f"parameter fairness {experiment} has inconsistent zero-gradient counts"
                )
            expected_graph_ratio = graph / reference_graph
            for key in (
                "graph_participating_ratio_to_eptnet",
                "executed_parameter_ratio_to_eptnet",
            ):
                observed_ratio = _finite(record.get(key), f"parameter fairness {experiment}:{key}")
                if not math.isclose(
                    observed_ratio,
                    expected_graph_ratio,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(f"parameter fairness {experiment} graph ratio is inconsistent")
            trainable = expected["trainable_parameters"]
            expected_graph_fraction = graph / trainable
            for key in ("graph_participating_fraction", "executed_fraction"):
                observed_fraction = _finite(
                    record.get(key), f"parameter fairness {experiment}:{key}"
                )
                if not math.isclose(
                    observed_fraction,
                    expected_graph_fraction,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"parameter fairness {experiment} graph fraction is inconsistent"
                    )
            for key, numerator in (
                ("nonzero_gradient_parameter_fraction", nonzero_parameters),
                ("nonzero_gradient_element_fraction", nonzero_elements),
            ):
                observed_fraction = _finite(
                    record.get(key), f"parameter fairness {experiment}:{key}"
                )
                if not math.isclose(
                    observed_fraction,
                    numerator / trainable,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"parameter fairness {experiment} non-zero fraction is inconsistent"
                    )
            _finite(
                record.get("mean_audit_loss"),
                f"parameter fairness {experiment}:mean_audit_loss",
            )
        reference_nonzero_parameters = int(
            indexed["eptnet_bci_subjects_no_text"]["nonzero_gradient_parameters"]
        )
        reference_nonzero_elements = int(
            indexed["eptnet_bci_subjects_no_text"]["nonzero_gradient_elements"]
        )
        for experiment, record in indexed.items():
            for key, numerator, denominator in (
                (
                    "nonzero_gradient_parameter_ratio_to_eptnet",
                    int(record["nonzero_gradient_parameters"]),
                    reference_nonzero_parameters,
                ),
                (
                    "nonzero_gradient_element_ratio_to_eptnet",
                    int(record["nonzero_gradient_elements"]),
                    reference_nonzero_elements,
                ),
            ):
                observed_ratio = _finite(record.get(key), f"parameter fairness {experiment}:{key}")
                if not math.isclose(
                    observed_ratio,
                    numerator / denominator,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"parameter fairness {experiment} non-zero ratio is inconsistent"
                    )
        for experiment in (
            "baseline_early_fusion_gru_bci_subjects_no_text",
            "baseline_fusion_transformer_bci_subjects_no_text",
        ):
            if abs(float(indexed[experiment]["graph_participating_ratio_to_eptnet"]) - 1.0) > 0.03:
                raise ValueError(f"{experiment} exceeds the predeclared 3% graph-parameter band")
        return indexed, []
    except (OSError, ValueError) as exc:
        return {}, [str(exc)]


def _cohort_issues(results: Sequence[Mapping[str, Any]], cohort: str) -> list[str]:
    issues: list[str] = []
    hashes = {
        result["provenance_sha256"] for result in results if result["provenance_sha256"] is not None
    }
    if len(hashes) > 1:
        issues.append(f"{cohort}: available experiments do not share one provenance")
    devices = {device for result in results for device in result["devices"]}
    if len(devices) > 1:
        issues.append(f"{cohort}: available experiments do not share one GPU model")
    return issues


def _format_mean_sd(summary: Mapping[str, Any] | None) -> str:
    if not summary or summary.get("mean") is None:
        if summary and summary.get("n") == 0:
            return f"undefined (n=0; {summary.get('missing', 0)} undef.)"
        return "—"
    rendered = f"{float(summary['mean']):.4f} ± {float(summary['sample_std']):.4f}"
    if int(summary.get("missing", 0)):
        rendered += f" (n={int(summary['n'])}; {int(summary['missing'])} undef.)"
    return rendered


def _format_seed(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _require_primary_scope(results: Sequence[Mapping[str, Any]]) -> None:
    observed = tuple(result["spec"].directory for result in results)
    if observed != PRIMARY_EXPERIMENT_DIRECTORIES:
        raise ValueError(
            "release results must contain the four primary experiments in frozen order"
        )
    if any(result["spec"].cohort != PRIMARY_COHORT for result in results):
        raise ValueError("release results contain a non-primary cohort")


def _write_summary_csv(
    path: Path,
    results: Sequence[Mapping[str, Any]],
    fairness: Mapping[str, Mapping[str, Any]],
) -> None:
    _require_primary_scope(results)
    fieldnames = [
        "cohort",
        "role",
        "experiment",
        "display_name",
        "status",
        "evaluation_variant",
        "aggregation_unit",
        "metric",
        "metric_label",
        "seed_mean",
        "seed_sample_std",
        "seed_n",
        "seed_missing",
        "expected_seeds",
        "completed_seeds",
        "allocated_parameters",
        "executed_parameters",
        "executed_ratio_to_eptnet",
        "provenance_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            spec: ExperimentSpec = result["spec"]
            fair = fairness.get(spec.directory, {})
            for variant in ("selected", "fixed_0_5"):
                for metric in PRIMARY_METRICS:
                    summary = result["summaries"].get((variant, metric.metric_id), {})
                    writer.writerow(
                        {
                            "cohort": spec.cohort,
                            "role": spec.role,
                            "experiment": spec.directory,
                            "display_name": spec.display_name,
                            "status": "complete" if result["complete"] else "incomplete",
                            "evaluation_variant": variant,
                            "aggregation_unit": metric.aggregation_unit,
                            "metric": metric.metric_id,
                            "metric_label": metric.label,
                            "seed_mean": _csv(summary.get("mean")),
                            "seed_sample_std": _csv(summary.get("sample_std")),
                            "seed_n": summary.get("n", ""),
                            "seed_missing": summary.get("missing", ""),
                            "expected_seeds": ";".join(map(str, EXPECTED_SEEDS)),
                            "completed_seeds": ";".join(str(run["seed"]) for run in result["runs"]),
                            "allocated_parameters": result["allocated_parameter_count"] or "",
                            "executed_parameters": fair.get("executed_parameters", ""),
                            "executed_ratio_to_eptnet": _csv(
                                fair.get("executed_parameter_ratio_to_eptnet")
                            ),
                            "provenance_sha256": result["provenance_sha256"] or "",
                        }
                    )


def _write_seed_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    _require_primary_scope(results)
    fieldnames = [
        "cohort",
        "role",
        "experiment",
        "display_name",
        "seed",
        "status",
        "evaluation_variant",
        "aggregation_unit",
        "metric",
        "metric_label",
        "value",
        "participant_n",
        "participant_missing",
        "participant_sample_std",
        "participant_bootstrap_ci95_lower",
        "participant_bootstrap_ci95_upper",
        "selected_frame_threshold",
        "allocated_parameters",
        "provenance_sha256",
        "training_config_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for result in results:
            spec: ExperimentSpec = result["spec"]
            by_seed = {run["seed"]: run for run in result["runs"]}
            for seed in EXPECTED_SEEDS:
                run = by_seed.get(seed)
                for variant in ("selected", "fixed_0_5"):
                    for metric in PRIMARY_METRICS:
                        details = (
                            run["subject_details"].get((variant, metric.metric_id), {})
                            if run
                            else {}
                        )
                        writer.writerow(
                            {
                                "cohort": spec.cohort,
                                "role": spec.role,
                                "experiment": spec.directory,
                                "display_name": spec.display_name,
                                "seed": seed,
                                "status": "complete" if run else "missing_or_invalid",
                                "evaluation_variant": variant,
                                "aggregation_unit": metric.aggregation_unit,
                                "metric": metric.metric_id,
                                "metric_label": metric.label,
                                "value": _csv(
                                    run["values"].get((variant, metric.metric_id)) if run else None
                                ),
                                "participant_n": details.get("subject_n", ""),
                                "participant_missing": details.get("subject_missing", ""),
                                "participant_sample_std": _csv(details.get("subject_sample_std")),
                                "participant_bootstrap_ci95_lower": _csv(
                                    details.get("subject_bootstrap_ci95_lower")
                                ),
                                "participant_bootstrap_ci95_upper": _csv(
                                    details.get("subject_bootstrap_ci95_upper")
                                ),
                                "selected_frame_threshold": _csv(
                                    run.get("selected_frame_threshold") if run else None
                                ),
                                "allocated_parameters": (
                                    run["allocated_parameter_count"] if run else ""
                                ),
                                "provenance_sha256": (run["provenance_sha256"] if run else ""),
                                "training_config_sha256": (
                                    run["training_config_sha256"] if run else ""
                                ),
                            }
                        )


def _missing_lines(results: Sequence[Mapping[str, Any]], global_issues: Sequence[str]) -> list[str]:
    lines: list[str] = []
    for result in results:
        spec: ExperimentSpec = result["spec"]
        lines.extend(f"- `{spec.directory}`：缺少 `{item}`" for item in result["missing"])
        lines.extend(f"- `{spec.directory}`：校验失败：{item}" for item in result["errors"])
    lines.extend(f"- 全局门禁：{issue}" for issue in global_issues)
    return lines


def _summary(result: Mapping[str, Any], variant: str, metric_id: str) -> Mapping[str, Any] | None:
    return result["summaries"].get((variant, metric_id))


def _write_report(
    path: Path,
    primary: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    fairness: Mapping[str, Mapping[str, Any]],
    global_issues: Sequence[str],
    release_ready: bool,
) -> None:
    _require_primary_scope(primary)
    lines = [
        "# EPT-Net 发布实验报告",
        "",
        f"产物门禁：**{'PASS' if release_ready else 'INCOMPLETE（不得引用为最终结果）'}**  ",
        "科学证据等级：**固定 subject-disjoint split 的 pilot**；不是交叉验证，也不是人口级泛化证据。",
        "",
        "## 1. 证据边界与门禁",
        "",
    ]
    if release_ready:
        lines.append(
            "Primary BCI 四组模型 × seeds 13/42/73 均通过文件、GPU、确定性、数据计数、"
            "校准、参数与 provenance 一致性校验。"
        )
    else:
        lines.extend(
            [
                "正式矩阵尚未完整。任何不完整实验的局部均值均被抑制；补齐并重新运行构建器前，"
                "不得据此比较模型、宣称机制有效或声称达到顶会实证标准。",
                "",
                "### 缺失或无效产物",
                "",
                *_missing_lines(primary, global_issues),
            ]
        )
    lines.extend(
        [
            "",
            "本公开 release 仅包含 primary 4×3 矩阵。历史单会话工程结果不进入 release readiness，"
            "也不进入公开机器可读数值。产物门禁 PASS 只表示本地冻结协议可追溯，不等于科学结论"
            "或顶会证据充分。",
            "",
            "## 2. Primary：18-subject fixed-split pilot",
            "",
            "- 训练前冻结、subject-disjoint 的 12/2/4 train/validation/test 划分；这是 label-aware、"
            "modality-constrained stratification，且仅有一个 fixed split。",
            "- 模型观察完整交互上下文，但 loss、阈值选择和测试指标仅在 target-speaker mask 内计算；"
            "无效说话人区间会关闭且不能开启预测事件。",
            "- text 与 audio 输入禁用。阈值只依据 2 个 validation subjects 的 participant-mean Macro-F1 "
            "在 0.10–0.90 的 81 点网格上选择，随后冻结到 test。",
            "- Event F1、early detection 与 latency 使用冻结帧阈值的因果 decoder；event AP/mAP 则对每个 "
            "target-valid step 生成一个由正类概率评分、offset 定界的稠密候选，按连续 target segment 截断，"
            "不使用 score threshold 或 NMS。因此 mAP 不随 validation-selected / fixed-0.5 operating point 改变。",
            "- 每个 seed 的 participant-macro 指标附 10,000 次 participant bootstrap 区间；下表的 `±` "
            "则是三个模型初始化 seed 之间的样本标准差。两者都不能替代 repeated group splits。",
            "",
        ]
    )
    if audit:
        test = audit["splits"]["test"]
        lines.extend(
            [
                f"冻结数据计数：18 subjects、{audit['total_context_steps']} context steps、"
                f"{audit['total_target_valid_steps']} target-valid steps；test 为 "
                f"{test['num_subject_sessions']} subjects / {test['num_context_steps']} context / "
                f"{test['num_target_valid_steps']} target-valid steps / {test['num_deception_events']} events。",
                "",
            ]
        )
    lines.extend(
        [
            "### Participant-macro 主结果（validation-selected threshold）",
            "",
            "| 模型 | 状态 | Frame Macro-F1 | Balanced Acc. | AUROC | AP | Boundary F1 | Event F1@0.5 | Threshold-free Event mAP | TTD seconds |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    primary_columns = (
        "subject_frame_macro_f1",
        "subject_frame_balanced_accuracy",
        "subject_frame_auroc",
        "subject_frame_average_precision",
        "subject_boundary_macro_f1",
        "subject_event_f1_iou_0_5",
        "subject_event_map",
        "matched_event_delay_seconds",
    )
    for result in primary:
        values = [
            _format_mean_sd(_summary(result, "selected", metric_id))
            for metric_id in primary_columns
        ]
        lines.append(
            f"| {result['spec'].display_name} | "
            f"{'complete' if result['complete'] else 'incomplete'} | " + " | ".join(values) + " |"
        )
    lines.extend(
        [
            "",
            "TTD 是成功匹配事件条件下的指标。没有匹配事件的 seed 保持 `undefined`，汇总明确显示"
            "有效 seed n 与 undefined 数，绝不补成 0。participant-level 的 n、missing 和 bootstrap CI "
            "保存在 `release_seed_metrics.csv`。",
            "",
            "### 固定阈值 0.5 敏感性分析",
            "",
            "该敏感性分析只改变 operating-point 指标。Threshold-free event AP/mAP 使用相同稠密候选排名，"
            "构建器会逐 seed 强制验证 selected 与 fixed-0.5 的 AP@各 IoU 和 mAP 完全一致。",
            "",
            "| 模型 | Selected Frame Macro-F1 | Fixed-0.5 Frame Macro-F1 | Selected Event F1@0.5 | Fixed-0.5 Event F1@0.5 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for result in primary:
        cells = [
            _format_mean_sd(_summary(result, "selected", "subject_frame_macro_f1")),
            _format_mean_sd(_summary(result, "fixed_0_5", "subject_frame_macro_f1")),
            _format_mean_sd(_summary(result, "selected", "subject_event_f1_iou_0_5")),
            _format_mean_sd(_summary(result, "fixed_0_5", "subject_event_f1_iou_0_5")),
        ]
        lines.append(f"| {result['spec'].display_name} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### 参数公平性",
            "",
            "`allocated` 是模型对象中全部可训练参数；`executed` 是完整多任务 loss 反传后实际收到"
            "梯度的参数。baseline 比较按 executed 参数控制在 EPT-Net 的 ±3% 内。",
            "",
            "| 模型 | Allocated | Trainable | Executed | Executed / EPT |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for result in primary:
        spec: ExperimentSpec = result["spec"]
        record = fairness.get(spec.directory)
        if record:
            lines.append(
                f"| {spec.display_name} | {int(record['allocated_parameters']):,} | "
                f"{int(record['trainable_parameters']):,} | "
                f"{int(record['executed_parameters']):,} | "
                f"{float(record['executed_parameter_ratio_to_eptnet']):.3f}× |"
            )
        else:
            lines.append(f"| {spec.display_name} | — | — | — | — |")
    lines.extend(
        [
            "",
            "`w/o recurrent fusion update` 仅执行 EPT-Net 的 82.6% 参数，因此是容量混杂的机制消融，"
            "不能把差异单独归因于 recurrent fusion update。",
            "",
            "## 3. 可追溯性与发布判断",
            "",
            "- 每个 seed 的 checkpoint、evaluation 与 run metadata provenance 必须相同；primary "
            "矩阵的所有可用实验必须共享同一 provenance。",
            "- Primary evaluation provenance 中的 dataset summary、feature schema、normalization statistics "
            "和 train/val/test manifest+tensors 摘要必须逐项匹配匿名数据审计。",
            "- `aggregate.json` 必须与三份 seed metrics 在 identity、数值、样本标准差、n/missing 和"
            "conditional latency 上逐项一致。",
            "- `release_summary.csv` 提供四模型的三-seed 汇总；`release_seed_metrics.csv` 保留原始 "
            "seed 值、participant bootstrap、校准阈值和 provenance。",
            "- 报告不含生成时间，JSON key、行顺序与 LF 换行固定，输入不变时输出字节稳定。",
            "",
        ]
    )
    if release_ready:
        lines.append(
            "工程/产物门禁：**PASS**。科学判断：**pilot only**。可以如实报告冻结 fixed-split 的"
            "本地结果，但当前 n=18、validation n=2、test n=4、模态缺失且没有 repeated group split，"
            "不能宣称已达到顶会级实证充分性或人口级泛化。"
        )
    else:
        lines.append(
            "工程/产物门禁：**INCOMPLETE**。在缺失项补齐前，不输出模型优劣、机制有效性、"
            "统计显著性或顶会性能结论。"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _experiment_status(result: Mapping[str, Any]) -> dict[str, Any]:
    spec: ExperimentSpec = result["spec"]
    return {
        "experiment": spec.directory,
        "display_name": spec.display_name,
        "role": spec.role,
        "complete": result["complete"],
        "expected_seeds": list(EXPECTED_SEEDS),
        "completed_seeds": [run["seed"] for run in result["runs"]],
        "missing": result["missing"],
        "errors": result["errors"],
        "provenance_sha256": result["provenance_sha256"],
        "training_config_sha256": result["training_config_sha256"],
        "devices": result["devices"],
    }


def _status_payload(
    primary: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    audit_issues: Sequence[str],
    fairness_issues: Sequence[str],
    primary_issues: Sequence[str],
    release_ready: bool,
) -> dict[str, Any]:
    _require_primary_scope(primary)
    return {
        "schema_version": 3,
        "release_scope": "primary_only",
        "matrix_shape": {"experiments": 4, "seeds_per_experiment": 3},
        "included_cohorts": [PRIMARY_COHORT],
        "release_ready": release_ready,
        "artifact_release_ready": release_ready,
        "scientific_evidence_level": "single_fixed_subject_disjoint_split_pilot",
        "top_conference_claim_ready": False,
        "top_conference_claim_blockers": [
            "Only one pre-training fixed 12/2/4 subject split is evaluated.",
            "Validation and test contain only 2 and 4 subjects, respectively.",
            "EEG/physiology availability is missing-not-at-random across subjects.",
            "No repeated grouped split or nested group cross-validation is available.",
        ],
        "deterministic_output": {
            "contains_generation_timestamp": False,
            "line_endings": "LF",
            "cross_cohort_pooling": False,
        },
        "primary_data_audit": {
            "complete": not audit_issues,
            "issues": list(audit_issues),
            "data_fingerprint_sha256": audit.get("data_fingerprint_sha256"),
            "expected_counts": {
                "subjects": 18,
                "train_val_test_subjects": [12, 2, 4],
                **PRIMARY_COUNTS,
            },
        },
        "parameter_fairness": {
            "complete": not fairness_issues,
            "issues": list(fairness_issues),
            "baseline_executed_parameter_tolerance_fraction": 0.03,
            "ablation_capacity_confounded": True,
        },
        "cohorts": [
            {
                "cohort": PRIMARY_COHORT,
                "role": "primary",
                "release_complete": all(result["complete"] for result in primary)
                and not primary_issues,
                "issues": list(primary_issues),
                "expected_seeds": list(EXPECTED_SEEDS),
                "expected_test_counts": PRIMARY_COUNTS,
                "required_calibration_method": "validation_subject_macro_f1_grid",
                "requires_subject_macro": True,
                "requires_fixed_threshold_0_5_sensitivity": True,
                "event_f1_protocol": EVENT_F1_PROTOCOL,
                "event_ap_protocol": EVENT_AP_PROTOCOL,
                "experiments": [_experiment_status(result) for result in primary],
            },
        ],
        "outputs": [
            "release_report.md",
            "release_report_status.json",
            "release_summary.csv",
            "release_seed_metrics.csv",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results",
        help="Directory containing experiment artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: results directory)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return zero for development even when the status remains INCOMPLETE",
    )
    args = parser.parse_args(argv)
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit, audit_binding, audit_issues = _validate_bci_audit(results_dir)
    fairness, fairness_issues = _load_parameter_fairness(results_dir)
    primary = [
        _collect_experiment(results_dir, spec, audit_binding, fairness)
        for spec in PRIMARY_EXPERIMENTS
    ]
    _require_primary_scope(primary)
    primary_issues = _cohort_issues(primary, PRIMARY_COHORT)
    global_issues = [
        *audit_issues,
        *fairness_issues,
        *primary_issues,
    ]
    release_ready = (
        not global_issues
        and len(primary) == len(PRIMARY_EXPERIMENTS)
        and all(result["complete"] for result in primary)
    )

    summary_path = output_dir / "release_summary.csv"
    seed_path = output_dir / "release_seed_metrics.csv"
    report_path = output_dir / "release_report.md"
    status_path = output_dir / "release_report_status.json"
    _write_summary_csv(summary_path, primary, fairness)
    _write_seed_csv(seed_path, primary)
    _write_report(
        report_path,
        primary,
        audit,
        fairness,
        global_issues,
        release_ready,
    )
    status = _status_payload(
        primary,
        audit,
        audit_issues,
        fairness_issues,
        primary_issues,
        release_ready,
    )
    status_path.write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "release_ready": release_ready,
                "report": report_path.name,
                "status": status_path.name,
                "summary_csv": summary_path.name,
                "seed_csv": seed_path.name,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if release_ready or args.allow_incomplete else 2


if __name__ == "__main__":
    sys.exit(main())
