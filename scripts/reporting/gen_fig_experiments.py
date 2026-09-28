#!/usr/bin/env python3
"""Generate publication figures for the fixed-split BCI-subject pilot.

The primary estimator is the per-subject macro score on four held-out subject
sessions. Each seed therefore carries a participant-level, fixed-seed 10,000
resample bootstrap interval. Variation across the three model initializations
is reported separately as sample standard deviation. Pooled step/event scores
are retained only as secondary estimands.

No score is embedded in this file. The complete 4-experiment x 3-seed matrix,
its aggregate files, the de-identified dataset audit, and the parameter audit
must all validate before any existing final figure is replaced.

Usage
-----
    python scripts/reporting/gen_fig_experiments.py
    python scripts/reporting/gen_fig_experiments.py --check-only
    python scripts/reporting/gen_fig_experiments.py --verify-determinism
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

CODE_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = CODE_ROOT / "figures"
RESULTS_DIR = CODE_ROOT / "results"
DATA_AUDIT_PATH = RESULTS_DIR / "bci_subjects_data_audit.json"
PARAMETER_AUDIT_PATH = RESULTS_DIR / "parameter_fairness_bci_subjects.json"

EXPECTED_SEEDS = [13, 42, 73]
EXPECTED_TEST_SUBJECTS = 4
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

# Okabe--Ito: color-vision-deficiency safe. Marker shape and hatching provide
# redundant encodings for print and grayscale reproduction.
OI = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "pink": "#CC79A7",
    "black": "#000000",
    "gray": "#7A7A7A",
    "light_gray": "#D8D8D8",
    "very_light_gray": "#EFEFEF",
}

MAIN_MODELS: Mapping[str, str] = {
    "EPT-Net": "eptnet_bci_subjects_no_text",
    "Early-fusion GRU": "baseline_early_fusion_gru_bci_subjects_no_text",
    "Fusion Transformer": "baseline_fusion_transformer_bci_subjects_no_text",
}

MECHANISM_MODELS: Mapping[str, str] = {
    "Full EPT-Net": "eptnet_bci_subjects_no_text",
    "No recurrent\nfusion update": "ablation_no_recurrent_fusion_bci_subjects_no_text",
}

ALL_EXPERIMENTS = tuple(dict.fromkeys((*MAIN_MODELS.values(), *MECHANISM_MODELS.values())))

# Display name -> (subject-macro key, pooled section/key).
METRICS: Mapping[str, tuple[str, tuple[str, str]]] = {
    "AUROC": ("frame_auroc", ("frame", "auroc")),
    "Macro-F1": ("frame_macro_f1", ("frame", "macro_f1")),
    "Boundary\nF1": ("boundary_macro_f1", ("boundary", "boundary_macro_f1")),
    "Event F1\n@ IoU 0.5": ("event_f1_iou_0.5", ("event", "event_f1_iou_0.5")),
    "Threshold-free\nEvent mAP": ("event_map", ("event", "event_map")),
}

MODEL_COLORS = [OI["vermillion"], OI["sky"], OI["green"]]
MODEL_HATCHES = ["///", "...", "xx"]
SEED_MARKERS = ["o", "s", "^"]


def configure_style() -> None:
    """Apply conference-width typography and deterministic export defaults."""

    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.4,
            "axes.titlesize": 9.4,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.8,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid.axis": "y",
            "grid.color": "#B8B8B8",
            "grid.alpha": 0.24,
            "grid.linewidth": 0.55,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7.2,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required resolved configuration is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def checked_score(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Boolean is not a score for {label}")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Expected a finite unit-interval score for {label}: {value!r}")
    return score


def checked_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Boolean is not a numeric value for {label}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"Expected a finite non-negative value for {label}: {value!r}")
    return number


def nested_value(payload: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Missing {'.'.join(path)} for {label}")
        current = current[key]
    return current


def close_enough(actual: float, expected: float) -> bool:
    return bool(np.isclose(actual, expected, rtol=1e-10, atol=1e-12))


def load_bci_run(
    experiment: str,
    seed: int,
    sources: list[Path],
) -> dict[str, Any]:
    """Load one run and validate identity, protocol, provenance, and estimands."""

    run_dir = RESULTS_DIR / experiment / f"seed_{seed}"
    metrics_path = run_dir / "test_metrics.json"
    config_path = run_dir / "resolved_config.yaml"
    metrics = load_json(metrics_path)
    config = load_yaml(config_path)
    sources.extend([metrics_path, config_path])

    checkpoint = metrics.get("checkpoint", {})
    if checkpoint.get("experiment") != experiment or checkpoint.get("seed") != seed:
        raise ValueError(
            f"Run identity mismatch in {metrics_path}: expected {experiment}/seed_{seed}"
        )
    config_checks = {
        "experiment.name": (
            nested_value(config, ("experiment", "name"), str(config_path)),
            experiment,
        ),
        "experiment.seed": (nested_value(config, ("experiment", "seed"), str(config_path)), seed),
        "data.dataset_name": (
            nested_value(config, ("data", "dataset_name"), str(config_path)),
            "bci_subjects_ept_v1",
        ),
        "training.sequence_protocol": (
            nested_value(config, ("training", "sequence_protocol"), str(config_path)),
            "continuous_session",
        ),
        "training.batch_size": (
            nested_value(config, ("training", "batch_size"), str(config_path)),
            1,
        ),
        "model.use_text": (
            nested_value(config, ("model", "use_text"), str(config_path)),
            False,
        ),
        "model.use_audio": (
            nested_value(config, ("model", "use_audio"), str(config_path)),
            False,
        ),
        "evaluation.calibrate_frame_threshold": (
            nested_value(
                config,
                ("evaluation", "calibrate_frame_threshold"),
                str(config_path),
            ),
            True,
        ),
    }
    mismatches = {
        key: {"actual": actual, "expected": expected}
        for key, (actual, expected) in config_checks.items()
        if actual != expected
    }
    if mismatches:
        raise ValueError(f"BCI pilot protocol mismatch in {config_path}: {mismatches}")

    provenance = metrics.get("provenance", {})
    provenance_sha = provenance.get("provenance_sha256")
    if not is_sha256(provenance_sha) or checkpoint.get("provenance_sha256") != provenance_sha:
        raise ValueError(f"Missing or inconsistent source/data provenance in {metrics_path}")
    if not is_sha256(checkpoint.get("training_config_sha256")):
        raise ValueError(f"Missing training-config fingerprint in {metrics_path}")

    calibration = metrics.get("calibration", {})
    expected_calibration = {
        "enabled": True,
        "method": "validation_subject_macro_f1_grid",
        "selection_split": "validation",
    }
    if any(calibration.get(key) != value for key, value in expected_calibration.items()):
        raise ValueError(f"Unexpected threshold calibration protocol in {metrics_path}")
    checked_score(calibration.get("frame_threshold"), f"{experiment}/seed_{seed}:threshold")

    data = metrics.get("data", {})
    if data.get("num_sequences") != EXPECTED_TEST_SUBJECTS:
        raise ValueError(
            f"{metrics_path} evaluates {data.get('num_sequences')!r} sequences, "
            f"expected {EXPECTED_TEST_SUBJECTS} held-out subjects"
        )
    unique_steps = int(data.get("num_unique_steps", 0))
    target_steps = int(data.get("num_evaluated_target_steps", 0))
    target_events = int(data.get("num_target_events", 0))
    if not 0 < target_steps <= unique_steps or target_events <= 0:
        raise ValueError(f"Invalid target-only evaluation counts in {metrics_path}")
    if data.get("num_dense_event_proposals") != target_steps:
        raise ValueError(f"Dense event-proposal count is invalid in {metrics_path}")

    protocol = metrics.get("protocol", {})
    if "target-speaker intervals" not in str(protocol.get("target_mask_definition", "")):
        raise ValueError(f"Target-speaker mask contract is absent from {metrics_path}")
    if protocol.get("event_ap_protocol") != EVENT_AP_PROTOCOL:
        raise ValueError(f"Threshold-free event-AP contract is absent from {metrics_path}")

    fixed = metrics.get("fixed_threshold_0_5_sensitivity")
    if not isinstance(fixed, Mapping):
        raise ValueError(f"Fixed-threshold sensitivity is absent from {metrics_path}")
    for key in ("event_ap_iou_0.3", "event_ap_iou_0.5", "event_ap_iou_0.7", "event_map"):
        selected_value = checked_score(
            nested_value(metrics, ("event", key), str(metrics_path)),
            f"{experiment}/seed_{seed}:selected:{key}",
        )
        fixed_value = checked_score(
            nested_value(fixed, ("event", key), str(metrics_path)),
            f"{experiment}/seed_{seed}:fixed:{key}",
        )
        if not close_enough(selected_value, fixed_value):
            raise ValueError(f"Threshold-free {key} changes with frame threshold in {metrics_path}")
    selected_subject_map = checked_score(
        nested_value(
            metrics,
            ("subject_macro", "metrics", "event_map", "mean"),
            str(metrics_path),
        ),
        f"{experiment}/seed_{seed}:selected:subject_event_map",
    )
    fixed_subject_map = checked_score(
        nested_value(
            fixed,
            ("subject_macro", "metrics", "event_map", "mean"),
            str(metrics_path),
        ),
        f"{experiment}/seed_{seed}:fixed:subject_event_map",
    )
    if not close_enough(selected_subject_map, fixed_subject_map):
        raise ValueError(
            f"Subject-macro threshold-free mAP changes with frame threshold in {metrics_path}"
        )

    subject_macro = metrics.get("subject_macro", {})
    if (
        subject_macro.get("unit") != "held-out subject/session"
        or subject_macro.get("num_subjects") != EXPECTED_TEST_SUBJECTS
        or "10000-resample" not in str(subject_macro.get("uncertainty", ""))
    ):
        raise ValueError(f"Invalid subject-macro uncertainty contract in {metrics_path}")
    return metrics


def subject_metric_record(
    metrics: Mapping[str, Any],
    metric_key: str,
    experiment: str,
    seed: int,
) -> dict[str, Any]:
    label = f"{experiment}/seed_{seed}:subject_macro.{metric_key}"
    summary = nested_value(metrics, ("subject_macro", "metrics", metric_key), label)
    if not isinstance(summary, Mapping):
        raise TypeError(f"Expected a subject summary for {label}")
    value = checked_score(summary.get("mean"), f"{label}:mean")
    lower = checked_score(summary.get("bootstrap_ci95_lower"), f"{label}:ci_lower")
    upper = checked_score(summary.get("bootstrap_ci95_upper"), f"{label}:ci_upper")
    if lower > upper:
        raise ValueError(f"Reversed participant-bootstrap interval for {label}")
    sample_std = checked_nonnegative(summary.get("sample_std"), f"{label}:sample_std")
    n = summary.get("n")
    missing = summary.get("missing")
    if n != EXPECTED_TEST_SUBJECTS or missing != 0:
        raise ValueError(
            f"Publication figure requires all {EXPECTED_TEST_SUBJECTS} test subjects for "
            f"{label}; observed n={n!r}, missing={missing!r}"
        )
    return {
        "seed": seed,
        "mean": value,
        "participant_sample_std": sample_std,
        "bootstrap_ci95_lower": lower,
        "bootstrap_ci95_upper": upper,
        "n_subjects": n,
        "missing_subjects": missing,
    }


def pooled_metric_value(
    metrics: Mapping[str, Any],
    metric_path: tuple[str, str],
    experiment: str,
    seed: int,
) -> float:
    label = f"{experiment}/seed_{seed}:pooled.{'.'.join(metric_path)}"
    return checked_score(nested_value(metrics, metric_path, label), label)


def validate_aggregate_metric(
    aggregate: Mapping[str, Any],
    flat_key: str,
    seed_values: Sequence[float],
    experiment: str,
) -> tuple[float, float]:
    summary = nested_value(aggregate, ("aggregate", flat_key), experiment)
    if not isinstance(summary, Mapping):
        raise TypeError(f"Aggregate summary is not an object: {experiment}:{flat_key}")
    mean = checked_score(summary.get("mean"), f"{experiment}:{flat_key}:mean")
    std = checked_nonnegative(summary.get("std"), f"{experiment}:{flat_key}:std")
    if summary.get("n") != len(EXPECTED_SEEDS) or summary.get("missing") != 0:
        raise ValueError(f"Incomplete aggregate metric for {experiment}:{flat_key}")
    expected_mean = float(np.mean(seed_values))
    expected_std = float(np.std(seed_values, ddof=1))
    if not close_enough(mean, expected_mean) or not close_enough(std, expected_std):
        raise ValueError(f"Stored aggregate disagrees with seed values: {experiment}:{flat_key}")

    aggregate_runs = aggregate.get("runs", [])
    if len(aggregate_runs) != len(EXPECTED_SEEDS):
        raise ValueError(f"Aggregate does not contain exactly three runs: {experiment}")
    runs_by_seed = {run.get("seed"): run for run in aggregate_runs}
    for seed, value in zip(EXPECTED_SEEDS, seed_values, strict=True):
        stored = checked_score(
            nested_value(runs_by_seed[seed], ("metrics", flat_key), experiment),
            f"{experiment}:aggregate-run:{flat_key}:seed_{seed}",
        )
        if not close_enough(stored, value):
            raise ValueError(
                f"Aggregate run value disagrees with source metrics: "
                f"{experiment}:{flat_key}:seed_{seed}"
            )
    return mean, std


def load_experiment(experiment: str, sources: list[Path]) -> dict[str, Any]:
    """Load one exact three-seed experiment and cross-check its aggregate."""

    aggregate_path = RESULTS_DIR / experiment / "aggregate.json"
    aggregate = load_json(aggregate_path)
    sources.append(aggregate_path)
    if (
        aggregate.get("experiment") != experiment
        or aggregate.get("num_runs") != len(EXPECTED_SEEDS)
        or aggregate.get("seeds") != EXPECTED_SEEDS
    ):
        raise ValueError(f"Incomplete or misidentified three-seed aggregate: {aggregate_path}")
    if not is_sha256(aggregate.get("training_config_sha256")) or not is_sha256(
        aggregate.get("provenance_sha256")
    ):
        raise ValueError(f"Aggregate fingerprints are missing in {aggregate_path}")

    aggregate_runs = aggregate.get("runs", [])
    if len(aggregate_runs) != len(EXPECTED_SEEDS):
        raise ValueError(f"Aggregate does not contain exactly three runs: {aggregate_path}")
    runs_by_seed = {run.get("seed"): run for run in aggregate_runs}
    if set(runs_by_seed) != set(EXPECTED_SEEDS):
        raise ValueError(f"Aggregate seed records are incomplete in {aggregate_path}")
    runs: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        aggregate_run = runs_by_seed[seed]
        if (
            aggregate_run.get("experiment") != experiment
            or aggregate_run.get("provenance_sha256") != aggregate.get("provenance_sha256")
            or aggregate_run.get("training_config_sha256")
            != aggregate.get("training_config_sha256")
        ):
            raise ValueError(f"Aggregate run identity/fingerprint mismatch in {aggregate_path}")
        metrics = load_bci_run(experiment, seed, sources)
        if metrics["checkpoint"]["provenance_sha256"] != aggregate.get(
            "provenance_sha256"
        ) or metrics["checkpoint"]["training_config_sha256"] != aggregate.get(
            "training_config_sha256"
        ):
            raise ValueError(f"Aggregate/source fingerprint mismatch for {experiment}/seed_{seed}")
        runs[seed] = metrics

    required_flat_keys = {
        f"subject_macro.metrics.{subject_key}.mean" for subject_key, _ in METRICS.values()
    } | {".".join(pooled_path) for _, pooled_path in METRICS.values()}
    incomplete = set(aggregate.get("incomplete_metrics", [])) & required_flat_keys
    if incomplete:
        raise ValueError(
            f"Plotted aggregate metrics are incomplete in {aggregate_path}: {incomplete}"
        )
    return {"aggregate": aggregate, "runs": runs}


def load_matrix(sources: list[Path]) -> dict[str, dict[str, Any]]:
    matrix = {experiment: load_experiment(experiment, sources) for experiment in ALL_EXPERIMENTS}
    provenances = {record["aggregate"]["provenance_sha256"] for record in matrix.values()}
    data_contracts = {
        json.dumps(record["aggregate"].get("data"), sort_keys=True) for record in matrix.values()
    }
    protocol_contracts = {
        json.dumps(record["aggregate"].get("protocol"), sort_keys=True)
        for record in matrix.values()
    }
    calibration_contracts = {
        json.dumps(record["aggregate"].get("calibration_protocol"), sort_keys=True)
        for record in matrix.values()
    }
    if len(provenances) != 1:
        raise ValueError("Formal BCI experiments do not share one source/data provenance")
    if len(data_contracts) != 1 or len(protocol_contracts) != 1:
        raise ValueError("Formal BCI experiments do not share one test-set/evaluation contract")
    if len(calibration_contracts) != 1:
        raise ValueError("Formal BCI experiments do not share one calibration procedure")
    return matrix


def load_data_audit(sources: list[Path]) -> dict[str, Any]:
    audit = load_json(DATA_AUDIT_PATH)
    sources.append(DATA_AUDIT_PATH)
    if audit.get("schema_version") != 1 or audit.get("dataset") != "bci_subjects_ept_v1":
        raise ValueError(f"Unexpected BCI dataset audit schema: {DATA_AUDIT_PATH}")
    expected_privacy = {
        "deidentified_aggregate_only": True,
        "contains_subject_ids": False,
        "contains_source_filenames": False,
        "contains_row_level_data": False,
    }
    if audit.get("privacy_contract") != expected_privacy:
        raise ValueError("Dataset audit is not de-identified aggregate-only output")
    protocol = audit.get("protocol", {})
    protocol_checks = {
        "subject_disjoint_fixed_split": True,
        "pilot_not_cross_validation": True,
        "split_frozen_before_training": True,
        "label_aware_pretraining_stratification": True,
        "target_mask_required": True,
        "text_enabled": False,
        "audio_enabled": False,
        "split_strategy": "exhaustive_constrained_subject_stratification_v1",
        "split_seed": 42,
    }
    if any(protocol.get(key) != value for key, value in protocol_checks.items()):
        raise ValueError("Dataset audit does not describe the frozen target-only pilot protocol")
    if not is_sha256(audit.get("data_fingerprint_sha256")):
        raise ValueError("Dataset audit lacks a valid data fingerprint")
    if (
        not is_sha256(protocol.get("split_inputs_sha256"))
        or not is_sha256(protocol.get("split_assignment_sha256"))
        or int(protocol.get("split_feasible_assignments", 0)) <= 0
    ):
        raise ValueError("Dataset audit lacks a reproducible constrained-split record")

    split_sizes = {"train": 12, "val": 2, "test": 4}
    context_total = 0
    target_total = 0
    for split, expected_subjects in split_sizes.items():
        record = nested_value(audit, ("splits", split), str(DATA_AUDIT_PATH))
        if record.get("num_subject_sessions") != expected_subjects:
            raise ValueError(f"Unexpected {split} subject count in dataset audit")
        context = int(record.get("num_context_steps", 0))
        target = int(record.get("num_target_valid_steps", 0))
        labels = record.get("label_counts", {})
        positive = int(labels.get("0", -1))
        negative = int(labels.get("1", -1))
        events = int(record.get("num_deception_events", 0))
        if not 0 < target <= context or positive + negative != target or events <= 0:
            raise ValueError(f"Inconsistent target-only counts for {split} in dataset audit")
        prevalence = checked_score(record.get("deception_prevalence"), f"{split}:prevalence")
        if not close_enough(prevalence, positive / target):
            raise ValueError(f"Deception prevalence disagrees with label counts for {split}")
        context_total += context
        target_total += target
    if (
        audit.get("total_subject_sessions") != sum(split_sizes.values())
        or audit.get("total_context_steps") != context_total
        or audit.get("total_target_valid_steps") != target_total
    ):
        raise ValueError("Dataset totals disagree with split-level counts")

    modality_coverage = audit.get("modality_coverage", {})
    for modality in ("eeg_time", "eeg_spectral", "physiology", "video", "audio", "text"):
        record = modality_coverage.get(modality, {})
        sessions = int(record.get("sessions_available", -1))
        steps = int(record.get("available_steps", -1))
        if not 0 <= sessions <= audit["total_subject_sessions"] or not 0 <= steps <= context_total:
            raise ValueError(f"Invalid modality coverage for {modality}")
    if (
        modality_coverage["audio"]["sessions_available"] != 0
        or modality_coverage["text"]["sessions_available"] != audit["total_subject_sessions"]
    ):
        raise ValueError("Dataset audit does not match the frozen audio/text availability")
    return audit


def load_parameter_audit(sources: list[Path]) -> dict[str, Any]:
    audit = load_json(PARAMETER_AUDIT_PATH)
    sources.append(PARAMETER_AUDIT_PATH)
    if audit.get("reference_experiment") != MAIN_MODELS["EPT-Net"]:
        raise ValueError("Parameter audit uses an unexpected reference model")
    records = audit.get("records", [])
    by_experiment = {record.get("experiment"): record for record in records}
    if len(records) != len(ALL_EXPERIMENTS) or set(by_experiment) != set(ALL_EXPERIMENTS):
        raise ValueError("Parameter audit does not exactly cover the formal BCI matrix")
    reference_parameters = int(by_experiment[MAIN_MODELS["EPT-Net"]]["executed_parameters"])
    if reference_parameters <= 0:
        raise ValueError("Parameter audit has an invalid EPT-Net reference count")
    for label, experiment in MAIN_MODELS.items():
        record = by_experiment[experiment]
        executed = int(record.get("executed_parameters", 0))
        ratio = checked_nonnegative(
            record.get("executed_parameter_ratio_to_eptnet"),
            f"{label}:executed_parameter_ratio",
        )
        if executed <= 0 or not close_enough(ratio, executed / reference_parameters):
            raise ValueError(f"Parameter ratio is inconsistent for {label}")
        if label != "EPT-Net" and abs(ratio - 1.0) > 0.05:
            raise ValueError(f"Main baseline is not within 5% executed parameters: {label}")
    ablation = by_experiment[MECHANISM_MODELS["No recurrent\nfusion update"]]
    ablation_ratio = checked_nonnegative(
        ablation.get("executed_parameter_ratio_to_eptnet"), "mechanism-ablation ratio"
    )
    if not 0.0 < ablation_ratio < 1.0:
        raise ValueError("No-recurrent-fusion condition does not expose its capacity reduction")
    return {
        "raw": audit,
        "by_experiment": by_experiment,
        "ablation_capacity_reduction_percent": 100.0 * (1.0 - ablation_ratio),
    }


def validate_audit_against_matrix(
    audit: Mapping[str, Any], matrix: Mapping[str, Mapping[str, Any]]
) -> None:
    """Bind the de-identified audit to the prepared artifacts used at evaluation."""

    reference = matrix[MAIN_MODELS["EPT-Net"]]["runs"][EXPECTED_SEEDS[0]]
    provenance = reference.get("provenance", {})
    provenance_artifacts = provenance.get("required_artifacts", {})
    audit_artifacts = nested_value(audit, ("fingerprint_inputs", "artifacts"), "data audit")
    for key in ("dataset_summary", "feature_schema", "normalization_stats"):
        provenance_sha = nested_value(provenance_artifacts, (key, "sha256"), "provenance")
        if audit_artifacts.get(key) != provenance_sha:
            raise ValueError(f"Dataset audit/provenance artifact mismatch: {key}")

    provenance_manifests = {
        Path(record["path"]).stem: record["manifest_and_tensors_sha256"]
        for record in provenance.get("manifests", [])
    }
    audit_manifests = nested_value(
        audit, ("fingerprint_inputs", "manifests_and_tensors"), "data audit"
    )
    if provenance_manifests != audit_manifests:
        raise ValueError("Dataset audit manifest/tensor hashes do not match evaluation provenance")


def metric_bundle(experiment_record: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated primary and secondary estimates for all plotted metrics."""

    aggregate = experiment_record["aggregate"]
    runs = experiment_record["runs"]
    experiment = aggregate["experiment"]
    bundle: dict[str, Any] = {}
    for display_name, (subject_key, pooled_path) in METRICS.items():
        subject_records = [
            subject_metric_record(runs[seed], subject_key, experiment, seed)
            for seed in EXPECTED_SEEDS
        ]
        subject_values = [record["mean"] for record in subject_records]
        subject_flat_key = f"subject_macro.metrics.{subject_key}.mean"
        subject_mean, subject_std = validate_aggregate_metric(
            aggregate, subject_flat_key, subject_values, experiment
        )

        pooled_values = [
            pooled_metric_value(runs[seed], pooled_path, experiment, seed)
            for seed in EXPECTED_SEEDS
        ]
        pooled_flat_key = ".".join(pooled_path)
        pooled_mean, pooled_std = validate_aggregate_metric(
            aggregate, pooled_flat_key, pooled_values, experiment
        )
        bundle[display_name.replace("\n", " ")] = {
            "subject_macro": {
                "mean_across_seeds": subject_mean,
                "sample_std_across_seeds": subject_std,
                "seeds": subject_records,
            },
            "pooled_secondary": {
                "mean_across_seeds": pooled_mean,
                "sample_std_across_seeds": pooled_std,
                "seeds": [
                    {"seed": seed, "value": value}
                    for seed, value in zip(EXPECTED_SEEDS, pooled_values, strict=True)
                ],
            },
        }
    return bundle


def calibration_metadata(matrix: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for experiment, record in matrix.items():
        result[experiment] = {
            "selected_frame_threshold_by_seed": {
                str(seed): checked_score(
                    record["runs"][seed]["calibration"]["frame_threshold"],
                    f"{experiment}/seed_{seed}:frame_threshold",
                )
                for seed in EXPECTED_SEEDS
            },
            "test_counts": record["aggregate"]["data"],
            "training_config_sha256": record["aggregate"]["training_config_sha256"],
            "provenance_sha256": record["aggregate"]["provenance_sha256"],
        }
    return result


def save_figure(fig: mpl.figure.Figure, stem: str, output_dir: Path) -> list[Path]:
    outputs = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    fig.savefig(
        outputs[0],
        format="pdf",
        metadata={
            "Title": stem,
            "Author": "EPT-Net",
            "Subject": "Fixed subject-disjoint BCI pilot",
            "Keywords": "EPT-Net, BCI, subject-disjoint pilot",
            "Creator": "EPT-Net deterministic figure pipeline",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        outputs[1],
        format="png",
        dpi=300,
        metadata={"Software": "EPT-Net deterministic figure pipeline"},
    )
    plt.close(fig)
    for output in outputs:
        if not output.is_file() or output.stat().st_size < 10_000:
            raise RuntimeError(f"Figure export failed or is suspiciously small: {output}")
    return outputs


def _bar_seed_overlay(
    ax: mpl.axes.Axes,
    x_position: float,
    estimate: Mapping[str, Any],
    *,
    color: str,
    hatch: str,
    width: float,
    participant_intervals: bool,
) -> None:
    """Draw one across-seed bar plus independently encoded seed estimates."""

    mean = estimate["mean_across_seeds"]
    std = estimate["sample_std_across_seeds"]
    ax.bar(
        x_position,
        mean,
        width=width,
        yerr=std,
        capsize=2.3,
        color=color,
        alpha=0.66,
        edgecolor="#222222",
        linewidth=0.55,
        hatch=hatch,
        error_kw={"elinewidth": 1.15, "capthick": 1.15, "ecolor": "#222222"},
        zorder=2,
    )
    seed_records = estimate["seeds"]
    jitter = np.linspace(-width * 0.22, width * 0.22, len(seed_records))
    for index, (record, delta) in enumerate(zip(seed_records, jitter, strict=True)):
        value = record.get("mean", record.get("value"))
        if participant_intervals:
            ax.vlines(
                x_position + delta,
                record["bootstrap_ci95_lower"],
                record["bootstrap_ci95_upper"],
                color=color,
                linewidth=0.8,
                alpha=0.95,
                zorder=3,
            )
        ax.scatter(
            x_position + delta,
            value,
            s=16,
            marker=SEED_MARKERS[index],
            facecolor="white",
            edgecolor="#202020",
            linewidth=0.6,
            zorder=4,
        )


def plot_main_comparison(
    matrix: Mapping[str, Mapping[str, Any]], output_dir: Path
) -> tuple[list[Path], dict[str, Any]]:
    """Plot subject-macro primary estimates and pooled secondary estimates."""

    plotted = {
        model: metric_bundle(matrix[experiment]) for model, experiment in MAIN_MODELS.items()
    }
    metric_keys = [name.replace("\n", " ") for name in METRICS]
    x = np.arange(len(metric_keys), dtype=float)
    offsets = np.array([-0.25, 0.0, 0.25])
    width = 0.205

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.35, 5.15),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.10},
    )
    for model_index, (_model, metrics) in enumerate(plotted.items()):
        for metric_index, metric_name in enumerate(metric_keys):
            position = x[metric_index] + offsets[model_index]
            _bar_seed_overlay(
                axes[0],
                position,
                metrics[metric_name]["subject_macro"],
                color=MODEL_COLORS[model_index],
                hatch=MODEL_HATCHES[model_index],
                width=width,
                participant_intervals=True,
            )
            _bar_seed_overlay(
                axes[1],
                position,
                metrics[metric_name]["pooled_secondary"],
                color=MODEL_COLORS[model_index],
                hatch=MODEL_HATCHES[model_index],
                width=width,
                participant_intervals=False,
            )

    for ax in axes:
        ax.set_ylim(0.0, 1.04)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.set_ylabel("Score")
        ax.set_axisbelow(True)
    axes[0].set_title("(a) Primary: subject-macro test performance", loc="left")
    axes[1].set_title("(b) Secondary: pooled target-step/event performance", loc="left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(METRICS.keys())

    model_handles = [
        Patch(
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.55,
            alpha=0.66,
            hatch=hatch,
            label=model,
        )
        for model, color, hatch in zip(MAIN_MODELS, MODEL_COLORS, MODEL_HATCHES, strict=True)
    ]
    seed_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=marker,
            markerfacecolor="white",
            markeredgecolor="#202020",
            markersize=4.2,
            label=f"seed {seed}",
        )
        for seed, marker in zip(EXPECTED_SEEDS, SEED_MARKERS, strict=True)
    ]
    fig.legend(
        handles=[*model_handles, *seed_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=6,
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.suptitle(
        "Fixed subject-disjoint pilot: 4 held-out subjects × 3 initialization seeds",
        y=0.995,
        fontsize=10.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.008,
        "Bars: mean over seeds; thick black whiskers: seed sample SD. "
        "Panel (a) thin colored whiskers: per-seed 95% bootstrap CI over all n=4 test subjects. "
        "Single fixed split; not cross-validation.",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.115)
    return save_figure(fig, "fig_main_comparison", output_dir), plotted


def plot_mechanism_ablation(
    matrix: Mapping[str, Mapping[str, Any]],
    parameter_audit: Mapping[str, Any],
    output_dir: Path,
) -> tuple[list[Path], dict[str, Any]]:
    """Plot the repeated fusion-update diagnostic and paired-seed effects."""

    absolute = {
        variant: metric_bundle(matrix[experiment])
        for variant, experiment in MECHANISM_MODELS.items()
    }
    metric_keys = [name.replace("\n", " ") for name in METRICS]
    x = np.arange(len(metric_keys), dtype=float)
    variant_offsets = [-0.18, 0.18]
    variant_colors = [OI["vermillion"], OI["blue"]]
    width = 0.30

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.65),
        gridspec_kw={"width_ratios": [1.16, 1.0]},
    )

    ax = axes[0]
    for variant_index, (_variant, metrics) in enumerate(absolute.items()):
        for metric_index, metric_name in enumerate(metric_keys):
            _bar_seed_overlay(
                ax,
                x[metric_index] + variant_offsets[variant_index],
                metrics[metric_name]["subject_macro"],
                color=variant_colors[variant_index],
                hatch=MODEL_HATCHES[variant_index],
                width=width,
                participant_intervals=True,
            )
    ax.set_ylim(0.0, 1.04)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS.keys(), rotation=8, ha="right")
    ax.set_ylabel("Subject-macro score")
    ax.set_title("(a) Absolute scores with participant uncertainty", loc="left")
    ax.legend(
        handles=[
            Patch(
                facecolor=color,
                edgecolor="#222222",
                alpha=0.66,
                hatch=hatch,
                label=variant.replace("\n", " "),
            )
            for variant, color, hatch in zip(
                MECHANISM_MODELS, variant_colors, MODEL_HATCHES, strict=True
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=1,
    )

    full = absolute["Full EPT-Net"]
    ablated = absolute["No recurrent\nfusion update"]
    deltas: dict[str, Any] = {}
    all_deltas: list[float] = []
    ax = axes[1]
    seed_jitter = np.array([-0.075, 0.0, 0.075])
    for metric_index, metric_name in enumerate(metric_keys):
        subject_full = [record["mean"] for record in full[metric_name]["subject_macro"]["seeds"]]
        subject_ablated = [
            record["mean"] for record in ablated[metric_name]["subject_macro"]["seeds"]
        ]
        pooled_full = [record["value"] for record in full[metric_name]["pooled_secondary"]["seeds"]]
        pooled_ablated = [
            record["value"] for record in ablated[metric_name]["pooled_secondary"]["seeds"]
        ]
        subject_delta = np.asarray(subject_full) - np.asarray(subject_ablated)
        pooled_delta = np.asarray(pooled_full) - np.asarray(pooled_ablated)
        all_deltas.extend(subject_delta.tolist())
        all_deltas.extend(pooled_delta.tolist())
        for seed_index, delta_x in enumerate(seed_jitter):
            ax.plot(
                [x[metric_index] + delta_x - 0.018, x[metric_index] + delta_x + 0.018],
                [subject_delta[seed_index], pooled_delta[seed_index]],
                color="#B0B0B0",
                linewidth=0.55,
                zorder=1,
            )
            ax.scatter(
                x[metric_index] + delta_x - 0.018,
                subject_delta[seed_index],
                s=22,
                marker=SEED_MARKERS[seed_index],
                color=OI["vermillion"],
                edgecolor="#222222",
                linewidth=0.45,
                zorder=3,
            )
            ax.scatter(
                x[metric_index] + delta_x + 0.018,
                pooled_delta[seed_index],
                s=22,
                marker=SEED_MARKERS[seed_index],
                facecolor="white",
                edgecolor=OI["blue"],
                linewidth=0.9,
                zorder=3,
            )
        subject_mean = float(subject_delta.mean())
        subject_std = float(subject_delta.std(ddof=1))
        pooled_mean = float(pooled_delta.mean())
        pooled_std = float(pooled_delta.std(ddof=1))
        ax.errorbar(
            x[metric_index] - 0.115,
            subject_mean,
            yerr=subject_std,
            marker="D",
            markersize=4.5,
            color=OI["vermillion"],
            ecolor=OI["vermillion"],
            elinewidth=1.35,
            capsize=2.4,
            zorder=4,
        )
        ax.errorbar(
            x[metric_index] + 0.115,
            pooled_mean,
            yerr=pooled_std,
            marker="D",
            markerfacecolor="white",
            markeredgecolor=OI["blue"],
            markersize=4.5,
            color=OI["blue"],
            ecolor=OI["blue"],
            elinewidth=1.35,
            capsize=2.4,
            zorder=4,
        )
        deltas[metric_name] = {
            "definition": "full_ept_minus_no_recurrent_fusion_update",
            "subject_macro_primary": {
                "mean": subject_mean,
                "sample_std": subject_std,
                "by_seed": dict(zip(map(str, EXPECTED_SEEDS), subject_delta.tolist(), strict=True)),
            },
            "pooled_secondary": {
                "mean": pooled_mean,
                "sample_std": pooled_std,
                "by_seed": dict(zip(map(str, EXPECTED_SEEDS), pooled_delta.tolist(), strict=True)),
            },
        }

    limit = max(0.1, math.ceil((max(abs(value) for value in all_deltas) + 0.04) * 10) / 10)
    ax.axhline(0.0, color="#333333", linewidth=0.9, zorder=0)
    ax.set_ylim(-limit, limit)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS.keys(), rotation=8, ha="right")
    ax.set_ylabel(r"Paired effect $\Delta$ (full $-$ no update)")
    ax.set_title("(b) Paired-seed effects", loc="left")
    ax.text(
        0.02,
        0.98,
        "positive favors recurrent fusion update",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color="#4A4A4A",
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                color=OI["vermillion"],
                label="subject macro (primary)",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor=OI["blue"],
                color=OI["blue"],
                label="pooled (secondary)",
            ),
        ],
        loc="lower left",
        fontsize=6.5,
    )

    reduction = parameter_audit["ablation_capacity_reduction_percent"]
    fig.suptitle(
        "Recurrent-fusion update diagnostic (three paired initialization seeds)",
        y=0.99,
        fontsize=10.3,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.008,
        f"Mechanism diagnostic only: the no-update condition still gives the reader its prior "
        f"state and executes {reduction:.1f}% fewer parameters. "
        "Panel (a) thin whiskers are per-seed n=4 subject-bootstrap 95% CIs; "
        "panel (b) diamonds show mean ± seed sample SD.",
        ha="center",
        va="bottom",
        fontsize=6.25,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.89, bottom=0.20, wspace=0.24)
    plotted = {
        "absolute_subject_macro": absolute,
        "paired_effects": deltas,
        "capacity_context": {
            "executed_parameter_reduction_percent": reduction,
            "reader_still_receives_previous_state": True,
            "causal_mechanism_claim_supported": False,
        },
    }
    return save_figure(fig, "fig_mechanism_ablation", output_dir), plotted


def plot_data_integrity(
    audit: Mapping[str, Any], output_dir: Path
) -> tuple[list[Path], dict[str, Any]]:
    """Plot only de-identified aggregate properties of the 18-subject dataset."""

    split_keys = ["train", "val", "test"]
    split_labels = ["Train", "Validation", "Test"]
    splits = audit["splits"]
    context = np.asarray([splits[key]["num_context_steps"] for key in split_keys], dtype=float)
    target = np.asarray([splits[key]["num_target_valid_steps"] for key in split_keys], dtype=float)
    target_percent = 100.0 * target / context

    fig, axes = plt.subplots(1, 3, figsize=(7.45, 3.25))

    ax = axes[0]
    y = np.arange(len(split_keys))
    ax.barh(y, target_percent, color=OI["green"], height=0.58, label="target-valid")
    ax.barh(
        y,
        100.0 - target_percent,
        left=target_percent,
        color=OI["light_gray"],
        height=0.58,
        label="context only",
    )
    for index, percentage in enumerate(target_percent):
        ax.text(
            percentage / 2,
            index,
            f"{int(target[index]):,}/{int(context[index]):,}\n({percentage:.1f}%)",
            ha="center",
            va="center",
            fontsize=6.5,
            color="white" if percentage > 35 else "#222222",
            fontweight="bold",
        )
    ax.set_yticks(y)
    ax.set_yticklabels(split_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of interaction timeline (%)")
    ax.set_title("(a) Target-decision coverage", loc="left")
    ax.legend(loc="lower right", fontsize=6.2)

    ax = axes[1]
    modality_keys = ["eeg_time", "eeg_spectral", "physiology", "video", "text", "audio"]
    modality_labels = [
        "EEG time",
        "EEG spectral",
        "Physiology",
        "Video",
        "Text (disabled)",
        "Audio (disabled)",
    ]
    modality_counts = [
        int(audit["modality_coverage"][key]["sessions_available"]) for key in modality_keys
    ]
    modality_colors = [
        OI["sky"],
        OI["blue"],
        OI["green"],
        OI["orange"],
        OI["gray"],
        OI["light_gray"],
    ]
    y = np.arange(len(modality_keys))
    bars = ax.barh(
        y,
        modality_counts,
        color=modality_colors,
        edgecolor="#333333",
        linewidth=0.45,
        height=0.58,
    )
    for bar, count in zip(bars, modality_counts, strict=True):
        ax.text(
            min(count + 0.45, 17.4),
            bar.get_y() + bar.get_height() / 2,
            f"{count}/18",
            ha="left" if count < 17 else "right",
            va="center",
            fontsize=6.6,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(modality_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 18)
    ax.set_xticks([0, 6, 12, 18])
    ax.set_xlabel("Subjects with any available data")
    ax.set_title("(b) Modality availability", loc="left")

    ax = axes[2]
    positive = np.asarray(
        [int(splits[key]["label_counts"]["0"]) for key in split_keys], dtype=float
    )
    negative = target - positive
    positive_percent = 100.0 * positive / target
    negative_percent = 100.0 - positive_percent
    x = np.arange(len(split_keys), dtype=float)
    ax.bar(
        x,
        positive_percent,
        width=0.62,
        color=OI["vermillion"],
        edgecolor="#333333",
        linewidth=0.45,
        label="deception",
    )
    ax.bar(
        x,
        negative_percent,
        width=0.62,
        bottom=positive_percent,
        color=OI["very_light_gray"],
        edgecolor="#333333",
        linewidth=0.45,
        label="other target speech",
    )
    for index, percentage in enumerate(positive_percent):
        events = int(splits[split_keys[index]]["num_deception_events"])
        ax.text(
            index,
            percentage / 2,
            f"{percentage:.1f}%",
            ha="center",
            va="center",
            fontsize=7.0,
            color="white",
            fontweight="bold",
        )
        ax.text(
            index,
            103,
            f"{events} events",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color="#333333",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\n(n={int(target[index]):,} valid)" for index, label in enumerate(split_labels)]
    )
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Target-valid labels (%)")
    ax.set_title("(c) Label mix and event support", loc="left")
    ax.legend(loc="center right", fontsize=6.1)

    fig.suptitle(
        "Dataset integrity: 18-subject fixed-split pilot (12 train / 2 validation / 4 test)",
        y=0.99,
        fontsize=10.3,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.008,
        "Split frozen before training via label/modality-aware stratification. "
        "Labels and metrics are target-speaker-only; missing modalities use explicit masks. "
        "One fixed split and approximate cross-stream alignment limit generalization claims.",
        ha="center",
        va="bottom",
        fontsize=6.25,
        color="#4A4A4A",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.87, bottom=0.19, wspace=0.44)

    plotted = {
        "split_target_coverage": {
            split: {
                "subjects": splits[split]["num_subject_sessions"],
                "context_steps": splits[split]["num_context_steps"],
                "target_valid_steps": splits[split]["num_target_valid_steps"],
                "target_valid_percent": float(target_percent[index]),
            }
            for index, split in enumerate(split_keys)
        },
        "modality_subject_coverage": dict(zip(modality_keys, modality_counts, strict=True)),
        "target_only_label_support": {
            split: {
                "deception_steps": int(positive[index]),
                "other_target_steps": int(negative[index]),
                "deception_prevalence": float(positive_percent[index] / 100.0),
                "deception_events": splits[split]["num_deception_events"],
            }
            for index, split in enumerate(split_keys)
        },
        "data_fingerprint_sha256": audit["data_fingerprint_sha256"],
        "limitations": audit["limitations"],
    }
    return save_figure(fig, "fig_data_integrity_audit", output_dir), plotted


def write_manifest(
    sources: Iterable[Path],
    outputs: Iterable[Path],
    plotted_data: Mapping[str, Any],
    matrix: Mapping[str, Mapping[str, Any]],
    parameter_audit: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    unique_sources = sorted(set(path.resolve() for path in sources))
    unique_outputs = sorted(set(path.resolve() for path in outputs))
    manifest = {
        "schema_version": 4,
        "artifact_type": "publication_figure_bundle",
        "generator": {
            "path": Path(__file__).resolve().relative_to(CODE_ROOT).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "study_design": {
            "dataset": "bci_subjects_ept_v1",
            "split": {"train_subjects": 12, "validation_subjects": 2, "test_subjects": 4},
            "subject_disjoint": True,
            "fixed_split_pilot_not_cross_validation": True,
            "model_initialization_seeds": EXPECTED_SEEDS,
            "primary_estimand": (
                "per-seed mean across four held-out subjects, summarized across three "
                "model-initialization seeds"
            ),
            "participant_uncertainty": (
                "per-seed fixed-seed 10000-resample percentile bootstrap across n=4 "
                "held-out subjects"
            ),
            "initialization_uncertainty": "sample standard deviation across three seeds",
            "secondary_estimand": "pooled target-step/event metric",
            "frame_threshold": "selected on validation subjects independently for each seed",
            "event_f1": "thresholded causal decoder at the selected operating point",
            "event_map": {
                "threshold_free": True,
                **EVENT_AP_PROTOCOL,
            },
            "text_enabled": False,
            "audio_enabled": False,
            "target_speaker_mask_required": True,
        },
        "interpretation_guardrails": {
            "pilot_only": True,
            "population_level_claim_supported": False,
            "mechanism_ablation_capacity_matched": False,
            "mechanism_ablation_executed_parameter_reduction_percent": parameter_audit[
                "ablation_capacity_reduction_percent"
            ],
            "no_recurrent_fusion_reader_still_receives_previous_state": True,
            "pooled_results_are_secondary": True,
        },
        "plotting_contract": {
            "palette": "Okabe-Ito with redundant marker/hatch encodings",
            "exports": ["vector PDF", "300 dpi PNG"],
            "pdf_dates_removed": True,
            "no_generation_timestamp": True,
            "missing_or_inconsistent_input_behavior": "fail closed before replacing final files",
        },
        "experiment_metadata": calibration_metadata(matrix),
        "sources": [
            {
                "path": path.relative_to(FIGURE_DIR.parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in unique_sources
        ],
        "outputs": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in unique_outputs
        ],
        "plotted_data": plotted_data,
    }
    manifest_path = output_dir / "figure_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return manifest_path


def expected_input_paths() -> list[Path]:
    paths = [DATA_AUDIT_PATH, PARAMETER_AUDIT_PATH]
    for experiment in ALL_EXPERIMENTS:
        paths.append(RESULTS_DIR / experiment / "aggregate.json")
        for seed in EXPECTED_SEEDS:
            run_dir = RESULTS_DIR / experiment / f"seed_{seed}"
            paths.extend([run_dir / "test_metrics.json", run_dir / "resolved_config.yaml"])
    return sorted(set(paths))


def preflight_inputs() -> None:
    missing = [path for path in expected_input_paths() if not path.is_file()]
    if not missing:
        return
    formatted = "\n".join(
        f"  - {path.relative_to(FIGURE_DIR.parent).as_posix()}" for path in missing
    )
    raise FileNotFoundError(
        "The formal BCI 4-experiment x 3-seed matrix is incomplete. Existing final "
        f"figures were not modified. Missing artifacts:\n{formatted}"
    )


def render_bundle(
    output_dir: Path,
    *,
    sources: Sequence[Path],
    matrix: Mapping[str, Mapping[str, Any]],
    data_audit: Mapping[str, Any],
    parameter_audit: Mapping[str, Any],
) -> list[Path]:
    plotted_data: dict[str, Any] = {}
    outputs: list[Path] = []

    produced, plotted_data["main_comparison"] = plot_main_comparison(matrix, output_dir)
    outputs.extend(produced)
    produced, plotted_data["mechanism_ablation"] = plot_mechanism_ablation(
        matrix, parameter_audit, output_dir
    )
    outputs.extend(produced)
    produced, plotted_data["data_integrity_audit"] = plot_data_integrity(data_audit, output_dir)
    outputs.extend(produced)
    manifest = write_manifest(
        sources,
        outputs,
        plotted_data,
        matrix,
        parameter_audit,
        output_dir,
    )
    return [*outputs, manifest]


def assert_identical_bundles(first: Sequence[Path], second: Sequence[Path]) -> None:
    first_by_name = {path.name: path for path in first}
    second_by_name = {path.name: path for path in second}
    if set(first_by_name) != set(second_by_name):
        raise RuntimeError("Determinism check produced different file sets")
    mismatches = [
        name
        for name in sorted(first_by_name)
        if sha256(first_by_name[name]) != sha256(second_by_name[name])
    ]
    if mismatches:
        raise RuntimeError(f"Non-deterministic figure artifacts: {mismatches}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate validated BCI-subject pilot publication figures"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate all required inputs without rendering or replacing figures",
    )
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="render twice and require byte-identical PDFs, PNGs, and manifest before publish",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight_inputs()
    sources: list[Path] = []
    matrix = load_matrix(sources)
    data_audit = load_data_audit(sources)
    parameter_audit = load_parameter_audit(sources)
    validate_audit_against_matrix(data_audit, matrix)

    if args.check_only:
        print(
            "PASS: complete BCI matrix, subject-macro/pooled estimands, provenance, "
            "dataset audit, and parameter audit are internally consistent."
        )
        return

    configure_style()
    with tempfile.TemporaryDirectory(prefix=".figure-build-a-", dir=FIGURE_DIR) as first_tmp:
        first_dir = Path(first_tmp)
        first_bundle = render_bundle(
            first_dir,
            sources=sources,
            matrix=matrix,
            data_audit=data_audit,
            parameter_audit=parameter_audit,
        )
        if args.verify_determinism:
            with tempfile.TemporaryDirectory(
                prefix=".figure-build-b-", dir=FIGURE_DIR
            ) as second_tmp:
                second_bundle = render_bundle(
                    Path(second_tmp),
                    sources=sources,
                    matrix=matrix,
                    data_audit=data_audit,
                    parameter_audit=parameter_audit,
                )
                assert_identical_bundles(first_bundle, second_bundle)

        # All validation and optional byte-for-byte reproducibility checks have
        # succeeded. Only now may the last complete published bundle be replaced.
        published: list[Path] = []
        for staged_path in first_bundle:
            final_path = FIGURE_DIR / staged_path.name
            os.replace(staged_path, final_path)
            published.append(final_path)

    print(f"Generated {len(published) - 1} figure files and figure_manifest.json")
    if args.verify_determinism:
        print("PASS: two independent staged renders were byte-identical")
    for path in published:
        print(f"  {path.name}: {path.stat().st_size:,} bytes; sha256={sha256(path)}")


if __name__ == "__main__":
    main()
