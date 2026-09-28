#!/usr/bin/env python3
"""Audit entry point for the prepared, de-identified multi-subject BCI dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import Any

import torch

from eptnet.config import load_config
from eptnet.data import StitchedManifestDataset
from eptnet.provenance import manifest_fingerprint, sha256_file

SPLITS = ("train", "val", "test")
PHYSIOLOGY_MODALITIES = ("eeg_time", "eeg_spectral", "physiology")
BEHAVIOR_MODALITIES = ("video", "audio", "text")
TARGET_MASK_SCHEMA = (
    "target identity is frozen from the complete transcript by maximum total "
    "non-whitespace characters before annotation marks are read; true only when "
    "that speaker has a strict majority of characters assigned to the timestep"
)
TARGET_SPEAKER_INFERENCE = "maximum_total_non_whitespace_transcript_characters"


def _event_lengths(
    labels: torch.Tensor, positive_class: int, target_mask: torch.Tensor
) -> list[int]:
    lengths: list[int] = []
    current = 0
    for label, valid in zip(labels.tolist(), target_mask.bool().tolist(), strict=True):
        if valid and int(label) == positive_class:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _availability(
    sample: Mapping[str, Any], key: str, names: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    mask = sample.get(key)
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.shape[1] != len(names):
        raise ValueError(f"{key} must have shape [T, {len(names)}]")
    return {
        name: {
            "available_steps": int(mask[:, index].bool().sum().item()),
            "session_available": int(bool(mask[:, index].bool().any().item())),
        }
        for index, name in enumerate(names)
    }


def build_audit(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    data_config = config["data"]
    positive_class = int(config["evaluation"]["positive_class"])
    split_payloads: dict[str, Any] = {}
    seen_sessions: dict[str, set[str]] = {}
    aggregate_availability = {
        name: {"sessions_available": 0, "available_steps": 0}
        for name in (*PHYSIOLOGY_MODALITIES, *BEHAVIOR_MODALITIES)
    }
    total_sessions = 0
    total_steps = 0
    total_target_steps = 0
    total_deception_steps = 0
    total_deception_events = 0

    manifest_paths: dict[str, Path] = {}
    dataset_root: Path | None = None
    for split in SPLITS:
        manifest = Path(str(data_config[f"{split}_manifest"])).resolve()
        manifest_paths[split] = manifest
        current_root = manifest.parent.parent
        if dataset_root is None:
            dataset_root = current_root
        elif current_root != dataset_root:
            raise ValueError("All split manifests must belong to one prepared dataset")

        dataset = StitchedManifestDataset(str(manifest), require_targets=True)
        dataset.materialize()
        session_ids: set[str] = set()
        label_counts = {"0": 0, "1": 0}
        event_lengths: list[int] = []
        split_steps = 0
        split_target_steps = 0
        split_modality_sessions = {
            name: 0 for name in (*PHYSIOLOGY_MODALITIES, *BEHAVIOR_MODALITIES)
        }

        for index in range(len(dataset)):
            sample = dataset[index]
            metadata = sample.get("metadata", {})
            session_id = str(metadata.get("session_id", "")).strip()
            if not session_id or session_id in session_ids:
                raise ValueError(f"Invalid or duplicate session identity in {split}")
            session_ids.add(session_id)
            labels = sample["labels"].long()
            target_mask = sample.get("target_mask")
            if not isinstance(target_mask, torch.Tensor):
                raise ValueError(f"Prepared {split} session is missing target_mask")
            target_mask = target_mask.bool()
            if target_mask.shape != labels.shape or not bool(target_mask.any().item()):
                raise ValueError(f"Prepared {split} session has an invalid target_mask")
            split_steps += int(labels.numel())
            split_target_steps += int(target_mask.sum().item())
            for label in (0, 1):
                label_counts[str(label)] += int(((labels == label) & target_mask).sum().item())
            event_lengths.extend(_event_lengths(labels, positive_class, target_mask))

            availability = {
                **_availability(sample, "physiology_mask", PHYSIOLOGY_MODALITIES),
                **_availability(sample, "modality_mask", BEHAVIOR_MODALITIES),
            }
            for name, values in availability.items():
                aggregate_availability[name]["sessions_available"] += values["session_available"]
                aggregate_availability[name]["available_steps"] += values["available_steps"]
                split_modality_sessions[name] += values["session_available"]

        seen_sessions[split] = session_ids
        total_sessions += len(session_ids)
        total_steps += split_steps
        total_target_steps += split_target_steps
        total_deception_steps += label_counts[str(positive_class)]
        total_deception_events += len(event_lengths)
        split_payloads[split] = {
            "num_subject_sessions": len(session_ids),
            "num_context_steps": split_steps,
            "num_target_valid_steps": split_target_steps,
            "label_counts": label_counts,
            "deception_prevalence": label_counts[str(positive_class)] / split_target_steps,
            "num_deception_events": len(event_lengths),
            "modality_session_counts": split_modality_sessions,
            "event_length_steps": {
                "minimum": min(event_lengths),
                "median": float(median(event_lengths)),
                "maximum": max(event_lengths),
                "single_step_events": sum(length == 1 for length in event_lengths),
            },
        }

    assert dataset_root is not None
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlap = seen_sessions[left] & seen_sessions[right]
            if overlap:
                raise ValueError(f"Subject/session leakage between {left} and {right}")

    feature_schema_path = dataset_root / "feature_schema.json"
    feature_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    if not isinstance(feature_schema, Mapping):
        raise ValueError("feature_schema.json must contain an object")
    if feature_schema.get("target_mask") != TARGET_MASK_SCHEMA:
        raise ValueError("Prepared data uses an unexpected target-speaker inference rule")

    alignment_mode = str(feature_schema.get("alignment_mode", "duration_normalized"))
    alignment_report_path = dataset_root / "alignment_report.json"
    alignment_report = json.loads(alignment_report_path.read_text(encoding="utf-8"))
    if not isinstance(alignment_report, Mapping):
        raise ValueError("alignment_report.json must contain an object")
    source_manifest_path = dataset_root / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_manifest, Mapping):
        raise ValueError("source_manifest.json must contain an object")

    source_policy = feature_schema.get("source_inclusion_policy")
    if not isinstance(source_policy, Mapping):
        raise ValueError("feature_schema.json is missing source_inclusion_policy")
    facial_csv_included = bool(source_policy.get("include_facial_csv"))
    facial_source_records = sum(
        1
        for session in source_manifest.get("sessions", [])
        if isinstance(session, Mapping)
        for record in session.get("files", [])
        if isinstance(record, Mapping)
        and "fac" in str(record.get("role", "")).lower()
        and str(record.get("role", "")).lower() not in {"interface", "artifact"}
    )
    if not facial_csv_included and facial_source_records:
        raise ValueError("Facial CSV is excluded by policy but remains in source_manifest.json")

    alignment_aggregate = alignment_report.get("aggregate")
    if not isinstance(alignment_aggregate, Mapping):
        raise ValueError("alignment_report.json is missing aggregate coverage")
    if int(alignment_aggregate.get("facial_source_records", -1)) != facial_source_records:
        raise ValueError("Facial source count disagrees between alignment and source manifests")
    if alignment_mode == "absolute_time":
        if alignment_report.get("alignment_scope") != "absolute_timestamp_intersection":
            raise ValueError("Absolute-time data lacks timestamp-intersection provenance")
        if alignment_report.get("physical_clock_synchronization") is not False:
            raise ValueError("Physical clock synchronization must be stated explicitly")
        acceptance = alignment_report.get("acceptance_criteria")
        if not isinstance(acceptance, Mapping) or not acceptance.get(
            "session_024_eeg_rejected"
        ):
            raise ValueError("Known non-overlapping EEG control was not rejected")
        sessions = alignment_report.get("sessions")
        if not isinstance(sessions, Mapping):
            raise ValueError("alignment_report.json is missing session evidence")
        session_024 = sessions.get("session_024")
        if not isinstance(session_024, Mapping):
            raise ValueError("Known non-overlapping EEG control is missing")
        session_024_eeg = session_024.get("eeg")
        if not isinstance(session_024_eeg, Mapping) or (
            session_024_eeg.get("status") != "no_overlap"
            or int(session_024_eeg.get("valid_steps", -1)) != 0
        ):
            raise ValueError("Known non-overlapping EEG control leaked into valid features")
    elif alignment_mode == "session_registered_time":
        if alignment_report.get("alignment_scope") != (
            "session_registered_timestamp_intersection"
        ):
            raise ValueError("Session-registered data lacks clock-registration provenance")
        if alignment_report.get("physical_clock_synchronization") is not False:
            raise ValueError("Clock registration must not claim physical synchronization")
        acceptance = alignment_report.get("acceptance_criteria")
        if not isinstance(acceptance, Mapping) or not acceptance.get(
            "session_024_eeg_registered"
        ):
            raise ValueError("Confirmed clock-mismatched EEG control was not registered")
        sessions = alignment_report.get("sessions")
        if not isinstance(sessions, Mapping):
            raise ValueError("alignment_report.json is missing session evidence")
        session_024 = sessions.get("session_024")
        if not isinstance(session_024, Mapping):
            raise ValueError("Confirmed clock-mismatched EEG control is missing")
        session_024_eeg = session_024.get("eeg")
        if not isinstance(session_024_eeg, Mapping) or (
            session_024_eeg.get("status") != "aligned_end_anchored_clock_offset"
            or not session_024_eeg.get("clock_offset_applied")
            or int(session_024_eeg.get("valid_steps", 0)) <= 0
        ):
            raise ValueError("Confirmed clock-mismatched EEG was not registered safely")

    split_strategy_path = dataset_root / "split_strategy.json"
    split_strategy = json.loads(split_strategy_path.read_text(encoding="utf-8"))
    if not isinstance(split_strategy, Mapping):
        raise ValueError("split_strategy.json must contain an object")
    supported_split_algorithms = {
        "exhaustive_constrained_subject_stratification_v1",
        "fixed_seed_constrained_monte_carlo_stratification_v2",
    }
    if split_strategy.get("algorithm") not in supported_split_algorithms:
        raise ValueError("Prepared data uses an unexpected split strategy")
    if not split_strategy.get("model_independent") or not split_strategy.get(
        "frozen_before_training"
    ):
        raise ValueError("The subject split is not declared model-independent and frozen")
    declared_splits = split_strategy.get("split_sessions")
    if not isinstance(declared_splits, Mapping):
        raise ValueError("split_strategy.json is missing split_sessions")
    for split in SPLITS:
        if set(declared_splits.get(split, [])) != seen_sessions[split]:
            raise ValueError(f"Prepared {split} manifest disagrees with split_strategy.json")
    declared_quotas = split_strategy.get("modality_quotas")
    if not isinstance(declared_quotas, Mapping):
        raise ValueError("split_strategy.json is missing modality quotas")
    for modality, mask_name in {
        "eeg": "eeg_time",
        "physiology": "physiology",
        "video": "video",
    }.items():
        quota = declared_quotas.get(modality)
        if not isinstance(quota, Mapping):
            raise ValueError(f"Missing split quota for {modality}")
        for split in SPLITS:
            observed = split_payloads[split]["modality_session_counts"][mask_name]
            if quota.get(split) != observed:
                raise ValueError(f"Observed {modality} coverage violates the frozen quota")
    declared_totals = split_strategy.get("balance_totals")
    expected_totals = {
        "target_valid_steps": total_target_steps,
        "deception_steps": total_deception_steps,
        "deception_events": total_deception_events,
    }
    if declared_totals != expected_totals:
        raise ValueError("Prepared target totals disagree with split_strategy.json")

    artifact_hashes = {
        name: sha256_file(dataset_root / filename)
        for name, filename in {
            "dataset_summary": "dataset_summary.json",
            "feature_schema": "feature_schema.json",
            "alignment_report": "alignment_report.json",
            "normalization_stats": "normalization_stats.npz",
            "source_manifest": "source_manifest.json",
            "split_strategy": "split_strategy.json",
        }.items()
    }
    manifest_hashes = {
        split: manifest_fingerprint(path, artifact_root=dataset_root)["manifest_and_tensors_sha256"]
        for split, path in manifest_paths.items()
    }
    fingerprint_inputs = {
        "artifacts": artifact_hashes,
        "manifests_and_tensors": manifest_hashes,
    }

    return {
        "schema_version": 1,
        "dataset": str(data_config["dataset_name"]),
        "privacy_contract": {
            "deidentified_aggregate_only": True,
            "contains_subject_ids": False,
            "contains_source_filenames": False,
            "contains_row_level_data": False,
        },
        "protocol": {
            "subject_disjoint_fixed_split": True,
            "pilot_not_cross_validation": True,
            "split_strategy": split_strategy["algorithm"],
            "split_frozen_before_training": True,
            "label_aware_pretraining_stratification": bool(
                split_strategy.get("uses_ground_truth_for_stratification")
            ),
            "split_seed": split_strategy["seed_used_only_for_exact_score_ties"],
            "split_objective_value": split_strategy["objective_value"],
            "split_feasible_assignments": split_strategy["feasible_assignments"],
            "split_inputs_sha256": split_strategy["stratification_inputs_sha256"],
            "split_assignment_sha256": split_strategy["assignment_sha256"],
            "positive_class": positive_class,
            "text_enabled": bool(config["model"]["use_text"]),
            "audio_enabled": bool(config["model"]["use_audio"]),
            "target_mask_required": True,
            "target_speaker_inference": TARGET_SPEAKER_INFERENCE,
            "target_speaker_inference_uses_annotation_marks": False,
            "target_speaker_inference_scope": "retrospective_complete_transcript",
            "alignment_mode": alignment_mode,
            "alignment_scope": alignment_report.get("alignment_scope"),
            "physical_clock_synchronization": bool(
                alignment_report.get("physical_clock_synchronization")
            ),
            "facial_csv_included": facial_csv_included,
        },
        "total_subject_sessions": total_sessions,
        "total_context_steps": total_steps,
        "total_target_valid_steps": total_target_steps,
        "splits": split_payloads,
        "modality_coverage": aggregate_availability,
        "alignment": {
            "scope": alignment_report.get("alignment_scope"),
            "physical_clock_synchronization": bool(
                alignment_report.get("physical_clock_synchronization")
            ),
            "eeg_sessions_present": int(
                alignment_aggregate.get("eeg_sessions_present", 0)
            ),
            "eeg_sessions_with_valid_overlap": int(
                alignment_aggregate.get("eeg_sessions_with_overlap", 0)
            ),
            "eeg_valid_steps": int(alignment_aggregate.get("eeg_valid_steps", 0)),
            "physiology_sessions_present": int(
                alignment_aggregate.get("physiology_sessions_present", 0)
            ),
            "physiology_sessions_with_valid_overlap": int(
                alignment_aggregate.get("physiology_sessions_with_overlap", 0)
            ),
            "physiology_valid_steps": int(
                alignment_aggregate.get("physiology_valid_steps", 0)
            ),
            "facial_source_records": facial_source_records,
            "eeg_clock_registered_sessions": int(
                alignment_aggregate.get("eeg_clock_registered_sessions", 0)
            ),
            "known_nonoverlap_control_rejected": bool(
                alignment_report.get("acceptance_criteria", {}).get(
                    "session_024_eeg_rejected"
                )
            ),
            "known_clock_mismatch_registered": bool(
                alignment_report.get("acceptance_criteria", {}).get(
                    "session_024_eeg_registered"
                )
            ),
        },
        "data_fingerprint_sha256": _canonical_sha256(fingerprint_inputs),
        "fingerprint_inputs": fingerprint_inputs,
        "limitations": [
            "The fixed split uses labels and modality availability for pre-training stratification.",
            "One fixed "
            + "/".join(str(len(seen_sessions[split])) for split in SPLITS)
            + " split is a pilot protocol, not cross-validation or population-level validation.",
            "Speaker-turn timestamps require within-turn character interpolation.",
            (
                "Confirmed same-session EEG first uses direct absolute intersection; "
                "when its device clock does not overlap, one documented session-end "
                "constant offset is applied. This is registration, not shared hardware "
                "synchronization, so sub-second lag claims remain unsupported."
                if alignment_mode == "session_registered_time"
                else
                "Sensor samples are retained only where source and transcript absolute-time "
                "intervals intersect; device clocks did not share a hardware trigger, so "
                "sub-second lag claims remain unsupported."
                if alignment_mode == "absolute_time"
                else "Sensor streams are duration-normalized because acquisition clocks are inconsistent."
            ),
            (
                f"Only {int(alignment_aggregate.get('eeg_sessions_with_overlap', 0))} of "
                f"{total_sessions} sessions contain valid EEG overlap after the declared "
                "clock policy and "
                f"{int(alignment_aggregate.get('physiology_sessions_with_overlap', 0))} "
                "contain valid PPG overlap."
            ),
            (
                "The frozen validation and test splits contain no valid EEG sessions; this "
                "dataset split cannot support an EEG generalization claim."
                if split_payloads["val"]["modality_session_counts"]["eeg_time"] == 0
                and split_payloads["test"]["modality_session_counts"]["eeg_time"] == 0
                else "EEG split coverage must be interpreted together with the reported masks."
            ),
            "EEG and physiology are missing for a subset of subjects and are handled by explicit masks.",
            "The selected prepared dataset has no audio representation; audio is disabled "
            "and unavailable for every subject.",
            "Target identity is inferred retrospectively from complete-transcript speaking "
            "volume; it is not supplied role metadata or an online role-discovery method.",
            "The target-speaker heuristic and highlighted-label semantics still require "
            "independent confirmation by the data owner.",
            "This audit does not establish annotation validity or population-level generalization.",
        ],
    }


def _write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    split_sizes = "/".join(
        str(payload["splits"][split]["num_subject_sessions"]) for split in SPLITS
    )
    strategy = payload["protocol"]["split_strategy"]
    lines = [
        "# De-identified BCI-subject dataset audit",
        "",
        "This audit reports aggregate dataset properties only. It contains no subject identifiers, "
        "source filenames, or row-level signals.",
        "",
        f"The {split_sizes} split was frozen before training by model-independent "
        f"label/modality-aware stratification (`{strategy}`). It is a fixed pilot split, "
        "not cross-validation.",
        "",
        "## Subject-disjoint split",
        "",
        "| Split | Subjects | Context steps | Target-valid steps | Deception / valid | Prevalence | Events | Median event length |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = payload["splits"][split]
        positive = item["label_counts"]["0"]
        lines.append(
            f"| {split} | {item['num_subject_sessions']} | {item['num_context_steps']} | "
            f"{item['num_target_valid_steps']} | {positive} / {item['num_target_valid_steps']} | "
            f"{item['deception_prevalence']:.3f} | "
            f"{item['num_deception_events']} | {item['event_length_steps']['median']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Modality allocation by split",
            "",
            "| Modality | Train subjects | Validation subjects | Test subjects |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("eeg_time", "physiology", "video"):
        lines.append(
            f"| {name} | {payload['splits']['train']['modality_session_counts'][name]} | "
            f"{payload['splits']['val']['modality_session_counts'][name]} | "
            f"{payload['splits']['test']['modality_session_counts'][name]} |"
        )
    lines.extend(
        [
            "",
            "## Modality coverage",
            "",
            "| Modality | Subjects with any data | Available steps |",
            "|---|---:|---:|",
        ]
    )
    for name, item in payload["modality_coverage"].items():
        lines.append(
            f"| {name} | {item['sessions_available']} / {payload['total_subject_sessions']} | "
            f"{item['available_steps']} / {payload['total_context_steps']} |"
        )
    alignment = payload["alignment"]
    lines.extend(
        [
            "",
            "## Clock alignment",
            "",
            f"- Scope: `{alignment['scope']}`",
            f"- Shared hardware clock: `{alignment['physical_clock_synchronization']}`",
            f"- EEG valid overlap: {alignment['eeg_sessions_with_valid_overlap']} / "
            f"{alignment['eeg_sessions_present']} source-present sessions "
            f"({alignment['eeg_valid_steps']} one-second steps)",
            f"- PPG valid overlap: {alignment['physiology_sessions_with_valid_overlap']} / "
            f"{alignment['physiology_sessions_present']} source-present sessions "
            f"({alignment['physiology_valid_steps']} one-second steps)",
            f"- Facial CSV source records: {alignment['facial_source_records']}",
            f"- EEG sessions using a constant clock offset: "
            f"{alignment['eeg_clock_registered_sessions']}",
        ]
    )
    if payload["protocol"]["alignment_mode"] == "session_registered_time":
        lines.append(
            "- Confirmed clock-mismatch control registered: "
            f"`{alignment['known_clock_mismatch_registered']}`"
        )
    else:
        lines.append(
            "- Known non-overlap control rejected: "
            f"`{alignment['known_nonoverlap_control_rejected']}`"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    lines.extend(
        [
            "",
            f"Prepared-data fingerprint: `{payload['data_fingerprint_sha256']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bci_subjects.yaml"))
    parser.add_argument(
        "--json-output", type=Path, default=Path("results/bci_subjects_data_audit.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("results/bci_subjects_data_audit.md")
    )
    args = parser.parse_args()

    payload = build_audit(args.config)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_markdown(args.markdown_output, payload)
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
