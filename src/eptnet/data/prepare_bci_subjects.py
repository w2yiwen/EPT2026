from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Literal, overload
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import torch

from ..models.behavior.face.marlin import MARLIN_OUTPUT_DIM
from .behavior_features import (
    MACBERT_MODEL_ID,
    MACBERT_REVISION,
    OPENFACE_FEATURE_NAMES,
    OPENFACE_FRAME_COLUMNS,
    WAVLM_MODEL_ID,
    WAVLM_REVISION,
    BehaviorFeatureConfig,
    BehaviorFeaturePipeline,
)
from .session_package import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_jsonl,
    write_session_package,
)
from .session_package import (
    sha256_file as package_sha256_file,
)
from .whisper_alignment import load_alignment

DATASET_NAME = "bci_subjects_ept_v1"
SESSION_ID_PATTERN = re.compile(r"^session_(?P<number>\d{3,})$")
STEP_SECONDS = 1.0
LOCAL_TIMEZONE = timezone(timedelta(hours=8))
ACQUISITION_YEAR = 2025
# Empirical effective rate from two intact files written by the same PPG
# acquisition program: 385.005 Hz and 385.559 Hz.  The rounded value avoids
# pretending that timestamp-free recordings have per-session clock precision.
PPG_EMPIRICAL_SAMPLE_RATE_HZ = 385.0
EEG_TIME_DIM = 8
EEG_SPECTRAL_DIM = 40
PHYSIOLOGY_DIM = 40
VIDEO_DIM = 3
AUDIO_DIM = 50
TEXT_DIM = 768
PHYSIOLOGY_MODALITIES = ("eeg_time", "eeg_spectral", "physiology")
BEHAVIOR_MODALITIES = ("video", "audio", "text")
FEATURE_NAMES = PHYSIOLOGY_MODALITIES + BEHAVIOR_MODALITIES
COMPLIANCE_PROFILES: dict[str, tuple[str, ...]] = {
    "model_contract": (
        "annotation_source",
        "timeline",
        "tensor_schema",
        "finite_features",
        "explicit_missingness",
        "target_supervision",
    ),
    "legacy_face_text": (
        "annotation_source",
        "timeline",
        "tensor_schema",
        "finite_features",
        "explicit_missingness",
        "target_supervision",
        "face_features",
        "text_features",
    ),
    "strong_behavior_sources": (
        "annotation_source",
        "timeline",
        "tensor_schema",
        "finite_features",
        "explicit_missingness",
        "target_supervision",
        "raw_video",
        "raw_audio",
        "text_features",
    ),
    "strong_behavior_features": (
        "annotation_source",
        "timeline",
        "tensor_schema",
        "finite_features",
        "explicit_missingness",
        "target_supervision",
        "openface_features",
        "wavlm_features",
        "macbert_features",
    ),
    "complete_multimodal_sources": (
        "annotation_source",
        "timeline",
        "tensor_schema",
        "finite_features",
        "explicit_missingness",
        "target_supervision",
        "raw_video",
        "raw_audio",
        "text_features",
        "eeg_input",
        "paired_ppg_input",
    ),
}
COMPLIANCE_CHECK_DESCRIPTIONS = {
    "annotation_source": "an annotated transcript source exists",
    "timeline": "the one-second timeline is non-empty, contiguous, and monotonic",
    "tensor_schema": "every feature, mask, target, and timestamp has the declared shape",
    "finite_features": "every stored feature and timestamp is finite",
    "explicit_missingness": "unavailable feature rows are zero and guarded by false masks",
    "target_supervision": "target-valid truth and deception steps plus an event are present",
    "face_features": "at least one aligned facial-action feature step is available",
    "text_features": "at least one aligned non-empty transcript step is available",
    "raw_video": "a real source video is available for a frozen face extractor",
    "raw_audio": "a real source audio file is available for a frozen speech extractor",
    "eeg_input": "raw EEG exists and yields at least one valid aligned step",
    "paired_ppg_input": "paired PPG exists and yields at least one valid aligned step",
    "openface_features": "frozen OpenFace features are available for at least one step",
    "wavlm_features": "frozen WavLM Base+ features are available for at least one step",
    "macbert_features": "frozen MacBERT features are available for at least one step",
}
SPEAKER_TIME = re.compile(
    r"(?:说话人|speaker|发言人)\s*(?P<speaker>\d+)\s*[:：]?\s*"
    r"(?P<minute>\d{1,3})\s*[:：]\s*(?P<second>\d{2})",
    re.IGNORECASE,
)
MIN_TARGET_MARKED_COVERAGE = 0.95
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
AlignmentMode = Literal[
    "duration_normalized",
    "absolute_time",
    "session_registered_time",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_event_targets(
    labels: np.ndarray,
    event_label: int = 0,
    valid_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build contiguous event targets, optionally restricted to valid timesteps.

    ``labels`` are returned unchanged for auditability. ``valid_mask`` gates only
    event supervision, so annotations that bleed into a non-target speaker turn
    cannot create or connect events.
    """
    labels = np.asarray(labels, dtype=np.int64)
    positive = labels == event_label
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != labels.shape:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} does not match labels {labels.shape}"
            )
        positive &= valid_mask
    boundaries = np.zeros((len(labels), 2), dtype=np.float32)
    offsets = np.zeros((len(labels), 2), dtype=np.float32)
    index = 0
    while index < len(labels):
        if not positive[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(labels) and positive[index + 1]:
            index += 1
        end = index
        boundaries[start, 0] = 1.0
        boundaries[end, 1] = 1.0
        positions = np.arange(start, end + 1)
        offsets[start : end + 1, 0] = positions - start
        offsets[start : end + 1, 1] = end - positions
        index += 1
    return {
        "labels": labels.astype(np.int64),
        "boundaries": boundaries,
        "offsets": offsets,
        "positive_mask": positive,
    }


def window_starts(length: int, window_size: int, stride: int, min_window_size: int) -> list[int]:
    if length < min_window_size:
        raise ValueError(f"Session length {length} is below min_window_size {min_window_size}")
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@dataclass(frozen=True)
class SourceSession:
    session_id: str
    subject_id: str
    root: Path
    annotation: Path
    annotation_mode: str
    face_csv: Path | None
    openface_csv: Path | None
    ppg_files: tuple[Path, Path] | None
    eeg_file: Path | None
    audio_file: Path | None
    video_file: Path | None
    metadata: Mapping[str, object] | None = None


@dataclass
class PreparedSession:
    dataset_name: str
    session_id: str
    subject_id: str
    source: SourceSession
    duration_seconds: float
    step_seconds: float
    features: dict[str, np.ndarray]
    physiology_mask: np.ndarray
    modality_mask: np.ndarray
    target_mask: np.ndarray
    labels: np.ndarray
    boundaries: np.ndarray
    offsets: np.ndarray
    positive_mask: np.ndarray
    timestamps: np.ndarray
    step_text: list[str]
    annotation_statistics: dict[str, bool | int | float | str]
    behavior_provenance: dict[str, object]
    alignment: dict[str, object]


@dataclass(frozen=True)
class SessionClock:
    timeline_start_epoch_s: float | None
    timeline_start_local: str | None
    timeline_origin_evidence: str
    timeline_origin_confidence: str
    ppg_start_epoch_s: float | None
    ppg_start_local: str | None
    session_end_epoch_s: float | None
    session_end_local: str | None
    session_end_evidence: str
    session_end_confidence: str
    media_timestamp_semantics: str | None


def _local_datetime_to_epoch_seconds(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=LOCAL_TIMEZONE)
    return value.timestamp()


def _epoch_seconds_to_local(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=LOCAL_TIMEZONE).isoformat(timespec="milliseconds")


def _parse_local_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("/", "-")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _metadata_file_records(source: SourceSession) -> list[Mapping[str, object]]:
    if source.metadata is None:
        return []
    records = source.metadata.get("files", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, Mapping)]


def _record_for_staged_path(
    source: SourceSession, path: Path | None
) -> Mapping[str, object] | None:
    if path is None:
        return None
    for record in _metadata_file_records(source):
        staged = str(record.get("staged_relative_path", "")).replace("\\", "/")
        if staged.rsplit("/", 1)[-1] == path.name:
            return record
    return None


def _source_name(record: Mapping[str, object] | None) -> str:
    if record is None:
        return ""
    value = str(record.get("source_relative_path", "")).replace("\\", "/")
    return value.rsplit("/", 1)[-1]


def _month_day_time_from_name(name: str, *, year: int = ACQUISITION_YEAR) -> datetime | None:
    match = re.search(
        r"(?<!\d)(?P<month>\d{2})-(?P<day>\d{2})-(?P<hour>\d{2})-"
        r"(?P<minute>\d{2})-(?P<second>\d{2})(?!\d)",
        name,
    )
    if match is None:
        return None
    try:
        return datetime(
            year,
            int(match.group("month")),
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
        )
    except ValueError:
        return None


def _compact_datetime_from_name(name: str) -> datetime | None:
    match = re.search(
        r"(?P<year>20\d{2})[-_]?" r"(?P<month>\d{2})(?P<day>\d{2})[_-]"
        r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})",
        name,
    )
    if match is None:
        return None
    try:
        return datetime(*(int(match.group(key)) for key in (
            "year", "month", "day", "hour", "minute", "second"
        )))
    except ValueError:
        return None


def _ppg_start_datetime(source: SourceSession) -> datetime | None:
    if source.ppg_files is None:
        return None
    record = _record_for_staged_path(source, source.ppg_files[0])
    return _month_day_time_from_name(_source_name(record))


def _record_duration_seconds(record: Mapping[str, object] | None) -> float | None:
    if record is None:
        return None
    try:
        value = float(record.get("source_duration_seconds"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _media_interval_datetime(
    source: SourceSession, ppg_start: datetime | None
) -> tuple[datetime | None, datetime | None, str, str | None]:
    for path in (source.audio_file, source.video_file):
        record = _record_for_staged_path(source, path)
        if record is None:
            continue
        name = _source_name(record)
        duration = _record_duration_seconds(record)
        recorded_at = _parse_local_datetime(record.get("source_recorded_at"))
        if recorded_at is not None:
            try:
                clip_start = float(record.get("clip_start_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                clip_start = 0.0
            timestamp_semantics = (
                "recording_end" if name.upper().startswith("VID_") else "recording_start"
            )
            if timestamp_semantics == "recording_end" and duration is not None:
                source_end = recorded_at
                source_start = recorded_at - timedelta(seconds=duration)
                evidence = "media_end_timestamp_minus_duration_plus_clip_offset"
            else:
                source_start = recorded_at
                source_end = (
                    recorded_at + timedelta(seconds=duration) if duration is not None else None
                )
                evidence = "media_start_timestamp_plus_clip_offset"
            return (
                source_start + timedelta(seconds=clip_start),
                source_end,
                evidence,
                timestamp_semantics,
            )
        compact = _compact_datetime_from_name(name)
        if compact is not None:
            timestamp_semantics = (
                "recording_end" if name.upper().startswith("VID_") else "recording_start"
            )
            if timestamp_semantics == "recording_end" and duration is not None:
                return (
                    compact - timedelta(seconds=duration),
                    compact,
                    "media_filename_end_minus_duration",
                    timestamp_semantics,
                )
            return (
                compact,
                compact + timedelta(seconds=duration) if duration is not None else None,
                "media_filename_start",
                timestamp_semantics,
            )
        if ppg_start is not None:
            clock = re.search(
                r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})"
                r"(?:-\d{3})?(?=\.[^.]+$)",
                name,
            )
            if clock is not None:
                start = datetime(
                    ppg_start.year,
                    ppg_start.month,
                    ppg_start.day,
                    int(clock.group("hour")),
                    int(clock.group("minute")),
                    int(clock.group("second")),
                )
                return (
                    start,
                    start + timedelta(seconds=duration) if duration is not None else None,
                    "media_filename_clock_with_session_date",
                    "recording_start",
                )
    return None, None, "none", None


def _facial_reference_start_datetime(source: SourceSession) -> datetime | None:
    for record in _metadata_file_records(source):
        staged = str(record.get("staged_relative_path", "")).replace("\\", "/").lower()
        if not staged.endswith("_facial_actions.csv"):
            continue
        name = _source_name(record)
        match = re.search(
            r"action_info_(?P<year>20\d{2})(?P<month>\d{2})(?P<day>\d{2})_"
            r"(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})",
            name,
        )
        if match is None:
            continue
        try:
            return datetime(*(int(match.group(key)) for key in (
                "year", "month", "day", "hour", "minute", "second"
            )))
        except ValueError:
            continue
    return None


def resolve_session_clock(
    source: SourceSession, timeline_duration_seconds: float | None = None
) -> SessionClock:
    """Resolve an auditable local-time origin without reading facial values."""

    ppg_start = _ppg_start_datetime(source)
    media_start, media_end, media_evidence, media_semantics = _media_interval_datetime(
        source, ppg_start
    )
    if media_start is not None:
        timeline_start = media_start
        evidence = media_evidence
        confidence = "high"
    elif ppg_start is not None:
        timeline_start = ppg_start
        evidence = "ppg_filename_start"
        confidence = "medium"
    else:
        facial_reference = _facial_reference_start_datetime(source)
        timeline_start = facial_reference
        evidence = (
            "facial_action_filename_reference_only" if facial_reference is not None else "none"
        )
        confidence = "medium" if facial_reference is not None else "unavailable"
    timeline_epoch = (
        _local_datetime_to_epoch_seconds(timeline_start) if timeline_start is not None else None
    )
    ppg_epoch = _local_datetime_to_epoch_seconds(ppg_start) if ppg_start is not None else None
    if media_end is not None:
        session_end = media_end
        session_end_evidence = (
            "media_recording_end_from_end_timestamp"
            if media_semantics == "recording_end"
            else "media_recording_end_from_start_plus_duration"
        )
        session_end_confidence = "high"
    elif timeline_start is not None and timeline_duration_seconds is not None:
        session_end = timeline_start + timedelta(seconds=timeline_duration_seconds)
        session_end_evidence = "transcript_origin_plus_duration"
        session_end_confidence = "low"
    else:
        session_end = None
        session_end_evidence = "none"
        session_end_confidence = "unavailable"
    session_end_epoch = (
        _local_datetime_to_epoch_seconds(session_end) if session_end is not None else None
    )
    return SessionClock(
        timeline_start_epoch_s=timeline_epoch,
        timeline_start_local=_epoch_seconds_to_local(timeline_epoch),
        timeline_origin_evidence=evidence,
        timeline_origin_confidence=confidence,
        ppg_start_epoch_s=ppg_epoch,
        ppg_start_local=_epoch_seconds_to_local(ppg_epoch),
        session_end_epoch_s=session_end_epoch,
        session_end_local=_epoch_seconds_to_local(session_end_epoch),
        session_end_evidence=session_end_evidence,
        session_end_confidence=session_end_confidence,
        media_timestamp_semantics=media_semantics,
    )


def _is_zero_where_missing(values: np.ndarray, mask: np.ndarray) -> bool:
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    return bool(np.all(values[~mask] == 0.0))


def _session_compliance_checks(session: PreparedSession) -> dict[str, bool]:
    """Evaluate one de-identified session without imputing absent source data."""
    num_steps = len(session.labels)
    video_shape = np.asarray(session.features["video"]).shape
    audio_shape = np.asarray(session.features["audio"]).shape
    expected_dimensions = {
        "eeg_time": EEG_TIME_DIM,
        "eeg_spectral": EEG_SPECTRAL_DIM,
        "physiology": PHYSIOLOGY_DIM,
        "video": video_shape[1] if len(video_shape) == 2 else -1,
        "audio": audio_shape[1] if len(audio_shape) == 2 else -1,
        "text": TEXT_DIM,
    }
    behavior_widths_ok = expected_dimensions["video"] in {
        VIDEO_DIM,
        len(OPENFACE_FEATURE_NAMES),
        MARLIN_OUTPUT_DIM,
    } and expected_dimensions["audio"] in {AUDIO_DIM, 768}
    feature_shapes_ok = all(
        np.asarray(session.features[name]).shape == (num_steps, dimension)
        for name, dimension in expected_dimensions.items()
    )
    target_shapes_ok = (
        np.asarray(session.physiology_mask).shape == (num_steps, len(PHYSIOLOGY_MODALITIES))
        and np.asarray(session.modality_mask).shape == (num_steps, len(BEHAVIOR_MODALITIES))
        and np.asarray(session.target_mask).shape == (num_steps,)
        and np.asarray(session.labels).shape == (num_steps,)
        and np.asarray(session.boundaries).shape == (num_steps, 2)
        and np.asarray(session.offsets).shape == (num_steps, 2)
        and np.asarray(session.positive_mask).shape == (num_steps,)
        and np.asarray(session.timestamps).shape == (num_steps, 2)
        and len(session.step_text) == num_steps
    )
    timestamps = np.asarray(session.timestamps)
    expected_starts = np.arange(num_steps, dtype=np.float64) * session.step_seconds
    timeline_ok = (
        num_steps > 0
        and session.step_seconds > 0
        and timestamps.shape == (num_steps, 2)
        and np.allclose(timestamps[:, 0], expected_starts)
        and np.allclose(timestamps[:, 1], expected_starts + session.step_seconds)
    )
    finite_ok = all(np.isfinite(np.asarray(values)).all() for values in session.features.values())
    finite_ok = finite_ok and np.isfinite(timestamps).all()

    physiology_mask = np.asarray(session.physiology_mask, dtype=bool)
    modality_mask = np.asarray(session.modality_mask, dtype=bool)
    explicit_missingness_ok = feature_shapes_ok and target_shapes_ok
    if explicit_missingness_ok:
        masks = {
            "eeg_time": physiology_mask[:, 0],
            "eeg_spectral": physiology_mask[:, 1],
            "physiology": physiology_mask[:, 2],
            "video": modality_mask[:, 0],
            "audio": modality_mask[:, 1],
            "text": modality_mask[:, 2],
        }
        explicit_missingness_ok = all(
            _is_zero_where_missing(session.features[name], mask) for name, mask in masks.items()
        )

    target_mask = np.asarray(session.target_mask, dtype=bool)
    target_labels = np.asarray(session.labels)[target_mask]
    target_supervision_ok = (
        target_labels.size > 0
        and bool(np.any(target_labels == 0))
        and bool(np.any(target_labels == 1))
        and int(np.asarray(session.boundaries)[:, 0].sum()) > 0
    )
    source = session.source
    behavior = session.behavior_provenance
    return {
        "annotation_source": source.annotation.is_file(),
        "timeline": bool(timeline_ok),
        "tensor_schema": bool(feature_shapes_ok and target_shapes_ok and behavior_widths_ok),
        "finite_features": bool(finite_ok),
        "explicit_missingness": bool(explicit_missingness_ok),
        "target_supervision": bool(target_supervision_ok),
        "face_features": bool(modality_mask[:, 0].any()),
        "text_features": bool(modality_mask[:, 2].any()),
        "raw_video": source.video_file is not None and source.video_file.is_file(),
        "raw_audio": source.audio_file is not None and source.audio_file.is_file(),
        "eeg_input": (
            source.eeg_file is not None
            and source.eeg_file.is_file()
            and bool(physiology_mask[:, 0].any())
        ),
        "paired_ppg_input": (
            source.ppg_files is not None
            and all(path.is_file() for path in source.ppg_files)
            and bool(physiology_mask[:, 2].any())
        ),
        "openface_features": (
            behavior.get("video_backend") in {"marlin", "openface"}
            and bool(modality_mask[:, 0].any())
        ),
        "wavlm_features": (
            behavior.get("audio_backend") == "wavlm" and bool(modality_mask[:, 1].any())
        ),
        "macbert_features": (
            behavior.get("text_backend") == "macbert" and bool(modality_mask[:, 2].any())
        ),
    }


def build_session_compliance(
    sessions: Sequence[PreparedSession], required_profile: str = "model_contract"
) -> dict:
    """Build a deterministic audit that separates structural and source completeness."""
    if required_profile not in COMPLIANCE_PROFILES:
        choices = ", ".join(COMPLIANCE_PROFILES)
        raise ValueError(
            f"Unknown compliance profile {required_profile!r}; choose one of {choices}"
        )
    if not sessions:
        raise ValueError("At least one prepared session is required for compliance auditing")

    session_records: dict[str, dict] = {}
    profile_counts = {profile: 0 for profile in COMPLIANCE_PROFILES}
    failed_check_counts = {check: 0 for check in COMPLIANCE_CHECK_DESCRIPTIONS}
    for session in sorted(sessions, key=lambda item: item.session_id):
        checks = _session_compliance_checks(session)
        profile_status = {
            profile: all(checks[check] for check in requirements)
            for profile, requirements in COMPLIANCE_PROFILES.items()
        }
        for profile, passed in profile_status.items():
            profile_counts[profile] += int(passed)
        missing_required = [
            check for check in COMPLIANCE_PROFILES[required_profile] if not checks[check]
        ]
        for check in missing_required:
            failed_check_counts[check] += 1
        target_mask = np.asarray(session.target_mask, dtype=bool)
        target_labels = np.asarray(session.labels)[target_mask]
        session_records[session.session_id] = {
            "compliant": profile_status[required_profile],
            "missing_required_checks": missing_required,
            "checks": checks,
            "profile_status": profile_status,
            "counts": {
                "steps": len(session.labels),
                "target_valid_steps": int(target_mask.sum()),
                "target_deception_steps": int((target_labels == 0).sum()),
                "target_truth_steps": int((target_labels == 1).sum()),
                "deception_events": int(np.asarray(session.boundaries)[:, 0].sum()),
                "face_feature_steps": int(np.asarray(session.modality_mask)[:, 0].sum()),
                "audio_feature_steps": int(np.asarray(session.modality_mask)[:, 1].sum()),
                "text_feature_steps": int(np.asarray(session.modality_mask)[:, 2].sum()),
            },
        }

    total = len(sessions)
    profiles = {
        profile: {
            "requirements": list(requirements),
            "compliant_sessions": profile_counts[profile],
            "total_sessions": total,
            "coverage": profile_counts[profile] / total,
        }
        for profile, requirements in COMPLIANCE_PROFILES.items()
    }
    selected_count = profile_counts[required_profile]
    return {
        "schema_version": 1,
        "dataset": sessions[0].dataset_name,
        "observational_unit": "one chronological interview session per subject",
        "missingness_policy": (
            "structural absence is encoded as an all-zero feature row only when the matching "
            "availability mask is false; absent raw signals are never imputed or synthesized"
        ),
        "selected_profile": required_profile,
        "all_sessions_compliant": selected_count == total,
        "compliant_sessions": selected_count,
        "total_sessions": total,
        "profiles": profiles,
        "check_descriptions": COMPLIANCE_CHECK_DESCRIPTIONS,
        "selected_profile_failed_check_counts": {
            check: count for check, count in failed_check_counts.items() if count > 0
        },
        "sessions": session_records,
    }


def render_session_compliance_markdown(report: Mapping[str, object]) -> str:
    profiles = report["profiles"]
    sessions = report["sessions"]
    lines = [
        "# BCI session compliance report",
        "",
        f"Selected profile: `{report['selected_profile']}`.",
        "",
        (
            f"Result: **{report['compliant_sessions']}/{report['total_sessions']} sessions "
            f"compliant**."
        ),
        "",
        "Missing raw signals are not imputed or synthesized. A zero feature row is valid only "
        "when its availability mask is false.",
        "",
        "## Profile coverage",
        "",
        "| Profile | Compliant sessions | Coverage |",
        "|---|---:|---:|",
    ]
    assert isinstance(profiles, Mapping)
    for profile, raw_summary in profiles.items():
        assert isinstance(raw_summary, Mapping)
        lines.append(
            f"| `{profile}` | {raw_summary['compliant_sessions']}/"
            f"{raw_summary['total_sessions']} | {float(raw_summary['coverage']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Selected-profile failures",
            "",
            "| Session | Status | Missing requirements |",
            "|---|---|---|",
        ]
    )
    assert isinstance(sessions, Mapping)
    for session_id, raw_record in sessions.items():
        assert isinstance(raw_record, Mapping)
        missing = raw_record["missing_required_checks"]
        assert isinstance(missing, list)
        lines.append(
            f"| `{session_id}` | {'pass' if raw_record['compliant'] else 'fail'} | "
            f"{', '.join(f'`{item}`' for item in missing) if missing else '—'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `model_contract` means the session can safely enter the masked EPT pipeline.",
            "- `legacy_face_text` additionally requires observed face and transcript features.",
            "- `strong_behavior_sources` requires real video and audio for frozen published "
            "extractors; placeholders do not pass.",
            "- `strong_behavior_features` requires observed OpenFace, WavLM Base+, and "
            "MacBERT features after extraction.",
            "- `complete_multimodal_sources` further requires usable EEG and paired PPG.",
            "",
        ]
    )
    return "\n".join(lines)


def _run_text(run: ET.Element) -> str:
    return "".join(node.text or "" for node in run.iter(f"{W_NS}t"))


def _is_yellow(run: ET.Element) -> bool:
    properties = run.find(f"{W_NS}rPr")
    if properties is None:
        return False
    for node in properties.findall(f"{W_NS}highlight"):
        if node.attrib.get(f"{W_NS}val", "").lower() == "yellow":
            return True
    return False


def _is_bold(run: ET.Element) -> bool:
    properties = run.find(f"{W_NS}rPr")
    if properties is None:
        return False
    bold = properties.find(f"{W_NS}b")
    if bold is None:
        return False
    return bold.attrib.get(f"{W_NS}val", "true").lower() not in {"0", "false", "off"}


def _document_paragraphs(path: Path) -> list[list[tuple[str, bool, bool]]]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs: list[list[tuple[str, bool, bool]]] = []
    for paragraph in root.iter(f"{W_NS}p"):
        runs = []
        for run in paragraph.iter(f"{W_NS}r"):
            text = _run_text(run)
            if text:
                runs.append((text, _is_yellow(run), _is_bold(run)))
        if runs:
            paragraphs.append(runs)
    return paragraphs


def _document_text_paragraphs(path: Path) -> list[list[tuple[str, bool, bool]]]:
    """Read DOCX text without inspecting run-format annotation properties."""
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs: list[list[tuple[str, bool, bool]]] = []
    for paragraph in root.iter(f"{W_NS}p"):
        runs = []
        for run in paragraph.iter(f"{W_NS}r"):
            text = _run_text(run)
            if text:
                # Keep the parser's common tuple shape, but do not call either
                # formatting reader until target identity has been frozen.
                runs.append((text, False, False))
        if runs:
            paragraphs.append(runs)
    return paragraphs


def _annotation_counts(path: Path) -> tuple[int, int]:
    yellow = 0
    bold_after_first_time = 0
    seen_time = False
    for paragraph in _document_paragraphs(path):
        paragraph_text = "".join(text for text, _, _ in paragraph)
        seen_time = seen_time or SPEAKER_TIME.search(paragraph_text) is not None
        for text, is_yellow, is_bold in paragraph:
            yellow += len(text.strip()) if is_yellow else 0
            bold_after_first_time += len(text.strip()) if seen_time and is_bold else 0
    return yellow, bold_after_first_time


def select_annotation(documents: Sequence[Path]) -> tuple[Path, str]:
    """Select the marked transcript and identify its span-marking convention."""
    candidates = []
    for path in documents:
        yellow, bold = _annotation_counts(path)
        mode = "yellow_highlight" if yellow else "bold"
        marked = yellow if yellow else bold
        candidates.append((marked, path.stat().st_size, path.name, path, mode))
    if not candidates:
        raise FileNotFoundError("A session has no DOCX transcript")
    marked, _, _, path, mode = max(candidates)
    if marked <= 0:
        raise ValueError(
            f"No yellow-highlight or bold deception annotation found in {[p.name for p in documents]}"
        )
    return path, mode


def _looks_like_eeg(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return "%OpenBCI Raw EEG Data" in "".join(handle.readline() for _ in range(6))


def _discover_ppg_files(text_files: Sequence[Path]) -> tuple[Path, Path] | None:
    candidates = [
        path for path in text_files if "ch2" in path.name.lower() or "ch3" in path.name.lower()
    ]
    if len(candidates) < 2:
        return None
    ch2 = sorted(path for path in candidates if "ch2" in path.name.lower())
    ch3 = sorted(path for path in candidates if "ch3" in path.name.lower())
    if ch2 and ch3:
        return ch2[0], ch3[0]
    # One acquisition used two files both named ch2 with (1)/(2) suffixes.
    return tuple(sorted(candidates)[:2])  # type: ignore[return-value]


def _csv_has_columns(path: Path, required: set[str]) -> bool:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    return required.issubset({name.strip() for name in header})


def _load_session_metadata(session_root: Path) -> dict[str, object] | None:
    """Load normalized-session metadata and reject identity/path inconsistencies.

    A staged dataset is a trust boundary: directory names, embedded identifiers, and
    file paths must agree before any label or feature is read.  Original subject
    folders do not have this metadata and continue through the legacy discovery path.
    """

    metadata_path = session_root / "session_metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid session metadata: {metadata_path}") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"Session metadata must be an object: {metadata_path}")
    expected_session_id = session_root.name
    if metadata.get("session_id") != expected_session_id:
        raise ValueError(
            f"Session identity mismatch for {metadata_path}: "
            f"expected {expected_session_id!r}, found {metadata.get('session_id')!r}"
        )
    return metadata


def _first_existing(session_root: Path, patterns: Sequence[str]) -> Path | None:
    return _first_existing_across((session_root,), patterns)


def _first_existing_across(roots: Sequence[Path], patterns: Sequence[str]) -> Path | None:
    candidates = sorted(
        {
            path.resolve()
            for root in roots
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file()
        }
    )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous canonical source across staged roots: {[str(path) for path in candidates]}"
        )
    return candidates[0] if candidates else None


def _discover_normalized_session(
    session_root: Path, metadata: Mapping[str, object]
) -> SourceSession | None:
    """Discover an already staged ``session_NNN`` without renumbering it."""

    session_id = session_root.name
    if SESSION_ID_PATTERN.fullmatch(session_id) is None:
        raise ValueError(f"Non-canonical staged session directory: {session_root}")
    availability = metadata.get("available_sources", {})
    if availability is not None and not isinstance(availability, dict):
        raise ValueError(f"available_sources must be an object in {session_root}")
    if isinstance(availability, dict) and availability.get("annotated_transcript") is False:
        return None

    annotation = _first_existing(session_root, [f"{session_id}_annotations.docx"])
    if annotation is None:
        raise ValueError(f"Staged session declares annotations but none exist: {session_root}")
    annotation_mode = str(metadata.get("annotation_mode", "")).strip()
    if annotation_mode not in {"yellow_highlight", "bold"}:
        annotation, annotation_mode = select_annotation([annotation])

    centralized_face_root = session_root.parent / "facial_csv" / session_id
    face_roots = (session_root, centralized_face_root)
    face_csv = _first_existing_across(face_roots, [f"{session_id}_facial_actions.csv"])
    openface_csv = _first_existing_across(face_roots, [f"{session_id}_openface_features.csv"])
    eeg_file = _first_existing(session_root, [f"{session_id}_eeg_raw.txt"])
    ppg_first = _first_existing(session_root, [f"{session_id}_ppg_channel_1.txt"])
    ppg_second = _first_existing(session_root, [f"{session_id}_ppg_channel_2.txt"])
    if (ppg_first is None) != (ppg_second is None):
        raise ValueError(f"Incomplete staged PPG pair in {session_root}")
    audio_file = _first_existing(
        session_root,
        [f"{session_id}_audio.wav", f"{session_id}_audio.mp3", f"{session_id}_audio.flac"],
    )
    video_file = _first_existing(
        session_root,
        [f"{session_id}_video.mp4", f"{session_id}_video.avi", f"{session_id}_video.mov"],
    )

    # Processed artifacts expose only the stable pseudonymous session identifier.
    # The private raw metadata retains any owner-provided subject mapping.
    return SourceSession(
        session_id=session_id,
        subject_id=session_id,
        root=session_root.resolve(),
        annotation=annotation,
        annotation_mode=annotation_mode,
        face_csv=face_csv,
        openface_csv=openface_csv,
        ppg_files=(ppg_first, ppg_second) if ppg_first and ppg_second else None,
        eeg_file=eeg_file,
        audio_file=audio_file,
        video_file=video_file,
        metadata=dict(metadata),
    )


def discover_sessions(source_root: Path) -> list[SourceSession]:
    sessions: list[SourceSession] = []
    session_roots = sorted(path for path in source_root.iterdir() if path.is_dir())
    for session_root in session_roots:
        metadata = _load_session_metadata(session_root)
        if metadata is not None:
            normalized = _discover_normalized_session(session_root, metadata)
            if normalized is not None:
                sessions.append(normalized)
            continue
        documents = sorted(session_root.glob("*.docx"))
        if not documents:
            continue
        annotation, annotation_mode = select_annotation(documents)
        text_files = sorted(session_root.glob("*.txt"))
        eeg_files = [path for path in text_files if _looks_like_eeg(path)]
        face_files = sorted(session_root.glob("*.csv"))
        legacy_face_files = [
            path
            for path in face_files
            if _csv_has_columns(path, {"timestamp", "blink", "pressed_lips", "furrow_brow"})
        ]
        openface_files = [
            path
            for path in face_files
            if _csv_has_columns(path, {"timestamp", *OPENFACE_FRAME_COLUMNS})
        ]
        audio_files = sorted(
            path for pattern in ("*.wav", "*.mp3", "*.flac") for path in session_root.glob(pattern)
        )
        video_files = sorted(
            path for pattern in ("*.mp4", "*.avi", "*.mov") for path in session_root.glob(pattern)
        )
        sessions.append(
            SourceSession(
                session_id=f"session_{len(sessions) + 1:03d}",
                subject_id=session_root.name,
                root=session_root,
                annotation=annotation,
                annotation_mode=annotation_mode,
                face_csv=legacy_face_files[0] if legacy_face_files else None,
                openface_csv=openface_files[0] if openface_files else None,
                ppg_files=_discover_ppg_files(text_files),
                eeg_file=eeg_files[0] if eeg_files else None,
                audio_file=audio_files[0] if audio_files else None,
                video_file=video_files[0] if video_files else None,
            )
        )
    if not sessions:
        raise ValueError(f"No subject sessions found under {source_root}")
    session_ids = [session.session_id for session in sessions]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError(f"Duplicate session identifiers discovered under {source_root}")
    return sessions


def _marked_paragraph_chars(
    paragraph: Sequence[tuple[str, bool, bool]], annotation_mode: str
) -> tuple[str, list[bool]]:
    text_parts: list[str] = []
    marks: list[bool] = []
    for text, yellow, bold in paragraph:
        text_parts.append(text)
        marked = yellow if annotation_mode == "yellow_highlight" else bold
        marks.extend([marked] * len(text))
    return "".join(text_parts), marks


def _parse_speaker_turns(
    paragraphs: Sequence[Sequence[tuple[str, bool, bool]]],
    path: Path,
    annotation_mode: str | None = None,
) -> list[dict]:
    """Parse speaker turns, optionally attaching annotation marks.

    The unannotated pass reads only run text. ``transcript_timeline`` uses that
    pass to freeze target identity before a second, explicitly post-inference
    pass reads highlight/bold marks for labels and consistency auditing.
    """
    turns: list[dict] = []
    current: dict | None = None
    for paragraph in paragraphs:
        if annotation_mode is None:
            text = "".join(run[0] for run in paragraph)
            marks: list[bool] | None = None
        else:
            text, marks = _marked_paragraph_chars(paragraph, annotation_mode)
        match = SPEAKER_TIME.search(text)
        if match is not None:
            if current is not None:
                turns.append(current)
            speaker_id = str(int(match.group("speaker")))
            minute = int(match.group("minute"))
            second = int(match.group("second"))
            if second >= 60:
                raise ValueError(f"Transcript {path} contains an invalid timestamp")
            timestamp = minute * 60 + second
            content = text[match.end() :].strip()
            current = {
                "speaker_id": speaker_id,
                "start": float(timestamp),
                "text": content,
            }
            if marks is not None:
                left_trim = len(text[match.end() :]) - len(text[match.end() :].lstrip())
                current["marks"] = marks[
                    match.end() + left_trim : match.end() + left_trim + len(content)
                ]
        elif current is not None:
            content = text.strip()
            if content:
                current["text"] += content
                if marks is not None:
                    leading = len(text) - len(text.lstrip())
                    current["marks"].extend(marks[leading : leading + len(content)])
    if current is not None:
        turns.append(current)
    return turns


@overload
def transcript_timeline(
    path: Path,
    annotation_mode: str,
    step_seconds: float = STEP_SECONDS,
    *,
    include_target_mask: Literal[False] = False,
) -> tuple[list[str], np.ndarray, dict[str, bool | int | float | str]]: ...


@overload
def transcript_timeline(
    path: Path,
    annotation_mode: str,
    step_seconds: float = STEP_SECONDS,
    *,
    include_target_mask: Literal[True],
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, bool | int | float | str]]: ...


def transcript_timeline(
    path: Path,
    annotation_mode: str,
    step_seconds: float = STEP_SECONDS,
    *,
    include_target_mask: bool = False,
) -> (
    tuple[list[str], np.ndarray, dict[str, bool | int | float | str]]
    | tuple[list[str], np.ndarray, np.ndarray, dict[str, bool | int | float | str]]
):
    """Map run-level DOCX annotations to fixed-duration causal steps.

    Timestamps exist at speaker-turn granularity. Characters inside each turn are
    therefore distributed uniformly between adjacent timestamp anchors. This is
    approximate alignment, and the strategy is recorded in the output metadata.
    Target identity uses the complete transcript and is therefore retrospective;
    "causal" describes the post-preprocessing model feature stream, not online
    discovery of the participant role.
    """
    # Freeze target identity in a text-only pass. Do not inspect annotation
    # marks above this point: held-out labels must not choose the speaker.
    text_paragraphs = _document_text_paragraphs(path)
    turns = _parse_speaker_turns(text_paragraphs, path)
    if len(turns) < 2:
        raise ValueError(f"Transcript {path} has fewer than two timestamped speaker turns")

    characters_by_speaker: Counter[str] = Counter()
    for turn in turns:
        characters_by_speaker.setdefault(turn["speaker_id"], 0)
        characters_by_speaker[turn["speaker_id"]] += sum(
            not character.isspace() for character in turn["text"]
        )
    total_transcript_characters = sum(characters_by_speaker.values())
    if total_transcript_characters <= 0:
        raise ValueError(f"Transcript {path} contains no non-whitespace speaker text")
    ranked_speakers = sorted(
        characters_by_speaker.items(), key=lambda item: (-item[1], int(item[0]))
    )
    if len(ranked_speakers) > 1 and ranked_speakers[0][1] == ranked_speakers[1][1]:
        raise ValueError(
            f"Transcript {path} has ambiguous target-speaker transcript length: "
            f"the two leading speakers are tied at {ranked_speakers[0][1]} characters"
        )
    target_speaker, target_speaker_characters = ranked_speakers[0]
    runner_up_characters = ranked_speakers[1][1] if len(ranked_speakers) > 1 else 0
    target_speaker_character_share = target_speaker_characters / total_transcript_characters
    target_speaker_margin_characters = target_speaker_characters - runner_up_characters

    # Only after target identity is immutable may annotation marks be read.
    annotated_paragraphs = _document_paragraphs(path)
    annotated_turns = _parse_speaker_turns(annotated_paragraphs, path, annotation_mode)

    def turn_identity(turn: Mapping[str, object]) -> tuple[object, object, object]:
        return turn["speaker_id"], turn["start"], turn["text"]

    if [turn_identity(turn) for turn in annotated_turns] != [turn_identity(turn) for turn in turns]:
        raise RuntimeError("Annotated and text-only transcript passes disagree")
    for turn, annotated_turn in zip(turns, annotated_turns, strict=True):
        turn["marks"] = annotated_turn["marks"]

    marked_by_speaker: Counter[str] = Counter(
        {speaker_id: 0 for speaker_id in characters_by_speaker}
    )
    for turn in turns:
        marked_by_speaker[turn["speaker_id"]] += sum(
            marked and not character.isspace()
            for character, marked in zip(turn["text"], turn["marks"], strict=True)
        )
    total_marked_characters = sum(marked_by_speaker.values())
    if total_marked_characters <= 0:
        raise ValueError(f"Transcript {path} contains no marked transcript characters")
    target_marked_characters = marked_by_speaker[target_speaker]
    target_marked_coverage = target_marked_characters / total_marked_characters
    if target_marked_coverage < MIN_TARGET_MARKED_COVERAGE:
        raise ValueError(
            f"Transcript {path} fails target-label consistency: text-inferred speaker "
            f"owns {target_marked_coverage:.3%} of marked characters, below "
            f"{MIN_TARGET_MARKED_COVERAGE:.0%}"
        )

    observed_rates = []
    for turn, following in zip(turns, turns[1:], strict=False):
        gap = following["start"] - turn["start"]
        characters = sum(not character.isspace() for character in turn["text"])
        if gap > 0 and characters > 0:
            observed_rates.append(characters / gap)
    character_rate = float(np.median(observed_rates)) if observed_rates else 4.0
    character_rate = min(max(character_rate, 1.0), 12.0)

    for index, turn in enumerate(turns):
        if index + 1 < len(turns):
            turn["end"] = turns[index + 1]["start"]
        else:
            characters = sum(not character.isspace() for character in turn["text"])
            turn["end"] = turn["start"] + max(2.0, characters / character_rate)
        if turn["end"] <= turn["start"]:
            turn["end"] = turn["start"] + step_seconds

    duration = max(turn["end"] for turn in turns)
    num_steps = max(16, int(math.ceil(duration / step_seconds)))
    step_text: list[list[str]] = [[] for _ in range(num_steps)]
    marked_counts = np.zeros(num_steps, dtype=np.int64)
    target_character_counts = np.zeros(num_steps, dtype=np.int64)
    non_target_character_counts = np.zeros(num_steps, dtype=np.int64)
    for turn in turns:
        units = [
            (character, marked)
            for character, marked in zip(turn["text"], turn["marks"], strict=True)
            if not character.isspace()
        ]
        if not units:
            continue
        turn_duration = turn["end"] - turn["start"]
        for index, (character, marked) in enumerate(units):
            midpoint = turn["start"] + (index + 0.5) * turn_duration / len(units)
            step = min(int(midpoint // step_seconds), num_steps - 1)
            step_text[step].append(character)
            if turn["speaker_id"] == target_speaker:
                target_character_counts[step] += 1
            else:
                non_target_character_counts[step] += 1
            if marked and turn["speaker_id"] == target_speaker:
                marked_counts[step] += 1
    occupied_steps = (target_character_counts + non_target_character_counts) > 0
    mixed_steps = (target_character_counts > 0) & (non_target_character_counts > 0)
    tied_steps = occupied_steps & (target_character_counts == non_target_character_counts)
    target_mask = (target_character_counts > 0) & (
        target_character_counts > non_target_character_counts
    )
    labels = np.where(marked_counts > 0, 0, 1).astype(np.int64)
    joined_text = ["".join(characters) for characters in step_text]
    stats: dict[str, bool | int | float | str] = {
        "annotation_mode": annotation_mode,
        "alignment": "uniform_character_interpolation_between_speaker_timestamps",
        "speaker_step_assignment": (
            "strict_target_character_majority_per_step; empty_and_tied_steps_invalid"
        ),
        "num_speaker_turns": len(turns),
        "num_speakers": len(characters_by_speaker),
        "duration_seconds": float(duration),
        "character_rate_per_second": character_rate,
        "marked_characters": total_marked_characters,
        "target_speaker_id": target_speaker,
        "target_speaker_inference": "maximum_total_non_whitespace_transcript_characters",
        "target_speaker_inference_uses_annotation_marks": False,
        "target_speaker_transcript_characters": target_speaker_characters,
        "total_speaker_transcript_characters": total_transcript_characters,
        "target_speaker_character_share": target_speaker_character_share,
        "runner_up_speaker_transcript_characters": runner_up_characters,
        "target_speaker_margin_characters": target_speaker_margin_characters,
        "target_speaker_margin_fraction_of_total": (
            target_speaker_margin_characters / total_transcript_characters
        ),
        "target_speaker_tie_policy": "fail_closed",
        "target_speaker_validation": (
            "post_inference_annotation_consistency_only; marked coverage >= 0.95 "
            "and marked characters > 0"
        ),
        "target_speaker_annotation_minimum_coverage": MIN_TARGET_MARKED_COVERAGE,
        "target_marked_characters": target_marked_characters,
        "non_target_marked_characters": total_marked_characters - target_marked_characters,
        "discarded_non_target_marked_characters": (
            total_marked_characters - target_marked_characters
        ),
        "label_marked_characters": target_marked_characters,
        "target_marked_coverage": target_marked_coverage,
        "target_valid_steps": int(target_mask.sum()),
        "target_valid_fraction": float(target_mask.mean()),
        "mixed_speaker_steps": int(mixed_steps.sum()),
        "tied_speaker_steps": int(tied_steps.sum()),
        "deception_steps": int((labels == 0).sum()),
        "truth_steps": int((labels == 1).sum()),
    }
    if include_target_mask:
        return joined_text, labels, target_mask, stats
    return joined_text, labels, stats


def hashed_text_features(texts: Sequence[str], dimension: int = TEXT_DIM) -> np.ndarray:
    """Create deterministic local text vectors without an external encoder."""
    output = np.zeros((len(texts), dimension), dtype=np.float32)
    for row, text in enumerate(texts):
        compact = "".join(text.split())
        tokens = list(compact) + [compact[index : index + 2] for index in range(len(compact) - 1)]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            column = value % dimension
            output[row, column] += 1.0 if (value >> 63) == 0 else -1.0
        norm = float(np.linalg.norm(output[row]))
        if norm > 0:
            output[row] /= norm
    return output


def _aggregate_face(path: Path | None, num_steps: int) -> tuple[np.ndarray, np.ndarray]:
    output = np.zeros((num_steps, VIDEO_DIM), dtype=np.float32)
    mask = np.zeros(num_steps, dtype=bool)
    if path is None:
        return output, mask
    timestamps: list[np.datetime64] = []
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "blink", "pressed_lips", "furrow_brow"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Unexpected facial CSV schema in {path}")
        for row in reader:
            try:
                timestamps.append(np.datetime64(row["timestamp"]))
                values.append(
                    [float(row["blink"]), float(row["pressed_lips"]), float(row["furrow_brow"])]
                )
            except (TypeError, ValueError):
                continue
    if not values:
        return output, mask
    matrix = np.asarray(values, dtype=np.float64)
    elapsed = np.asarray(
        [(timestamp - timestamps[0]) / np.timedelta64(1, "s") for timestamp in timestamps],
        dtype=np.float64,
    )
    source_duration = max(float(elapsed[-1]), 1e-6)
    aligned = np.clip(elapsed / source_duration * num_steps, 0, num_steps - 1e-7)
    bins = aligned.astype(np.int64)
    for step in range(num_steps):
        selected = matrix[bins == step]
        if len(selected):
            output[step] = selected.mean(axis=0).astype(np.float32)
            mask[step] = True
    return output, mask


def _read_scalar_signal(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = float(line.strip())
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
    return np.asarray(values, dtype=np.float64)


def _signal_features(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    if len(signal) < 3:
        return np.zeros(20, dtype=np.float32)
    signal = np.asarray(signal, dtype=np.float64)
    mean = float(signal.mean())
    centered = signal - mean
    std = float(centered.std())
    safe_std = max(std, 1e-8)
    difference = np.diff(signal)
    rms = float(np.sqrt(np.mean(np.square(signal))))
    skewness = float(np.mean(np.power(centered / safe_std, 3))) if std > 1e-8 else 0.0
    kurtosis = float(np.mean(np.power(centered / safe_std, 4)) - 3.0) if std > 1e-8 else 0.0
    zero_crossings = float(np.mean(centered[:-1] * centered[1:] < 0))
    peaks = int(np.sum((signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:])))
    peak_rate = peaks / max(len(signal) / max(sample_rate, 1e-8), 1e-8)
    windowed = centered * np.hanning(len(centered))
    power = np.square(np.abs(np.fft.rfft(windowed)))
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / max(sample_rate, 1e-8))
    if len(power) > 1 and power[1:].sum() > 0:
        dominant_frequency = float(frequencies[1 + int(np.argmax(power[1:]))])
        probability = power[1:] / power[1:].sum()
        spectral_entropy = float(-np.sum(probability * np.log(probability + 1e-12)))
    else:
        dominant_frequency = 0.0
        spectral_entropy = 0.0
    autocorrelation = (
        float(np.dot(centered[:-1], centered[1:]) / np.dot(centered, centered))
        if np.dot(centered, centered) > 1e-12
        else 0.0
    )
    features = np.asarray(
        [
            mean,
            std,
            float(signal.min()),
            float(signal.max()),
            float(np.median(signal)),
            float(np.quantile(signal, 0.25)),
            float(np.quantile(signal, 0.75)),
            float(np.ptp(signal)),
            float(np.median(np.abs(centered))),
            rms,
            float(np.mean(np.abs(signal))),
            skewness,
            kurtosis,
            float(difference.mean()),
            float(difference.std()),
            float(np.mean(np.abs(difference))),
            zero_crossings,
            peak_rate,
            dominant_frequency,
            spectral_entropy + autocorrelation,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def _ppg_features_with_alignment(
    paths: tuple[Path, Path] | None,
    num_steps: int,
    duration_seconds: float,
    *,
    alignment_mode: AlignmentMode = "duration_normalized",
    timeline_start_epoch_s: float | None = None,
    source_start_epoch_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    output = np.zeros((num_steps, PHYSIOLOGY_DIM), dtype=np.float32)
    mask = np.zeros(num_steps, dtype=bool)
    if paths is None:
        return (
            output,
            mask,
            {
                "present": False,
                "status": "unavailable",
                "method": "none",
                "target_steps": num_steps,
                "valid_steps": 0,
            },
        )
    channels = [_read_scalar_signal(path) for path in paths]
    channel_lengths = [len(channel) for channel in channels]
    length = min(map(len, channels))
    if length < 3:
        return (
            output,
            mask,
            {
                "present": True,
                "status": "insufficient_samples",
                "method": (
                    "absolute_time_intersection"
                    if alignment_mode != "duration_normalized"
                    else "uniform_index_bins_on_transcript_timeline"
                ),
                "channel_samples": channel_lengths,
                "common_samples": length,
                "target_steps": num_steps,
                "target_duration_seconds": duration_seconds,
                "valid_steps": 0,
                "native_duration_seconds": None,
                "duration_scale_factor": None,
            },
        )
    channels = [channel[:length] for channel in channels]
    step_seconds = duration_seconds / max(num_steps, 1)
    if alignment_mode != "duration_normalized":
        sample_rate = PPG_EMPIRICAL_SAMPLE_RATE_HZ
        if timeline_start_epoch_s is None or source_start_epoch_s is None:
            return (
                output,
                mask,
                {
                    "present": True,
                    "status": "unanchored",
                    "method": "absolute_time_intersection",
                    "channel_samples": channel_lengths,
                    "common_samples": length,
                    "sample_rate_hz": sample_rate,
                    "sample_rate_basis": "empirical_same_acquisition_program",
                    "target_steps": num_steps,
                    "valid_steps": 0,
                    "source_start_local": _epoch_seconds_to_local(source_start_epoch_s),
                    "source_end_local": None,
                },
            )
        source_end_epoch_s = source_start_epoch_s + length / sample_rate
        temporal_overlap_steps = 0
        for step in range(num_steps):
            step_start = timeline_start_epoch_s + step * step_seconds
            step_end = step_start + step_seconds
            start = max(0, int(math.ceil((step_start - source_start_epoch_s) * sample_rate)))
            end = min(length, int(math.ceil((step_end - source_start_epoch_s) * sample_rate)))
            temporal_overlap_steps += int(end > start)
            if end - start < 3:
                continue
            output[step] = np.concatenate(
                [_signal_features(channel[start:end], sample_rate) for channel in channels]
            )
            mask[step] = np.isfinite(output[step]).all()
        method = "absolute_time_intersection"
        status = (
            "aligned_absolute_overlap"
            if mask.any()
            else "overlap_but_insufficient_samples"
            if temporal_overlap_steps
            else "no_overlap"
        )
        native_duration = length / sample_rate
        duration_scale = None
    else:
        sample_rate = length / max(duration_seconds, STEP_SECONDS)
        edges = np.linspace(0, length, num_steps + 1, dtype=np.int64)
        for step in range(num_steps):
            start, end = int(edges[step]), int(edges[step + 1])
            if end - start < 3:
                continue
            output[step] = np.concatenate(
                [_signal_features(channel[start:end], sample_rate) for channel in channels]
            )
            mask[step] = np.isfinite(output[step]).all()
        source_end_epoch_s = None
        method = "uniform_index_bins_on_transcript_timeline"
        status = "aligned"
        temporal_overlap_steps = num_steps
        native_duration = None
        duration_scale = None
    output[~mask] = 0.0
    return (
        output,
        mask,
        {
            "present": True,
            "status": status,
            "method": method,
            "channel_samples": channel_lengths,
            "common_samples": length,
            "sample_rate_hz": sample_rate,
            "sample_rate_basis": (
                "empirical_same_acquisition_program"
                if alignment_mode != "duration_normalized"
                else "inferred_from_target_duration"
            ),
            "target_steps": num_steps,
            "target_duration_seconds": duration_seconds,
            "valid_steps": int(mask.sum()),
            "temporal_overlap_steps": temporal_overlap_steps,
            "native_duration_seconds": native_duration,
            "duration_scale_factor": duration_scale,
            "source_start_local": _epoch_seconds_to_local(source_start_epoch_s),
            "source_end_local": _epoch_seconds_to_local(source_end_epoch_s),
        },
    )


def ppg_features(
    paths: tuple[Path, Path] | None, num_steps: int, duration_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    output, mask, _ = _ppg_features_with_alignment(paths, num_steps, duration_seconds)
    return output, mask


def _openbci_epoch_seconds(value: str) -> float | None:
    try:
        raw = float(value.strip())
    except ValueError:
        return None
    if 1_700_000_000_000 <= raw < 1_800_000_000_000:
        return raw / 1000.0
    if 1_700_000_000 <= raw < 1_800_000_000:
        return raw
    return None


def _read_eeg(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    sample_rate = 500.0
    rows: list[list[float]] = []
    epoch_seconds: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            if line.startswith("%"):
                match = re.search(r"Sample Rate\s*=\s*([0-9.]+)", line)
                if match:
                    sample_rate = float(match.group(1))
                continue
            parts = [value.strip() for value in line.split(",")]
            if len(parts) < 9:
                continue
            try:
                channels = [float(value) for value in parts[1:9]]
            except ValueError:
                continue
            if np.isfinite(channels).all():
                rows.append(channels)
                timestamp = _openbci_epoch_seconds(parts[-1])
                epoch_seconds.append(timestamp if timestamp is not None else float("nan"))
    if not rows:
        return (
            np.empty((0, EEG_TIME_DIM), dtype=np.float64),
            sample_rate,
            np.empty(0, dtype=np.float64),
        )
    return (
        np.asarray(rows, dtype=np.float64),
        sample_rate,
        np.asarray(epoch_seconds, dtype=np.float64),
    )


def _eeg_features_with_alignment(
    path: Path | None,
    num_steps: int,
    duration_seconds: float,
    *,
    alignment_mode: AlignmentMode = "duration_normalized",
    timeline_start_epoch_s: float | None = None,
    session_end_epoch_s: float | None = None,
    session_end_evidence: str = "none",
    session_end_confidence: str = "unavailable",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    time_features = np.zeros((num_steps, EEG_TIME_DIM), dtype=np.float32)
    spectral_features = np.zeros((num_steps, EEG_SPECTRAL_DIM), dtype=np.float32)
    mask = np.zeros(num_steps, dtype=bool)
    if path is None:
        return (
            time_features,
            spectral_features,
            mask,
            {
                "present": False,
                "status": "unavailable",
                "method": "none",
                "target_steps": num_steps,
                "valid_steps": 0,
            },
        )
    signal, sample_rate, epoch_seconds = _read_eeg(path)
    native_duration = len(signal) / sample_rate if sample_rate > 0 else None
    duration_scale = (
        duration_seconds / native_duration
        if native_duration is not None and native_duration > 0
        else None
    )
    if alignment_mode != "duration_normalized" and timeline_start_epoch_s is None:
        return (
            time_features,
            spectral_features,
            mask,
            {
                "present": True,
                "status": "unanchored",
                "method": "embedded_epoch_intersection",
                "source_samples": len(signal),
                "timestamped_samples": int(np.isfinite(epoch_seconds).sum()),
                "declared_sample_rate_hz": sample_rate,
                "target_steps": num_steps,
                "valid_steps": 0,
            },
        )
    if alignment_mode == "duration_normalized" and len(signal) < num_steps * 32:
        return (
            time_features,
            spectral_features,
            mask,
            {
                "present": True,
                "status": "insufficient_samples",
                "method": "uniform_index_bins_on_transcript_timeline",
                "source_samples": len(signal),
                "declared_sample_rate_hz": sample_rate,
                "native_duration_seconds": native_duration,
                "target_duration_seconds": duration_seconds,
                "duration_scale_factor": duration_scale,
                "target_steps": num_steps,
                "valid_steps": 0,
            },
        )
    signal = signal - np.median(signal, axis=0, keepdims=True)
    bands = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))

    clock_offset_seconds: float | None = None
    clock_offset_applied = False
    direct_temporal_overlap_steps = 0
    raw_source_start_epoch_s: float | None = None
    raw_source_end_epoch_s: float | None = None
    if alignment_mode != "duration_normalized":
        timestamp_mask = np.isfinite(epoch_seconds)
        aligned_signal = signal[timestamp_mask]
        aligned_epoch = epoch_seconds[timestamp_mask]
        order = np.argsort(aligned_epoch, kind="stable")
        reordered_samples = int(np.count_nonzero(order != np.arange(len(order))))
        aligned_signal = aligned_signal[order]
        aligned_epoch = aligned_epoch[order]
        raw_source_start_epoch_s = float(aligned_epoch[0]) if len(aligned_epoch) else None
        raw_source_end_epoch_s = float(aligned_epoch[-1]) if len(aligned_epoch) else None
        step_seconds = duration_seconds / max(num_steps, 1)

        def epoch_ranges(values: np.ndarray) -> list[tuple[int, int]]:
            output_ranges: list[tuple[int, int]] = []
            for step in range(num_steps):
                step_start = float(timeline_start_epoch_s) + step * step_seconds
                step_end = step_start + step_seconds
                start = int(np.searchsorted(values, step_start, side="left"))
                end = int(np.searchsorted(values, step_end, side="left"))
                output_ranges.append((start, end))
            return output_ranges

        ranges = epoch_ranges(aligned_epoch)
        direct_temporal_overlap_steps = sum(end > start for start, end in ranges)
        if (
            alignment_mode == "session_registered_time"
            and direct_temporal_overlap_steps == 0
            and raw_source_end_epoch_s is not None
            and session_end_epoch_s is not None
        ):
            clock_offset_seconds = session_end_epoch_s - raw_source_end_epoch_s
            aligned_epoch = aligned_epoch + clock_offset_seconds
            ranges = epoch_ranges(aligned_epoch)
            clock_offset_applied = True
            method = "embedded_epoch_plus_constant_session_end_offset"
        else:
            method = "embedded_epoch_intersection"
        source_start_epoch_s = float(aligned_epoch[0]) if len(aligned_epoch) else None
        source_end_epoch_s = float(aligned_epoch[-1]) if len(aligned_epoch) else None
    else:
        aligned_signal = signal
        source_start_epoch_s = None
        source_end_epoch_s = None
        reordered_samples = 0
        edges = np.linspace(0, len(signal), num_steps + 1, dtype=np.int64)
        ranges = [(int(edges[step]), int(edges[step + 1])) for step in range(num_steps)]
        method = "uniform_index_bins_on_transcript_timeline"

    temporal_overlap_steps = sum(end > start for start, end in ranges)
    for step, (start, end) in enumerate(ranges):
        segment = aligned_signal[start:end]
        if len(segment) < 32 or not np.any(np.ptp(segment, axis=0) > 1e-6):
            continue
        time_features[step] = segment.mean(axis=0).astype(np.float32)
        centered = segment - segment.mean(axis=0, keepdims=True)
        spectrum = np.fft.rfft(centered * np.hanning(len(centered))[:, None], axis=0)
        power = np.square(np.abs(spectrum)) / max(len(centered), 1)
        frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
        band_values = []
        for low, high in bands:
            selected = (frequencies >= low) & (frequencies < high)
            band_power = power[selected].mean(axis=0) if selected.any() else np.zeros(8)
            band_values.append(np.log1p(band_power))
        # Channel-major ordering keeps the five bands adjacent for each electrode.
        spectral_features[step] = np.stack(band_values, axis=1).reshape(-1).astype(np.float32)
        mask[step] = (
            np.isfinite(time_features[step]).all() and np.isfinite(spectral_features[step]).all()
        )
    time_features[~mask] = 0.0
    spectral_features[~mask] = 0.0
    if alignment_mode != "duration_normalized":
        if clock_offset_applied:
            status = (
                "aligned_end_anchored_clock_offset"
                if mask.any()
                else "registered_overlap_but_invalid_signal"
                if temporal_overlap_steps
                else "registered_no_overlap"
            )
        else:
            status = (
                "aligned_absolute_overlap"
                if mask.any()
                else "overlap_but_invalid_signal"
                if temporal_overlap_steps
                else "no_overlap"
            )
    else:
        status = "aligned"
    return (
        time_features,
        spectral_features,
        mask,
        {
            "present": True,
            "status": status,
            "method": method,
            "source_samples": len(signal),
            "timestamped_samples": int(np.isfinite(epoch_seconds).sum()),
            "reordered_samples_for_epoch_monotonicity": reordered_samples,
            "declared_sample_rate_hz": sample_rate,
            "native_duration_seconds": (
                source_end_epoch_s - source_start_epoch_s
                if source_start_epoch_s is not None and source_end_epoch_s is not None
                else native_duration
            ),
            "target_duration_seconds": duration_seconds,
            "duration_scale_factor": (
                None if alignment_mode != "duration_normalized" else duration_scale
            ),
            "target_steps": num_steps,
            "valid_steps": int(mask.sum()),
            "temporal_overlap_steps": temporal_overlap_steps,
            "direct_temporal_overlap_steps": direct_temporal_overlap_steps,
            "clock_offset_applied": clock_offset_applied,
            "clock_offset_seconds": clock_offset_seconds,
            "clock_offset_anchor_evidence": (
                session_end_evidence if clock_offset_applied else None
            ),
            "clock_offset_anchor_confidence": (
                session_end_confidence if clock_offset_applied else None
            ),
            "raw_source_start_local": _epoch_seconds_to_local(raw_source_start_epoch_s),
            "raw_source_end_local": _epoch_seconds_to_local(raw_source_end_epoch_s),
            "source_start_local": _epoch_seconds_to_local(source_start_epoch_s),
            "source_end_local": _epoch_seconds_to_local(source_end_epoch_s),
        },
    )


def eeg_features(
    path: Path | None, num_steps: int, duration_seconds: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    duration = duration_seconds if duration_seconds is not None else num_steps * STEP_SECONDS
    time_features, spectral_features, mask, _ = _eeg_features_with_alignment(
        path, num_steps, duration
    )
    return time_features, spectral_features, mask


def prepare_session(
    source: SourceSession,
    step_seconds: float = STEP_SECONDS,
    *,
    behavior_config: BehaviorFeatureConfig | None = None,
    behavior_pipeline: BehaviorFeaturePipeline | None = None,
    dataset_name: str = DATASET_NAME,
    alignment_mode: AlignmentMode = "duration_normalized",
    whisper_alignment: Mapping[str, object] | None = None,
) -> PreparedSession:
    behavior_config = behavior_config or BehaviorFeatureConfig()
    behavior_config.validate()
    behavior_pipeline = behavior_pipeline or BehaviorFeaturePipeline(behavior_config)
    if whisper_alignment is None:
        step_text, labels, target_mask, annotation_statistics = transcript_timeline(
            source.annotation,
            source.annotation_mode,
            step_seconds=step_seconds,
            include_target_mask=True,
        )
        video_time_offset_seconds = 0.0
        transcript_method = "uniform_character_interpolation_between_speaker_timestamps"
    else:
        if not math.isclose(float(whisper_alignment["step_seconds"]), step_seconds):
            raise ValueError(f"Whisper step size mismatch for {source.session_id}")
        step_text = [str(value) for value in whisper_alignment["step_text"]]
        labels = np.asarray(whisper_alignment["labels"], dtype=np.int64)
        target_mask = np.asarray(whisper_alignment["target_mask"], dtype=bool)
        expected_steps = int(whisper_alignment["num_steps"])
        if not (
            len(step_text) == len(labels) == len(target_mask) == expected_steps
            and expected_steps > 0
        ):
            raise ValueError(f"Malformed Whisper timeline for {source.session_id}")
        annotation_statistics = dict(whisper_alignment["statistics"])
        video_time_offset_seconds = float(whisper_alignment["video_alignment"]["offset_seconds"])
        transcript_method = str(annotation_statistics["alignment"])
    num_steps = len(labels)
    duration = num_steps * step_seconds
    clock = (
        resolve_session_clock(source, timeline_duration_seconds=duration)
        if alignment_mode != "duration_normalized"
        else None
    )
    eeg_time, eeg_spectral, eeg_mask, eeg_alignment = _eeg_features_with_alignment(
        source.eeg_file,
        num_steps,
        duration,
        alignment_mode=alignment_mode,
        timeline_start_epoch_s=(clock.timeline_start_epoch_s if clock else None),
        session_end_epoch_s=(clock.session_end_epoch_s if clock else None),
        session_end_evidence=(clock.session_end_evidence if clock else "none"),
        session_end_confidence=(clock.session_end_confidence if clock else "unavailable"),
    )
    physiology, ppg_mask, ppg_alignment = _ppg_features_with_alignment(
        source.ppg_files,
        num_steps,
        duration,
        alignment_mode=alignment_mode,
        timeline_start_epoch_s=(clock.timeline_start_epoch_s if clock else None),
        source_start_epoch_s=(clock.ppg_start_epoch_s if clock else None),
    )
    if behavior_config.video_backend in {"marlin", "openface"}:
        video_source = (
            source.video_file
            if behavior_config.video_backend == "marlin"
            else source.openface_csv or source.video_file
        )
        video, video_mask = behavior_pipeline.encode_video(
            video_source,
            num_steps=num_steps,
            step_seconds=step_seconds if whisper_alignment is not None else None,
            time_offset_seconds=video_time_offset_seconds,
        )
        video_feature_alignment_method = behavior_pipeline.last_video_alignment_method
    else:
        video, video_mask = _aggregate_face(source.face_csv, num_steps)
        video_feature_alignment_method = "legacy_facial_feature_aggregation"
    if behavior_config.text_backend == "macbert":
        text = behavior_pipeline.encode_text(step_text)
    else:
        text = hashed_text_features(step_text)
    text_mask = np.asarray([bool(value.strip()) for value in step_text], dtype=bool)
    audio, audio_mask = behavior_pipeline.encode_audio(
        source.audio_file,
        num_steps=num_steps,
        step_seconds=step_seconds,
    )
    physiology_mask = np.stack((eeg_mask, eeg_mask, ppg_mask), axis=-1)
    modality_mask = np.stack((video_mask, audio_mask, text_mask), axis=-1)
    targets = build_event_targets(labels, event_label=0, valid_mask=target_mask)
    starts = np.arange(num_steps, dtype=np.float32) * step_seconds
    timestamps = np.stack((starts, starts + step_seconds), axis=-1)
    features = {
        "eeg_time": eeg_time,
        "eeg_spectral": eeg_spectral,
        "physiology": physiology,
        "video": video,
        "audio": audio,
        "text": text,
    }
    return PreparedSession(
        dataset_name=dataset_name,
        session_id=source.session_id,
        subject_id=source.subject_id,
        source=source,
        duration_seconds=duration,
        step_seconds=step_seconds,
        features=features,
        physiology_mask=physiology_mask,
        modality_mask=modality_mask,
        target_mask=target_mask,
        labels=targets["labels"],
        boundaries=targets["boundaries"],
        offsets=targets["offsets"],
        positive_mask=targets["positive_mask"],
        timestamps=timestamps,
        step_text=step_text,
        annotation_statistics=annotation_statistics,
        behavior_provenance=behavior_config.provenance(),
        alignment={
            "reference": "annotated_transcript",
            "alignment_mode": alignment_mode,
            "transcript_method": transcript_method,
            "video_feature_alignment_method": video_feature_alignment_method,
            "whisper_alignment": (
                {
                    key: value
                    for key, value in whisper_alignment.items()
                    if key
                    not in {"aligned_characters", "step_text", "labels", "target_mask"}
                }
                if whisper_alignment is not None
                else None
            ),
            "step_seconds": step_seconds,
            "timeline_steps": num_steps,
            "timeline_duration_seconds": duration,
            "timeline_start_local": clock.timeline_start_local if clock else None,
            "timeline_origin_evidence": (
                clock.timeline_origin_evidence if clock else "relative_transcript_only"
            ),
            "timeline_origin_confidence": (
                clock.timeline_origin_confidence if clock else "not_applicable"
            ),
            "session_end_local": clock.session_end_local if clock else None,
            "session_end_evidence": (
                clock.session_end_evidence if clock else "relative_transcript_only"
            ),
            "session_end_confidence": (
                clock.session_end_confidence if clock else "not_applicable"
            ),
            "media_timestamp_semantics": (
                clock.media_timestamp_semantics if clock else None
            ),
            "eeg": eeg_alignment,
            "physiology": ppg_alignment,
        },
    )


def assign_subject_splits(subject_ids: Sequence[str], seed: int = 42) -> dict[str, list[str]]:
    if len(subject_ids) < 3:
        raise ValueError("At least three subjects are required for subject-disjoint splits")
    ordered = sorted(set(subject_ids))
    random.Random(seed).shuffle(ordered)
    train_count = max(1, int(math.floor(len(ordered) * 0.70)))
    val_count = max(1, int(math.floor(len(ordered) * 0.15)))
    if train_count + val_count >= len(ordered):
        train_count = len(ordered) - 2
        val_count = 1
    return {
        "train": sorted(ordered[:train_count]),
        "val": sorted(ordered[train_count : train_count + val_count]),
        "test": sorted(ordered[train_count + val_count :]),
    }


def _largest_remainder_quotas(total: int, split_sizes: Mapping[str, int]) -> dict[str, int]:
    """Allocate an integer modality total in proportion to fixed split sizes."""
    population = sum(split_sizes.values())
    exact = {split: Fraction(total * size, population) for split, size in split_sizes.items()}
    quotas = {split: value.numerator // value.denominator for split, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = {split: index for index, split in enumerate(split_sizes)}
    ranked = sorted(
        split_sizes,
        key=lambda split: (
            -(exact[split] - quotas[split]),
            order[split],
        ),
    )
    for split in ranked[:remainder]:
        quotas[split] += 1
    return quotas


def _session_split_statistics(session: PreparedSession) -> dict[str, int]:
    target_mask = np.asarray(session.target_mask, dtype=bool)
    labels = np.asarray(session.labels, dtype=np.int64)
    if labels.shape != target_mask.shape or not target_mask.any():
        raise ValueError(f"Session {session.session_id} has an invalid target mask")
    deception_steps = int(((labels == 0) & target_mask).sum())
    truth_steps = int(((labels == 1) & target_mask).sum())
    return {
        "target_valid_steps": int(target_mask.sum()),
        "deception_steps": deception_steps,
        "truth_steps": truth_steps,
        "deception_events": int(np.asarray(session.boundaries)[:, 0].sum()),
        "eeg": int(bool(np.asarray(session.physiology_mask)[:, 0].any())),
        "physiology": int(bool(np.asarray(session.physiology_mask)[:, 2].any())),
        "video": int(bool(np.asarray(session.modality_mask)[:, 0].any())),
    }


def assign_constrained_subject_splits(
    sessions: Sequence[PreparedSession], seed: int = 42
) -> tuple[dict[str, list[str]], dict[str, object]]:
    """Freeze a model-independent, label-aware subject split before training.

    Small cohorts are exhaustively enumerated.  For larger cohorts, where exhaustive
    enumeration grows combinatorially, a fixed-seed unique Monte-Carlo search is used
    with the same hard constraints and objective.  Model outputs are never inputs to
    either procedure.
    """
    ordered_sessions = sorted(sessions, key=lambda session: session.session_id)
    if len(ordered_sessions) < 3:
        raise ValueError("At least three sessions are required for subject-disjoint splits")
    session_ids = [session.session_id for session in ordered_sessions]
    subject_ids = [session.subject_id for session in ordered_sessions]
    if len(set(session_ids)) != len(session_ids) or len(set(subject_ids)) != len(subject_ids):
        raise ValueError("The constrained split requires one unique session per subject")

    train_count = max(1, int(math.floor(len(session_ids) * 0.70)))
    val_count = max(1, int(math.floor(len(session_ids) * 0.15)))
    if train_count + val_count >= len(session_ids):
        train_count = len(session_ids) - 2
        val_count = 1
    split_sizes = {
        "train": train_count,
        "val": val_count,
        "test": len(session_ids) - train_count - val_count,
    }
    statistics = {
        session.session_id: _session_split_statistics(session) for session in ordered_sessions
    }
    balance_keys = ("target_valid_steps", "deception_steps", "deception_events")
    modality_keys = ("eeg", "physiology", "video")
    totals = {
        key: sum(item[key] for item in statistics.values())
        for key in (*balance_keys, *modality_keys)
    }
    if any(totals[key] <= 0 for key in balance_keys):
        raise ValueError("Constrained stratification requires valid positive supervision")
    modality_quotas = {
        key: _largest_remainder_quotas(totals[key], split_sizes) for key in modality_keys
    }

    def aggregate(ids: Sequence[str]) -> dict[str, int]:
        return {
            key: sum(statistics[session_id][key] for session_id in ids)
            for key in statistics[session_ids[0]]
        }

    def valid_assignment(groups: Mapping[str, Sequence[str]]) -> bool:
        for split, ids in groups.items():
            counts = aggregate(ids)
            if counts["deception_steps"] <= 0 or counts["truth_steps"] <= 0:
                return False
            if any(counts[key] != modality_quotas[key][split] for key in modality_keys):
                return False
        return True

    def objective(groups: Mapping[str, Sequence[str]]) -> Fraction:
        score = Fraction(0, 1)
        population = len(session_ids)
        for split, ids in groups.items():
            counts = aggregate(ids)
            split_size = split_sizes[split]
            for key in balance_keys:
                denominator = split_size * totals[key]
                residual = counts[key] * population - denominator
                score += Fraction(residual * residual, denominator * denominator)
        return score

    best: tuple[Fraction, str, dict[str, list[str]]] | None = None
    feasible_assignments = 0
    enumerated_assignments = 0
    all_ids = set(session_ids)
    exact_search_size = math.comb(len(session_ids), val_count) * math.comb(
        len(session_ids) - val_count, split_sizes["test"]
    )
    exact_search_limit = 2_000_000
    randomized_search_trials = 250_000

    def consider(groups: dict[str, list[str]]) -> None:
        nonlocal best, feasible_assignments, enumerated_assignments
        enumerated_assignments += 1
        if not valid_assignment(groups):
            return
        feasible_assignments += 1
        score = objective(groups)
        val_ids = groups["val"]
        test_ids = groups["test"]
        tie_break = hashlib.sha256(
            f"{seed}|val={','.join(val_ids)}|test={','.join(test_ids)}".encode()
        ).hexdigest()
        candidate = (score, tie_break, groups)
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    if exact_search_size <= exact_search_limit:
        search_method = "exhaustive"
        for val_ids_tuple in combinations(session_ids, val_count):
            remaining = sorted(all_ids - set(val_ids_tuple))
            for test_ids_tuple in combinations(remaining, split_sizes["test"]):
                val_ids = sorted(val_ids_tuple)
                test_ids = sorted(test_ids_tuple)
                consider(
                    {
                        "train": sorted(all_ids - set(val_ids) - set(test_ids)),
                        "val": val_ids,
                        "test": test_ids,
                    }
                )
    else:
        search_method = "fixed_seed_unique_monte_carlo"
        generator = random.Random(seed)
        seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
        attempts = 0
        max_attempts = randomized_search_trials * 3
        while len(seen) < randomized_search_trials and attempts < max_attempts:
            attempts += 1
            shuffled = list(session_ids)
            generator.shuffle(shuffled)
            train_end = split_sizes["train"]
            val_end = train_end + split_sizes["val"]
            groups = {
                "train": sorted(shuffled[:train_end]),
                "val": sorted(shuffled[train_end:val_end]),
                "test": sorted(shuffled[val_end:]),
            }
            key = (tuple(groups["val"]), tuple(groups["test"]))
            if key in seen:
                continue
            seen.add(key)
            consider(groups)

    if best is None:
        raise ValueError("No subject split satisfies the declared stratification constraints")
    score, tie_break, splits = best
    split_counts = {split: aggregate(ids) for split, ids in splits.items()}
    assignment_sha256 = hashlib.sha256(
        json.dumps(splits, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    stratification_inputs_sha256 = hashlib.sha256(
        json.dumps(statistics, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    strategy: dict[str, object] = {
        "schema_version": 1,
        "algorithm": (
            "exhaustive_constrained_subject_stratification_v1"
            if search_method == "exhaustive"
            else "fixed_seed_constrained_monte_carlo_stratification_v2"
        ),
        "search_method": search_method,
        "exact_search_space_size": exact_search_size,
        "exact_search_limit": exact_search_limit,
        "randomized_search_trials": (
            randomized_search_trials if search_method != "exhaustive" else 0
        ),
        "model_independent": True,
        "frozen_before_training": True,
        "uses_ground_truth_for_stratification": True,
        "seed_used_only_for_exact_score_ties": seed,
        "split_size_rule": "floor(0.70*N), floor(0.15*N), remainder",
        "split_sizes": split_sizes,
        "modality_quota_method": "largest_remainder_proportional_to_split_subject_counts",
        "modality_quotas": modality_quotas,
        "balance_totals": {key: totals[key] for key in balance_keys},
        "objective": (
            "sum over splits and {target_valid_steps,deception_steps,deception_events} "
            "of squared relative deviation from the split-size proportional target"
        ),
        "objective_exact": f"{score.numerator}/{score.denominator}",
        "objective_value": float(score),
        "enumerated_assignments": enumerated_assignments,
        "feasible_assignments": feasible_assignments,
        "tie_break_sha256": tie_break,
        "stratification_inputs_sha256": stratification_inputs_sha256,
        "assignment_sha256": assignment_sha256,
        "split_sessions": splits,
        "split_counts": split_counts,
    }
    return splits, strategy


def _feature_mask(session: PreparedSession, feature: str) -> np.ndarray:
    if feature in PHYSIOLOGY_MODALITIES:
        return session.physiology_mask[:, PHYSIOLOGY_MODALITIES.index(feature)]
    return session.modality_mask[:, BEHAVIOR_MODALITIES.index(feature)]


def fit_normalization(
    sessions: Sequence[PreparedSession], train_sessions: Iterable[str]
) -> dict[str, dict[str, np.ndarray]]:
    train_set = set(train_sessions)
    stats: dict[str, dict[str, np.ndarray]] = {}
    for feature in FEATURE_NAMES:
        available = [
            session.features[feature][_feature_mask(session, feature)]
            for session in sessions
            if session.session_id in train_set and _feature_mask(session, feature).any()
        ]
        dimension = sessions[0].features[feature].shape[1]
        if available:
            values = np.concatenate(available, axis=0).astype(np.float64)
            mean = values.mean(axis=0)
            scale = values.std(axis=0)
            scale[scale < 1e-8] = 1.0
        else:
            mean = np.zeros(dimension, dtype=np.float64)
            scale = np.ones(dimension, dtype=np.float64)
        stats[feature] = {"mean": mean, "scale": scale}
    return stats


def apply_normalization(
    sessions: Sequence[PreparedSession], stats: Mapping[str, Mapping[str, np.ndarray]]
) -> None:
    for session in sessions:
        for feature in FEATURE_NAMES:
            mask = _feature_mask(session, feature)
            values = session.features[feature]
            normalized = (values - stats[feature]["mean"]) / stats[feature]["scale"]
            normalized = np.asarray(normalized, dtype=np.float32)
            normalized[~mask] = 0.0
            if not np.isfinite(normalized).all():
                raise ValueError(f"Non-finite {feature} values for subject {session.subject_id}")
            session.features[feature] = normalized


def _compact_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(values).copy())


def write_session_windows(
    session: PreparedSession,
    split: str,
    output_root: Path,
    window_size: int,
    stride: int,
    min_window_size: int,
) -> list[dict]:
    window_root = output_root / "sessions" / session.session_id / "windows"
    window_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for window_index, start in enumerate(
        window_starts(len(session.labels), window_size, stride, min_window_size)
    ):
        end = min(start + window_size, len(session.labels))
        sample_id = f"{session.session_id}_{split}_window_{window_index:04d}"
        filename = f"{sample_id}.pt"
        sample = {
            **{
                feature: _compact_tensor(values[start:end])
                for feature, values in session.features.items()
            },
            "physiology_mask": _compact_tensor(session.physiology_mask[start:end]),
            "modality_mask": _compact_tensor(session.modality_mask[start:end]),
            "target_mask": _compact_tensor(session.target_mask[start:end]),
            "labels": _compact_tensor(session.labels[start:end]),
            "boundaries": _compact_tensor(session.boundaries[start:end]),
            "offsets": _compact_tensor(session.offsets[start:end]),
            "positive_mask": _compact_tensor(session.positive_mask[start:end]),
            "timestamps": _compact_tensor(session.timestamps[start:end]),
            "row_indices": torch.arange(start, end, dtype=torch.long),
            "words": session.step_text[start:end],
        }
        atomic_torch_save(window_root / filename, sample)
        records.append(
            {
                "sample_id": sample_id,
                "tensor_file": f"../sessions/{session.session_id}/windows/{filename}",
                "metadata": {
                    "dataset": session.dataset_name,
                    "session_id": session.session_id,
                    "subject_id": session.subject_id,
                    "split": split,
                    "source_row_start": start,
                    "source_row_end_exclusive": end,
                    "start_time_seconds": float(session.timestamps[start, 0]),
                    "end_time_seconds": float(session.timestamps[end - 1, 1]),
                },
            }
        )
    return records


def _write_json(path: Path, value: object) -> None:
    atomic_write_json(path, value)


def _is_facial_csv_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("_facial_actions.csv") or name.endswith("_openface_features.csv")


def validate_reused_staged_raw(
    raw_root: Path,
    sessions: Sequence[SourceSession],
    *,
    dataset_name: str,
    include_facial_csv: bool = True,
) -> dict[str, object]:
    """Verify an existing normalized raw tree without copying or rewriting it.

    The returned manifest is deliberately de-identified and contains only paths
    relative to the staged root.  Original source names and absolute paths remain
    confined to the private raw-data layer.
    """

    raw_root = raw_root.resolve()
    session_records: list[dict[str, object]] = []
    all_files: list[dict[str, object]] = []
    excluded_facial_records = 0
    for session in sorted(sessions, key=lambda item: item.session_id):
        if session.root.resolve().parent != raw_root:
            raise ValueError(f"Session escapes staged raw root: {session.root}")
        metadata = _load_session_metadata(session.root)
        if metadata is None:
            raise ValueError(f"Missing staged metadata for {session.session_id}")
        raw_files = metadata.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError(f"Missing file ledger for {session.session_id}")
        verified_files: list[dict[str, object]] = []
        seen_paths: set[Path] = set()
        for raw_record in raw_files:
            if not isinstance(raw_record, dict):
                raise ValueError(f"Invalid file ledger entry for {session.session_id}")
            relative = raw_record.get("staged_relative_path")
            expected_hash = raw_record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise ValueError(f"Incomplete file ledger entry for {session.session_id}")
            relative_path = Path(relative)
            if not include_facial_csv and _is_facial_csv_path(relative_path):
                excluded_facial_records += 1
                continue
            expected_hash = expected_hash.strip().lower()
            if len(expected_hash) != 64 or any(
                character not in "0123456789abcdef" for character in expected_hash
            ):
                raise ValueError(f"Invalid SHA-256 ledger value for {session.session_id}")
            if relative_path.is_absolute():
                raise ValueError(f"Absolute staged path is forbidden: {relative}")
            staged_path = (raw_root / relative_path).resolve()
            try:
                staged_path.relative_to(raw_root)
            except ValueError as error:
                raise ValueError(f"Staged path escapes raw root: {relative}") from error
            allowed_parents = {
                session.root.resolve(),
                (raw_root / "facial_csv" / session.session_id).resolve(),
            }
            if staged_path.parent not in allowed_parents:
                raise ValueError(
                    f"File ledger crosses session boundary for {session.session_id}: {relative}"
                )
            if staged_path in seen_paths:
                raise ValueError(f"Duplicate staged path in ledger: {relative}")
            seen_paths.add(staged_path)
            if not staged_path.is_file():
                raise FileNotFoundError(staged_path)
            actual_hash = package_sha256_file(staged_path)
            if actual_hash != expected_hash:
                raise ValueError(f"Raw checksum mismatch: {staged_path}")
            record = {
                "session_id": session.session_id,
                "role": staged_path.stem.removeprefix(f"{session.session_id}_"),
                "staged_relative_path": relative_path.as_posix(),
                "bytes": staged_path.stat().st_size,
                "sha256": actual_hash,
            }
            verified_files.append(record)
            all_files.append(record)
        session_records.append(
            {
                "session_id": session.session_id,
                "participant_id": session.session_id,
                "raw_path": session.root.relative_to(raw_root).as_posix(),
                "files": verified_files,
            }
        )
    included_ids = {str(record["session_id"]) for record in session_records}
    excluded_sessions: list[dict[str, object]] = []
    for session_root in sorted(path for path in raw_root.iterdir() if path.is_dir()):
        if SESSION_ID_PATTERN.fullmatch(session_root.name) is None:
            continue
        if session_root.name in included_ids:
            continue
        metadata = _load_session_metadata(session_root)
        if metadata is None:
            raise ValueError(f"Untracked canonical session directory: {session_root}")
        availability = metadata.get("available_sources", {})
        annotated = isinstance(availability, dict) and bool(
            availability.get("annotated_transcript", False)
        )
        excluded_sessions.append(
            {
                "session_id": session_root.name,
                "reason": (
                    "missing_annotated_transcript"
                    if not annotated
                    else "not_discovered_despite_declared_annotation"
                ),
                "available_sources": {
                    str(name): bool(value)
                    for name, value in availability.items()
                    if isinstance(name, str)
                }
                if isinstance(availability, dict)
                else {},
            }
        )
    return {
        "schema_version": 1,
        "dataset": dataset_name,
        "source_kind": "checksum_verified_staged_raw",
        "deidentified": True,
        "session_count": len(session_records),
        "sessions": session_records,
        "excluded_session_count": len(excluded_sessions),
        "excluded_sessions": excluded_sessions,
        "source_policy": {
            "include_facial_csv": include_facial_csv,
            "excluded_facial_csv_records": excluded_facial_records,
        },
        "files": all_files,
    }


def _normalized_raw_names(
    session: SourceSession, *, include_facial_csv: bool = True
) -> dict[Path, str]:
    names: dict[Path, str] = {}
    excluded_paths = {
        path.resolve()
        for path in (session.face_csv, session.openface_csv)
        if path is not None and not include_facial_csv
    }

    def register(path: Path | None, suffix: str) -> None:
        if path is not None:
            names[path] = f"{session.session_id}_{suffix}"

    register(session.annotation, "annotations.docx")
    if include_facial_csv:
        register(session.face_csv, "facial_actions.csv")
        register(session.openface_csv, "openface_features.csv")
    register(session.eeg_file, "eeg_raw.txt")
    if session.ppg_files is not None:
        register(session.ppg_files[0], "ppg_channel_1.txt")
        register(session.ppg_files[1], "ppg_channel_2.txt")
    if session.audio_file is not None:
        register(session.audio_file, f"audio{session.audio_file.suffix.lower()}")
    if session.video_file is not None:
        register(session.video_file, f"video{session.video_file.suffix.lower()}")

    transcript_index = 0
    supplementary_index = 0
    for path in sorted(value for value in session.root.iterdir() if value.is_file()):
        if path in names or path.resolve() in excluded_paths:
            continue
        if path.suffix.lower() == ".docx":
            transcript_index += 1
            names[path] = f"{session.session_id}_transcript_unmarked_{transcript_index:02d}.docx"
        else:
            supplementary_index += 1
            names[path] = (
                f"{session.session_id}_supplementary_{supplementary_index:02d}{path.suffix.lower()}"
            )
    return names


def stage_raw_sessions(
    source_root: Path,
    raw_root: Path,
    sessions: Sequence[SourceSession],
    *,
    dataset_name: str = DATASET_NAME,
    include_facial_csv: bool = True,
) -> dict:
    """Copy every source artifact into matching session_00n raw directories."""
    raw_root.mkdir(parents=True, exist_ok=True)
    all_files = []
    session_index = []
    for session in sessions:
        session_root = raw_root / session.session_id
        session_root.mkdir(parents=True, exist_ok=True)
        records = []
        for source, normalized_name in _normalized_raw_names(
            session, include_facial_csv=include_facial_csv
        ).items():
            destination = session_root / normalized_name
            source_hash = sha256_file(source)
            if not destination.exists() or sha256_file(destination) != source_hash:
                shutil.copy2(source, destination)
            if sha256_file(destination) != source_hash:
                raise OSError(f"Checksum mismatch after staging {source} to {destination}")
            record = {
                "session_id": session.session_id,
                "subject_id": session.subject_id,
                "source_relative_path": source.relative_to(source_root).as_posix(),
                "staged_relative_path": destination.relative_to(raw_root).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": source_hash,
            }
            records.append(record)
            all_files.append(record)
        metadata = {
            "dataset": dataset_name,
            "session_id": session.session_id,
            "subject_id": session.subject_id,
            "annotation_mode": session.annotation_mode,
            "available_sources": {
                "eeg": session.eeg_file is not None,
                "paired_ppg": session.ppg_files is not None,
                "facial_actions": session.face_csv is not None,
                "openface_features": session.openface_csv is not None,
                "audio": session.audio_file is not None,
                "video": session.video_file is not None,
                "annotated_transcript": True,
            },
            "missing_source_reasons": {
                name: "source_artifact_absent"
                for name, available in {
                    "eeg": session.eeg_file is not None,
                    "paired_ppg": session.ppg_files is not None,
                    "facial_actions": session.face_csv is not None,
                    "openface_features": session.openface_csv is not None,
                    "audio": session.audio_file is not None,
                    "video": session.video_file is not None,
                    "annotated_transcript": True,
                }.items()
                if not available
            },
            "files": records,
        }
        _write_json(session_root / "session_metadata.json", metadata)
        _write_json(session_root / "source_manifest.json", metadata)
        session_index.append(
            {
                "session_id": session.session_id,
                "subject_id": session.subject_id,
                "raw_path": session_root.relative_to(raw_root.parent.parent).as_posix(),
            }
        )
    manifest = {
        "dataset": dataset_name,
        "source_root": str(source_root.resolve()),
        "raw_root": str(raw_root.resolve()),
        "source_policy": {
            "include_facial_csv": include_facial_csv,
            "excluded_facial_csv_records": sum(
                int(session.face_csv is not None) + int(session.openface_csv is not None)
                for session in sessions
            )
            if not include_facial_csv
            else 0,
        },
        "sessions": session_index,
        "files": all_files,
    }
    _write_json(raw_root / "source_manifest.json", manifest)
    _write_json(
        raw_root / "session_index.json", {"dataset": dataset_name, "sessions": session_index}
    )
    return manifest


def _safe_reset_processed_sessions(output_root: Path) -> None:
    target = (output_root / "sessions").resolve()
    if target.parent != output_root.resolve():
        raise ValueError(f"Refusing to reset unexpected processed path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _safe_reset_processed_manifests(output_root: Path) -> None:
    target = (output_root / "manifests").resolve()
    if target.parent != output_root.resolve():
        raise ValueError(f"Refusing to reset unexpected manifest path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _safe_remove_legacy_windows(output_root: Path) -> None:
    target = (output_root / "windows").resolve()
    if target.parent != output_root.resolve():
        raise ValueError(f"Refusing to remove unexpected legacy path: {target}")
    if target.exists():
        shutil.rmtree(target)


def _write_data_catalog(data_root: Path, session_count: int, dataset_name: str) -> None:
    bci_entries = []
    for name, config in (
        (DATASET_NAME, "../configs/bci_subjects.yaml"),
        (dataset_name, "../configs/bci_subjects_official_text.yaml"),
    ):
        if any(entry["name"] == name for entry in bci_entries):
            continue
        bci_entries.append(
            {
                "name": name,
                "session_count": session_count,
                "raw_root": f"raw/{name}",
                "processed_root": f"processed/{name}",
                "train_manifest": f"processed/{name}/manifests/train.jsonl",
                "val_manifest": f"processed/{name}/manifests/val.jsonl",
                "test_manifest": f"processed/{name}/manifests/test.jsonl",
                "all_manifest": f"processed/{name}/manifests/all.jsonl",
                "config": config,
            }
        )
    catalog = {
        "layout_version": 2,
        "session_naming": "session_001 ... session_NNN",
        "datasets": [
            {
                "name": "bci_truth_deception_v1",
                "session_count": 1,
                "raw_root": "raw/bci_truth_deception_v1",
                "processed_root": "processed/bci_truth_deception_v1",
                "train_manifest": "processed/bci_truth_deception_v1/manifests/train.jsonl",
                "val_manifest": "processed/bci_truth_deception_v1/manifests/val.jsonl",
                "test_manifest": "processed/bci_truth_deception_v1/manifests/test.jsonl",
                "all_manifest": "processed/bci_truth_deception_v1/manifests/all.jsonl",
                "config": "../configs/default.yaml",
            },
            *bci_entries,
        ],
    }
    _write_json(data_root / "catalog.json", catalog)


def _dataset_readme(summary: Mapping[str, object]) -> str:
    dataset_name = summary["dataset"]
    dimensions = summary["feature_dimensions"]
    split_sessions = summary["split_sessions"]
    split_strategy = summary["split_strategy"]
    split_sizes = split_strategy["split_sizes"]
    behavior = summary["behavior_features"]
    compatibility_note = (
        "- Compatibility window manifests are also available as "
        "`manifests/{train,val,test,all}.jsonl`."
        if summary["storage_mode"] == "session_timelines_plus_compatibility_windows"
        else "- Overlapping compatibility windows are omitted to avoid duplicate storage."
    )
    alignment_note = (
        "- Sensor alignment: direct absolute intersection is attempted first; confirmed "
        "same-session EEG with a non-overlapping device clock receives one documented "
        "session-end constant offset. Within-stream timing and gaps are preserved."
        if summary.get("alignment_mode") == "session_registered_time"
        else
        "- Sensor alignment: EEG row epochs and PPG acquisition-start/sample-rate clocks "
        "are intersected with the absolute UTC+08:00 transcript bins. Non-overlap is "
        "zero-filled with a false mask; clocks were not hardware-triggered together."
        if summary.get("alignment_mode") == "absolute_time"
        else "- Sensor alignment: each source stream is duration-normalized to its own "
        "annotated interview timeline because acquisition clocks are incomplete/inconsistent."
    )
    return f"""# {dataset_name}

This is a self-contained processed dataset generated from the local acquisition
collection. It includes real serialized session tensors, portable relative manifests,
feature/alignment metadata, and checksums; raw recordings are not included. It can be
loaded with `eptnet.data.ManifestDataset` or directly with PyTorch after extraction.

- Steps: fixed {summary["step_seconds"]}-second causal intervals.
- Labels: `0=deception`, `1=truth`; highlighted/bold transcript spans are deception.
- Target identity: inferred before annotation marks are read, using the speaker with
  the largest total number of non-whitespace transcript characters. This is a
  retrospective, label-independent conversation-length heuristic, not supplied
  participant-role metadata or an online role-discovery method. The winning character
  share and runner-up margin are recorded per private session summary.
- Annotation consistency: after target identity is frozen, at least 95% of marked
  characters must belong to that speaker or preparation fails closed. Any remaining
  non-target marks are retained only in audit counts and cannot create labels/events.
- `target_mask`: valid only in bins where the text-inferred target speaker contributes
  a strict majority of assigned transcript characters; empty and tied bins are invalid.
- Dimensions: `{json.dumps(dimensions, ensure_ascii=False)}`.
- Subject-disjoint split: `{json.dumps(split_sessions, ensure_ascii=False)}`.
- Split protocol: `{split_strategy["algorithm"]}`, model-independent and frozen before
  training; labels and modality availability are used only to balance this fixed split.
- This {split_sizes["train"]}/{split_sizes["val"]}/{split_sizes["test"]} fixed split is
  not cross-validation or population-level validation.
- Raw and processed subjects use stable `session_NNN` pseudonyms for the
  {summary["num_subjects"]} included participants.
- `manifests/sessions_{{train,val,test,all}}.jsonl` loads one complete non-overlapping
  timeline per session. Every record is SHA-256 verified by `ManifestDataset`.
{compatibility_note}
{alignment_note}
- Missing streams are zero-filled only after normalization and are explicitly marked
  by `physiology_mask` (`eeg_time/eeg_spectral/physiology`) or `modality_mask`
  (`video/audio/text`).
- Frozen behavior feature contract: `{json.dumps(behavior, ensure_ascii=False)}`.
- Missing raw media remain unavailable and are never replaced by fabricated features.
- `FILES.sha256` verifies every file in a downloadable package. `PORTABLE_PACKAGE.json`
  identifies the entry manifests and records the portability boundary.

Minimal direct loading without the EPT-Net repository:

```python
import hashlib
import json
from pathlib import Path

import torch

root = Path("{dataset_name}")
manifest = root / "manifests" / "sessions_train.jsonl"
record = json.loads(next(line for line in manifest.read_text(encoding="utf-8").splitlines() if line))
tensor_path = (manifest.parent / record["tensor_file"]).resolve()
assert hashlib.sha256(tensor_path.read_bytes()).hexdigest() == record["sha256"]
sample = torch.load(tensor_path, map_location="cpu", weights_only=True)
print(sample["labels"].shape, sample["physiology_mask"].shape)
```

Technical portability does not itself grant public redistribution rights. Confirm
participant consent, data governance, and a dataset license before public release.
"""


def _build_alignment_report(
    sessions: Sequence[PreparedSession],
    *,
    dataset_name: str,
    include_facial_csv: bool,
    video_backend: str,
    staged_manifest: Mapping[str, object],
) -> dict[str, object]:
    alignment_mode = str(sessions[0].alignment["alignment_mode"])
    uses_absolute_time = alignment_mode != "duration_normalized"
    uses_whisper = all(session.alignment.get("whisper_alignment") for session in sessions)
    sessions_by_id = {session.session_id: session for session in sessions}
    facial_records = sum(
        1
        for record in staged_manifest.get("files", [])
        if isinstance(record, dict)
        and str(record.get("role", "")) in {"facial_actions", "openface_features"}
    )
    return {
        "schema_version": 3 if uses_whisper else 2,
        "dataset": dataset_name,
        "reference_timeline": "external_audio" if uses_whisper else "annotated_transcript",
        "step_seconds": float(sessions[0].step_seconds),
        "alignment_scope": (
            "session_registered_timestamp_intersection"
            if alignment_mode == "session_registered_time"
            else "absolute_timestamp_intersection"
            if uses_absolute_time
            else "coarse_duration_normalization"
        ),
        "physical_clock_synchronization": False,
        "clock_policy": (
            "direct_absolute_clocks_then_confirmed_session_end_constant_offset"
            if alignment_mode == "session_registered_time"
            else "local_utc_plus_08_absolute_clocks_without_shared_hardware_trigger"
            if uses_absolute_time
            else "relative_duration_only"
        ),
        "methods": {
            "transcript": (
                "openai_whisper_word_timestamps_plus_exact_reference_character_mapping"
                if uses_whisper
                else "uniform_character_interpolation_between_speaker_timestamps"
            ),
            "eeg": (
                "OpenBCI epoch intersection with optional confirmed-session end offset"
                if alignment_mode == "session_registered_time"
                else "OpenBCI embedded epoch intersection"
                if uses_absolute_time
                else "uniform_index_bins_on_transcript_timeline"
            ),
            "physiology": (
                "PPG filename start plus 385 Hz empirical device rate intersection"
                if uses_absolute_time
                else "uniform_index_bins_on_transcript_timeline_after_common_length_crop"
            ),
            "facial_csv": "excluded" if not include_facial_csv else "configured_backend",
            "video": (
                "official_marlin_embeddings_at_audio_time_plus_embedded_audio_offset"
                if video_backend == "marlin" and uses_whisper
                else "official_marlin_embeddings_from_paired_raw_video"
                if video_backend == "marlin"
                else "official_openface_features_from_paired_raw_video"
                if video_backend == "openface"
                else "legacy_facial_csv_features"
                if include_facial_csv
                else "disabled"
            ),
            "audio": "selected_behavior_backend",
            "video_audio_offset": (
                "embedded_video_audio_vs_external_audio_rms_cross_correlation"
                if uses_whisper
                else "not_applied"
            ),
        },
        "acceptance_criteria": {
            "session_count": len(sessions),
            "shared_time_axis_per_session": True,
            "contiguous_one_second_bins": all(session.step_seconds == 1.0 for session in sessions),
            "facial_source_records": 0 if not include_facial_csv else facial_records,
            "video_masks_all_false": all(
                not session.modality_mask[:, 0].any() for session in sessions
            ),
            "session_024_eeg_rejected": (
                sessions_by_id["session_024"].alignment["eeg"].get("status")
                == "no_overlap"
                if alignment_mode == "absolute_time" and "session_024" in sessions_by_id
                else None
            ),
            "session_024_eeg_registered": (
                bool(
                    sessions_by_id["session_024"].alignment["eeg"].get(
                        "clock_offset_applied"
                    )
                )
                and int(
                    sessions_by_id["session_024"].alignment["eeg"].get(
                        "valid_steps", 0
                    )
                )
                > 0
                if alignment_mode == "session_registered_time"
                and "session_024" in sessions_by_id
                else None
            ),
        },
        "aggregate": {
            "sessions": len(sessions),
            "eeg_sessions_present": sum(
                bool(session.alignment["eeg"].get("present")) for session in sessions
            ),
            "physiology_sessions_present": sum(
                bool(session.alignment["physiology"].get("present")) for session in sessions
            ),
            "eeg_sessions_with_overlap": sum(
                bool(session.alignment["eeg"].get("valid_steps")) for session in sessions
            ),
            "physiology_sessions_with_overlap": sum(
                bool(session.alignment["physiology"].get("valid_steps"))
                for session in sessions
            ),
            "eeg_valid_steps": sum(
                int(session.alignment["eeg"].get("valid_steps", 0)) for session in sessions
            ),
            "eeg_clock_registered_sessions": sum(
                bool(session.alignment["eeg"].get("clock_offset_applied"))
                for session in sessions
            ),
            "physiology_valid_steps": sum(
                int(session.alignment["physiology"].get("valid_steps", 0))
                for session in sessions
            ),
            "timeline_origin_evidence_counts": dict(
                sorted(Counter(
                    str(session.alignment["timeline_origin_evidence"])
                    for session in sessions
                ).items())
            ),
            "facial_source_records": facial_records,
        },
        "sessions": {
            session.session_id: session.alignment
            for session in sorted(sessions, key=lambda item: item.session_id)
        },
        "interpretation": (
            "Sensor features exist only for one-second transcript bins with an "
            "absolute-time intersection. Device clocks were not driven by one shared "
            "hardware trigger, so residual clock error remains and sub-second lag "
            "claims are unsupported."
            if alignment_mode == "absolute_time"
            else "Confirmed same-session EEG first uses direct absolute intersection; "
            "when device clocks do not overlap, one session-end constant offset is "
            "applied while preserving embedded relative timing and gaps. This is clock "
            "registration, not shared hardware synchronization, so sub-second lag "
            "claims remain unsupported."
            if alignment_mode == "session_registered_time"
            else "All included tensors share the transcript-derived timeline. Duration "
            "normalization supports coarse multimodal fusion, not physiological lag "
            "or hardware-clock synchronization claims."
        ),
    }


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    raw_root: Path | None = None,
    window_size: int = 64,
    stride: int = 32,
    min_window_size: int = 16,
    step_seconds: float = STEP_SECONDS,
    split_seed: int = 42,
    dataset_name: str = DATASET_NAME,
    behavior_config: BehaviorFeatureConfig | None = None,
    required_profile: str = "model_contract",
    reuse_staged_raw: bool = False,
    overwrite: bool = False,
    write_compatibility_windows: bool = True,
    include_facial_csv: bool = True,
    alignment_mode: AlignmentMode = "duration_normalized",
    complete_behavior_sources_only: bool = False,
    whisper_alignment_dir: Path | None = None,
    require_whisper_alignment: bool = False,
    excluded_session_ids: Sequence[str] = (),
) -> dict:
    if not dataset_name.strip():
        raise ValueError("dataset_name must be non-empty")
    if alignment_mode not in {
        "duration_normalized",
        "absolute_time",
        "session_registered_time",
    }:
        raise ValueError(f"Unsupported alignment_mode: {alignment_mode}")
    behavior_config = behavior_config or BehaviorFeatureConfig()
    behavior_config.validate()
    behavior_pipeline = BehaviorFeaturePipeline(behavior_config)
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    whisper_alignment_dir = (
        whisper_alignment_dir.expanduser().resolve() if whisper_alignment_dir else None
    )
    if require_whisper_alignment and whisper_alignment_dir is None:
        raise ValueError("--require-whisper-alignment requires --whisper-alignment-dir")
    raw_root = (
        raw_root.expanduser().resolve()
        if raw_root is not None
        else output_root.parent.parent / "raw" / dataset_name
    )
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty processed dataset {output_root}. "
            "Choose a new versioned output path or pass --overwrite explicitly."
        )
    if reuse_staged_raw and raw_root != source_root:
        raise ValueError(
            "--reuse-staged-raw requires --source and --raw-output to resolve to the same path"
        )
    discovered = discover_sessions(source_root)
    staged_manifest = (
        validate_reused_staged_raw(
            raw_root,
            discovered,
            dataset_name=dataset_name,
            include_facial_csv=include_facial_csv,
        )
        if reuse_staged_raw
        else stage_raw_sessions(
            source_root,
            raw_root,
            discovered,
            dataset_name=dataset_name,
            include_facial_csv=include_facial_csv,
        )
    )
    behavior_source_exclusions: list[dict[str, object]] = []
    explicit_exclusions = set(excluded_session_ids)
    discovered_ids = {source.session_id for source in discovered}
    unknown_exclusions = explicit_exclusions.difference(discovered_ids)
    if unknown_exclusions:
        raise ValueError(f"Unknown session exclusions: {sorted(unknown_exclusions)}")
    if explicit_exclusions:
        for session_id in sorted(explicit_exclusions):
            behavior_source_exclusions.append(
                {
                    "session_id": session_id,
                    "missing_sources": [],
                    "reason": "explicit_alignment_quality_exclusion",
                }
            )
        discovered = [
            source for source in discovered if source.session_id not in explicit_exclusions
        ]
        if len(discovered) < 3:
            raise ValueError("Explicit exclusions leave fewer than three sessions")
    if complete_behavior_sources_only:
        selected: list[SourceSession] = []
        for source in discovered:
            missing = [
                name
                for name, available in (
                    ("video", source.video_file is not None),
                    ("audio", source.audio_file is not None),
                    ("text", source.annotation is not None),
                )
                if not available
            ]
            if missing:
                behavior_source_exclusions.append(
                    {"session_id": source.session_id, "missing_sources": missing}
                )
            else:
                selected.append(source)
        discovered = selected
        if len(discovered) < 3:
            raise ValueError(
                "Complete behavior-source filtering left fewer than three sessions; "
                "video, audio, and annotated text are all required"
            )
    model_sources = (
        discovered
        if include_facial_csv
        else [replace(source, face_csv=None, openface_csv=None) for source in discovered]
    )
    sessions: list[PreparedSession] = []
    for index, source in enumerate(model_sources, start=1):
        alignment_artifact = None
        if whisper_alignment_dir is not None:
            artifact_path = whisper_alignment_dir / f"{source.session_id}.json"
            if artifact_path.is_file():
                alignment_artifact = load_alignment(artifact_path)
                expected_sources = {
                    "annotation_sha256": source.annotation,
                    "audio_sha256": source.audio_file,
                    "video_sha256": source.video_file,
                }
                if alignment_artifact.get("session_id") != source.session_id:
                    raise ValueError(f"Whisper session mismatch: {artifact_path}")
                for field, path in expected_sources.items():
                    if path is None or alignment_artifact.get(field) != sha256_file(path):
                        raise ValueError(f"Whisper source fingerprint mismatch: {field} in {artifact_path}")
            elif require_whisper_alignment:
                raise FileNotFoundError(f"Required Whisper alignment is missing: {artifact_path}")
        print(
            json.dumps(
                {
                    "event": "prepare_session_start",
                    "session_id": source.session_id,
                    "index": index,
                    "total": len(model_sources),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        sessions.append(
            prepare_session(
                source,
                step_seconds=step_seconds,
                behavior_config=behavior_config,
                behavior_pipeline=behavior_pipeline,
                dataset_name=dataset_name,
                alignment_mode=alignment_mode,
                whisper_alignment=alignment_artifact,
            )
        )
        print(
            json.dumps(
                {
                    "event": "prepare_session_complete",
                    "session_id": source.session_id,
                    "index": index,
                    "total": len(model_sources),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    compliance = build_session_compliance(sessions, required_profile=required_profile)
    if not reuse_staged_raw:
        _write_json(raw_root / "session_compliance.json", compliance)
        (raw_root / "session_compliance.md").write_text(
            render_session_compliance_markdown(compliance), encoding="utf-8", newline="\n"
        )
    if not compliance["all_sessions_compliant"]:
        failed = compliance["selected_profile_failed_check_counts"]
        raise ValueError(
            f"Session compliance profile {required_profile!r} failed: "
            f"{compliance['compliant_sessions']}/{compliance['total_sessions']} sessions pass; "
            f"missing requirements={failed}. See {raw_root / 'session_compliance.md'}"
        )
    splits, split_strategy = assign_constrained_subject_splits(sessions, seed=split_seed)
    stats = fit_normalization(sessions, splits["train"])
    apply_normalization(sessions, stats)
    output_root.mkdir(parents=True, exist_ok=True)
    _safe_reset_processed_sessions(output_root)
    _safe_reset_processed_manifests(output_root)
    manifest_root = output_root / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    by_session = {session.session_id: session for session in sessions}
    all_records = []
    all_session_records: list[dict[str, object]] = []
    window_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {}
    session_to_split = {
        session_id: split for split, session_ids in splits.items() for session_id in session_ids
    }
    for split, session_ids in splits.items():
        records = []
        session_records: list[dict[str, object]] = []
        for session_id in session_ids:
            session_records.append(
                write_session_package(by_session[session_id], split, output_root)
            )
            if write_compatibility_windows:
                records.extend(
                    write_session_windows(
                        by_session[session_id],
                        split,
                        output_root,
                        window_size,
                        stride,
                        min_window_size,
                    )
                )
        if write_compatibility_windows:
            atomic_write_jsonl(manifest_root / f"{split}.jsonl", records)
        atomic_write_jsonl(manifest_root / f"sessions_{split}.jsonl", session_records)
        all_records.extend(records)
        all_session_records.extend(session_records)
        window_counts[split] = len(records)
        session_counts[split] = len(session_records)
    all_records.sort(key=lambda record: (record["metadata"]["session_id"], record["sample_id"]))
    if write_compatibility_windows:
        atomic_write_jsonl(manifest_root / "all.jsonl", all_records)
    all_session_records.sort(key=lambda record: str(record["sample_id"]))
    atomic_write_jsonl(manifest_root / "sessions_all.jsonl", all_session_records)

    np.savez(
        output_root / "normalization_stats.npz",
        **{
            f"{feature}_{statistic}": values
            for feature, feature_stats in stats.items()
            for statistic, values in feature_stats.items()
        },
    )
    feature_dimensions = {
        "eeg_time": EEG_TIME_DIM,
        "eeg_spectral": EEG_SPECTRAL_DIM,
        "physiology": PHYSIOLOGY_DIM,
        "video": behavior_config.video_dim,
        "audio": behavior_config.audio_dim,
        "text": behavior_config.text_dim,
    }
    feature_schema = {
        "dataset": dataset_name,
        "input_mode": "precomputed_features",
        "source_inclusion_policy": {
            "include_facial_csv": include_facial_csv,
            "facial_csv_processed": include_facial_csv,
            "complete_behavior_sources_only": complete_behavior_sources_only,
            "behavior_source_exclusions": behavior_source_exclusions,
        },
        "alignment_mode": alignment_mode,
        "feature_dimensions": feature_dimensions,
        "eeg_spectral_order": "channel-major [delta, theta, alpha, beta, gamma] log power",
        "physiology_order": "20 statistics for ch2 followed by 20 statistics for ch3",
        "video_order": (
            [f"marlin_{index:03d}" for index in range(behavior_config.video_dim)]
            if behavior_config.video_backend == "marlin"
            else list(OPENFACE_FEATURE_NAMES)
            if behavior_config.video_backend == "openface"
            else ["blink", "pressed_lips", "furrow_brow"]
            if include_facial_csv
            else []
        ),
        "video": (
            "official MARLIN-small frozen embeddings extracted from paired raw video"
            if behavior_config.video_backend == "marlin"
            else "official OpenFace features extracted from paired raw video"
            if behavior_config.video_backend == "openface"
            else "configured legacy behavior-video features"
            if include_facial_csv
            else "zero placeholder; facial CSV excluded and modality_mask is false"
        ),
        "audio": (
            "zero placeholder; modality_mask is false"
            if behavior_config.audio_backend == "none"
            else "Microsoft WavLM Base+ last-hidden-state pooled per causal step"
        ),
        "text": (
            "deterministic signed character unigram/bigram hashing"
            if behavior_config.text_backend == "hash"
            else "HFL Chinese MacBERT-base CLS embedding per causal step"
        ),
        "behavior_features": behavior_config.provenance(),
        "physiology_mask_order": list(PHYSIOLOGY_MODALITIES),
        "modality_mask_order": list(BEHAVIOR_MODALITIES),
        "target_mask": (
            "target identity is frozen from the complete transcript by maximum total "
            "non-whitespace characters before annotation marks are read; true only "
            "when that speaker has a strict majority of characters assigned to the "
            "timestep"
        ),
        "label_mapping": {"0": "deception", "1": "truth"},
        "event_positive_label": 0,
    }
    _write_json(output_root / "feature_schema.json", feature_schema)
    _write_json(output_root / "source_manifest.json", staged_manifest)
    alignment_report = _build_alignment_report(
        sessions,
        dataset_name=dataset_name,
        include_facial_csv=include_facial_csv,
        video_backend=behavior_config.video_backend,
        staged_manifest=staged_manifest,
    )
    _write_json(output_root / "alignment_report.json", alignment_report)
    if reuse_staged_raw:
        _write_json(
            output_root / "excluded_sessions.json",
            {
                "dataset": dataset_name,
                "policy": "excluded_before_split_and_normalization",
                "sessions": staged_manifest.get("excluded_sessions", []),
            },
        )
    _write_json(output_root / "split_strategy.json", split_strategy)
    _write_json(output_root / "session_compliance.json", compliance)
    (output_root / "session_compliance.md").write_text(
        render_session_compliance_markdown(compliance), encoding="utf-8", newline="\n"
    )

    session_summaries = {}
    for session in sessions:
        session_summary = {
            "session_id": session.session_id,
            "subject_id": session.subject_id,
            "split": session_to_split[session.session_id],
            "num_steps": len(session.labels),
            "duration_seconds": session.duration_seconds,
            "target_valid_steps": int(session.target_mask.sum()),
            "label_counts": {
                "0": int((session.labels == 0).sum()),
                "1": int((session.labels == 1).sum()),
            },
            "target_label_counts": {
                "0": int(((session.labels == 0) & session.target_mask).sum()),
                "1": int(((session.labels == 1) & session.target_mask).sum()),
                "ignored": int((~session.target_mask).sum()),
            },
            "deception_events": int(session.boundaries[:, 0].sum()),
            "fully_missing_steps": int(
                (~session.physiology_mask.any(axis=1) & ~session.modality_mask.any(axis=1)).sum()
            ),
            "available_steps": {
                **{
                    name: int(session.physiology_mask[:, index].sum())
                    for index, name in enumerate(PHYSIOLOGY_MODALITIES)
                },
                **{
                    name: int(session.modality_mask[:, index].sum())
                    for index, name in enumerate(BEHAVIOR_MODALITIES)
                },
            },
            "sources": {
                "annotation": session.source.annotation.name,
                "face_csv": session.source.face_csv.name if session.source.face_csv else None,
                "openface_csv": (
                    session.source.openface_csv.name if session.source.openface_csv else None
                ),
                "ppg": [path.name for path in session.source.ppg_files]
                if session.source.ppg_files
                else None,
                "eeg": session.source.eeg_file.name if session.source.eeg_file else None,
                "audio": session.source.audio_file.name if session.source.audio_file else None,
                "video": session.source.video_file.name if session.source.video_file else None,
            },
            "annotation": session.annotation_statistics,
            "alignment": session.alignment,
        }
        session_summaries[session.session_id] = session_summary
        _write_json(
            output_root / "sessions" / session.session_id / "session_summary.json",
            session_summary,
        )
    total_labels = np.concatenate([session.labels for session in sessions])
    total_target_mask = np.concatenate([session.target_mask for session in sessions])
    total_boundaries = np.concatenate([session.boundaries for session in sessions])
    summary = {
        "dataset": dataset_name,
        "num_subjects": len(sessions),
        "num_steps": int(sum(len(session.labels) for session in sessions)),
        "step_seconds": step_seconds,
        "feature_dimensions": feature_dimensions,
        "behavior_features": behavior_config.provenance(),
        "source_inclusion_policy": {
            "include_facial_csv": include_facial_csv,
            "complete_behavior_sources_only": complete_behavior_sources_only,
            "behavior_source_exclusions": behavior_source_exclusions,
            "excluded_facial_csv_records": staged_manifest.get("source_policy", {}).get(
                "excluded_facial_csv_records", 0
            ),
        },
        "alignment_mode": alignment_mode,
        "label_counts": {
            "0": int((total_labels == 0).sum()),
            "1": int((total_labels == 1).sum()),
        },
        "target_label_counts": {
            "0": int(((total_labels == 0) & total_target_mask).sum()),
            "1": int(((total_labels == 1) & total_target_mask).sum()),
            "ignored": int((~total_target_mask).sum()),
        },
        "deception_events": int(total_boundaries[:, 0].sum()),
        "fully_missing_steps": int(
            sum(
                (~session.physiology_mask.any(axis=1) & ~session.modality_mask.any(axis=1)).sum()
                for session in sessions
            )
        ),
        "excluded_raw_sessions": int(staged_manifest.get("excluded_session_count", 0)),
        "split_seed": split_seed,
        "split_strategy": split_strategy,
        "split_sessions": splits,
        "subject_disjoint": True,
        "session_compliance": {
            "selected_profile": required_profile,
            "all_sessions_compliant": compliance["all_sessions_compliant"],
            "compliant_sessions": compliance["compliant_sessions"],
            "total_sessions": compliance["total_sessions"],
            "profile_coverage": {
                profile: values["coverage"] for profile, values in compliance["profiles"].items()
            },
        },
        "windows_per_split": window_counts,
        "storage_mode": (
            "session_timelines_plus_compatibility_windows"
            if write_compatibility_windows
            else "session_timelines"
        ),
        "sessions_per_split": session_counts,
        "all_manifest_windows": len(all_records),
        "all_manifest_sessions": len(all_session_records),
        "window_size": window_size,
        "stride": stride,
        "sessions": session_summaries,
        "limitations": [
            "the fixed split uses labels and modality availability for pre-training stratification",
            "a single fixed split is a pilot protocol, not cross-validation",
            "target identity uses a retrospective label-independent transcript-length "
            "heuristic, not supplied role metadata or online role discovery",
            "speaker-turn timestamps require within-turn character interpolation",
            (
                "confirmed same-session EEG may use a documented session-end constant "
                "clock offset; within-stream timing is preserved, but the result is not "
                "shared hardware synchronization and cannot support sub-second lag claims"
                if alignment_mode == "session_registered_time"
                else
                "sensor rows are retained only on absolute-time overlap; clocks were not "
                "hardware-synchronized, so sub-second lag claims remain unsupported"
                if alignment_mode == "absolute_time"
                else "sensor streams are duration-normalized because device clocks are inconsistent"
            ),
            (
                "audio is unavailable except where a raw recording exists"
                if behavior_config.audio_backend == "wavlm"
                else "raw audio is not decoded because the selected backend is none"
            ),
            (
                "MARLIN-small embeddings are extracted from each paired raw video"
                if behavior_config.video_backend == "marlin"
                else "OpenFace frame features are extracted from each paired raw video and "
                "cached by source SHA-256"
                if behavior_config.video_backend == "openface"
                else (
                    "facial CSV is explicitly excluded from this dataset version"
                    if not include_facial_csv
                    else "facial input uses the legacy three-variable action CSV"
                )
            ),
            f"raw EEG is present for {sum(source.eeg_file is not None for source in discovered)} "
            f"of {len(discovered)} subjects",
            f"paired PPG is present for {sum(source.ppg_files is not None for source in discovered)} "
            f"of {len(discovered)} subjects",
        ],
    }
    session_index = {
        "dataset": dataset_name,
        "sessions": [
            {
                "session_id": session.session_id,
                "subject_id": session.subject_id,
                "split": session_to_split[session.session_id],
                "raw_path": f"raw/{source_root.name}/{session.session_id}",
                "processed_path": f"processed/{dataset_name}/sessions/{session.session_id}",
            }
            for session in sessions
        ],
    }
    _write_json(output_root / "session_index.json", session_index)
    if not reuse_staged_raw:
        _write_json(raw_root / "session_index.json", session_index)
    _write_json(output_root / "dataset_summary.json", summary)
    (output_root / "README.md").write_text(_dataset_readme(summary), encoding="utf-8", newline="\n")
    package_index = [
        {
            "session_id": record["sample_id"],
            "tensor_sha256": record["sha256"],
            "num_steps": record["metadata"]["num_steps"],
            "validated_before_write": True,
        }
        for record in all_session_records
    ]
    integrity_manifest = {
        "schema_version": 1,
        "dataset": dataset_name,
        "status": "complete",
        "session_count": len(package_index),
        "split_assignment_sha256": split_strategy["assignment_sha256"],
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(staged_manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "alignment_report_sha256": package_sha256_file(output_root / "alignment_report.json"),
        "sessions": package_index,
    }
    _write_json(output_root / "dataset_integrity.json", integrity_manifest)
    _write_json(
        output_root / "_SUCCESS.json",
        {
            "dataset": dataset_name,
            "dataset_integrity_sha256": package_sha256_file(output_root / "dataset_integrity.json"),
        },
    )
    if not reuse_staged_raw:
        _write_data_catalog(raw_root.parent.parent, len(sessions), dataset_name)
    _safe_remove_legacy_windows(output_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the per-subject BCI directory to model-ready EPT windows"
    )
    parser.add_argument("--source", required=True, help="Path to the BCI directory")
    parser.add_argument(
        "--output",
        default=f"data/processed/{DATASET_NAME}",
        help="New EPT dataset folder",
    )
    parser.add_argument(
        "--raw-output",
        default=f"data/raw/{DATASET_NAME}",
        help="Session-normalized raw dataset folder",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min-window-size", type=int, default=16)
    parser.add_argument("--step-seconds", type=float, default=STEP_SECONDS)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument(
        "--required-profile",
        choices=tuple(COMPLIANCE_PROFILES),
        default="model_contract",
        help=(
            "Fail before split generation unless every session satisfies this profile. "
            "Use strong_behavior_sources before formal audio-video-text extraction."
        ),
    )
    parser.add_argument(
        "--video-backend", choices=("legacy", "marlin", "openface"), default="marlin"
    )
    parser.add_argument("--audio-backend", choices=("none", "wavlm"), default="none")
    parser.add_argument("--text-backend", choices=("hash", "macbert"), default="hash")
    parser.add_argument("--text-model-id", default=MACBERT_MODEL_ID)
    parser.add_argument("--text-revision", default=MACBERT_REVISION)
    parser.add_argument("--audio-model-id", default=WAVLM_MODEL_ID)
    parser.add_argument("--audio-revision", default=WAVLM_REVISION)
    parser.add_argument("--behavior-device", default="auto")
    parser.add_argument("--behavior-batch-size", type=int, default=16)
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--marlin-checkpoint", default=None)
    parser.add_argument(
        "--marlin-cache-dir",
        default=None,
        help="Persistent MARLIN embedding cache; completed videos survive interruption",
    )
    parser.add_argument(
        "--marlin-crop-face",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--openface-cache-dir",
        default=None,
        help="Persistent OpenFace CSV cache; completed videos are reused after interruption",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--openface-min-confidence", type=float, default=0.8)
    parser.add_argument(
        "--reuse-staged-raw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat --source as an existing session_NNN raw tree, verify its file "
            "ledger, preserve session IDs, and never rewrite the raw directory"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Explicitly allow replacement of an existing processed output",
    )
    parser.add_argument(
        "--write-compatibility-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also materialize overlapping window .pt files. Disable when downstream "
            "models consume complete session timelines."
        ),
    )
    parser.add_argument(
        "--include-facial-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include staged legacy/OpenFace facial CSV files in provenance, split "
            "statistics, and video features. Disable to build a no-facial dataset."
        ),
    )
    parser.add_argument(
        "--complete-behavior-sources-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Before splitting, retain only sessions with paired video, audio, and "
            "annotated transcript sources"
        ),
    )
    parser.add_argument(
        "--exclude-session",
        action="append",
        default=[],
        help="Explicitly exclude a session before splitting; may be repeated",
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("duration_normalized", "absolute_time", "session_registered_time"),
        default="duration_normalized",
        help=(
            "Use legacy duration normalization, direct absolute intersection, or "
            "confirmed-session constant-offset clock registration"
        ),
    )
    parser.add_argument(
        "--whisper-alignment-dir",
        default=None,
        help="Directory containing verified per-session Whisper alignment JSON files",
    )
    parser.add_argument(
        "--require-whisper-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail closed if a Whisper alignment artifact is absent or stale",
    )
    args = parser.parse_args()
    behavior_config = BehaviorFeatureConfig(
        video_backend=args.video_backend,
        audio_backend=args.audio_backend,
        text_backend=args.text_backend,
        text_model_id=args.text_model_id,
        text_revision=args.text_revision,
        audio_model_id=args.audio_model_id,
        audio_revision=args.audio_revision,
        device=args.behavior_device,
        batch_size=args.behavior_batch_size,
        cache_dir=args.model_cache,
        marlin_checkpoint=args.marlin_checkpoint,
        marlin_crop_face=args.marlin_crop_face,
        marlin_cache_dir=args.marlin_cache_dir,
        openface_cache_dir=args.openface_cache_dir,
        local_files_only=args.local_files_only,
        openface_min_confidence=args.openface_min_confidence,
    )
    summary = prepare_dataset(
        Path(args.source),
        Path(args.output),
        raw_root=Path(args.raw_output),
        window_size=args.window_size,
        stride=args.stride,
        min_window_size=args.min_window_size,
        step_seconds=args.step_seconds,
        split_seed=args.split_seed,
        dataset_name=args.dataset_name,
        behavior_config=behavior_config,
        required_profile=args.required_profile,
        reuse_staged_raw=args.reuse_staged_raw,
        overwrite=args.overwrite,
        write_compatibility_windows=args.write_compatibility_windows,
        include_facial_csv=args.include_facial_csv,
        alignment_mode=args.alignment_mode,
        complete_behavior_sources_only=args.complete_behavior_sources_only,
        whisper_alignment_dir=(
            Path(args.whisper_alignment_dir) if args.whisper_alignment_dir else None
        ),
        require_whisper_alignment=args.require_whisper_alignment,
        excluded_session_ids=args.exclude_session,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
