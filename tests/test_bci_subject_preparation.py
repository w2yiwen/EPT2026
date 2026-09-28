from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import eptnet.data.prepare_bci_subjects as preparation
from eptnet.data.prepare_bci_subjects import (
    TEXT_DIM,
    PreparedSession,
    SourceSession,
    assign_constrained_subject_splits,
    assign_subject_splits,
    build_session_compliance,
    hashed_text_features,
    ppg_features,
    prepare_session,
    transcript_timeline,
    write_session_windows,
)


def _speaker_paragraph(
    speaker: int, timestamp: str, content: str, *, marked: bool = False
) -> list[tuple[str, bool, bool]]:
    return [
        (f"说话人{speaker} {timestamp} ", False, False),
        (content, marked, False),
    ]


def _patch_document(
    monkeypatch: pytest.MonkeyPatch,
    paragraphs: list[list[tuple[str, bool, bool]]],
) -> None:
    monkeypatch.setattr(preparation, "_document_text_paragraphs", lambda _path: paragraphs)
    monkeypatch.setattr(preparation, "_document_paragraphs", lambda _path: paragraphs)


def test_subject_splits_are_deterministic_and_disjoint() -> None:
    subjects = [f"subject_{index:02d}" for index in range(15)]
    first = assign_subject_splits(subjects, seed=42)
    second = assign_subject_splits(list(reversed(subjects)), seed=42)
    assert first == second
    assert set(first["train"]).isdisjoint(first["val"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["val"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(subjects)


def test_constrained_subject_splits_are_deterministic_and_meet_modality_quotas() -> None:
    sessions = []
    for index in range(6):
        physiology_mask = np.zeros((4, 3), dtype=bool)
        modality_mask = np.zeros((4, 3), dtype=bool)
        if index < 5:
            physiology_mask[:, :2] = True
        if index < 5:
            physiology_mask[:, 2] = True
        modality_mask[:, 0] = True
        boundaries = np.zeros((4, 2), dtype=np.float32)
        boundaries[0, 0] = 1.0
        boundaries[0, 1] = 1.0
        sessions.append(
            SimpleNamespace(
                session_id=f"session_{index:03d}",
                subject_id=f"subject_{index:03d}",
                target_mask=np.ones(4, dtype=bool),
                labels=np.array([0, 1, 1, 1], dtype=np.int64),
                boundaries=boundaries,
                physiology_mask=physiology_mask,
                modality_mask=modality_mask,
            )
        )

    first, strategy = assign_constrained_subject_splits(sessions, seed=42)
    second, second_strategy = assign_constrained_subject_splits(list(reversed(sessions)), seed=42)

    assert first == second
    assert strategy["assignment_sha256"] == second_strategy["assignment_sha256"]
    assert strategy["split_sizes"] == {"train": 4, "val": 1, "test": 1}
    assert strategy["modality_quotas"] == {
        "eeg": {"train": 3, "val": 1, "test": 1},
        "physiology": {"train": 3, "val": 1, "test": 1},
        "video": {"train": 4, "val": 1, "test": 1},
    }
    assert strategy["feasible_assignments"] > 0
    assert set().union(*map(set, first.values())) == {session.session_id for session in sessions}


def test_text_hashes_are_finite_deterministic_and_fixed_width() -> None:
    texts = ["测试文本", "", "欺骗片段"]
    first = hashed_text_features(texts)
    second = hashed_text_features(texts)
    assert first.shape == (3, TEXT_DIM)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.count_nonzero(first[1]) == 0


def test_ppg_adapter_produces_ept_width_and_availability(tmp_path: Path) -> None:
    x = np.sin(np.linspace(0, 40 * np.pi, 1200)) + 2.0
    y = np.cos(np.linspace(0, 30 * np.pi, 1200)) + 3.0
    first = tmp_path / "ch2.txt"
    second = tmp_path / "ch3.txt"
    first.write_text("\n".join(map(str, x)), encoding="utf-8")
    second.write_text("\n".join(map(str, y)), encoding="utf-8")
    features, mask = ppg_features((first, second), num_steps=20, duration_seconds=20.0)
    assert features.shape == (20, 40)
    assert mask.shape == (20,)
    assert mask.all()
    assert np.isfinite(features).all()


def test_target_mask_removes_annotation_bleed_and_splits_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paragraphs = [
        [("timestamp前的空白文本", True, False)],
        _speaker_paragraph(1, "00:02", "问问"),
        _speaker_paragraph(2, "00:04", "目" * 20, marked=True),
        _speaker_paragraph(1, "00:06", "误", marked=True),
        _speaker_paragraph(2, "00:08", "标" * 20, marked=True),
        _speaker_paragraph(1, "00:10", "结束"),
    ]
    _patch_document(monkeypatch, paragraphs)

    _, labels, target_mask, stats = transcript_timeline(
        tmp_path / "annotations.docx",
        "yellow_highlight",
        include_target_mask=True,
    )
    targets = preparation.build_event_targets(labels, valid_mask=target_mask)

    assert stats["target_speaker_id"] == "2"
    assert stats["target_speaker_inference"] == (
        "maximum_total_non_whitespace_transcript_characters"
    )
    assert stats["target_speaker_inference_uses_annotation_marks"] is False
    assert stats["target_speaker_transcript_characters"] == 40
    assert stats["runner_up_speaker_transcript_characters"] == 5
    assert stats["target_speaker_character_share"] == pytest.approx(40 / 45)
    assert stats["target_speaker_margin_characters"] == 35
    assert stats["target_marked_coverage"] == pytest.approx(40 / 41)
    assert stats["discarded_non_target_marked_characters"] == 1
    assert not target_mask[:4].any()
    assert target_mask[4:6].all()
    assert not target_mask[6:8].any()
    assert target_mask[8:10].all()
    assert not target_mask[10:].any()
    assert labels[7] == 1  # Non-target marks are audit-only, never label evidence.
    assert not targets["positive_mask"][7]
    assert np.flatnonzero(targets["boundaries"][:, 0]).tolist() == [4, 8]
    assert np.flatnonzero(targets["boundaries"][:, 1]).tolist() == [5, 9]


def test_target_mask_requires_strict_character_majority_in_mixed_bins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paragraphs = [
        _speaker_paragraph(2, "00:00", "AAA", marked=True),
        _speaker_paragraph(1, "00:01", "B"),
        _speaker_paragraph(2, "00:02", "C", marked=True),
        _speaker_paragraph(1, "00:03", "D"),
        _speaker_paragraph(1, "00:04", "E"),
    ]
    _patch_document(monkeypatch, paragraphs)

    _, _, target_mask, stats = transcript_timeline(
        tmp_path / "annotations.docx",
        "yellow_highlight",
        step_seconds=2.0,
        include_target_mask=True,
    )

    assert target_mask[0]
    assert not target_mask[1]
    assert stats["mixed_speaker_steps"] == 2
    assert stats["tied_speaker_steps"] == 1


def test_non_target_mark_cannot_label_a_mixed_target_majority_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paragraphs = [
        _speaker_paragraph(2, "00:00", "AA"),
        _speaker_paragraph(1, "00:01", "X", marked=True),
        _speaker_paragraph(2, "00:02", "B" * 20, marked=True),
        _speaker_paragraph(1, "00:04", "Y"),
    ]
    _patch_document(monkeypatch, paragraphs)

    _, labels, target_mask, stats = transcript_timeline(
        tmp_path / "annotations.docx",
        "yellow_highlight",
        step_seconds=2.0,
        include_target_mask=True,
    )
    targets = preparation.build_event_targets(labels, valid_mask=target_mask)

    assert target_mask[0]  # Two target chars versus one marked non-target char.
    assert labels[0] == 1
    assert not targets["positive_mask"][0]
    assert stats["target_marked_coverage"] == pytest.approx(20 / 21)
    assert stats["discarded_non_target_marked_characters"] == 1


@pytest.mark.parametrize(
    "paragraphs, expected_message",
    [
        (
            [
                _speaker_paragraph(1, "00:00", "AA"),
                _speaker_paragraph(2, "00:02", "B"),
            ],
            "no marked transcript characters",
        ),
        (
            [
                _speaker_paragraph(1, "00:00", "A" * 20, marked=True),
                _speaker_paragraph(2, "00:02", "B" * 10, marked=True),
            ],
            "target-label consistency",
        ),
        (
            [
                _speaker_paragraph(1, "00:00", "A" * 10, marked=True),
                _speaker_paragraph(2, "00:02", "B" * 10),
            ],
            "ambiguous target-speaker transcript length",
        ),
    ],
)
def test_target_speaker_inference_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    paragraphs: list[list[tuple[str, bool, bool]]],
    expected_message: str,
) -> None:
    _patch_document(monkeypatch, paragraphs)
    with pytest.raises(ValueError, match=expected_message):
        transcript_timeline(
            tmp_path / "annotations.docx",
            "yellow_highlight",
            include_target_mask=True,
        )


def test_target_speaker_and_mask_do_not_follow_highlight_location(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def paragraphs(mark_first_target: bool) -> list[list[tuple[str, bool, bool]]]:
        return [
            _speaker_paragraph(1, "00:00", "aa"),
            _speaker_paragraph(2, "00:02", "B" * 12, marked=mark_first_target),
            _speaker_paragraph(1, "00:04", "cc"),
            _speaker_paragraph(2, "00:06", "D" * 12, marked=not mark_first_target),
            _speaker_paragraph(1, "00:08", "ee"),
        ]

    _patch_document(monkeypatch, paragraphs(True))
    _, first_labels, first_mask, first_stats = transcript_timeline(
        tmp_path / "annotations.docx",
        "yellow_highlight",
        include_target_mask=True,
    )
    _patch_document(monkeypatch, paragraphs(False))
    _, second_labels, second_mask, second_stats = transcript_timeline(
        tmp_path / "annotations.docx",
        "yellow_highlight",
        include_target_mask=True,
    )

    assert first_stats["target_speaker_id"] == second_stats["target_speaker_id"] == "2"
    assert np.array_equal(first_mask, second_mask)
    assert not np.array_equal(first_labels, second_labels)
    assert (
        first_stats["target_speaker_character_share"]
        == second_stats["target_speaker_character_share"]
    )

    swapped_to_non_target = [
        _speaker_paragraph(1, "00:00", "aa", marked=True),
        _speaker_paragraph(2, "00:02", "B" * 12),
        _speaker_paragraph(1, "00:04", "cc", marked=True),
        _speaker_paragraph(2, "00:06", "D" * 12),
        _speaker_paragraph(1, "00:08", "ee", marked=True),
    ]
    _patch_document(monkeypatch, swapped_to_non_target)
    with pytest.raises(ValueError, match="target-label consistency"):
        transcript_timeline(
            tmp_path / "annotations.docx",
            "yellow_highlight",
            include_target_mask=True,
        )


def test_prepared_session_and_window_persist_target_mask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paragraphs = [
        _speaker_paragraph(1, "00:02", "问问"),
        _speaker_paragraph(2, "00:04", "目" * 20, marked=True),
        _speaker_paragraph(1, "00:06", "回答"),
    ]
    _patch_document(monkeypatch, paragraphs)
    source = SourceSession(
        session_id="session_001",
        subject_id="private_subject",
        root=tmp_path,
        annotation=tmp_path / "annotations.docx",
        annotation_mode="yellow_highlight",
        face_csv=None,
        openface_csv=None,
        ppg_files=None,
        eeg_file=None,
        audio_file=None,
        video_file=None,
    )

    session = prepare_session(source)
    records = write_session_windows(
        session,
        "train",
        tmp_path / "processed",
        window_size=64,
        stride=32,
        min_window_size=16,
    )
    sample_path = tmp_path / "processed" / "sessions" / "session_001" / "windows"
    sample = torch.load(sample_path / Path(records[0]["tensor_file"]).name, weights_only=True)

    assert session.target_mask.dtype == np.bool_
    assert torch.equal(sample["target_mask"], torch.from_numpy(session.target_mask))
    assert not sample["positive_mask"][~sample["target_mask"]].any()


def test_absolute_clock_prefers_clipped_media_over_ppg_start(tmp_path: Path) -> None:
    source = SourceSession(
        session_id="session_001",
        subject_id="session_001",
        root=tmp_path,
        annotation=tmp_path / "session_001_annotations.docx",
        annotation_mode="yellow_highlight",
        face_csv=None,
        openface_csv=None,
        ppg_files=(
            tmp_path / "session_001_ppg_channel_1.txt",
            tmp_path / "session_001_ppg_channel_2.txt",
        ),
        eeg_file=None,
        audio_file=tmp_path / "session_001_audio.wav",
        video_file=tmp_path / "session_001_video.mp4",
        metadata={
            "files": [
                {
                    "source_relative_path": "subject/s-03-06-19-44-15_ch2.txt",
                    "staged_relative_path": "session_001/session_001_ppg_channel_1.txt",
                },
                {
                    "source_relative_path": "VID20250306194315.mp4",
                    "staged_relative_path": "session_001/session_001_audio.wav",
                    "source_recorded_at": "2025-03-06T19:43:15",
                    "clip_start_seconds": 76.92,
                },
            ]
        },
    )

    clock = preparation.resolve_session_clock(source)

    assert clock.timeline_start_local == "2025-03-06T19:44:31.920+08:00"
    assert clock.timeline_origin_evidence == "media_start_timestamp_plus_clip_offset"
    assert clock.ppg_start_local == "2025-03-06T19:44:15.000+08:00"


def test_vid_underscore_timestamp_is_recording_end(tmp_path: Path) -> None:
    source = SourceSession(
        session_id="session_004",
        subject_id="session_004",
        root=tmp_path,
        annotation=tmp_path / "session_004_annotations.docx",
        annotation_mode="yellow_highlight",
        face_csv=None,
        openface_csv=None,
        ppg_files=None,
        eeg_file=None,
        audio_file=tmp_path / "session_004_audio.wav",
        video_file=tmp_path / "session_004_video.mp4",
        metadata={
            "files": [
                {
                    "source_relative_path": "927/VID_20250305_211356.mp4",
                    "staged_relative_path": "session_004/session_004_audio.wav",
                    "source_recorded_at": "2025-03-05T21:13:56",
                    "source_duration_seconds": 763.541,
                    "clip_start_seconds": 328.46,
                }
            ]
        },
    )

    clock = preparation.resolve_session_clock(source, timeline_duration_seconds=445.0)

    assert clock.timeline_start_local == "2025-03-05T21:06:40.919+08:00"
    assert clock.session_end_local == "2025-03-05T21:13:56.000+08:00"
    assert clock.media_timestamp_semantics == "recording_end"
    assert (
        clock.timeline_origin_evidence
        == "media_end_timestamp_minus_duration_plus_clip_offset"
    )


def test_absolute_ppg_alignment_masks_non_overlapping_steps(tmp_path: Path) -> None:
    first = tmp_path / "ch2.txt"
    second = tmp_path / "ch3.txt"
    values = "\n".join(str(index % 17) for index in range(770))
    first.write_text(values, encoding="utf-8")
    second.write_text(values, encoding="utf-8")
    timeline_start = 1_741_260_000.0

    features, mask, alignment = preparation._ppg_features_with_alignment(
        (first, second),
        4,
        4.0,
        alignment_mode="absolute_time",
        timeline_start_epoch_s=timeline_start,
        source_start_epoch_s=timeline_start + 1.0,
    )

    assert mask.tolist() == [False, True, True, False]
    assert np.all(features[~mask] == 0)
    assert alignment["status"] == "aligned_absolute_overlap"
    assert alignment["sample_rate_hz"] == 385.0


def test_absolute_eeg_alignment_uses_embedded_epoch_and_masks_gaps(tmp_path: Path) -> None:
    eeg = tmp_path / "eeg.txt"
    timeline_start = 1_741_260_000.0
    lines = [
        "%OpenBCI Raw EEG Data",
        "%Sample Rate = 500.0 Hz",
    ]
    sample_index = 0
    for step in (1, 2):
        for offset in range(40):
            epoch_ms = int((timeline_start + step + offset / 50.0) * 1000)
            channels = ", ".join(str(sample_index + channel) for channel in range(8))
            lines.append(f"{sample_index}, {channels}, 0, 0, 0, 00:00:00, {epoch_ms}")
            sample_index += 1
    eeg.write_text("\n".join(lines), encoding="utf-8")

    time_features, spectral_features, mask, alignment = (
        preparation._eeg_features_with_alignment(
            eeg,
            4,
            4.0,
            alignment_mode="absolute_time",
            timeline_start_epoch_s=timeline_start,
        )
    )

    assert mask.tolist() == [False, True, True, False]
    assert np.all(time_features[~mask] == 0)
    assert np.all(spectral_features[~mask] == 0)
    assert alignment["method"] == "embedded_epoch_intersection"


def test_confirmed_session_eeg_uses_constant_end_offset_without_stretching(
    tmp_path: Path,
) -> None:
    eeg = tmp_path / "eeg.txt"
    raw_start = 1_700_000_100.0
    timeline_start = 1_741_260_000.0
    lines = ["%OpenBCI Raw EEG Data", "%Sample Rate = 500.0 Hz"]
    sample_index = 0
    for step in range(4):
        for offset in range(40):
            epoch_ms = int((raw_start + step + offset / 50.0) * 1000)
            channels = ", ".join(str(sample_index + channel) for channel in range(8))
            lines.append(f"{sample_index}, {channels}, 0, 0, 0, 00:00:00, {epoch_ms}")
            sample_index += 1
    eeg.write_text("\n".join(lines), encoding="utf-8")

    _, _, mask, alignment = preparation._eeg_features_with_alignment(
        eeg,
        4,
        4.0,
        alignment_mode="session_registered_time",
        timeline_start_epoch_s=timeline_start,
        session_end_epoch_s=timeline_start + 4.0,
        session_end_evidence="media_recording_end_from_start_plus_duration",
        session_end_confidence="high",
    )

    assert mask.tolist() == [True, True, True, True]
    assert alignment["status"] == "aligned_end_anchored_clock_offset"
    assert alignment["method"] == "embedded_epoch_plus_constant_session_end_offset"
    assert alignment["clock_offset_applied"] is True
    assert alignment["clock_offset_anchor_confidence"] == "high"
    assert alignment["direct_temporal_overlap_steps"] == 0
    assert alignment["raw_source_start_local"] != alignment["source_start_local"]


def _compliance_session(tmp_path: Path, *, with_behavior_media: bool) -> PreparedSession:
    tmp_path.mkdir(parents=True, exist_ok=True)
    annotation = tmp_path / "annotations.docx"
    annotation.write_bytes(b"test")
    audio = tmp_path / "audio.wav" if with_behavior_media else None
    video = tmp_path / "video.mp4" if with_behavior_media else None
    if audio is not None:
        audio.write_bytes(b"audio")
    if video is not None:
        video.write_bytes(b"video")
    source = SourceSession(
        session_id="session_001",
        subject_id="private_subject",
        root=tmp_path,
        annotation=annotation,
        annotation_mode="yellow_highlight",
        face_csv=None,
        openface_csv=None,
        ppg_files=None,
        eeg_file=None,
        audio_file=audio,
        video_file=video,
    )
    num_steps = 4
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    boundaries = np.zeros((num_steps, 2), dtype=np.float32)
    boundaries[0, 0] = 1.0
    boundaries[1, 1] = 1.0
    return PreparedSession(
        dataset_name="test_dataset",
        session_id=source.session_id,
        subject_id=source.subject_id,
        source=source,
        duration_seconds=4.0,
        step_seconds=1.0,
        features={
            "eeg_time": np.zeros((num_steps, 8), dtype=np.float32),
            "eeg_spectral": np.zeros((num_steps, 40), dtype=np.float32),
            "physiology": np.zeros((num_steps, 40), dtype=np.float32),
            "video": np.zeros((num_steps, 3), dtype=np.float32),
            "audio": np.zeros((num_steps, 50), dtype=np.float32),
            "text": np.ones((num_steps, TEXT_DIM), dtype=np.float32),
        },
        physiology_mask=np.zeros((num_steps, 3), dtype=bool),
        modality_mask=np.asarray([[False, False, True]] * num_steps, dtype=bool),
        target_mask=np.ones(num_steps, dtype=bool),
        labels=labels,
        boundaries=boundaries,
        offsets=np.zeros((num_steps, 2), dtype=np.float32),
        positive_mask=labels == 0,
        timestamps=np.stack((np.arange(num_steps), np.arange(1, num_steps + 1)), axis=-1),
        step_text=["text"] * num_steps,
        annotation_statistics={},
        behavior_provenance={
            "video_backend": "legacy",
            "audio_backend": "none",
            "text_backend": "hash",
        },
        alignment={
            "reference": "annotated_transcript",
            "eeg": {"present": False},
            "physiology": {"present": False},
        },
    )


def test_session_compliance_separates_model_contract_from_raw_media(tmp_path: Path) -> None:
    masked_session = _compliance_session(tmp_path / "masked", with_behavior_media=False)
    source_ready_session = _compliance_session(tmp_path / "ready", with_behavior_media=True)

    report = build_session_compliance(
        [masked_session, source_ready_session], required_profile="strong_behavior_sources"
    )

    assert report["profiles"]["model_contract"]["compliant_sessions"] == 2
    assert report["profiles"]["strong_behavior_sources"]["compliant_sessions"] == 1
    assert report["all_sessions_compliant"] is False
    assert report["selected_profile_failed_check_counts"] == {
        "raw_video": 1,
        "raw_audio": 1,
    }


def test_session_compliance_rejects_unmasked_placeholder_values(tmp_path: Path) -> None:
    session = _compliance_session(tmp_path, with_behavior_media=False)
    session.features["audio"][0, 0] = 1.0

    report = build_session_compliance([session], required_profile="model_contract")

    assert report["all_sessions_compliant"] is False
    assert report["sessions"]["session_001"]["checks"]["explicit_missingness"] is False
