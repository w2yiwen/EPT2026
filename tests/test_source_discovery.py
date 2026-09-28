from __future__ import annotations

from pathlib import Path

import pytest

from eptnet.data.prepare_bci import _resolve_source_file


def test_source_discovery_uses_anonymous_unique_patterns(tmp_path: Path) -> None:
    acquisition = tmp_path / "01_raw_acquisition"
    acquisition.mkdir()
    expected = acquisition / "participant_video.mp4"
    expected.write_bytes(b"video fixture")

    resolved = _resolve_source_file(tmp_path, "01_raw_acquisition/*.mp4")

    assert resolved == expected


def test_source_discovery_rejects_ambiguous_matches(tmp_path: Path) -> None:
    acquisition = tmp_path / "01_raw_acquisition"
    acquisition.mkdir()
    (acquisition / "session_a.mp4").write_bytes(b"a")
    (acquisition / "session_b.mp4").write_bytes(b"b")

    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_source_file(tmp_path, "01_raw_acquisition/*.mp4")


def test_source_discovery_reports_missing_required_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="has no file"):
        _resolve_source_file(tmp_path, "02_alignment_and_labels/*_marked.docx")
