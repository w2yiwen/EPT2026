from __future__ import annotations

import numpy as np
import pytest

from eptnet.data.whisper_alignment import (
    build_reference_timeline,
    estimate_video_audio_offset,
)


def _turns():
    plain = [
        {"speaker_id": "1", "text": "你好", "start": 0.0},
        {"speaker_id": "2", "text": "是的", "start": 1.0},
        {"speaker_id": "1", "text": "再见", "start": 2.0},
    ]
    marked = [
        {**plain[0], "marks": [False, True]},
        {**plain[1], "marks": [False, False]},
        {**plain[2], "marks": [False, False]},
    ]
    return plain, marked


def test_reference_text_is_placed_on_whisper_time() -> None:
    plain, marked = _turns()
    result = {
        "segments": [
            {
                "text": "你好是的再见",
                "start": 1.0,
                "end": 5.0,
                "words": [
                    {"word": "你好", "start": 1.0, "end": 2.0, "probability": 0.9},
                    {"word": "是的", "start": 2.0, "end": 3.0, "probability": 0.9},
                    {"word": "再见", "start": 4.0, "end": 5.0, "probability": 0.9},
                ],
            }
        ]
    }

    aligned = build_reference_timeline(
        plain, marked, result, audio_duration_seconds=6.0, minimum_reference_coverage=1.0
    )

    assert aligned["step_text"][:6] == ["", "你好", "是的", "", "再见", ""]
    assert aligned["target_mask"][:6] == [False, True, False, False, True, False]
    assert len(aligned["step_text"]) == 16
    assert aligned["labels"][1] == 0


def test_reference_alignment_rejects_low_coverage() -> None:
    plain, marked = _turns()
    result = {
        "segments": [
            {"text": "错", "start": 0.0, "end": 1.0, "words": []},
        ]
    }
    with pytest.raises(ValueError, match="coverage"):
        build_reference_timeline(
            plain, marked, result, audio_duration_seconds=6.0, minimum_reference_coverage=0.5
        )


def test_video_audio_offset_sign() -> None:
    sample_rate = 1_000
    rng = np.random.default_rng(7)
    external = np.zeros(8 * sample_rate, dtype=np.float32)
    external[1_000:5_000] = rng.normal(size=4_000)
    video = np.concatenate([np.zeros(500, dtype=np.float32), external])

    offset, score = estimate_video_audio_offset(
        external, video, sample_rate=sample_rate, maximum_offset_seconds=2.0
    )

    assert offset == pytest.approx(0.5, abs=0.03)
    assert score > 0.95
