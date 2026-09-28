"""Data validation entry point for the 927 video augmentation."""

from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from pathlib import Path

import av


def media_summary(path: Path) -> tuple[float, bool, bool]:
    with av.open(str(path), mode="r") as container:
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        return duration, bool(container.streams.video), bool(container.streams.audio)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve(strict=True)
    rows = json.loads((root / "video_pairing_audit.json").read_text(encoding="utf-8"))
    assert len(rows) == 25, len(rows)
    counts = Counter(row["status"] for row in rows)
    assert counts == {
        "assigned_existing_session": 12,
        "created_new_session": 12,
        "unresolved_non_session_clip": 1,
    }, counts

    for row in rows:
        video = Path(row["staged_video"])
        assert video.is_file() and video.stat().st_size > 0, video
        duration, has_video, has_audio = media_summary(video)
        assert has_video and has_audio, (video, has_video, has_audio)
        expected = float(row["source_duration_seconds"]) - float(row["clip_start_seconds"])
        assert abs(duration - expected) < 12.0, (video, duration, expected)
        if row.get("staged_audio"):
            audio = Path(row["staged_audio"])
            assert audio.is_file() and audio.stat().st_size > 0, audio
            with wave.open(str(audio), "rb") as handle:
                assert handle.getnchannels() == 1, audio
                assert handle.getframerate() == 16000, audio

    index = json.loads((root / "session_index.json").read_text(encoding="utf-8"))
    assert len(index["sessions"]) == 41, len(index["sessions"])
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["sessions"]) == 41, len(manifest["sessions"])
    assert not list(root.rglob("*.partial.mp4"))
    assert not list(root.rglob("*.partial.wav"))
    print(json.dumps({"status": "ok", "audit_rows": len(rows), "counts": counts}, default=dict))


if __name__ == "__main__":
    main()
