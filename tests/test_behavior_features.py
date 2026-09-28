from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import eptnet.data.behavior_features as behavior_features
from eptnet.data.behavior_features import (
    MARLIN_OUTPUT_DIM,
    OPENFACE_FEATURE_NAMES,
    OPENFACE_FRAME_COLUMNS,
    BehaviorFeatureConfig,
    BehaviorFeaturePipeline,
    extract_openface_features,
)
from eptnet.models.behavior.common import EncodedSequence


def test_behavior_feature_dimensions_are_explicit() -> None:
    legacy = BehaviorFeatureConfig()
    official = BehaviorFeatureConfig(
        video_backend="openface", audio_backend="wavlm", text_backend="macbert"
    )

    assert (legacy.video_dim, legacy.audio_dim, legacy.text_dim) == (3, 50, 768)
    assert (official.video_dim, official.audio_dim, official.text_dim) == (
        len(OPENFACE_FEATURE_NAMES),
        768,
        768,
    )
    assert official.provenance()["text_revision"]
    assert official.provenance()["audio_revision"]

    marlin = BehaviorFeatureConfig(video_backend="marlin")
    assert marlin.video_dim == MARLIN_OUTPUT_DIM
    assert len(marlin.provenance()["video_feature_order"]) == MARLIN_OUTPUT_DIM


def test_openface_adapter_filters_failures_and_pools_official_statistics(tmp_path: Path) -> None:
    frame_columns = list(OPENFACE_FRAME_COLUMNS)
    csv_path = tmp_path / "openface.csv"
    header = ["timestamp", "confidence", "success", *frame_columns]
    rows = [
        [0.0, 0.99, 1, *([1.0] * len(frame_columns))],
        [0.5, 0.99, 1, *([3.0] * len(frame_columns))],
        [1.0, 0.20, 1, *([100.0] * len(frame_columns))],
        [1.5, 0.99, 0, *([100.0] * len(frame_columns))],
        [1.9, 0.99, 1, *([5.0] * len(frame_columns))],
    ]
    csv_path.write_text(
        "\n".join(",".join(map(str, row)) for row in [header, *rows]),
        encoding="utf-8",
    )

    features, mask = extract_openface_features(csv_path, num_steps=2)

    assert features.shape == (2, len(OPENFACE_FEATURE_NAMES))
    assert mask.tolist() == [True, True]
    width = len(frame_columns)
    assert np.allclose(features[0, :width], 2.0)
    assert np.allclose(features[0, width : 2 * width], 1.0)
    assert np.allclose(features[0, 2 * width : 3 * width], 3.0)
    assert np.allclose(features[0, 3 * width : 4 * width], 2.0)
    assert np.allclose(features[0, 4 * width :], 4.0)
    assert np.allclose(features[1, :width], 5.0)
    assert np.allclose(features[1, width : 2 * width], 0.0)
    assert np.allclose(features[1, 2 * width : 3 * width], 5.0)
    assert np.allclose(features[1, 3 * width :], 0.0)
    assert np.isfinite(features).all()


def test_openface_adapter_fails_closed_on_legacy_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_text(
        "timestamp,blink,pressed_lips,furrow_brow\n2025-01-01 00:00:00,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing .*columns"):
        extract_openface_features(csv_path, num_steps=2)


def test_openface_video_cache_survives_pipeline_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "session_video.mp4"
    video.write_bytes(b"video-source")
    cache = tmp_path / "cache"
    calls = {"extract": 0}

    class FakeOpenFaceEncoder:
        def __init__(self, *, min_confidence: float) -> None:
            assert min_confidence == 0.8

        def extract_video(self, _video: Path, output_dir: str | Path) -> Path:
            calls["extract"] += 1
            generated = Path(output_dir) / "generated.csv"
            generated.write_text("cached", encoding="utf-8")
            return generated

        def encode_csv(self, _csv: Path, *, num_steps: int) -> SimpleNamespace:
            return SimpleNamespace(
                features=np.ones((num_steps, len(OPENFACE_FEATURE_NAMES)), dtype=np.float32),
                mask=np.ones(num_steps, dtype=bool),
            )

    monkeypatch.setattr(behavior_features, "OpenFaceFeatureEncoder", FakeOpenFaceEncoder)
    config = BehaviorFeatureConfig(
        video_backend="openface", openface_cache_dir=str(cache)
    )

    first = BehaviorFeaturePipeline(config).encode_video(video, num_steps=2)
    second = BehaviorFeaturePipeline(config).encode_video(video, num_steps=2)

    assert calls["extract"] == 1
    assert np.array_equal(first[0], second[0])
    assert list(cache.glob("*.csv"))


def test_pipeline_routes_raw_video_to_marlin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "session.mp4"
    video.write_bytes(b"video")
    calls: dict[str, object] = {}

    class FakeMarlinEncoder:
        def __init__(self, **kwargs: object) -> None:
            calls["init"] = kwargs

        def encode_video(self, path: Path, *, num_steps: int) -> EncodedSequence:
            calls["path"] = path
            return EncodedSequence(
                features=np.ones((num_steps, MARLIN_OUTPUT_DIM), dtype=np.float32),
                mask=np.ones(num_steps, dtype=bool),
            )

    monkeypatch.setattr(behavior_features, "MarlinFeatureEncoder", FakeMarlinEncoder)
    pipeline = BehaviorFeaturePipeline(
        BehaviorFeatureConfig(video_backend="marlin", device="cuda", batch_size=8)
    )

    features, mask = pipeline.encode_video(video, num_steps=3)

    assert features.shape == (3, MARLIN_OUTPUT_DIM)
    assert mask.tolist() == [True, True, True]
    assert calls["path"] == video
    assert calls["init"]["device"] == "cuda"
    assert calls["init"]["batch_size"] == 8


def test_marlin_video_cache_survives_pipeline_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "session.mp4"
    video.write_bytes(b"video")
    cache = tmp_path / "marlin-cache"
    calls = {"encode": 0}

    class FakeMarlinEncoder:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def encode_video(self, _path: Path, *, num_steps: int) -> EncodedSequence:
            calls["encode"] += 1
            return EncodedSequence(
                features=np.ones((num_steps, MARLIN_OUTPUT_DIM), dtype=np.float32),
                mask=np.ones(num_steps, dtype=bool),
            )

    monkeypatch.setattr(behavior_features, "MarlinFeatureEncoder", FakeMarlinEncoder)
    config = BehaviorFeatureConfig(
        video_backend="marlin",
        marlin_cache_dir=str(cache),
        marlin_crop_face=False,
    )

    first = BehaviorFeaturePipeline(config).encode_video(video, num_steps=2)
    second = BehaviorFeaturePipeline(config).encode_video(video, num_steps=2)

    assert calls["encode"] == 1
    assert np.array_equal(first[0], second[0])
    assert list(cache.glob("*.npz"))


def test_pipeline_rejects_csv_for_marlin(tmp_path: Path) -> None:
    csv_path = tmp_path / "face.csv"
    csv_path.write_text("legacy", encoding="utf-8")
    pipeline = BehaviorFeaturePipeline(BehaviorFeatureConfig(video_backend="marlin"))

    with pytest.raises(ValueError, match="raw video"):
        pipeline.encode_video(csv_path, num_steps=2)
